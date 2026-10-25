"""
AirTeach AI - Interactive Solid-State Physics: Band Theory & Kronig-Penney Simulator
====================================================================================
Authentic Quantum Mechanics & Semiconductor Physics:
- 1D Periodic Potential & Kronig-Penney Model:
    P * sin(α*a)/(α*a) + cos(α*a) = cos(k*a)
    where α = sqrt(2*m*E / ħ^2), a = lattice spacing, P = m*V0*a*b / ħ^2 (barrier strength).
    Allowed energy bands: |f(α*a)| <= 1  ==>  k*a = arccos(f(α*a))
    Forbidden bandgap E_g: |f(α*a)| > 1
- Energy Band Structure & Direct/Indirect Gaps:
    Effective mass m* = ħ^2 / (d^2E / dk^2)
    Valence Band Maximum (E_v) and Conduction Band Minimum (E_c)
- Fermi-Dirac Distribution & Statistical Mechanics:
    f(E) = 1 / [ 1 + exp((E - E_F) / (k_B * T)) ]
- Thermal Carrier Excitation & Law of Mass Action:
    Intrinsic carrier density: n_i = sqrt(N_c * N_v) * exp(-E_g / (2 * k_B * T))
    Mass action law: n * p = n_i^2
- Extrinsic Doping & Fermi Level Shifting:
    n-type: Donor density N_d, E_F = E_c - k_B*T * ln(N_c / N_d)
    p-type: Acceptor density N_a, E_F = E_v + k_B*T * ln(N_v / N_a)
- 5 Curriculum Presets:
    1. INTRINSIC_SILICON: Pure Si, E_g = 1.12 eV, mid-gap E_F, thermal e-h pairs.
    2. N_TYPE_DOPED: Phosphorus-doped Si (N_d = 1e16 cm^-3), donor level E_d, high electron density.
    3. P_TYPE_DOPED: Boron-doped Si (N_a = 1e16 cm^-3), acceptor level E_a, high hole density.
    4. DIRECT_GAP_GAAS: GaAs, E_g = 1.42 eV, direct gap optoelectronic photon recombination.
    5. WIDE_GAP_INSULATOR: Diamond, E_g = 5.47 eV, wide gap insulator with zero conduction electrons.
- Touchless webcam air gestures (air-tap, sliders, band dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Physical constants
BOLTZMANN_K_EV: float = 8.617333262e-5  # eV / K
PLANCK_HBAR_EV_S: float = 6.582119569e-16  # eV * s
E_CHARGE: float = 1.602176634e-19       # Coulombs


@dataclass
class BandTheoryPreset:
    """Preset configuration for educational band theory & semiconductor experiments."""
    key: str
    name: str
    desc: str
    material: str
    eg_ev: float
    is_direct: bool
    doping_type: str  # "intrinsic", "n_type", "p_type", "insulator"
    default_doping_cm3: float
    default_temp_k: float
    p_barrier: float


PRESETS: List[BandTheoryPreset] = [
    BandTheoryPreset(
        key="intrinsic_silicon",
        name="Silic Tinh khiết (Intrinsic Si)",
        desc="Bán dẫn Si tinh khiết (Eg = 1.12 eV), mức Fermi EF nằm chính giữa vùng cấm, hạt tải sinh ra do kích thích nhiệt",
        material="Silicon (Si, Tinh khiết)",
        eg_ev=1.12,
        is_direct=False,
        doping_type="intrinsic",
        default_doping_cm3=0.0,
        default_temp_k=300.0,
        p_barrier=6.0,
    ),
    BandTheoryPreset(
        key="n_type_doped",
        name="Silic Pha tạp loại n (n-type Si:P)",
        desc="Pha tạp Photpho (Nd = 1e16 cm^-3), tạo mức donor Ed sát đáy vùng dẫn, mức Fermi EF dịch mạnh lên phía trên",
        material="Si:P (Bán dẫn loại n)",
        eg_ev=1.12,
        is_direct=False,
        doping_type="n_type",
        default_doping_cm3=1e16,
        default_temp_k=300.0,
        p_barrier=6.0,
    ),
    BandTheoryPreset(
        key="p_type_doped",
        name="Silic Pha tạp loại p (p-type Si:B)",
        desc="Pha tạp Bo (Na = 1e16 cm^-3), tạo mức acceptor Ea sát đỉnh vùng hóa trị, mức Fermi EF dịch mạnh xuống dưới",
        material="Si:B (Bán dẫn loại p)",
        eg_ev=1.12,
        is_direct=False,
        doping_type="p_type",
        default_doping_cm3=1e16,
        default_temp_k=300.0,
        p_barrier=6.0,
    ),
    BandTheoryPreset(
        key="direct_gap_gaas",
        name="Gali Asenua Vùng cấm thẳng (GaAs)",
        desc="Bán dẫn vùng cấm thẳng GaAs (Eg = 1.42 eV), cho phép electron và lỗ trống tái hợp phát quang trực tiếp (LED, Laser)",
        material="Gallium Arsenide (GaAs)",
        eg_ev=1.42,
        is_direct=True,
        doping_type="intrinsic",
        default_doping_cm3=0.0,
        default_temp_k=300.0,
        p_barrier=7.5,
    ),
    BandTheoryPreset(
        key="wide_gap_insulator",
        name="Chất cách điện Vùng cấm rộng (Kim cương)",
        desc="Kim cương Carbon (Eg = 5.47 eV), năng lượng kích thích nhiệt ở nhiệt độ phòng hoàn toàn không đủ để vượt qua vùng cấm",
        material="Kim cương (Diamond, C)",
        eg_ev=5.47,
        is_direct=False,
        doping_type="insulator",
        default_doping_cm3=0.0,
        default_temp_k=300.0,
        p_barrier=15.0,
    ),
]


class InteractiveBandTheoryItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Solid-State Band Theory, Kronig-Penney
    dispersion relations, Fermi-Dirac distribution, and semiconductor doping.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 940.0
    HEIGHT: float = 600.0

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        p = PRESETS[self._preset_index]
        self._temp_k: float = p.default_temp_k
        self._eg_ev: float = p.eg_ev
        self._p_barrier: float = p.p_barrier
        self._doping_cm3: float = p.default_doping_cm3

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance carrier generation/recombination animation."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

    @property
    def current_preset(self) -> BandTheoryPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active band theory preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._temp_k = p.default_temp_k
            self._eg_ev = p.eg_ev
            self._p_barrier = p.p_barrier
            self._doping_cm3 = p.default_doping_cm3
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_temperature(self, t_kelvin: float) -> None:
        """Set temperature in Kelvin."""
        self._temp_k = max(10.0, min(t_kelvin, 600.0))
        self.property_changed.emit("temperature", self._temp_k)
        self.update()

    def set_bandgap(self, eg_ev: float) -> None:
        """Set bandgap energy in eV."""
        self._eg_ev = max(0.1, min(eg_ev, 6.5))
        self.property_changed.emit("bandgap", self._eg_ev)
        self.update()

    def set_barrier_strength(self, p_val: float) -> None:
        """Set Kronig-Penney barrier strength P."""
        self._p_barrier = max(0.5, min(p_val, 25.0))
        self.property_changed.emit("p_barrier", self._p_barrier)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset defaults."""
        self.set_preset(self._preset_index)

    # -------------------------------------------------------------------------
    # Authentic Physics Calculations
    # -------------------------------------------------------------------------

    def compute_fermi_level(self) -> float:
        """
        Compute Fermi level E_F relative to midgap E_i = 0.0 eV.
        Conduction band Ec = +Eg / 2, Valence band Ev = -Eg / 2.
        Returns: E_F in eV.
        """
        p = self.current_preset
        t = self._temp_k
        kb_t = BOLTZMANN_K_EV * t
        ec = self._eg_ev / 2.0
        ev = -self._eg_ev / 2.0

        if p.doping_type == "intrinsic" or p.doping_type == "insulator":
            return 0.0  # Exactly at midgap
        elif p.doping_type == "n_type":
            # Shifted up toward Ec
            shift = min(ec - 0.05, kb_t * math.log(max(1.0, self._doping_cm3 / 1e10)))
            return float(min(ec - 0.02, shift))
        elif p.doping_type == "p_type":
            # Shifted down toward Ev
            shift = min(abs(ev) - 0.05, kb_t * math.log(max(1.0, self._doping_cm3 / 1e10)))
            return float(max(ev + 0.02, -shift))
        return 0.0

    def fermi_dirac(self, e_ev: float) -> float:
        """Compute Fermi-Dirac occupation probability f(E)."""
        ef = self.compute_fermi_level()
        kb_t = max(1e-5, BOLTZMANN_K_EV * self._temp_k)
        delta = (e_ev - ef) / kb_t
        if delta > 80.0:
            return 0.0
        elif delta < -80.0:
            return 1.0
        return float(1.0 / (1.0 + math.exp(delta)))

    def compute_carrier_densities(self) -> Tuple[float, float, float]:
        """
        Compute intrinsic density n_i, electron density n, and hole density p (cm^-3).
        Returns: (n_i, n, p)
        """
        t = self._temp_k
        kb_t = BOLTZMANN_K_EV * t
        # Effective density of states Nc, Nv ≈ 2.8e19 * (T/300)^1.5
        nc = 2.8e19 * (t / 300.0)**1.5
        nv = 1.04e19 * (t / 300.0)**1.5

        # n_i = sqrt(Nc * Nv) * exp(-Eg / (2 * kB * T))
        exponent = -self._eg_ev / (2.0 * kb_t)
        if exponent < -80.0:
            ni = 0.0
        else:
            ni = math.sqrt(nc * nv) * math.exp(exponent)

        p = self.current_preset
        if p.doping_type == "n_type":
            nd = self._doping_cm3
            n = 0.5 * (nd + math.sqrt(nd**2 + 4.0 * ni**2))
            hole = (ni**2) / n if n > 0 else 0.0
        elif p.doping_type == "p_type":
            na = self._doping_cm3
            hole = 0.5 * (na + math.sqrt(na**2 + 4.0 * ni**2))
            n = (ni**2) / hole if hole > 0 else 0.0
        else:
            n = ni
            hole = ni

        return float(ni), float(n), float(hole)

    def step_physics(self, dt: float) -> None:
        """Advance time parameter for carrier fluctuation rendering."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, 56.0)

    def _left_dispersion_rect(self) -> QRectF:
        return QRectF(15.0, 65.0, 310.0, 420.0)

    def _center_band_rect(self) -> QRectF:
        return QRectF(335.0, 65.0, 290.0, 420.0)

    def _right_fermi_rect(self) -> QRectF:
        return QRectF(635.0, 65.0, 290.0, 420.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, BandTheoryPreset]]:
        h_rect = self._header_rect()
        btn_w = 95.0
        btn_h = 28.0
        gap = 6.0
        start_x = 350.0
        y = (h_rect.height() - btn_h) / 2.0
        rects = []
        for i, p in enumerate(PRESETS):
            bx = start_x + i * (btn_w + gap)
            rects.append((QRectF(bx, y, btn_w, btn_h), p))
        return rects

    def paint(
        self,
        painter: QPainter,
        option: Optional[QStyleOptionGraphicsItem] = None,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Main Background Card
        bg_brush = QBrush(QColor(11, 15, 25, 245))
        border_pen = QPen(QColor(99, 102, 241, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_kronig_penney_dispersion(painter)
        self._paint_energy_band_diagram(painter)
        self._paint_fermi_dirac_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ LÝ THUYẾT VÙNG NĂNG LƯỢNG & MÔ HÌNH KRONIG-PENNEY")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        gap_type = "Thẳng (Direct)" if p.is_direct else "Xiên (Indirect)"
        painter.drawText(20, 44, f"Vật liệu: {p.material}  |  Vùng cấm Eg = {self._eg_ev:.2f} eV ({gap_type})  |  T = {self._temp_k:.0f} K")

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(99, 102, 241, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(129, 140, 248), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            labels = ["Si thuần", "Si loại n", "Si loại p", "GaAs thẳng", "Kim cương"]
            label = labels[i] if i < len(labels) else preset.name[:8]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        badge_color = QColor(16, 185, 129) if p.doping_type == "intrinsic" else (
            QColor(56, 189, 248) if p.doping_type == "n_type" else (
                QColor(244, 114, 182) if p.doping_type == "p_type" else QColor(250, 204, 21)
            )
        )
        painter.setBrush(QColor(badge_color.red(), badge_color.green(), badge_color.blue(), 40))
        painter.setPen(QPen(badge_color, 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(badge_color)
        painter.drawText(badge_rect, Qt.AlignCenter, p.material.split("(")[0].strip()[:14])

    def _paint_kronig_penney_dispersion(self, painter: QPainter) -> None:
        """Render Left Panel: E(k) Dispersion Bands in 1st Brillouin Zone."""
        rect = self._left_dispersion_rect()

        # Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "📈 ĐỒ THỊ TÁN SẮC E(k) (KRONIG-PENNEY)")

        # Plot boundary
        plot_rect = QRectF(rect.left() + 25.0, rect.top() + 45.0, rect.width() - 40.0, 310.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(plot_rect)

        # Brillouin Zone center (k=0) and edges (k = ±pi/a)
        cx = plot_rect.center().x()
        painter.setPen(QPen(QColor(51, 65, 85), 1.0, Qt.DashLine))
        painter.drawLine(int(cx), int(plot_rect.top()), int(cx), int(plot_rect.bottom()))
        painter.drawLine(int(plot_rect.left()), int(plot_rect.center().y()), int(plot_rect.right()), int(plot_rect.center().y()))

        # Shaded Forbidden Bandgap (Eg)
        eg_scaled = (self._eg_ev / 6.0) * plot_rect.height()
        gap_top = plot_rect.center().y() - eg_scaled / 2.0
        gap_bot = plot_rect.center().y() + eg_scaled / 2.0
        gap_rect = QRectF(plot_rect.left(), gap_top, plot_rect.width(), eg_scaled)

        painter.setBrush(QColor(239, 68, 68, 25))
        painter.setPen(Qt.NoPen)
        painter.drawRect(gap_rect)

        # Draw E(k) parabolic bands
        # Conduction Band (parabolic upwards)
        cb_path = QPainterPath()
        vb_path = QPainterPath()
        p = self.current_preset
        k_shift = 0.0 if p.is_direct else 0.45  # Indirect gap shift in Si

        num_pts = 60
        half_w = plot_rect.width() / 2.0

        for i in range(num_pts + 1):
            norm_k = (i / num_pts) * 2.0 - 1.0  # k from -1 to +1 (units of pi/a)
            px = cx + norm_k * half_w

            # Conduction band minimum at k_shift
            e_cb = gap_top - 90.0 * ((abs(norm_k) - k_shift)**2 if not p.is_direct else norm_k**2)
            # Valence band maximum at k=0
            e_vb = gap_bot + 80.0 * (norm_k**2)

            if i == 0:
                cb_path.moveTo(px, e_cb)
                vb_path.moveTo(px, e_vb)
            else:
                cb_path.lineTo(px, e_cb)
                vb_path.lineTo(px, e_vb)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawPath(cb_path)

        painter.setPen(QPen(QColor(244, 114, 182), 2.0))
        painter.drawPath(vb_path)

        # Labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(plot_rect.left() + 5.0, gap_top - 18.0, 120.0, 16.0), Qt.AlignLeft, "Vùng dẫn (CB)")

        painter.setPen(QColor(244, 114, 182))
        painter.drawText(QRectF(plot_rect.left() + 5.0, gap_bot + 4.0, 120.0, 16.0), Qt.AlignLeft, "Vùng hóa trị (VB)")

        painter.setPen(QColor(239, 68, 68))
        painter.drawText(gap_rect, Qt.AlignCenter, f"VÙNG CẤM Eg = {self._eg_ev:.2f} eV")

        # Bottom text
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(int(rect.left() + 25.0), int(rect.bottom() - 25.0), f"Hệ số rào P = {self._p_barrier:.1f}  |  -π/a ≤ k ≤ +π/a")

    def _paint_energy_band_diagram(self, painter: QPainter) -> None:
        """Render Center Panel: Real-Space Band Diagram & Electron/Hole Carriers."""
        rect = self._center_band_rect()

        # Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "🔋 VÙNG NĂNG LƯỢNG & HẠT TẢI")

        # Band boundaries
        plot_rect = QRectF(rect.left() + 20.0, rect.top() + 45.0, rect.width() - 40.0, 310.0)
        eg_scaled = (self._eg_ev / 6.0) * plot_rect.height()
        ec_y = plot_rect.center().y() - eg_scaled / 2.0
        ev_y = plot_rect.center().y() + eg_scaled / 2.0

        # 1. Conduction Band Block
        cb_rect = QRectF(plot_rect.left(), plot_rect.top(), plot_rect.width(), ec_y - plot_rect.top())
        cb_grad = QLinearGradient(cb_rect.left(), cb_rect.bottom(), cb_rect.left(), cb_rect.top())
        cb_grad.setColorAt(0.0, QColor(14, 165, 233, 120))
        cb_grad.setColorAt(1.0, QColor(14, 165, 233, 30))
        painter.setBrush(cb_grad)
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(cb_rect, 6.0, 6.0)

        # 2. Valence Band Block
        vb_rect = QRectF(plot_rect.left(), ev_y, plot_rect.width(), plot_rect.bottom() - ev_y)
        vb_grad = QLinearGradient(vb_rect.left(), vb_rect.top(), vb_rect.left(), vb_rect.bottom())
        vb_grad.setColorAt(0.0, QColor(236, 72, 153, 120))
        vb_grad.setColorAt(1.0, QColor(236, 72, 153, 30))
        painter.setBrush(vb_grad)
        painter.setPen(QPen(QColor(244, 114, 182), 1.5))
        painter.drawRoundedRect(vb_rect, 6.0, 6.0)

        # 3. Fermi Level Line (E_F)
        ef_val = self.compute_fermi_level()
        ef_y = plot_rect.center().y() - (ef_val / 3.0) * (plot_rect.height() / 2.0)
        painter.setPen(QPen(QColor(250, 204, 21), 1.8, Qt.DashLine))
        painter.drawLine(int(plot_rect.left()), int(ef_y), int(plot_rect.right()), int(ef_y))

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        painter.drawText(QRectF(plot_rect.left() + 10.0, ef_y - 18.0, 100.0, 16.0), Qt.AlignLeft, f"EF = {ef_val:+.2f} eV")

        # 4. Animated Carriers (Electrons in CB, Holes in VB)
        _, n_dens, p_dens = self.compute_carrier_densities()
        num_e = min(20, max(1, int(math.log10(max(10.0, n_dens)) - 9))) if n_dens > 1e9 else 0
        num_h = min(20, max(1, int(math.log10(max(10.0, p_dens)) - 9))) if p_dens > 1e9 else 0

        # Draw conduction electrons (cyan dots)
        painter.setBrush(QColor(56, 189, 248))
        painter.setPen(QPen(QColor(255, 255, 255), 1.0))
        for ei in range(num_e):
            ex = plot_rect.left() + 25.0 + (ei * 22.0 + math.sin(self._sim_time * 3.0 + ei) * 8.0) % (plot_rect.width() - 50.0)
            ey = ec_y - 12.0 - (ei % 3) * 10.0
            painter.drawEllipse(QPointF(ex, ey), 4.0, 4.0)

        # Draw valence holes (pink hollow rings)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(244, 114, 182), 1.8))
        for hi in range(num_h):
            hx = plot_rect.left() + 25.0 + (hi * 22.0 + math.cos(self._sim_time * 3.0 + hi) * 8.0) % (plot_rect.width() - 50.0)
            hy = ev_y + 12.0 + (hi % 3) * 10.0
            painter.drawEllipse(QPointF(hx, hy), 4.5, 4.5)

        # Telemetry Text
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        hud_lines = [
            f"Ec = {self._eg_ev/2.0:+.2f} eV  |  Ev = {-self._eg_ev/2.0:+.2f} eV",
            f"Mật độ e-: n = {n_dens:.2e} cm⁻³",
            f"Mật độ lỗ trống: p = {p_dens:.2e} cm⁻³",
        ]
        ty = rect.bottom() - 48.0
        for line in hud_lines:
            painter.drawText(int(rect.left() + 20.0), int(ty), line)
            ty += 16.0

    def _paint_fermi_dirac_panel(self, painter: QPainter) -> None:
        """Render Right Panel: Fermi-Dirac f(E) Distribution Curve & Statistics."""
        rect = self._right_fermi_rect()

        # Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "📊 XÁC SUẤT FERMI-DIRAC f(E)")

        # Plot boundary
        plot_rect = QRectF(rect.left() + 25.0, rect.top() + 45.0, rect.width() - 40.0, 310.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(plot_rect)

        # Draw f(E) curve: Energy E on vertical axis (from -3.0 eV to +3.0 eV), f(E) on horizontal axis (0.0 to 1.0)
        f_path = QPainterPath()
        num_pts = 60
        e_range = 3.0

        for i in range(num_pts + 1):
            e_val = -e_range + (i / num_pts) * (2.0 * e_range)
            f_val = self.fermi_dirac(e_val)

            px = plot_rect.left() + f_val * plot_rect.width()
            py = plot_rect.bottom() - (i / num_pts) * plot_rect.height()

            if i == 0:
                f_path.moveTo(px, py)
            else:
                f_path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(250, 204, 21), 2.2))
        painter.drawPath(f_path)

        # 0.5 probability vertical line at E_F
        ef_val = self.compute_fermi_level()
        ef_norm = (ef_val + e_range) / (2.0 * e_range)
        ef_py = plot_rect.bottom() - ef_norm * plot_rect.height()

        painter.setPen(QPen(QColor(250, 204, 21, 140), 1.0, Qt.DashLine))
        painter.drawLine(int(plot_rect.left() + 0.5 * plot_rect.width()), int(plot_rect.top()), int(plot_rect.left() + 0.5 * plot_rect.width()), int(plot_rect.bottom()))

        # Point (f=0.5, E=EF)
        painter.setBrush(QColor(250, 204, 21))
        painter.drawEllipse(QPointF(plot_rect.left() + 0.5 * plot_rect.width(), ef_py), 4.5, 4.5)

        # Labels & Explanations
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(int(plot_rect.left() + 5.0), int(plot_rect.bottom() - 5.0), "f=0.0")
        painter.drawText(int(plot_rect.left() + 0.5 * plot_rect.width() - 15.0), int(plot_rect.bottom() - 5.0), "f=0.5")
        painter.drawText(int(plot_rect.right() - 25.0), int(plot_rect.bottom() - 5.0), "f=1.0")

        # Summary text
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(int(rect.left() + 20.0), int(rect.bottom() - 32.0), f"k_B·T = {BOLTZMANN_K_EV * self._temp_k * 1000.0:.1f} meV  |  f(EF) ≡ 50%")

    def _paint_controls(self, painter: QPainter) -> None:
        ctrl_rect = self._controls_rect()

        # Controls Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(ctrl_rect, 10.0, 10.0)

        col_w = (ctrl_rect.width() - 40.0) / 4.0

        sliders = [
            ("temp", "Nhiệt độ T (K)", f"{self._temp_k:.0f} K", self._temp_k, 10.0, 600.0),
            ("gap", "Độ rộng vùng cấm Eg (eV)", f"{self._eg_ev:.2f} eV", self._eg_ev, 0.2, 5.5),
            ("barrier", "Hệ số thế rào P (KP)", f"{self._p_barrier:.1f}", self._p_barrier, 1.0, 20.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(129, 140, 248))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill
            norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(99, 102, 241))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(99, 102, 241), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Reset button in 4th column
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        painter.setBrush(QColor(99, 102, 241, 40))
        painter.setPen(QPen(QColor(129, 140, 248), 1.2))
        painter.drawRoundedRect(btn_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(129, 140, 248))
        painter.drawText(btn_rect, Qt.AlignCenter, "🔄 ĐẶT LẠI BAN ĐẦU")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Reset Button
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        if btn_rect.contains(pos):
            self.reset_simulation()
            event.accept()
            return

        # Check Sliders
        for i, key in enumerate(["temp", "gap", "barrier"]):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["temp", "gap", "barrier"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        if key == "temp":
            self.set_temperature(10.0 + norm * 590.0)
        elif key == "gap":
            self.set_bandgap(0.2 + norm * 5.3)
        elif key == "barrier":
            self.set_barrier_strength(1.0 + norm * 19.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveBandTheoryItem",
            "preset_key": self.current_preset.key,
            "temp_k": self._temp_k,
            "eg_ev": self._eg_ev,
            "p_barrier": self._p_barrier,
            "doping_cm3": self._doping_cm3,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveBandTheoryItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "intrinsic_silicon")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "temp_k" in data:
            item._temp_k = float(data["temp_k"])
        if "eg_ev" in data:
            item._eg_ev = float(data["eg_ev"])
        if "p_barrier" in data:
            item._p_barrier = float(data["p_barrier"])
        if "doping_cm3" in data:
            item._doping_cm3 = float(data["doping_cm3"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
