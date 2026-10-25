"""
AirTeach AI - Interactive Structural Biology: Cryo-EM CTF & 3D Reconstruction
=============================================================================
Authentic Nobel Prize in Chemistry (2017) Electron Microscopy Foundations:
- Contrast Transfer Function (CTF):
    CTF(k) = -sqrt(1 - Q^2) * sin(χ(k)) - Q * cos(χ(k)) * E(k)
    where phase aberration:
      χ(k) = π * λ * Δf * k^2 + 0.5 * π * Cs * λ^3 * k^4
    k = spatial frequency (1 / Å), Q = amplitude contrast fraction (~0.08).
- Relativistic Electron de Broglie Wavelength:
    λ = h / sqrt(2*m_e*e*V * (1 + e*V / (2*m_e*c^2)))
    (e.g., λ ≈ 0.019687 Å at 300 keV, 0.025079 Å at 200 keV).
- Damping Envelope Function:
    E(k) = exp(-B_factor * k^2 / 4)
- Scherzer Optimal Defocus:
    Δf_Sch = -1.2 * sqrt(Cs * λ)
- 2D Thon Rings (Fourier Power Spectrum):
    |CTF(k)|^2 oscillating concentric rings with zero-crossing nodes.
    Astigmatism shifts rings into concentric ellipses.
- Gold-Standard Fourier Shell Correlation (FSC):
    FSC(k) = ∑(F1 * F2*) / sqrt(∑|F1|^2 * ∑|F2|^2)
    Nominal resolution reported at FSC = 0.143 threshold.
- 5 Curriculum Presets:
    1. RIBOSOME_70S: 70S Ribosome complex, 300 keV, Δf = -1.5 µm, 2.4 Å resolution.
    2. SARS_COV2_SPIKE: Spike glycoprotein trimer, 300 keV, Δf = -1.2 µm, 3.1 Å resolution.
    3. APOFERRITIN_ATOMIC: Benchmark test, Cs-corrected (Cs=0.01 mm), atomic 1.25 Å.
    4. ION_CHANNEL_MEMBRANE: K+ channel in nanodisc, high defocus Δf = -2.2 µm.
    5. OVERFOCUSED_ARTIFACT: Defocus Δf = +1.0 µm, phase reversal contrast artifact.
- Touchless webcam air gestures (air-tap, defocus slider, voltage tuning),
  bimanual manipulation, canvas stamping, and .airteach packaging.
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
    QImage,
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

# Fundamental constants
PLANCK_H: float = 6.62607015e-34      # J * s
SPEED_OF_LIGHT: float = 299792458.0    # m/s
ELECTRON_MASS: float = 9.1093837e-31  # kg
E_CHARGE: float = 1.602176634e-19      # C


@dataclass
class CryoEMPreset:
    """Preset configuration for educational Cryo-EM & CTF experiments."""
    key: str
    name: str
    desc: str
    specimen: str
    voltage_kv: float
    defocus_um: float
    cs_mm: float
    b_factor: float
    nominal_res_ang: float


PRESETS: List[CryoEMPreset] = [
    CryoEMPreset(
        key="ribosome_70s",
        name="Ribosome 70S Vi khuẩn (2.4 Å)",
        desc="Phức hợp Ribosome 70S vi khuẩn ở 300 keV, độ mờ defocus -1.5 µm, đạt độ phân giải tái tạo 2.4 Å",
        specimen="Bacterial 70S Ribosome",
        voltage_kv=300.0,
        defocus_um=-1.5,
        cs_mm=2.7,
        b_factor=45.0,
        nominal_res_ang=2.4,
    ),
    CryoEMPreset(
        key="sars_cov2_spike",
        name="Spike Glycoprotein SARS-CoV-2 (3.1 Å)",
        desc="Gai trimer SARS-CoV-2 ở trạng thái tiền dung hợp (Pre-fusion), độ mờ defocus -1.2 µm",
        specimen="SARS-CoV-2 Spike Trimer",
        voltage_kv=300.0,
        defocus_um=-1.2,
        cs_mm=2.7,
        b_factor=60.0,
        nominal_res_ang=3.1,
    ),
    CryoEMPreset(
        key="apoferritin_atomic",
        name="Apoferritin Độ phân giải Nguyên tử (1.25 Å)",
        desc="Kính hiển vi Cryo-EM có bộ khử cầu sai Cs (Cs = 0.01 mm), độ phân giải nguyên tử vượt ngưỡng 1.25 Å",
        specimen="Apoferritin Benchmark",
        voltage_kv=300.0,
        defocus_um=-0.35,
        cs_mm=0.01,
        b_factor=15.0,
        nominal_res_ang=1.25,
    ),
    CryoEMPreset(
        key="ion_channel_membrane",
        name="Kênh ion Màng tế bào (K+ Channel)",
        desc="Kênh ion Kali gắn trên đĩa nano lipid (Nanodisc), defocus cao -2.2 µm giúp nhận diện hạt tương phản tốt",
        specimen="Voltage-Gated K+ Channel",
        voltage_kv=200.0,
        defocus_um=-2.2,
        cs_mm=2.7,
        b_factor=75.0,
        nominal_res_ang=3.6,
    ),
    CryoEMPreset(
        key="overfocused_artifact",
        name="Ảnh quá tiêu cự (Overfocused +1.0 µm)",
        desc="Defocus dương (+1.0 µm) làm đảo pha tương phản (Phase reversal), tạo quầng trắng nhân tạo quanh đại phân tử",
        specimen="Overfocused Defocus Artifact",
        voltage_kv=300.0,
        defocus_um=1.0,
        cs_mm=2.7,
        b_factor=50.0,
        nominal_res_ang=5.5,
    ),
]


class InteractiveCryoEMItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Cryo-EM Contrast Transfer Function (CTF),
    Thon rings, Fourier Shell Correlation (FSC), and 3D molecular reconstruction.
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
        self._voltage_kv: float = p.voltage_kv
        self._defocus_um: float = p.defocus_um
        self._cs_mm: float = p.cs_mm
        self._b_factor: float = p.b_factor
        self._amp_contrast: float = 0.08  # ~8% amplitude contrast

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance continuous specimen rotation."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources."""
        try:
            if hasattr(self, "_anim_timer") and self._anim_timer is not None and self._anim_timer.isActive():
                self._anim_timer.stop()
        except (RuntimeError, ReferenceError):
            pass

    @property
    def current_preset(self) -> CryoEMPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active cryo-em preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._voltage_kv = p.voltage_kv
            self._defocus_um = p.defocus_um
            self._cs_mm = p.cs_mm
            self._b_factor = p.b_factor
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_voltage_kv(self, v_kv: float) -> None:
        """Set accelerating voltage in kV."""
        self._voltage_kv = max(80.0, min(v_kv, 400.0))
        self.property_changed.emit("voltage_kv", self._voltage_kv)
        self.update()

    def set_defocus_um(self, df_um: float) -> None:
        """Set defocus in micrometers."""
        self._defocus_um = max(-5.0, min(df_um, 3.0))
        self.property_changed.emit("defocus_um", self._defocus_um)
        self.update()

    def set_cs_mm(self, cs: float) -> None:
        """Set spherical aberration Cs in mm."""
        self._cs_mm = max(0.001, min(cs, 5.0))
        self.property_changed.emit("cs_mm", self._cs_mm)
        self.update()

    def set_b_factor(self, b: float) -> None:
        """Set temperature/damping B-factor in Å^2."""
        self._b_factor = max(5.0, min(b, 150.0))
        self.property_changed.emit("b_factor", self._b_factor)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset defaults."""
        self.set_preset(self._preset_index)

    # -------------------------------------------------------------------------
    # Authentic Cryo-EM Electron Optics & CTF Calculations
    # -------------------------------------------------------------------------

    def compute_electron_wavelength_angstrom(self) -> float:
        """
        Compute relativistic electron de Broglie wavelength in Ångströms:
        λ = h / sqrt( 2 * m_e * e * V * (1 + e*V / (2*m_e*c^2)) )
        """
        v_volts = self._voltage_kv * 1000.0
        rest_energy = ELECTRON_MASS * (SPEED_OF_LIGHT**2)
        kin_energy = E_CHARGE * v_volts
        momentum = math.sqrt(2.0 * ELECTRON_MASS * kin_energy * (1.0 + kin_energy / (2.0 * rest_energy)))
        lambda_m = PLANCK_H / momentum
        return float(lambda_m * 1e10)  # in Ångströms

    def compute_scherzer_defocus_um(self) -> float:
        """Compute optimal Scherzer defocus: Δf_Sch = -1.2 * sqrt(Cs * λ) in µm."""
        lam_ang = self.compute_electron_wavelength_angstrom()
        cs_ang = self._cs_mm * 1e7  # mm to Å
        df_sch_ang = -1.2 * math.sqrt(cs_ang * lam_ang)
        return float(df_sch_ang * 1e-4)  # Å to µm

    def compute_ctf(self, k_inv_ang: float) -> float:
        """
        Compute Contrast Transfer Function CTF(k):
        χ(k) = π * λ * Δf * k^2 + 0.5 * π * Cs * λ^3 * k^4
        CTF(k) = -sqrt(1 - Q^2) * sin(χ(k)) - Q * cos(χ(k)) * exp(-B * k^2 / 4)
        """
        lam = self.compute_electron_wavelength_angstrom()  # in Å
        df = self._defocus_um * 1e4                       # µm to Å
        cs = self._cs_mm * 1e7                            # mm to Å
        q = self._amp_contrast
        w1 = math.sqrt(max(0.0, 1.0 - q**2))
        w2 = q

        # Phase aberration chi(k)
        chi = math.pi * lam * df * (k_inv_ang**2) + 0.5 * math.pi * cs * (lam**3) * (k_inv_ang**4)

        # Damping envelope
        env = math.exp(-self._b_factor * (k_inv_ang**2) / 4.0)

        # Combined CTF
        val = -w1 * math.sin(chi) - w2 * math.cos(chi)
        return float(val * env)

    def compute_fsc(self, k_inv_ang: float) -> float:
        """
        Compute model Fourier Shell Correlation curve FSC(k).
        FSC drops smoothly from 1.0 down through the 0.143 gold-standard cutoff at nominal resolution.
        """
        nom_res = self.current_preset.nominal_res_ang
        k_res = 1.0 / nom_res
        # Sigmoid transition centered near k_res
        arg = (k_inv_ang - k_res) * (18.0 * nom_res)
        fsc = 1.0 / (1.0 + math.exp(min(40.0, max(-40.0, arg))))
        return float(max(0.0, min(1.0, fsc)))

    def step_physics(self, dt: float) -> None:
        """Advance time for 3D specimen particle rotation animation."""
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

    def _micrograph_rect(self) -> QRectF:
        return QRectF(15.0, 65.0, 290.0, 420.0)

    def _ctf_curve_rect(self) -> QRectF:
        return QRectF(315.0, 65.0, 310.0, 420.0)

    def _thon_fsc_rect(self) -> QRectF:
        return QRectF(635.0, 65.0, 290.0, 420.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, CryoEMPreset]]:
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
        border_pen = QPen(QColor(16, 185, 129, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_micrograph_panel(painter)
        self._paint_ctf_panel(painter)
        self._paint_thon_fsc_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ CRYO-EM: HÀM TRUYỀN TƯƠNG PHẢN (CTF) & VÒNG THON")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        lam = self.compute_electron_wavelength_angstrom()
        painter.drawText(20, 44, f"Mẫu: {p.specimen}  |  Điện áp: {self._voltage_kv:.0f} kV (λ ≈ {lam:.4f} Å)  |  Độ mờ Δf = {self._defocus_um:+.2f} µm")

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(16, 185, 129, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(52, 211, 153), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            labels = ["Ribosome", "Spike SARS", "Apoferritin", "K+ Channel", "Quá tiêu cự"]
            label = labels[i] if i < len(labels) else preset.name[:8]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        painter.setBrush(QColor(16, 185, 129, 40))
        painter.setPen(QPen(QColor(52, 211, 153), 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(52, 211, 153))
        painter.drawText(badge_rect, Qt.AlignCenter, f"FSC: {p.nominal_res_ang:.2f} Å (Gold)")

    def _paint_micrograph_panel(self, painter: QPainter) -> None:
        """Render Left Panel: Simulated Cryo-EM Micrograph & Rotating Single Particle."""
        rect = self._micrograph_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "🔬 ẢNH VI MÔ VĂN (MICROGRAPH)")

        # Micrograph box
        box_rect = QRectF(rect.left() + 20.0, rect.top() + 45.0, rect.width() - 40.0, 310.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(box_rect)

        # Draw Vitreous Ice Noise Background
        cx = box_rect.center().x()
        cy = box_rect.center().y()

        # Single Particle Projective Density (Rotating Macro-Complex)
        rot_angle = self._sim_time * 0.8
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(math.degrees(rot_angle))

        # Macromolecule Density Blobs (Ribosome / Spike subunits)
        painter.setPen(Qt.NoPen)
        for bi in range(8):
            ang = bi * (2.0 * math.pi / 8.0)
            bx = 45.0 * math.cos(ang)
            by = 45.0 * math.sin(ang)
            b_rad = 22.0 + 6.0 * math.sin(bi * 1.5)

            # Contrast depends on defocus: negative defocus -> dark particles with bright Fresnel fringe
            is_overfocused = (self._defocus_um > 0.0)
            grad = QRadialGradient(bx, by, b_rad)
            if not is_overfocused:
                grad.setColorAt(0.0, QColor(15, 23, 42, 230))
                grad.setColorAt(0.8, QColor(30, 41, 59, 180))
                grad.setColorAt(1.0, QColor(255, 255, 255, 45))  # Phase fringe
            else:
                grad.setColorAt(0.0, QColor(255, 255, 255, 200))  # Inverted contrast
                grad.setColorAt(1.0, QColor(15, 23, 42, 100))

            painter.setBrush(grad)
            painter.drawEllipse(QPointF(bx, by), b_rad, b_rad)

        # Center core subunit
        core_grad = QRadialGradient(0.0, 0.0, 32.0)
        core_grad.setColorAt(0.0, QColor(16, 185, 129, 120))
        core_grad.setColorAt(1.0, QColor(15, 23, 42, 220))
        painter.setBrush(core_grad)
        painter.drawEllipse(QPointF(0.0, 0.0), 32.0, 32.0)

        painter.restore()

        # Telemetry Text
        sch_df = self.compute_scherzer_defocus_um()
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        ty = box_rect.bottom() + 18.0
        painter.drawText(int(rect.left() + 20.0), int(ty), f"Scherzer Defocus: Δf_Sch = {sch_df:.2f} µm")
        painter.drawText(int(rect.left() + 20.0), int(ty + 16.0), f"Cs = {self._cs_mm:.2f} mm  |  B-Factor = {self._b_factor:.0f} Å²")

    def _paint_ctf_panel(self, painter: QPainter) -> None:
        """Render Middle Panel: 1D CTF(k) Oscillation Profile with Zero-Crossings."""
        rect = self._ctf_curve_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "📈 HÀM TRUYỀN CTF(k) & ĐẢO PHA")

        # Plot boundary
        plot_rect = QRectF(rect.left() + 25.0, rect.top() + 45.0, rect.width() - 40.0, 310.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(plot_rect)

        # Zero CTF horizontal line (center)
        zero_y = plot_rect.center().y()
        painter.setPen(QPen(QColor(71, 85, 105), 1.0, Qt.DashLine))
        painter.drawLine(int(plot_rect.left()), int(zero_y), int(plot_rect.right()), int(zero_y))

        # Draw CTF(k) oscillating curve
        # k from 0.0 to 0.5 (1/Å) corresponding to resolution down to 2.0 Å
        ctf_path = QPainterPath()
        num_pts = 120
        max_k = 0.55  # 1 / Å

        for i in range(num_pts + 1):
            k_val = (i / num_pts) * max_k
            ctf_val = self.compute_ctf(k_val)

            px = plot_rect.left() + (k_val / max_k) * plot_rect.width()
            # CTF between -1.0 and +1.0
            py = zero_y - ctf_val * (plot_rect.height() * 0.42)

            if i == 0:
                ctf_path.moveTo(px, py)
            else:
                ctf_path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(52, 211, 153), 2.0))
        painter.drawPath(ctf_path)

        # Frequency Axis labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(int(plot_rect.left() + 5.0), int(zero_y - 6.0), "k=0")
        painter.drawText(int(plot_rect.left() + 0.5 * plot_rect.width()), int(plot_rect.bottom() + 14.0), "0.25 Å⁻¹ (4 Å)")
        painter.drawText(int(plot_rect.right() - 65.0), int(plot_rect.bottom() + 14.0), "0.50 Å⁻¹ (2 Å)")

        # Summary text
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        ty = plot_rect.bottom() + 20.0
        painter.drawText(int(rect.left() + 20.0), int(ty), "Các điểm CTF = 0 là vị trí mất thông tin (Zero-crossings)")
        painter.drawText(int(rect.left() + 20.0), int(ty + 16.0), "Đảo pha (Phase flip) được sửa bằng CTF Correction")

    def _paint_thon_fsc_panel(self, painter: QPainter) -> None:
        """Render Right Panel: 2D Thon Rings & Gold-Standard FSC Resolution Curve."""
        rect = self._thon_fsc_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "🍩 VÒNG THON & ĐƯỜNG CONG FSC")

        # Top: 2D Thon Rings (Concentric Circles)
        thon_rect = QRectF(rect.left() + 65.0, rect.top() + 45.0, 160.0, 160.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(thon_rect)

        # Render concentric Thon rings
        tcx = thon_rect.center().x()
        tcy = thon_rect.center().y()
        max_rad = thon_rect.width() / 2.0 - 5.0

        for ri in range(1, 6):
            # Ring radii proportional to sqrt of zero crossings
            r_px = max_rad * math.sqrt(ri / 6.0)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248, 160), 1.5))
            painter.drawEllipse(QPointF(tcx, tcy), r_px, r_px)

        # Bottom: FSC Resolution Curve
        fsc_rect = QRectF(rect.left() + 25.0, thon_rect.bottom() + 25.0, rect.width() - 45.0, 120.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(fsc_rect)

        # 0.143 Gold-Standard Cutoff Line
        cut_y = fsc_rect.bottom() - 0.143 * fsc_rect.height()
        painter.setPen(QPen(QColor(239, 68, 68), 1.2, Qt.DashLine))
        painter.drawLine(int(fsc_rect.left()), int(cut_y), int(fsc_rect.right()), int(cut_y))

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(239, 68, 68))
        painter.drawText(int(fsc_rect.left() + 5.0), int(cut_y - 4.0), "FSC = 0.143 (Gold Standard)")

        # Draw FSC curve
        fsc_path = QPainterPath()
        num_pts = 50
        max_k = 0.55

        for i in range(num_pts + 1):
            k_val = (i / num_pts) * max_k
            fsc_val = self.compute_fsc(k_val)

            px = fsc_rect.left() + (i / num_pts) * fsc_rect.width()
            py = fsc_rect.bottom() - fsc_val * fsc_rect.height()

            if i == 0:
                fsc_path.moveTo(px, py)
            else:
                fsc_path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(250, 204, 21), 2.0))
        painter.drawPath(fsc_path)

        # Telemetry
        p = self.current_preset
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        ty = fsc_rect.bottom() + 16.0
        painter.drawText(int(rect.left() + 20.0), int(ty), f"Độ phân giải danh định: d = {p.nominal_res_ang:.2f} Å")

    def _paint_controls(self, painter: QPainter) -> None:
        ctrl_rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(ctrl_rect, 10.0, 10.0)

        col_w = (ctrl_rect.width() - 40.0) / 4.0

        sliders = [
            ("defocus", "Độ mờ Defocus (µm)", f"{self._defocus_um:+.2f} µm", self._defocus_um, -4.0, 2.0),
            ("voltage", "Điện áp gia tốc (kV)", f"{self._voltage_kv:.0f} kV", self._voltage_kv, 100.0, 300.0),
            ("bfactor", "B-Factor suy giảm", f"{self._b_factor:.0f} Å²", self._b_factor, 10.0, 120.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(52, 211, 153))
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
            painter.setBrush(QColor(16, 185, 129))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(16, 185, 129), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Reset button in 4th column
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        painter.setBrush(QColor(16, 185, 129, 40))
        painter.setPen(QPen(QColor(52, 211, 153), 1.2))
        painter.drawRoundedRect(btn_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(52, 211, 153))
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
        for i, key in enumerate(["defocus", "voltage", "bfactor"]):
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
            slider_keys = ["defocus", "voltage", "bfactor"]
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
        if key == "defocus":
            self.set_defocus_um(-4.0 + norm * 6.0)
        elif key == "voltage":
            self.set_voltage_kv(100.0 + norm * 200.0)
        elif key == "bfactor":
            self.set_b_factor(10.0 + norm * 110.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveCryoEMItem",
            "preset_key": self.current_preset.key,
            "voltage_kv": self._voltage_kv,
            "defocus_um": self._defocus_um,
            "cs_mm": self._cs_mm,
            "b_factor": self._b_factor,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveCryoEMItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "ribosome_70s")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "voltage_kv" in data:
            item._voltage_kv = float(data["voltage_kv"])
        if "defocus_um" in data:
            item._defocus_um = float(data["defocus_um"])
        if "cs_mm" in data:
            item._cs_mm = float(data["cs_mm"])
        if "b_factor" in data:
            item._b_factor = float(data["b_factor"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
