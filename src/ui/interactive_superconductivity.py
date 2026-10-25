"""
AirTeach AI - Interactive Superconductivity, Meissner Effect & Quantum Levitation Simulator
==========================================================================================
Authentic BCS / London equations, Abrikosov flux pinning & Ginzburg-Landau phenomenology:
- Critical Field & Phase Boundary:
    B_c(T) = B_c(0) * [1 - (T / T_c)^2]
- London Penetration Depth (London & London, 1935):
    λ_L(T) = λ_0 / sqrt(1 - (T / T_c)^4)
    B(x) = B_ext * exp(-x / λ_L)
- Ginzburg-Landau Coherence & Type Classification:
    κ = λ / ξ
    κ < 1 / sqrt(2) ≈ 0.707: Type I (Complete Meissner, abrupt first-order transition at B_c)
    κ > 1 / sqrt(2): Type II (Two critical fields B_c1, B_c2; mixed Abrikosov vortex state)
- Quantized Magnetic Flux Vortices:
    Φ_0 = h / (2 * e) ≈ 2.0678e-15 Wb
    Normal vortex cores of radius ≈ ξ surrounded by circulating supercurrents.
- Quantum Levitation & Flux Pinning:
    Pinning energy barrier U_p traps vortices at defect sites, creating a 3D restoring force:
      F_restore = -k_p * (z - z_0) - γ * dz/dt
    enabling stable levitation and inverted suspension without active power.
- 5 Curriculum Presets:
    1. TYPE1_MEISSNER: Lead/Mercury (T_c = 7.2 K, Type I), total flux expulsion outside bulk.
    2. TYPE2_FLUX_PINNING: YBCO (T_c = 93 K, Type II, mixed state), pinned vortices & stable levitation.
    3. THERMAL_QUENCH: LN2 evaporation warming (77 K -> 110 K), sudden Meissner breakdown.
    4. MAGNETIC_BREAKDOWN: High field B > B_c2 driving superconductor into normal metallic state.
    5. QUANTUM_TRACK_LOOP: Levitating magnet locked stably along magnetic guideway rails.
- Touchless webcam air gestures (air-tap, magnet dragging, sliders), bimanual manipulation,
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

# Fundamental physical constants
FLUX_QUANTUM_PHI0: float = 2.067833848e-15  # Wb (Tesla * m^2)
BOLTZMANN_K: float = 1.380649e-23           # J/K
MU_0: float = 4.0 * math.pi * 1e-7          # H/m


@dataclass
class SuperconductorPreset:
    """Preset configuration for educational superconductivity & levitation experiments."""
    key: str
    name: str
    desc: str
    material: str
    tc_k: float
    bc0_t: float
    bc2_t: float
    lambda0_nm: float
    xi0_nm: float
    is_type2: bool
    default_temp_k: float
    default_field_t: float
    pinning_density: float


PRESETS: List[SuperconductorPreset] = [
    SuperconductorPreset(
        key="type1_meissner",
        name="Loại I: Hiệu ứng Meissner (Chì Pb)",
        desc="Chất siêu dẫn loại I (κ < 0.707). Từ trường ngoài bị đẩy hoàn toàn khỏi thể tích vật thể (B=0)",
        material="Chì (Pb, Type I)",
        tc_k=7.2,
        bc0_t=0.08,
        bc2_t=0.08,
        lambda0_nm=39.0,
        xi0_nm=83.0,
        is_type2=False,
        default_temp_k=4.2,
        default_field_t=0.03,
        pinning_density=0.0,
    ),
    SuperconductorPreset(
        key="type2_flux_pinning",
        name="Loại II: Găm từ thông & Bay lượng tử (YBCO)",
        desc="Gốm siêu dẫn nhiệt độ cao YBCO. Các xoáy từ thông lượng tử (Abrikosov) bị găm chặt vào khuyết tật tinh thể",
        material="YBa₂Cu₃O₇ (YBCO, Type II)",
        tc_k=93.0,
        bc0_t=0.1,
        bc2_t=120.0,
        lambda0_nm=150.0,
        xi0_nm=1.5,
        is_type2=True,
        default_temp_k=77.0,
        default_field_t=0.5,
        pinning_density=0.85,
    ),
    SuperconductorPreset(
        key="thermal_quench",
        name="Sự bay hơi Nitơ lỏng & Mất siêu dẫn",
        desc="Nhiệt độ nóng dần từ 77K vượt qua nhiệt độ tới hạn Tc=93K, từ trường đột ngột xuyên qua và nam châm rơi xuống",
        material="YBCO (Nóng dần qua Tc)",
        tc_k=93.0,
        bc0_t=0.1,
        bc2_t=120.0,
        lambda0_nm=150.0,
        xi0_nm=1.5,
        is_type2=True,
        default_temp_k=88.0,
        default_field_t=0.4,
        pinning_density=0.8,
    ),
    SuperconductorPreset(
        key="magnetic_breakdown",
        name="Phá hủy siêu dẫn bằng từ trường cao",
        desc="Từ trường ngoài vượt quá giới hạn trên Bc2, năng lượng từ phá vỡ các cặp Cooper chuyển về kim loại bình thường",
        material="Nb-Ti Hợp kim (Type II)",
        tc_k=9.8,
        bc0_t=0.05,
        bc2_t=15.0,
        lambda0_nm=240.0,
        xi0_nm=5.0,
        is_type2=True,
        default_temp_k=4.2,
        default_field_t=12.0,
        pinning_density=0.9,
    ),
    SuperconductorPreset(
        key="quantum_track_loop",
        name="Đường ray đệm từ lượng tử (MagLev Rail)",
        desc="Nam châm bay ổn định khóa chặt cả chiều ngang lẫn phương đứng dọc theo đường ray từ vĩnh cửu",
        material="YBCO Đệm từ MagLev",
        tc_k=93.0,
        bc0_t=0.1,
        bc2_t=120.0,
        lambda0_nm=150.0,
        xi0_nm=1.5,
        is_type2=True,
        default_temp_k=77.0,
        default_field_t=0.6,
        pinning_density=0.95,
    ),
]


class InteractiveSuperconductivityItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Superconductivity, Meissner Effect,
    London Penetration Depth, Abrikosov Flux Pinning, and Quantum Levitation.
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
        self._preset_index: int = 1  # Default to YBCO Quantum Levitation
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        p = PRESETS[self._preset_index]
        self._temp_k: float = p.default_temp_k
        self._b_ext: float = p.default_field_t
        self._pinning: float = p.pinning_density

        # Levitating permanent magnet dynamic state: height z (mm) and velocity dz (mm/s)
        self._magnet_z: float = 38.0
        self._magnet_dz: float = 0.0
        self._target_equilibrium_z: float = 38.0
        self._is_dragging_magnet: bool = False
        self._drag_start_y: float = 0.0

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance continuous dynamic trajectory."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

    @property
    def current_preset(self) -> SuperconductorPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active superconductivity preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._temp_k = p.default_temp_k
            self._b_ext = p.default_field_t
            self._pinning = p.pinning_density
            self._magnet_z = 38.0
            self._magnet_dz = 0.0
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_temperature(self, t_kelvin: float) -> None:
        """Set temperature in Kelvin."""
        p = self.current_preset
        max_t = 150.0 if p.tc_k > 50.0 else 20.0
        self._temp_k = max(0.1, min(t_kelvin, max_t))
        self.property_changed.emit("temperature", self._temp_k)
        self.update()

    def set_magnetic_field(self, b_tesla: float) -> None:
        """Set external magnetic field in Tesla."""
        self._b_ext = max(0.0, min(b_tesla, 25.0))
        self.property_changed.emit("b_ext", self._b_ext)
        self.update()

    def set_pinning_density(self, density: float) -> None:
        """Set pinning defect density (0.0 to 1.0)."""
        self._pinning = max(0.0, min(density, 1.0))
        self.property_changed.emit("pinning", self._pinning)
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

    def compute_critical_fields(self) -> Tuple[float, float, float]:
        """
        Compute thermodynamic Bc(T), lower Bc1(T), and upper Bc2(T) in Tesla.
        Returns: (Bc, Bc1, Bc2)
        """
        p = self.current_preset
        t = self._temp_k
        tc = p.tc_k

        if t >= tc:
            return 0.0, 0.0, 0.0

        t_ratio = t / tc
        t_factor = max(0.0, 1.0 - t_ratio**2)
        bc = p.bc0_t * t_factor

        if not p.is_type2:
            return bc, bc, bc

        # Type II: Bc1 and Bc2
        bc2 = p.bc2_t * t_factor
        kappa = p.lambda0_nm / max(0.1, p.xi0_nm)
        # Bc1 ≈ (Bc / (sqrt(2) * kappa)) * ln(kappa)
        bc1 = (bc / (math.sqrt(2.0) * kappa)) * math.log(max(1.1, kappa))
        return bc, bc1, bc2

    def compute_london_penetration_depth(self) -> float:
        """Compute London penetration depth λ_L(T) in nanometers."""
        p = self.current_preset
        t = self._temp_k
        tc = p.tc_k

        if t >= tc:
            return 1e6  # Infinite penetration (normal metal)

        t_ratio = t / tc
        denom = max(1e-4, 1.0 - t_ratio**4)
        return float(p.lambda0_nm / math.sqrt(denom))

    def compute_coherence_length(self) -> float:
        """Compute Ginzburg-Landau coherence length ξ(T) in nanometers."""
        p = self.current_preset
        t = self._temp_k
        tc = p.tc_k

        if t >= tc:
            return 0.0

        t_ratio = t / tc
        denom = max(1e-4, 1.0 - t_ratio)
        return float(p.xi0_nm / math.sqrt(denom))

    def get_superconducting_state(self) -> str:
        """
        Evaluate physical state:
        'meissner': Perfect diamagnetic expulsion (B < Bc1 and T < Tc)
        'vortex': Mixed / Shubnikov state with Abrikosov flux tubes (Bc1 <= B <= Bc2)
        'normal': Normal resistive metallic state (T >= Tc or B > Bc2)
        """
        p = self.current_preset
        if self._temp_k >= p.tc_k:
            return "normal"

        bc, bc1, bc2 = self.compute_critical_fields()

        if not p.is_type2:
            return "meissner" if self._b_ext <= bc else "normal"

        if self._b_ext < bc1:
            return "meissner"
        elif self._b_ext <= bc2:
            return "vortex"
        else:
            return "normal"

    def step_physics(self, dt: float) -> None:
        """Advance levitating magnet dynamics and thermal drift."""
        if self._is_paused or dt <= 0.0:
            return

        self._sim_time += dt
        state = self.get_superconducting_state()

        # If in Thermal Quench preset, simulate LN2 evaporation warming (0.5 K/s)
        if self.current_preset.key == "thermal_quench":
            self._temp_k = min(115.0, self._temp_k + 0.35 * dt)

        # Dynamic levitation height calculation
        # Equilibrium height depends on state and pinning
        if state == "normal":
            # Magnetic field penetrates fully; magnet drops onto superconductor surface (z -> 0 mm)
            target_z = 3.0  # Touching surface
            k_spring = 25.0
            damping = 10.0
        elif state == "meissner":
            # Perfect diamagnetism pushes magnet up strongly
            target_z = 42.0
            k_spring = 40.0
            damping = 4.5
        else:  # "vortex" (mixed state with flux pinning)
            # Pinned vortices lock magnet at current frozen height
            # Height modulated by pinning density
            target_z = 25.0 + 20.0 * self._pinning
            k_spring = 60.0 * self._pinning + 20.0
            damping = 8.0 * self._pinning + 3.0

        self._target_equilibrium_z = target_z

        if not self._is_dragging_magnet:
            # Damped harmonic restoring force
            error_z = self._target_equilibrium_z - self._magnet_z
            accel_z = k_spring * error_z - damping * self._magnet_dz
            self._magnet_dz += accel_z * dt
            self._magnet_z += self._magnet_dz * dt
            self._magnet_z = max(2.0, min(self._magnet_z, 75.0))

        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, 56.0)

    def _canvas_rect(self) -> QRectF:
        return QRectF(15.0, 65.0, 600.0, 420.0)

    def _side_panel_rect(self) -> QRectF:
        return QRectF(625.0, 65.0, 300.0, 420.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, SuperconductorPreset]]:
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
        border_pen = QPen(QColor(56, 189, 248, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_levitation_canvas(painter)
        self._paint_phase_diagram(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ HIỆU ỨNG MEISSNER & BAY LƯỢNG TỬ (SUPERCONDUCTIVITY)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        state = self.get_superconducting_state().upper()
        painter.drawText(20, 44, f"Vật liệu: {p.material}  |  Trạng thái: {state}  |  Tc = {p.tc_k:.1f} K")

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(14, 165, 233, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(56, 189, 248), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            labels = ["Loại I (Pb)", "Găm từ (YBCO)", "Nóng dần", "Từ trường cao", "Đường ray"]
            label = labels[i] if i < len(labels) else preset.name[:8]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        state_color = {
            "meissner": QColor(16, 185, 129),     # Green
            "vortex": QColor(56, 189, 248),       # Cyan
            "normal": QColor(239, 68, 68),        # Red
        }[self.get_superconducting_state()]

        painter.setBrush(QColor(state_color.red(), state_color.green(), state_color.blue(), 40))
        painter.setPen(QPen(state_color, 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(state_color)
        status_labels = {
            "meissner": "🛡️ MEISSNER (B=0)",
            "vortex": "🌀 XOÁY TỪ THÔNG",
            "normal": "⚠️ BÌNH THƯỜNG",
        }
        painter.drawText(badge_rect, Qt.AlignCenter, status_labels[self.get_superconducting_state()])

    def _paint_levitation_canvas(self, painter: QPainter) -> None:
        c_rect = self._canvas_rect()

        # Canvas Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(c_rect, 10.0, 10.0)

        # Cryogenic Nitrogen Vapor Effect (if T <= 85 K)
        if self._temp_k <= 85.0:
            for vi in range(5):
                vx = c_rect.left() + 100.0 + vi * 80.0 + math.sin(self._sim_time * 2.0 + vi) * 15.0
                vy = c_rect.bottom() - 60.0 - (vi % 3) * 15.0
                vapor_grad = QRadialGradient(vx, vy, 45.0)
                vapor_grad.setColorAt(0.0, QColor(224, 242, 254, 45))
                vapor_grad.setColorAt(1.0, QColor(224, 242, 254, 0))
                painter.setBrush(vapor_grad)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(vx, vy), 50.0, 22.0)

        # 1. Superconductor Puck Geometry
        puck_w = 260.0
        puck_h = 36.0
        puck_x = c_rect.center().x() - puck_w / 2.0
        puck_y = c_rect.bottom() - 110.0
        puck_rect = QRectF(puck_x, puck_y, puck_w, puck_h)

        # Puck Gradient
        state = self.get_superconducting_state()
        puck_grad = QLinearGradient(puck_rect.left(), puck_rect.top(), puck_rect.left(), puck_rect.bottom())
        if state == "normal":
            puck_grad.setColorAt(0.0, QColor(100, 116, 139))
            puck_grad.setColorAt(1.0, QColor(51, 65, 85))
        else:
            puck_grad.setColorAt(0.0, QColor(30, 58, 138))
            puck_grad.setColorAt(0.5, QColor(2, 132, 199))
            puck_grad.setColorAt(1.0, QColor(14, 116, 144))

        painter.setBrush(puck_grad)
        painter.setPen(QPen(QColor(56, 189, 248) if state != "normal" else QColor(148, 163, 184), 1.8))
        painter.drawRoundedRect(puck_rect, 8.0, 8.0)

        # Puck Label
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        puck_title = f"{self.current_preset.material} (T = {self._temp_k:.1f} K)"
        painter.drawText(puck_rect, Qt.AlignCenter, puck_title)

        # 2. Levitating Neodymium Magnet
        mag_w = 140.0
        mag_h = 32.0
        mag_center_x = c_rect.center().x()
        # Scale z (mm) to pixels (approx 2.5 px per mm)
        mag_y = puck_y - (self._magnet_z * 2.4) - mag_h
        mag_rect = QRectF(mag_center_x - mag_w / 2.0, mag_y, mag_w, mag_h)

        # Magnet Halo Glow
        glow_grad = QRadialGradient(mag_rect.center(), mag_w)
        glow_grad.setColorAt(0.0, QColor(236, 72, 153, 60))
        glow_grad.setColorAt(1.0, QColor(236, 72, 153, 0))
        painter.setBrush(glow_grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(mag_rect.center(), mag_w * 0.9, mag_h * 1.8)

        # Magnet Body: North (Red) & South (Blue) split
        mag_left = QRectF(mag_rect.left(), mag_rect.top(), mag_w / 2.0, mag_h)
        mag_right = QRectF(mag_rect.left() + mag_w / 2.0, mag_rect.top(), mag_w / 2.0, mag_h)

        painter.setBrush(QColor(225, 29, 72))  # North Pole (Red)
        painter.setPen(QPen(QColor(244, 63, 94), 1.2))
        painter.drawRoundedRect(mag_left, 5.0, 5.0)

        painter.setBrush(QColor(37, 99, 235))   # South Pole (Blue)
        painter.setPen(QPen(QColor(96, 165, 250), 1.2))
        painter.drawRoundedRect(mag_right, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(mag_left, Qt.AlignCenter, "N (Cực Bắc)")
        painter.drawText(mag_right, Qt.AlignCenter, "S (Cực Nam)")

        # 3. Magnetic Flux Lines
        self._paint_flux_lines(painter, c_rect, puck_rect, mag_rect, state)

        # 4. Trapped Quantized Vortices (in Vortex state)
        if state == "vortex":
            self._paint_pinned_vortices(painter, puck_rect)

        # Levitation Height Indicator HUD
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(244, 114, 182))
        painter.drawText(
            QRectF(c_rect.left() + 15.0, c_rect.top() + 15.0, 240.0, 20.0),
            Qt.AlignLeft,
            f"Độ cao bay z = {self._magnet_z:.1f} mm  (Vận tốc: {self._magnet_dz:.1f} mm/s)"
        )

    def _paint_flux_lines(
        self,
        painter: QPainter,
        c_rect: QRectF,
        puck_rect: QRectF,
        mag_rect: QRectF,
        state: str,
    ) -> None:
        """Draw curved magnetic field lines bending around or threading the puck."""
        flux_pen = QPen(QColor(56, 189, 248, 140), 1.2, Qt.DashLine if state == "vortex" else Qt.SolidLine)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(flux_pen)

        mag_cx = mag_rect.center().x()
        mag_bot = mag_rect.bottom()
        puck_top = puck_rect.top()

        for offset in [-60.0, -35.0, -15.0, 15.0, 35.0, 60.0]:
            path = QPainterPath()
            sx = mag_cx + offset
            sy = mag_bot
            path.moveTo(sx, sy)

            if state == "meissner":
                # Expelled outside: lines curve away around the edges of the puck
                deflect = 70.0 if offset > 0 else -70.0
                ctrl_x = sx + deflect
                ctrl_y = (sy + puck_top) / 2.0 + 10.0
                end_x = puck_rect.center().x() + (offset * 2.2)
                end_y = puck_rect.bottom() + 25.0
                path.quadTo(ctrl_x, ctrl_y, end_x, end_y)

            elif state == "vortex":
                # Some lines penetrate through quantized flux channels
                mid_y = (sy + puck_top) / 2.0
                path.lineTo(sx, puck_top)
                path.lineTo(sx, puck_rect.bottom())
                path.lineTo(sx + offset * 0.3, puck_rect.bottom() + 30.0)

            else:  # "normal"
                # Uniform unhindered penetration
                path.lineTo(sx, puck_rect.bottom() + 35.0)

            painter.drawPath(path)

    def _paint_pinned_vortices(self, painter: QPainter, puck_rect: QRectF) -> None:
        """Render glowing Abrikosov vortex flux tubes inside the superconductor."""
        num_vortices = int(6 + self._pinning * 10)
        puck_w = puck_rect.width()
        step = puck_w / (num_vortices + 1)

        for i in range(num_vortices):
            vx = puck_rect.left() + (i + 1) * step
            # Normal core (radius ≈ xi)
            painter.setBrush(QColor(239, 68, 68, 200))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(vx - 2.0, puck_rect.top(), 4.0, puck_rect.height()))

            # Screening supercurrent arrows (circulating)
            painter.setPen(QPen(QColor(250, 204, 21), 1.0))
            painter.drawLine(int(vx - 7.0), int(puck_rect.center().y()), int(vx - 7.0), int(puck_rect.center().y() - 6))
            painter.drawLine(int(vx + 7.0), int(puck_rect.center().y()), int(vx + 7.0), int(puck_rect.center().y() + 6))

    def _paint_phase_diagram(self, painter: QPainter) -> None:
        s_rect = self._side_panel_rect()

        # Side Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(s_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(s_rect.left(), s_rect.top() + 12.0, s_rect.width(), 20.0), Qt.AlignCenter, "📊 GIẢN ĐỒ PHA B - T (PHASE DIAGRAM)")

        # Phase plot box
        plot_rect = QRectF(s_rect.left() + 25.0, s_rect.top() + 45.0, s_rect.width() - 45.0, 180.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(plot_rect)

        # Draw Phase Curves: Bc(T) or Bc1(T) and Bc2(T)
        p = self.current_preset
        max_t = p.tc_k * 1.25
        max_b = (p.bc2_t if p.is_type2 else p.bc0_t) * 1.2
        max_b = max(0.1, max_b)

        # Bc2 curve (or Bc curve for Type I)
        path_bc2 = QPainterPath()
        path_bc1 = QPainterPath()

        num_pts = 40
        for i in range(num_pts + 1):
            t_val = (i / num_pts) * p.tc_k
            px = plot_rect.left() + (t_val / max_t) * plot_rect.width()

            # Bc2
            b2 = p.bc2_t * (1.0 - (t_val / p.tc_k)**2) if p.is_type2 else p.bc0_t * (1.0 - (t_val / p.tc_k)**2)
            py2 = plot_rect.bottom() - (b2 / max_b) * plot_rect.height()

            if i == 0:
                path_bc2.moveTo(px, py2)
            else:
                path_bc2.lineTo(px, py2)

            # Bc1 (if Type II)
            if p.is_type2:
                b1 = (b2 / 15.0) * math.log(max(1.1, p.lambda0_nm / p.xi0_nm))
                py1 = plot_rect.bottom() - (b1 / max_b) * plot_rect.height()
                if i == 0:
                    path_bc1.moveTo(px, py1)
                else:
                    path_bc1.lineTo(px, py1)

        # Draw curves
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(239, 68, 68), 1.5))  # Bc2 / Bc boundary
        painter.drawPath(path_bc2)

        if p.is_type2:
            painter.setPen(QPen(QColor(16, 185, 129), 1.5, Qt.DashLine))  # Bc1 boundary
            painter.drawPath(path_bc1)

        # Plot current state point (T, B)
        curr_t = self._temp_k
        curr_b = self._b_ext
        pt_x = plot_rect.left() + min(1.0, curr_t / max_t) * plot_rect.width()
        pt_y = plot_rect.bottom() - min(1.0, curr_b / max_b) * plot_rect.height()

        painter.setBrush(QColor(250, 204, 21))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(pt_x, pt_y), 5.0, 5.0)

        # Physical Parameters HUD Text
        bc, bc1, bc2 = self.compute_critical_fields()
        lambda_l = self.compute_london_penetration_depth()
        xi = self.compute_coherence_length()
        kappa = p.lambda0_nm / max(0.1, p.xi0_nm)

        hud_y = plot_rect.bottom() + 18.0
        lines = [
            f"Nhiệt độ hiện tại: T = {self._temp_k:.1f} K",
            f"Từ trường ngoài: B₀ = {self._b_ext:.2f} T",
            f"Độ sâu London: λ_L(T) = {lambda_l:.1f} nm" if lambda_l < 9999 else "Độ sâu London: λ_L(T) = ∞ (Kim loại)",
            f"Độ dài kết hợp: ξ(T) = {xi:.1f} nm",
            f"Tham số GL κ = λ/ξ = {kappa:.2f} ({'Loại II' if p.is_type2 else 'Loại I'})",
            f"Từ trường tới hạn Bc1 = {bc1:.2f} T, Bc2 = {bc2:.2f} T",
            f"Lực khôi phục găm: k_p = {60.0 * self._pinning + 20.0:.0f} N/m",
        ]

        painter.setFont(QFont("Segoe UI", 8))
        for line in lines:
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(s_rect.left() + 20.0), int(hud_y), line)
            hud_y += 18.0

    def _paint_controls(self, painter: QPainter) -> None:
        ctrl_rect = self._controls_rect()

        # Controls Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(ctrl_rect, 10.0, 10.0)

        col_w = (ctrl_rect.width() - 40.0) / 4.0
        p = self.current_preset
        max_t = 150.0 if p.tc_k > 50.0 else 20.0

        sliders = [
            ("temp", "Nhiệt độ T (K)", f"{self._temp_k:.1f} K", self._temp_k, 0.1, max_t),
            ("field", "Từ trường ngoài B₀ (T)", f"{self._b_ext:.2f} T", self._b_ext, 0.0, 20.0),
            ("pinning", "Mật độ găm khuyết tật", f"{self._pinning * 100:.0f}%", self._pinning, 0.0, 1.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(56, 189, 248))
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
            painter.setBrush(QColor(14, 165, 233))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(14, 165, 233), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Reset button in 4th column
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        painter.setBrush(QColor(14, 165, 233, 40))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(btn_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
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
        for i, key in enumerate(["temp", "field", "pinning"]):
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

        # Check Magnet Direct Dragging (to feel restoring force)
        c_rect = self._canvas_rect()
        puck_y = c_rect.bottom() - 110.0
        mag_y = puck_y - (self._magnet_z * 2.4) - 32.0
        mag_rect = QRectF(c_rect.center().x() - 70.0, mag_y - 10.0, 140.0, 52.0)
        if mag_rect.contains(pos):
            self._is_dragging_magnet = True
            self._drag_start_y = pos.y()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["temp", "field", "pinning"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        if self._is_dragging_magnet:
            c_rect = self._canvas_rect()
            puck_y = c_rect.bottom() - 110.0
            # Dragging magnet directly
            dy = self._drag_start_y - pos.y()
            delta_z = dy / 2.4
            self._magnet_z = max(2.0, min(self._magnet_z + delta_z, 75.0))
            self._drag_start_y = pos.y()
            self._magnet_dz = 0.0
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        self._is_dragging_magnet = False
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        p = self.current_preset
        max_t = 150.0 if p.tc_k > 50.0 else 20.0

        if key == "temp":
            self.set_temperature(0.1 + norm * (max_t - 0.1))
        elif key == "field":
            self.set_magnetic_field(0.0 + norm * 20.0)
        elif key == "pinning":
            self.set_pinning_density(norm)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveSuperconductivityItem",
            "preset_key": self.current_preset.key,
            "temp_k": self._temp_k,
            "b_ext": self._b_ext,
            "pinning": self._pinning,
            "magnet_z": self._magnet_z,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveSuperconductivityItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "type2_flux_pinning")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "temp_k" in data:
            item._temp_k = float(data["temp_k"])
        if "b_ext" in data:
            item._b_ext = float(data["b_ext"])
        if "pinning" in data:
            item._pinning = float(data["pinning"])
        if "magnet_z" in data:
            item._magnet_z = float(data["magnet_z"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
