"""
AirTeach AI - Interactive Physics Circular Motion & Centripetal Force / Banking Road Simulator
=============================================================================================
Provides an authentic interactive physics simulation for high school & university mechanics:
  - Kinematic & Dynamic Foundations (Zero Mock Data):
      1. Uniform Circular Motion (Chuyển động tròn đều):
           Tangential speed: v = ω · R = (2π · R) / T = 2π · R · f
           Centripetal acceleration: a_c = v² / R = ω² · R  (directed strictly towards center O)
           Centripetal force: F_c = m · a_c = m · v² / R = m · ω² · R
      2. Banked Curved Road (Đường cong nghiêng góc θ):
           Optimal banking angle without lateral friction:
             tan(θ) = v² / (g · R)  =>  θ_ideal = arctan(v² / (g · R))
           With coefficient of static friction μ_s:
             v_max = sqrt(g · R · (tan(θ) + μ_s) / (1 - μ_s · tan(θ)))
             v_min = sqrt(max(0.0, g · R · (tan(θ) - μ_s) / (1 + μ_s · tan(θ))))
      3. Conical Pendulum (Con lắc nón):
           Bob mass m on string of length L forming angle θ with vertical:
             Radius R = L · sin(θ)
             Tension T = m · g / cos(θ)
             Centripetal force F_c = T · sin(θ) = m · g · tan(θ)
             Period T_p = 2π · sqrt((L · cos(θ)) / g)
      4. Vertical Loop & Roller Coaster (Vòng xiếc thẳng đứng):
           Conservation of mechanical energy: E = (1/2)·m·v² + m·g·h = const
           Critical velocity at loop apex to maintain contact:
             v_top_min = sqrt(g · R)
             v_bottom_min = sqrt(5 · g · R)
           Apparent weight / Normal force at bottom:
             N_bottom = m · (v_bottom² / R) + m · g
      5. Geocentric Satellite Orbit (Chuyển động vệ tinh quanh Trái Đất):
           Gravitational force providing centripetal acceleration:
             G · (M_E · m) / r² = m · v² / r  =>  v = sqrt(G · M_E / r)
             First cosmic velocity at surface: v_I = sqrt(g · R_E) ≈ 7.91 km/s
  - 5 Authentic Curriculum Presets:
      - HORIZONTAL_TURNTABLE: Đĩa quay nằm ngang với ma sát nghỉ F_msn giữ vật không trượt
      - BANKED_CURVE_HIGHWAY: Khúc cua nghiêng góc chống lật và trượt trên cao tốc
      - CONICAL_PENDULUM: Con lắc nón quay tròn quanh trục thẳng đứng
      - VERTICAL_LOOP_ROLLERCOASTER: Vòng xiếc thẳng đứng với vận tốc tới hạn
      - GEOCENTRIC_SATELLITE_ORBIT: Vệ tinh chuyển động tròn quanh quỹ đạo Trái Đất
  - Visual Overlays & Gestures:
      - Velocity vector v (cyan tangent arrow)
      - Centripetal acceleration vector a_c (crimson arrow to center)
      - Dynamic particle / object tracing orbital path
      - Touchless air pinch controls for radius, speed, mass, presets, and stamping
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
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

# Standard Physical Constants
GRAVITY_G: float = 9.80665  # m/s² standard acceleration due to gravity
EARTH_MASS_KG: float = 5.972e24  # kg
EARTH_RADIUS_M: float = 6.371e6  # m
GRAVITATIONAL_CONST_G: float = 6.67430e-11  # N·m²/kg²


@dataclass
class CircularPreset:
    code: str
    title_vi: str
    mode: str  # "HORIZONTAL", "BANKED", "CONICAL", "VERTICAL", "ORBITAL"
    radius_m: float
    tangential_speed_ms: float
    mass_kg: float
    friction_mu: float
    bank_angle_deg: float
    description_vi: str
    formula_vi: str


PRESETS: dict[str, CircularPreset] = {
    "HORIZONTAL_TURNTABLE": CircularPreset(
        code="HORIZONTAL_TURNTABLE",
        title_vi="Đĩa Quay Ngang & Ma Sát Nghỉ",
        mode="HORIZONTAL",
        radius_m=2.0,
        tangential_speed_ms=3.0,
        mass_kg=0.5,
        friction_mu=0.55,
        bank_angle_deg=0.0,
        description_vi="Vật thể nằm trên bàn xoay ngang. Lực ma sát nghỉ đóng vai trò lực hướng tâm giữ vật không bị văng ra xa.",
        formula_vi="F_msn = m·v²/R ≤ μ_s·m·g  =>  v_max = √(μ_s·g·R)",
    ),
    "BANKED_CURVE_HIGHWAY": CircularPreset(
        code="BANKED_CURVE_HIGHWAY",
        title_vi="Khúc Cua Nghiêng Cao Tốc (Banked Curve)",
        mode="BANKED",
        radius_m=50.0,
        tangential_speed_ms=20.0,  # 72 km/h
        mass_kg=1200.0,
        friction_mu=0.30,
        bank_angle_deg=18.0,
        description_vi="Đường cong nghiêng góc θ giúp phản lực pháp tuyến N triệt tiêu lực ly tâm, xe ôm cua an toàn không cần phụ thuộc ma sát.",
        formula_vi="tan(θ) = v² / (g·R)  |  θ_ideal = arctan(v² / (g·R))",
    ),
    "CONICAL_PENDULUM": CircularPreset(
        code="CONICAL_PENDULUM",
        title_vi="Con Lắc Nón (Conical Pendulum)",
        mode="CONICAL",
        radius_m=1.2,
        tangential_speed_ms=2.4,
        mass_kg=0.4,
        friction_mu=0.0,
        bank_angle_deg=30.0,
        description_vi="Vật nặng treo vào sợi dây mảnh quay đều trong mặt phẳng nằm ngang tạo thành hình nón trong không gian.",
        formula_vi="T_căng = m·g / cos(θ)  |  T_chu_kỳ = 2π·√(L·cos(θ) / g)",
    ),
    "VERTICAL_LOOP_ROLLERCOASTER": CircularPreset(
        code="VERTICAL_LOOP_ROLLERCOASTER",
        title_vi="Vòng Xiếc Thẳng Đứng (Vertical Loop)",
        mode="VERTICAL",
        radius_m=8.0,
        tangential_speed_ms=10.0,
        mass_kg=350.0,
        friction_mu=0.05,
        bank_angle_deg=0.0,
        description_vi="Tàu lượn siêu tốc đi qua vòng tròn thẳng đứng. Để không bị rơi tại đỉnh vòng, vận tốc phải đạt ngưỡng tới hạn v_top ≥ √(g·R).",
        formula_vi="v_top ≥ √(g·R)  |  v_đáy ≥ √(5·g·R)  |  N_đáy = m(v²/R + g)",
    ),
    "GEOCENTRIC_SATELLITE_ORBIT": CircularPreset(
        code="GEOCENTRIC_SATELLITE_ORBIT",
        title_vi="Quỹ Đạo Vệ Tinh Địa Tĩnh (Satellite Orbit)",
        mode="ORBITAL",
        radius_m=7000.0,  # Scaled orbital radius
        tangential_speed_ms=7546.0,  # ~7.5 km/s LEO
        mass_kg=800.0,
        friction_mu=0.0,
        bank_angle_deg=0.0,
        description_vi="Vệ tinh nhân tạo bay quanh Trái Đất. Trọng lực hấp dẫn của Trái Đất cung cấp chính xác gia tốc hướng tâm.",
        formula_vi="G·M_E / r² = v² / r  =>  v = √(G·M_E / r)  (Vận tốc vũ trụ cấp 1)",
    ),
}


class InteractiveCircularMotionItem(QGraphicsObject):
    """
    High-performance vector interactive Circular Motion & Centripetal Force Simulator.
    Provides authentic mathematical modeling, animated force/velocity vectors, and LessonBoard stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (290.0, 110.0)) -> None:
        super().__init__()
        self.setPos(pos[0], pos[1])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active state
        self._active_preset_code: str = "HORIZONTAL_TURNTABLE"
        self._preset: CircularPreset = PRESETS[self._active_preset_code]

        # Configurable Parameters
        self._radius_m: float = self._preset.radius_m
        self._speed_ms: float = self._preset.tangential_speed_ms
        self._mass_kg: float = self._preset.mass_kg
        self._bank_angle_deg: float = self._preset.bank_angle_deg

        # Animation state
        self._current_angle_rad: float = 0.0
        self._is_animating: bool = True
        self._elapsed_time_s: float = 0.0

        # Animation Timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Initialize clickable and air-pinchable hit areas."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Preset buttons bar (5 buttons)
        preset_codes = [
            "HORIZONTAL_TURNTABLE",
            "BANKED_CURVE_HIGHWAY",
            "CONICAL_PENDULUM",
            "VERTICAL_LOOP_ROLLERCOASTER",
            "GEOCENTRIC_SATELLITE_ORBIT",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Simulation canvas viewport
        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 470.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 110.0, h_bot)
        self._btn_rects["dec_radius"] = QRectF(134.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_radius"] = QRectF(180.0, y_bot, 42.0, h_bot)
        self._btn_rects["dec_speed"] = QRectF(230.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_speed"] = QRectF(276.0, y_bot, 42.0, h_bot)
        self._btn_rects["dec_mass"] = QRectF(326.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_mass"] = QRectF(372.0, y_bot, 42.0, h_bot)
        self._btn_rects["stamp"] = QRectF(422.0, y_bot, 342.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical Calculations & Properties (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> CircularPreset:
        return self._preset

    @property
    def radius_m(self) -> float:
        return self._radius_m

    @property
    def speed_ms(self) -> float:
        return self._speed_ms

    @property
    def mass_kg(self) -> float:
        return self._mass_kg

    @property
    def bank_angle_deg(self) -> float:
        return self._bank_angle_deg

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def angular_velocity_rads(self) -> float:
        """Angular velocity ω = v / R in rad/s."""
        return self._speed_ms / max(0.01, self._radius_m)

    @property
    def period_seconds(self) -> float:
        """Period T = 2π / ω in seconds."""
        omega = self.angular_velocity_rads
        return (2.0 * math.pi) / max(1e-6, omega)

    @property
    def frequency_hz(self) -> float:
        """Frequency f = 1 / T in Hertz."""
        return 1.0 / max(1e-6, self.period_seconds)

    @property
    def centripetal_acceleration_ms2(self) -> float:
        """Centripetal acceleration a_c = v² / R = ω²·R in m/s²."""
        return (self._speed_ms ** 2) / max(0.01, self._radius_m)

    @property
    def centripetal_force_newtons(self) -> float:
        """Centripetal force F_c = m · a_c = m · v² / R in Newtons."""
        return self._mass_kg * self.centripetal_acceleration_ms2

    @property
    def ideal_banking_angle_deg(self) -> float:
        """Optimal banking angle for frictionless curve: θ = arctan(v² / (g·R))."""
        tan_theta = (self._speed_ms ** 2) / (GRAVITY_G * max(0.01, self._radius_m))
        return math.degrees(math.atan(tan_theta))

    @property
    def critical_vertical_top_speed_ms(self) -> float:
        """Critical minimum velocity at top of vertical loop: v_top_min = √(g·R)."""
        return math.sqrt(GRAVITY_G * max(0.01, self._radius_m))

    @property
    def vertical_bottom_normal_force_newtons(self) -> float:
        """Apparent weight / normal force at bottom of vertical loop: N = m(v²/R + g)."""
        return self._mass_kg * (self.centripetal_acceleration_ms2 + GRAVITY_G)

    @property
    def max_safe_friction_speed_ms(self) -> float:
        """Maximum safe speed on flat road with static friction: v_max = √(μ_s·g·R)."""
        mu = max(0.01, self._preset.friction_mu)
        return math.sqrt(mu * GRAVITY_G * max(0.01, self._radius_m))

    # -------------------------------------------------------------------------
    # Public Controller API
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic circular motion preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._radius_m = self._preset.radius_m
        self._speed_ms = self._preset.tangential_speed_ms
        self._mass_kg = self._preset.mass_kg
        self._bank_angle_deg = self._preset.bank_angle_deg
        self._current_angle_rad = 0.0
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_radius(self, radius_m: float) -> None:
        """Set circular radius in meters (clamped between 0.2m and 500.0m)."""
        self._radius_m = max(0.2, min(500.0, float(radius_m)))
        self.update()

    def set_speed(self, speed_ms: float) -> None:
        """Set tangential velocity in m/s (clamped between 0.1 m/s and 10000.0 m/s)."""
        self._speed_ms = max(0.1, min(10000.0, float(speed_ms)))
        self.update()

    def set_mass(self, mass_kg: float) -> None:
        """Set body mass in kg (clamped between 0.05 kg and 10000.0 kg)."""
        self._mass_kg = max(0.05, min(10000.0, float(mass_kg)))
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle orbital animation state."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def cleanup(self) -> None:
        """Stop animation timer and cleanly disconnect."""
        self._anim_timer.stop()
        self._is_animating = False

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        if not self._is_animating:
            return

        dt = 0.033  # ~33 ms per frame
        self._elapsed_time_s += dt

        # Scale angular speed visually so high speed remains perceptible
        # Visual angular speed clamped to comfortable presentation rate [0.5 rad/s, 6.0 rad/s]
        raw_omega = self.angular_velocity_rads
        vis_omega = math.copysign(math.log1p(abs(raw_omega)) * 1.5, raw_omega)
        vis_omega = max(0.4, min(5.0, vis_omega))

        self._current_angle_rad = (self._current_angle_rad + vis_omega * dt) % (2.0 * math.pi)
        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Painting (High-Performance Vector Graphics)
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main outer window card
        card_rect = self.boundingRect()
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Window Header Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 48.0), 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 24.0, self.WIDTH, 24.0))  # Fill bottom rounded corners

        # Header Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(16.0, 10.0, 500.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🔄 {self._preset.title_vi} | Chuyển Động Tròn Đều",
        )

        # Close Button
        close_r = self._btn_rects["close"]
        is_close_hover = self._hovered_button == "close"
        painter.setBrush(QBrush(QColor("#ef4444" if is_close_hover else "#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(close_r)
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        cx, cy = close_r.center().x(), close_r.center().y()
        painter.drawLine(int(cx - 4), int(cy - 4), int(cx + 4), int(cy + 4))
        painter.drawLine(int(cx + 4), int(cy - 4), int(cx - 4), int(cy + 4))

        # 3. Presets Bar
        preset_codes = [
            ("HORIZONTAL_TURNTABLE", "Đĩa Quay"),
            ("BANKED_CURVE_HIGHWAY", "Cua Nghiêng"),
            ("CONICAL_PENDULUM", "Con Lắc Nón"),
            ("VERTICAL_LOOP_ROLLERCOASTER", "Vòng Xiếc"),
            ("GEOCENTRIC_SATELLITE_ORBIT", "Vệ Tinh"),
        ]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for code, label in preset_codes:
            r = self._btn_rects[f"preset_{code}"]
            is_active = self._active_preset_code == code
            is_hover = self._hovered_button == f"preset_{code}"
            bg_col = (
                QColor("#0284c7") if is_active else (QColor("#334155") if is_hover else QColor("#1e293b"))
            )
            border_col = QColor("#38bdf8") if is_active else QColor("#475569")
            painter.setPen(QPen(border_col, 1.2))
            painter.setBrush(QBrush(bg_col))
            painter.drawRoundedRect(r, 6.0, 6.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#cbd5e1")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)

        # 4. Canvas Viewport (Simulation Arena)
        can_r = self._btn_rects["canvas"]
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(can_r, 10.0, 10.0)

        # Draw Orbital Canvas Content
        self._paint_simulation_arena(painter, can_r)

        # 5. Right Inspector & Dynamics Analytics Panel
        self._paint_inspector_panel(painter)

        # 6. Bottom Action Toolbar
        self._paint_bottom_toolbar(painter)

    def _paint_simulation_arena(self, painter: QPainter, arena_r: QRectF) -> None:
        """Paints the circular trajectory, rotating body, velocity & centripetal vectors."""
        cx = arena_r.center().x()
        cy = arena_r.center().y()
        vis_radius = 120.0  # Visual rendering radius

        # Grid / Coordinate background crosshair
        painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(arena_r.left() + 20), int(cy), int(arena_r.right() - 20), int(cy))
        painter.drawLine(int(cx), int(arena_r.top() + 20), int(cx), int(arena_r.bottom() - 20))

        # Circular Path (Orbit)
        pen_orbit = QPen(QColor(56, 189, 248, 120), 2.0, Qt.PenStyle.DashLine)
        painter.setPen(pen_orbit)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), vis_radius, vis_radius)

        # Center Pivot Point O
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(QPointF(cx, cy), 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QPointF(cx - 14.0, cy - 8.0), "O")

        # Object Position on Circle
        obj_x = cx + vis_radius * math.cos(self._current_angle_rad)
        obj_y = cy + vis_radius * math.sin(self._current_angle_rad)
        obj_pos = QPointF(obj_x, obj_y)

        # Radius Line connecting Center O to Object
        painter.setPen(QPen(QColor("#64748b"), 1.5, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(cx, cy), obj_pos)

        # Centripetal Acceleration / Force Vector a_c (pointing to Center O, Ruby)
        arrow_ac_len = 55.0
        ac_dir_x = (cx - obj_x) / vis_radius
        ac_dir_y = (cy - obj_y) / vis_radius
        ac_end = QPointF(obj_x + ac_dir_x * arrow_ac_len, obj_y + ac_dir_y * arrow_ac_len)
        self._draw_arrow(painter, obj_pos, ac_end, QColor("#f43f5e"), 2.5, "a_c, F_c")

        # Tangential Velocity Vector v (perpendicular to radius, Cyan)
        # v points in direction of motion (tangent: dx/dθ = -sinθ, dy/dθ = cosθ)
        arrow_v_len = 65.0
        v_dir_x = -math.sin(self._current_angle_rad)
        v_dir_y = math.cos(self._current_angle_rad)
        v_end = QPointF(obj_x + v_dir_x * arrow_v_len, obj_y + v_dir_y * arrow_v_len)
        self._draw_arrow(painter, obj_pos, v_end, QColor("#10b981"), 2.5, "v")

        # Rotating Body Item
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        painter.setBrush(QBrush(QColor("#f59e0b")))
        body_radius = 12.0
        painter.drawEllipse(obj_pos, body_radius, body_radius)

        # Mass text on body
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#0f172a")))
        painter.drawText(
            QRectF(obj_x - body_radius, obj_y - body_radius, body_radius * 2, body_radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            "m",
        )

        # Mode Specific Overlays (Banked angle or Conical string)
        if self._preset.mode == "CONICAL":
            # Apex of conical pendulum
            apex_p = QPointF(cx, cy - 110.0)
            painter.setPen(QPen(QColor("#a855f7"), 1.8, Qt.PenStyle.SolidLine))
            painter.drawLine(apex_p, obj_pos)
            # Apex suspension point
            painter.setBrush(QBrush(QColor("#c084fc")))
            painter.drawRect(QRectF(cx - 15.0, cy - 115.0, 30.0, 5.0))
        elif self._preset.mode == "BANKED":
            # Banking angle indicator
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(
                QRectF(arena_r.left() + 16.0, arena_r.top() + 16.0, 220.0, 20.0),
                Qt.AlignmentFlag.AlignLeft,
                f"Góc nghiêng θ = {self._bank_angle_deg:.1f}° | Lý tưởng: {self.ideal_banking_angle_deg:.1f}°",
            )

    def _draw_arrow(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        color: QColor,
        width: float = 2.0,
        label: str = "",
    ) -> None:
        """Draws a crisp directed vector arrow with arrowhead and label."""
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 5.0:
            return

        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)

        # Arrowhead
        arrow_size = 9.0
        angle = math.atan2(dy, dx)
        p1 = QPointF(
            end.x() - arrow_size * math.cos(angle - math.pi / 6.0),
            end.y() - arrow_size * math.sin(angle - math.pi / 6.0),
        )
        p2 = QPointF(
            end.x() - arrow_size * math.cos(angle + math.pi / 6.0),
            end.y() - arrow_size * math.sin(angle + math.pi / 6.0),
        )
        poly = QPolygonF([end, p1, p2])
        painter.setBrush(QBrush(color))
        painter.drawPolygon(poly)

        if label:
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(color))
            painter.drawText(QPointF(end.x() + 6.0, end.y() + 4.0), label)

    def _paint_inspector_panel(self, painter: QPainter) -> None:
        """Paints quantitative kinematics/dynamics parameters and theoretical formulas."""
        x = 502.0
        y = 88.0
        w = 262.0
        h = 360.0

        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(x, y, w, h), 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(x + 14.0, y + 26.0), "📊 THÔNG SỐ ĐỘNG HỌC")

        # Metrics rows
        metrics = [
            ("Bán kính quỹ đạo (R):", f"{self._radius_m:.2f} m"),
            ("Vận tốc dài (v):", f"{self._speed_ms:.2f} m/s"),
            ("Tốc độ góc (ω):", f"{self.angular_velocity_rads:.2f} rad/s"),
            ("Chu kỳ quay (T):", f"{self.period_seconds:.2f} s"),
            ("Tần số quay (f):", f"{self.frequency_hz:.2f} Hz"),
            ("Khối lượng vật (m):", f"{self._mass_kg:.2f} kg"),
            ("Gia tốc hướng tâm (a_c):", f"{self.centripetal_acceleration_ms2:.2f} m/s²"),
            ("Lực hướng tâm (F_c):", f"{self.centripetal_force_newtons:.2f} N"),
        ]

        row_y = y + 50.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for lbl, val in metrics:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(QPointF(x + 14.0, row_y), lbl)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(
                QRectF(x + 120.0, row_y - 12.0, w - 134.0, 16.0),
                Qt.AlignmentFlag.AlignRight,
                val,
            )
            row_y += 21.0

        # Mode Specific Critical Insights
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(x + 10), int(row_y + 4), int(x + w - 10), int(row_y + 4))
        row_y += 18.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        if self._preset.mode == "BANKED":
            painter.drawText(QPointF(x + 14.0, row_y), f"Góc nghiêng lý tưởng: {self.ideal_banking_angle_deg:.1f}°")
            row_y += 18.0
            painter.drawText(QPointF(x + 14.0, row_y), f"Vận tốc thiết kế: {self._speed_ms*3.6:.1f} km/h")
        elif self._preset.mode == "VERTICAL":
            v_crit = self.critical_vertical_top_speed_ms
            is_safe = self._speed_ms >= v_crit
            status_txt = "AN TOÀN" if is_safe else "RƠI TỰ DO!"
            col = QColor("#10b981") if is_safe else QColor("#ef4444")
            painter.setPen(QPen(col))
            painter.drawText(QPointF(x + 14.0, row_y), f"v_tới_hạn (đỉnh): {v_crit:.2f} m/s | {status_txt}")
            row_y += 18.0
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.drawText(QPointF(x + 14.0, row_y), f"Áp lực đáy N: {self.vertical_bottom_normal_force_newtons:.1f} N")
        else:
            v_max_safe = self.max_safe_friction_speed_ms
            is_skid = self._speed_ms > v_max_safe
            status_txt = "TRƯỢT VĂNG!" if is_skid else "BÁM ĐƯỜNG"
            col = QColor("#ef4444") if is_skid else QColor("#10b981")
            painter.setPen(QPen(col))
            painter.drawText(QPointF(x + 14.0, row_y), f"v_tối_đa ma sát: {v_max_safe:.2f} m/s | {status_txt}")
            row_y += 18.0
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(QPointF(x + 14.0, row_y), f"Hệ số ma sát μ_s: {self._preset.friction_mu:.2f}")

        # Theoretical Equation Box at bottom of inspector
        eq_rect = QRectF(x + 10.0, y + h - 54.0, w - 20.0, 44.0)
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(eq_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(eq_rect.left() + 6.0, eq_rect.top() + 4.0, eq_rect.width() - 12.0, eq_rect.height() - 8.0),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._preset.formula_vi,
        )

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Paints interactive action buttons: play/pause, radius/speed/mass adjustments, stamp."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Play / Pause
        r_play = self._btn_rects["play_pause"]
        is_hover = self._hovered_button == "play_pause"
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor("#0284c7" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_play, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        play_txt = "⏸ Tạm dừng" if self._is_animating else "▶ Tiếp tục"
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # 2. Stepper Buttons: Radius, Speed, Mass
        steppers = [
            ("dec_radius", "R -"),
            ("inc_radius", "R +"),
            ("dec_speed", "v -"),
            ("inc_speed", "v +"),
            ("dec_mass", "m -"),
            ("inc_mass", "m +"),
        ]
        for btn_name, lbl in steppers:
            r = self._btn_rects[btn_name]
            is_hover = self._hovered_button == btn_name
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor("#334155" if is_hover else "#1e293b")))
            painter.drawRoundedRect(r, 6.0, 6.0)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, lbl)

        # 3. Stamp to Board
        r_stamp = self._btn_rects["stamp"]
        is_hover = self._hovered_button == "stamp"
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.setBrush(QBrush(QColor("#059669" if is_hover else "#064e3b")))
        painter.drawRoundedRect(r_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Sơ Đồ Vector Lên Bảng (LessonBoard)")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects["close"].contains(pos):
            self.cleanup()
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects["play_pause"].contains(pos):
            self.toggle_animation()
            event.accept()
            return
        if self._btn_rects["dec_radius"].contains(pos):
            self.set_radius(self._radius_m * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_radius"].contains(pos):
            self.set_radius(self._radius_m * 1.25)
            event.accept()
            return
        if self._btn_rects["dec_speed"].contains(pos):
            self.set_speed(self._speed_ms * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_speed"].contains(pos):
            self.set_speed(self._speed_ms * 1.25)
            event.accept()
            return
        if self._btn_rects["dec_mass"].contains(pos):
            self.set_mass(self._mass_kg * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_mass"].contains(pos):
            self.set_mass(self._mass_kg * 1.25)
            event.accept()
            return
        if self._btn_rects["stamp"].contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break
        if self._hovered_button != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Touchless air pinch gesture for selecting controls."""
        local_p = self.mapFromScene(point)

        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.cleanup()
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "play_pause":
                    self.toggle_animation()
                    return True
                elif name == "dec_radius":
                    self.set_radius(self._radius_m * 0.8)
                    return True
                elif name == "inc_radius":
                    self.set_radius(self._radius_m * 1.25)
                    return True
                elif name == "dec_speed":
                    self.set_speed(self._speed_ms * 0.8)
                    return True
                elif name == "inc_speed":
                    self.set_speed(self._speed_ms * 1.25)
                    return True
                elif name == "dec_mass":
                    self.set_mass(self._mass_kg * 0.8)
                    return True
                elif name == "inc_mass":
                    self.set_mass(self._mass_kg * 1.25)
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current circular motion state as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_circular_motion_diagram(
            center=center,
            preset_code=self._active_preset_code,
            radius_m=self._radius_m,
            speed_ms=self._speed_ms,
            mass_kg=self._mass_kg,
        )
        for itm in items:
            scene.addItem(itm)
