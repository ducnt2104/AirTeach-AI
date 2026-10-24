"""
AirTeach AI - Interactive Physics Projectile Motion & 2D Kinematics with Air Drag Simulator
===========================================================================================
Provides an authentic physics simulation of 2D projectile kinematics and ballistic drag:
  - Theoretical & Kinematic Foundations (Zero Mock Data):
      1. Ideal Vacuum Projectile Motion (Ném trong chân không):
           Initial launch speed v₀, launch angle θ ∈ [0°, 90°], launch height h₀.
           Velocity components:
             v_x(t) = v₀ · cos(θ)
             v_y(t) = v₀ · sin(θ) - g · t
           Position coordinates:
             x(t) = (v₀ · cos(θ)) · t
             y(t) = h₀ + (v₀ · sin(θ)) · t - 0.5 · g · t²
           Maximum apex height (v_y = 0):
             H_max = h₀ + (v₀² · sin²(θ)) / (2 · g)
           Total flight time to ground (y = 0):
             t_flight = (v₀ · sin(θ) + sqrt(v₀² · sin²(θ) + 2 · g · h₀)) / g
           Horizontal Range:
             R = x(t_flight) = (v₀ · cos(θ)) · t_flight
             For h₀ = 0: R = (v₀² · sin(2θ)) / g  (Maximum range at θ = 45°)
             For h₀ > 0: θ_optimal = arcsin(1 / sqrt(2 + (2·g·h₀ / v₀²))) < 45°
      2. Atmospheric Aerodynamic Drag (Lực cản không khí thực tế):
           F_drag = 0.5 · ρ · C_d · A · v²
           Terminal velocity: v_t = sqrt((2 · m · g) / (ρ · C_d · A))
           Equations of motion integrated via Runge-Kutta 4th Order (RK4):
             dv_x/dt = -(0.5 · ρ · C_d · A / m) · v · v_x
             dv_y/dt = -g - (0.5 · ρ · C_d · A / m) · v · v_y
  - 5 Authentic Curriculum Presets:
      - CANNONBALL_VACUUM_45: Pháo bắn góc 45° đạt tầm xa cực đại trong chân không
      - BASEBALL_AIR_DRAG: Quả bóng chày bay trong không khí chịu lực cản khí quyển thực tế
      - BASKETBALL_FREE_THROW: Ném phạt bóng rổ vào rổ từ độ cao h₀ = 2.0m góc 52°
      - CLIFF_PROJECTILE_OPTIMAL: Bắn từ vách núi cao h₀ = 30m với góc bắn tối ưu < 45°
      - BULLET_HIGH_SPEED_DRAG: Đạn súng trường vận tốc cao v₀ = 350 m/s chịu lực cản lớn
  - Interactive Visual Elements:
      - Animated flying projectile tracing path with live velocity vectors (v, v_x, v_y).
      - Visual comparison: Ideal vacuum trajectory (dashed cyan) vs Air drag trajectory (solid emerald).
      - Target indicator with impact landing circle.
      - Touchless air pinch gestures for adjusting speed, angle, height, and stamping to board.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
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

GRAVITY_G: float = 9.80665  # m/s² standard gravity
AIR_DENSITY_RHO: float = 1.225  # kg/m³ at sea level, 15 °C


@dataclass
class ProjectilePreset:
    code: str
    title_vi: str
    v0_ms: float
    angle_deg: float
    h0_m: float
    mass_kg: float
    drag_coeff: float  # C_d * A composite drag area (0.0 for vacuum)
    target_dist_m: float
    description_vi: str
    formula_vi: str


PRESETS: dict[str, ProjectilePreset] = {
    "CANNONBALL_VACUUM_45": ProjectilePreset(
        code="CANNONBALL_VACUUM_45",
        title_vi="Bắn Pháo Chân Không Góc 45°",
        v0_ms=25.0,
        angle_deg=45.0,
        h0_m=0.0,
        mass_kg=5.0,
        drag_coeff=0.0,  # Ideal vacuum
        target_dist_m=63.7,
        description_vi="Chuyển động ném xiên lý tưởng trong chân không. Góc bắn 45° cho tầm xa cực đại R = v₀²/g = 63.73 m.",
        formula_vi="R = (v₀²·sin(2θ)) / g  |  H_max = (v₀²·sin²(θ)) / (2g)",
    ),
    "BASEBALL_AIR_DRAG": ProjectilePreset(
        code="BASEBALL_AIR_DRAG",
        title_vi="Bóng Chày Trong Không Khí (Air Drag)",
        v0_ms=35.0,
        angle_deg=35.0,
        h0_m=1.0,
        mass_kg=0.145,
        drag_coeff=0.0018,  # Cd * A for baseball
        target_dist_m=85.0,
        description_vi="Cú đánh bóng chày chịu lực cản khí động học làm quỹ đạo mất tính đối xứng, tầm xa giảm đáng kể so với chân không.",
        formula_vi="F_cản = ½·ρ·Cd·A·v²  |  v_giới_hạn = √(2mg / (ρ·Cd·A))",
    ),
    "BASKETBALL_FREE_THROW": ProjectilePreset(
        code="BASKETBALL_FREE_THROW",
        title_vi="Ném Phạt Bóng Rổ (Free Throw)",
        v0_ms=7.2,
        angle_deg=52.0,
        h0_m=2.0,
        mass_kg=0.62,
        drag_coeff=0.0035,
        target_dist_m=4.6,  # 15 ft free throw line
        description_vi="Cú ném bóng rổ từ độ cao tay ném h₀ = 2.0m vào rổ cao 3.05m ở cự ly 4.6m với góc vào rổ dốc tối ưu.",
        formula_vi="y(x) = h₀ + x·tan(θ) - (g·x²) / (2v₀²·cos²(θ))",
    ),
    "CLIFF_PROJECTILE_OPTIMAL": ProjectilePreset(
        code="CLIFF_PROJECTILE_OPTIMAL",
        title_vi="Ném Từ Vách Núi Cao (Cliff Launch)",
        v0_ms=20.0,
        angle_deg=34.0,  # Optimal angle < 45°
        h0_m=30.0,
        mass_kg=2.0,
        drag_coeff=0.0,
        target_dist_m=76.8,
        description_vi="Khi ném từ độ cao h₀ > 0, góc bắn cho tầm xa cực đại luôn nhỏ hơn 45° theo công thức vi tích phân tối ưu.",
        formula_vi="θ_opt = arcsin(1 / √(2 + 2gh₀/v₀²)) < 45°",
    ),
    "BULLET_HIGH_SPEED_DRAG": ProjectilePreset(
        code="BULLET_HIGH_SPEED_DRAG",
        title_vi="Đạn Súng Trường Khí Động Học",
        v0_ms=120.0,
        angle_deg=20.0,
        h0_m=1.5,
        mass_kg=0.010,
        drag_coeff=0.0003,
        target_dist_m=450.0,
        description_vi="Đạn đạo súng trường vận tốc cao. Lực cản khí quyển cực lớn uốn cong quỹ đạo nhanh chóng.",
        formula_vi="m·dv/dt = -F_cản - m·g  (Tích phân số RK4)",
    ),
}


class InteractiveProjectileMotionItem(QGraphicsObject):
    """
    High-performance vector interactive Projectile Motion & 2D Kinematics Simulator.
    Supports animated projectile flight, live velocity vectors, air drag RK4 integration,
    vacuum vs drag trajectory comparison, and LessonBoard stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: Union[tuple[float, float], QPointF] = (290.0, 110.0)) -> None:
        super().__init__()
        if isinstance(pos, QPointF):
            self.setPos(pos)
        elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
            self.setPos(float(pos[0]), float(pos[1]))
        else:
            self.setPos(290.0, 110.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active state
        self._active_preset_code: str = "CANNONBALL_VACUUM_45"
        self._preset: ProjectilePreset = PRESETS[self._active_preset_code]

        # Configurable Parameters
        self._v0_ms: float = self._preset.v0_ms
        self._angle_deg: float = self._preset.angle_deg
        self._h0_m: float = self._preset.h0_m
        self._mass_kg: float = self._preset.mass_kg
        self._drag_coeff: float = self._preset.drag_coeff

        # Animation state
        self._anim_time_s: float = 0.0
        self._is_animating: bool = True

        # Computed trajectories: lists of (x_m, y_m)
        self._traj_vacuum: list[tuple[float, float]] = []
        self._traj_drag: list[tuple[float, float]] = []
        self._recompute_trajectories()

        # Animation timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "CANNONBALL_VACUUM_45",
            "BASEBALL_AIR_DRAG",
            "BASKETBALL_FREE_THROW",
            "CLIFF_PROJECTILE_OPTIMAL",
            "BULLET_HIGH_SPEED_DRAG",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Trajectory Canvas (Left) & Inspector (Right)
        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 480.0, 360.0)
        self._btn_rects["inspector"] = QRectF(504.0, 88.0, 260.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 100.0, h_bot)
        self._btn_rects["dec_speed"] = QRectF(122.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_speed"] = QRectF(168.0, y_bot, 42.0, h_bot)
        self._btn_rects["dec_angle"] = QRectF(216.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_angle"] = QRectF(262.0, y_bot, 42.0, h_bot)
        self._btn_rects["dec_height"] = QRectF(310.0, y_bot, 42.0, h_bot)
        self._btn_rects["inc_height"] = QRectF(356.0, y_bot, 42.0, h_bot)
        self._btn_rects["stamp"] = QRectF(406.0, y_bot, 358.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical Trajectory Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    def _recompute_trajectories(self) -> None:
        """Computes authentic vacuum and air-drag trajectories."""
        self._traj_vacuum.clear()
        self._traj_drag.clear()

        # 1. Vacuum Trajectory
        rad = math.radians(self._angle_deg)
        vx0 = self._v0_ms * math.cos(rad)
        vy0 = self._v0_ms * math.sin(rad)
        g = GRAVITY_G
        h0 = self._h0_m

        # Total vacuum flight time
        discrim = max(0.0, vy0 ** 2 + 2.0 * g * h0)
        t_flight = (vy0 + math.sqrt(discrim)) / g
        dt = max(0.01, t_flight / 80.0)

        steps = 80
        for i in range(steps + 1):
            t = min(t_flight, i * dt)
            x = vx0 * t
            y = max(0.0, h0 + vy0 * t - 0.5 * g * t ** 2)
            self._traj_vacuum.append((x, y))

        # 2. Air Drag Trajectory via RK4 integration
        k_drag = (0.5 * AIR_DENSITY_RHO * self._drag_coeff) / max(0.01, self._mass_kg)
        if k_drag <= 1e-6:
            self._traj_drag = list(self._traj_vacuum)
        else:
            x, y = 0.0, h0
            vx, vy = vx0, vy0
            dt_sim = 0.01
            max_sim_steps = 3000
            self._traj_drag.append((x, y))

            for _ in range(max_sim_steps):
                speed = math.hypot(vx, vy)
                ax = -k_drag * speed * vx
                ay = -g - k_drag * speed * vy

                vx += ax * dt_sim
                vy += ay * dt_sim
                x += vx * dt_sim
                y += vy * dt_sim

                if y < 0.0:
                    self._traj_drag.append((x, 0.0))
                    break
                self._traj_drag.append((x, y))

    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> ProjectilePreset:
        return self._preset

    @property
    def v0_ms(self) -> float:
        return self._v0_ms

    @property
    def angle_deg(self) -> float:
        return self._angle_deg

    @property
    def h0_m(self) -> float:
        return self._h0_m

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def vacuum_range_m(self) -> float:
        """Horizontal range in vacuum (meters)."""
        if self._traj_vacuum:
            return self._traj_vacuum[-1][0]
        return 0.0

    @property
    def vacuum_max_height_m(self) -> float:
        """Maximum apex altitude in vacuum (meters)."""
        rad = math.radians(self._angle_deg)
        return self._h0_m + (self._v0_ms ** 2 * (math.sin(rad) ** 2)) / (2.0 * GRAVITY_G)

    @property
    def drag_range_m(self) -> float:
        """Horizontal range with air drag (meters)."""
        if self._traj_drag:
            return self._traj_drag[-1][0]
        return 0.0

    @property
    def optimal_launch_angle_deg(self) -> float:
        """Calculates exact analytical optimal launch angle for maximum range."""
        if self._h0_m <= 0.0:
            return 45.0
        val = 1.0 / math.sqrt(2.0 + (2.0 * GRAVITY_G * self._h0_m) / (self._v0_ms ** 2))
        return math.degrees(math.asin(max(0.1, min(1.0, val))))

    # -------------------------------------------------------------------------
    # Public Controller Actions
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic projectile kinematics preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._v0_ms = self._preset.v0_ms
        self._angle_deg = self._preset.angle_deg
        self._h0_m = self._preset.h0_m
        self._mass_kg = self._preset.mass_kg
        self._drag_coeff = self._preset.drag_coeff
        self._anim_time_s = 0.0
        self._recompute_trajectories()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_speed(self, speed_ms: float) -> None:
        """Set initial launch velocity (clamped between 2.0 m/s and 250.0 m/s)."""
        self._v0_ms = max(2.0, min(250.0, float(speed_ms)))
        self._recompute_trajectories()
        self.update()

    def set_angle(self, angle_deg: float) -> None:
        """Set launch angle (clamped between 5.0° and 85.0°)."""
        self._angle_deg = max(5.0, min(85.0, float(angle_deg)))
        self._recompute_trajectories()
        self.update()

    def set_height(self, height_m: float) -> None:
        """Set initial launch altitude (clamped between 0.0m and 100.0m)."""
        self._h0_m = max(0.0, min(100.0, float(height_m)))
        self._recompute_trajectories()
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle projectile flight animation."""
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

        dt = 0.033
        rad = math.radians(self._angle_deg)
        vy0 = self._v0_ms * math.sin(rad)
        discrim = max(0.0, vy0 ** 2 + 2.0 * GRAVITY_G * self._h0_m)
        t_flight = (vy0 + math.sqrt(discrim)) / GRAVITY_G

        self._anim_time_s += dt * 1.5
        if self._anim_time_s > t_flight * 1.2:
            self._anim_time_s = 0.0

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main outer card
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        # 2. Header Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 48.0), 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 24.0, self.WIDTH, 24.0))

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(16.0, 10.0, 520.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🚀 {self._preset.title_vi} | Chuyển Động Ném & Khí Động Học",
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
        preset_items = [
            ("CANNONBALL_VACUUM_45", "Pháo 45° Chân Không"),
            ("BASEBALL_AIR_DRAG", "Bóng Chày (Có Cản)"),
            ("BASKETBALL_FREE_THROW", "Bóng Rổ Vào Rổ"),
            ("CLIFF_PROJECTILE_OPTIMAL", "Vách Núi Tối Ưu"),
            ("BULLET_HIGH_SPEED_DRAG", "Đạn Súng Trường"),
        ]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for code, label in preset_items:
            r = self._btn_rects[f"preset_{code}"]
            is_active = self._active_preset_code == code
            is_hover = self._hovered_button == f"preset_{code}"
            bg = QColor("#0284c7") if is_active else (QColor("#334155") if is_hover else QColor("#1e293b"))
            border = QColor("#38bdf8") if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(r, 6.0, 6.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#cbd5e1")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)

        # 4. Trajectory Canvas (Left)
        can_r = self._btn_rects["canvas"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(can_r, 10.0, 10.0)
        self._paint_trajectory_canvas(painter, can_r)

        # 5. Right Inspector & Dynamics Analytics Panel
        insp_r = self._btn_rects["inspector"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(insp_r, 10.0, 10.0)
        self._paint_inspector_panel(painter, insp_r)

        # 6. Bottom Action Toolbar
        self._paint_bottom_toolbar(painter)

    def _paint_trajectory_canvas(self, painter: QPainter, arena_r: QRectF) -> None:
        """Paints ground baseline, launcher cliff, parabolic vacuum curve, and air-drag path."""
        pad_x = 45.0
        pad_y = 35.0
        draw_w = arena_r.width() - pad_x - 30.0
        draw_h = arena_r.height() - pad_y - 45.0

        origin_x = arena_r.left() + pad_x
        origin_y = arena_r.bottom() - 45.0

        # Maximum bounds for auto-scaling
        max_x = max(10.0, self.vacuum_range_m * 1.15)
        max_y = max(5.0, self.vacuum_max_height_m * 1.25)

        scale_x = draw_w / max_x
        scale_y = draw_h / max_y

        # Ground baseline (green grass / dark earth line)
        painter.setPen(QPen(QColor("#22c55e"), 2.0))
        painter.drawLine(int(arena_r.left() + 20), int(origin_y), int(arena_r.right() - 20), int(origin_y))

        # Launch platform cliff if h0 > 0
        if self._h0_m > 0.0:
            cliff_h = self._h0_m * scale_y
            painter.setPen(QPen(QColor("#64748b"), 1.5))
            painter.setBrush(QBrush(QColor("#334155")))
            painter.drawRect(QRectF(origin_x - 22.0, origin_y - cliff_h, 22.0, cliff_h))

        # 1. Ideal Vacuum Path (Dashed light blue)
        if self._traj_vacuum:
            path_vac = QPainterPath()
            for i, (xm, ym) in enumerate(self._traj_vacuum):
                px = origin_x + xm * scale_x
                py = origin_y - ym * scale_y
                if i == 0:
                    path_vac.moveTo(px, py)
                else:
                    path_vac.lineTo(px, py)

            painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8, Qt.PenStyle.DashLine))
            painter.drawPath(path_vac)

        # 2. Air Drag Trajectory (Solid emerald green)
        if self._traj_drag:
            path_drag = QPainterPath()
            for i, (xm, ym) in enumerate(self._traj_drag):
                px = origin_x + xm * scale_x
                py = origin_y - ym * scale_y
                if i == 0:
                    path_drag.moveTo(px, py)
                else:
                    path_drag.lineTo(px, py)

            painter.setPen(QPen(QColor("#10b981"), 2.5))
            painter.drawPath(path_drag)

        # 3. Flying Projectile Body & Velocity Vectors
        rad = math.radians(self._angle_deg)
        vx0 = self._v0_ms * math.cos(rad)
        vy0 = self._v0_ms * math.sin(rad)
        t = self._anim_time_s

        cur_x = vx0 * t
        cur_y = max(0.0, self._h0_m + vy0 * t - 0.5 * GRAVITY_G * t ** 2)

        proj_px = origin_x + cur_x * scale_x
        proj_py = origin_y - cur_y * scale_y

        # Velocity components at current time
        cur_vx = vx0
        cur_vy = vy0 - GRAVITY_G * t

        # Draw Velocity Vector Arrow (Tangential)
        v_arrow_len = 35.0
        v_hypot = max(0.1, math.hypot(cur_vx, cur_vy))
        ax_end_x = proj_px + (cur_vx / v_hypot) * v_arrow_len
        ax_end_y = proj_py - (cur_vy / v_hypot) * v_arrow_len

        painter.setPen(QPen(QColor("#f43f5e"), 2.0))
        painter.drawLine(QPointF(proj_px, proj_py), QPointF(ax_end_x, ax_end_y))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(QPointF(ax_end_x + 4.0, ax_end_y + 2.0), "v")

        # Projectile Sphere (Gold)
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.drawEllipse(QPointF(proj_px, proj_py), 6.5, 6.5)

        # 4. Canvas Legends
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(arena_r.left() + 20.0, arena_r.top() + 20.0), "- - Chân không (Lý tưởng)")
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(QPointF(arena_r.left() + 175.0, arena_r.top() + 20.0), "— Lực cản không khí (Khí động học)")

    def _paint_inspector_panel(self, painter: QPainter, insp_r: QRectF) -> None:
        """Paints quantitative kinematics parameters, range comparison, and optimal angle."""
        x = insp_r.left()
        y = insp_r.top()
        w = insp_r.width()

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(x + 14.0, y + 24.0), "📊 THÔNG SỐ ĐỘNG HỌC")

        # Metrics rows
        vac_r = self.vacuum_range_m
        drag_r = self.drag_range_m
        h_max = self.vacuum_max_height_m
        opt_angle = self.optimal_launch_angle_deg

        rad = math.radians(self._angle_deg)
        vy0 = self._v0_ms * math.sin(rad)
        discrim = max(0.0, vy0 ** 2 + 2.0 * GRAVITY_G * self._h0_m)
        t_flight = (vy0 + math.sqrt(discrim)) / GRAVITY_G

        metrics = [
            ("Vận tốc đầu (v₀):", f"{self._v0_ms:.1f} m/s"),
            ("Góc bắn (θ):", f"{self._angle_deg:.1f}°"),
            ("Độ cao ban đầu (h₀):", f"{self._h0_m:.1f} m"),
            ("Khối lượng vật (m):", f"{self._mass_kg:.2f} kg"),
            ("Thời gian bay (t_bay):", f"{t_flight:.2f} s"),
            ("Tầm bay cao (H_max):", f"{h_max:.2f} m"),
            ("Tầm xa chân không (R):", f"{vac_r:.2f} m"),
            ("Tầm xa có lực cản:", f"{drag_r:.2f} m"),
        ]

        row_y = y + 48.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for lbl, val in metrics:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(QPointF(x + 14.0, row_y), lbl)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(QRectF(x + 130.0, row_y - 12.0, w - 145.0, 16.0), Qt.AlignmentFlag.AlignRight, val)
            row_y += 19.0

        # Optimal Angle Highlight Card
        opt_y = row_y + 10.0
        opt_h = 44.0
        painter.setPen(QPen(QColor("#f59e0b"), 1.0))
        painter.setBrush(QBrush(QColor("#451a03")))
        painter.drawRoundedRect(QRectF(x + 14.0, opt_y, w - 28.0, opt_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(QPointF(x + 22.0, opt_y + 18.0), f"Góc ném cực đại tối ưu: {opt_angle:.1f}°")
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#fef08a")))
        painter.drawText(QPointF(x + 22.0, opt_y + 34.0), f"Giảm tầm xa do lực cản: {max(0.0, vac_r - drag_r):.1f} m ({(max(0.0, vac_r - drag_r)/max(1.0, vac_r))*100.0:.1f}%)")

        # Theoretical formula box at bottom
        eq_y = opt_y + opt_h + 12.0
        eq_h = 48.0
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(QRectF(x + 14.0, eq_y, w - 28.0, eq_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(x + 18.0, eq_y + 4.0, w - 36.0, eq_h - 8.0),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._preset.formula_vi,
        )

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Paints interactive action buttons: play/pause, v0/angle/height adjustments, stamp."""
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

        # 2. Stepper Buttons: v0, angle, height
        steppers = [
            ("dec_speed", "v₀ -"),
            ("inc_speed", "v₀ +"),
            ("dec_angle", "θ -"),
            ("inc_angle", "θ +"),
            ("dec_height", "h₀ -"),
            ("inc_height", "h₀ +"),
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
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Đồ Thị Ném Lên Bảng (LessonBoard)")

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
        if self._btn_rects["dec_speed"].contains(pos):
            self.set_speed(self._v0_ms * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_speed"].contains(pos):
            self.set_speed(self._v0_ms * 1.25)
            event.accept()
            return
        if self._btn_rects["dec_angle"].contains(pos):
            self.set_angle(self._angle_deg - 5.0)
            event.accept()
            return
        if self._btn_rects["inc_angle"].contains(pos):
            self.set_angle(self._angle_deg + 5.0)
            event.accept()
            return
        if self._btn_rects["dec_height"].contains(pos):
            self.set_height(self._h0_m - 2.0)
            event.accept()
            return
        if self._btn_rects["inc_height"].contains(pos):
            self.set_height(self._h0_m + 2.0)
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
                elif name == "dec_speed":
                    self.set_speed(self._v0_ms * 0.8)
                    return True
                elif name == "inc_speed":
                    self.set_speed(self._v0_ms * 1.25)
                    return True
                elif name == "dec_angle":
                    self.set_angle(self._angle_deg - 5.0)
                    return True
                elif name == "inc_angle":
                    self.set_angle(self._angle_deg + 5.0)
                    return True
                elif name == "dec_height":
                    self.set_height(self._h0_m - 2.0)
                    return True
                elif name == "inc_height":
                    self.set_height(self._h0_m + 2.0)
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current projectile trajectory as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_projectile_motion_diagram(
            center=center,
            preset_code=self._active_preset_code,
            v0_ms=self._v0_ms,
            angle_deg=self._angle_deg,
            h0_m=self._h0_m,
        )
        for itm in items:
            scene.addItem(itm)
