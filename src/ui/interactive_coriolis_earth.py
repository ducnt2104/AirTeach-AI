"""
AirTeach AI - Interactive Atmospheric Physics: Coriolis Effect, Geostrophic Flow & Cyclonic Vorticity
======================================================================================================
Authentic geophysical fluid dynamics and atmospheric physics on rotating planets:
- Reference Frames & Coriolis Acceleration:
    - Inertial Frame: Planetary rotation vector Ω = Ω k̂
    - Rotating Frame (Observer on Earth):
        a_C = -2 (Ω × v)
    - At Latitude φ ∈ [-90°, +90°]:
        Vertical component: Ω_z = Ω sin(φ)
        Coriolis Parameter: f = 2 Ω sin(φ)
        - Northern Hemisphere (φ > 0 => f > 0): Deflection to the RIGHT of motion
        - Southern Hemisphere (φ < 0 => f < 0): Deflection to the LEFT of motion
        - Equator (φ = 0 => f = 0): Zero Coriolis deflection (no tropical cyclones form within 5° of equator)
- Geostrophic Wind Balance (Cân bằng Địa chuyển):
    - Pressure Gradient Force (PGF): F_pgf = -(1/ρ) ∇P
    - Coriolis Force: F_C = -f (k̂ × v)
    - Balance condition: v_g = (1 / (ρ f)) (k̂ × ∇P)
    - Wind blows parallel to isobars (đường đẳng áp), with lower pressure to the left in Northern Hemisphere (Buys Ballot's Law)
- Cyclonic & Anticyclonic Vorticity:
    - Low-Pressure System (Tâm Áp thấp / Bão nhiệt đới):
        - Inward PGF balanced by Coriolis deflection:
        - Northern Hemisphere: Counterclockwise (Ngược chiều kim đồng hồ) cyclonic spiral
        - Southern Hemisphere: Clockwise (Cùng chiều kim đồng hồ) cyclonic spiral
    - High-Pressure System (Tâm Áp cao / Anticyclone):
        - Outward PGF balanced by Coriolis deflection:
        - Northern Hemisphere: Clockwise spiral
        - Southern Hemisphere: Counterclockwise spiral
- Ballistic Deflection (Độ lệch đạn đạo):
    - Fired projectile trajectory integrated in rotating frame showing eastward/westward Coriolis drift
- Foucault Pendulum Precession:
    - Precession rate: ω_F = Ω sin(φ) (15°/hour · sin φ)
    - Rotation Period: T = 24 hours / |sin φ|
- 5 Curriculum Presets:
    1. NORTH_CYCLONE: Northern hemisphere tropical cyclone (Bão Bắc Bán Cầu, φ = +21° Hanoi)
    2. SOUTH_CYCLONE: Southern hemisphere cyclone (Bão Nam Bán Cầu, φ = -21°)
    3. GEOSTROPHIC_WIND: Mid-latitude geostrophic wind & isobar balance (φ = +45°)
    4. BALLISTIC_CANNON: Projectile artillery deflection comparison (Inertial vs Rotating frame)
    5. FOUCAULT_PENDULUM: Foucault pendulum precession with latitude knob
- Touchless webcam air gestures (air-tap on buttons, latitude slider, rotation speed slider),
  bimanual manipulation, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
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

# Base Earth rotation rate in scaled simulation units (approx 7.292e-5 rad/s)
OMEGA_EARTH_BASE = 1.0


@dataclass(frozen=True)
class CoriolisPreset:
    code: str
    name_vi: str
    scenario_type: str  # "cyclone", "anticyclone", "geostrophic", "ballistic", "foucault"
    latitude_deg: float
    rotation_speed: float  # multiplier (0.0 to 3.0)
    pressure_gradient: float  # 0.5 to 5.0
    description: str


PRESETS: dict[str, CoriolisPreset] = {
    "NORTH_CYCLONE": CoriolisPreset(
        code="NORTH_CYCLONE",
        name_vi="1. Bão Nhiệt Đới Bắc Bán Cầu",
        scenario_type="cyclone",
        latitude_deg=21.0,  # Hanoi latitude
        rotation_speed=1.0,
        pressure_gradient=3.5,
        description="Lực Gradien Áp suất hút gió vào tâm, lực Coriolis làm lệch phải -> Xoáy bão ngược chiều kim đồng hồ.",
    ),
    "SOUTH_CYCLONE": CoriolisPreset(
        code="SOUTH_CYCLONE",
        name_vi="2. Bão Nhiệt Đới Nam Bán Cầu",
        scenario_type="cyclone",
        latitude_deg=-21.0,
        rotation_speed=1.0,
        pressure_gradient=3.5,
        description="Ở Nam Bán Cầu (f < 0), lực Coriolis làm lệch trái -> Xoáy bão cùng chiều kim đồng hồ.",
    ),
    "GEOSTROPHIC_WIND": CoriolisPreset(
        code="GEOSTROPHIC_WIND",
        name_vi="3. Gió Địa Chuyển & Đường Đẳng Áp",
        scenario_type="geostrophic",
        latitude_deg=45.0,
        rotation_speed=1.0,
        pressure_gradient=2.8,
        description="Cân bằng địa chuyển F_PGF = F_Coriolis: Gió thổi song song với các đường đẳng áp (Định luật Buys-Ballot).",
    ),
    "BALLISTIC_CANNON": CoriolisPreset(
        code="BALLISTIC_CANNON",
        name_vi="4. Độ Lệch Đạn Đạo Pháo Binh",
        scenario_type="ballistic",
        latitude_deg=35.0,
        rotation_speed=1.2,
        pressure_gradient=1.0,
        description="Quỹ đạo đạn thẳng trong hệ quy chiếu quán tính nhưng bị bẻ cong sang phải đối với quan sát viên trên Trái Đất.",
    ),
    "FOUCAULT_PENDULUM": CoriolisPreset(
        code="FOUCAULT_PENDULUM",
        name_vi="5. Con Lắc Foucault (Trái Đất tự quay)",
        scenario_type="foucault",
        latitude_deg=48.8,  # Paris
        rotation_speed=1.0,
        pressure_gradient=1.0,
        description="Mặt phẳng dao động quay với chu kỳ T = 24h / |sin φ|. Tại vĩ độ 48.8°, chu kỳ quay là ~32 giờ.",
    ),
}


class InteractiveCoriolisEarthItem(QGraphicsObject):
    """
    AirTeach AI interactive Coriolis Effect, Geostrophic Wind & Cyclone Simulation.
    Visualizes streamline particles, Coriolis acceleration vectors, and vorticity metrics.
    """

    closed = Signal()

    WIDTH = 760.0
    HEIGHT = 500.0

    def __init__(
        self,
        pos: tuple[float, float] | QPointF = (120.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(float(pos[0]), float(pos[1]))

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Physics State
        self._preset_code: str = "NORTH_CYCLONE"
        self._preset: CoriolisPreset = PRESETS[self._preset_code]
        self._latitude_deg: float = self._preset.latitude_deg
        self._rotation_speed: float = self._preset.rotation_speed
        self._pressure_gradient: float = self._preset.pressure_gradient

        # View options
        self._show_isobars: bool = True
        self._show_force_vectors: bool = True
        self._show_inertial_path: bool = False

        # Particle Streamline Simulation
        self._sim_time: float = 0.0
        self._is_playing: bool = True
        self._num_particles: int = 36
        self._particle_pos: np.ndarray = np.zeros((self._num_particles, 2), dtype=float)
        self._particle_age: np.ndarray = np.zeros(self._num_particles, dtype=float)
        self._reset_particles()

        # Ballistic Projectile State
        self._ballistic_t: float = 0.0
        self._ballistic_traj_rot: list[tuple[float, float]] = []
        self._ballistic_traj_inertial: list[tuple[float, float]] = []

        # Foucault Pendulum Precession Angle (radians)
        self._foucault_angle: float = 0.0

        # Animation timer
        self._timer: QTimer = QTimer()
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # UI Hitboxes
        self._btn_rects: dict[str, QRectF] = {}
        self._slider_rect_lat: QRectF = QRectF()
        self._slider_rect_omega: QRectF = QRectF()
        self._drag_target: Optional[str] = None

        self._setup_layout()

    # ── Geometry & Layout ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _setup_layout(self) -> None:
        """Calculate hitboxes for glassmorphic controls and buttons."""
        self._btn_rects.clear()

        # Row 1: 5 Curriculum Presets (y = 40.0)
        p_keys = [
            "NORTH_CYCLONE",
            "SOUTH_CYCLONE",
            "GEOSTROPHIC_WIND",
            "BALLISTIC_CANNON",
            "FOUCAULT_PENDULUM",
        ]
        p_labels = [
            "1. Bão Bắc Bán Cầu",
            "2. Bão Nam Bán Cầu",
            "3. Gió Địa Chuyển",
            "4. Lệch Đạn Đạo",
            "5. Con Lắc Foucault",
        ]
        p_widths = [135.0, 135.0, 140.0, 140.0, 150.0]
        cur_x = 16.0
        for code, label, pw in zip(p_keys, p_labels, p_widths):
            self._btn_rects[f"preset_{code}"] = QRectF(cur_x, 40.0, pw, 24.0)
            cur_x += pw + 6.0

        # Row 2: Control Buttons (y = 70.0)
        r2_x = 16.0
        # Lat -15 / +15
        self._btn_rects["lat_minus"] = QRectF(r2_x, 70.0, 60.0, 24.0)
        r2_x += 64.0
        self._btn_rects["lat_plus"] = QRectF(r2_x, 70.0, 60.0, 24.0)
        r2_x += 66.0

        # Equator shortcut (φ = 0°)
        self._btn_rects["set_equator"] = QRectF(r2_x, 70.0, 80.0, 24.0)
        r2_x += 86.0

        # Toggles: Isobars, Force Vectors
        self._btn_rects["toggle_isobars"] = QRectF(r2_x, 70.0, 95.0, 24.0)
        r2_x += 101.0
        self._btn_rects["toggle_forces"] = QRectF(r2_x, 70.0, 90.0, 24.0)
        r2_x += 96.0

        # Play / Pause, Reset
        self._btn_rects["btn_play_pause"] = QRectF(r2_x, 70.0, 75.0, 24.0)
        r2_x += 81.0
        self._btn_rects["btn_reset"] = QRectF(r2_x, 70.0, 70.0, 24.0)
        r2_x += 76.0

        # Close button
        self._btn_rects["btn_close"] = QRectF(self.WIDTH - 64.0, 8.0, 48.0, 22.0)

        # Bottom Slider Tracks (y = 448.0)
        self._slider_rect_lat = QRectF(100.0, 448.0, 250.0, 14.0)
        self._slider_rect_omega = QRectF(450.0, 448.0, 250.0, 14.0)

    # ── Atmospheric & Geophysical Physics Engine ───────────────────────────────

    def get_coriolis_parameter(self) -> float:
        """
        Compute Coriolis parameter f = 2 Ω sin(φ).
        Returns scaled parameter (negative in Southern hemisphere, 0 at equator).
        """
        phi_rad = math.radians(self._latitude_deg)
        omega = OMEGA_EARTH_BASE * self._rotation_speed
        return 2.0 * omega * math.sin(phi_rad)

    def compute_velocity_field(self, x: float, y: float) -> tuple[float, float]:
        """
        Compute air parcel velocity vector (u, v) under PGF and Coriolis deflection.
        Coordinates are normalized relative to center of visualizer (-3.0 to +3.0).
        """
        scenario = self._preset.scenario_type
        f = self.get_coriolis_parameter()
        pgf = self._pressure_gradient

        if scenario == "cyclone":
            # Radial distance to low pressure center (0, 0)
            r = math.hypot(x, y)
            if r < 0.15:
                return 0.0, 0.0
            # Radial pressure gradient force directed inward: F_r = -pgf / r
            # Tangential Coriolis deflection:
            # Northern hemisphere (f > 0): deflection to the right -> counterclockwise
            # Southern hemisphere (f < 0): deflection to the left -> clockwise
            inward_v = -0.35 * (pgf / max(0.5, r))
            tangential_v = 0.85 * f * (pgf / max(0.8, r**0.7))

            # Direction unit vectors
            ux = x / r
            uy = y / r
            tx = -uy  # Counterclockwise tangential
            ty = ux

            vx = (inward_v * ux) + (tangential_v * tx)
            vy = (inward_v * uy) + (tangential_v * ty)
            return vx, vy

        elif scenario == "geostrophic":
            # Isobars running horizontally (parallel to x-axis)
            # Pressure decreases towards positive y (North): ∂P/∂y > 0
            # PGF points towards -y: F_y = -pgf
            # Geostrophic balance: u_g = -(1/f) ∂P/∂y
            # If f > 0 (Northern hemisphere), wind blows eastward (+x) with low pressure to the left!
            # If f == 0 (Equator), air accelerates straight down gradient (-y).
            if abs(f) < 0.1:
                return 0.0, -0.6 * pgf
            u_g = 0.5 * (pgf / f)
            v_ageostrophic = -0.1 * pgf * (1.0 / (1.0 + abs(f)))
            return u_g, v_ageostrophic

        elif scenario == "ballistic":
            # Projectile fired northward (+y) with speed v0
            # Coriolis acceleration: a_x = f * v_y
            return 0.0, 1.2

        elif scenario == "foucault":
            # Harmonic pendulum oscillation along dominant axis with precession
            return 0.0, 0.0

        return 0.0, 0.0

    def _reset_particles(self) -> None:
        """Scatter particles randomly across the simulation domain."""
        np.random.seed(42)
        angles = np.random.uniform(0.0, 2.0 * math.pi, self._num_particles)
        radii = np.random.uniform(0.5, 2.6, self._num_particles)
        self._particle_pos[:, 0] = radii * np.cos(angles)
        self._particle_pos[:, 1] = radii * np.sin(angles)
        self._particle_age = np.random.uniform(0.0, 1.0, self._num_particles)

        # Reset ballistic
        self._ballistic_t = 0.0
        self._ballistic_traj_rot = [(0.0, -2.2)]
        self._ballistic_traj_inertial = [(0.0, -2.2)]

    # ── Simulation Tick ────────────────────────────────────────────────────────

    def _on_sim_tick(self) -> None:
        """Advance time, particle advection, and pendulum precession."""
        if not self._is_playing:
            return

        dt = 0.033
        self._sim_time += dt

        scenario = self._preset.scenario_type
        f = self.get_coriolis_parameter()

        if scenario in ("cyclone", "geostrophic"):
            # Advect particles
            for i in range(self._num_particles):
                px = self._particle_pos[i, 0]
                py = self._particle_pos[i, 1]
                vx, vy = self.compute_velocity_field(px, py)
                self._particle_pos[i, 0] += vx * dt * 0.7
                self._particle_pos[i, 1] += vy * dt * 0.7
                self._particle_age[i] += dt * 0.4

                # Recycle particle if too close to center or out of bounds or expired
                r = math.hypot(self._particle_pos[i, 0], self._particle_pos[i, 1])
                if r < 0.25 or r > 2.8 or self._particle_age[i] > 3.0:
                    ang = np.random.uniform(0.0, 2.0 * math.pi)
                    rad = np.random.uniform(1.8, 2.7)
                    self._particle_pos[i, 0] = rad * math.cos(ang)
                    self._particle_pos[i, 1] = rad * math.sin(ang)
                    self._particle_age[i] = 0.0

        elif scenario == "ballistic":
            # Integrate artillery flight
            if self._ballistic_t < 4.0:
                self._ballistic_t += dt
                t = self._ballistic_t
                v0 = 1.4
                # Inertial frame: straight line path y(t) = y0 + v0 * t
                y_iner = -2.2 + (v0 * t)
                x_iner = 0.0
                self._ballistic_traj_inertial.append((x_iner, y_iner))

                # Rotating frame: x''(t) = f * y'(t) => x(t) = 1/2 * f * v0 * t^2
                # In Northern hemisphere (f > 0), x drifts rightward (+x)!
                x_rot = 0.5 * f * v0 * (t**2) * 0.35
                y_rot = -2.2 + (v0 * t) - (0.04 * (t**2))
                self._ballistic_traj_rot.append((x_rot, y_rot))
            else:
                # Loop ballistic animation
                self._ballistic_t = 0.0
                self._ballistic_traj_rot = [(0.0, -2.2)]
                self._ballistic_traj_inertial = [(0.0, -2.2)]

        elif scenario == "foucault":
            # Precession rate dθ/dt = -Ω sin(φ)
            phi_rad = math.radians(self._latitude_deg)
            omega_foucault = 0.4 * self._rotation_speed * math.sin(phi_rad)
            self._foucault_angle -= omega_foucault * dt

        self.update()

    # ── State Mutators ─────────────────────────────────────────────────────────

    def load_preset(self, code: str) -> None:
        """Load curriculum preset."""
        if code not in PRESETS:
            return
        self._preset_code = code
        self._preset = PRESETS[code]
        self._latitude_deg = self._preset.latitude_deg
        self._rotation_speed = self._preset.rotation_speed
        self._pressure_gradient = self._preset.pressure_gradient
        self._reset_particles()
        self.update()

    def set_latitude(self, lat_deg: float) -> None:
        """Set latitude with clamped bounds [-90.0°, +90.0°]."""
        self._latitude_deg = max(-90.0, min(90.0, lat_deg))
        self.update()

    def set_rotation_speed(self, speed: float) -> None:
        """Set Earth rotation speed multiplier [0.0, 3.0]."""
        self._rotation_speed = max(0.0, min(3.0, speed))
        self.update()

    def toggle_play_pause(self) -> None:
        self._is_playing = not self._is_playing
        self.update()

    def toggle_isobars(self) -> None:
        self._show_isobars = not self._show_isobars
        self.update()

    def toggle_forces(self) -> None:
        self._show_force_vectors = not self._show_force_vectors
        self.update()

    # ── Touch & Gesture Interactions ───────────────────────────────────────────

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process touchless air tap gesture from webcam tracking."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # 1. Preset Buttons
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "lat_minus":
                    self.set_latitude(self._latitude_deg - 15.0)
                elif key == "lat_plus":
                    self.set_latitude(self._latitude_deg + 15.0)
                elif key == "set_equator":
                    self.set_latitude(0.0)
                elif key == "toggle_isobars":
                    self.toggle_isobars()
                elif key == "toggle_forces":
                    self.toggle_forces()
                elif key == "btn_play_pause":
                    self.toggle_play_pause()
                elif key == "btn_reset":
                    self._reset_particles()
                elif key == "btn_close":
                    self.closed.emit()
                return True

        # 2. Latitude Slider Tap
        if self._slider_rect_lat.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            frac = (pos.x() - self._slider_rect_lat.left()) / self._slider_rect_lat.width()
            frac = max(0.0, min(1.0, frac))
            # Map [0, 1] to [-90, +90]
            lat_val = -90.0 + (frac * 180.0)
            self.set_latitude(round(lat_val, 1))
            return True

        # 3. Rotation Speed Slider Tap
        if self._slider_rect_omega.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            frac = (pos.x() - self._slider_rect_omega.left()) / self._slider_rect_omega.width()
            frac = max(0.0, min(1.0, frac))
            self.set_rotation_speed(round(frac * 3.0, 2))
            return True

        return True

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check button clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                event.accept()
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "lat_minus":
                    self.set_latitude(self._latitude_deg - 15.0)
                elif key == "lat_plus":
                    self.set_latitude(self._latitude_deg + 15.0)
                elif key == "set_equator":
                    self.set_latitude(0.0)
                elif key == "toggle_isobars":
                    self.toggle_isobars()
                elif key == "toggle_forces":
                    self.toggle_forces()
                elif key == "btn_play_pause":
                    self.toggle_play_pause()
                elif key == "btn_reset":
                    self._reset_particles()
                elif key == "btn_close":
                    self.closed.emit()
                return

        # Check slider drags
        if self._slider_rect_lat.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            event.accept()
            self._drag_target = "latitude"
            frac = (pos.x() - self._slider_rect_lat.left()) / self._slider_rect_lat.width()
            self.set_latitude(round(-90.0 + (max(0.0, min(1.0, frac)) * 180.0), 1))
            return

        if self._slider_rect_omega.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            event.accept()
            self._drag_target = "omega"
            frac = (pos.x() - self._slider_rect_omega.left()) / self._slider_rect_omega.width()
            self.set_rotation_speed(round(max(0.0, min(1.0, frac)) * 3.0, 2))
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._drag_target == "latitude":
            event.accept()
            frac = (pos.x() - self._slider_rect_lat.left()) / self._slider_rect_lat.width()
            self.set_latitude(round(-90.0 + (max(0.0, min(1.0, frac)) * 180.0), 1))
            return
        elif self._drag_target == "omega":
            event.accept()
            frac = (pos.x() - self._slider_rect_omega.left()) / self._slider_rect_omega.width()
            self.set_rotation_speed(round(max(0.0, min(1.0, frac)) * 3.0, 2))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_target is not None:
            event.accept()
            self._drag_target = None
            return
        super().mouseReleaseEvent(event)

    # ── Painting & Rendering ───────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self.WIDTH
        h = self.HEIGHT

        # 1. Dark Glass Card Background
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, 12.0, 12.0)
        painter.fillPath(card_path, QBrush(QColor(15, 23, 42, 245)))
        painter.setPen(QPen(QColor(56, 189, 248, 80), 1.2))
        painter.drawPath(card_path)

        # 2. Header & Title Bar
        self._paint_header(painter)

        # 3. Control Buttons (Rows 1 & 2)
        self._paint_controls(painter)

        # 4. Main Simulation Canvas (Coriolis Vector Field & Particles)
        self._paint_simulation_canvas(painter)

        # 5. Telemetry & Atmospheric Dynamics HUD (Right Side)
        self._paint_telemetry_hud(painter)

        # 6. Sliders (Latitude and Rotation Speed)
        self._paint_sliders(painter)

        painter.restore()

    def _paint_header(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(16, 26, "Vật Lý Khí Quyển: Hiệu Ứng Coriolis, Gió Địa Chuyển & Xoáy Bão")

        # Close button [X]
        c_rect = self._btn_rects.get("btn_close", QRectF())
        if not c_rect.isEmpty():
            painter.setPen(QPen(QColor("#ef4444"), 1.0))
            painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
            painter.drawRoundedRect(c_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#fca5a5")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕ Đóng")

    def _paint_controls(self, painter: QPainter) -> None:
        # Row 1 Presets
        for code, preset in PRESETS.items():
            rect = self._btn_rects.get(f"preset_{code}")
            if not rect:
                continue
            is_active = (self._preset_code == code)
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(56, 189, 248, 60) if is_active else QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, preset.name_vi.split(" (")[0])

        # Row 2 Controls
        # Lat - / +
        for key, label in (("lat_minus", "-15° Vĩ"), ("lat_plus", "+15° Vĩ")):
            rect = self._btn_rects.get(key)
            if rect:
                painter.setPen(QPen(QColor("#64748b"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
                painter.drawRoundedRect(rect, 4.0, 4.0)
                painter.setPen(QPen(QColor("#e2e8f0")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Equator (0°)
        eq_rect = self._btn_rects.get("set_equator")
        if eq_rect:
            is_eq = (abs(self._latitude_deg) < 1.0)
            painter.setPen(QPen(QColor("#10b981") if is_eq else QColor("#64748b"), 1.0))
            painter.setBrush(QBrush(QColor(16, 185, 129, 40) if is_eq else QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(eq_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#34d399") if is_eq else QColor("#cbd5e1")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_eq else QFont.Weight.Normal))
            painter.drawText(eq_rect, Qt.AlignmentFlag.AlignCenter, "Xích đạo (0°)")

        # Toggles
        toggles = [
            ("toggle_isobars", "Đường đẳng áp", self._show_isobars, "#f59e0b"),
            ("toggle_forces", "Vectơ lực", self._show_force_vectors, "#38bdf8"),
        ]
        for key, label, active, col_hex in toggles:
            rect = self._btn_rects.get(key)
            if rect:
                c = QColor(col_hex)
                painter.setPen(QPen(c if active else QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 50 if active else 15)))
                painter.drawRoundedRect(rect, 4.0, 4.0)
                painter.setPen(QPen(c if active else QColor("#94a3b8")))
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if active else QFont.Weight.Normal))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Play / Pause and Reset
        pp_rect = self._btn_rects.get("btn_play_pause")
        if pp_rect:
            painter.setPen(QPen(QColor("#a855f7"), 1.0))
            painter.setBrush(QBrush(QColor(168, 85, 247, 40)))
            painter.drawRoundedRect(pp_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#d8b4fe")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(pp_rect, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if self._is_playing else "▶ Chạy")

        res_rect = self._btn_rects.get("btn_reset")
        if res_rect:
            painter.setPen(QPen(QColor("#64748b"), 1.0))
            painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(res_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(res_rect, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

    def _paint_simulation_canvas(self, painter: QPainter) -> None:
        """Paint visualizer domain with rotating Earth background, isobars, and flow particles."""
        canvas_rect = QRectF(16.0, 102.0, 440.0, 335.0)

        # Canvas Frame
        bg_path = QPainterPath()
        bg_path.addRoundedRect(canvas_rect, 6.0, 6.0)
        painter.fillPath(bg_path, QBrush(QColor(10, 15, 30, 240)))
        painter.setPen(QPen(QColor(30, 41, 59, 200), 1.0))
        painter.drawPath(bg_path)

        cx = canvas_rect.center().x()
        cy = canvas_rect.center().y()
        scale = 55.0  # pixels per coordinate unit

        scenario = self._preset.scenario_type
        f = self.get_coriolis_parameter()

        # 1. Paint Isobars (Đường đẳng áp)
        if self._show_isobars:
            painter.setPen(QPen(QColor(245, 158, 11, 70), 1.0, Qt.PenStyle.DashLine))
            if scenario == "cyclone":
                # Concentric circles representing isobars around low pressure eye
                for r_u in (0.6, 1.2, 1.8, 2.4):
                    painter.drawEllipse(QPointF(cx, cy), r_u * scale, r_u * scale)
                # Low pressure marker [L]
                painter.setPen(QPen(QColor("#ef4444")))
                painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                painter.drawText(int(cx - 7), int(cy + 6), "L")
                painter.setFont(QFont("Segoe UI", 6))
                painter.drawText(int(cx - 18), int(cy + 18), "Áp thấp")
            elif scenario == "geostrophic":
                # Horizontal parallel isobars
                for idx, y_u in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
                    py = cy - (y_u * scale)
                    painter.drawLine(QPointF(canvas_rect.left() + 10, py), QPointF(canvas_rect.right() - 10, py))
                    painter.setPen(QPen(QColor("#f59e0b")))
                    painter.setFont(QFont("Consolas", 6))
                    p_val = 1000 + (idx * 4)
                    painter.drawText(int(canvas_rect.left() + 14), int(py - 3), f"{p_val} hPa")
                    painter.setPen(QPen(QColor(245, 158, 11, 70), 1.0, Qt.PenStyle.DashLine))

        # 2. Paint Scenario Dynamics
        if scenario in ("cyclone", "geostrophic"):
            # Air parcel flow particles
            for i in range(self._num_particles):
                px = cx + (self._particle_pos[i, 0] * scale)
                py = cy - (self._particle_pos[i, 1] * scale)
                if not canvas_rect.contains(px, py):
                    continue

                vx, vy = self.compute_velocity_field(self._particle_pos[i, 0], self._particle_pos[i, 1])
                speed = math.hypot(vx, vy)

                # Particle head
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#38bdf8")))
                painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

                # Particle tail (direction vector)
                if speed > 0.05:
                    tx = px - (vx * 12.0)
                    ty = py + (vy * 12.0)
                    tail_pen = QPen(QColor(56, 189, 248, 120), 1.2)
                    painter.setPen(tail_pen)
                    painter.drawLine(QPointF(px, py), QPointF(tx, ty))

            # Force Vectors (PGF and Coriolis) at probe location
            if self._show_force_vectors:
                probe_x = 1.3
                probe_y = 0.5
                pr_px = cx + (probe_x * scale)
                pr_py = cy - (probe_y * scale)

                # PGF Vector (Inward towards 0, 0) - Amber
                pgf_len = 38.0
                pr_ang = math.atan2(-probe_y, -probe_x)
                pgf_x = pr_px + (pgf_len * math.cos(pr_ang))
                pgf_y = pr_py - (pgf_len * math.sin(pr_ang))
                painter.setPen(QPen(QColor("#f59e0b"), 2.0))
                painter.drawLine(QPointF(pr_px, pr_py), QPointF(pgf_x, pgf_y))
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                painter.drawText(int(pgf_x + 4), int(pgf_y + 4), "F_PGF")

                # Coriolis Vector (Deflected 90 degrees relative to velocity) - Emerald
                vx, vy = self.compute_velocity_field(probe_x, probe_y)
                v_mag = math.hypot(vx, vy)
                if v_mag > 0.05:
                    c_ang = math.atan2(vy, vx) + (math.pi / 2.0 if f < 0 else -math.pi / 2.0)
                    c_len = min(50.0, abs(f) * 35.0)
                    cor_x = pr_px + (c_len * math.cos(c_ang))
                    cor_y = pr_py - (c_len * math.sin(c_ang))
                    painter.setPen(QPen(QColor("#10b981"), 2.0))
                    painter.drawLine(QPointF(pr_px, pr_py), QPointF(cor_x, cor_y))
                    painter.drawText(int(cor_x + 4), int(cor_y + 4), "F_Coriolis")

        elif scenario == "ballistic":
            # Paint comparison of Inertial vs Rotating Frame Projectile Paths
            # Firing cannon at bottom
            cannon_pt = QPointF(cx, cy + (2.2 * scale))
            painter.setPen(QPen(QColor("#64748b"), 1.0))
            painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
            painter.drawEllipse(cannon_pt, 8.0, 8.0)

            # Target target at top
            target_pt = QPointF(cx, cy - (2.2 * scale))
            painter.setPen(QPen(QColor("#ef4444"), 1.5))
            painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
            painter.drawEllipse(target_pt, 12.0, 12.0)
            painter.drawEllipse(target_pt, 4.0, 4.0)

            # 1. Inertial Straight Path (Dashed White)
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(cannon_pt, target_pt)

            # 2. Rotating Frame Deflected Path (Emerald Curve)
            if len(self._ballistic_traj_rot) > 1:
                path_rot = QPainterPath()
                path_rot.moveTo(cx + (self._ballistic_traj_rot[0][0] * scale), cy - (self._ballistic_traj_rot[0][1] * scale))
                for pt in self._ballistic_traj_rot[1:]:
                    path_rot.lineTo(cx + (pt[0] * scale), cy - (pt[1] * scale))
                painter.setPen(QPen(QColor("#10b981"), 2.2))
                painter.drawPath(path_rot)

                # Projectile current position
                cur_pt = self._ballistic_traj_rot[-1]
                painter.setPen(QPen(QColor("#ffffff"), 1.0))
                painter.setBrush(QBrush(QColor("#38bdf8")))
                painter.drawEllipse(QPointF(cx + (cur_pt[0] * scale), cy - (cur_pt[1] * scale)), 5.0, 5.0)

            # Legend
            painter.setPen(QPen(QColor(255, 255, 255, 180)))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(canvas_rect.left() + 14), int(canvas_rect.top() + 20), "-- Quỹ đạo hệ quán tính (Thẳng)")
            painter.setPen(QPen(QColor("#10b981")))
            painter.drawText(int(canvas_rect.left() + 14), int(canvas_rect.top() + 35), "― Quỹ đạo trên Trái Đất (Lệch Coriolis)")

        elif scenario == "foucault":
            # Paint Foucault Pendulum Precession
            # Outer ring with hour marks (0 to 24h)
            ring_r = 120.0
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
            painter.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

            # Compass Cardinal Points
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(int(cx - 4), int(cy - ring_r - 6), "B")
            painter.drawText(int(cx - 4), int(cy + ring_r + 14), "N")
            painter.drawText(int(cx + ring_r + 6), int(cy + 4), "Đ")
            painter.drawText(int(cx - ring_r - 16), int(cy + 4), "T")

            # Plane of oscillation (rotated by _foucault_angle)
            ang = self._foucault_angle
            p1 = QPointF(cx + (ring_r * math.cos(ang)), cy - (ring_r * math.sin(ang)))
            p2 = QPointF(cx - (ring_r * math.cos(ang)), cy + (ring_r * math.sin(ang)))
            painter.setPen(QPen(QColor("#38bdf8"), 2.2))
            painter.drawLine(p1, p2)

            # Bob oscillation along line
            bob_phase = math.sin(self._sim_time * 3.5)
            bob_r = ring_r * 0.85 * bob_phase
            bob_pt = QPointF(cx + (bob_r * math.cos(ang)), cy - (bob_r * math.sin(ang)))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.drawEllipse(bob_pt, 7.0, 7.0)

            # Precession rotation arc
            painter.setPen(QPen(QColor(245, 158, 11, 150), 1.2, Qt.PenStyle.DashLine))
            painter.drawEllipse(QPointF(cx, cy), ring_r * 0.4, ring_r * 0.4)

    def _paint_telemetry_hud(self, painter: QPainter) -> None:
        """Paint exact atmospheric dynamics metrics, rotation periods, and laws."""
        panel_rect = QRectF(466.0, 102.0, 278.0, 335.0)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(panel_rect, 6.0, 6.0)
        painter.fillPath(bg_path, QBrush(QColor(10, 15, 30, 240)))
        painter.setPen(QPen(QColor(30, 41, 59, 200), 1.0))
        painter.drawPath(bg_path)

        x_left = panel_rect.left() + 14.0
        cur_y = panel_rect.top() + 20.0

        # Title
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(x_left), int(cur_y), "THÔNG SỐ ĐỘNG LỰC HỌC KHÍ QUYỂN")
        cur_y += 18.0

        f = self.get_coriolis_parameter()
        phi = self._latitude_deg
        omega = self._rotation_speed

        # Hemisphere classification
        if phi > 0.5:
            hem_str = "Bắc Bán Cầu (Lệch PHẢI)"
            hem_col = "#38bdf8"
        elif phi < -0.5:
            hem_str = "Nam Bán Cầu (Lệch TRÁI)"
            hem_col = "#f472b6"
        else:
            hem_str = "Xích Đạo (Coriolis = 0)"
            hem_col = "#10b981"

        # Foucault Precession Period T = 24 / |sin φ|
        sin_phi = abs(math.sin(math.radians(phi)))
        if sin_phi > 0.05:
            foucault_t_str = f"{24.0 / sin_phi:.1f} giờ"
        else:
            foucault_t_str = "∞ (Không quay tại Xích đạo)"

        metrics = [
            ("Vĩ độ quan sát (φ):", f"{phi:+.1f}°", hem_col),
            ("Bán cầu Trái Đất:", hem_str, hem_col),
            ("Thông số Coriolis (f):", f"{f:+.3f} (2Ω sin φ)", "#f59e0b"),
            ("Tốc độ tự quay (Ω):", f"{omega:.2f}× chuẩn", "#cbd5e1"),
            ("Chu kỳ Foucault (T):", foucault_t_str, "#a855f7"),
            ("Hướng xoáy bão:", "Ngược chiều KĐH" if phi > 0 else ("Cùng chiều KĐH" if phi < 0 else "Không tạo bão"), "#34d399"),
        ]

        for label, val_str, col in metrics:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(x_left), int(cur_y), label)

            painter.setPen(QPen(QColor(col)))
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.drawText(int(x_left + 125.0), int(cur_y), val_str)
            cur_y += 18.0

        # Mathematical Formula Box
        f_rect = QRectF(x_left, cur_y + 4.0, panel_rect.width() - 28.0, 72.0)
        f_path = QPainterPath()
        f_path.addRoundedRect(f_rect, 5.0, 5.0)
        painter.fillPath(f_path, QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(56, 189, 248, 100), 1.0))
        painter.drawPath(f_path)

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 16), "Gia tốc Coriolis & Gió địa chuyển:")
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 32), " a_C = -2 (Ω × v) = 2 Ω v sin(φ)")
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 48), " v_g = (1 / ρf) · (k̂ × ∇P)")
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 64), " Định luật Buys-Ballot: Hạ áp nằm bên trái gió")

        # Description
        desc_rect = QRectF(x_left, f_rect.bottom() + 10.0, panel_rect.width() - 28.0, 48.0)
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(desc_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), self._preset.description)

    def _paint_sliders(self, painter: QPainter) -> None:
        """Paint interactive latitude and rotation speed sliders."""
        # Slider 1: Latitude φ (-90° to +90°)
        lat_rect = self._slider_rect_lat
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(int(lat_rect.left() - 85.0), int(lat_rect.center().y() + 4.0), "Vĩ độ (φ):")

        # Track
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(lat_rect, 3.0, 3.0)

        # Center line (Equator = 0°)
        eq_x = lat_rect.left() + (lat_rect.width() * 0.5)
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawLine(QPointF(eq_x, lat_rect.top() - 2.0), QPointF(eq_x, lat_rect.bottom() + 2.0))

        # Filled bar from equator to current lat
        lat_frac = (self._latitude_deg - (-90.0)) / 180.0
        cur_x = lat_rect.left() + (lat_frac * lat_rect.width())
        bar_x = min(eq_x, cur_x)
        bar_w = abs(cur_x - eq_x)
        fill_col = QColor("#38bdf8") if self._latitude_deg >= 0 else QColor("#f472b6")
        painter.fillRect(QRectF(bar_x, lat_rect.top(), bar_w, lat_rect.height()), QBrush(fill_col))

        # Handle
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(fill_col))
        painter.drawEllipse(QPointF(cur_x, lat_rect.center().y()), 6.0, 6.0)

        painter.setPen(QPen(fill_col))
        painter.drawText(int(lat_rect.right() + 8.0), int(lat_rect.center().y() + 4.0), f"{self._latitude_deg:+.1f}°")

        # Slider 2: Rotation Speed Ω (0.0 to 3.0)
        om_rect = self._slider_rect_omega
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(om_rect.left() - 75.0), int(om_rect.center().y() + 4.0), "Tự quay (Ω):")

        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(om_rect, 3.0, 3.0)

        om_frac = self._rotation_speed / 3.0
        fill_om = om_rect.width() * max(0.0, min(1.0, om_frac))
        painter.fillRect(QRectF(om_rect.left(), om_rect.top(), fill_om, om_rect.height()), QBrush(QColor("#f59e0b")))

        hx = om_rect.left() + fill_om
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(QColor("#fbbf24")))
        painter.drawEllipse(QPointF(hx, om_rect.center().y()), 6.0, 6.0)

        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(int(om_rect.right() + 8.0), int(om_rect.center().y() + 4.0), f"{self._rotation_speed:.2f}×")

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Stamp current Coriolis simulation onto LessonBoard."""
        stamped: list[QGraphicsItem] = []
        copy_item = InteractiveCoriolisEarthItem(pos=(self.x(), self.y()))
        copy_item.load_preset(self._preset_code)
        copy_item.set_latitude(self._latitude_deg)
        copy_item.set_rotation_speed(self._rotation_speed)
        board.addItem(copy_item)
        stamped.append(copy_item)
        return stamped

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach lesson packages."""
        return {
            "type": "interactive_coriolis_earth",
            "pos": [self.x(), self.y()],
            "preset_code": self._preset_code,
            "latitude_deg": round(self._latitude_deg, 2),
            "rotation_speed": round(self._rotation_speed, 2),
            "pressure_gradient": round(self._pressure_gradient, 2),
            "show_isobars": self._show_isobars,
            "show_force_vectors": self._show_force_vectors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractiveCoriolisEarthItem:
        """Reconstruct visualizer from serialized dictionary."""
        pos = data.get("pos", [120.0, 50.0])
        item = cls(pos=(float(pos[0]), float(pos[1])))
        if data.get("preset_code"):
            item.load_preset(str(data.get("preset_code")))
        item.set_latitude(float(data.get("latitude_deg", 21.0)))
        item.set_rotation_speed(float(data.get("rotation_speed", 1.0)))
        item._show_isobars = bool(data.get("show_isobars", True))
        item._show_force_vectors = bool(data.get("show_force_vectors", True))
        return item
