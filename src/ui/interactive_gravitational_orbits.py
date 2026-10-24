"""
AirTeach AI - Interactive Gravitational N-Body Planetary Orbit & Lagrange Points Simulator
========================================================================================
Authentic celestial mechanics and orbital dynamics engine:
  - Newton's Law of Universal Gravitation: F_ij = G * m_i * m_j / (r_ij^2 + epsilon^2)
  - Symplectic Velocity Verlet integrator (O(dt^2)) strictly preserving total mechanical energy E = Ek + U
    and total angular momentum L_z over prolonged orbital integrations without secular dissipation.
  - 5 Authentic Presets:
      1. INNER_SOLAR_SYSTEM: Sun (M☉) and 4 terrestrial planets (Mercury, Venus, Earth, Mars).
      2. FIGURE_EIGHT_3BODY: Famous Chenciner-Montgomery (2000) 3-body equal-mass choreographic orbit.
      3. EARTH_MOON_LAGRANGE: Earth-Moon system with 5 Lagrange equilibrium points (L1, L2, L3, L4, L5).
      4. BINARY_STAR_PLANET: Equal-mass binary star pair orbiting barycenter with circumbinary P-type planet.
      5. CHAOTIC_3BODY: Burrau (1913) Pythagorean 3-body problem showing chaotic gravitational slingshots.
  - Touchless air gestures (Air Pinch to drag planet position / velocity vector, Air Tap on controls).
  - Vector diagram stamping to LessonBoard via PhysicsTools.create_gravitational_orbits_diagram.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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
)

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class CelestialBody:
    name: str
    mass: float              # Mass in simulation units
    x: float                 # Position X relative to simulator center
    y: float                 # Position Y relative to simulator center
    vx: float                # Velocity X
    vy: float                # Velocity Y
    radius: float            # Display circle radius in px
    color: str               # Hex color string
    fixed: bool = False      # If True, mass does not accelerate
    trail: deque = field(default_factory=lambda: deque(maxlen=70))
    # Scratch accelerations for Velocity Verlet
    ax: float = 0.0
    ay: float = 0.0


# Gravitational Constant in simulation units
G_CONST = 1000.0
PLUMMER_EPSILON = 6.0  # Softening distance to prevent infinite forces on close encounters


class InteractiveGravitationalOrbitsItem(QGraphicsObject):
    """
    Interactive Gravitational N-Body Planetary Orbit & Lagrange Points Simulator item.
    """

    orbits_updated = Signal(float, float, float)  # (energy_total, kinetic_energy, potential_energy)
    preset_changed = Signal(str)
    stamp_requested = Signal()

    PRESET_NAMES = [
        "INNER_SOLAR_SYSTEM",
        "FIGURE_EIGHT_3BODY",
        "EARTH_MOON_LAGRANGE",
        "BINARY_STAR_PLANET",
        "CHAOTIC_3BODY",
    ]

    PRESET_LABELS = {
        "INNER_SOLAR_SYSTEM": "Hệ Mặt Trời Cận Điểm",
        "FIGURE_EIGHT_3BODY": "Số 8 Chenciner-Montgomery",
        "EARTH_MOON_LAGRANGE": "5 Điểm Lagrange (L₁-L₅)",
        "BINARY_STAR_PLANET": "Sao Đôi & Hành Tinh",
        "CHAOTIC_3BODY": "Tam Giác 3-4-5 Burrau",
    }

    def __init__(
        self,
        pos: tuple[float, float] = (450.0, 240.0),
        width: float = 800.0,
        height: float = 520.0,
        preset: str = "INNER_SOLAR_SYSTEM",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Simulation Time & Physics Engine
        self._dt = 0.008
        self._speed_multiplier = 1.0
        self._is_paused = False
        self._show_trails = True
        self._show_vectors = True
        self._show_lagrange = True
        self._preset = preset.upper() if preset.upper() in self.PRESET_NAMES else "INNER_SOLAR_SYSTEM"

        # Bodies list
        self._bodies: list[CelestialBody] = []
        self._center_canvas = QPointF(270.0, 210.0)  # Center of space canvas relative to item

        # Energy & Metrics
        self._energy_k = 0.0
        self._energy_u = 0.0
        self._energy_total = 0.0
        self._angular_momentum_z = 0.0

        # Dragging state for air gestures or mouse
        self._dragged_body_idx: Optional[int] = None
        self._is_dragging_velocity: bool = False

        # Build initial preset
        self._load_preset(self._preset)

        # 30 FPS physics / render timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Layout Geometry Constants
        self._canvas_x = 20.0
        self._canvas_y = 48.0
        self._canvas_w = 500.0
        self._canvas_h = 330.0

        self._hud_x = 535.0
        self._hud_y = 48.0
        self._hud_w = self._width - self._hud_x - 20.0  # 245.0 px
        self._hud_h = 330.0

        # Window Controls
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Deck Row 1: Presets (y = 390.0, h = 26.0)
        self._rect_btn_prev_preset = QRectF(20.0, 390.0, 30.0, 26.0)
        self._rect_btn_preset = QRectF(55.0, 390.0, 250.0, 26.0)
        self._rect_btn_next_preset = QRectF(310.0, 390.0, 30.0, 26.0)

        # Deck Row 1 (Right): Speed Controls
        self._rect_btn_pause = QRectF(355.0, 390.0, 75.0, 26.0)
        self._rect_btn_speed_05 = QRectF(435.0, 390.0, 48.0, 26.0)
        self._rect_btn_speed_10 = QRectF(488.0, 390.0, 48.0, 26.0)
        self._rect_btn_speed_20 = QRectF(541.0, 390.0, 48.0, 26.0)
        self._rect_btn_speed_50 = QRectF(594.0, 390.0, 48.0, 26.0)

        # Deck Row 2: Toggles & Actions (y = 426.0, h = 26.0)
        self._rect_btn_trails = QRectF(20.0, 426.0, 105.0, 26.0)
        self._rect_btn_vectors = QRectF(130.0, 426.0, 115.0, 26.0)
        self._rect_btn_lagrange = QRectF(250.0, 426.0, 125.0, 26.0)
        self._rect_btn_reset = QRectF(380.0, 426.0, 85.0, 26.0)
        self._rect_btn_stamp = QRectF(470.0, 426.0, 172.0, 26.0)

    # ─── Preset Initialization ───────────────────────────────────────────────

    def _load_preset(self, preset_name: str) -> None:
        """Initializes celestial bodies for selected preset."""
        self._bodies.clear()
        name = preset_name.upper()

        if name == "FIGURE_EIGHT_3BODY":
            # Authentic Chenciner & Montgomery (2000) choreography
            m = 1000.0
            scale = 88.0
            v_scale = math.sqrt(G_CONST * m / scale)
            x1 = 0.97000436 * scale
            y1 = -0.24308753 * scale
            vx1 = 0.46620531 * v_scale
            vy1 = 0.43236573 * v_scale

            b1 = CelestialBody("Vật thể 1", m, x1, y1, vx1, vy1, 8.0, "#f59e0b")
            b2 = CelestialBody("Vật thể 2", m, -x1, -y1, vx1, vy1, 8.0, "#10b981")
            b3 = CelestialBody("Vật thể 3", m, 0.0, 0.0, -2.0 * vx1, -2.0 * vy1, 8.0, "#ec4899")
            self._bodies = [b1, b2, b3]

        elif name == "EARTH_MOON_LAGRANGE":
            # Earth (Primary) and Moon (Secondary)
            m_earth = 10000.0
            m_moon = 150.0
            r_moon = 135.0
            # Orbital speed for circular orbit v = sqrt(G*M/R)
            v_moon = math.sqrt(G_CONST * m_earth / r_moon)

            earth = CelestialBody("Trái Đất (M₁)", m_earth, -2.0, 0.0, 0.0, -v_moon * (m_moon / m_earth), 12.0, "#0284c7")
            moon = CelestialBody("Mặt Trăng (M₂)", m_moon, r_moon, 0.0, 0.0, v_moon, 5.5, "#cbd5e1")

            # Satellite near L4 equilateral point
            l4_x = r_moon * math.cos(math.pi / 3.0)
            l4_y = -r_moon * math.sin(math.pi / 3.0)
            v_l4_x = -v_moon * math.sin(math.pi / 3.0)
            v_l4_y = -v_moon * math.cos(math.pi / 3.0)
            sat = CelestialBody("Vệ Tinh (L₄)", 0.01, l4_x, l4_y, v_l4_x, v_l4_y, 4.0, "#facc15")

            self._bodies = [earth, moon, sat]

        elif name == "BINARY_STAR_PLANET":
            # Two stars of equal mass orbiting barycenter
            m_star = 4500.0
            sep = 60.0
            v_star = math.sqrt(G_CONST * m_star / (4.0 * sep))

            star_a = CelestialBody("Sao A", m_star, -sep, 0.0, 0.0, -v_star, 9.5, "#f59e0b")
            star_b = CelestialBody("Sao B", m_star, sep, 0.0, 0.0, v_star, 9.5, "#38bdf8")

            # Circumbinary planet in outer P-type orbit
            r_p = 155.0
            v_p = math.sqrt(G_CONST * (2.0 * m_star) / r_p)
            planet = CelestialBody("Hành Tinh P", 0.05, 0.0, -r_p, v_p, 0.0, 4.5, "#22c55e")

            self._bodies = [star_a, star_b, planet]

        elif name == "CHAOTIC_3BODY":
            # Burrau (1913) Pythagorean 3-4-5 problem
            b1 = CelestialBody("Khối 3m (3000)", 3000.0, -40.0, 50.0, 0.0, 0.0, 7.5, "#ef4444")
            b2 = CelestialBody("Khối 4m (4000)", 4000.0, 50.0, -20.0, 0.0, 0.0, 8.5, "#3b82f6")
            b3 = CelestialBody("Khối 5m (5000)", 5000.0, -20.0, -40.0, 0.0, 0.0, 9.5, "#10b981")
            self._bodies = [b1, b2, b3]

        else:
            # INNER_SOLAR_SYSTEM default
            sun = CelestialBody("Mặt Trời (M☉)", 10000.0, 0.0, 0.0, 0.0, 0.0, 14.0, "#fbbf24", fixed=True)
            # Planets: (name, mass, dist, radius, color)
            specs = [
                ("Sao Thủy", 0.05, 42.0, 3.5, "#94a3b8"),
                ("Sao Kim", 0.82, 68.0, 4.8, "#eab308"),
                ("Trái Đất", 1.00, 102.0, 5.2, "#38bdf8"),
                ("Sao Hỏa", 0.11, 142.0, 4.0, "#ef4444"),
            ]
            self._bodies = [sun]
            for pname, pmass, pdist, prad, pcol in specs:
                # Circular orbital velocity v = sqrt(G*M/r)
                v_circ = math.sqrt(G_CONST * 10000.0 / pdist)
                planet = CelestialBody(pname, pmass, 0.0, -pdist, v_circ, 0.0, prad, pcol)
                self._bodies.append(planet)

        # Initialize initial accelerations
        self._compute_all_accelerations()
        self._update_energies()

    # ─── Symplectic Velocity Verlet Integrator ────────────────────────────────

    def _compute_all_accelerations(self) -> None:
        """Computes gravitational acceleration vectors for all active bodies."""
        n = len(self._bodies)
        for i in range(n):
            self._bodies[i].ax = 0.0
            self._bodies[i].ay = 0.0

        for i in range(n):
            if self._bodies[i].fixed:
                continue
            xi = self._bodies[i].x
            yi = self._bodies[i].y
            for j in range(n):
                if i == j:
                    continue
                dx = self._bodies[j].x - xi
                dy = self._bodies[j].y - yi
                dist_sq = dx * dx + dy * dy + PLUMMER_EPSILON * PLUMMER_EPSILON
                dist = math.sqrt(dist_sq)
                inv_dist3 = 1.0 / (dist_sq * dist)
                f_factor = G_CONST * self._bodies[j].mass * inv_dist3
                self._bodies[i].ax += f_factor * dx
                self._bodies[i].ay += f_factor * dy

    def step_simulation(self, dt: float) -> None:
        """
        Executes one full symplectic Velocity Verlet integration time step:
        1. r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. Recompute a(t + dt)
        3. v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
        """
        n = len(self._bodies)

        # 1. Update positions using current velocities and accelerations
        for i in range(n):
            b = self._bodies[i]
            if b.fixed:
                continue
            b.x += b.vx * dt + 0.5 * b.ax * dt * dt
            b.y += b.vy * dt + 0.5 * b.ay * dt * dt

        # Store old accelerations
        old_ax = [b.ax for b in self._bodies]
        old_ay = [b.ay for b in self._bodies]

        # 2. Compute new accelerations at updated positions
        self._compute_all_accelerations()

        # 3. Update velocities using average acceleration
        for i in range(n):
            b = self._bodies[i]
            if b.fixed:
                continue
            b.vx += 0.5 * (old_ax[i] + b.ax) * dt
            b.vy += 0.5 * (old_ay[i] + b.ay) * dt

            # Append current position to trail
            if self._show_trails:
                b.trail.append((b.x, b.y))

        self._update_energies()

    def _update_energies(self) -> None:
        """Computes kinetic, potential, total energy and angular momentum."""
        ek = 0.0
        ep = 0.0
        lz = 0.0
        n = len(self._bodies)

        for i in range(n):
            b = self._bodies[i]
            v_sq = b.vx * b.vx + b.vy * b.vy
            ek += 0.5 * b.mass * v_sq
            # Angular momentum L_z = m * (x*vy - y*vx)
            lz += b.mass * (b.x * b.vy - b.y * b.vx)

            for j in range(i + 1, n):
                bj = self._bodies[j]
                dx = bj.x - b.x
                dy = bj.y - b.y
                dist = math.sqrt(dx * dx + dy * dy + PLUMMER_EPSILON * PLUMMER_EPSILON)
                ep -= (G_CONST * b.mass * bj.mass) / dist

        self._energy_k = ek
        self._energy_u = ep
        self._energy_total = ek + ep
        self._angular_momentum_z = lz
        self.orbits_updated.emit(self._energy_total, self._energy_k, self._energy_u)

    def _on_sim_tick(self) -> None:
        """Timer callback (30 FPS) executing sub-stepped physics for numerical stability."""
        if self._is_paused:
            return

        # 3 sub-steps per frame for smooth symplectic stability
        sub_steps = 3
        sub_dt = (self._dt * self._speed_multiplier) / sub_steps
        for _ in range(sub_steps):
            self.step_simulation(sub_dt)

        self.update()

    # ─── Public Properties & Setters ─────────────────────────────────────────

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def speed_multiplier(self) -> float:
        return self._speed_multiplier

    @property
    def bodies(self) -> list[CelestialBody]:
        return self._bodies

    @property
    def total_energy(self) -> float:
        return self._energy_total

    @property
    def kinetic_energy(self) -> float:
        return self._energy_k

    @property
    def potential_energy(self) -> float:
        return self._energy_u

    @property
    def angular_momentum(self) -> float:
        return self._angular_momentum_z

    def set_preset(self, preset_name: str) -> None:
        """Switches active orbital preset and resets bodies."""
        name = preset_name.upper()
        if name in self.PRESET_NAMES and name != self._preset:
            self._preset = name
            self._load_preset(self._preset)
            self.preset_changed.emit(self._preset)
            self.update()

    def cycle_preset(self, step: int = 1) -> None:
        """Cycles to next/previous celestial preset."""
        idx = self.PRESET_NAMES.index(self._preset)
        next_idx = (idx + step) % len(self.PRESET_NAMES)
        self.set_preset(self.PRESET_NAMES[next_idx])

    def toggle_pause(self) -> bool:
        """Toggles simulation pause state."""
        self._is_paused = not self._is_paused
        self.update()
        return self._is_paused

    def set_speed(self, multiplier: float) -> None:
        """Sets time speed multiplier."""
        self._speed_multiplier = max(0.1, min(10.0, float(multiplier)))
        self.update()

    def toggle_trails(self) -> bool:
        """Toggles trajectory trails visibility."""
        self._show_trails = not self._show_trails
        if not self._show_trails:
            for b in self._bodies:
                b.trail.clear()
        self.update()
        return self._show_trails

    def toggle_vectors(self) -> bool:
        """Toggles velocity and force vectors visibility."""
        self._show_vectors = not self._show_vectors
        self.update()
        return self._show_vectors

    def toggle_lagrange(self) -> bool:
        """Toggles Lagrange points visualization."""
        self._show_lagrange = not self._show_lagrange
        self.update()
        return self._show_lagrange

    def reset_simulation(self) -> None:
        """Resets active preset to initial coordinates and velocities."""
        self._load_preset(self._preset)
        self.update()

    def close(self) -> None:
        """Closes and hides item."""
        self._timer.stop()
        self.setVisible(False)
        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps celestial gravitational orbits, velocity vectors, and mechanics equations to LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        items = PhysicsTools.create_gravitational_orbits_diagram(
            center=(cx, cy),
            width=460.0,
            height=270.0,
            preset=self._preset,
            show_lagrange=self._show_lagrange,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None and hasattr(scene, "add_item_with_undo"):
                target_board = scene  # type: ignore

        if target_board is not None and hasattr(target_board, "add_item_with_undo"):
            for item in items:
                target_board.add_item_with_undo(item)  # type: ignore
        elif self.scene() is not None:
            for item in items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        logger.info("Stamped Gravitational Orbits diagram to LessonBoard (%s)", self._preset)
        return items

    # ─── Air Gesture & Mouse Interactions ────────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF, prev_pos: Optional[QPointF] = None) -> bool:
        """
        Handles touchless air pinch gesture.
        If pinching a planet or velocity arrow, manipulates position or velocity vector.
        If pinching background and prev_pos is provided, translates the simulator window.
        """
        local_pos = self.mapFromScene(scene_pos)

        # Check celestial body hit
        body_idx = self._hit_test_body(local_pos)
        if body_idx is not None and prev_pos is not None:
            delta = scene_pos - prev_pos
            b = self._bodies[body_idx]
            if not b.fixed:
                b.x += delta.x()
                b.y += delta.y()
                self._compute_all_accelerations()
                self._update_energies()
                self.update()
                return True

        # Window drag if outside interactive canvas controls
        if prev_pos is not None:
            delta = scene_pos - prev_pos
            self.setPos(self.pos() + delta)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Handles touchless air tap gesture on buttons."""
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_prev_preset.contains(local_pos):
            self.cycle_preset(-1)
            return True
        elif self._rect_btn_next_preset.contains(local_pos) or self._rect_btn_preset.contains(local_pos):
            self.cycle_preset(1)
            return True
        elif self._rect_btn_pause.contains(local_pos):
            self.toggle_pause()
            return True
        elif self._rect_btn_speed_05.contains(local_pos):
            self.set_speed(0.5)
            return True
        elif self._rect_btn_speed_10.contains(local_pos):
            self.set_speed(1.0)
            return True
        elif self._rect_btn_speed_20.contains(local_pos):
            self.set_speed(2.0)
            return True
        elif self._rect_btn_speed_50.contains(local_pos):
            self.set_speed(5.0)
            return True
        elif self._rect_btn_trails.contains(local_pos):
            self.toggle_trails()
            return True
        elif self._rect_btn_vectors.contains(local_pos):
            self.toggle_vectors()
            return True
        elif self._rect_btn_lagrange.contains(local_pos):
            self.toggle_lagrange()
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True

        return False

    def _hit_test_body(self, local_pos: QPointF) -> Optional[int]:
        """Returns index of body clicked or pinched, or None."""
        cx = self._canvas_x + self._center_canvas.x()
        cy = self._canvas_y + self._center_canvas.y()

        for idx, b in enumerate(self._bodies):
            bx = cx + b.x
            by = cy + b.y
            dist = math.hypot(local_pos.x() - bx, local_pos.y() - by)
            if dist <= max(14.0, b.radius + 6.0):
                return idx
        return None

    def mousePressEvent(self, event) -> None:
        """Mouse click event handler."""
        pos = event.pos()
        if self.handle_air_tap(self.mapToScene(pos)):
            event.accept()
            return

        b_idx = self._hit_test_body(pos)
        if b_idx is not None:
            self._dragged_body_idx = b_idx
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Mouse drag on planet position."""
        if self._dragged_body_idx is not None and 0 <= self._dragged_body_idx < len(self._bodies):
            pos = event.pos()
            cx = self._canvas_x + self._center_canvas.x()
            cy = self._canvas_y + self._center_canvas.y()
            b = self._bodies[self._dragged_body_idx]
            if not b.fixed:
                b.x = pos.x() - cx
                b.y = pos.y() - cy
                self._compute_all_accelerations()
                self._update_energies()
                self.update()
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Mouse release handler."""
        self._dragged_body_idx = None
        super().mouseReleaseEvent(event)

    # ─── Bounding Rect & Painting ────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Renders deep-space canvas, orbits, celestial bodies, and orbital mechanics HUD."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Outer Window Frame
        painter.setBrush(QBrush(QColor("#0a0f1d")))  # Dark celestial slate
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(self.boundingRect(), 16.0, 16.0)

        # Title Bar
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self._width, 40.0), 16.0, 16.0)
        painter.drawRect(QRectF(0.0, 24.0, self._width, 16.0))  # Flatten bottom corners

        # Title text
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(20, 26, "🌌 MÔ PHỎNG VẠN VẬT HẤP DẪN & ĐIỂM LAGRANGE (N-BODY GRAVITATION)")

        # Close button [✕]
        painter.setBrush(QBrush(QColor("#ef4444").darker(130)))
        painter.setPen(QPen(QColor("#f87171"), 1.0))
        painter.drawRoundedRect(self._rect_btn_close, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # 2. Deep Space Canvas (Left Pane)
        can_rect = QRectF(self._canvas_x, self._canvas_y, self._canvas_w, self._canvas_h)
        painter.setBrush(QBrush(QColor("#030712")))  # Deep void obsidian
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(can_rect, 10.0, 10.0)

        # Background grid & distant stars
        painter.setClipRect(can_rect)
        cx = self._canvas_x + self._center_canvas.x()
        cy = self._canvas_y + self._center_canvas.y()

        # Origin reticle
        painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(can_rect.left()), int(cy), int(can_rect.right()), int(cy))
        painter.drawLine(int(cx), int(can_rect.top()), int(cx), int(can_rect.bottom()))

        # Distant background stars
        painter.setPen(Qt.PenStyle.NoPen)
        star_dots = [(cx - 190, cy - 110), (cx + 170, cy - 90), (cx - 140, cy + 120),
                     (cx + 190, cy + 110), (cx + 80, cy - 130), (cx - 60, cy + 140)]
        for sx, sy in star_dots:
            painter.setBrush(QBrush(QColor("#94a3b8")))
            painter.drawEllipse(QPointF(sx, sy), 1.2, 1.2)

        # Draw Lagrange equilibrium points if active in EARTH_MOON_LAGRANGE mode
        if self._preset == "EARTH_MOON_LAGRANGE" and self._show_lagrange and len(self._bodies) >= 2:
            self._paint_lagrange_points(painter, cx, cy)

        # 3. Draw Celestial Trajectory Trails
        if self._show_trails:
            for b in self._bodies:
                if len(b.trail) < 2:
                    continue
                t_color = QColor(b.color)
                t_path = QPainterPath()
                t_pts = list(b.trail)
                t_path.moveTo(cx + t_pts[0][0], cy + t_pts[0][1])
                for pt in t_pts[1:]:
                    t_path.lineTo(cx + pt[0], cy + pt[1])

                painter.setBrush(Qt.BrushStyle.NoBrush)
                t_pen = QPen(t_color, 1.3)
                t_pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(t_pen)
                painter.setOpacity(0.55)
                painter.drawPath(t_path)
                painter.setOpacity(1.0)

        # 4. Draw Celestial Bodies & Vectors
        for b in self._bodies:
            bx = cx + b.x
            by = cy + b.y

            # Body radial corona / atmosphere glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(b.color)))
            painter.setOpacity(0.25)
            painter.drawEllipse(QPointF(bx, by), b.radius * 1.8, b.radius * 1.8)
            painter.setOpacity(1.0)

            # Solid body sphere
            painter.setBrush(QBrush(QColor(b.color)))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(QPointF(bx, by), b.radius, b.radius)

            # Velocity vector arrow
            if self._show_vectors and not b.fixed:
                v_scale = 3.5
                vx_end = bx + b.vx * v_scale
                vy_end = by + b.vy * v_scale
                painter.setPen(QPen(QColor("#22c55e"), 1.4))
                painter.drawLine(QPointF(bx, by), QPointF(vx_end, vy_end))
                # Arrow head
                v_angle = math.atan2(b.vy, b.vx)
                arrow_len = 5.0
                p1 = QPointF(vx_end - arrow_len * math.cos(v_angle - 0.5), vy_end - arrow_len * math.sin(v_angle - 0.5))
                p2 = QPointF(vx_end - arrow_len * math.cos(v_angle + 0.5), vy_end - arrow_len * math.sin(v_angle + 0.5))
                painter.drawLine(QPointF(vx_end, vy_end), p1)
                painter.drawLine(QPointF(vx_end, vy_end), p2)

            # Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(int(bx + b.radius + 4), int(by + 4), b.name)

        painter.setClipping(False)

        # 5. Right Side: Orbital Mechanics & Energy HUD
        self._paint_hud(painter)

        # 6. Bottom Deck Controls
        self._paint_controls(painter)

    def _paint_lagrange_points(self, painter: QPainter, cx: float, cy: float) -> None:
        """Paints L1-L5 points and equilateral triangle lines for Earth-Moon system."""
        b_earth = self._bodies[0]
        b_moon = self._bodies[1]
        ex, ey = cx + b_earth.x, cy + b_earth.y
        mx, my = cx + b_moon.x, cy + b_moon.y

        dx = mx - ex
        dy = my - ey
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        angle = math.atan2(dy, dx)

        # Lagrange points relative to Earth-Moon axis
        # L1 between Earth and Moon (inside)
        l1_x = ex + (dist * 0.84) * math.cos(angle)
        l1_y = ey + (dist * 0.84) * math.sin(angle)
        # L2 beyond Moon
        l2_x = ex + (dist * 1.16) * math.cos(angle)
        l2_y = ey + (dist * 1.16) * math.sin(angle)
        # L3 opposite side
        l3_x = ex - (dist * 1.02) * math.cos(angle)
        l3_y = ey - (dist * 1.02) * math.sin(angle)
        # L4, L5 equilateral points (+- 60 degrees)
        l4_x = ex + dist * math.cos(angle + math.pi / 3.0)
        l4_y = ey + dist * math.sin(angle + math.pi / 3.0)
        l5_x = ex + dist * math.cos(angle - math.pi / 3.0)
        l5_y = ey + dist * math.sin(angle - math.pi / 3.0)

        # Equilateral triangle dashed lines connecting Earth, Moon, L4/L5
        painter.setPen(QPen(QColor("#f59e0b"), 0.9, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(ex, ey), QPointF(l4_x, l4_y))
        painter.drawLine(QPointF(mx, my), QPointF(l4_x, l4_y))
        painter.drawLine(QPointF(ex, ey), QPointF(l5_x, l5_y))
        painter.drawLine(QPointF(mx, my), QPointF(l5_x, l5_y))

        pts = [("L₁", l1_x, l1_y, "#f43f5e"),
               ("L₂", l2_x, l2_y, "#fb7185"),
               ("L₃", l3_x, l3_y, "#e879f9"),
               ("L₄", l4_x, l4_y, "#facc15"),
               ("L₅", l5_x, l5_y, "#facc15")]

        for name, px, py, col in pts:
            painter.setBrush(QBrush(QColor(col)))
            painter.setPen(QPen(QColor("#ffffff"), 0.8))
            painter.drawEllipse(QPointF(px, py), 3.5, 3.5)

            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor(col))
            painter.drawText(int(px - 6), int(py - 6), name)

    def _paint_hud(self, painter: QPainter) -> None:
        """Paints energy metrics and orbital mechanics equations card."""
        hud_rect = QRectF(self._hud_x, self._hud_y, self._hud_w, self._hud_h)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Header
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(self._hud_x, self._hud_y, self._hud_w, 28.0), 10.0, 10.0)
        painter.drawRect(QRectF(self._hud_x, self._hud_y + 16.0, self._hud_w, 12.0))

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(int(self._hud_x + 10), int(self._hud_y + 18), "BẢNG ĐO CƠ HỌC & NĂNG LƯỢNG")

        curr_y = self._hud_y + 45.0

        # Energy Gauges
        # Total Mechanical Energy E
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(int(self._hud_x + 10), int(curr_y), "Cơ Năng Toàn Phần (E = Ek + U):")
        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(int(self._hud_x + 10), int(curr_y + 14), f"{self._energy_total:,.1f} J")
        curr_y += 32.0

        # Kinetic Energy Ek
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#22c55e"))
        painter.drawText(int(self._hud_x + 10), int(curr_y), "Động Năng (Ek = ½ ∑ m·v²):")
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(int(self._hud_x + 10), int(curr_y + 13), f"{self._energy_k:,.1f} J")
        curr_y += 26.0

        # Gravitational Potential U
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(int(self._hud_x + 10), int(curr_y), "Thế Năng Hấp Dẫn (U = -∑ GMm/r):")
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(int(self._hud_x + 10), int(curr_y + 13), f"{self._energy_u:,.1f} J")
        curr_y += 26.0

        # Angular Momentum Lz
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(int(self._hud_x + 10), int(curr_y), "Mômen Động Lượng (Lz = const):")
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(int(self._hud_x + 10), int(curr_y + 13), f"{self._angular_momentum_z:,.1f} kg·m²/s")
        curr_y += 32.0

        # Horizontal separator
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(self._hud_x + 8), int(curr_y), int(self._hud_x + self._hud_w - 8), int(curr_y))
        curr_y += 12.0

        # Theory Equations
        formulas = [
            ("Lực Hấp Dẫn Newton:", "F = G · (m₁ · m₂) / r²"),
            ("Tích Phân Symplectic:", "Velocity Verlet (O(dt²))"),
            ("Định Luật Kepler III:", "T² / a³ = 4π² / (G·M)"),
            ("Điểm Cân Bằng Lagrange:", "∇Φ_eff(x, y) = 0"),
        ]

        for lbl, form in formulas:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(int(self._hud_x + 10), int(curr_y), lbl)
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(int(self._hud_x + 10), int(curr_y + 12), form)
            curr_y += 24.0

    def _paint_controls(self, painter: QPainter) -> None:
        """Paints lower control deck buttons."""
        # Row 1: Presets
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # [◀]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_prev_preset, 5.0, 5.0)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(self._rect_btn_prev_preset, Qt.AlignmentFlag.AlignCenter, "◀")

        # Preset Name Button
        preset_vi = self.PRESET_LABELS.get(self._preset, self._preset)
        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(self._rect_btn_preset, 5.0, 5.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_preset, Qt.AlignmentFlag.AlignCenter, f"🌌 {preset_vi}")

        # [▶]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_next_preset, 5.0, 5.0)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(self._rect_btn_next_preset, Qt.AlignmentFlag.AlignCenter, "▶")

        # Speed Controls: Pause & Multipliers
        # Pause
        pause_txt = "▶ Tiếp tục" if self._is_paused else "⏸ Tạm dừng"
        pause_col = QColor("#f59e0b") if self._is_paused else QColor("#1e293b")
        painter.setBrush(QBrush(pause_col))
        painter.setPen(QPen(QColor("#fbbf24"), 1.0))
        painter.drawRoundedRect(self._rect_btn_pause, 5.0, 5.0)
        painter.setPen(QColor("#0f172a" if self._is_paused else "#f8fafc"))
        painter.drawText(self._rect_btn_pause, Qt.AlignmentFlag.AlignCenter, pause_txt)

        # 0.5x, 1x, 2x, 5x
        speeds = [
            (self._rect_btn_speed_05, 0.5, "0.5×"),
            (self._rect_btn_speed_10, 1.0, "1×"),
            (self._rect_btn_speed_20, 2.0, "2×"),
            (self._rect_btn_speed_50, 5.0, "5×"),
        ]
        for s_rect, s_val, s_lbl in speeds:
            is_cur = abs(self._speed_multiplier - s_val) < 0.1
            bg_col = QColor("#2563eb") if is_cur else QColor("#1e293b")
            painter.setBrush(QBrush(bg_col))
            painter.setPen(QPen(QColor("#60a5fa" if is_cur else "#475569"), 1.0))
            painter.drawRoundedRect(s_rect, 5.0, 5.0)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, s_lbl)

        # Row 2: Toggles & Actions
        # Trails
        tr_bg = QColor("#0284c7") if self._show_trails else QColor("#1e293b")
        painter.setBrush(QBrush(tr_bg))
        painter.setPen(QPen(QColor("#38bdf8" if self._show_trails else "#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_trails, 5.0, 5.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_trails, Qt.AlignmentFlag.AlignCenter, "✨ Vết quỹ đạo")

        # Vectors
        vec_bg = QColor("#16a34a") if self._show_vectors else QColor("#1e293b")
        painter.setBrush(QBrush(vec_bg))
        painter.setPen(QPen(QColor("#4ade80" if self._show_vectors else "#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_vectors, 5.0, 5.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_vectors, Qt.AlignmentFlag.AlignCenter, "🏹 Vector v & F")

        # Lagrange
        lag_bg = QColor("#7c3aed") if self._show_lagrange else QColor("#1e293b")
        painter.setBrush(QBrush(lag_bg))
        painter.setPen(QPen(QColor("#c084fc" if self._show_lagrange else "#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_lagrange, 5.0, 5.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_lagrange, Qt.AlignmentFlag.AlignCenter, "🪐 Điểm L₁-L₅")

        # Reset
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(self._rect_btn_reset, 5.0, 5.0)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

        # Stamp Button
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#34d399"), 1.2))
        painter.drawRoundedRect(self._rect_btn_stamp, 5.0, 5.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Dập vào bài giảng")
