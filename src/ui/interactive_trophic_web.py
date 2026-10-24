"""
AirTeach AI - Interactive Biology Ecosystem Trophic Food Web & Lotka-Volterra Simulator
=======================================================================================
Provides an authentic biological and ecological simulation of trophic dynamics:
  - 4 Trophic Levels:
      1. Primary Producers (Thực vật / Cỏ)
      2. Primary Consumers (Động vật ăn cỏ / Thỏ)
      3. Secondary Consumers (Động vật ăn thịt nhỏ / Cáo)
      4. Tertiary / Apex Predators (Động vật ăn thịt đầu bảng / Đại bàng)
  - Directed Food Web network with draggable species nodes and animated energy particles.
  - Coupled Lotka-Volterra Predator-Prey Differential Equations:
      dx/dt = α * x - β * x * y   (Prey / Thỏ)
      dy/dt = δ * x * y - γ * y   (Predator / Cáo)
  - 4th-Order Runge-Kutta (RK4) numerical integration for smooth, energy-conserving orbits.
  - Conserved quantity (First Integral / Liapunov invariant):
      V(x, y) = δ * x - γ * ln(x) + β * y - α * ln(y) = const
  - Equilibrium fixed points:
      x* = γ / δ,   y* = α / β
  - Real-time dual charts:
      1. Time-domain population waves x(t) and y(t) with phase shift.
      2. Phase space portrait (x, y) showing closed orbital trajectories.
  - Ecological perturbations & shocks:
      Drought (Hạn hán: -50% prey), Epidemic (Dịch bệnh: -60% predators),
      Population boosters (+30 Thỏ, +20 Cáo), and Equilibrium reset.
  - Lindeman 10% ecological energy transfer pyramid HUD.
  - Touchless air pinch dragging and air tap controls.
  - Vector diagram stamping onto LessonBoard via BiologyTools.create_trophic_pyramid.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

from src.ui.stem_tools import BiologyTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class SpeciesNode:
    """Represents a species node in the interactive food web graph."""
    node_id: str
    name: str
    trophic_level: int  # 1: Producer, 2: Primary, 3: Secondary, 4: Apex
    color: QColor
    badge: str
    desc: str
    pos: QPointF
    radius: float = 22.0


class InteractiveTrophicWebItem(QGraphicsObject):
    """
    Interactive Biology Ecosystem Trophic Food Web & Lotka-Volterra Simulator.
    """

    population_changed = Signal(float, float)  # (prey_x, predator_y)
    parameters_changed = Signal(float, float)  # (alpha, beta)
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 720.0,
        height: float = 470.0,
        alpha: float = 0.80,
        beta: float = 0.02,
        gamma: float = 0.40,
        delta: float = 0.005,
        prey_init: float = 100.0,
        predator_init: float = 30.0,
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

        # Lotka-Volterra parameters
        self._alpha = max(0.1, min(2.5, float(alpha)))     # Prey birth rate
        self._beta = max(0.005, min(0.1, float(beta)))     # Predation coefficient
        self._gamma = max(0.1, min(1.5, float(gamma)))     # Predator mortality rate
        self._delta = max(0.001, min(0.02, float(delta)))  # Reproduction conversion

        # Current population state
        self._x = max(1.0, float(prey_init))       # Prey population (Thỏ)
        self._y = max(1.0, float(predator_init))   # Predator population (Cáo)
        self._time_elapsed = 0.0

        # Shock alert banner message & timer
        self._shock_alert: Optional[str] = None
        self._shock_alert_countdown: int = 0

        # History buffer for wave charting: list of (time, x, y)
        self._history: list[tuple[float, float, float]] = [(0.0, self._x, self._y)]
        self._max_history = 200

        # Food Web Graph Nodes (local coordinates in left pane)
        self._nodes: dict[str, SpeciesNode] = {
            "producer": SpeciesNode(
                node_id="producer",
                name="Cỏ / Thực vật",
                trophic_level=1,
                color=QColor("#10b981"),
                badge="🌱",
                desc="10,000 kcal (SX)",
                pos=QPointF(80.0, 260.0),
            ),
            "herbivore": SpeciesNode(
                node_id="herbivore",
                name="Thỏ rừng",
                trophic_level=2,
                color=QColor("#06b6d4"),
                badge="🐇",
                desc="x(t) Con mồi",
                pos=QPointF(210.0, 200.0),
            ),
            "carnivore": SpeciesNode(
                node_id="carnivore",
                name="Cáo lửa",
                trophic_level=3,
                color=QColor("#f59e0b"),
                badge="🦊",
                desc="y(t) Săn mồi",
                pos=QPointF(220.0, 100.0),
            ),
            "apex": SpeciesNode(
                node_id="apex",
                name="Đại bàng",
                trophic_level=4,
                color=QColor("#ef4444"),
                badge="🦅",
                desc="Ăn thịt đầu bảng",
                pos=QPointF(80.0, 80.0),
            ),
        }

        # Directed Trophic Connections: (source_id, target_id)
        self._edges: list[tuple[str, str]] = [
            ("producer", "herbivore"),
            ("herbivore", "carnivore"),
            ("carnivore", "apex"),
            ("herbivore", "apex"),
        ]

        # Animated energy flow particles along edges: list of float phases [0.0, 1.0]
        self._energy_phases: list[float] = [0.1, 0.4, 0.7, 0.25, 0.85]

        # Pyramidal Energy view toggle
        self._show_pyramid = False

        # Simulation Loop Timer (30 FPS)
        self._is_running = True
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Interaction & Dragging
        self._active_drag_node: Optional[str] = None
        self._drag_offset = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Interactive Button Rectangles (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_run = QRectF(25.0, 355.0, 95.0, 28.0)
        self._rect_btn_add_prey = QRectF(128.0, 355.0, 95.0, 28.0)
        self._rect_btn_add_pred = QRectF(231.0, 355.0, 95.0, 28.0)
        self._rect_btn_drought = QRectF(334.0, 355.0, 95.0, 28.0)
        self._rect_btn_disease = QRectF(437.0, 355.0, 95.0, 28.0)
        self._rect_btn_pyramid = QRectF(540.0, 355.0, 75.0, 28.0)
        self._rect_btn_reset = QRectF(623.0, 355.0, 72.0, 28.0)

        # Parameter Adjust Steppers (row 2)
        self._rect_btn_alpha_dec = QRectF(25.0, 395.0, 24.0, 24.0)
        self._rect_btn_alpha_inc = QRectF(105.0, 395.0, 24.0, 24.0)
        self._rect_btn_beta_dec = QRectF(160.0, 395.0, 24.0, 24.0)
        self._rect_btn_beta_inc = QRectF(245.0, 395.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(580.0, 393.0, 115.0, 28.0)

    # ─── Public Properties & Accessors ───────────────────────────────────────

    @property
    def prey_population(self) -> float:
        return self._x

    @property
    def predator_population(self) -> float:
        return self._y

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def equilibrium_fixed_point(self) -> tuple[float, float]:
        """Calculates stable fixed point (x*, y*) = (gamma / delta, alpha / beta)."""
        x_star = self._gamma / self._delta if self._delta > 0 else 0.0
        y_star = self._alpha / self._beta if self._beta > 0 else 0.0
        return x_star, y_star

    @property
    def conserved_quantity(self) -> float:
        """Calculates first integral / Liapunov invariant V(x, y)."""
        x_safe = max(self._x, 0.001)
        y_safe = max(self._y, 0.001)
        return (
            self._delta * x_safe
            - self._gamma * math.log(x_safe)
            + self._beta * y_safe
            - self._alpha * math.log(y_safe)
        )

    # ─── Lotka-Volterra ODE Numerical Solver (RK4) ───────────────────────────

    def _derivatives(self, x: float, y: float) -> tuple[float, float]:
        """Computes dx/dt and dy/dt with non-negative species boundaries."""
        x_safe = max(x, 0.01)
        y_safe = max(y, 0.01)
        dxdt = self._alpha * x_safe - self._beta * x_safe * y_safe
        dydt = self._delta * x_safe * y_safe - self._gamma * y_safe
        return dxdt, dydt

    def step_simulation(self, dt: float = 0.04) -> None:
        """
        Executes a 4th-Order Runge-Kutta numerical integration step,
        ensuring smooth orbits and exact phase trajectory evolution.
        """
        x, y = self._x, self._y

        # RK4 intermediate slopes
        k1x, k1y = self._derivatives(x, y)
        k2x, k2y = self._derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = self._derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = self._derivatives(x + dt * k3x, y + dt * k3y)

        # 4th-order weighted average
        new_x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        new_y = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

        # Clamping to biological non-extinction margins
        self._x = max(1.0, min(new_x, 600.0))
        self._y = max(1.0, min(new_y, 400.0))
        self._time_elapsed += dt

        # Buffer history for graphing
        self._history.append((self._time_elapsed, self._x, self._y))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Update energy flow particle animation phases
        for i in range(len(self._energy_phases)):
            self._energy_phases[i] = (self._energy_phases[i] + 0.035) % 1.0

        # Handle shock alert countdown
        if self._shock_alert_countdown > 0:
            self._shock_alert_countdown -= 1
            if self._shock_alert_countdown == 0:
                self._shock_alert = None

        self.population_changed.emit(self._x, self._y)
        self.update()

    def _on_sim_tick(self) -> None:
        if self._is_running:
            self.step_simulation(0.04)

    # ─── Ecological Perturbations & Controls ──────────────────────────────────

    def toggle_running(self) -> None:
        self._is_running = not self._is_running
        self.update()

    def add_prey(self, amount: float = 30.0) -> None:
        """Injects additional prey biomass into ecosystem."""
        self._x = min(self._x + amount, 500.0)
        self._shock_alert = f"+{int(amount)} Thỏ di cư vào"
        self._shock_alert_countdown = 30
        self.population_changed.emit(self._x, self._y)
        self.update()

    def add_predators(self, amount: float = 20.0) -> None:
        """Injects additional predators into ecosystem."""
        self._y = min(self._y + amount, 350.0)
        self._shock_alert = f"+{int(amount)} Cáo di cư vào"
        self._shock_alert_countdown = 30
        self.population_changed.emit(self._x, self._y)
        self.update()

    def trigger_drought(self) -> None:
        """Simulates severe drought: -50% primary consumers due to vegetation loss."""
        self._x = max(5.0, self._x * 0.5)
        self._shock_alert = "⚡ HẠN HÁN: Con mồi giảm 50%"
        self._shock_alert_countdown = 40
        self.population_changed.emit(self._x, self._y)
        self.update()

    def trigger_disease(self) -> None:
        """Simulates predator epidemic: -60% predator population collapse."""
        self._y = max(2.0, self._y * 0.4)
        self._shock_alert = "🎯 DỊCH BỆNH: Kẻ săn mồi giảm 60%"
        self._shock_alert_countdown = 40
        self.population_changed.emit(self._x, self._y)
        self.update()

    def reset_simulation(self) -> None:
        """Restores baseline ecological populations and history."""
        self._x = 100.0
        self._y = 30.0
        self._time_elapsed = 0.0
        self._history = [(0.0, self._x, self._y)]
        self._shock_alert = "Đã đặt lại hệ sinh thái"
        self._shock_alert_countdown = 25
        self.population_changed.emit(self._x, self._y)
        self.update()

    def set_alpha(self, alpha: float) -> None:
        self._alpha = max(0.1, min(2.5, float(alpha)))
        self.parameters_changed.emit(self._alpha, self._beta)
        self.update()

    def set_beta(self, beta: float) -> None:
        self._beta = max(0.005, min(0.1, float(beta)))
        self.parameters_changed.emit(self._alpha, self._beta)
        self.update()

    def toggle_pyramid_view(self) -> None:
        self._show_pyramid = not self._show_pyramid
        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps the 4-tier Lindeman ecological energy pyramid vector diagram
        onto the active LessonBoard canvas. Returns the list of created items.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        pyramid_items = BiologyTools.create_trophic_pyramid(
            center=(cx, cy),
            base_width=380.0,
            total_height=260.0,
            efficiency=0.10,
            producer_energy=10000.0,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None:
                if hasattr(scene, "add_item_with_undo"):
                    target_board = scene
                else:
                    target_board = getattr(scene, "board", None)

        if target_board and hasattr(target_board, "add_item_with_undo"):
            for item in pyramid_items:
                target_board.add_item_with_undo(item)
        elif self.scene():
            for item in pyramid_items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        return pyramid_items

    # ─── Touchless Air Gestures ──────────────────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF, is_pressed: bool) -> bool:
        """
        Handles touchless air pinch gesture for dragging species nodes
        within the food web graph.
        """
        local_pos = self.mapFromScene(scene_pos)

        if is_pressed:
            # Check species node drag
            for node_id, node in self._nodes.items():
                dx = local_pos.x() - node.pos.x()
                dy = local_pos.y() - node.pos.y()
                if math.hypot(dx, dy) <= node.radius + 8.0:
                    self._active_drag_node = node_id
                    self._drag_offset = QPointF(dx, dy)
                    return True
        else:
            if self._active_drag_node:
                self._active_drag_node = None
                return True

        if self._active_drag_node and is_pressed:
            target_node = self._nodes[self._active_drag_node]
            # Constrain to left food web pane
            new_x = max(45.0, min(local_pos.x() - self._drag_offset.x(), 305.0))
            new_y = max(65.0, min(local_pos.y() - self._drag_offset.y(), 310.0))
            target_node.pos = QPointF(new_x, new_y)
            self.update()
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """
        Handles touchless air tap gesture for clicking control buttons.
        """
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_run.contains(local_pos):
            self.toggle_running()
            return True
        elif self._rect_btn_add_prey.contains(local_pos):
            self.add_prey(30.0)
            return True
        elif self._rect_btn_add_pred.contains(local_pos):
            self.add_predators(20.0)
            return True
        elif self._rect_btn_drought.contains(local_pos):
            self.trigger_drought()
            return True
        elif self._rect_btn_disease.contains(local_pos):
            self.trigger_disease()
            return True
        elif self._rect_btn_pyramid.contains(local_pos):
            self.toggle_pyramid_view()
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True
        elif self._rect_btn_alpha_dec.contains(local_pos):
            self.set_alpha(self._alpha - 0.05)
            return True
        elif self._rect_btn_alpha_inc.contains(local_pos):
            self.set_alpha(self._alpha + 0.05)
            return True
        elif self._rect_btn_beta_dec.contains(local_pos):
            self.set_beta(self._beta - 0.002)
            return True
        elif self._rect_btn_beta_inc.contains(local_pos):
            self.set_beta(self._beta + 0.002)
            return True

        return False

    def close(self) -> None:
        """Closes the interactive simulator and removes it from scene."""
        self._timer.stop()
        scene = self.scene()
        if scene:
            scene.removeItem(self)

    # ─── Mouse Event Handlers ────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        local_pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(local_pos):
            self.close()
            event.accept()
            return
        elif self._rect_btn_run.contains(local_pos):
            self.toggle_running()
            event.accept()
            return
        elif self._rect_btn_add_prey.contains(local_pos):
            self.add_prey(30.0)
            event.accept()
            return
        elif self._rect_btn_add_pred.contains(local_pos):
            self.add_predators(20.0)
            event.accept()
            return
        elif self._rect_btn_drought.contains(local_pos):
            self.trigger_drought()
            event.accept()
            return
        elif self._rect_btn_disease.contains(local_pos):
            self.trigger_disease()
            event.accept()
            return
        elif self._rect_btn_pyramid.contains(local_pos):
            self.toggle_pyramid_view()
            event.accept()
            return
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            event.accept()
            return
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas()
            event.accept()
            return
        elif self._rect_btn_alpha_dec.contains(local_pos):
            self.set_alpha(self._alpha - 0.05)
            event.accept()
            return
        elif self._rect_btn_alpha_inc.contains(local_pos):
            self.set_alpha(self._alpha + 0.05)
            event.accept()
            return
        elif self._rect_btn_beta_dec.contains(local_pos):
            self.set_beta(self._beta - 0.002)
            event.accept()
            return
        elif self._rect_btn_beta_inc.contains(local_pos):
            self.set_beta(self._beta + 0.002)
            event.accept()
            return

        # Check species node drag
        for node_id, node in self._nodes.items():
            dx = local_pos.x() - node.pos.x()
            dy = local_pos.y() - node.pos.y()
            if math.hypot(dx, dy) <= node.radius + 6.0:
                self._active_drag_node = node_id
                self._drag_offset = QPointF(dx, dy)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._active_drag_node:
            target_node = self._nodes[self._active_drag_node]
            local_pos = event.pos()
            new_x = max(45.0, min(local_pos.x() - self._drag_offset.x(), 305.0))
            new_y = max(65.0, min(local_pos.y() - self._drag_offset.y(), 310.0))
            target_node.pos = QPointF(new_x, new_y)
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._active_drag_node:
            self._active_drag_node = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    # ─── Bounding Rect & Painting ────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Frosted Card
        bg_path = QPainterPath()
        bg_path.addRoundedRect(QRectF(0.0, 0.0, self._width, self._height), 14.0, 14.0)
        painter.fillPath(bg_path, QBrush(QColor(15, 23, 42, 235)))  # Slate 900
        painter.strokePath(bg_path, QPen(QColor(51, 65, 85, 200), 1.5))

        # 2. Header Bar
        title_font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(22, 28, "🌿 HỆ SINH THÁI & LƯỚI THỨC ĂN (LOTKA-VOLTERRA)")

        # Status & Shock Alert
        status_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(status_font)
        if self._shock_alert:
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(390, 28, self._shock_alert)
        elif self._is_running:
            painter.setPen(QColor("#10b981"))
            painter.drawText(430, 28, "● Đang mô phỏng")
        else:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(430, 28, "⏸ Tạm dừng")

        # Close button [✕]
        painter.setPen(QPen(QColor("#64748b"), 1.5))
        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # 3. Left Panel: Food Web Graph or Energy Pyramid View
        left_rect = QRectF(20.0, 48.0, 310.0, 280.0)
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(left_rect, 8.0, 8.0)

        if self._show_pyramid:
            self._paint_energy_pyramid_view(painter, left_rect)
        else:
            self._paint_food_web_graph(painter, left_rect)

        # 4. Right Panel: Dynamic Wave Plot and Phase Portrait
        right_rect = QRectF(345.0, 48.0, 350.0, 280.0)
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(right_rect, 8.0, 8.0)

        self._paint_population_charts(painter, right_rect)

        # 5. Bottom Control Deck & Buttons
        self._paint_control_deck(painter)

        painter.restore()

    def _paint_food_web_graph(self, painter: QPainter, rect: QRectF) -> None:
        """Paints the directed food web graph with glowing nodes and moving energy particles."""
        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(rect.x() + 10.0, rect.y() + 20.0, "🕸️ Lưới thức ăn & Dòng năng lượng")

        # Draw Directed Trophic Edges
        edge_pen = QPen(QColor(100, 116, 139, 140), 2.0)
        painter.setPen(edge_pen)

        for idx, (src_id, dst_id) in enumerate(self._edges):
            src_node = self._nodes.get(src_id)
            dst_node = self._nodes.get(dst_id)
            if not src_node or not dst_node:
                continue

            p1 = src_node.pos
            p2 = dst_node.pos

            # Draw curved edge path
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            dist = math.hypot(dx, dy)
            if dist < 1.0:
                continue

            # Slight perpendicular curvature
            ctrl = QPointF(
                (p1.x() + p2.x()) / 2.0 - dy * 0.12,
                (p1.y() + p2.y()) / 2.0 + dx * 0.12,
            )

            path = QPainterPath()
            path.moveTo(p1)
            path.quadTo(ctrl, p2)
            painter.setPen(edge_pen)
            painter.drawPath(path)

            # Draw animated energy particle along bezier curve
            phase = (self._energy_phases[idx % len(self._energy_phases)])
            t = phase
            omt = 1.0 - t
            bx = omt * omt * p1.x() + 2.0 * omt * t * ctrl.x() + t * t * p2.x()
            by = omt * omt * p1.y() + 2.0 * omt * t * ctrl.y() + t * t * p2.y()

            particle_grad = QRadialGradient(bx, by, 5.0)
            particle_grad.setColorAt(0.0, QColor("#fbbf24"))
            particle_grad.setColorAt(1.0, QColor(251, 191, 36, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(particle_grad))
            painter.drawEllipse(QPointF(bx, by), 5.0, 5.0)

        # Draw Species Nodes
        node_title_font = QFont("Segoe UI", 8, QFont.Bold)
        desc_font = QFont("Segoe UI", 7)

        for node_id, node in self._nodes.items():
            pos = node.pos
            r = node.radius

            # Radial glow
            grad = QRadialGradient(pos.x(), pos.y(), r + 8.0)
            grad.setColorAt(0.0, node.color)
            grad.setColorAt(0.7, QColor(node.color.red(), node.color.green(), node.color.blue(), 180))
            grad.setColorAt(1.0, QColor(node.color.red(), node.color.green(), node.color.blue(), 0))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(pos, r + 6.0, r + 6.0)

            # Node body
            painter.setPen(QPen(QColor("#ffffff"), 1.8))
            painter.setBrush(QBrush(node.color))
            painter.drawEllipse(pos, r, r)

            # Badge Icon
            painter.setFont(QFont("Segoe UI Emoji", 11))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r),
                Qt.AlignmentFlag.AlignCenter,
                node.badge,
            )

            # Label below node
            painter.setFont(node_title_font)
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(pos.x() - 40.0, pos.y() + r + 13.0, node.name)

            painter.setFont(desc_font)
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(pos.x() - 40.0, pos.y() + r + 24.0, node.desc)

    def _paint_energy_pyramid_view(self, painter: QPainter, rect: QRectF) -> None:
        """Paints mini 4-tier Lindeman ecological energy pyramid overlay."""
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(rect.x() + 10.0, rect.y() + 20.0, "📐 Kim tự tháp năng lượng (10%)")

        tiers = [
            ("Bậc 4: Đại bàng", "10 kcal (0.1%)", "#fce8e6", "#ef4444", 60.0),
            ("Bậc 3: Cáo lửa", "100 kcal (1.0%)", "#fef7e0", "#f59e0b", 120.0),
            ("Bậc 2: Thỏ rừng", "1,000 kcal (10%)", "#e0f2f1", "#06b6d4", 180.0),
            ("Bậc 1: Thực vật", "10,000 kcal (100%)", "#e6f4ea", "#10b981", 240.0),
        ]

        start_y = rect.y() + 45.0
        h_tier = 50.0
        cx = rect.x() + rect.width() / 2.0

        for i, (name, kcal, fill, border, w) in enumerate(tiers):
            ty = start_y + i * h_tier
            t_rect = QRectF(cx - w / 2.0, ty, w, h_tier - 6.0)

            painter.setPen(QPen(QColor(border), 1.5))
            painter.setBrush(QBrush(QColor(fill)))
            painter.drawRoundedRect(t_rect, 6.0, 6.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(border))
            painter.drawText(t_rect, Qt.AlignmentFlag.AlignCenter, f"{name}: {kcal}")

    def _paint_population_charts(self, painter: QPainter, rect: QRectF) -> None:
        """Paints real-time wave graph and phase portrait orbit."""
        # Upper: Wave Graph (x(t) in cyan vs y(t) in amber)
        wave_rect = QRectF(rect.x() + 12.0, rect.y() + 10.0, rect.width() - 24.0, 140.0)
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(wave_rect, 6.0, 6.0)

        # Wave Graph Title & Legend
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#06b6d4"))
        painter.drawText(wave_rect.x() + 8.0, wave_rect.y() + 16.0, f"― Thỏ x(t): {self._x:.1f}")
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(wave_rect.x() + 130.0, wave_rect.y() + 16.0, f"― Cáo y(t): {self._y:.1f}")

        # Horizontal equilibrium guide lines
        x_star, y_star = self.equilibrium_fixed_point
        max_pop = max(180.0, max(self._x, self._y) * 1.25)

        def map_y(val: float) -> float:
            ratio = max(0.0, min(val / max_pop, 1.0))
            return wave_rect.bottom() - 10.0 - ratio * (wave_rect.height() - 32.0)

        # Plot curves
        if len(self._history) >= 2:
            path_x = QPainterPath()
            path_y = QPainterPath()
            num_pts = len(self._history)
            dx = (wave_rect.width() - 20.0) / float(self._max_history)

            first_t, first_x, first_y = self._history[0]
            x_screen_0 = wave_rect.x() + 10.0
            path_x.moveTo(x_screen_0, map_y(first_x))
            path_y.moveTo(x_screen_0, map_y(first_y))

            for i in range(1, num_pts):
                t_val, xv, yv = self._history[i]
                sx = wave_rect.x() + 10.0 + i * dx
                path_x.lineTo(sx, map_y(xv))
                path_y.lineTo(sx, map_y(yv))

            painter.setPen(QPen(QColor("#06b6d4"), 2.0))
            painter.drawPath(path_x)

            painter.setPen(QPen(QColor("#f59e0b"), 2.0))
            painter.drawPath(path_y)

        # Lower: Phase Portrait & Metric HUD Card
        phase_rect = QRectF(rect.x() + 12.0, rect.y() + 162.0, 130.0, 105.0)
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(phase_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(phase_rect.x() + 6.0, phase_rect.y() + 14.0, "Không gian pha (x, y)")

        # Phase trajectory loop
        if len(self._history) >= 2:
            phase_path = QPainterPath()
            for i, (_, xv, yv) in enumerate(self._history[-120:]):
                px = phase_rect.x() + 10.0 + (xv / 250.0) * (phase_rect.width() - 20.0)
                py = phase_rect.bottom() - 10.0 - (yv / 180.0) * (phase_rect.height() - 25.0)
                if i == 0:
                    phase_path.moveTo(px, py)
                else:
                    phase_path.lineTo(px, py)

            painter.setPen(QPen(QColor("#a855f7"), 1.8))
            painter.drawPath(phase_path)

            # Fixed point marker
            fx = phase_rect.x() + 10.0 + (x_star / 250.0) * (phase_rect.width() - 20.0)
            fy = phase_rect.bottom() - 10.0 - (y_star / 180.0) * (phase_rect.height() - 25.0)
            painter.setPen(QPen(QColor("#ef4444"), 1.5))
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.drawEllipse(QPointF(fx, fy), 2.5, 2.5)

        # Metric Readout Card
        metric_rect = QRectF(rect.x() + 152.0, rect.y() + 162.0, rect.width() - 164.0, 105.0)
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(metric_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(metric_rect.x() + 8.0, metric_rect.y() + 18.0, "📊 Chỉ số động học quần thể:")

        v_inv = self.conserved_quantity
        metrics_lines = [
            f"• Điểm cân bằng: x*={x_star:.1f}, y*={y_star:.1f}",
            f"• Tích phân bảo toàn V: {v_inv:.2f}",
            f"• Tốc độ sinh α: {self._alpha:.2f} | Săn mồi β: {self._beta:.3f}",
            f"• Tử vong γ: {self._gamma:.2f} | Chuyển hóa δ: {self._delta:.3f}",
        ]
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        for idx, line in enumerate(metrics_lines):
            painter.drawText(metric_rect.x() + 8.0, metric_rect.y() + 36.0 + idx * 16.0, line)

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive action buttons and parameter step controls."""
        btn_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(btn_font)

        def draw_button(rect: QRectF, text: str, bg_hex: str, text_hex: str = "#ffffff") -> None:
            painter.setPen(QPen(QColor(bg_hex).lighter(120), 1.0))
            painter.setBrush(QBrush(QColor(bg_hex)))
            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(QColor(text_hex))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Row 1: Action triggers
        run_txt = "⏸ Dừng" if self._is_running else "▶ Chạy"
        run_col = "#0284c7" if self._is_running else "#16a34a"
        draw_button(self._rect_btn_run, run_txt, run_col)

        draw_button(self._rect_btn_add_prey, "🌾 +30 Thỏ", "#0891b2")
        draw_button(self._rect_btn_add_pred, "🦊 +20 Cáo", "#d97706")
        draw_button(self._rect_btn_drought, "⚡ Hạn hán", "#b45309")
        draw_button(self._rect_btn_disease, "🎯 Dịch bệnh", "#be123c")

        pyr_txt = "📐 Đồ thị" if self._show_pyramid else "📐 Kim tự tháp"
        draw_button(self._rect_btn_pyramid, pyr_txt, "#4f46e5")
        draw_button(self._rect_btn_reset, "🔄 Đặt lại", "#475569")

        # Row 2: Parameter Steppers and Stamping
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(55.0, 412.0, f"α={self._alpha:.2f}")
        draw_button(self._rect_btn_alpha_dec, "−", "#334155")
        draw_button(self._rect_btn_alpha_inc, "+", "#334155")

        painter.drawText(190.0, 412.0, f"β={self._beta:.3f}")
        draw_button(self._rect_btn_beta_dec, "−", "#334155")
        draw_button(self._rect_btn_beta_inc, "+", "#334155")

        draw_button(self._rect_btn_stamp, "🖊️ Đóng dấu bảng", "#059669")
