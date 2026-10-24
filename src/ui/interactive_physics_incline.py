"""
AirTeach AI - Interactive Physics Vector Force Decomposer & Inclined Plane Suite
===============================================================================
Provides an interactive physics mechanics demonstration tool for inclined planes:
  - Dynamic slope angle adjustment: alpha in [5°, 75°].
  - Gravity force vector P = m*g decomposed into:
      P_parallel = P * sin(alpha)   (down the slope)
      P_perp = P * cos(alpha)       (into the surface)
  - Normal reaction force N = P_perp.
  - Friction force F_ms = mu * N opposing motion.
  - Net force F_net = P_parallel - F_ms and acceleration a = g*(sin(alpha) - mu*cos(alpha)).
  - Motion condition test: Slips if tan(alpha) > mu, balances if tan(alpha) <= mu.
  - Draggable slope vertex with mouse and touchless air gestures.
  - Stamping permanent vector diagrams onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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


class InteractiveInclinedPlaneItem(QGraphicsObject):
    """
    Interactive physics inclined plane with real-time vector force decomposition.
    """

    angle_changed = Signal(float)          # (alpha_degrees)
    physics_updated = Signal(float, float) # (f_net, acceleration)

    def __init__(
        self,
        base_pos: tuple[float, float] = (700.0, 720.0),
        base_length: float = 420.0,
        alpha_deg: float = 30.0,
        mass_kg: float = 2.0,
        mu_friction: float = 0.25,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._base_x = float(base_pos[0])
        self._base_y = float(base_pos[1])
        self._base_len = max(200.0, float(base_length))
        self._alpha = max(5.0, min(75.0, float(alpha_deg)))
        self._mass = max(0.1, float(mass_kg))
        self._mu = max(0.0, min(1.0, float(mu_friction)))
        self._g = 9.8  # m/s^2

        self._show_components: bool = True
        self._show_friction: bool = True
        self._show_net_force: bool = True
        self._show_hud_card: bool = True

        self._is_dragging_incline: bool = False
        self._handle_radius: float = 14.0

        self.setZValue(14.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    # ── Parameters & Properties ────────────────────────────────────────────────

    def alpha(self) -> float:
        return self._alpha

    def set_alpha(self, deg: float) -> None:
        """Set slope incline angle clamped to [5.0°, 75.0°]."""
        self.prepareGeometryChange()
        self._alpha = max(5.0, min(75.0, float(deg)))
        self.update()
        self.angle_changed.emit(self._alpha)
        f_net, a = self.net_force(), self.acceleration()
        self.physics_updated.emit(f_net, a)

    def mass(self) -> float:
        return self._mass

    def set_mass(self, m: float) -> None:
        self.prepareGeometryChange()
        self._mass = max(0.1, float(m))
        self.update()
        self.physics_updated.emit(self.net_force(), self.acceleration())

    def friction_coeff(self) -> float:
        return self._mu

    def set_friction_coeff(self, mu: float) -> None:
        self.prepareGeometryChange()
        self._mu = max(0.0, min(1.0, float(mu)))
        self.update()
        self.physics_updated.emit(self.net_force(), self.acceleration())

    # ── Physics Calculations ───────────────────────────────────────────────────

    def gravity_force(self) -> float:
        """P = m * g (Newtons)."""
        return self._mass * self._g

    def parallel_force(self) -> float:
        """P_parallel = P * sin(alpha)."""
        rad = math.radians(self._alpha)
        return self.gravity_force() * math.sin(rad)

    def perpendicular_force(self) -> float:
        """P_perp = P * cos(alpha)."""
        rad = math.radians(self._alpha)
        return self.gravity_force() * math.cos(rad)

    def normal_force(self) -> float:
        """N = P_perp = P * cos(alpha)."""
        return self.perpendicular_force()

    def friction_force(self) -> float:
        """F_ms = mu * N."""
        return self._mu * self.normal_force()

    def is_sliding(self) -> bool:
        """Returns True if block overcomes friction and slides (tan(alpha) > mu)."""
        rad = math.radians(self._alpha)
        return math.tan(rad) > self._mu

    def net_force(self) -> float:
        """F_net along slope (0 if static equilibrium)."""
        if not self.is_sliding():
            return 0.0
        return self.parallel_force() - self.friction_force()

    def acceleration(self) -> float:
        """a = F_net / m (m/s^2)."""
        return self.net_force() / self._mass

    # ── Geometry Points ────────────────────────────────────────────────────────

    def wedge_vertices(self) -> tuple[QPointF, QPointF, QPointF]:
        """
        Returns (W1_bottom_left, W2_bottom_right, W3_top_left).
        Slope runs from W3 down to W2.
        """
        rad = math.radians(self._alpha)
        h = self._base_len * math.tan(rad)
        w1 = QPointF(self._base_x, self._base_y)
        w2 = QPointF(self._base_x + self._base_len, self._base_y)
        w3 = QPointF(self._base_x, self._base_y - h)
        return (w1, w2, w3)

    def block_center(self) -> QPointF:
        """Center of mass on the slope hypotenuse."""
        _, w2, w3 = self.wedge_vertices()
        mid_x = (w3.x() + w2.x()) / 2.0
        mid_y = (w3.y() + w2.y()) / 2.0
        # Offset outward perpendicular to slope by half block height (18px)
        rad = math.radians(self._alpha)
        # Normal direction outward is (-sin(alpha), -cos(alpha))
        nx = -math.sin(rad) * 18.0
        ny = -math.cos(rad) * 18.0
        return QPointF(mid_x + nx, mid_y + ny)

    # ── Touchless Air Gestures ─────────────────────────────────────────────────

    def air_drag_slope(self, canvas_y: float) -> bool:
        """Touchless air gesture: adjust slope height by dragging top handle."""
        dy = self._base_y - canvas_y
        if dy > 10.0:
            tan_a = dy / self._base_len
            deg = math.degrees(math.atan(tan_a))
            self.set_alpha(deg)
            return True
        return False

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """Touchless air tap near top handle to cycle angle (+5°)."""
        _, _, w3 = self.wedge_vertices()
        if math.hypot(canvas_x - w3.x(), canvas_y - w3.y()) <= 24.0:
            new_a = self._alpha + 5.0
            if new_a > 60.0:
                new_a = 15.0
            self.set_alpha(new_a)
            return True
        return False

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        w1, w2, w3 = self.wedge_vertices()
        margin = 180.0
        xs = [w1.x(), w2.x(), w3.x()]
        ys = [w1.y(), w2.y(), w3.y()]
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def _draw_vector_arrow(
        self,
        painter: QPainter,
        start: QPointF,
        vec: QPointF,
        color: QColor,
        label: str,
        dashed: bool = False,
    ) -> None:
        """Helper to draw physics styled vector arrow with arrowhead."""
        end = QPointF(start.x() + vec.x(), start.y() + vec.y())
        pen_style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        painter.setPen(QPen(color, 2.4, pen_style, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)

        # Arrowhead
        ang = math.atan2(vec.y(), vec.x())
        head_len = 10.0
        left_head = QPointF(
            end.x() - head_len * math.cos(ang - math.pi / 6.0),
            end.y() - head_len * math.sin(ang - math.pi / 6.0),
        )
        right_head = QPointF(
            end.x() - head_len * math.cos(ang + math.pi / 6.0),
            end.y() - head_len * math.sin(ang + math.pi / 6.0),
        )
        painter.setPen(QPen(color, 2.4))
        painter.drawLine(end, left_head)
        painter.drawLine(end, right_head)

        # Label
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(color))
        lx = end.x() + 8.0 * math.cos(ang)
        ly = end.y() + 8.0 * math.sin(ang)
        painter.drawText(int(lx - 10.0), int(ly + 5.0), label)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w1, w2, w3 = self.wedge_vertices()

        # 1. Draw Wedge Body
        wedge_path = QPainterPath()
        wedge_path.moveTo(w1)
        wedge_path.lineTo(w2)
        wedge_path.lineTo(w3)
        wedge_path.closeSubpath()

        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.setPen(QPen(QColor("#64748b"), 2.2))
        painter.drawPath(wedge_path)

        # Angle arc at bottom-right W2
        arc_r = 50.0
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.setBrush(QBrush(QColor(56, 189, 248, 30)))
        # Arc from horizontal (180°) up to slope (180° - alpha)
        arc_rect = QRectF(w2.x() - arc_r, w2.y() - arc_r, 2 * arc_r, 2 * arc_r)
        painter.drawPie(arc_rect, 180 * 16, int(-self._alpha * 16))

        # Angle label alpha
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(int(w2.x() - arc_r - 28.0), int(w2.y() - 10.0), f"α={self._alpha:.0f}°")

        # 2. Draw Block on Slope
        bc = self.block_center()
        rad = math.radians(self._alpha)
        bw, bh = 56.0, 32.0

        painter.save()
        painter.translate(bc.x(), bc.y())
        painter.rotate(-self._alpha)  # Slope orientation
        block_rect = QRectF(-bw / 2.0, -bh / 2.0, bw, bh)
        painter.setBrush(QBrush(QColor(51, 65, 85)))
        painter.setPen(QPen(QColor("#94a3b8"), 2.0))
        painter.drawRoundedRect(block_rect, 4.0, 4.0)
        # Block label "m"
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(block_rect, Qt.AlignmentFlag.AlignCenter, f"{self._mass:.1f}kg")
        painter.restore()

        # 3. Vector Force Decomposition
        scale_px = 5.0  # pixels per Newton
        p_val = self.gravity_force()
        p_par_val = self.parallel_force()
        p_perp_val = self.perpendicular_force()
        n_val = self.normal_force()
        f_ms_val = self.friction_force()
        f_net_val = self.net_force()

        # Gravity P: vertically down (0, +P*scale)
        vec_p = QPointF(0.0, p_val * scale_px)
        self._draw_vector_arrow(painter, bc, vec_p, QColor("#ef4444"), "P⃗")

        # Unit vectors along and normal to slope:
        # Down-slope vector: (cos(alpha), sin(alpha))
        u_down = QPointF(math.cos(rad), math.sin(rad))
        # Normal outward vector: (-sin(alpha), -cos(alpha))
        u_norm_out = QPointF(-math.sin(rad), -math.cos(rad))
        # Normal inward vector: (sin(alpha), cos(alpha))
        u_norm_in = QPointF(math.sin(rad), math.cos(rad))

        # Normal Reaction Force N: outward from surface
        vec_n = QPointF(u_norm_out.x() * n_val * scale_px, u_norm_out.y() * n_val * scale_px)
        self._draw_vector_arrow(painter, bc, vec_n, QColor("#10b981"), "N⃗")

        # Parallel component P_parallel: down the slope
        if self._show_components:
            vec_ppar = QPointF(u_down.x() * p_par_val * scale_px, u_down.y() * p_par_val * scale_px)
            self._draw_vector_arrow(painter, bc, vec_ppar, QColor("#f59e0b"), "P⃗∥", dashed=True)

            # Perpendicular component P_perp: into the surface
            vec_pperp = QPointF(u_norm_in.x() * p_perp_val * scale_px, u_norm_in.y() * p_perp_val * scale_px)
            self._draw_vector_arrow(painter, bc, vec_pperp, QColor("#94a3b8"), "P⃗⊥", dashed=True)

        # Friction Force F_ms: opposing slide (up the slope)
        if self._show_friction and f_ms_val > 0.1:
            u_up = QPointF(-u_down.x(), -u_down.y())
            vec_fms = QPointF(u_up.x() * f_ms_val * scale_px, u_up.y() * f_ms_val * scale_px)
            self._draw_vector_arrow(painter, bc, vec_fms, QColor("#8b5cf6"), "F⃗ms")

        # Net Force F_net: along slope
        if self._show_net_force and f_net_val > 0.1:
            vec_fnet = QPointF(u_down.x() * f_net_val * scale_px, u_down.y() * f_net_val * scale_px)
            # Offset slightly to be clearly visible
            start_fnet = QPointF(bc.x() + u_norm_out.x() * 14.0, bc.y() + u_norm_out.y() * 14.0)
            self._draw_vector_arrow(painter, start_fnet, vec_fnet, QColor("#06b6d4"), "F⃗hl")

        # 4. Top Incline Handle W3 (Draggable)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 2.2))
        painter.drawEllipse(w3, self._handle_radius, self._handle_radius)
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(w3, 4.0, 4.0)

        # 5. Physics Calculus HUD Card
        if self._show_hud_card:
            card_w, card_h = 190.0, 84.0
            card_x = self._base_x + 20.0
            card_y = self._base_y - card_h - 20.0
            card_rect = QRectF(card_x, card_y, card_w, card_h)

            card_bg = QColor(15, 23, 42, 235)
            painter.setBrush(QBrush(card_bg))
            painter.setPen(QPen(QColor("#38bdf8"), 1.3))
            painter.drawRoundedRect(card_rect, 6.0, 6.0)

            # Left strip
            strip = QRectF(card_x, card_y + 4.0, 3.5, card_h - 8.0)
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(strip, 1.5, 1.5)

            # Motion State
            sliding = self.is_sliding()
            status_txt = "Trượt xuống dốc ↘" if sliding else "Đứng yên cân bằng ⏹"
            status_color = QColor("#38bdf8") if sliding else QColor("#10b981")
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(status_color))
            painter.drawText(int(card_x + 10.0), int(card_y + 18.0), status_txt)

            # Physical quantities
            a_val = self.acceleration()
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QPen(QColor("#e2e8f0")))
            painter.drawText(int(card_x + 10.0), int(card_y + 36.0), f"P={p_val:.1f}N | N={n_val:.1f}N")
            painter.drawText(int(card_x + 10.0), int(card_y + 52.0), f"F_ms={f_ms_val:.1f}N (μ={self._mu:.2f})")
            painter.drawText(int(card_x + 10.0), int(card_y + 68.0), f"F_hl={f_net_val:.1f}N | a={a_val:.2f}m/s²")

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        _, _, w3 = self.wedge_vertices()
        pos = event.pos()
        if math.hypot(pos.x() - w3.x(), pos.y() - w3.y()) <= self._handle_radius * 1.5:
            self._is_dragging_incline = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_incline:
            pos = event.pos()
            self.air_drag_slope(pos.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_incline:
            self._is_dragging_incline = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Permanently stamp inclined plane and vector forces onto LessonBoard."""
        from src.ui.stem_tools import PhysicsTools
        w1, w2, w3 = self.wedge_vertices()
        bc = self.block_center()
        stamped: list[QGraphicsItem] = []

        # 1. Wedge outline
        wedge_path = QPainterPath()
        wedge_path.moveTo(w1)
        wedge_path.lineTo(w2)
        wedge_path.lineTo(w3)
        wedge_path.closeSubpath()
        from PySide6.QtWidgets import QGraphicsPathItem
        wedge_item = QGraphicsPathItem(wedge_path)
        wedge_item.setPen(QPen(QColor("#64748b"), 2.2))
        wedge_item.setBrush(QBrush(QColor(30, 41, 59, 120)))
        board.addItem(wedge_item)
        stamped.append(wedge_item)

        # 2. Gravity vector P
        scale_px = 5.0
        p_val = self.gravity_force()
        p_end = (bc.x(), bc.y() + p_val * scale_px)
        p_items = PhysicsTools.create_force_vector(
            start=(bc.x(), bc.y()),
            end=p_end,
            label="P⃗",
            color=QColor("#ef4444"),
        )
        for itm in p_items:
            board.addItem(itm)
            stamped.append(itm)

        # 3. Normal force vector N
        rad = math.radians(self._alpha)
        n_val = self.normal_force()
        u_norm_out = (-math.sin(rad), -math.cos(rad))
        n_end = (bc.x() + u_norm_out[0] * n_val * scale_px, bc.y() + u_norm_out[1] * n_val * scale_px)
        n_items = PhysicsTools.create_force_vector(
            start=(bc.x(), bc.y()),
            end=n_end,
            label="N⃗",
            color=QColor("#10b981"),
        )
        for itm in n_items:
            board.addItem(itm)
            stamped.append(itm)

        return stamped
