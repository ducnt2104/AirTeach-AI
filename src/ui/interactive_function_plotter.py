"""
AirTeach AI - Interactive Function Graph Plotter & Tangent Line Inspector Suite
==============================================================================
Provides an interactive mathematical function plotter with real-time calculus inspection:
  - Continuous curve plotting for polynomial, trigonometric, exponential functions.
  - Interactive tracer probe point M(x0, y0) gliding smoothly along the curve.
  - High-precision numerical derivative calculation: k = f'(x0) = lim (f(x0+h) - f(x0-h))/(2h).
  - Dynamic Tangent Line (Tiếp tuyến): y - y0 = k*(x - x0) updated in real time.
  - Calculus HUD badge: coordinates M(x0, y0), slope k, tangent equation, and monotonicity tag (Đồng biến/Nghịch biến/Cực trị).
  - Mouse dragging and touchless air gesture tracking.
  - Vector stamping to LessonBoard canvas.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

import numpy as np
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


class InteractiveFunctionPlotterItem(QGraphicsObject):
    """
    Interactive function graph item with dynamic tangent line and calculus probe.
    Allows teachers and students to visually explore derivatives and rate of change.
    """

    probe_moved = Signal(float, float, float)  # (x0, y0, slope_k)
    expression_changed = Signal(str)

    def __init__(
        self,
        expr: str = "0.25 * x**3 - x",
        origin: tuple[float, float] = (960.0, 540.0),
        grid_step: float = 60.0,
        x_min: float = -6.0,
        x_max: float = 6.0,
        probe_x: float = 1.0,
        curve_color: QColor | str = "#3b82f6",
        tangent_color: QColor | str = "#f59e0b",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._origin_x = float(origin[0])
        self._origin_y = float(origin[1])
        self._grid_step = max(15.0, float(grid_step))
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._probe_x = float(probe_x)
        self._expr = expr.strip()

        self._curve_color = QColor(curve_color) if isinstance(curve_color, str) else curve_color
        self._tangent_color = QColor(tangent_color) if isinstance(tangent_color, str) else tangent_color

        self._show_tangent: bool = True
        self._show_hud_badge: bool = True
        self._is_dragging_probe: bool = False

        self._cached_path: QPainterPath = QPainterPath()
        self._rebuild_curve_path()

        self.setZValue(14.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    # ── Safe Expression Evaluation ─────────────────────────────────────────────

    def evaluate_f(self, x_val: float) -> Optional[float]:
        """Safely compute f(x) for a scalar float."""
        try:
            safe_dict = {
                "x": x_val,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "exp": math.exp,
                "sqrt": math.sqrt,
                "abs": abs,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }
            clean = self._expr.replace("^", "**")
            res = float(eval(clean, {"__builtins__": {}}, safe_dict))  # noqa: S307
            return res if math.isfinite(res) and abs(res) < 1000.0 else None
        except Exception:
            return None

    def evaluate_derivative(self, x0: float) -> float:
        """Compute numerical derivative f'(x0) using symmetric difference quotient."""
        h = 1e-4
        y_plus = self.evaluate_f(x0 + h)
        y_minus = self.evaluate_f(x0 - h)
        if y_plus is not None and y_minus is not None:
            return (y_plus - y_minus) / (2.0 * h)
        return 0.0

    def set_expression(self, expr: str) -> bool:
        """Update mathematical function expression and rebuild path."""
        old_expr = self._expr
        self._expr = expr.strip()
        try:
            # Test evaluation
            test_val = self.evaluate_f(0.0)
            if test_val is None:
                test_val = self.evaluate_f(1.0)
            self.prepareGeometryChange()
            self._rebuild_curve_path()
            self.update()
            self.expression_changed.emit(self._expr)
            return True
        except Exception as exc:
            logger.warning("Failed to set expression %s: %s", expr, exc)
            self._expr = old_expr
            return False

    def expression(self) -> str:
        return self._expr

    # ── Geometry & Path Construction ───────────────────────────────────────────

    def _rebuild_curve_path(self) -> None:
        """Sample function curve and construct anti-aliased QPainterPath."""
        self._cached_path = QPainterPath()
        num_samples = 450
        xs = np.linspace(self._x_min, self._x_max, num_samples)

        first = True
        for x in xs:
            y = self.evaluate_f(float(x))
            if y is None or abs(y) > 40.0:
                first = True
                continue
            sx = self._origin_x + float(x) * self._grid_step
            sy = self._origin_y - float(y) * self._grid_step
            if first:
                self._cached_path.moveTo(sx, sy)
                first = False
            else:
                self._cached_path.lineTo(sx, sy)

    def set_origin(self, ox: float, oy: float) -> None:
        self.prepareGeometryChange()
        self._origin_x = float(ox)
        self._origin_y = float(oy)
        self._rebuild_curve_path()
        self.update()

    def set_grid_step(self, step: float) -> None:
        self.prepareGeometryChange()
        self._grid_step = max(15.0, float(step))
        self._rebuild_curve_path()
        self.update()

    # ── Probe Point & Tangent Inspection ───────────────────────────────────────

    def probe_x(self) -> float:
        return self._probe_x

    def set_probe_x(self, x0: float) -> None:
        """Set tracer probe x-coordinate clamped to [x_min, x_max]."""
        self.prepareGeometryChange()
        self._probe_x = max(self._x_min, min(self._x_max, float(x0)))
        y0 = self.evaluate_f(self._probe_x)
        slope_k = self.evaluate_derivative(self._probe_x)
        self.update()
        if y0 is not None:
            self.probe_moved.emit(self._probe_x, y0, slope_k)

    def probe_scene_pos(self) -> Optional[QPointF]:
        """Return scene coordinates of current probe point M(x0, y0)."""
        y0 = self.evaluate_f(self._probe_x)
        if y0 is None:
            return None
        sx = self._origin_x + self._probe_x * self._grid_step
        sy = self._origin_y - y0 * self._grid_step
        return QPointF(sx, sy)

    def air_drag_probe(self, canvas_x: float, canvas_y: float) -> bool:
        """Touchless air gesture support: translates canvas x-pos to probe x0."""
        target_x0 = (canvas_x - self._origin_x) / self._grid_step
        self.set_probe_x(target_x0)
        return True

    # ── Overlays Toggles ───────────────────────────────────────────────────────

    def toggle_tangent(self) -> bool:
        self.prepareGeometryChange()
        self._show_tangent = not self._show_tangent
        self.update()
        return self._show_tangent

    def toggle_hud_badge(self) -> bool:
        self.prepareGeometryChange()
        self._show_hud_badge = not self._show_hud_badge
        self.update()
        return self._show_hud_badge

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        margin = 150.0
        min_sx = self._origin_x + self._x_min * self._grid_step - margin
        max_sx = self._origin_x + self._x_max * self._grid_step + margin
        return QRectF(
            min_sx,
            self._origin_y - 600.0,
            max_sx - min_sx,
            1200.0,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Neon Glow Halo on Function Curve
        glow_pen = QPen(
            QColor(self._curve_color.red(), self._curve_color.green(), self._curve_color.blue(), 50),
            6.0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.strokePath(self._cached_path, glow_pen)

        # Core Curve
        core_pen = QPen(
            self._curve_color,
            2.6,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.strokePath(self._cached_path, core_pen)

        # 2. Probe Point M(x0, y0) and Tangent Line
        probe_pos = self.probe_scene_pos()
        if probe_pos is None:
            return

        x0 = self._probe_x
        y0 = self.evaluate_f(x0)
        slope_k = self.evaluate_derivative(x0)

        if self._show_tangent and y0 is not None:
            # Draw Tangent Line y - y0 = k*(x - x0)
            dx_arm = 3.5  # units left & right
            tx_left = x0 - dx_arm
            ty_left = y0 - slope_k * dx_arm
            tx_right = x0 + dx_arm
            ty_right = y0 + slope_k * dx_arm

            p_start = QPointF(
                self._origin_x + tx_left * self._grid_step,
                self._origin_y - ty_left * self._grid_step,
            )
            p_end = QPointF(
                self._origin_x + tx_right * self._grid_step,
                self._origin_y - ty_right * self._grid_step,
            )

            # Tangent Glow & Dash Line
            painter.setPen(QPen(QColor(self._tangent_color.red(), self._tangent_color.green(), self._tangent_color.blue(), 80), 5.0))
            painter.drawLine(p_start, p_end)
            painter.setPen(QPen(self._tangent_color, 2.0, Qt.PenStyle.DashLine))
            painter.drawLine(p_start, p_end)

        # 3. Probe Point Marker
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(self._tangent_color, 2.4))
        painter.drawEllipse(probe_pos, 7.5, 7.5)
        painter.setBrush(QBrush(self._tangent_color))
        painter.drawEllipse(probe_pos, 3.5, 3.5)

        # 4. Floating Calculus HUD Badge
        if self._show_hud_badge and y0 is not None:
            badge_w, badge_h = 160.0, 56.0
            bx = probe_pos.x() + 18.0
            by = probe_pos.y() - badge_h - 12.0

            # Clamp badge inside scene margins
            if bx + badge_w > 1920.0 - 20.0:
                bx = probe_pos.x() - badge_w - 18.0
            if by < 20.0:
                by = probe_pos.y() + 20.0

            badge_rect = QRectF(bx, by, badge_w, badge_h)
            badge_bg = QColor(15, 23, 42, 235)
            painter.setBrush(QBrush(badge_bg))
            painter.setPen(QPen(self._tangent_color, 1.3))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)

            # Left accent strip
            strip = QRectF(bx, by + 4.0, 3.5, badge_h - 8.0)
            painter.setBrush(QBrush(self._tangent_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(strip, 1.5, 1.5)

            # Rate tag
            if abs(slope_k) <= 0.05:
                tag_txt = "Cực trị (k≈0)"
                tag_color = QColor("#10b981")  # Emerald
            elif slope_k > 0:
                tag_txt = "Đồng biến ↗"
                tag_color = QColor("#38bdf8")  # Cyan
            else:
                tag_txt = "Nghịch biến ↘"
                tag_color = QColor("#f43f5e")  # Rose

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(tag_color))
            painter.drawText(int(bx + 10.0), int(by + 16.0), tag_txt)

            # Coordinates M(x0, y0)
            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(int(bx + 10.0), int(by + 33.0), f"M({x0:.2f}, {y0:.2f})")

            # Tangent slope & equation
            # y - y0 = k(x - x0) -> y = k*x + (y0 - k*x0)
            c_intercept = y0 - slope_k * x0
            sign_c = "+" if c_intercept >= 0 else "-"
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.drawText(int(bx + 10.0), int(by + 48.0), f"y = {slope_k:.2f}x {sign_c} {abs(c_intercept):.2f}")

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        probe_pos = self.probe_scene_pos()
        if probe_pos is not None:
            pos = event.pos()
            if math.hypot(pos.x() - probe_pos.x(), pos.y() - probe_pos.y()) <= 20.0:
                self._is_dragging_probe = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_probe:
            pos = event.pos()
            target_x0 = (pos.x() - self._origin_x) / self._grid_step
            self.set_probe_x(target_x0)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_probe:
            self._is_dragging_probe = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Permanently stamp function curve and tangent line onto LessonBoard."""
        from src.ui.stem_tools import MathTools
        stamped: list[QGraphicsItem] = []

        # 1. Stamp function curve
        curve_item = MathTools.create_function_graph(
            expr=self._expr,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
            x_min=self._x_min,
            x_max=self._x_max,
            grid_step_px=self._grid_step,
            line_color=self._curve_color,
            line_width=2.5,
        )
        if curve_item:
            board.addItem(curve_item)
            stamped.append(curve_item)

        # 2. Stamp tangent line if enabled
        y0 = self.evaluate_f(self._probe_x)
        slope_k = self.evaluate_derivative(self._probe_x)
        if y0 is not None:
            dx_arm = 3.5
            tx_left = self._probe_x - dx_arm
            ty_left = y0 - slope_k * dx_arm
            tx_right = self._probe_x + dx_arm
            ty_right = y0 + slope_k * dx_arm

            p_start = QPointF(
                self._origin_x + tx_left * self._grid_step,
                self._origin_y - ty_left * self._grid_step,
            )
            p_end = QPointF(
                self._origin_x + tx_right * self._grid_step,
                self._origin_y - ty_right * self._grid_step,
            )
            tangent_path = QPainterPath()
            tangent_path.moveTo(p_start)
            tangent_path.lineTo(p_end)

            from PySide6.QtWidgets import QGraphicsPathItem
            t_item = QGraphicsPathItem(tangent_path)
            t_item.setPen(QPen(self._tangent_color, 2.0, Qt.PenStyle.DashLine))
            t_item.setZValue(12.5)
            board.addItem(t_item)
            stamped.append(t_item)

        return stamped
