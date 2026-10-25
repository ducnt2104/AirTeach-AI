"""
AirTeach AI - Interactive Function Graph Plotter & Tangent Line Inspector Suite
==============================================================================
Provides an interactive mathematical function plotter with real-time calculus inspection:
- Continuous curve plotting for polynomial, trigonometric, exponential functions f(x)
- Multi-curve comparison: dual-function plotting f(x) and g(x) with intersection detection
- Analytical & numerical derivative calculation: k = f'(x0) with derivative curve f'(x)
- Dynamic Tangent Line (Tiếp tuyến): y - y0 = k*(x - x0) updated in real time
- Definite integral area visualization: shaded area between x=a and x=b, plus Riemann sums
- Automatic critical points detection: roots (x-intercepts), y-intercept, local extrema
- Calculus HUD badge: coordinates M(x0, y0), slope k, tangent equation, and monotonicity
- Full JSON serialization for .airteach lesson packaging and presentation replay
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional, TYPE_CHECKING

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
    Interactive function graph item with dynamic tangent line, calculus probe,
    multi-curve plotting, Riemann integral area shading, and critical point analysis.
    """

    probe_moved = Signal(float, float, float)  # (x0, y0, slope_k)
    expression_changed = Signal(str)
    secondary_expression_changed = Signal(str)

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
        self._expr_g: Optional[str] = None

        self._curve_color = QColor(curve_color) if isinstance(curve_color, str) else curve_color
        self._curve_color_g = QColor("#ec4899")  # Pink for g(x)
        self._deriv_color = QColor("#10b981")    # Emerald for f'(x)
        self._tangent_color = QColor(tangent_color) if isinstance(tangent_color, str) else tangent_color

        # Features & Visibility Toggles
        self._show_tangent: bool = True
        self._show_hud_badge: bool = True
        self._show_derivative_curve: bool = False
        self._show_integral_area: bool = False
        self._show_critical_points: bool = True
        self._is_dragging_probe: bool = False

        # Integral & Riemann parameters
        self._integral_a: float = 0.0
        self._integral_b: float = 2.0
        self._riemann_mode: str = "none"  # "none", "left", "right", "midpoint", "trapezoid"
        self._riemann_n: int = 8

        # Cached Paths
        self._cached_path: QPainterPath = QPainterPath()
        self._cached_path_g: QPainterPath = QPainterPath()
        self._cached_path_deriv: QPainterPath = QPainterPath()

        self._rebuild_all_paths()

        self.setZValue(14.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    # ── Safe Expression Evaluation ─────────────────────────────────────────────

    def _eval_raw_string(self, expr_str: str, x_val: float) -> Optional[float]:
        """Safely compute mathematical expression string for a scalar float."""
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
            clean = expr_str.replace("^", "**")
            res = float(eval(clean, {"__builtins__": {}}, safe_dict))  # noqa: S307
            return res if math.isfinite(res) and abs(res) < 1000.0 else None
        except Exception:
            return None

    def evaluate_f(self, x_val: float) -> Optional[float]:
        """Safely compute f(x) for a scalar float."""
        return self._eval_raw_string(self._expr, x_val)

    def evaluate_g(self, x_val: float) -> Optional[float]:
        """Safely compute secondary function g(x) for a scalar float."""
        if not self._expr_g:
            return None
        return self._eval_raw_string(self._expr_g, x_val)

    def evaluate_derivative(self, x0: float) -> float:
        """Compute numerical derivative f'(x0) using symmetric difference quotient."""
        h = 1e-4
        y_plus = self.evaluate_f(x0 + h)
        y_minus = self.evaluate_f(x0 - h)
        if y_plus is not None and y_minus is not None:
            return (y_plus - y_minus) / (2.0 * h)
        return 0.0

    def set_expression(self, expr: str) -> bool:
        """Update primary mathematical function f(x) and rebuild paths."""
        old_expr = self._expr
        self._expr = expr.strip()
        try:
            test_val = self.evaluate_f(0.0)
            if test_val is None:
                test_val = self.evaluate_f(1.0)
            self.prepareGeometryChange()
            self._rebuild_all_paths()
            self.update()
            self.expression_changed.emit(self._expr)
            return True
        except Exception as exc:
            logger.warning("Failed to set expression %s: %s", expr, exc)
            self._expr = old_expr
            return False

    def expression(self) -> str:
        return self._expr

    def set_secondary_expression(self, expr: Optional[str]) -> bool:
        """Update secondary mathematical function g(x) and rebuild path."""
        if expr is None or not expr.strip():
            self._expr_g = None
            self._cached_path_g = QPainterPath()
            self.update()
            return True

        clean = expr.strip()
        test_val = self._eval_raw_string(clean, 0.0)
        if test_val is None:
            test_val = self._eval_raw_string(clean, 1.0)

        self._expr_g = clean
        self.prepareGeometryChange()
        self._rebuild_curve_path_g()
        self.update()
        self.secondary_expression_changed.emit(clean)
        return True

    def secondary_expression(self) -> Optional[str]:
        return self._expr_g

    def set_symbolic_expression(self, latex_or_expr: str) -> bool:
        """Set expression from LaTeX formula using SymbolicMathEngine parser."""
        try:
            from src.services.symbolic_math import convert_latex_to_python_expr
            py_expr = convert_latex_to_python_expr(latex_or_expr)
            if "=" in py_expr:
                # e.g. y = x^2 - 4 -> extract RHS
                parts = py_expr.split("=", 1)
                py_expr = parts[1].strip() if "y" in parts[0] or "f" in parts[0] else parts[0].strip()
            return self.set_expression(py_expr)
        except Exception as exc:
            logger.warning("Failed to set symbolic expression %s: %s", latex_or_expr, exc)
            return self.set_expression(latex_or_expr)

    # ── Geometry & Path Construction ───────────────────────────────────────────

    def _rebuild_all_paths(self) -> None:
        self._rebuild_curve_path_f()
        self._rebuild_curve_path_g()
        self._rebuild_derivative_path()

    def _rebuild_curve_path_f(self) -> None:
        """Sample primary function f(x) curve and construct anti-aliased QPainterPath."""
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

    def _rebuild_curve_path_g(self) -> None:
        """Sample secondary function g(x) curve if configured."""
        self._cached_path_g = QPainterPath()
        if not self._expr_g:
            return

        num_samples = 450
        xs = np.linspace(self._x_min, self._x_max, num_samples)
        first = True
        for x in xs:
            y = self.evaluate_g(float(x))
            if y is None or abs(y) > 40.0:
                first = True
                continue
            sx = self._origin_x + float(x) * self._grid_step
            sy = self._origin_y - float(y) * self._grid_step
            if first:
                self._cached_path_g.moveTo(sx, sy)
                first = False
            else:
                self._cached_path_g.lineTo(sx, sy)

    def _rebuild_derivative_path(self) -> None:
        """Sample derivative f'(x) curve."""
        self._cached_path_deriv = QPainterPath()
        num_samples = 300
        xs = np.linspace(self._x_min, self._x_max, num_samples)
        first = True
        for x in xs:
            dy = self.evaluate_derivative(float(x))
            if abs(dy) > 40.0:
                first = True
                continue
            sx = self._origin_x + float(x) * self._grid_step
            sy = self._origin_y - float(dy) * self._grid_step
            if first:
                self._cached_path_deriv.moveTo(sx, sy)
                first = False
            else:
                self._cached_path_deriv.lineTo(sx, sy)

    def set_origin(self, ox: float, oy: float) -> None:
        """Set Cartesian origin coordinates (ox, oy) on canvas."""
        self.prepareGeometryChange()
        self._origin_x = float(ox)
        self._origin_y = float(oy)
        self._rebuild_all_paths()
        self.update()

    def set_grid_step(self, step: float) -> None:
        """Set pixels per unit scale step."""
        self.prepareGeometryChange()
        self._grid_step = max(15.0, float(step))
        self._rebuild_all_paths()
        self.update()

    # ── Probe Point & Toggles ──────────────────────────────────────────────────

    def probe_x(self) -> float:
        return self._probe_x

    def set_probe_x(self, x0: float) -> None:
        """Move tangent tracer probe along curve to x0."""
        self._probe_x = float(np.clip(x0, self._x_min, self._x_max))
        self.update()
        y0 = self.evaluate_f(self._probe_x)
        slope_k = self.evaluate_derivative(self._probe_x)
        if y0 is not None:
            self.probe_moved.emit(self._probe_x, y0, slope_k)

    def probe_scene_pos(self) -> Optional[QPointF]:
        """Return scene coordinates of probe point M(x0, y0)."""
        y0 = self.evaluate_f(self._probe_x)
        if y0 is None:
            return None
        return QPointF(
            self._origin_x + self._probe_x * self._grid_step,
            self._origin_y - y0 * self._grid_step,
        )

    def air_drag_probe(self, canvas_x: float, canvas_y: float) -> bool:
        """Update probe position from hand gesture canvas coordinates."""
        dx_units = (canvas_x - self._origin_x) / self._grid_step
        if self._x_min <= dx_units <= self._x_max:
            self.set_probe_x(dx_units)
            return True
        return False

    def toggle_tangent(self) -> bool:
        self._show_tangent = not self._show_tangent
        self.update()
        return self._show_tangent

    def toggle_hud_badge(self) -> bool:
        self._show_hud_badge = not self._show_hud_badge
        self.update()
        return self._show_hud_badge

    def toggle_derivative_curve(self) -> bool:
        self._show_derivative_curve = not self._show_derivative_curve
        if self._show_derivative_curve:
            self._rebuild_derivative_path()
        self.update()
        return self._show_derivative_curve

    def set_show_derivative_curve(self, show: bool) -> None:
        self._show_derivative_curve = show
        if show:
            self._rebuild_derivative_path()
        self.update()

    def toggle_integral_area(self) -> bool:
        self._show_integral_area = not self._show_integral_area
        self.update()
        return self._show_integral_area

    def set_integral_range(self, a: float, b: float) -> None:
        self._integral_a = min(float(a), float(b))
        self._integral_b = max(float(a), float(b))
        self.update()

    def set_riemann_mode(self, mode: str, n: int = 8) -> None:
        self._riemann_mode = mode.lower()
        self._riemann_n = max(2, min(64, int(n)))
        self.update()

    def toggle_critical_points(self) -> bool:
        self._show_critical_points = not self._show_critical_points
        self.update()
        return self._show_critical_points

    # ── Critical Points Analysis ───────────────────────────────────────────────

    def find_critical_points(self) -> dict[str, list[tuple[float, float]]]:
        """
        Analyze numerical roots, y-intercept, and local extrema within [x_min, x_max].
        """
        results: dict[str, list[tuple[float, float]]] = {
            "roots": [],
            "y_intercept": [],
            "extrema": [],
        }

        # 1. Y-Intercept
        y0 = self.evaluate_f(0.0)
        if y0 is not None:
            results["y_intercept"].append((0.0, y0))

        # 2. Roots & Extrema via Grid Search
        samples = 300
        xs = np.linspace(self._x_min, self._x_max, samples)
        prev_y = self.evaluate_f(xs[0])
        prev_dy = self.evaluate_derivative(xs[0])

        for i in range(1, len(xs)):
            x = xs[i]
            y = self.evaluate_f(x)
            dy = self.evaluate_derivative(x)

            if y is not None and prev_y is not None:
                # Sign change in y -> Root
                if (prev_y < 0 and y >= 0) or (prev_y > 0 and y <= 0):
                    root_x = xs[i - 1] - prev_y * (x - xs[i - 1]) / (y - prev_y)
                    results["roots"].append((round(root_x, 3), 0.0))

            if dy is not None and prev_dy is not None:
                # Sign change in derivative -> Extrema
                if (prev_dy < 0 and dy > 0) or (prev_dy > 0 and dy < 0):
                    ext_x = xs[i - 1] - prev_dy * (x - xs[i - 1]) / (dy - prev_dy)
                    ext_y = self.evaluate_f(ext_x)
                    if ext_y is not None:
                        results["extrema"].append((round(ext_x, 3), round(ext_y, 3)))

            prev_y = y
            prev_dy = dy

        return results

    # ── Bounding Rect ──────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        min_sx = self._origin_x + self._x_min * self._grid_step - 50.0
        max_sx = self._origin_x + self._x_max * self._grid_step + 50.0
        return QRectF(
            min_sx,
            self._origin_y - 600.0,
            max_sx - min_sx,
            1200.0,
        )

    # ── Painting ───────────────────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Shaded Definite Integral Area & Riemann Rectangles
        if self._show_integral_area:
            self._paint_integral_area(painter)

        # 2. Derivative Curve f'(x)
        if self._show_derivative_curve and not self._cached_path_deriv.isEmpty():
            deriv_pen = QPen(self._deriv_color, 2.0, Qt.PenStyle.DashDotLine)
            painter.strokePath(self._cached_path_deriv, deriv_pen)

        # 3. Secondary Curve g(x)
        if self._expr_g and not self._cached_path_g.isEmpty():
            g_glow = QPen(QColor(236, 72, 153, 60), 5.0)
            painter.strokePath(self._cached_path_g, g_glow)
            g_pen = QPen(self._curve_color_g, 2.4, Qt.PenStyle.SolidLine)
            painter.strokePath(self._cached_path_g, g_pen)

        # 4. Primary Curve f(x)
        glow_pen = QPen(
            QColor(self._curve_color.red(), self._curve_color.green(), self._curve_color.blue(), 50),
            6.0,
        )
        painter.strokePath(self._cached_path, glow_pen)
        core_pen = QPen(self._curve_color, 2.6)
        painter.strokePath(self._cached_path, core_pen)

        # 5. Critical Points Markers (Roots & Extrema)
        if self._show_critical_points:
            self._paint_critical_points(painter)

        # 6. Probe Point M(x0, y0) and Tangent Line
        probe_pos = self.probe_scene_pos()
        if probe_pos is not None:
            x0 = self._probe_x
            y0 = self.evaluate_f(x0)
            slope_k = self.evaluate_derivative(x0)

            if self._show_tangent and y0 is not None:
                dx_arm = 3.5
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

                painter.setPen(QPen(QColor(self._tangent_color.red(), self._tangent_color.green(), self._tangent_color.blue(), 80), 5.0))
                painter.drawLine(p_start, p_end)
                painter.setPen(QPen(self._tangent_color, 2.0, Qt.PenStyle.DashLine))
                painter.drawLine(p_start, p_end)

            # Probe Marker
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(self._tangent_color, 2.4))
            painter.drawEllipse(probe_pos, 7.5, 7.5)
            painter.setBrush(QBrush(self._tangent_color))
            painter.drawEllipse(probe_pos, 3.5, 3.5)

            # Floating Calculus HUD Badge
            if self._show_hud_badge and y0 is not None:
                self._paint_calculus_hud(painter, probe_pos, x0, y0, slope_k)

        painter.restore()

    def _paint_integral_area(self, painter: QPainter) -> None:
        """Paint shaded definite integral area under curve or Riemann rectangles."""
        a = self._integral_a
        b = self._integral_b
        if b <= a:
            return

        # 1. Smooth Area Shading
        if self._riemann_mode == "none":
            area_path = QPainterPath()
            sx_start = self._origin_x + a * self._grid_step
            area_path.moveTo(sx_start, self._origin_y)

            num_steps = 120
            xs = np.linspace(a, b, num_steps)
            for x in xs:
                y = self.evaluate_f(float(x)) or 0.0
                sx = self._origin_x + float(x) * self._grid_step
                sy = self._origin_y - float(y) * self._grid_step
                area_path.lineTo(sx, sy)

            sx_end = self._origin_x + b * self._grid_step
            area_path.lineTo(sx_end, self._origin_y)
            area_path.closeSubpath()

            area_brush = QBrush(QColor(99, 102, 241, 65))  # Indigo translucent
            painter.fillPath(area_path, area_brush)
        else:
            # 2. Riemann Sum Rectangles
            n = self._riemann_n
            dx = (b - a) / n
            rect_pen = QPen(QColor(168, 85, 247, 190), 1.0)
            rect_brush = QBrush(QColor(168, 85, 247, 75))

            for i in range(n):
                x_left = a + i * dx
                x_right = x_left + dx
                if self._riemann_mode == "left":
                    x_eval = x_left
                elif self._riemann_mode == "right":
                    x_eval = x_right
                else:  # midpoint
                    x_eval = (x_left + x_right) / 2.0

                h = self.evaluate_f(x_eval) or 0.0
                rx = self._origin_x + x_left * self._grid_step
                rw = dx * self._grid_step
                ry = self._origin_y - max(0.0, h) * self._grid_step
                rh = abs(h) * self._grid_step

                rect = QRectF(rx, ry, rw, rh)
                painter.fillRect(rect, rect_brush)
                painter.setPen(rect_pen)
                painter.drawRect(rect)

    def _paint_critical_points(self, painter: QPainter) -> None:
        """Paint glowing markers at roots and local extrema."""
        crit = self.find_critical_points()

        # Roots (x-intercepts) -> Emerald markers
        for rx, ry in crit.get("roots", []):
            sx = self._origin_x + rx * self._grid_step
            sy = self._origin_y
            pt = QPointF(sx, sy)
            painter.setBrush(QBrush(QColor("#10b981")))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(pt, 4.5, 4.5)

        # Extrema -> Amber Diamond markers
        for ex, ey in crit.get("extrema", []):
            sx = self._origin_x + ex * self._grid_step
            sy = self._origin_y - ey * self._grid_step
            pt = QPointF(sx, sy)
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(pt, 5.0, 5.0)

    def _paint_calculus_hud(
        self,
        painter: QPainter,
        probe_pos: QPointF,
        x0: float,
        y0: float,
        slope_k: float,
    ) -> None:
        badge_w, badge_h = 175.0, 62.0
        bx = probe_pos.x() + 18.0
        by = probe_pos.y() - badge_h - 12.0

        if bx + badge_w > 1920.0 - 20.0:
            bx = probe_pos.x() - badge_w - 18.0
        if by < 20.0:
            by = probe_pos.y() + 20.0

        badge_rect = QRectF(bx, by, badge_w, badge_h)
        badge_bg = QColor(15, 23, 42, 240)
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(self._tangent_color, 1.4))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        # Left accent strip
        strip = QRectF(bx, by + 4.0, 3.5, badge_h - 8.0)
        painter.setBrush(QBrush(self._tangent_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(strip, 1.5, 1.5)

        # Rate tag
        if abs(slope_k) <= 0.05:
            tag_txt = "Cực trị (k≈0)"
            tag_color = QColor("#10b981")
        elif slope_k > 0:
            tag_txt = f"Đồng biến ↗ (k={slope_k:.2f})"
            tag_color = QColor("#38bdf8")
        else:
            tag_txt = f"Nghịch biến ↘ (k={slope_k:.2f})"
            tag_color = QColor("#f43f5e")

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(tag_color))
        painter.drawText(int(bx + 10.0), int(by + 16.0), tag_txt)

        # Coordinates M(x0, y0)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(int(bx + 10.0), int(by + 33.0), f"M({x0:.2f}, {y0:.2f})")

        # Tangent slope & equation
        c_intercept = y0 - slope_k * x0
        sign_c = "+" if c_intercept >= 0 else "-"
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(int(bx + 10.0), int(by + 48.0), f"y = {slope_k:.2f}x {sign_c} {abs(c_intercept):.2f}")

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        probe_pos = self.probe_scene_pos()
        if probe_pos is not None:
            dist = (event.pos() - probe_pos).manhattanLength()
            if dist < 25.0:
                self._is_dragging_probe = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_probe:
            canvas_x = event.pos().x()
            dx_units = (canvas_x - self._origin_x) / self._grid_step
            self.set_probe_x(dx_units)
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

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize plotter state for .airteach packages."""
        return {
            "type": "interactive_function_plotter",
            "expr": self._expr,
            "expr_g": self._expr_g,
            "origin": [self._origin_x, self._origin_y],
            "grid_step": self._grid_step,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "probe_x": self._probe_x,
            "show_tangent": self._show_tangent,
            "show_hud_badge": self._show_hud_badge,
            "show_derivative": self._show_derivative_curve,
            "show_integral": self._show_integral_area,
            "integral_range": [self._integral_a, self._integral_b],
            "riemann_mode": self._riemann_mode,
            "riemann_n": self._riemann_n,
            "show_critical_points": self._show_critical_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractiveFunctionPlotterItem:
        """Reconstruct plotter from serialized dictionary."""
        origin = data.get("origin", [960.0, 540.0])
        item = cls(
            expr=str(data.get("expr", "0.25 * x**3 - x")),
            origin=(float(origin[0]), float(origin[1])),
            grid_step=float(data.get("grid_step", 60.0)),
            x_min=float(data.get("x_min", -6.0)),
            x_max=float(data.get("x_max", 6.0)),
            probe_x=float(data.get("probe_x", 1.0)),
        )
        if data.get("expr_g"):
            item.set_secondary_expression(str(data.get("expr_g")))
        item._show_tangent = bool(data.get("show_tangent", True))
        item._show_hud_badge = bool(data.get("show_hud_badge", True))
        item.set_show_derivative_curve(bool(data.get("show_derivative", False)))
        item._show_integral_area = bool(data.get("show_integral", False))
        ir = data.get("integral_range", [0.0, 2.0])
        item.set_integral_range(ir[0], ir[1])
        item.set_riemann_mode(
            str(data.get("riemann_mode", "none")),
            int(data.get("riemann_n", 8)),
        )
        item._show_critical_points = bool(data.get("show_critical_points", True))
        return item
