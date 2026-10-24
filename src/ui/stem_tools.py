"""
AirTeach AI - STEM Teaching Suite & Subject Workspaces
======================================================
Provides interactive subject-specific teaching tools for:
  - Mathematics: Cartesian axes (Oxy), grid, function graphing, geometric shapes
  - Physics: Force vectors, vector decomposition (Fx, Fy), circuit schematics
  - Chemistry: Chemical formula auto-formatting (H₂SO₄), reaction arrows, benzene/bonds
  - Biology: Callout leaders, histological/anatomical labeled pins
  - Informatics: Flowchart shapes (Process, Decision, I/O, Terminator), code blocks
  - General Teaching: Spotlight, highlighter, measurement guides
"""
from __future__ import annotations

import logging
import math
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QObject
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
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ─── Subject Workspace Enum ───────────────────────────────────────────────────

class StemSubject(Enum):
    GENERAL     = "Chung"
    MATHEMATICS = "Toán học"
    PHYSICS     = "Vật lý"
    CHEMISTRY   = "Hóa học"
    BIOLOGY     = "Sinh học"
    INFORMATICS = "Tin học"


# ─── Unicode Chemical & Mathematical Formatters ───────────────────────────────

_SUBSCRIPT_MAP = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
}

_SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ',
}


def format_chemical_formula(raw: str) -> str:
    """
    Convert raw chemical formulas into clean Unicode formatted formulas.
    E.g. H2SO4 -> H₂SO₄, Ca(OH)2 -> Ca(OH)₂, Cu2+ -> Cu²⁺, Fe3+ -> Fe³⁺, SO4^2- -> SO₄²⁻
    """
    result = []
    i = 0
    n = len(raw)
    while i < n:
        char = raw[i]
        if char == '^':
            # Superscript group
            i += 1
            if i < n and raw[i] == '{':
                i += 1
                while i < n and raw[i] != '}':
                    result.append(_SUPERSCRIPT_MAP.get(raw[i], raw[i]))
                    i += 1
                if i < n and raw[i] == '}':
                    i += 1
            elif i < n:
                result.append(_SUPERSCRIPT_MAP.get(raw[i], raw[i]))
                i += 1
        elif char.isdigit():
            # Check if this digit is followed by + or - (ionic charge e.g. 2+, 3-)
            if i + 1 < n and raw[i + 1] in ('+', '-'):
                result.append(_SUPERSCRIPT_MAP.get(char, char))
                result.append(_SUPERSCRIPT_MAP.get(raw[i + 1], raw[i + 1]))
                i += 2
            elif (
                (i > 0 and (raw[i - 1].isalpha() or raw[i - 1] in (')', ']', '}')))
                or (result and result[-1] in _SUBSCRIPT_MAP.values())
            ):
                # Subscript index following an element symbol, closing bracket, or another subscript
                result.append(_SUBSCRIPT_MAP.get(char, char))
                i += 1
            else:
                # Leading stoichiometric coefficient (e.g. 2H2O -> 2H₂O)
                result.append(char)
                i += 1
        elif char in ('+', '-') and (i > 0 and (raw[i - 1].isalpha() or raw[i - 1] in _SUBSCRIPT_MAP.values())):
            result.append(_SUPERSCRIPT_MAP.get(char, char))
            i += 1
        else:
            result.append(char)
            i += 1
    return "".join(result)


# ─── Vector Tools for Mathematics ─────────────────────────────────────────────

class MathTools:
    """Mathematical vector geometry generators."""

    @staticmethod
    def create_cartesian_axes(
        origin_x: float = 960.0,
        origin_y: float = 540.0,
        x_min: float = -8.0,
        x_max: float = 8.0,
        y_min: float = -5.0,
        y_max: float = 5.0,
        grid_step_px: float = 60.0,
        show_grid: bool = True,
        pen_color: QColor = QColor("#202124"),
    ) -> list[QGraphicsItem]:
        """
        Create a high-precision Cartesian coordinate plane (Oxy)
        with grid lines, tick marks, arrows, and axis labels.
        """
        items: list[QGraphicsItem] = []

        # 1. Background Grid (optional)
        if show_grid:
            grid_path = QPainterPath()
            curr_x = origin_x + x_min * grid_step_px
            max_x = origin_x + x_max * grid_step_px
            min_y = origin_y - y_max * grid_step_px
            max_y = origin_y - y_min * grid_step_px

            # Vertical grid lines
            x = curr_x
            while x <= max_x + 0.1:
                grid_path.moveTo(x, min_y)
                grid_path.lineTo(x, max_y)
                x += grid_step_px

            # Horizontal grid lines
            y = min_y
            while y <= max_y + 0.1:
                grid_path.moveTo(curr_x, y)
                grid_path.lineTo(max_x, y)
                y += grid_step_px

            grid_item = QGraphicsPathItem(grid_path)
            grid_pen = QPen(QColor(220, 224, 230), 1.0, Qt.PenStyle.DashLine)
            grid_item.setPen(grid_pen)
            grid_item.setZValue(5.0)
            items.append(grid_item)

        # 2. Main Axes (X and Y with arrows)
        axis_pen = QPen(pen_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        left_x = origin_x + x_min * grid_step_px - 20
        right_x = origin_x + x_max * grid_step_px + 20
        bottom_y = origin_y - y_min * grid_step_px + 20
        top_y = origin_y - y_max * grid_step_px - 20

        # X-Axis
        x_axis = QGraphicsLineItem(left_x, origin_y, right_x, origin_y)
        x_axis.setPen(axis_pen)
        x_axis.setZValue(11.0)
        items.append(x_axis)

        # Y-Axis
        y_axis = QGraphicsLineItem(origin_x, bottom_y, origin_x, top_y)
        y_axis.setPen(axis_pen)
        y_axis.setZValue(11.0)
        items.append(y_axis)

        # Arrowheads for X and Y
        arrow_size = 12.0
        # X arrow
        x_arrow_poly = QPolygonF([
            QPointF(right_x, origin_y),
            QPointF(right_x - arrow_size, origin_y - arrow_size * 0.4),
            QPointF(right_x - arrow_size, origin_y + arrow_size * 0.4),
        ])
        x_arrow = QGraphicsPolygonItem(x_arrow_poly)
        x_arrow.setBrush(QBrush(pen_color))
        x_arrow.setPen(Qt.PenStyle.NoPen)
        x_arrow.setZValue(11.5)
        items.append(x_arrow)

        # Y arrow
        y_arrow_poly = QPolygonF([
            QPointF(origin_x, top_y),
            QPointF(origin_x - arrow_size * 0.4, top_y + arrow_size),
            QPointF(origin_x + arrow_size * 0.4, top_y + arrow_size),
        ])
        y_arrow = QGraphicsPolygonItem(y_arrow_poly)
        y_arrow.setBrush(QBrush(pen_color))
        y_arrow.setPen(Qt.PenStyle.NoPen)
        y_arrow.setZValue(11.5)
        items.append(y_arrow)

        # Axis label texts: "x", "y", "O"
        font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        lbl_x = QGraphicsTextItem("x")
        lbl_x.setFont(font)
        lbl_x.setDefaultTextColor(pen_color)
        lbl_x.setPos(right_x - 10, origin_y + 8)
        lbl_x.setZValue(12.0)
        items.append(lbl_x)

        lbl_y = QGraphicsTextItem("y")
        lbl_y.setFont(font)
        lbl_y.setDefaultTextColor(pen_color)
        lbl_y.setPos(origin_x + 12, top_y - 10)
        lbl_y.setZValue(12.0)
        items.append(lbl_y)

        lbl_o = QGraphicsTextItem("O")
        lbl_o.setFont(QFont("Segoe UI", 11, QFont.Weight.Normal))
        lbl_o.setDefaultTextColor(pen_color)
        lbl_o.setPos(origin_x - 18, origin_y + 4)
        lbl_o.setZValue(12.0)
        items.append(lbl_o)

        # 3. Ticks and Numbers
        tick_len = 4.0
        tick_font = QFont("Segoe UI", 9)
        for unit_x in range(int(x_min), int(x_max) + 1):
            if unit_x == 0:
                continue
            px = origin_x + unit_x * grid_step_px
            tick = QGraphicsLineItem(px, origin_y - tick_len, px, origin_y + tick_len)
            tick.setPen(axis_pen)
            tick.setZValue(11.0)
            items.append(tick)

            t_item = QGraphicsTextItem(str(unit_x))
            t_item.setFont(tick_font)
            t_item.setDefaultTextColor(pen_color)
            t_item.setPos(px - 8, origin_y + 5)
            t_item.setZValue(12.0)
            items.append(t_item)

        for unit_y in range(int(y_min), int(y_max) + 1):
            if unit_y == 0:
                continue
            py = origin_y - unit_y * grid_step_px
            tick = QGraphicsLineItem(origin_x - tick_len, py, origin_x + tick_len, py)
            tick.setPen(axis_pen)
            tick.setZValue(11.0)
            items.append(tick)

            t_item = QGraphicsTextItem(str(unit_y))
            t_item.setFont(tick_font)
            t_item.setDefaultTextColor(pen_color)
            t_item.setPos(origin_x - 22, py - 10)
            t_item.setZValue(12.0)
            items.append(t_item)

        return items

    @staticmethod
    def create_function_graph(
        expr: str,
        origin_x: float = 960.0,
        origin_y: float = 540.0,
        x_min: float = -6.0,
        x_max: float = 6.0,
        grid_step_px: float = 60.0,
        line_color: QColor = QColor("#1a73e8"),
        line_width: float = 3.0,
    ) -> Optional[QGraphicsPathItem]:
        """
        Plot continuous function curves (e.g. 'x**2', 'sin(x)', '2*x - 1', 'exp(x)', 'sqrt(x)').
        Uses safe mathematical evaluation over numpy array.
        """
        try:
            x_vals = np.linspace(x_min, x_max, 400)
            safe_dict = {
                "x": x_vals,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "log": np.log,
                "abs": np.abs,
                "pi": np.pi,
                "e": np.e,
            }
            # Clean expression
            clean_expr = expr.replace("^", "**")
            y_vals = eval(clean_expr, {"__builtins__": {}}, safe_dict)  # noqa: S307

            path = QPainterPath()
            first = True
            for x, y in zip(x_vals, y_vals):
                if not np.isfinite(y) or abs(y) > 50.0:
                    first = True
                    continue
                px = origin_x + float(x) * grid_step_px
                py = origin_y - float(y) * grid_step_px
                if first:
                    path.moveTo(px, py)
                    first = False
                else:
                    path.lineTo(px, py)

            item = QGraphicsPathItem(path)
            pen = QPen(line_color, line_width, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            item.setPen(pen)
            item.setZValue(13.0)
            return item
        except Exception as exc:
            logger.warning("Function plot error for %s: %s", expr, exc)
            return None


# ─── Vector Tools for Physics ─────────────────────────────────────────────────

    @staticmethod
    def create_calculus_riemann_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'RIEMANN_SINE_INTEGRAL',
        func_name: str = 'f(x) = sin(x)',
        exact_val: float = 2.0,
        approx_val: float = 1.974,
        n_partitions: int = 8,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Riemann Calculus and Numerical Quadrature."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#a855f7'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'GIẢI TÍCH & TÍCH PHÂN RIEMANN: {func_name} – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 320.0
        vh = 195.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Baseline & grid
        base_y = vy0 + vh - 30.0
        axis = QGraphicsLineItem(vx0 + 15.0, base_y, vx0 + vw - 15.0, base_y)
        axis.setPen(QPen(QColor('#475569'), 1.0))
        axis.setZValue(10.3)
        items.append(axis)

        # Draw representative Riemann partition rectangles
        rw = (vw - 60.0) / float(n_partitions)
        for i in range(n_partitions):
            rx = vx0 + 30.0 + i * rw
            rh = 50.0 + 40.0 * math.sin((i + 0.5) * math.pi / float(n_partitions))
            rect = QGraphicsRectItem(rx, base_y - rh, rw - 1.0, rh)
            rect.setBrush(QBrush(QColor(168, 85, 247, 40)))
            rect.setPen(QPen(QColor('#a855f7'), 1.2))
            rect.setZValue(10.4)
            items.append(rect)

        # Right box
        rx0 = vx0 + vw + 14.0
        rw_box = width - (rx0 - x0) - 16.0
        r_box = QGraphicsRectItem(rx0, vy0, rw_box, vh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info = QGraphicsTextItem(
            f'THÔNG SỐ TÍCH PHÂN RIEMANN:\n'
            f'• Hàm số: {func_name}\n'
            f'• Giá trị chính xác (FTC): {exact_val:.4f}\n'
            f'• Xấp xỉ Cầu phương: {approx_val:.4f}\n'
            f'• Số phân khoảng n: {n_partitions}\n'
            f'• Sai số tuyệt đối: {abs(exact_val - approx_val):.4f}'
        )
        info.setFont(QFont('Segoe UI', 7))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(rx0 + 10.0, vy0 + 12.0)
        info.setZValue(10.4)
        items.append(info)

        return items



class PhysicsTools:
    """Physics vector mechanics, force decomposition, and circuit symbols."""

    @staticmethod
    def create_force_vector(
        start: tuple[float, float],
        end: tuple[float, float],
        label: str = "F⃗",
        color: QColor = QColor("#ea4335"),
        width: float = 3.0,
    ) -> list[QGraphicsItem]:
        """
        Create a styled force vector arrow with physics notation (e.g. F⃗, v⃗, a⃗, P⃗).
        """
        items: list[QGraphicsItem] = []
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-2:
            return items

        ux, uy = dx / length, dy / length
        arrow_len = min(18.0, length * 0.3)
        angle = math.radians(24)

        # Main vector line
        line = QGraphicsLineItem(x0, y0, x1, y1)
        pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        line.setPen(pen)
        line.setZValue(12.0)
        items.append(line)

        # Arrowhead
        p_tip = QPointF(x1, y1)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        l1 = QPointF(x1 - (ux * cos_a - uy * sin_a) * arrow_len,
                     y1 - (ux * sin_a + uy * cos_a) * arrow_len)
        l2 = QPointF(x1 - (ux * cos_a + uy * sin_a) * arrow_len,
                     y1 - (-ux * sin_a + uy * cos_a) * arrow_len)

        arrow_poly = QPolygonF([p_tip, l1, l2])
        arrow_head = QGraphicsPolygonItem(arrow_poly)
        arrow_head.setBrush(QBrush(color))
        arrow_head.setPen(Qt.PenStyle.NoPen)
        arrow_head.setZValue(12.5)
        items.append(arrow_head)

        # Text label
        if label:
            txt = QGraphicsTextItem(label)
            txt.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            txt.setDefaultTextColor(color)
            # Position offset perpendicular to arrow
            perp_x = -uy * 18.0
            perp_y = ux * 18.0
            txt.setPos(x1 + perp_x - 10, y1 + perp_y - 15)
            txt.setZValue(13.0)
            items.append(txt)

        return items

    @staticmethod
    def create_vector_decomposition(
        origin: tuple[float, float],
        vector_end: tuple[float, float],
        label: str = "F⃗",
        color_main: QColor = QColor("#ea4335"),
        color_comp: QColor = QColor("#1a73e8"),
    ) -> list[QGraphicsItem]:
        """
        Decomposes a 2D vector into its orthogonal components Fx and Fy
        with dashed projection guidelines.
        """
        items: list[QGraphicsItem] = []
        ox, oy = origin
        vx, vy = vector_end
        fx_end = (vx, oy)
        fy_end = (ox, vy)

        # 1. Main Vector
        items.extend(PhysicsTools.create_force_vector(origin, vector_end, label=label, color=color_main, width=3.2))

        # 2. X-Component (Fx)
        items.extend(PhysicsTools.create_force_vector(origin, fx_end, label=f"{label}x", color=color_comp, width=2.2))

        # 3. Y-Component (Fy)
        items.extend(PhysicsTools.create_force_vector(origin, fy_end, label=f"{label}y", color=color_comp, width=2.2))

        # 4. Dashed Projection Box Lines
        dash_pen = QPen(QColor("#5f6368"), 1.2, Qt.PenStyle.DashLine)
        proj_x = QGraphicsLineItem(vx, oy, vx, vy)
        proj_x.setPen(dash_pen)
        proj_x.setZValue(11.0)
        items.append(proj_x)

        proj_y = QGraphicsLineItem(ox, vy, vx, vy)
        proj_y.setPen(dash_pen)
        proj_y.setZValue(11.0)
        items.append(proj_y)

        return items

    @staticmethod
    def create_circuit_component(
        component: str,
        start: tuple[float, float],
        end: tuple[float, float],
        label: str = "",
        color: QColor = QColor("#202124"),
    ) -> list[QGraphicsItem]:
        """
        Render electronic schematic symbols:
        Resistor (R), Capacitor (C), Battery/DC Source, Switch, Ground.
        """
        items: list[QGraphicsItem] = []
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 20.0:
            return items

        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular normal
        pen = QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        comp = component.lower()
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0

        if "resistor" in comp or "điện trở" in comp:
            # Zig-zag resistor symbol
            lead_len = (length - 40.0) / 2.0
            p_in = (x0 + ux * lead_len, y0 + uy * lead_len)
            p_out = (x1 - ux * lead_len, y1 - uy * lead_len)

            path = QPainterPath()
            path.moveTo(x0, y0)
            path.lineTo(*p_in)

            # 4 zig-zag teeth
            teeth = 4
            teeth_len = 40.0 / teeth
            for i in range(teeth):
                sign = 1 if i % 2 == 0 else -1
                t_mid = (p_in[0] + ux * (i + 0.5) * teeth_len + px * sign * 10.0,
                         p_in[1] + uy * (i + 0.5) * teeth_len + py * sign * 10.0)
                t_end = (p_in[0] + ux * (i + 1) * teeth_len,
                         p_in[1] + uy * (i + 1) * teeth_len)
                path.lineTo(*t_mid)
                path.lineTo(*t_end)

            path.lineTo(x1, y1)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(12.0)
            items.append(item)

        elif "capacitor" in comp or "tụ điện" in comp:
            # Parallel plates
            lead_len = (length - 12.0) / 2.0
            p_in = (x0 + ux * lead_len, y0 + uy * lead_len)
            p_out = (x1 - ux * lead_len, y1 - uy * lead_len)

            path = QPainterPath()
            path.moveTo(x0, y0)
            path.lineTo(*p_in)
            path.moveTo(p_in[0] + px * 16.0, p_in[1] + py * 16.0)
            path.lineTo(p_in[0] - px * 16.0, p_in[1] - py * 16.0)

            path.moveTo(p_out[0] + px * 16.0, p_out[1] + py * 16.0)
            path.lineTo(p_out[0] - px * 16.0, p_out[1] - py * 16.0)
            path.moveTo(*p_out)
            path.lineTo(x1, y1)

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(12.0)
            items.append(item)

        elif "battery" in comp or "nguồn" in comp:
            # Long thin plate (positive) and short thick plate (negative)
            lead_len = (length - 12.0) / 2.0
            p_in = (x0 + ux * lead_len, y0 + uy * lead_len)
            p_out = (x1 - ux * lead_len, y1 - uy * lead_len)

            path = QPainterPath()
            path.moveTo(x0, y0)
            path.lineTo(*p_in)
            # Long plate
            path.moveTo(p_in[0] + px * 20.0, p_in[1] + py * 20.0)
            path.lineTo(p_in[0] - px * 20.0, p_in[1] - py * 20.0)

            # Short plate
            path.moveTo(p_out[0] + px * 10.0, p_out[1] + py * 10.0)
            path.lineTo(p_out[0] - px * 10.0, p_out[1] - py * 10.0)
            path.moveTo(*p_out)
            path.lineTo(x1, y1)

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(12.0)
            items.append(item)

        if label:
            txt = QGraphicsTextItem(label)
            txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            txt.setDefaultTextColor(color)
            txt.setPos(mid_x + px * 14.0 - 10, mid_y + py * 14.0 - 10)
            txt.setZValue(13.0)
            items.append(txt)

        return items


    @staticmethod
    def create_michelson_interferometer_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'HE_NE_CIRCULAR',
        mode: str = 'circular',
        source_name: str = 'Laser He-Ne Đỏ (632.8 nm)',
        wavelength_nm: float = 632.8,
        delta_d_um: float = 10.0,
        visibility_pct: float = 100.0,
        fringe_count: float = 31.6,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Michelson Interferometer & LIGO Gravitational Wave Detector."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'QUANG HỌC & THIÊN VĂN: GIAO THOA KẾ MICHELSON & SÓNG HẤP DẪN LIGO – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 42.0
        vw = 320.0
        vh = 200.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        bx_mid = vx0 + vw / 2.0
        by_mid = vy0 + vh / 2.0

        # Beam Splitter (45 deg)
        bs_line = QGraphicsLineItem(bx_mid - 15.0, by_mid - 15.0, bx_mid + 15.0, by_mid + 15.0)
        bs_line.setPen(QPen(QColor('#38bdf8'), 2.5))
        bs_line.setZValue(10.4)
        items.append(bs_line)

        # Mirror M1 (Top)
        m1 = QGraphicsLineItem(bx_mid - 25.0, vy0 + 20.0, bx_mid + 25.0, vy0 + 20.0)
        m1.setPen(QPen(QColor('#f8fafc'), 3.0))
        m1.setZValue(10.4)
        items.append(m1)

        m1_lbl = QGraphicsTextItem('Gương M₁ (Cố định)')
        m1_lbl.setFont(QFont('Segoe UI', 6))
        m1_lbl.setDefaultTextColor(QColor('#94a3b8'))
        m1_lbl.setPos(bx_mid - 35.0, vy0 + 6.0)
        m1_lbl.setZValue(10.5)
        items.append(m1_lbl)

        # Mirror M2 (Right)
        m2_x = vx0 + vw - 25.0
        m2 = QGraphicsLineItem(m2_x, by_mid - 25.0, m2_x, by_mid + 25.0)
        m2.setPen(QPen(QColor('#f59e0b'), 3.0))
        m2.setZValue(10.4)
        items.append(m2)

        m2_lbl = QGraphicsTextItem('Gương M₂\n(Di động)')
        m2_lbl.setFont(QFont('Segoe UI', 6))
        m2_lbl.setDefaultTextColor(QColor('#f59e0b'))
        m2_lbl.setPos(m2_x - 35.0, by_mid + 28.0)
        m2_lbl.setZValue(10.5)
        items.append(m2_lbl)

        # Light Source (Left)
        src = QGraphicsRectItem(vx0 + 15.0, by_mid - 12.0, 30.0, 24.0)
        src.setBrush(QBrush(QColor('#ef4444')))
        src.setPen(QPen(QColor('#ffffff'), 1.0))
        src.setZValue(10.4)
        items.append(src)

        # Laser Ray Paths
        ray_pen = QPen(QColor('#ef4444'), 1.8)
        # Source to BS
        r1 = QGraphicsLineItem(vx0 + 45.0, by_mid, bx_mid, by_mid)
        r1.setPen(ray_pen)
        r1.setZValue(10.3)
        items.append(r1)

        # BS to M1
        r2 = QGraphicsLineItem(bx_mid, by_mid, bx_mid, vy0 + 20.0)
        r2.setPen(ray_pen)
        r2.setZValue(10.3)
        items.append(r2)

        # BS to M2
        r3 = QGraphicsLineItem(bx_mid, by_mid, m2_x, by_mid)
        r3.setPen(ray_pen)
        r3.setZValue(10.3)
        items.append(r3)

        # BS to Detector
        r4 = QGraphicsLineItem(bx_mid, by_mid, bx_mid, vy0 + vh - 20.0)
        r4.setPen(QPen(QColor('#ef4444'), 2.2))
        r4.setZValue(10.3)
        items.append(r4)

        # Detector (Bottom)
        det = QGraphicsRectItem(bx_mid - 25.0, vy0 + vh - 18.0, 50.0, 10.0)
        det.setBrush(QBrush(QColor('#22c55e')))
        det.setPen(QPen(QColor('#ffffff'), 1.0))
        det.setZValue(10.4)
        items.append(det)

        # Right Panel: Interference pattern and metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Concentric rings simulation
        ring_cx = rx0 + 60.0
        ring_cy = vy0 + 60.0
        for rad in [10.0, 20.0, 30.0, 40.0, 50.0]:
            ring = QGraphicsEllipseItem(ring_cx - rad, ring_cy - rad, rad * 2.0, rad * 2.0)
            ring.setPen(QPen(QColor('#ef4444'), 1.5))
            ring.setBrush(Qt.BrushStyle.NoBrush)
            ring.setZValue(10.3)
            items.append(ring)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ GIAO THOA:\n'
            f'• Nguồn: {source_name}\n'
            f'• Bước sóng λ: {wavelength_nm:.1f} nm\n'
            f'• Hiệu quang trình: {2.0 * delta_d_um:+.2f} μm\n'
            f'• Độ lệch vị trí M₂: {delta_d_um:+.2f} μm\n'
            f'• Số vân dịch chuyển: {fringe_count:+.1f} vân\n'
            f'• Độ tương phản V: {visibility_pct:.1f}%\n'
            f'• Chế độ: {mode.upper()}'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 120.0, vy0 + 15.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Giao thoa kế Michelson: Tách biên độ sóng ánh sáng thành hai chùm tia trực giao. Hiệu quang trình Δ = 2(d₂ - d₁) '
            'quy định hệ vân giao thoa (vân tròn đồng tâm Haidinger hoặc vân nêm Fizeau). Đài quan sát LIGO ứng dụng nguyên lý này '
            'với cánh tay 4 km để bắt sóng hấp dẫn với độ nhạy biến dạng cỡ h ~ 10⁻²¹.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


# ─── Vector Tools for Chemistry ───────────────────────────────────────────────


    @staticmethod
    def create_photoelectric_effect_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CESIUM_VISIBLE_EMISSION',
        metal_name: str = 'Xesi (Cs)',
        work_function_ev: float = 2.14,
        wavelength_nm: float = 500.0,
        stopping_voltage_v: float = 0.34,
        is_emitted: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Photoelectric Effect phototube."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HIỆN TƯỢNG QUANG ĐIỆN NGOÀI EINSTEIN – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        # Vacuum Phototube
        tube = QGraphicsRectItem(x0 + 30.0, y0 + 50.0, 240.0, 130.0)
        tube.setBrush(QBrush(QColor(15, 23, 42, 100)))
        tube.setPen(QPen(QColor('#38bdf8'), 1.5))
        tube.setZValue(10.2)
        items.append(tube)

        # Cathode
        cat = QGraphicsRectItem(x0 + 55.0, y0 + 75.0, 12.0, 80.0)
        cat.setBrush(QBrush(QColor('#f59e0b')))
        cat.setPen(QPen(QColor('#ffffff'), 1.0))
        cat.setZValue(10.4)
        items.append(cat)

        # Anode
        an = QGraphicsRectItem(x0 + 230.0, y0 + 75.0, 12.0, 80.0)
        an.setBrush(QBrush(QColor('#64748b')))
        an.setPen(QPen(QColor('#ffffff'), 1.0))
        an.setZValue(10.4)
        items.append(an)

        # Photon Beam
        beam = QGraphicsLineItem(x0 + 15.0, y0 + 115.0, x0 + 55.0, y0 + 115.0)
        beam.setPen(QPen(QColor('#22c55e'), 2.5))
        beam.setZValue(10.3)
        items.append(beam)

        info = QGraphicsTextItem(
            f'THÔNG SỐ QUANG ĐIỆN:\n'
            f'• Kim loại: {metal_name} | Công thoát: {work_function_ev:.2f} eV\n'
            f'• Bước sóng tới: {wavelength_nm:.1f} nm | Điện thế hãm: {stopping_voltage_v:.2f} V\n'
            f'• Trạng thái: {"Có quang điện (λ ≤ λ₀)" if is_emitted else "Triệt tiêu (λ > λ₀)"}'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 50.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_hohmann_transfer_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'LEO_TO_GEO_TRANSFER',
        r1_km: float = 6678.0,
        r2_km: float = 42164.0,
        delta_v1_kms: float = 2.425,
        delta_v2_kms: float = 1.465,
        transfer_hours: float = 5.26,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hohmann Orbital Transfer."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'CƠ HỌC THIÊN THỂ: CHUYỂN QUỸ ĐẠO HOHMANN – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        ox = x0 + 150.0
        oy = y0 + 160.0

        # Central Body (Earth)
        body = QGraphicsEllipseItem(ox - 18.0, oy - 18.0, 36.0, 36.0)
        body.setBrush(QBrush(QColor('#0284c7')))
        body.setPen(QPen(QColor('#38bdf8'), 1.5))
        body.setZValue(10.3)
        items.append(body)

        # Orbit 1 (LEO)
        o1 = QGraphicsEllipseItem(ox - 45.0, oy - 45.0, 90.0, 90.0)
        o1.setPen(QPen(QColor('#38bdf8'), 1.2, Qt.PenStyle.DashLine))
        o1.setBrush(Qt.BrushStyle.NoBrush)
        o1.setZValue(10.2)
        items.append(o1)

        # Orbit 2 (GEO)
        o2 = QGraphicsEllipseItem(ox - 110.0, oy - 110.0, 220.0, 220.0)
        o2.setPen(QPen(QColor('#f59e0b'), 1.2, Qt.PenStyle.DashLine))
        o2.setBrush(Qt.BrushStyle.NoBrush)
        o2.setZValue(10.2)
        items.append(o2)

        # Transfer Ellipse
        trans = QGraphicsEllipseItem(ox - 110.0, oy - 77.5, 155.0, 155.0)
        trans.setPen(QPen(QColor('#22c55e'), 1.8))
        trans.setBrush(Qt.BrushStyle.NoBrush)
        trans.setZValue(10.3)
        items.append(trans)

        info = QGraphicsTextItem(
            f'THÔNG SỐ QUỸ ĐẠO:\n'
            f'• Bán kính r₁: {r1_km:,.0f} km | Bán kính r₂: {r2_km:,.0f} km\n'
            f'• Δv₁: {delta_v1_kms:.3f} km/s | Δv₂: {delta_v2_kms:.3f} km/s\n'
            f'• Tổng Δv: {delta_v1_kms + delta_v2_kms:.3f} km/s\n'
            f'• Thời gian bay: {transfer_hours:.2f} giờ'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_quantum_well_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'GROUND_STATE_N1',
        n_quantum: int = 1,
        well_width_nm: float = 1.0,
        energy_ev: float = 0.376,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of 1D Quantum Well and Schrödinger Wavefunction."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'CƠ HỌC LƯỢNG TỬ: GIẾNG THẾ VÔ HẠN 1D (SCHRÖDINGER) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        wx0 = x0 + 40.0
        wy0 = y0 + 60.0
        ww = 220.0
        wh = 140.0

        # Left Barrier (V = infinity)
        b_l = QGraphicsLineItem(wx0, wy0, wx0, wy0 + wh)
        b_l.setPen(QPen(QColor('#38bdf8'), 3.0))
        b_l.setZValue(10.3)
        items.append(b_l)

        # Right Barrier
        b_r = QGraphicsLineItem(wx0 + ww, wy0, wx0 + ww, wy0 + wh)
        b_r.setPen(QPen(QColor('#38bdf8'), 3.0))
        b_r.setZValue(10.3)
        items.append(b_r)

        # Well Floor
        floor = QGraphicsLineItem(wx0, wy0 + wh, wx0 + ww, wy0 + wh)
        floor.setPen(QPen(QColor('#475569'), 1.5))
        floor.setZValue(10.2)
        items.append(floor)

        # Standing Wavefunction psi_n(x)
        wave_path = QPainterPath()
        wave_y = wy0 + wh - 40.0
        wave_path.moveTo(wx0, wave_y)
        for i in range(50):
            frac = i / 49.0
            px = wx0 + frac * ww
            py = wave_y - 30.0 * math.sin(n_quantum * math.pi * frac)
            wave_path.lineTo(px, py)
        wave_item = QGraphicsPathItem(wave_path)
        wave_item.setPen(QPen(QColor('#a855f7'), 2.0))
        wave_item.setZValue(10.4)
        items.append(wave_item)

        info = QGraphicsTextItem(
            f'THÔNG SỐ TRẠNG THÁI DỪNG:\n'
            f'• Mức lượng tử n = {n_quantum} | Bề rộng giếng L = {well_width_nm:.2f} nm\n'
            f'• Năng lượng lượng tử hóa E_n: {energy_ev:.4f} eV\n'
            f'• Hàm sóng: ψ_n(x) = √(2/L) · sin(nπx/L)'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 300.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_cyclotron_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CLASSICAL_PROTON_ACCELERATOR',
        particle_name: str = 'Proton (¹H⁺)',
        b_field_tesla: float = 1.5,
        dee_voltage_kv: float = 50.0,
        freq_mhz: float = 22.88,
        max_energy_mev: float = 16.5,
        r_max_m: float = 0.8,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Cyclotron Particle Accelerator."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'VẬT LÝ HẠT NHÂN: MÁY GIA TỐC CYCLOTRON (LAWRENCE 1939) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 310.0
        vh = 195.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        bx = vx0 + vw / 2.0
        by = vy0 + vh / 2.0
        r_dee = 75.0

        # Left Dee (D1)
        dee_l = QGraphicsEllipseItem(bx - r_dee - 4.0, by - r_dee, 2 * r_dee, 2 * r_dee)
        dee_l.setSpanAngle(180 * 16)
        dee_l.setStartAngle(90 * 16)
        dee_l.setBrush(QBrush(QColor(15, 23, 42)))
        dee_l.setPen(QPen(QColor('#0284c7'), 2.0))
        dee_l.setZValue(10.3)
        items.append(dee_l)

        # Right Dee (D2)
        dee_r = QGraphicsEllipseItem(bx - r_dee + 4.0, by - r_dee, 2 * r_dee, 2 * r_dee)
        dee_r.setSpanAngle(180 * 16)
        dee_r.setStartAngle(-90 * 16)
        dee_r.setBrush(QBrush(QColor(15, 23, 42)))
        dee_r.setPen(QPen(QColor('#0284c7'), 2.0))
        dee_r.setZValue(10.3)
        items.append(dee_r)

        # Spiral path
        spiral_path = QPainterPath()
        spiral_path.moveTo(bx, by)
        for i in range(1, 45):
            rad = i * 1.5
            angle = i * 0.4
            sx = bx + rad * math.cos(angle)
            sy = by + rad * math.sin(angle)
            spiral_path.lineTo(sx, sy)

        sp_item = QGraphicsPathItem(spiral_path)
        sp_item.setPen(QPen(QColor('#ef4444'), 1.8))
        sp_item.setZValue(10.4)
        items.append(sp_item)

        # Right box: Readouts & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info = QGraphicsTextItem(
            f'THÔNG SỐ GIA TỐC CYCLOTRON:\n'
            f'• Hạt tích điện: {particle_name}\n'
            f'• Cảm ứng từ B: {b_field_tesla:.2f} T (vuông góc mặt phẳng)\n'
            f'• Điện thế cực Dees: {dee_voltage_kv:.1f} kV\n'
            f'• Tần số cộng hưởng f_c: {freq_mhz:.2f} MHz\n'
            f'• Bán kính buồng R_max: {r_max_m:.2f} m\n'
            f'• Động năng cực đại tại bia: {max_energy_mev:.2f} MeV'
        )
        info.setFont(QFont('Segoe UI', 7))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(rx0 + 10.0, vy0 + 12.0)
        info.setZValue(10.4)
        items.append(info)

        cap = QGraphicsTextItem(
            'Máy gia tốc Cyclotron: Sử dụng từ trường tĩnh uốn hạt theo quỹ đạo xoắn ốc và điện trường xoay chiều '
            'tần số cao gia tốc hạt mỗi khi qua khe hở giữa 2 cực Dees. Ở vận tốc lớn (v > 0.1c), hiệu ứng tương đối tính '
            'làm tăng khối lượng m = γ·m₀, đòi hỏi biến điệu tần số RF (Synchrocyclotron).'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


    @staticmethod
    def create_electromagnetic_wave_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "LINEAR_POLARIZATION_45_DEG",
        e0x: float = 100.0,
        e0y: float = 100.0,
        delta_rad: float = 0.0,
        wavelength_nm: float = 532.0,
        intensity_w_m2: float = 26.54,
        b0_ut: float = 0.472,
        rad_pressure_upa: float = 0.089,
        polarization_name: str = "Tuyến tính (Linear 45°)",
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of transverse Electromagnetic Waves, Poynting vector, and Polarization."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'VẬT LÝ & ĐIỆN ĐỘNG LỰC HỌC: SÓNG ĐIỆN TỪ & VECTƠ POYNTING – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 240.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # 3D Propagation Axis +z
        axis_y = vy0 + vh * 0.55
        z_line = QGraphicsLineItem(vx0 + 30.0, axis_y, vx0 + vw - 20.0, axis_y)
        z_line.setPen(QPen(QColor('#94a3b8'), 1.5))
        z_line.setZValue(10.3)
        items.append(z_line)

        k_lbl = QGraphicsTextItem('k⃗ (+z)')
        k_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        k_lbl.setDefaultTextColor(QColor('#f59e0b'))
        k_lbl.setPos(vx0 + vw - 35.0, axis_y - 18.0)
        k_lbl.setZValue(10.4)
        items.append(k_lbl)

        # Sinusoidal E-field path (Vertical/Transverse, Red)
        e_path = QPainterPath()
        e_path.moveTo(vx0 + 40.0, axis_y)
        num_pts = 60
        span = vw - 80.0
        for i in range(num_pts + 1):
            px = vx0 + 40.0 + (i / float(num_pts)) * span
            phase = (i / float(num_pts)) * 4.0 * math.pi
            ey = axis_y - 50.0 * math.cos(phase)
            e_path.lineTo(px, ey)
        e_item = QGraphicsPathItem(e_path)
        e_item.setPen(QPen(QColor('#ef4444'), 2.2))
        e_item.setZValue(10.4)
        items.append(e_item)

        # Sinusoidal B-field path (Orthogonal Transverse, Cyan)
        b_path = QPainterPath()
        b_path.moveTo(vx0 + 40.0, axis_y)
        for i in range(num_pts + 1):
            px = vx0 + 40.0 + (i / float(num_pts)) * span
            phase = (i / float(num_pts)) * 4.0 * math.pi
            # Tilted 3D horizontal perspective
            by = axis_y + 28.0 * math.cos(phase)
            bx = px + 15.0 * math.cos(phase)
            b_path.lineTo(bx, by)
        b_item = QGraphicsPathItem(b_path)
        b_item.setPen(QPen(QColor('#38bdf8'), 1.8, Qt.PenStyle.DashLine))
        b_item.setZValue(10.3)
        items.append(b_item)

        # Poynting Vector Arrow (Gold, along propagation direction)
        s_arr = QGraphicsLineItem(vx0 + vw * 0.45, axis_y - 25.0, vx0 + vw * 0.45 + 35.0, axis_y - 25.0)
        s_arr.setPen(QPen(QColor('#f59e0b'), 2.2))
        s_arr.setZValue(10.5)
        items.append(s_arr)

        s_lbl = QGraphicsTextItem('S⃗ = (1/μ₀)(E⃗ × B⃗)')
        s_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor('#f59e0b'))
        s_lbl.setPos(vx0 + vw * 0.45 - 15.0, axis_y - 42.0)
        s_lbl.setZValue(10.6)
        items.append(s_lbl)

        # Legend
        leg_e = QGraphicsTextItem('■ E⃗: Điện trường')
        leg_e.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        leg_e.setDefaultTextColor(QColor('#ef4444'))
        leg_e.setPos(vx0 + 10.0, vy0 + 8.0)
        leg_e.setZValue(10.5)
        items.append(leg_e)

        leg_b = QGraphicsTextItem('■ B⃗: Từ trường')
        leg_b.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        leg_b.setDefaultTextColor(QColor('#38bdf8'))
        leg_b.setPos(vx0 + 100.0, vy0 + 8.0)
        leg_b.setZValue(10.5)
        items.append(leg_b)

        # Right Panel: Polarization Screen & Stokes Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Transverse Screen Crosshair
        sc_cx = rx0 + rw / 2.0
        sc_cy = vy0 + 60.0
        ch_h = QGraphicsLineItem(sc_cx - 40.0, sc_cy, sc_cx + 40.0, sc_cy)
        ch_h.setPen(QPen(QColor('#334155'), 1.0, Qt.PenStyle.DashLine))
        ch_h.setZValue(10.3)
        items.append(ch_h)

        ch_v = QGraphicsLineItem(sc_cx, sc_cy - 40.0, sc_cx, sc_cy + 40.0)
        ch_v.setPen(QPen(QColor('#334155'), 1.0, Qt.PenStyle.DashLine))
        ch_v.setZValue(10.3)
        items.append(ch_v)

        # Polarization Orbit Path (Circle, Line, or Ellipse)
        orb_path = QPainterPath()
        for deg in range(361):
            th = math.radians(deg)
            ox = sc_cx + 30.0 * math.cos(th + delta_rad)
            oy = sc_cy - 30.0 * math.cos(th)
            if deg == 0:
                orb_path.moveTo(ox, oy)
            else:
                orb_path.lineTo(ox, oy)
        orb_item = QGraphicsPathItem(orb_path)
        orb_item.setPen(QPen(QColor('#a855f7'), 2.0))
        orb_item.setZValue(10.4)
        items.append(orb_item)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ SÓNG ĐIỆN TỪ MAXWELL:\n'
            f'• Bước sóng: {wavelength_nm:.1f} nm | Tần số: {299792458.0 / (wavelength_nm * 1e-9) * 1e-12:.1f} THz\n'
            f'• Biên độ điện trường: E₀ = {math.hypot(e0x, e0y):.1f} V/m\n'
            f'• Cảm ứng từ B₀ = E₀/c: {b0_ut:.3f} μT\n'
            f'• Độ lệch pha δ: {math.degrees(delta_rad):.1f}°\n'
            f'• Trạng thái phân cực: {polarization_name}\n'
            f'• Cường độ Poynting <S>: {intensity_w_m2:.2f} W/m²\n'
            f'• Áp suất bức xạ P_rad: {rad_pressure_upa:.3f} μPa'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 115.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Hệ phương trình Maxwell khẳng định sóng điện từ lan truyền trong chân không với vận tốc ánh sáng c = 1/√(ε₀μ₀). '
            'Vectơ điện E⃗ và từ B⃗ luôn dao động vuông góc với nhau và vuông góc với phương truyền sóng k⃗. '
            'Vectơ Poynting S⃗ = (1/μ₀)(E⃗ × B⃗) định lượng dòng năng lượng và áp suất bức xạ truyền tải trên mỗi đơn vị diện tích.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items

    @staticmethod
    def create_fabry_perot_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HELIUM_NEON_LASER_CAVITY",
        wavelength_nm: float = 632.8,
        cavity_d_um: float = 20.0,
        reflectance_r: float = 0.96,
        finesse: float = 76.8,
        fsr_ghz: float = 7494.8,
        fwhm_ghz: float = 97.6,
        transmittance: float = 1.0,
        order_m: float = 63.2,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of a Fabry-Pérot Cavity & Airy Transmission Spectrum.
        Includes dual semi-reflective dielectric mirrors, multi-beam internal bounces,
        transmitted interference rays, and plotted Airy resonance curve with FSR and FWHM metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"GIAO THOA KẾ FABRY-PÉROT & HỐC CỘNG HƯỞNG LASER: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Bước sóng λ = {wavelength_nm:.1f} nm | Khoảng cách gương d = {cavity_d_um:.1f} μm | "
            f"Phản xạ R = {reflectance_r:.3f} | Finesse F = {finesse:.1f} | FSR = {fsr_ghz:.1f} GHz"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Cavity & Multi-beam reflections
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("HỐC QUANG HỌC & ĐA PHẢN XẠ NHIỀU CHÙM TIA:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Mirrors M1 & M2
        m1_x = cx + cw * 0.28
        m2_x = cx + cw * 0.74
        m_top = cy + 30.0
        m_h = ch - 50.0
        m_thick = 12.0

        m1 = QGraphicsRectItem(m1_x - m_thick, m_top, m_thick, m_h)
        m1.setBrush(QBrush(QColor(56, 189, 248, 180)))
        m1.setPen(QPen(QColor(148, 163, 184), 1.2))
        m1.setZValue(10.3)
        items.append(m1)

        m2 = QGraphicsRectItem(m2_x, m_top, m_thick, m_h)
        m2.setBrush(QBrush(QColor(56, 189, 248, 180)))
        m2.setPen(QPen(QColor(148, 163, 184), 1.2))
        m2.setZValue(10.3)
        items.append(m2)

        # Incident Beam
        beam_y = m_top + 35.0
        in_line = QGraphicsLineItem(cx + 10.0, beam_y, m1_x, beam_y)
        in_line.setPen(QPen(QColor("#ef4444" if wavelength_nm > 600 else "#22c55e"), 2.2))
        in_line.setZValue(10.3)
        items.append(in_line)

        # Internal Multi-reflection path
        b_path = QPainterPath()
        b_path.moveTo(m1_x, beam_y)
        b_path.lineTo(m2_x, beam_y + 20.0)
        b_path.lineTo(m1_x, beam_y + 40.0)
        b_path.lineTo(m2_x, beam_y + 60.0)
        b_path.lineTo(m1_x, beam_y + 80.0)
        b_item = QGraphicsPathItem(b_path)
        b_item.setPen(QPen(QColor("#fca5a5" if wavelength_nm > 600 else "#86efac"), 1.8))
        b_item.setZValue(10.3)
        items.append(b_item)

        # Transmitted output rays
        for ty in [beam_y + 20.0, beam_y + 60.0]:
            t_line = QGraphicsLineItem(m2_x + m_thick, ty, cx + cw - 10.0, ty)
            t_line.setPen(QPen(QColor("#ef4444" if wavelength_nm > 600 else "#22c55e"), 1.6))
            t_line.setZValue(10.3)
            items.append(t_line)

        # 4. Right Panel: Airy Transmission Curve
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("ĐƯỜNG TRUYỀN QUA HÀM AIRY T(δ) & VẠCH CỘNG HƯỞNG:")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Airy curve plot
        px = rx + 30.0
        py = ry + 25.0
        pw = rw - 45.0
        ph = rh - 85.0

        # Plot axes
        axis_x = QGraphicsLineItem(px, py + ph, px + pw, py + ph)
        axis_x.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_x.setZValue(10.2)
        items.append(axis_x)

        axis_y = QGraphicsLineItem(px, py, px, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Calculate Airy path over 3 FSR periods
        f_coeff = (4.0 * reflectance_r) / max(1e-4, (1.0 - reflectance_r) ** 2)
        a_path = QPainterPath()
        steps = int(pw)
        delta_span = 6.0 * 3.14159265
        for s in range(steps + 1):
            delta = -3.0 * 3.14159265 + (s / float(steps)) * delta_span
            t_val = 1.0 / (1.0 + f_coeff * (math.sin(delta / 2.0) ** 2))
            gx = px + s
            gy = py + ph - t_val * ph
            if s == 0:
                a_path.moveTo(gx, gy)
            else:
                a_path.lineTo(gx, gy)

        a_item = QGraphicsPathItem(a_path)
        a_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        a_item.setZValue(10.4)
        items.append(a_item)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Finesse F = π√R/(1-R): {finesse:.1f}    • Hệ số F = 4R/(1-R)²: {f_coeff:.1f}    • Khoảng phổ tự do FSR: {fsr_ghz:.1f} GHz\n"
            f"• Bề rộng nửa cực đại FWHM δν: {fwhm_ghz:.2f} GHz    • Bậc mode dọc m: {order_m:.1f}    • Độ truyền qua cực đại: {transmittance*100:.1f}%"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Hàm truyền qua Airy T(δ) = 1 / [1 + F sin²(δ/2)] mô tả giao thoa đa chùm tia phản xạ giữa hai gương song song. "
            "Khi phản xạ R tiến gần 1.0, độ sắc nét Finesse F tăng vọt, các đỉnh truyền qua trở nên siêu hẹp (FWHM << FSR), "
            "cho phép chọn lọc chính xác các mode dọc quang học trong buồng cộng hưởng laser và phân giải siêu cao phổ vạch nguyên tử."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_photoelectric_effect_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CESIUM_VISIBLE_LIGHT_THRESHOLD",
        metal_name: str = "Xesi (Cs)",
        work_function_ev: float = 2.14,
        wavelength_nm: float = 532.0,
        photon_energy_ev: float = 2.33,
        stopping_potential_v: float = 0.19,
        voltage_uak_v: float = 1.0,
        photocurrent_ua: float = 5.0,
        red_threshold_nm: float = 579.4,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Einstein's Photoelectric Effect.
        Includes evacuated quartz phototube, cathode metal plate, anode ring, light rays,
        photoelectron trajectories, current-voltage I-U_AK curve, and quantum metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"HIỆU ỨNG QUANG ĐIỆN EINSTEIN & THUYẾT LƯỢNG TỬ: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Catot: {metal_name} (Φ = {work_function_ev:.2f} eV) | λ = {wavelength_nm:.1f} nm (E = {photon_energy_ev:.2f} eV) | "
            f"Thế hãm U_h = {stopping_potential_v:.2f} V | U_AK = {voltage_uak_v:+.2f} V | I = {photocurrent_ua:.2f} μA"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Quartz Vacuum Phototube
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("TẾ BÀO QUANG ĐIỆN CHÂN KHÔNG:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Quartz bulb
        bulb = QGraphicsRectItem(cx + 25.0, cy + 28.0, cw - 50.0, ch - 56.0)
        bulb.setBrush(QBrush(QColor(15, 23, 42, 120)))
        bulb.setPen(QPen(QColor(56, 189, 248, 160), 1.5))
        bulb.setZValue(10.2)
        items.append(bulb)

        # Cathode Plate
        cat_x = cx + 45.0
        cat_y = cy + 45.0
        cat_w = 8.0
        cat_h = ch - 90.0
        cat = QGraphicsRectItem(cat_x, cat_y, cat_w, cat_h)
        cat.setBrush(QBrush(QColor(250, 204, 21)))
        cat.setPen(QPen(QColor(248, 250, 252), 1.2))
        cat.setZValue(10.3)
        items.append(cat)

        cat_lbl = QGraphicsTextItem("Catot (K)")
        cat_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        cat_lbl.setDefaultTextColor(QColor("#e2e8f0"))
        cat_lbl.setPos(cat_x - 8.0, cat_y - 14.0)
        cat_lbl.setZValue(10.3)
        items.append(cat_lbl)

        # Anode Ring
        an_x = cx + cw - 55.0
        an = QGraphicsEllipseItem(an_x, cat_y + 10.0, 10.0, cat_h - 20.0)
        an.setBrush(Qt.BrushStyle.NoBrush)
        an.setPen(QPen(QColor(148, 163, 184), 2.0))
        an.setZValue(10.3)
        items.append(an)

        an_lbl = QGraphicsTextItem("Anot (A)")
        an_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        an_lbl.setDefaultTextColor(QColor("#e2e8f0"))
        an_lbl.setPos(an_x - 4.0, cat_y - 14.0)
        an_lbl.setZValue(10.3)
        items.append(an_lbl)

        # Light Rays
        ray_color = QColor("#22c55e" if wavelength_nm > 500 else "#a855f7")
        for ry in [cat_y + 15.0, cat_y + 40.0, cat_y + 65.0]:
            ray = QGraphicsLineItem(cx + 8.0, ry - 15.0, cat_x, ry)
            ray.setPen(QPen(ray_color, 1.5, Qt.PenStyle.DashLine))
            ray.setZValue(10.3)
            items.append(ray)

        # Photoelectrons
        if photon_energy_ev >= work_function_ev:
            for ex, ey in [(cat_x + 30.0, cat_y + 25.0), (cat_x + 70.0, cat_y + 35.0), (cat_x + 110.0, cat_y + 20.0), (cat_x + 150.0, cat_y + 45.0)]:
                elec = QGraphicsEllipseItem(ex, ey, 5.0, 5.0)
                elec.setBrush(QBrush(QColor("#38bdf8")))
                elec.setPen(Qt.PenStyle.NoPen)
                elec.setZValue(10.4)
                items.append(elec)

        # Microammeter badge
        meter_bg = QGraphicsRectItem(cx + cw - 95.0, cy + ch - 24.0, 85.0, 18.0)
        meter_bg.setBrush(QBrush(QColor(9, 14, 25)))
        meter_bg.setPen(QPen(QColor(56, 189, 248), 1.0))
        meter_bg.setZValue(10.3)
        items.append(meter_bg)

        meter_txt = QGraphicsTextItem(f"I: {photocurrent_ua:.2f} μA")
        meter_txt.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        meter_txt.setDefaultTextColor(QColor("#22c55e" if photocurrent_ua > 0 else "#94a3b8"))
        meter_txt.setPos(cx + cw - 92.0, cy + ch - 24.0)
        meter_txt.setZValue(10.4)
        items.append(meter_txt)

        # 4. Right Panel: Current-Voltage Curve I - U_AK
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("ĐƯỜNG ĐẶC TRƯNG VÔN-AMPE I(U_AK):")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Axes
        px = rx + 30.0
        py = ry + 25.0
        pw = rw - 45.0
        ph = rh - 85.0

        axis_x = QGraphicsLineItem(px, py + ph, px + pw, py + ph)
        axis_x.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_x.setZValue(10.2)
        items.append(axis_x)

        zero_x = px + pw * 0.45
        axis_y = QGraphicsLineItem(zero_x, py, zero_x, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Plot I - U_AK curve
        iv_path = QPainterPath()
        steps = int(pw)
        uh_pt_x = zero_x - (stopping_potential_v / 3.0) * (zero_x - px)
        uh_pt_x = max(px, uh_pt_x)

        for s in range(steps + 1):
            gx = px + s
            if gx <= uh_pt_x:
                gy = py + ph
            elif gx < zero_x:
                frac = (gx - uh_pt_x) / max(1.0, (zero_x - uh_pt_x))
                gy = py + ph - (frac ** 2) * (ph * 0.7)
            else:
                gy = py + ph - ph * 0.7
            if s == 0:
                iv_path.moveTo(gx, gy)
            else:
                iv_path.lineTo(gx, gy)

        iv_item = QGraphicsPathItem(iv_path)
        iv_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        iv_item.setZValue(10.4)
        items.append(iv_item)

        # -Uh marker
        uh_line = QGraphicsLineItem(uh_pt_x, py + 10.0, uh_pt_x, py + ph)
        uh_line.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DashLine))
        uh_line.setZValue(10.3)
        items.append(uh_line)

        uh_lbl = QGraphicsTextItem(f"-U_h = -{stopping_potential_v:.2f}V")
        uh_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        uh_lbl.setDefaultTextColor(QColor("#f43f5e"))
        uh_lbl.setPos(uh_pt_x - 18.0, py - 4.0)
        uh_lbl.setZValue(10.3)
        items.append(uh_lbl)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Năng lượng photon E = hc/λ: {photon_energy_ev:.2f} eV    • Công thoát Φ: {work_function_ev:.2f} eV    • Giới hạn quang điện λ₀: {red_threshold_nm:.1f} nm\n"
            f"• Động năng cực đại E_k: {max(0.0, photon_energy_ev - work_function_ev):.2f} eV    • Thế hãm U_h: {stopping_potential_v:.2f} V    • Dòng quang điện I: {photocurrent_ua:.2f} μA"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình lượng tử Einstein E_k = hν - Φ = e·U_h chứng minh bản chất hạt của ánh sáng: mỗi photon truyền toàn bộ năng lượng hν cho một electron. "
            "Hiện tượng quang điện chỉ xảy ra khi năng lượng photon vượt qua công thoát kim loại (hν ≥ Φ), độc lập hoàn toàn với cường độ sáng."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_thin_lens_system_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CONVEX_LENS_REAL_IMAGE",
        f1_mm: float = 100.0,
        do_mm: float = 250.0,
        di_mm: float = 166.7,
        magnification: float = -0.67,
        is_real: bool = True,
        is_two_lens: bool = False,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Thin Lens Geometrical Optics Ray Tracing.
        Includes principal optical axis, convex/concave lens symbols, focal points,
        object arrow, principal rays (parallel and central), real/virtual image arrow, and ABCD matrix metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"QUANG HÌNH HỌC & HỆ THẤU KÍNH MỎNG: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        lens_type_str = "Hội tụ (Convex)" if f1_mm > 0 else "Phân kỳ (Concave)"
        img_type_str = "ẢNH THẬT (Real, Ngược chiều)" if is_real else "ẢNH ẢO (Virtual, Cùng chiều)"
        sub_txt = QGraphicsTextItem(
            f"Thấu kính: {lens_type_str} f = {f1_mm:+.1f} mm | d_o = {do_mm:.1f} mm | d_i = {di_mm:+.1f} mm | "
            f"Độ phóng đại m = {magnification:+.2f} | {img_type_str}"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Ray Tracing Optical Bench Panel
        bw = width - 32.0
        bh = height - 165.0
        bx = x0 + 16.0
        by = y0 + 58.0

        b_bg = QGraphicsRectItem(bx, by, bw, bh)
        b_bg.setBrush(QBrush(QColor(4, 7, 16)))
        b_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        b_bg.setZValue(10.1)
        items.append(b_bg)

        # Principal Axis
        axis_y = by + bh / 2.0
        axis_line = QGraphicsLineItem(bx + 10.0, axis_y, bx + bw - 10.0, axis_y)
        axis_line.setPen(QPen(QColor(71, 85, 105), 1.2, Qt.PenStyle.DashLine))
        axis_line.setZValue(10.2)
        items.append(axis_line)

        # Lens 1 Line & Arrowheads
        scale_x = (bw - 60.0) / 500.0
        lens_x = bx + bw * 0.52
        lens_top = axis_y - 65.0
        lens_bot = axis_y + 65.0

        lens_line = QGraphicsLineItem(lens_x, lens_top, lens_x, lens_bot)
        lens_line.setPen(QPen(QColor("#38bdf8"), 2.4))
        lens_line.setZValue(10.3)
        items.append(lens_line)

        # Arrowheads on Lens
        is_convex = f1_mm > 0
        arr_sz = 6.0
        if is_convex:
            # Arrows pointing out
            a1 = QGraphicsLineItem(lens_x - arr_sz, lens_top + arr_sz, lens_x, lens_top)
            a1.setPen(QPen(QColor("#38bdf8"), 2.0))
            a1.setZValue(10.3)
            items.append(a1)
            a2 = QGraphicsLineItem(lens_x + arr_sz, lens_top + arr_sz, lens_x, lens_top)
            a2.setPen(QPen(QColor("#38bdf8"), 2.0))
            a2.setZValue(10.3)
            items.append(a2)
        else:
            # Arrows pointing inward
            a1 = QGraphicsLineItem(lens_x - arr_sz, lens_top, lens_x, lens_top + arr_sz)
            a1.setPen(QPen(QColor("#38bdf8"), 2.0))
            a1.setZValue(10.3)
            items.append(a1)
            a2 = QGraphicsLineItem(lens_x + arr_sz, lens_top, lens_x, lens_top + arr_sz)
            a2.setPen(QPen(QColor("#38bdf8"), 2.0))
            a2.setZValue(10.3)
            items.append(a2)

        # Focal points F1 and F'1
        f_px = abs(f1_mm) * scale_x * 0.7
        f_left_x = lens_x - f_px
        f_right_x = lens_x + f_px

        dot_f1 = QGraphicsEllipseItem(f_left_x - 3.0, axis_y - 3.0, 6.0, 6.0)
        dot_f1.setBrush(QBrush(QColor("#f43f5e")))
        dot_f1.setPen(Qt.PenStyle.NoPen)
        dot_f1.setZValue(10.4)
        items.append(dot_f1)

        dot_f2 = QGraphicsEllipseItem(f_right_x - 3.0, axis_y - 3.0, 6.0, 6.0)
        dot_f2.setBrush(QBrush(QColor("#f43f5e")))
        dot_f2.setPen(Qt.PenStyle.NoPen)
        dot_f2.setZValue(10.4)
        items.append(dot_f2)

        f_lbl = QGraphicsTextItem(f"F (f={abs(f1_mm):.0f}mm)")
        f_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        f_lbl.setDefaultTextColor(QColor("#f43f5e"))
        f_lbl.setPos(f_left_x - 14.0, axis_y + 4.0)
        f_lbl.setZValue(10.4)
        items.append(f_lbl)

        # Object Arrow AB (Yellow)
        obj_dist_px = do_mm * scale_x * 0.7
        obj_x = lens_x - obj_dist_px
        obj_h_px = 35.0
        obj_tip_y = axis_y - obj_h_px

        obj_line = QGraphicsLineItem(obj_x, axis_y, obj_x, obj_tip_y)
        obj_line.setPen(QPen(QColor("#facc15"), 2.5))
        obj_line.setZValue(10.4)
        items.append(obj_line)

        obj_lbl = QGraphicsTextItem("Vật AB")
        obj_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        obj_lbl.setDefaultTextColor(QColor("#facc15"))
        obj_lbl.setPos(obj_x - 10.0, obj_tip_y - 14.0)
        obj_lbl.setZValue(10.4)
        items.append(obj_lbl)

        # Image Arrow A'B'
        img_dist_px = di_mm * scale_x * 0.7
        img_x = lens_x + img_dist_px
        img_h_px = obj_h_px * abs(magnification)
        img_tip_y = axis_y + (img_h_px if is_real else -img_h_px)

        img_col = QColor("#22c55e" if is_real else "#f43f5e")
        img_line = QGraphicsLineItem(img_x, axis_y, img_x, img_tip_y)
        img_line.setPen(QPen(img_col, 2.5, Qt.PenStyle.SolidLine if is_real else Qt.PenStyle.DashLine))
        img_line.setZValue(10.4)
        items.append(img_line)

        img_lbl = QGraphicsTextItem("Ảnh A'B'")
        img_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        img_lbl.setDefaultTextColor(img_col)
        img_lbl.setPos(img_x - 12.0, img_tip_y + (4.0 if is_real else -14.0))
        img_lbl.setZValue(10.4)
        items.append(img_lbl)

        # Ray 1: Parallel ray refracts through F'1
        r1_a = QGraphicsLineItem(obj_x, obj_tip_y, lens_x, obj_tip_y)
        r1_a.setPen(QPen(QColor("#ef4444"), 1.6))
        r1_a.setZValue(10.3)
        items.append(r1_a)

        r1_b = QGraphicsLineItem(lens_x, obj_tip_y, img_x, img_tip_y)
        r1_b.setPen(QPen(QColor("#ef4444"), 1.6))
        r1_b.setZValue(10.3)
        items.append(r1_b)

        # Ray 2: Central ray through optical center O
        r2 = QGraphicsLineItem(obj_x, obj_tip_y, img_x, img_tip_y)
        r2.setPen(QPen(QColor("#22c55e"), 1.6))
        r2.setZValue(10.3)
        items.append(r2)

        # 4. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Công thức Gauss: 1/f = 1/d_o + 1/d_i    • Độ phóng đại: m = -d_i / d_o = {magnification:+.2f}    • Độ tụ: D = 1/f = {1000.0/f1_mm:+.1f} dp\n"
            f"• Khoảng cách vật d_o: {do_mm:.1f} mm    • Khoảng cách ảnh d_i: {di_mm:+.1f} mm    • Ma trận ABCD: det(M) = AD - BC = 1.000"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(bx + 8.0, by + bh + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 5. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Công thức thấu kính mỏng Gauss 1/f = 1/d_o + 1/d_i và ma trận truyền tia ABCD cho phép xác định chính xác vị trí, độ lớn và tính chất ảnh. "
            "Thấu kính hội tụ (f > 0) cho ảnh thật ngược chiều khi d_o > f và ảnh ảo phóng đại khi d_o < f. Thấu kính phân kỳ (f < 0) luôn tạo ảnh ảo cùng chiều thu nhỏ."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_bragg_diffraction_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "SILICON_POWDER_STANDARD",
        crystal_type: str = "DIAMOND_CUBIC",
        lattice_a_nm: float = 0.5431,
        wavelength_nm: float = 0.15406,
        crystallite_d_nm: float = 45.0,
        plane_label: str = "(111)",
        d_spacing_nm: float = 0.3135,
        bragg_theta_deg: float = 14.22,
        two_theta_deg: float = 28.44,
        scherrer_fwhm_deg: float = 0.22,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Bragg's Law & XRD Crystallography.
        Includes atomic crystal lattice planes, incident/reflected X-ray rays, geometric path difference Delta = 2d*sin(theta),
        XRD powder diffractogram spectrum I(2theta), and Debye-Scherrer crystallite size metrics.
        """
        import math
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"NHIỄU XẠ TIA X BRAGG & TINH THỂ HỌC: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Mạng: {crystal_type} | a = {lattice_a_nm*10.0:.3f} Å | Mặt {plane_label}: d = {d_spacing_nm*10.0:.3f} Å | "
            f"Góc Bragg: θ_B = {bragg_theta_deg:.2f}° (2θ = {two_theta_deg:.2f}°) | Scherrer: D = {crystallite_d_nm:.1f} nm"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Crystal Lattice & Ray Reflection Geometry
        pw = (width - 40.0) / 2.0
        ph = height - 165.0
        p1_x = x0 + 16.0
        p_y = y0 + 58.0

        b_bg = QGraphicsRectItem(p1_x, p_y, pw, ph)
        b_bg.setBrush(QBrush(QColor(4, 7, 16)))
        b_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        b_bg.setZValue(10.1)
        items.append(b_bg)

        p1_lbl = QGraphicsTextItem("MẶT MẠNG & QUANG TRÌNH Δ = 2d·sin θ")
        p1_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # 3 Crystal Planes
        plane_y1 = p_y + 65.0
        d_vis = 48.0
        plane_y2 = plane_y1 + d_vis
        plane_y3 = plane_y2 + d_vis
        atom_cx = p1_x + pw * 0.50

        for r_idx, ry in enumerate([plane_y1, plane_y2, plane_y3]):
            line = QGraphicsLineItem(p1_x + 12.0, ry, p1_x + pw - 12.0, ry)
            line.setPen(QPen(QColor(51, 65, 85, 160), 1.0, Qt.PenStyle.DashLine))
            line.setZValue(10.2)
            items.append(line)

            # Draw atoms
            for col in range(-3, 4):
                ax = atom_cx + col * 36.0
                atom = QGraphicsEllipseItem(ax - 4.5, ry - 4.5, 9.0, 9.0)
                atom.setBrush(QBrush(QColor("#38bdf8") if (col == 0 and r_idx in (0, 1)) else QColor("#64748b")))
                atom.setPen(QPen(QColor("#000000"), 0.8))
                atom.setZValue(10.3)
                items.append(atom)

        # Dimension line for d
        dim_x = p1_x + pw - 24.0
        dim_line = QGraphicsLineItem(dim_x, plane_y1, dim_x, plane_y2)
        dim_line.setPen(QPen(QColor("#f59e0b"), 1.2))
        dim_line.setZValue(10.3)
        items.append(dim_line)

        dim_txt = QGraphicsTextItem(f"d = {d_spacing_nm:.3f} nm")
        dim_txt.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        dim_txt.setDefaultTextColor(QColor("#f59e0b"))
        dim_txt.setPos(dim_x - 48.0, (plane_y1 + plane_y2) / 2.0 - 8.0)
        dim_txt.setZValue(10.4)
        items.append(dim_txt)

        # Incident and Reflected Rays
        th_rad = math.radians(max(8.0, min(45.0, bragg_theta_deg)))
        cos_t = math.cos(th_rad)
        sin_t = math.sin(th_rad)
        r_len = 110.0

        # Ray 1 (Scatters at atom_cx, plane_y1)
        r1_in = QGraphicsLineItem(atom_cx - r_len * cos_t, plane_y1 - r_len * sin_t, atom_cx, plane_y1)
        r1_in.setPen(QPen(QColor("#06b6d4"), 1.8))
        r1_in.setZValue(10.4)
        items.append(r1_in)

        r1_out = QGraphicsLineItem(atom_cx, plane_y1, atom_cx + r_len * cos_t, plane_y1 - r_len * sin_t)
        r1_out.setPen(QPen(QColor("#10b981"), 2.0))
        r1_out.setZValue(10.4)
        items.append(r1_out)

        # Ray 2 (Scatters at atom_cx, plane_y2)
        r2_in = QGraphicsLineItem(atom_cx - r_len * cos_t, plane_y2 - r_len * sin_t, atom_cx, plane_y2)
        r2_in.setPen(QPen(QColor("#06b6d4"), 1.8))
        r2_in.setZValue(10.4)
        items.append(r2_in)

        r2_out = QGraphicsLineItem(atom_cx, plane_y2, atom_cx + r_len * cos_t, plane_y2 - r_len * sin_t)
        r2_out.setPen(QPen(QColor("#10b981"), 2.0))
        r2_out.setZValue(10.4)
        items.append(r2_out)

        # Path difference construction
        proj_in_x = atom_cx - (d_vis * sin_t) * cos_t
        proj_in_y = plane_y2 - (d_vis * sin_t) * sin_t
        proj_out_x = atom_cx + (d_vis * sin_t) * cos_t
        proj_out_y = plane_y2 - (d_vis * sin_t) * sin_t

        delta_line = QGraphicsLineItem(proj_in_x, proj_in_y, proj_out_x, proj_out_y)
        delta_line.setPen(QPen(QColor("#f59e0b"), 2.2))
        delta_line.setZValue(10.5)
        items.append(delta_line)

        # 4. Right Panel: XRD Powder Diffractogram Spectrum
        p2_x = p1_x + pw + 8.0
        d_bg = QGraphicsRectItem(p2_x, p_y, pw, ph)
        d_bg.setBrush(QBrush(QColor(4, 7, 16)))
        d_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        d_bg.setZValue(10.1)
        items.append(d_bg)

        p2_lbl = QGraphicsTextItem("PHỔ NHIỄU XẠ BỘT I(2θ) & ĐỈNH PHẢN XẠ")
        p2_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p2_lbl.setPos(p2_x + 8.0, p_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        # Plot frame
        gx = p2_x + 30.0
        gy = p_y + 32.0
        gw = pw - 42.0
        gh = ph - 55.0

        plt_box = QGraphicsRectItem(gx, gy, gw, gh)
        plt_box.setBrush(QBrush(QColor(3, 7, 15)))
        plt_box.setPen(QPen(QColor(30, 41, 59), 1.0))
        plt_box.setZValue(10.2)
        items.append(plt_box)

        # 2theta grid ticks
        for t2 in (20, 40, 60, 80):
            tx = gx + ((t2 - 10.0) / 85.0) * gw
            t_line = QGraphicsLineItem(tx, gy, tx, gy + gh)
            t_line.setPen(QPen(QColor(30, 41, 59, 120), 0.8, Qt.PenStyle.DotLine))
            t_line.setZValue(10.2)
            items.append(t_line)

            t_txt = QGraphicsTextItem(f"{t2}°")
            t_txt.setFont(QFont("Segoe UI", 6))
            t_txt.setDefaultTextColor(QColor("#64748b"))
            t_txt.setPos(tx - 8.0, gy + gh + 2.0)
            t_txt.setZValue(10.3)
            items.append(t_txt)

        # Diffractogram peak at two_theta_deg
        pk_x = gx + ((two_theta_deg - 10.0) / 85.0) * gw
        pk_top_y = gy + 20.0
        pk_base_y = gy + gh

        peak_path = QPainterPath()
        peak_path.moveTo(gx, pk_base_y)
        hwhm_px = max(4.0, (scherrer_fwhm_deg / 85.0) * gw * 3.5)
        for i in range(41):
            px_cur = gx + (i / 40.0) * gw
            delta_px = px_cur - pk_x
            y_val = pk_base_y - (gh - 25.0) / (1.0 + (delta_px / hwhm_px) ** 2)
            peak_path.lineTo(px_cur, y_val)

        peak_item = QGraphicsPathItem(peak_path)
        peak_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        peak_item.setZValue(10.3)
        items.append(peak_item)

        # Peak indicator dot & label
        pk_dot = QGraphicsEllipseItem(pk_x - 3.5, pk_top_y - 3.5, 7.0, 7.0)
        pk_dot.setBrush(QBrush(QColor("#f59e0b")))
        pk_dot.setPen(Qt.PenStyle.NoPen)
        pk_dot.setZValue(10.4)
        items.append(pk_dot)

        pk_lbl = QGraphicsTextItem(f"{plane_label} 2θ={two_theta_deg:.1f}°")
        pk_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        pk_lbl.setDefaultTextColor(QColor("#f59e0b"))
        pk_lbl.setPos(pk_x - 24.0, pk_top_y - 14.0)
        pk_lbl.setZValue(10.4)
        items.append(pk_lbl)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Định luật Bragg: n·λ = 2d·sin θ_B    • Gian mặt: d_hkl = a/√(h²+k²+l²) = {d_spacing_nm*10.0:.3f} Å    • Bước sóng: λ = {wavelength_nm*10.0:.3f} Å\n"
            f"• Góc quét: 2θ_B = {two_theta_deg:.2f}°    • Độ rộng Debye-Scherrer: β = K·λ/(D·cos θ) = {scherrer_fwhm_deg:.3f}° FWHM (D = {crystallite_d_nm:.1f} nm)"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(p1_x + 8.0, p_y + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Định luật Bragg n·λ = 2d·sin θ xác định điều kiện giao thoa tăng cường của chùm tia X phản xạ từ các mặt tinh thể song song. "
            "Phương trình Debye-Scherrer β = K·λ / (D·cos θ) liên hệ trực tiếp độ mở rộng chân vạch nhiễu xạ với kích thước hạt nano D."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_stern_gerlach_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "SINGLE_STAGE_SILVER_ATOMS",
        gradient_T_per_m: float = 1200.0,
        deflection_mm: float = 2.15,
        p_up: float = 1.0,
        p_down: float = 1.0,
        is_classical: bool = False,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of the Stern-Gerlach Experiment.
        Includes silver atomic oven source, collimator slits, inhomogeneous magnet poles (N knife-edge, S concave),
        parabolic beam deflection paths, glass detector plate, and spatial spin quantization metrics.
        """
        import math
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"THÍ NGHIỆM STERN - GERLACH & SPIN LƯỢNG TỬ 1/2: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Gradient: dB/dz = {gradient_T_per_m:.0f} T/m | Độ lệch chùm: Δz = ±{deflection_mm:.2f} mm | "
            f"Hình chiếu Spin: S_z = ±ħ/2 (m_s = ±1/2) | Chế độ: {'Cổ điển (Vết mờ)' if is_classical else 'Lượng tử hóa không gian'}"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Apparatus & Deflection Trajectories
        pw = (width - 40.0) / 2.0
        ph = height - 165.0
        p1_x = x0 + 16.0
        p_y = y0 + 58.0

        b_bg = QGraphicsRectItem(p1_x, p_y, pw, ph)
        b_bg.setBrush(QBrush(QColor(4, 7, 16)))
        b_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        b_bg.setZValue(10.1)
        items.append(b_bg)

        p1_lbl = QGraphicsTextItem("THIẾT BỊ NAM CHÂM GRADIENT & QUỸ ĐẠO TÁCH CHÙM")
        p1_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # Diagram parameters inside Left Panel
        oven_x = p1_x + 24.0
        axis_y = p_y + ph / 2.0 + 8.0

        # Thermal Oven
        ov_rect = QGraphicsRectItem(oven_x - 14.0, axis_y - 20.0, 28.0, 40.0)
        ov_rect.setBrush(QBrush(QColor("#ea580c")))
        ov_rect.setPen(QPen(QColor("#fdba74"), 1.0))
        ov_rect.setZValue(10.3)
        items.append(ov_rect)

        ov_lbl = QGraphicsTextItem("Lò Ag")
        ov_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        ov_lbl.setDefaultTextColor(QColor("#ffffff"))
        ov_lbl.setPos(oven_x - 12.0, axis_y - 12.0)
        ov_lbl.setZValue(10.4)
        items.append(ov_lbl)

        # Collimator Slits
        s1 = QGraphicsLineItem(oven_x + 26.0, axis_y - 24.0, oven_x + 26.0, axis_y - 4.0)
        s1.setPen(QPen(QColor("#64748b"), 2.2))
        s1.setZValue(10.3)
        items.append(s1)

        s2 = QGraphicsLineItem(oven_x + 26.0, axis_y + 4.0, oven_x + 26.0, axis_y + 24.0)
        s2.setPen(QPen(QColor("#64748b"), 2.2))
        s2.setZValue(10.3)
        items.append(s2)

        # Magnet Poles (N knife-edge, S concave)
        m_x = oven_x + 50.0
        m_w = 90.0

        north_poly = QPolygonF()
        north_poly.append(QPointF(m_x, axis_y - 45.0))
        north_poly.append(QPointF(m_x + m_w, axis_y - 45.0))
        north_poly.append(QPointF(m_x + m_w, axis_y - 24.0))
        north_poly.append(QPointF(m_x + m_w * 0.5, axis_y - 14.0))
        north_poly.append(QPointF(m_x, axis_y - 24.0))

        n_item = QGraphicsPolygonItem(north_poly)
        n_item.setBrush(QBrush(QColor("#1e293b")))
        n_item.setPen(QPen(QColor("#ef4444"), 1.2))
        n_item.setZValue(10.3)
        items.append(n_item)

        n_lbl = QGraphicsTextItem("N (Mũi dao)")
        n_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        n_lbl.setDefaultTextColor(QColor("#ef4444"))
        n_lbl.setPos(m_x + m_w * 0.5 - 18.0, axis_y - 40.0)
        n_lbl.setZValue(10.4)
        items.append(n_lbl)

        south_poly = QPolygonF()
        south_poly.append(QPointF(m_x, axis_y + 18.0))
        south_poly.append(QPointF(m_x + m_w * 0.5, axis_y + 14.0))
        south_poly.append(QPointF(m_x + m_w, axis_y + 18.0))
        south_poly.append(QPointF(m_x + m_w, axis_y + 45.0))
        south_poly.append(QPointF(m_x, axis_y + 45.0))

        s_item = QGraphicsPolygonItem(south_poly)
        s_item.setBrush(QBrush(QColor("#1e293b")))
        s_item.setPen(QPen(QColor("#3b82f6"), 1.2))
        s_item.setZValue(10.3)
        items.append(s_item)

        s_lbl = QGraphicsTextItem("S (Rãnh lõm)")
        s_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor("#3b82f6"))
        s_lbl.setPos(m_x + m_w * 0.5 - 18.0, axis_y + 26.0)
        s_lbl.setZValue(10.4)
        items.append(s_lbl)

        # Screen Target Line
        sc_x = p1_x + pw - 25.0
        sc_line = QGraphicsLineItem(sc_x, axis_y - 50.0, sc_x, axis_y + 50.0)
        sc_line.setPen(QPen(QColor("#e2e8f0"), 2.5))
        sc_line.setZValue(10.3)
        items.append(sc_line)

        # Split trajectories
        def_vis = min(40.0, max(10.0, deflection_mm * 16.0))

        path_up = QPainterPath()
        path_up.moveTo(oven_x + 26.0, axis_y)
        path_up.quadTo(m_x + m_w * 0.6, axis_y - def_vis * 0.4, sc_x, axis_y - def_vis)

        up_item = QGraphicsPathItem(path_up)
        up_item.setPen(QPen(QColor("#10b981"), 2.0))
        up_item.setZValue(10.4)
        items.append(up_item)

        path_down = QPainterPath()
        path_down.moveTo(oven_x + 26.0, axis_y)
        path_down.quadTo(m_x + m_w * 0.6, axis_y + def_vis * 0.4, sc_x, axis_y + def_vis)

        down_item = QGraphicsPathItem(path_down)
        down_item.setPen(QPen(QColor("#f43f5e"), 2.0))
        down_item.setZValue(10.4)
        items.append(down_item)

        # 4. Right Panel: Detector Glass Plate & Double Peak Intensity
        p2_x = p1_x + pw + 8.0
        d_bg = QGraphicsRectItem(p2_x, p_y, pw, ph)
        d_bg.setBrush(QBrush(QColor(4, 7, 16)))
        d_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        d_bg.setZValue(10.1)
        items.append(d_bg)

        p2_lbl = QGraphicsTextItem("BẢN KÍNH HỨNG & PHÂN BỐ CƯỜNG ĐỘ I(z)")
        p2_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p2_lbl.setPos(p2_x + 8.0, p_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        # Glass Plate graphic
        gp_x = p2_x + 25.0
        gp_y = p_y + 35.0
        gp_w = 90.0
        gp_h = ph - 55.0

        gp_rect = QGraphicsRectItem(gp_x, gp_y, gp_w, gp_h)
        gp_rect.setBrush(QBrush(QColor(15, 23, 42)))
        gp_rect.setPen(QPen(QColor(100, 116, 139), 1.4))
        gp_rect.setZValue(10.2)
        items.append(gp_rect)

        mid_y = gp_y + gp_h / 2.0

        # Silver deposit spots
        if is_classical:
            smudge = QGraphicsRectItem(gp_x + 30.0, mid_y - def_vis, 30.0, def_vis * 2.0)
            smudge.setBrush(QBrush(QColor(245, 158, 11, 160)))
            smudge.setPen(Qt.PenStyle.NoPen)
            smudge.setZValue(10.3)
            items.append(smudge)
        else:
            # Two discrete spots
            spot1 = QGraphicsEllipseItem(gp_x + 30.0, mid_y - def_vis - 5.0, 30.0, 10.0)
            spot1.setBrush(QBrush(QColor("#10b981")))
            spot1.setPen(Qt.PenStyle.NoPen)
            spot1.setZValue(10.3)
            items.append(spot1)

            spot2 = QGraphicsEllipseItem(gp_x + 30.0, mid_y + def_vis - 5.0, 30.0, 10.0)
            spot2.setBrush(QBrush(QColor("#f43f5e")))
            spot2.setPen(Qt.PenStyle.NoPen)
            spot2.setZValue(10.3)
            items.append(spot2)

        # Intensity Curve on Right side
        prof_x = gp_x + gp_w + 15.0
        prof_w = pw - gp_w - 55.0
        prof_path = QPainterPath()
        first = True
        for i in range(41):
            y_cur = gp_y + (i / 40.0) * gp_h
            z_off = mid_y - y_cur
            if is_classical:
                intens = 0.5 if abs(z_off) <= def_vis else 0.05
            else:
                sigma = 8.0
                g1 = math.exp(-((z_off - def_vis)**2) / (2.0 * sigma * sigma))
                g2 = math.exp(-((z_off + def_vis)**2) / (2.0 * sigma * sigma))
                intens = min(0.95, g1 + g2 + 0.04)

            x_cur = prof_x + intens * prof_w
            if first:
                prof_path.moveTo(x_cur, y_cur)
                first = False
            else:
                prof_path.lineTo(x_cur, y_cur)

        prof_item = QGraphicsPathItem(prof_path)
        prof_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        prof_item.setZValue(10.3)
        items.append(prof_item)

        # Spot labels
        l1 = QGraphicsTextItem("|↑z⟩ (Spin-Up)")
        l1.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        l1.setDefaultTextColor(QColor("#10b981"))
        l1.setPos(prof_x + 8.0, mid_y - def_vis - 8.0)
        l1.setZValue(10.4)
        items.append(l1)

        l2 = QGraphicsTextItem("|↓z⟩ (Spin-Down)")
        l2.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        l2.setDefaultTextColor(QColor("#f43f5e"))
        l2.setPos(prof_x + 8.0, mid_y + def_vis - 8.0)
        l2.setZValue(10.4)
        items.append(l2)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Lực từ: F_z = ±μ_B·(dB_z/dz) = ±{9.274e-24 * gradient_T_per_m:.2e} N    • Mômen từ: μ_B = 9.274×10⁻²⁴ J/T\n"
            f"• Độ lệch chùm hạt: Δz = ±{deflection_mm:.2f} mm    • Hình chiếu Spin: S_z = ±ħ/2    • Hiện tượng: LƯỢNG TỬ HÓA KHÔNG GIAN"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(p1_x + 8.0, p_y + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Thí nghiệm Stern-Gerlach bác bỏ dự đoán cổ điển về sự định hướng liên tục của mômen từ nguyên tử. "
            "Sự tách đôi của chùm nguyên tử bạc chứng minh electron có spin nội tại S = 1/2 bị lượng tử hóa không gian thành đúng hai mức năng lượng tách rời."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_poiseuille_flow_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "WATER_CAPILLARY_LAMINAR",
        fluid_name: str = "Nước Cất ở 20°C (Water)",
        radius_mm: float = 0.5,
        delta_p_Pa: float = 1200.0,
        viscosity_mPa_s: float = 1.002,
        pipe_length_m: float = 0.20,
        flow_rate_mL_per_s: float = 1.25,
        max_velocity_m_per_s: float = 0.75,
        average_velocity_m_per_s: float = 0.375,
        reynolds_number: float = 450.0,
        flow_regime: str = "LAMINAR",
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Hagen-Poiseuille Viscous Pipe Flow.
        Illustrates the parabolic velocity profile v(r) = (ΔP/4ηL)*(R² - r²), no-slip wall conditions,
        pressure gradient drop, 4th-power volumetric flow rate law Q ∝ R⁴, and Reynolds number regime.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#090e17")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 190.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("🌊 FLUID DYNAMICS • POISEUILLE")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("DÒNG CHẢY NHỚT HAGEN-POISEUILLE & SỐ REYNOLDS")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 215.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Chất lỏng: {fluid_name} | R = {radius_mm:.3f} mm | ΔP = {delta_p_Pa:.1f} Pa | L = {pipe_length_m * 1000.0:.1f} mm"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Cylindrical Pipe with Parabolic Vectors
        p1_x = x0 + 18.0
        p1_y = y0 + 64.0
        p1_w = 385.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#060a12")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Trường Vector Vận Tốc Parabol v(r) = vmax·(1 - r²/R²)")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # Pipe geometry
        tube_x0 = p1_x + 36.0
        tube_w = p1_w - 55.0
        tube_cy = p1_y + p1_h / 2.0 + 8.0
        tube_h = 120.0
        tube_top = tube_cy - tube_h / 2.0
        tube_bot = tube_cy + tube_h / 2.0

        # Fluid pressure gradient rect
        f_rect = QGraphicsRectItem(tube_x0, tube_top, tube_w, tube_h)
        f_rect.setBrush(QBrush(QColor(14, 165, 233, 35)))
        f_rect.setPen(Qt.PenStyle.NoPen)
        f_rect.setZValue(10.15)
        items.append(f_rect)

        # Walls
        top_wall = QGraphicsLineItem(tube_x0, tube_top, tube_x0 + tube_w, tube_top)
        top_wall.setPen(QPen(QColor("#64748b"), 2.5))
        top_wall.setZValue(10.2)
        items.append(top_wall)

        bot_wall = QGraphicsLineItem(tube_x0, tube_bot, tube_x0 + tube_w, tube_bot)
        bot_wall.setPen(QPen(QColor("#64748b"), 2.5))
        bot_wall.setZValue(10.2)
        items.append(bot_wall)

        # Wall hatching
        for hi in range(0, int(tube_w), 16):
            hx = tube_x0 + hi
            ht = QGraphicsLineItem(hx, tube_top, hx + 6.0, tube_top - 6.0)
            ht.setPen(QPen(QColor("#475569"), 1.0))
            ht.setZValue(10.2)
            items.append(ht)

            hb = QGraphicsLineItem(hx, tube_bot, hx + 6.0, tube_bot + 6.0)
            hb.setPen(QPen(QColor("#475569"), 1.0))
            hb.setZValue(10.2)
            items.append(hb)

        # Centerline
        c_line = QGraphicsLineItem(tube_x0, tube_cy, tube_x0 + tube_w, tube_cy)
        c_line.setPen(QPen(QColor("#38bdf8"), 1.0, Qt.PenStyle.DashLine))
        c_line.setZValue(10.2)
        items.append(c_line)

        # Velocity Vectors at Station
        col_x = tube_x0 + tube_w * 0.45
        s_line = QGraphicsLineItem(col_x, tube_top, col_x, tube_bot)
        s_line.setPen(QPen(QColor("#475569"), 1.0, Qt.PenStyle.DotLine))
        s_line.setZValue(10.2)
        items.append(s_line)

        num_vecs = 9
        max_vec_len = 65.0
        for vi in range(num_vecs):
            rel_r = -1.0 + vi * (2.0 / (num_vecs - 1))
            vy = tube_cy + rel_r * (tube_h / 2.0)
            v_frac = max(0.0, 1.0 - (rel_r ** 2))
            v_len = v_frac * max_vec_len
            if v_len > 1.0:
                v_line = QGraphicsLineItem(col_x, vy, col_x + v_len, vy)
                v_line.setPen(QPen(QColor("#38bdf8"), 1.8))
                v_line.setZValue(10.3)
                items.append(v_line)

                # Arrowhead
                if v_len > 5.0:
                    arr_path = QPainterPath()
                    arr_path.moveTo(col_x + v_len, vy)
                    arr_path.lineTo(col_x + v_len - 4.0, vy - 2.5)
                    arr_path.lineTo(col_x + v_len - 4.0, vy + 2.5)
                    arr_path.closeSubpath()
                    arr_poly = QGraphicsPathItem(arr_path)
                    arr_poly.setBrush(QBrush(QColor("#38bdf8")))
                    arr_poly.setPen(Qt.PenStyle.NoPen)
                    arr_poly.setZValue(10.3)
                    items.append(arr_poly)

        # No-slip condition label
        no_slip_lbl = QGraphicsTextItem("Thành ống: No-slip v(±R) = 0")
        no_slip_lbl.setFont(QFont("Segoe UI", 7))
        no_slip_lbl.setDefaultTextColor(QColor("#f43f5e"))
        no_slip_lbl.setPos(tube_x0 + 10.0, tube_top - 12.0)
        no_slip_lbl.setZValue(10.25)
        items.append(no_slip_lbl)

        # Centerline tag
        c_lbl = QGraphicsTextItem("Trục r=0 (vmax)")
        c_lbl.setFont(QFont("Consolas", 7))
        c_lbl.setDefaultTextColor(QColor("#38bdf8"))
        c_lbl.setPos(tube_x0 + tube_w - 65.0, tube_cy - 4.0)
        c_lbl.setZValue(10.25)
        items.append(c_lbl)

        # 4. Right Panel: 2D Velocity Profile Plot v(r)
        p2_x = x0 + 413.0
        p2_y = y0 + 64.0
        p2_w = 249.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#060a12")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Đồ Thị Phân Bố Vận Tốc v(r)")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        ox2 = p2_x + 36.0
        cy2 = p2_y + p2_h / 2.0 + 8.0
        plot_w2 = p2_w - 55.0
        plot_h2 = 120.0
        top_y2 = cy2 - plot_h2 / 2.0
        bot_y2 = cy2 + plot_h2 / 2.0

        # Axes
        ax2_y = QGraphicsLineItem(ox2, top_y2 - 6.0, ox2, bot_y2 + 6.0)
        ax2_y.setPen(QPen(QColor("#64748b"), 1.2))
        ax2_y.setZValue(10.2)
        items.append(ax2_y)

        ax2_x = QGraphicsLineItem(ox2, cy2, ox2 + plot_w2 + 6.0, cy2)
        ax2_x.setPen(QPen(QColor("#64748b"), 1.2))
        ax2_x.setZValue(10.2)
        items.append(ax2_x)

        # Parabolic path
        prof_path = QPainterPath()
        num_pts = 30
        for i in range(num_pts + 1):
            rel_r = -1.0 + i * (2.0 / num_pts)
            py_pt = cy2 + rel_r * (plot_h2 / 2.0)
            v_frac = max(0.0, 1.0 - (rel_r ** 2))
            px_pt = ox2 + v_frac * (plot_w2 * 0.88)
            if i == 0:
                prof_path.moveTo(px_pt, py_pt)
            else:
                prof_path.lineTo(px_pt, py_pt)

        prof_item = QGraphicsPathItem(prof_path)
        prof_item.setPen(QPen(QColor("#38bdf8"), 2.2))
        prof_item.setZValue(10.3)
        items.append(prof_item)

        # v_avg line (dashed)
        v_avg_px = ox2 + 0.5 * (plot_w2 * 0.88)
        avg_line = QGraphicsLineItem(v_avg_px, top_y2, v_avg_px, bot_y2)
        avg_line.setPen(QPen(QColor("#f59e0b"), 1.0, Qt.PenStyle.DashLine))
        avg_line.setZValue(10.25)
        items.append(avg_line)

        avg_tag = QGraphicsTextItem("v_tb = vmax/2")
        avg_tag.setFont(QFont("Consolas", 7))
        avg_tag.setDefaultTextColor(QColor("#f59e0b"))
        avg_tag.setPos(v_avg_px - 22.0, top_y2 - 14.0)
        avg_tag.setZValue(10.3)
        items.append(avg_tag)

        # Regime indicator badge
        regime_col = QColor("#10b981") if flow_regime == "LAMINAR" else (QColor("#f59e0b") if flow_regime == "TRANSITIONAL" else QColor("#ef4444"))
        r_box = QGraphicsRectItem(p2_x + 12.0, p2_y + p2_h - 28.0, p2_w - 24.0, 20.0)
        r_box.setBrush(QBrush(QColor(regime_col.red(), regime_col.green(), regime_col.blue(), 35)))
        r_box.setPen(QPen(regime_col, 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        r_txt = QGraphicsTextItem(f"CHẾ ĐỘ: {flow_regime} (Re = {reynolds_number:.1f})")
        r_txt.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_txt.setDefaultTextColor(regime_col)
        r_txt.setPos(p2_x + 22.0, p2_y + p2_h - 28.0)
        r_txt.setZValue(10.25)
        items.append(r_txt)

        # 5. Bottom Quantitative Metrics Bar
        m_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 36.0, 52.0)
        m_box.setBrush(QBrush(QColor("#0d1526")))
        m_box.setPen(QPen(QColor("#1e293b"), 1.0))
        m_box.setZValue(10.15)
        items.append(m_box)

        m_txt = QGraphicsTextItem(
            f"• Định luật Hagen-Poiseuille: Q = (π·R⁴·ΔP)/(8·η·L) = {flow_rate_mL_per_s:.4f} mL/s  ➔  Phụ thuộc lũy thừa 4 bán kính (Q ∝ R⁴)\n"
            f"• Vận tốc cực đại vmax: {max_velocity_m_per_s:.3f} m/s    • Vận tốc trung bình v_tb: {average_velocity_m_per_s:.3f} m/s    • Số Reynolds: Re = {reynolds_number:.1f}"
        )
        m_txt.setFont(QFont("Consolas", 8))
        m_txt.setDefaultTextColor(QColor("#e2e8f0"))
        m_txt.setPos(p1_x + 10.0, y0 + 274.0)
        m_txt.setZValue(10.25)
        items.append(m_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình Navier-Stokes cho dòng chảy tầng trong ống trụ tạo ra profile vận tốc parabol kinh điển v(r) = vmax·(1 - r²/R²). "
            "Lưu lượng thể tích tỷ lệ với R⁴ giải thích cơ chế huyết động học vi mạch (tiểu động mạch chỉ cần co thắt nhẹ đã làm giảm mạnh lưu lượng máu) "
            "và là nền tảng của các phương pháp đo nhớt kế mao quản."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_cosmic_expansion_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "MODERN_PLANCK_LAMBDA_CDM",
        target_name: str = "Thiên hà Chuẩn Cụm Coma",
        hubble_constant: float = 67.4,
        distance_mpc: float = 100.0,
        recessional_velocity_km_s: float = 6740.0,
        redshift_z: float = 0.0225,
        hubble_time_Gyr: float = 14.51,
        hubble_horizon_distance_Mpc: float = 4448.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of the Hubble-Lemaître Law of Cosmic Expansion
        and Hydrogen Balmer Absorption Line Redshifts.
        Illustrates expanding cosmic grid with receding galaxies, rest-frame vs observed redshifted
        spectral bands (Hα, Hβ, Hγ, Hδ), Hubble linear expansion slope v = H0·d, and cosmic age t_H = 1/H0.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#080c14")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 195.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("🌌 ASTROPHYSICS • HUBBLE LAW")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("ĐỊNH LUẬT HUBBLE-LEMAÎTRE & VŨ TRỤ GIÃN NỞ")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 220.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Mục tiêu: {target_name} | H₀ = {hubble_constant:.1f} km/s/Mpc | d = {distance_mpc:.1f} Mpc | v = {recessional_velocity_km_s:.0f} km/s"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Expanding Cosmic Grid with Receding Galaxies
        p1_x = x0 + 18.0
        p1_y = y0 + 64.0
        p1_w = 320.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#04070d")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Không Gian Giãn Nở Đẳng Hướng (Metric Expansion)")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        cx1 = p1_x + p1_w / 2.0
        cy1 = p1_y + p1_h / 2.0 + 8.0
        r_max = (p1_h / 2.0) - 22.0

        # Concentric distance circles
        for frac in [0.35, 0.70, 1.0]:
            r_c = frac * r_max
            ring = QGraphicsEllipseItem(cx1 - r_c, cy1 - r_c, 2.0 * r_c, 2.0 * r_c)
            ring.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
            ring.setBrush(Qt.BrushStyle.NoBrush)
            ring.setZValue(10.15)
            items.append(ring)

        # Central Observer (Milky Way)
        mw_dot = QGraphicsEllipseItem(cx1 - 4.5, cy1 - 4.5, 9.0, 9.0)
        mw_dot.setBrush(QBrush(QColor("#fbbf24")))
        mw_dot.setPen(QPen(QColor("#ffffff"), 1.2))
        mw_dot.setZValue(10.3)
        items.append(mw_dot)

        mw_lbl = QGraphicsTextItem("Ngân Hà")
        mw_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        mw_lbl.setDefaultTextColor(QColor("#fbbf24"))
        mw_lbl.setPos(cx1 - 20.0, cy1 - 18.0)
        mw_lbl.setZValue(10.35)
        items.append(mw_lbl)

        # Surrounding representative galaxies with outward vectors
        galaxy_samples = [
            (0.55, 0.35, "NGC 224", QColor("#38bdf8")),
            (-0.65, 0.40, "M87", QColor("#818cf8")),
            (0.40, -0.70, "NGC 1365", QColor("#f472b6")),
            (-0.50, -0.55, "Centaurus A", QColor("#fbbf24")),
            (0.85, -0.20, "Coma Cluster", QColor("#34d399")),
            (-0.80, 0.10, "Virgo Cluster", QColor("#c084fc")),
        ]

        for gx_rel, gy_rel, gname, gcol in galaxy_samples:
            gx = cx1 + gx_rel * r_max
            gy = cy1 + gy_rel * r_max

            # Vector line from center outward
            v_line = QGraphicsLineItem(cx1, cy1, gx, gy)
            v_line.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DotLine))
            v_line.setZValue(10.2)
            items.append(v_line)

            # Galaxy dot
            g_dot = QGraphicsEllipseItem(gx - 3.5, gy - 3.5, 7.0, 7.0)
            g_dot.setBrush(QBrush(gcol))
            g_dot.setPen(QPen(QColor("#ffffff"), 1.0))
            g_dot.setZValue(10.3)
            items.append(g_dot)

        # 4. Right Panel: Hydrogen Balmer Spectral Absorption Bands
        p2_x = x0 + 348.0
        p2_y = y0 + 64.0
        p2_w = 314.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#04070d")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Dịch Chuyển Đỏ Phổ Vạch Hydro (Balmer Redshift)")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        # Rest frame band
        s1_x = p2_x + 14.0
        s1_y = p2_y + 44.0
        s_w = p2_w - 28.0
        s_h = 24.0

        s1_lbl = QGraphicsTextItem("Phổ Chuẩn Phòng TN (Rest Frame, z = 0):")
        s1_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        s1_lbl.setDefaultTextColor(QColor("#94a3b8"))
        s1_lbl.setPos(s1_x, s1_y - 15.0)
        s1_lbl.setZValue(10.2)
        items.append(s1_lbl)

        spec1 = QGraphicsRectItem(s1_x, s1_y, s_w, s_h)
        spec1.setBrush(QBrush(QColor("#0ea5e9")))
        spec1.setPen(QPen(QColor("#334155"), 1.0))
        spec1.setZValue(10.15)
        items.append(spec1)

        # Observed redshifted band
        s2_y = s1_y + 58.0
        s2_lbl = QGraphicsTextItem(f"Phổ Thiên Hà Quan Sát (Dịch Đỏ z = {redshift_z:.4f}):")
        s2_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        s2_lbl.setDefaultTextColor(QColor("#f43f5e"))
        s2_lbl.setPos(s1_x, s2_y - 15.0)
        s2_lbl.setZValue(10.2)
        items.append(s2_lbl)

        spec2 = QGraphicsRectItem(s1_x, s2_y, s_w, s_h)
        spec2.setBrush(QBrush(QColor("#f43f5e")))
        spec2.setPen(QPen(QColor("#334155"), 1.0))
        spec2.setZValue(10.15)
        items.append(spec2)

        # Balmer absorption lines (Hδ, Hγ, Hβ, Hα)
        balmer = [("Hδ", 410.2), ("Hγ", 434.0), ("Hβ", 486.1), ("Hα", 656.3)]
        wl_min = 390.0
        wl_max = 700.0

        for b_name, b_wl in balmer:
            # Rest line
            rel_rest = (b_wl - wl_min) / (wl_max - wl_min)
            lx1 = s1_x + rel_rest * s_w
            line1 = QGraphicsLineItem(lx1, s1_y, lx1, s1_y + s_h)
            line1.setPen(QPen(QColor("#000000"), 2.0))
            line1.setZValue(10.25)
            items.append(line1)

            # Redshifted line
            wl_obs = b_wl * (1.0 + min(0.35, redshift_z))
            if wl_obs <= wl_max:
                rel_obs = (wl_obs - wl_min) / (wl_max - wl_min)
                lx2 = s1_x + rel_obs * s_w
                line2 = QGraphicsLineItem(lx2, s2_y, lx2, s2_y + s_h)
                line2.setPen(QPen(QColor("#000000"), 2.0))
                line2.setZValue(10.25)
                items.append(line2)

                # Displacement arrow
                arr_lead = QGraphicsLineItem(lx1, s1_y + s_h, lx2, s2_y)
                arr_lead.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DotLine))
                arr_lead.setZValue(10.2)
                items.append(arr_lead)

        # Formula badge in Plot 2
        f_rect = QGraphicsRectItem(s1_x, s2_y + s_h + 10.0, s_w, 24.0)
        f_rect.setBrush(QBrush(QColor("#0f172a")))
        f_rect.setPen(QPen(QColor("#f43f5e"), 1.0))
        f_rect.setZValue(10.2)
        items.append(f_rect)

        f_txt = QGraphicsTextItem(f"z = Δλ / λ₀ = {redshift_z:.4f}  |  v = c · [((1+z)²-1)/((1+z)²+1)]")
        f_txt.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        f_txt.setDefaultTextColor(QColor("#f43f5e"))
        f_txt.setPos(s1_x + 6.0, s2_y + s_h + 12.0)
        f_txt.setZValue(10.25)
        items.append(f_txt)

        # 5. Bottom Quantitative Metrics Bar
        m_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 36.0, 52.0)
        m_box.setBrush(QBrush(QColor("#0d1526")))
        m_box.setPen(QPen(QColor("#1e293b"), 1.0))
        m_box.setZValue(10.15)
        items.append(m_box)

        m_txt = QGraphicsTextItem(
            f"• Định luật Hubble-Lemaître: v = H₀·d = {recessional_velocity_km_s:.0f} km/s  ➔  Hằng số Hubble H₀ = {hubble_constant:.1f} km/s/Mpc\n"
            f"• Khoảng cách d = {distance_mpc:.1f} Mpc    • Độ dịch đỏ z = {redshift_z:.4f}    • Tuổi vũ trụ t_H = {hubble_time_Gyr:.2f} Gyr    • Bán kính Hubble: d_H = {hubble_horizon_distance_Mpc:.0f} Mpc"
        )
        m_txt.setFont(QFont("Consolas", 8))
        m_txt.setDefaultTextColor(QColor("#e2e8f0"))
        m_txt.setPos(p1_x + 10.0, y0 + 274.0)
        m_txt.setZValue(10.25)
        items.append(m_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Định luật Hubble-Lemaître (1929) chứng minh vũ trụ đang giãn nở theo phương trình trắc địa FLRW. "
            "Độ dịch chuyển đỏ quang phổ của vạch Balmer hydro không phải hiệu ứng Doppler cơ học cổ điển mà do bước sóng photon bị kéo dãn "
            "đồng thời với sự dãn nở của hệ số co giãn không gian a(t) trong suốt thời gian lan truyền hàng triệu năm."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_seismic_waves_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "TOHOKU_2011_MEGATHRUST",
        event_name: str = "Trận động đất Tohoku-Oki 2011",
        epicenter_x: float = 140.0,
        epicenter_y: float = -60.0,
        magnitude: float = 9.1,
        vp: float = 7.6,
        vs: float = 4.3,
        depth_km: float = 29.0,
        dist_a_km: float = 300.0,
        sp_lag_a_s: float = 30.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Seismic Waves,
        P-Wave & S-Wave travel times, and 3-Station Epicenter Triangulation.
        Illustrates intersecting circular station distances (xA, yA, dA),
        synthetic 3-phase seismogram trace, S-P arrival lag time, and Omori formula.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#080c14")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 190.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("🌍 GEOPHYSICS • SEISMOLOGY")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("SÓNG ĐỊA CHẤN & ĐỊNH VỊ TÂM CHẤN 3 TRẠM")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 215.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Sự kiện: {event_name} | M_w = {magnitude:.1f} | vp = {vp:.1f} km/s, vs = {vs:.1f} km/s | Tâm chấn: ({epicenter_x:+.0f}, {epicenter_y:+.0f}) km"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: 3-Station Triangulation Map
        p1_x = x0 + 18.0
        p1_y = y0 + 64.0
        p1_w = 320.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#050811")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Định Vị Giao Điểm 3 Trạm (Triangulation)")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        map_cx = p1_x + p1_w / 2.0
        map_cy = p1_y + p1_h / 2.0 + 8.0
        scale = (p1_w - 40.0) / 440.0

        def km_to_px(xk: float, yk: float) -> tuple[float, float]:
            return map_cx + xk * scale, map_cy - yk * scale

        # Origin Axes
        ox_px, oy_px = km_to_px(0.0, 0.0)
        ax_x = QGraphicsLineItem(p1_x + 10.0, oy_px, p1_x + p1_w - 10.0, oy_px)
        ax_x.setPen(QPen(QColor("#334155"), 0.8, Qt.PenStyle.DashLine))
        ax_x.setZValue(10.15)
        items.append(ax_x)

        ax_y = QGraphicsLineItem(ox_px, p1_y + 24.0, ox_px, p1_y + p1_h - 8.0)
        ax_y.setPen(QPen(QColor("#334155"), 0.8, Qt.PenStyle.DashLine))
        ax_y.setZValue(10.15)
        items.append(ax_y)

        # 3 Stations definition
        stations = [
            (-150.0, -100.0, "STA_A", QColor("#38bdf8")),
            (180.0, -120.0, "STA_B", QColor("#10b981")),
            (50.0, 160.0, "STA_C", QColor("#f59e0b")),
        ]

        # Station circles & markers
        for sx, sy, slbl, scol in stations:
            st_px, st_py = km_to_px(sx, sy)
            # Distance from station to epicenter
            d_km = math.sqrt((epicenter_x - sx) ** 2 + (epicenter_y - sy) ** 2 + depth_km ** 2)
            r_px = d_km * scale

            # Circle
            circ = QGraphicsEllipseItem(st_px - r_px, st_py - r_px, 2.0 * r_px, 2.0 * r_px)
            circ.setPen(QPen(scol, 1.0, Qt.PenStyle.DashLine))
            circ.setBrush(Qt.BrushStyle.NoBrush)
            circ.setZValue(10.2)
            items.append(circ)

            # Triangle marker
            tri = QPainterPath()
            tri.moveTo(st_px, st_py - 5.0)
            tri.lineTo(st_px - 4.5, st_py + 4.0)
            tri.lineTo(st_px + 4.5, st_py + 4.0)
            tri.closeSubpath()
            tri_item = QGraphicsPathItem(tri)
            tri_item.setBrush(QBrush(scol))
            tri_item.setPen(QPen(QColor("#ffffff"), 0.8))
            tri_item.setZValue(10.3)
            items.append(tri_item)

            tag = QGraphicsTextItem(slbl)
            tag.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
            tag.setDefaultTextColor(scol)
            tag.setPos(st_px + 5.0, st_py - 6.0)
            tag.setZValue(10.35)
            items.append(tag)

        # Epicenter marker
        ep_px, ep_py = km_to_px(epicenter_x, epicenter_y)
        ep_dot = QGraphicsEllipseItem(ep_px - 4.5, ep_py - 4.5, 9.0, 9.0)
        ep_dot.setBrush(QBrush(QColor("#f43f5e")))
        ep_dot.setPen(QPen(QColor("#ffffff"), 1.2))
        ep_dot.setZValue(10.4)
        items.append(ep_dot)

        ep_lbl = QGraphicsTextItem("Tâm Chấn")
        ep_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        ep_lbl.setDefaultTextColor(QColor("#f43f5e"))
        ep_lbl.setPos(ep_px - 18.0, ep_py - 16.0)
        ep_lbl.setZValue(10.45)
        items.append(ep_lbl)

        # 4. Right Panel: Synthetic Seismogram & S-P Lag
        p2_x = x0 + 348.0
        p2_y = y0 + 64.0
        p2_w = 314.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#050811")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Băng Địa Chấn Ký & Độ Trễ S-P (Station A)")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        t_x0 = p2_x + 24.0
        t_y0 = p2_y + 55.0
        t_w = p2_w - 40.0
        t_h = 75.0
        cy2 = t_y0 + t_h / 2.0

        # Baseline
        base_line = QGraphicsLineItem(t_x0, cy2, t_x0 + t_w, cy2)
        base_line.setPen(QPen(QColor("#1e293b"), 1.0))
        base_line.setZValue(10.15)
        items.append(base_line)

        tp = dist_a_km / vp
        ts = dist_a_km / vs
        max_t = max(60.0, ts * 1.35)

        # Seismogram waveform
        tr_path = QPainterPath()
        num_pts = 90
        import random
        random.seed(101)

        for i in range(num_pts + 1):
            ct = (i / num_pts) * max_t
            cx_pt = t_x0 + (ct / max_t) * t_w
            if ct < tp:
                amp = (random.random() - 0.5) * 2.5
            elif ct < ts:
                phase = (ct - tp) * 16.0
                amp = math.sin(phase) * 10.0 + (random.random() - 0.5) * 3.5
            else:
                phase_s = (ct - ts) * 7.5
                env = math.exp(-(ct - ts) * 0.04)
                amp = math.sin(phase_s) * 28.0 * env + (random.random() - 0.5) * 5.0
            cy_pt = cy2 + amp
            if i == 0:
                tr_path.moveTo(cx_pt, cy_pt)
            else:
                tr_path.lineTo(cx_pt, cy_pt)

        tr_item = QGraphicsPathItem(tr_path)
        tr_item.setPen(QPen(QColor("#38bdf8"), 1.4))
        tr_item.setZValue(10.25)
        items.append(tr_item)

        # P arrival line
        px_pt = t_x0 + (tp / max_t) * t_w
        p_line = QGraphicsLineItem(px_pt, t_y0 - 6.0, px_pt, t_y0 + t_h + 6.0)
        p_line.setPen(QPen(QColor("#10b981"), 1.0, Qt.PenStyle.DashLine))
        p_line.setZValue(10.3)
        items.append(p_line)

        p_tag = QGraphicsTextItem("P")
        p_tag.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p_tag.setDefaultTextColor(QColor("#10b981"))
        p_tag.setPos(px_pt - 6.0, t_y0 - 15.0)
        p_tag.setZValue(10.35)
        items.append(p_tag)

        # S arrival line
        sx_pt = t_x0 + (ts / max_t) * t_w
        s_line = QGraphicsLineItem(sx_pt, t_y0 - 6.0, sx_pt, t_y0 + t_h + 6.0)
        s_line.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DashLine))
        s_line.setZValue(10.3)
        items.append(s_line)

        s_tag = QGraphicsTextItem("S")
        s_tag.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        s_tag.setDefaultTextColor(QColor("#f43f5e"))
        s_tag.setPos(sx_pt - 6.0, t_y0 - 15.0)
        s_tag.setZValue(10.35)
        items.append(s_tag)

        # S-P interval bar
        sp_bar_y = t_y0 + t_h + 12.0
        sp_bar = QGraphicsLineItem(px_pt, sp_bar_y, sx_pt, sp_bar_y)
        sp_bar.setPen(QPen(QColor("#fbbf24"), 1.2))
        sp_bar.setZValue(10.3)
        items.append(sp_bar)

        sp_lbl = QGraphicsTextItem(f"Δt_SP = {sp_lag_a_s:.1f} s")
        sp_lbl.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        sp_lbl.setDefaultTextColor(QColor("#fbbf24"))
        sp_lbl.setPos((px_pt + sx_pt) / 2.0 - 24.0, sp_bar_y + 4.0)
        sp_lbl.setZValue(10.35)
        items.append(sp_lbl)

        # Formula tag
        f_box2 = QRectF(t_x0, p2_y + p2_h - 28.0, t_w, 20.0)
        f_box2_item = QGraphicsRectItem(f_box2)
        f_box2_item.setBrush(QBrush(QColor("#0f172a")))
        f_box2_item.setPen(QPen(QColor("#38bdf8"), 1.0))
        f_box2_item.setZValue(10.2)
        items.append(f_box2_item)

        f_txt2 = QGraphicsTextItem(f"d = Δt_SP · [vp·vs / (vp - vs)] = {dist_a_km:.1f} km")
        f_txt2.setFont(QFont("Consolas", 7))
        f_txt2.setDefaultTextColor(QColor("#38bdf8"))
        f_txt2.setPos(t_x0 + 8.0, p2_y + p2_h - 28.0)
        f_txt2.setZValue(10.25)
        items.append(f_txt2)

        # 5. Bottom Quantitative Metrics Bar
        m_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 36.0, 52.0)
        m_box.setBrush(QBrush(QColor("#0d1526")))
        m_box.setPen(QPen(QColor("#1e293b"), 1.0))
        m_box.setZValue(10.15)
        items.append(m_box)

        k_omori = (vp * vs) / max(0.1, vp - vs)
        m_txt = QGraphicsTextItem(
            f"• Phương trình Omori: d = k·Δt_SP  ➔  Hệ số vận tốc k = (vp·vs)/(vp - vs) = {k_omori:.2f} km/s\n"
            f"• Trạm A: dA = {dist_a_km:.1f} km (Δt_SP = {sp_lag_a_s:.1f} s)    • Sóng P (dọc): vp = {vp:.1f} km/s    • Sóng S (ngang): vs = {vs:.1f} km/s    • Độ sâu h = {depth_km:.1f} km"
        )
        m_txt.setFont(QFont("Consolas", 8))
        m_txt.setDefaultTextColor(QColor("#e2e8f0"))
        m_txt.setPos(p1_x + 10.0, y0 + 274.0)
        m_txt.setZValue(10.25)
        items.append(m_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương pháp đo đạc tam giác 3 trạm địa chấn (Triangulation) xác định duy nhất vị trí tâm chấn qua giao điểm của 3 đường tròn khoảng cách. "
            "Độ trễ thời gian giữa sóng dọc P (nhanh nhất) và sóng ngang S (chậm hơn, không truyền qua chất lỏng) là cơ sở của hệ thống cảnh báo sớm động đất (EEW) toàn cầu."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_mass_spectrometer_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CHLORINE_ISOTOPE_35_37",
        element_name: str = "Clo (Chlorine)",
        e_field_V_m: float = 50000.0,
        b1_Tesla: float = 0.20,
        b2_Tesla: float = 0.30,
        velocity_m_s: float = 250000.0,
        iso0_symbol: str = "³⁵Cl⁺",
        iso0_radius_mm: float = 45.3,
        iso1_symbol: str = "³⁷Cl⁺",
        iso1_radius_mm: float = 47.9,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of the Mass Spectrometer,
        Wien Velocity Selector, and Isotopic Semicircular Magnetic Deflection.
        Illustrates cross-field velocity filtering v0 = E/B1, circular trajectories r = (m·v)/(q·B2),
        impact detector separation, and isotopic abundance mass spectrum peaks.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#080c14")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 195.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("⚡ ATOMIC • MASS SPECTROMETRY")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("MÁY KHỐI PHỔ & BỘ LỌC VẬN TỐC WIEN")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 220.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Nguyên tố: {element_name} | v₀ = {velocity_m_s/1000.0:.1f} km/s | E = {e_field_V_m/1000.0:.0f} kV/m, B₁ = {b1_Tesla:.2f} T, B₂ = {b2_Tesla:.2f} T"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Physical Spectrometer Apparatus
        p1_x = x0 + 18.0
        p1_y = y0 + 64.0
        p1_w = 380.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#050811")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Lọc Vận Tốc Wien (v₀ = E/B₁) & Buồng Từ Lệch B₂")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # Ion source box
        src_x = p1_x + 22.0
        src_y = p1_y + 32.0
        src_w = 34.0
        src_h = 24.0
        src_box = QGraphicsRectItem(src_x, src_y, src_w, src_h)
        src_box.setBrush(QBrush(QColor("#334155")))
        src_box.setPen(QPen(QColor("#64748b"), 1.0))
        src_box.setZValue(10.2)
        items.append(src_box)

        src_txt = QGraphicsTextItem("ION")
        src_txt.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        src_txt.setDefaultTextColor(QColor("#ffffff"))
        src_txt.setPos(src_x + 6.0, src_y + 4.0)
        src_txt.setZValue(10.25)
        items.append(src_txt)

        # Wien Filter plates
        wien_x = src_x + src_w / 2.0 - 12.0
        wien_y = src_y + src_h + 4.0
        wien_w = 24.0
        wien_h = 45.0

        wp_l = QGraphicsRectItem(wien_x - 3.0, wien_y, 3.0, wien_h)
        wp_l.setBrush(QBrush(QColor("#f43f5e")))
        wp_l.setPen(Qt.PenStyle.NoPen)
        wp_l.setZValue(10.2)
        items.append(wp_l)

        wp_r = QGraphicsRectItem(wien_x + wien_w, wien_y, 3.0, wien_h)
        wp_r.setBrush(QBrush(QColor("#0ea5e9")))
        wp_r.setPen(Qt.PenStyle.NoPen)
        wp_r.setZValue(10.2)
        items.append(wp_r)

        # Slit
        slit_y = wien_y + wien_h + 4.0
        slit_x = wien_x + wien_w / 2.0
        slit_l = QGraphicsLineItem(p1_x + 10.0, slit_y, slit_x - 4.0, slit_y)
        slit_l.setPen(QPen(QColor("#64748b"), 2.0))
        slit_l.setZValue(10.2)
        items.append(slit_l)

        slit_r = QGraphicsLineItem(slit_x + 4.0, slit_y, p1_x + p1_w - 10.0, slit_y)
        slit_r.setPen(QPen(QColor("#64748b"), 2.0))
        slit_r.setZValue(10.2)
        items.append(slit_r)

        # Deflection chamber B2 fill
        b2_rect = QGraphicsRectItem(p1_x + 10.0, slit_y + 2.0, p1_w - 20.0, p1_h - (slit_y - p1_y) - 10.0)
        b2_rect.setBrush(QBrush(QColor(14, 165, 233, 20)))
        b2_rect.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
        b2_rect.setZValue(10.15)
        items.append(b2_rect)

        # Semicircular paths
        # Scale max radius ~50mm to ~110px
        r_scale = 110.0 / max(1.0, max(iso0_radius_mm, iso1_radius_mm))
        r0_px = iso0_radius_mm * r_scale
        r1_px = iso1_radius_mm * r_scale

        # Isotope 0 Arc
        path0 = QPainterPath()
        path0.moveTo(slit_x, slit_y)
        path0.arcTo(QRectF(slit_x, slit_y - r0_px, 2.0 * r0_px, 2.0 * r0_px), 180.0, -180.0)
        arc0_item = QGraphicsPathItem(path0)
        arc0_item.setPen(QPen(QColor("#38bdf8"), 1.8))
        arc0_item.setZValue(10.25)
        items.append(arc0_item)

        # Impact 0
        imp0 = QGraphicsEllipseItem(slit_x + 2.0 * r0_px - 3.5, slit_y - 3.5, 7.0, 7.0)
        imp0.setBrush(QBrush(QColor("#38bdf8")))
        imp0.setPen(QPen(QColor("#ffffff"), 1.0))
        imp0.setZValue(10.3)
        items.append(imp0)

        t0 = QGraphicsTextItem(iso0_symbol)
        t0.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        t0.setDefaultTextColor(QColor("#38bdf8"))
        t0.setPos(slit_x + 2.0 * r0_px - 14.0, slit_y - 18.0)
        t0.setZValue(10.35)
        items.append(t0)

        # Isotope 1 Arc
        path1 = QPainterPath()
        path1.moveTo(slit_x, slit_y)
        path1.arcTo(QRectF(slit_x, slit_y - r1_px, 2.0 * r1_px, 2.0 * r1_px), 180.0, -180.0)
        arc1_item = QGraphicsPathItem(path1)
        arc1_item.setPen(QPen(QColor("#f43f5e"), 1.8))
        arc1_item.setZValue(10.25)
        items.append(arc1_item)

        # Impact 1
        imp1 = QGraphicsEllipseItem(slit_x + 2.0 * r1_px - 3.5, slit_y - 3.5, 7.0, 7.0)
        imp1.setBrush(QBrush(QColor("#f43f5e")))
        imp1.setPen(QPen(QColor("#ffffff"), 1.0))
        imp1.setZValue(10.3)
        items.append(imp1)

        t1 = QGraphicsTextItem(iso1_symbol)
        t1.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        t1.setDefaultTextColor(QColor("#f43f5e"))
        t1.setPos(slit_x + 2.0 * r1_px - 14.0, slit_y - 18.0)
        t1.setZValue(10.35)
        items.append(t1)

        # 4. Right Panel: Mass Spectrum Peak Intensity Chart (m/z)
        p2_x = x0 + 408.0
        p2_y = y0 + 64.0
        p2_w = 254.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#050811")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Phổ Khối Đồng Vị Thực Tế (m/z)")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        ox = p2_x + 36.0
        oy = p2_y + p2_h - 32.0
        ax_w = p2_w - 50.0
        ax_h = 120.0

        # Axes
        ax_m = QGraphicsLineItem(ox, oy, ox + ax_w, oy)
        ax_m.setPen(QPen(QColor("#64748b"), 1.2))
        ax_m.setZValue(10.2)
        items.append(ax_m)

        ax_i = QGraphicsLineItem(ox, oy, ox, oy - ax_h - 6.0)
        ax_i.setPen(QPen(QColor("#64748b"), 1.2))
        ax_i.setZValue(10.2)
        items.append(ax_i)

        # Two peak bars (e.g. 75.8% and 24.2%)
        b1_w = 26.0
        b1_x = ox + 35.0
        b1_h = 0.758 * ax_h
        bar1 = QGraphicsRectItem(b1_x, oy - b1_h, b1_w, b1_h)
        bar1.setBrush(QBrush(QColor("#38bdf8")))
        bar1.setPen(QPen(QColor("#ffffff"), 0.8))
        bar1.setZValue(10.25)
        items.append(bar1)

        b1_tag = QGraphicsTextItem(iso0_symbol)
        b1_tag.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        b1_tag.setDefaultTextColor(QColor("#38bdf8"))
        b1_tag.setPos(b1_x - 4.0, oy + 4.0)
        b1_tag.setZValue(10.3)
        items.append(b1_tag)

        b2_w = 26.0
        b2_x = ox + 105.0
        b2_h = 0.242 * ax_h
        bar2 = QGraphicsRectItem(b2_x, oy - b2_h, b2_w, b2_h)
        bar2.setBrush(QBrush(QColor("#f43f5e")))
        bar2.setPen(QPen(QColor("#ffffff"), 0.8))
        bar2.setZValue(10.25)
        items.append(bar2)

        b2_tag = QGraphicsTextItem(iso1_symbol)
        b2_tag.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        b2_tag.setDefaultTextColor(QColor("#f43f5e"))
        b2_tag.setPos(b2_x - 4.0, oy + 4.0)
        b2_tag.setZValue(10.3)
        items.append(b2_tag)

        # 5. Bottom Quantitative Metrics Bar
        m_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 36.0, 52.0)
        m_box.setBrush(QBrush(QColor("#0d1526")))
        m_box.setPen(QPen(QColor("#1e293b"), 1.0))
        m_box.setZValue(10.15)
        items.append(m_box)

        dr_mm = abs(iso1_radius_mm - iso0_radius_mm)
        m_txt = QGraphicsTextItem(
            f"• Lọc vận tốc Wien: v₀ = E/B₁ = {velocity_m_s/1000.0:.1f} km/s  ➔  Bán kính quỹ đạo r = (m·v₀)/(q·B₂) = (m·E)/(q·B₁·B₂)\n"
            f"• {iso0_symbol}: r = {iso0_radius_mm:.2f} mm (D = {2*iso0_radius_mm:.2f} mm)    • {iso1_symbol}: r = {iso1_radius_mm:.2f} mm (D = {2*iso1_radius_mm:.2f} mm)    • Độ phân tách đầu dò: ΔD = {2*dr_mm:.2f} mm"
        )
        m_txt.setFont(QFont("Consolas", 8))
        m_txt.setDefaultTextColor(QColor("#e2e8f0"))
        m_txt.setPos(p1_x + 10.0, y0 + 274.0)
        m_txt.setZValue(10.25)
        items.append(m_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Máy khối phổ (Francis Aston, 1919) kết hợp bộ lọc vận tốc chéo trường Wien (E ⟂ B₁) và buồng làm lệch từ trường B₂ "
            "để phân tách các hạt ion có cùng điện tích theo tỷ số khối lượng trên điện tích (m/z). Bán kính quỹ đạo tỷ lệ thuận trực tiếp với khối lượng ion "
            "là nền tảng cho việc xác định thành phần đồng vị, cấu trúc phân tử và phân tích khối phổ hiện đại."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_hall_effect_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR",
        material_name: str = "Indium Antimonide (InSb n-type)",
        carrier_type: str = "electron",
        current_mA: float = 20.0,
        b_field_T: float = 0.50,
        thickness_um: float = 10.0,
        carrier_density_m3: float = 2.0e22,
        hall_voltage_V: float = -0.0156,
        hall_coefficient_m3_C: float = -3.12e-4,
        drift_velocity_m_s: float = 1248.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of the Hall Effect,
        Lorentz Transverse Force, Carrier Drift Velocity, and Hall Voltage.
        Illustrates isometric 3D semiconductor slab, perpendicular B-field,
        longitudinal current I, voltage probes, surface charge separation,
        and analytical equation V_H = (I·B) / (n·q·d).
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f18")))
        bg.setPen(QPen(QColor("#1e293b"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Badges
        title = QGraphicsTextItem("⚡ HIỆU ỨNG HALL & ĐỘ LINH ĐỘNG LORENTZ (EDWIN HALL, 1879)")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#f8fafc"))
        title.setPos(x0 + 16.0, y0 + 12.0)
        title.setZValue(10.2)
        items.append(title)

        is_hole = (carrier_type == "hole")
        sub_str = f"Vật liệu: {material_name} | Hạt mang điện đa số: {'Lỗ trống (p-type)' if is_hole else 'Điện tử (n-type / kim loại)'}"
        sub = QGraphicsTextItem(sub_str)
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 16.0, y0 + 32.0)
        sub.setZValue(10.2)
        items.append(sub)

        # 3. 3D Isometric Semiconductor Slab
        ox = x0 + 70.0
        oy = y0 + 230.0
        length_px = 150.0
        height_px = 50.0
        tilt_x = 65.0
        tilt_y = -38.0

        p_A = QPointF(ox, oy)
        p_B = QPointF(ox + length_px, oy)
        p_C = QPointF(ox + length_px, oy - height_px)
        p_D = QPointF(ox, oy - height_px)

        p_C_back = QPointF(ox + length_px + tilt_x, oy - height_px + tilt_y)
        p_D_back = QPointF(ox + tilt_x, oy - height_px + tilt_y)
        p_B_back = QPointF(ox + length_px + tilt_x, oy + tilt_y)

        # Front Face
        front_poly = QPolygonF([p_A, p_B, p_C, p_D])
        front_item = QGraphicsPolygonItem(front_poly)
        front_item.setBrush(QBrush(QColor("#0f2a4a")))
        front_item.setPen(QPen(QColor("#38bdf8"), 1.2))
        front_item.setZValue(10.1)
        items.append(front_item)

        # Top Face
        top_poly = QPolygonF([p_D, p_C, p_C_back, p_D_back])
        top_item = QGraphicsPolygonItem(top_poly)
        top_item.setBrush(QBrush(QColor("#1e3a5f")))
        top_item.setPen(QPen(QColor("#38bdf8"), 1.2))
        top_item.setZValue(10.1)
        items.append(top_item)

        # Side Face
        side_poly = QPolygonF([p_B, p_B_back, p_C_back, p_C])
        side_item = QGraphicsPolygonItem(side_poly)
        side_item.setBrush(QBrush(QColor("#0b1e36")))
        side_item.setPen(QPen(QColor("#38bdf8"), 1.2))
        side_item.setZValue(10.1)
        items.append(side_item)

        # 4. Longitudinal Current I Vectors (Gold)
        i_pen = QPen(QColor("#f59e0b"), 2.0)
        i_in = QGraphicsLineItem(ox - 35.0, oy - height_px * 0.5, ox, oy - height_px * 0.5)
        i_in.setPen(i_pen)
        i_in.setZValue(10.2)
        items.append(i_in)

        lbl_i = QGraphicsTextItem(f"I = {current_mA:.1f} mA →")
        lbl_i.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_i.setDefaultTextColor(QColor("#f59e0b"))
        lbl_i.setPos(ox - 50.0, oy - height_px * 0.5 - 20.0)
        lbl_i.setZValue(10.2)
        items.append(lbl_i)

        # 5. Vertical Magnetic Field B Vectors (Purple)
        b_pen = QPen(QColor("#c084fc"), 1.8)
        for fx in (0.35, 0.65):
            bx = ox + length_px * fx + tilt_x * 0.5
            by_top = oy - height_px + tilt_y * 0.5 - 35.0
            by_bot = oy - height_px + tilt_y * 0.5 + 20.0
            b_line = QGraphicsLineItem(bx, by_bot, bx, by_top)
            b_line.setPen(b_pen)
            b_line.setZValue(10.2)
            items.append(b_line)

        lbl_b = QGraphicsTextItem(f"B⃗ = {b_field_T:+.2f} T (↑ Từ trường)")
        lbl_b.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_b.setDefaultTextColor(QColor("#c084fc"))
        lbl_b.setPos(ox + 40.0, oy - height_px + tilt_y - 30.0)
        lbl_b.setZValue(10.2)
        items.append(lbl_b)

        # 6. Voltage Probe & Digital Voltmeter Box
        vm_rect = QGraphicsRectItem(ox + 10.0, oy + 25.0, 240.0, 36.0)
        vm_rect.setBrush(QBrush(QColor("#042f2e")))
        vm_rect.setPen(QPen(QColor("#14b8a6"), 1.4))
        vm_rect.setZValue(10.2)
        items.append(vm_rect)

        abs_v = abs(hall_voltage_V)
        if abs_v < 1e-4:
            v_str = f"{hall_voltage_V * 1e6:+.2f} µV"
        elif abs_v < 1.0:
            v_str = f"{hall_voltage_V * 1e3:+.2f} mV"
        else:
            v_str = f"{hall_voltage_V:+.4f} V"

        lbl_vm = QGraphicsTextItem(f"V_H = {v_str}")
        lbl_vm.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        lbl_vm.setDefaultTextColor(QColor("#2dd4bf"))
        lbl_vm.setPos(ox + 30.0, oy + 32.0)
        lbl_vm.setZValue(10.3)
        items.append(lbl_vm)

        # 7. Telemetry & Formula Panel (Right Side)
        panel_rect = QGraphicsRectItem(x0 + 340.0, y0 + 60.0, 320.0, 240.0)
        panel_rect.setBrush(QBrush(QColor("#0f172a")))
        panel_rect.setPen(QPen(QColor("#334155"), 1.0))
        panel_rect.setZValue(10.1)
        items.append(panel_rect)

        thdr = QGraphicsTextItem("📊 THÔNG SỐ VẬT LÝ & ĐỊNH LUẬT HALL")
        thdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        thdr.setDefaultTextColor(QColor("#fbbf24"))
        thdr.setPos(x0 + 350.0, y0 + 68.0)
        thdr.setZValue(10.2)
        items.append(thdr)

        lines_info = [
            ("Công thức Hall:", "V_H = (I · B) / (n · q · d)"),
            ("Hệ số Hall R_H:", f"{hall_coefficient_m3_C:+.3e} m³/C"),
            ("Mật độ hạt n:", f"{carrier_density_m3:.2e} m⁻³"),
            ("Vận tốc trôi v_d:", f"{drift_velocity_m_s:.2e} m/s"),
            ("Chiều dày phiến d:", f"{thickness_um:.1f} µm"),
            ("Dòng điện I:", f"{current_mA:.1f} mA"),
            ("Từ trường B:", f"{b_field_T:+.2f} T"),
            ("Cực tính Hall:", "Dương (+, Lỗ trống)" if is_hole else "Âm (−, Điện tử)"),
        ]

        curr_y = y0 + 94.0
        for lbl_k, lbl_v in lines_info:
            txt_k = QGraphicsTextItem(lbl_k)
            txt_k.setFont(QFont("Segoe UI", 8))
            txt_k.setDefaultTextColor(QColor("#94a3b8"))
            txt_k.setPos(x0 + 350.0, curr_y)
            txt_k.setZValue(10.2)
            items.append(txt_k)

            txt_v = QGraphicsTextItem(lbl_v)
            txt_v.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            txt_v.setDefaultTextColor(QColor("#f8fafc"))
            txt_v.setPos(x0 + 475.0, curr_y)
            txt_v.setZValue(10.2)
            items.append(txt_v)
            curr_y += 18.0

        # 8. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Hiệu ứng Hall (1879) là phương pháp thực nghiệm quyết định cho phép đo chính xác mật độ n và dấu điện tích của hạt mang điện đa số. "
            "Sự cân bằng giữa lực từ Lorentz q(v x B) và lực điện trường Hall q·E_H sinh ra hiệu điện thế V_H tỷ lệ nghịch với độ dày d và mật độ hạt n."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items

    @staticmethod
    def create_radiation_shielding_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "COBALT_60_GAMMA_LEAD_SHIELDING",
        isotope_name: str = "Cobalt-60 (⁶⁰Co, γ = 1.25 MeV)",
        radiation_type: str = "gamma",
        absorber_material: str = "Chì (Lead - Pb)",
        thickness_cm: float = 2.50,
        i0_cps: float = 1000.0,
        i_trans_cps: float = 194.5,
        transmission_pct: float = 19.45,
        hvl_cm: float = 1.058,
        tvl_cm: float = 3.515,
        linear_mu_cm_inv: float = 0.655,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Nuclear Radiation Penetration,
        Exponential Attenuation Law I(x) = I0 * exp(-mu * x), and Half-Value Layer (HVL).
        Illustrates radioactive source collimator (with trefoil ☢), shielding barrier,
        Geiger-Müller detector tube, transmitted ray intensity, and exact mathematical parameters.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f18")))
        bg.setPen(QPen(QColor("#1e293b"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Subtitle
        title = QGraphicsTextItem("☢ ĐỘ XUYÊN SÂU PHÓNG XẠ & ĐỊNH LUẬT SUY GIẢM HÀM MŨ (HVL / TVL)")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#f8fafc"))
        title.setPos(x0 + 16.0, y0 + 12.0)
        title.setZValue(10.2)
        items.append(title)

        sub_str = f"Nguồn bức xạ: {isotope_name} | Vật liệu chắn: {absorber_material}"
        sub = QGraphicsTextItem(sub_str)
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 16.0, y0 + 32.0)
        sub.setZValue(10.2)
        items.append(sub)

        # 3. Radiation Bench Graphics (Source -> Shield -> Geiger Tube)
        bench_y = y0 + 180.0

        # Radioactive Source Block with Trefoil
        src_block = QGraphicsRectItem(x0 + 35.0, bench_y - 45.0, 50.0, 65.0)
        src_block.setBrush(QBrush(QColor("#334155")))
        src_block.setPen(QPen(QColor("#eab308"), 1.5))
        src_block.setZValue(10.1)
        items.append(src_block)

        lbl_rad = QGraphicsTextItem("☢")
        lbl_rad.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        lbl_rad.setDefaultTextColor(QColor("#eab308"))
        lbl_rad.setPos(x0 + 47.0, bench_y - 36.0)
        lbl_rad.setZValue(10.2)
        items.append(lbl_rad)

        lbl_i0 = QGraphicsTextItem(f"I₀ = {i0_cps:.0f} cps")
        lbl_i0.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_i0.setDefaultTextColor(QColor("#facc15"))
        lbl_i0.setPos(x0 + 28.0, bench_y - 66.0)
        lbl_i0.setZValue(10.2)
        items.append(lbl_i0)

        # Shielding Absorber Barrier
        bar_thick_px = max(6.0, min(75.0, thickness_cm * 6.5))
        bar_x = x0 + 155.0
        bar_block = QGraphicsRectItem(bar_x, bench_y - 55.0, bar_thick_px, 85.0)
        bar_block.setBrush(QBrush(QColor("#475569")))
        bar_block.setPen(QPen(QColor("#94a3b8"), 1.4))
        bar_block.setZValue(10.1)
        items.append(bar_block)

        lbl_bar = QGraphicsTextItem(f"x = {thickness_cm:.2f} cm\n({absorber_material[:6]})")
        lbl_bar.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_bar.setDefaultTextColor(QColor("#e2e8f0"))
        lbl_bar.setPos(bar_x - 10.0, bench_y - 88.0)
        lbl_bar.setZValue(10.2)
        items.append(lbl_bar)

        # Geiger-Müller Detector Tube
        gm_box = QGraphicsRectItem(x0 + 265.0, bench_y - 40.0, 55.0, 55.0)
        gm_box.setBrush(QBrush(QColor("#042f2e")))
        gm_box.setPen(QPen(QColor("#14b8a6"), 1.4))
        gm_box.setZValue(10.1)
        items.append(gm_box)

        lbl_gm = QGraphicsTextItem("Ống GM")
        lbl_gm.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_gm.setDefaultTextColor(QColor("#2dd4bf"))
        lbl_gm.setPos(x0 + 268.0, bench_y - 25.0)
        lbl_gm.setZValue(10.2)
        items.append(lbl_gm)

        # Radiation Ray Lines
        ray_pen = QPen(QColor("#c084fc"), 1.5, Qt.PenStyle.DashLine)
        ray_in = QGraphicsLineItem(x0 + 85.0, bench_y - 12.0, bar_x, bench_y - 12.0)
        ray_in.setPen(ray_pen)
        ray_in.setZValue(10.2)
        items.append(ray_in)

        ray_out = QGraphicsLineItem(bar_x + bar_thick_px, bench_y - 12.0, x0 + 265.0, bench_y - 12.0)
        ray_out.setPen(QPen(QColor("#c084fc"), 1.0, Qt.PenStyle.DotLine))
        ray_out.setZValue(10.2)
        items.append(ray_out)

        # Transmitted Count Banner below bench
        cnt_banner = QGraphicsRectItem(x0 + 35.0, bench_y + 40.0, 285.0, 36.0)
        cnt_banner.setBrush(QBrush(QColor("#022c22")))
        cnt_banner.setPen(QPen(QColor("#059669"), 1.2))
        cnt_banner.setZValue(10.1)
        items.append(cnt_banner)

        lbl_cnt = QGraphicsTextItem(f"Cường độ truyền qua: I = {i_trans_cps:.1f} cps  ({transmission_pct:.2f}% I₀)")
        lbl_cnt.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        lbl_cnt.setDefaultTextColor(QColor("#34d399"))
        lbl_cnt.setPos(x0 + 42.0, bench_y + 48.0)
        lbl_cnt.setZValue(10.2)
        items.append(lbl_cnt)

        # 4. Telemetry and Formula Panel (Right Side)
        panel_rect = QGraphicsRectItem(x0 + 345.0, y0 + 60.0, 315.0, 240.0)
        panel_rect.setBrush(QBrush(QColor("#0f172a")))
        panel_rect.setPen(QPen(QColor("#334155"), 1.0))
        panel_rect.setZValue(10.1)
        items.append(panel_rect)

        thdr = QGraphicsTextItem("📊 THÔNG SỐ VẬT LÝ & ĐỊNH LUẬT SUY GIẢM")
        thdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        thdr.setDefaultTextColor(QColor("#fbbf24"))
        thdr.setPos(x0 + 355.0, y0 + 68.0)
        thdr.setZValue(10.2)
        items.append(thdr)

        hvl_count = thickness_cm / max(1e-4, hvl_cm)
        lines_info = [
            ("Định luật suy giảm:", "I(x) = I₀ · e^(−µ·x)"),
            ("Hệ số suy giảm tuyến tính µ:", f"{linear_mu_cm_inv:.4f} cm⁻¹"),
            ("Bề dày bán giảm (HVL):", f"{hvl_cm:.3f} cm  (ln2/µ)"),
            ("Bề dày giảm mười lần (TVL):", f"{tvl_cm:.3f} cm  (ln10/µ)"),
            ("Số lớp HVL tương đương:", f"{hvl_count:.2f} × HVL"),
            ("Tỷ lệ suy giảm liều (%A):", f"{100.0 - transmission_pct:.2f}%"),
            ("Cường độ nguồn ban đầu I₀:", f"{i0_cps:.0f} cps"),
            ("Cường độ còn lại I(x):", f"{i_trans_cps:.1f} cps"),
        ]

        curr_y = y0 + 94.0
        for lbl_k, lbl_v in lines_info:
            txt_k = QGraphicsTextItem(lbl_k)
            txt_k.setFont(QFont("Segoe UI", 8))
            txt_k.setDefaultTextColor(QColor("#94a3b8"))
            txt_k.setPos(x0 + 355.0, curr_y)
            txt_k.setZValue(10.2)
            items.append(txt_k)

            txt_v = QGraphicsTextItem(lbl_v)
            txt_v.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            txt_v.setDefaultTextColor(QColor("#f8fafc"))
            txt_v.setPos(x0 + 510.0, curr_y)
            txt_v.setZValue(10.2)
            items.append(txt_v)
            curr_y += 18.0

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Định luật suy giảm hàm mũ mô tả xác suất tương tác của bức xạ ion hóa với vật chất qua hiệu ứng quang điện, tán xạ Compton và tạo cặp. "
            "Bề dày bán giảm (Half-Value Layer HVL) là thông số kỹ thuật cốt lõi trong an toàn bức xạ và thiết kế che chắn che chở lò phản ứng."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items

    @staticmethod
    def create_maxwell_boltzmann_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HELIUM_GAS_ROOM_TEMP_300K",
        gas_name: str = "Helium (He)",
        formula: str = "⁴He",
        temperature_K: float = 300.0,
        pressure_atm: float = 1.0,
        molar_mass_kg_mol: float = 0.0040026,
        vp_m_s: float = 1115.8,
        v_avg_m_s: float = 1259.0,
        v_rms_m_s: float = 1366.6,
        mean_free_path_nm: float = 197.5,
        collision_freq_GHz: float = 6.37,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of the Maxwell-Boltzmann
        Molecular Speed Distribution and Kinetic Gas Theory.
        Illustrates mathematical distribution curve f(v), characteristic speeds
        (v_p < <v> < v_rms), 2D kinetic chamber, and statistical mechanics parameters.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f18")))
        bg.setPen(QPen(QColor("#1e293b"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Subtitle
        title = QGraphicsTextItem("🔥 PHÂN BỐ VẬN TỐC PHÂN TỬ MAXWELL-BOLTZMANN")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#f8fafc"))
        title.setPos(x0 + 16.0, y0 + 12.0)
        title.setZValue(10.2)
        items.append(title)

        t_c = temperature_K - 273.15
        sub_str = f"Chất khí: {gas_name} ({formula}) | T = {temperature_K:.1f} K ({t_c:+.1f}°C) | P = {pressure_atm:.1f} atm"
        sub = QGraphicsTextItem(sub_str)
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 16.0, y0 + 32.0)
        sub.setZValue(10.2)
        items.append(sub)

        # 3. Left Panel: 2D Kinetic Gas Container Schematic
        c_box = QGraphicsRectItem(x0 + 20.0, y0 + 56.0, 310.0, 160.0)
        c_box.setBrush(QBrush(QColor("#0b1329")))
        c_box.setPen(QPen(QColor("#38bdf8"), 1.2))
        c_box.setZValue(10.1)
        items.append(c_box)

        lbl_c = QGraphicsTextItem("MÔ HÌNH CHẤT KHÍ LÝ TƯỞNG (2D KINETIC GAS CHAMBER)")
        lbl_c.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_c.setDefaultTextColor(QColor("#38bdf8"))
        lbl_c.setPos(x0 + 28.0, y0 + 64.0)
        lbl_c.setZValue(10.2)
        items.append(lbl_c)

        # Sample gas molecules inside chamber
        for i in range(25):
            mx = x0 + 40.0 + (i * 37) % 270.0
            my = y0 + 90.0 + (i * 23) % 110.0
            p_itm = QGraphicsEllipseItem(mx, my, 6.0, 6.0)
            col = QColor("#38bdf8") if i % 3 == 0 else (QColor("#facc15") if i % 3 == 1 else QColor("#ef4444"))
            p_itm.setBrush(QBrush(col))
            p_itm.setPen(Qt.PenStyle.NoPen)
            p_itm.setZValue(10.2)
            items.append(p_itm)

        # Telemetry Table below chamber
        t_box = QGraphicsRectItem(x0 + 20.0, y0 + 224.0, 310.0, 105.0)
        t_box.setBrush(QBrush(QColor("#0f172a")))
        t_box.setPen(QPen(QColor("#334155"), 1.0))
        t_box.setZValue(10.1)
        items.append(t_box)

        lines_char = [
            ("Tốc độ xác suất cực đại v_p:", f"{vp_m_s:6.1f} m/s", QColor("#f87171")),
            ("Tốc độ trung bình <v>:", f"{v_avg_m_s:6.1f} m/s", QColor("#4ade80")),
            ("Tốc độ căn quân phương v_rms:", f"{v_rms_m_s:6.1f} m/s", QColor("#38bdf8")),
            ("Quãng đường tự do λ:", f"{mean_free_path_nm:.1f} nm", QColor("#fbbf24")),
        ]
        cy = y0 + 230.0
        for l_k, l_v, col_v in lines_char:
            txt_k = QGraphicsTextItem(l_k)
            txt_k.setFont(QFont("Segoe UI", 7))
            txt_k.setDefaultTextColor(QColor("#94a3b8"))
            txt_k.setPos(x0 + 28.0, cy)
            txt_k.setZValue(10.2)
            items.append(txt_k)

            txt_v = QGraphicsTextItem(l_v)
            txt_v.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            txt_v.setDefaultTextColor(col_v)
            txt_v.setPos(x0 + 215.0, cy)
            txt_v.setZValue(10.2)
            items.append(txt_v)
            cy += 22.0

        # 4. Right Panel: Maxwell-Boltzmann Distribution Curve f(v) Plot
        plot_rect = QGraphicsRectItem(x0 + 345.0, y0 + 56.0, 315.0, 273.0)
        plot_rect.setBrush(QBrush(QColor("#0f172a")))
        plot_rect.setPen(QPen(QColor("#334155"), 1.0))
        plot_rect.setZValue(10.1)
        items.append(plot_rect)

        thdr = QGraphicsTextItem("📈 ĐỒ THỊ HÀM MẬT ĐỘ XÁC SUẤT f(v)")
        thdr.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr.setDefaultTextColor(QColor("#fbbf24"))
        thdr.setPos(x0 + 355.0, y0 + 64.0)
        thdr.setZValue(10.2)
        items.append(thdr)

        # Plot axes
        ax_x0 = x0 + 380.0
        ax_y0 = y0 + 290.0
        ax_w = 260.0
        ax_h = 180.0

        v_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0)
        v_axis.setPen(QPen(QColor("#475569"), 1.2))
        v_axis.setZValue(10.2)
        items.append(v_axis)

        f_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h)
        f_axis.setPen(QPen(QColor("#475569"), 1.2))
        f_axis.setZValue(10.2)
        items.append(f_axis)

        # Markers for vp, v_avg, v_rms on plot
        max_v = max(1500.0, v_rms_m_s * 2.8)
        for val_v, name_v, col_m in [
            (vp_m_s, "v_p", QColor("#f87171")),
            (v_avg_m_s, "<v>", QColor("#4ade80")),
            (v_rms_m_s, "v_rms", QColor("#38bdf8")),
        ]:
            if val_v <= max_v:
                mx = ax_x0 + (val_v / max_v) * ax_w
                m_line = QGraphicsLineItem(mx, ax_y0, mx, ax_y0 - ax_h * 0.90)
                m_line.setPen(QPen(col_m, 1.2, Qt.PenStyle.DashLine))
                m_line.setZValue(10.2)
                items.append(m_line)

                lbl_m = QGraphicsTextItem(name_v)
                lbl_m.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                lbl_m.setDefaultTextColor(col_m)
                lbl_m.setPos(mx - 8.0, ax_y0 - ax_h * 0.98)
                lbl_m.setZValue(10.2)
                items.append(lbl_m)

        # Formula text in plot box
        f_formula = QGraphicsTextItem("f(v) = 4π (M / 2πRT)^(3/2) · v² · e^(-Mv²/2RT)")
        f_formula.setFont(QFont("Consolas", 7))
        f_formula.setDefaultTextColor(QColor("#94a3b8"))
        f_formula.setPos(x0 + 355.0, y0 + 82.0)
        f_formula.setZValue(10.2)
        items.append(f_formula)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Phân bố Maxwell-Boltzmann (1860) là cầu nối căn bản giữa nhiệt động lực học vĩ mô (T, P) và cơ học thống kê vi mô. "
            "Quan hệ bất biến v_p < <v> < v_rms (tỷ lệ 1 : 1.128 : 1.225) luôn đúng cho mọi chất khí lý tưởng ở trạng thái cân bằng nhiệt động."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items

    @staticmethod
    def create_rlc_resonance_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "AM_RADIO_TUNER_540KHZ_HIGH_Q",
        circuit_name: str = "AM Receiver Tuner (540 kHz)",
        f0_hz: float = 540000.0,
        freq_hz: float = 540000.0,
        r_ohms: float = 10.0,
        l_henries: float = 220.0e-6,
        c_farads: float = 395.7e-12,
        z_ohms: float = 10.0,
        i0_amps: float = 0.10,
        phi_deg: float = 0.0,
        quality_factor_Q: float = 74.6,
        bandwidth_hz: float = 7238.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of AC RLC Resonant Circuit,
        Complex Impedance, 2D Freznel Phasor Vectors, and Bode Resonance Curve.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f18")))
        bg.setPen(QPen(QColor("#1e293b"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Subtitle
        title = QGraphicsTextItem("⚡ CỘNG HƯỞNG RLC & GIẢN ĐỒ VÉC-TƠ PHASOR")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#f8fafc"))
        title.setPos(x0 + 16.0, y0 + 12.0)
        title.setZValue(10.2)
        items.append(title)

        f0_str = f"{f0_hz*1e-3:.1f} kHz" if f0_hz >= 1000.0 else f"{f0_hz:.1f} Hz"
        f_str = f"{freq_hz*1e-3:.1f} kHz" if freq_hz >= 1000.0 else f"{freq_hz:.1f} Hz"
        sub_str = f"Mạch: {circuit_name} | f₀ = {f0_str} | f hoạt động = {f_str} | Q = {quality_factor_Q:.1f}"
        sub = QGraphicsTextItem(sub_str)
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 16.0, y0 + 32.0)
        sub.setZValue(10.2)
        items.append(sub)

        # 3. Left Panel: 2D Phasor Vector Diagram (Freznel)
        p_box = QGraphicsRectItem(x0 + 20.0, y0 + 56.0, 310.0, 160.0)
        p_box.setBrush(QBrush(QColor("#0b1329")))
        p_box.setPen(QPen(QColor("#38bdf8"), 1.2))
        p_box.setZValue(10.1)
        items.append(p_box)

        lbl_p = QGraphicsTextItem("GIẢN ĐỒ VÉC-TƠ QUAY PHASOR (FREZNEL)")
        lbl_p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_p.setDefaultTextColor(QColor("#38bdf8"))
        lbl_p.setPos(x0 + 28.0, y0 + 64.0)
        lbl_p.setZValue(10.2)
        items.append(lbl_p)

        cx = x0 + 150.0
        cy = y0 + 145.0

        # Phasor axes
        ax_real = QGraphicsLineItem(cx - 70.0, cy, cx + 80.0, cy)
        ax_real.setPen(QPen(QColor("#334155"), 1.0))
        ax_real.setZValue(10.15)
        items.append(ax_real)

        ax_imag = QGraphicsLineItem(cx, cy - 65.0, cx, cy + 65.0)
        ax_imag.setPen(QPen(QColor("#334155"), 1.0))
        ax_imag.setZValue(10.15)
        items.append(ax_imag)

        # VR phasor (Green, +X axis)
        vr_line = QGraphicsLineItem(cx, cy, cx + 55.0, cy)
        vr_line.setPen(QPen(QColor("#4ade80"), 2.2))
        vr_line.setZValue(10.2)
        items.append(vr_line)

        lbl_vr = QGraphicsTextItem("V_R (i)")
        lbl_vr.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_vr.setDefaultTextColor(QColor("#4ade80"))
        lbl_vr.setPos(cx + 58.0, cy - 10.0)
        lbl_vr.setZValue(10.2)
        items.append(lbl_vr)

        # VL phasor (Purple, +Y axis)
        vl_line = QGraphicsLineItem(cx, cy, cx, cy - 45.0)
        vl_line.setPen(QPen(QColor("#c084fc"), 2.0))
        vl_line.setZValue(10.2)
        items.append(vl_line)

        lbl_vl = QGraphicsTextItem("V_L (+90°)")
        lbl_vl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_vl.setDefaultTextColor(QColor("#c084fc"))
        lbl_vl.setPos(cx + 4.0, cy - 55.0)
        lbl_vl.setZValue(10.2)
        items.append(lbl_vl)

        # VC phasor (Cyan, -Y axis)
        vc_line = QGraphicsLineItem(cx, cy, cx, cy + 45.0)
        vc_line.setPen(QPen(QColor("#38bdf8"), 2.0))
        vc_line.setZValue(10.2)
        items.append(vc_line)

        lbl_vc = QGraphicsTextItem("V_C (−90°)")
        lbl_vc.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_vc.setDefaultTextColor(QColor("#38bdf8"))
        lbl_vc.setPos(cx + 4.0, cy + 35.0)
        lbl_vc.setZValue(10.2)
        items.append(lbl_vc)

        # V0 phasor (Gold)
        phi_rad = math.radians(phi_deg)
        v0_end_x = cx + 55.0 * math.cos(phi_rad)
        v0_end_y = cy - 55.0 * math.sin(phi_rad)
        v0_line = QGraphicsLineItem(cx, cy, v0_end_x, v0_end_y)
        v0_line.setPen(QPen(QColor("#fbbf24"), 2.5))
        v0_line.setZValue(10.25)
        items.append(v0_line)

        lbl_v0 = QGraphicsTextItem(f"V₀ (φ={phi_deg:+.1f}°)")
        lbl_v0.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_v0.setDefaultTextColor(QColor("#fbbf24"))
        lbl_v0.setPos(v0_end_x + 4.0, v0_end_y - 8.0)
        lbl_v0.setZValue(10.25)
        items.append(lbl_v0)

        # Telemetry Box below Phasor (y = 224 to 329)
        t_box = QGraphicsRectItem(x0 + 20.0, y0 + 224.0, 310.0, 105.0)
        t_box.setBrush(QBrush(QColor("#0f172a")))
        t_box.setPen(QPen(QColor("#334155"), 1.0))
        t_box.setZValue(10.1)
        items.append(t_box)

        i_str = f"{i0_amps*1e3:.2f} mA" if i0_amps < 1.0 else f"{i0_amps:.3f} A"
        bw_str = f"{bandwidth_hz*1e-3:.2f} kHz" if bandwidth_hz >= 1000.0 else f"{bandwidth_hz:.1f} Hz"

        lines_ac = [
            ("Tổng trở |Z|:", f"{z_ohms:.2f} Ω (R = {r_ohms:.1f} Ω)", QColor("#4ade80")),
            ("Dòng cực đại I₀:", i_str, QColor("#38bdf8")),
            ("Độ lệch pha φ:", f"{phi_deg:+.2f}°", QColor("#fbbf24")),
            ("Dải thông −3dB (Δf):", bw_str, QColor("#f43f5e")),
        ]
        cy_t = y0 + 230.0
        for l_k, l_v, col_v in lines_ac:
            txt_k = QGraphicsTextItem(l_k)
            txt_k.setFont(QFont("Segoe UI", 7))
            txt_k.setDefaultTextColor(QColor("#94a3b8"))
            txt_k.setPos(x0 + 28.0, cy_t)
            txt_k.setZValue(10.2)
            items.append(txt_k)

            txt_v = QGraphicsTextItem(l_v)
            txt_v.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            txt_v.setDefaultTextColor(col_v)
            txt_v.setPos(x0 + 195.0, cy_t)
            txt_v.setZValue(10.2)
            items.append(txt_v)
            cy_t += 22.0

        # 4. Right Panel: Bode Resonance Curve I(f)
        plot_rect = QGraphicsRectItem(x0 + 345.0, y0 + 56.0, 315.0, 273.0)
        plot_rect.setBrush(QBrush(QColor("#0f172a")))
        plot_rect.setPen(QPen(QColor("#334155"), 1.0))
        plot_rect.setZValue(10.1)
        items.append(plot_rect)

        thdr = QGraphicsTextItem("📈 ĐƯỜNG CONG CỘNG HƯỞNG DÒNG ĐIỆN I(f)")
        thdr.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr.setDefaultTextColor(QColor("#fbbf24"))
        thdr.setPos(x0 + 355.0, y0 + 64.0)
        thdr.setZValue(10.2)
        items.append(thdr)

        ax_x0 = x0 + 380.0
        ax_y0 = y0 + 290.0
        ax_w = 260.0
        ax_h = 180.0

        f_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0)
        f_axis.setPen(QPen(QColor("#475569"), 1.2))
        f_axis.setZValue(10.2)
        items.append(f_axis)

        i_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h)
        i_axis.setPen(QPen(QColor("#475569"), 1.2))
        i_axis.setZValue(10.2)
        items.append(i_axis)

        # Mark f0 resonant peak line
        f0_line = QGraphicsLineItem(ax_x0 + ax_w * 0.5, ax_y0, ax_x0 + ax_w * 0.5, ax_y0 - ax_h * 0.90)
        f0_line.setPen(QPen(QColor("#fbbf24"), 1.4, Qt.PenStyle.DashLine))
        f0_line.setZValue(10.2)
        items.append(f0_line)

        lbl_f0_pk = QGraphicsTextItem("f₀ (Peak I_max = V₀/R)")
        lbl_f0_pk.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_f0_pk.setDefaultTextColor(QColor("#fbbf24"))
        lbl_f0_pk.setPos(ax_x0 + ax_w * 0.5 - 20.0, ax_y0 - ax_h * 0.98)
        lbl_f0_pk.setZValue(10.2)
        items.append(lbl_f0_pk)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Hiện tượng cộng hưởng điện RLC xảy ra khi tần số nguồn bằng f₀ = 1/(2π√LC), cảm kháng bằng dung kháng (Z_L = Z_C). "
            "Khi đó tổng trở đạt cực tiểu Z = R, dòng điện đạt cực đại cùng pha với điện áp (cosφ = 1), điện áp trên L và C phóng đại gấp Q lần nguồn."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items


    @staticmethod
    def create_stress_strain_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "STRUCTURAL_STEEL_A36_DUCTILE",
        material_name: str = "Thép Kết Cấu A36 (Structural Carbon Steel)",
        youngs_modulus_GPa: float = 200.0,
        yield_strength_MPa: float = 250.0,
        uts_strength_MPa: float = 400.0,
        current_strain_pct: float = 0.125,
        current_stress_MPa: float = 250.0,
        tensile_force_kN: float = 19.63,
        elongation_mm: float = 0.062,
        is_fractured: bool = False,
        regime: str = "Đàn Hồi Tuyến Tính (Hooke)",
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Mechanical Stress-Strain Curve,
        Hooke's Law Elasticity, Young's Modulus E, Yield Strength, UTS Necking and Fracture.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0f172a")))
        bg.setPen(QPen(QColor("#334155"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Material Subtitle
        hdr = QGraphicsTextItem("⚙ ĐƯỜNG CONG ỨNG SUẤT - BIẾN DẠNG (STRESS-STRAIN TENSILE TEST)")
        hdr.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor("#38bdf8"))
        hdr.setPos(x0 + 16.0, y0 + 12.0)
        hdr.setZValue(10.1)
        items.append(hdr)

        sub = QGraphicsTextItem(f"Vật liệu: {material_name} | Trạng thái: {regime}")
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 18.0, y0 + 34.0)
        sub.setZValue(10.1)
        items.append(sub)

        # 3. Left Panel: Hydraulic Tensile Test Bench & Dog-Bone Specimen
        bench_rect = QGraphicsRectItem(x0 + 16.0, y0 + 60.0, 240.0, 255.0)
        bench_rect.setBrush(QBrush(QColor("#1e293b")))
        bench_rect.setPen(QPen(QColor("#334155"), 1.0))
        bench_rect.setZValue(10.1)
        items.append(bench_rect)

        thdr_left = QGraphicsTextItem("🔬 MÁY KÉO THỦY LỰC & MẪU THỬ")
        thdr_left.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr_left.setDefaultTextColor(QColor("#38bdf8"))
        thdr_left.setPos(x0 + 22.0, y0 + 64.0)
        thdr_left.setZValue(10.2)
        items.append(thdr_left)

        # Grips (Upper & Lower)
        cx_grip = x0 + 136.0
        up_grip = QGraphicsRectItem(cx_grip - 40.0, y0 + 90.0, 80.0, 22.0)
        up_grip.setBrush(QBrush(QColor("#475569")))
        up_grip.setPen(QPen(QColor("#94a3b8"), 1.2))
        up_grip.setZValue(10.2)
        items.append(up_grip)

        dn_grip = QGraphicsRectItem(cx_grip - 40.0, y0 + 230.0, 80.0, 22.0)
        dn_grip.setBrush(QBrush(QColor("#475569")))
        dn_grip.setPen(QPen(QColor("#94a3b8"), 1.2))
        dn_grip.setZValue(10.2)
        items.append(dn_grip)

        # Dog-bone specimen
        specimen_color = QColor("#ef4444") if is_fractured else QColor("#38bdf8")
        spec_path = QPainterPath()
        spec_path.moveTo(cx_grip - 20.0, y0 + 112.0)
        spec_path.lineTo(cx_grip + 20.0, y0 + 112.0)
        spec_path.lineTo(cx_grip + 10.0, y0 + 130.0)
        if is_fractured:
            spec_path.lineTo(cx_grip + 8.0, y0 + 168.0)
            spec_path.lineTo(cx_grip - 8.0, y0 + 172.0)
            spec_path.lineTo(cx_grip - 10.0, y0 + 130.0)
            spec_path.closeSubpath()
            spec_path.moveTo(cx_grip + 8.0, y0 + 176.0)
            spec_path.lineTo(cx_grip + 10.0, y0 + 212.0)
            spec_path.lineTo(cx_grip + 20.0, y0 + 230.0)
            spec_path.lineTo(cx_grip - 20.0, y0 + 230.0)
            spec_path.lineTo(cx_grip - 10.0, y0 + 212.0)
            spec_path.lineTo(cx_grip - 8.0, y0 + 180.0)
            spec_path.closeSubpath()
        else:
            spec_path.lineTo(cx_grip + 10.0, y0 + 212.0)
            spec_path.lineTo(cx_grip + 20.0, y0 + 230.0)
            spec_path.lineTo(cx_grip - 20.0, y0 + 230.0)
            spec_path.lineTo(cx_grip - 10.0, y0 + 212.0)
            spec_path.lineTo(cx_grip - 10.0, y0 + 130.0)
            spec_path.closeSubpath()

        spec_item = QGraphicsPathItem(spec_path)
        spec_item.setBrush(QBrush(specimen_color))
        spec_item.setPen(QPen(specimen_color.lighter(130), 1.2))
        spec_item.setZValue(10.2)
        items.append(spec_item)

        # Force arrows
        arr_up = QGraphicsLineItem(cx_grip, y0 + 88.0, cx_grip, y0 + 72.0)
        arr_up.setPen(QPen(QColor("#fbbf24"), 2.0))
        arr_up.setZValue(10.3)
        items.append(arr_up)
        lbl_f_up = QGraphicsTextItem("▲ F_t")
        lbl_f_up.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_f_up.setDefaultTextColor(QColor("#fbbf24"))
        lbl_f_up.setPos(cx_grip + 4.0, y0 + 70.0)
        lbl_f_up.setZValue(10.3)
        items.append(lbl_f_up)

        # Left Readouts Box
        readout_text = (
            f"Lực kéo F: {tensile_force_kN:.2f} kN\n"
            f"Độ giãn dài ΔL: {elongation_mm:.3f} mm\n"
            f"Biến dạng ε: {current_strain_pct:.3f}%\n"
            f"Ứng suất σ: {current_stress_MPa:.1f} MPa"
        )
        ro_item = QGraphicsTextItem(readout_text)
        ro_item.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        ro_item.setDefaultTextColor(QColor("#f8fafc"))
        ro_item.setPos(x0 + 22.0, y0 + 258.0)
        ro_item.setZValue(10.2)
        items.append(ro_item)

        # 4. Right Panel: Stress-Strain Curve Plot (σ vs ε)
        plot_rect = QGraphicsRectItem(x0 + 266.0, y0 + 60.0, 398.0, 255.0)
        plot_rect.setBrush(QBrush(QColor("#0f172a")))
        plot_rect.setPen(QPen(QColor("#334155"), 1.0))
        plot_rect.setZValue(10.1)
        items.append(plot_rect)

        thdr_plot = QGraphicsTextItem("📈 ĐỒ THỊ ỨNG SUẤT - BIẾN DẠNG σ(ε)")
        thdr_plot.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr_plot.setDefaultTextColor(QColor("#38bdf8"))
        thdr_plot.setPos(x0 + 272.0, y0 + 64.0)
        thdr_plot.setZValue(10.2)
        items.append(thdr_plot)

        ax_x0 = x0 + 310.0
        ax_y0 = y0 + 270.0
        ax_w = 330.0
        ax_h = 175.0

        eps_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0)
        eps_axis.setPen(QPen(QColor("#475569"), 1.2))
        eps_axis.setZValue(10.2)
        items.append(eps_axis)

        sig_axis = QGraphicsLineItem(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h)
        sig_axis.setPen(QPen(QColor("#475569"), 1.2))
        sig_axis.setZValue(10.2)
        items.append(sig_axis)

        lbl_eps = QGraphicsTextItem("Biến dạng ε (%) →")
        lbl_eps.setFont(QFont("Segoe UI", 7))
        lbl_eps.setDefaultTextColor(QColor("#94a3b8"))
        lbl_eps.setPos(ax_x0 + ax_w - 90.0, ax_y0 + 2.0)
        lbl_eps.setZValue(10.2)
        items.append(lbl_eps)

        lbl_sig = QGraphicsTextItem("↑ Ứng suất σ (MPa)")
        lbl_sig.setFont(QFont("Segoe UI", 7))
        lbl_sig.setDefaultTextColor(QColor("#94a3b8"))
        lbl_sig.setPos(ax_x0 - 24.0, ax_y0 - ax_h - 18.0)
        lbl_sig.setZValue(10.2)
        items.append(lbl_sig)

        # Plot curve
        # Normalized coordinates for max stress and max strain
        max_sig = max(uts_strength_MPa * 1.15, 100.0)
        max_eps = 30.0

        curve_path = QPainterPath()
        curve_path.moveTo(ax_x0, ax_y0)

        # Key points on curve
        ey_pct = (yield_strength_MPa / (youngs_modulus_GPa * 1000.0)) * 100.0
        euts_pct = 15.0
        ef_pct = 25.0
        fracture_sig = uts_strength_MPa * 0.85

        px_y = ax_x0 + (ey_pct / max_eps) * ax_w
        py_y = ax_y0 - (yield_strength_MPa / max_sig) * ax_h

        px_uts = ax_x0 + (euts_pct / max_eps) * ax_w
        py_uts = ax_y0 - (uts_strength_MPa / max_sig) * ax_h

        px_f = ax_x0 + (ef_pct / max_eps) * ax_w
        py_f = ax_y0 - (fracture_sig / max_sig) * ax_h

        curve_path.lineTo(px_y, py_y)
        curve_path.quadTo((px_y + px_uts) * 0.5, py_uts - 5.0, px_uts, py_uts)
        curve_path.quadTo((px_uts + px_f) * 0.5, py_uts + (py_f - py_uts) * 0.4, px_f, py_f)

        curve_item = QGraphicsPathItem(curve_path)
        curve_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        curve_item.setZValue(10.3)
        items.append(curve_item)

        # Yield dashed guide
        ydash = QGraphicsLineItem(ax_x0, py_y, px_y, py_y)
        ydash.setPen(QPen(QColor("#f59e0b"), 1.0, Qt.PenStyle.DashLine))
        ydash.setZValue(10.2)
        items.append(ydash)

        lbl_sy = QGraphicsTextItem(f"σ_y = {yield_strength_MPa:.0f} MPa")
        lbl_sy.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_sy.setDefaultTextColor(QColor("#f59e0b"))
        lbl_sy.setPos(ax_x0 + 4.0, py_y - 12.0)
        lbl_sy.setZValue(10.3)
        items.append(lbl_sy)

        # UTS dashed guide
        utsdash = QGraphicsLineItem(ax_x0, py_uts, px_uts, py_uts)
        utsdash.setPen(QPen(QColor("#ec4899"), 1.0, Qt.PenStyle.DashLine))
        utsdash.setZValue(10.2)
        items.append(utsdash)

        lbl_suts = QGraphicsTextItem(f"σ_UTS = {uts_strength_MPa:.0f} MPa")
        lbl_suts.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_suts.setDefaultTextColor(QColor("#ec4899"))
        lbl_suts.setPos(px_uts - 40.0, py_uts - 14.0)
        lbl_suts.setZValue(10.3)
        items.append(lbl_suts)

        # Operating Point Marker
        cur_px = ax_x0 + min(max((current_strain_pct / max_eps) * ax_w, 0.0), ax_w)
        cur_py = ax_y0 - min(max((current_stress_MPa / max_sig) * ax_h, 0.0), ax_h)

        marker = QGraphicsEllipseItem(cur_px - 4.5, cur_py - 4.5, 9.0, 9.0)
        marker.setBrush(QBrush(QColor("#10b981")))
        marker.setPen(QPen(QColor("#ffffff"), 1.5))
        marker.setZValue(10.4)
        items.append(marker)

        # Material parameters summary box
        u_resilience = (yield_strength_MPa ** 2) / (2.0 * youngs_modulus_GPa * 1000.0) * 1000.0  # kJ/m³
        param_text = (
            f"Modun Đàn Hồi E: {youngs_modulus_GPa:.1f} GPa\n"
            f"Giới Hạn Chảy σ_y: {yield_strength_MPa:.1f} MPa\n"
            f"Độ Bền Kéo σ_UTS: {uts_strength_MPa:.1f} MPa\n"
            f"Modun Phục Hồi U_r: {u_resilience:.1f} kJ/m³"
        )
        p_item = QGraphicsTextItem(param_text)
        p_item.setFont(QFont("Segoe UI", 7))
        p_item.setDefaultTextColor(QColor("#cbd5e1"))
        p_item.setPos(ax_x0 + 130.0, y0 + 180.0)
        p_item.setZValue(10.3)
        items.append(p_item)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Định luật Hooke biểu diễn biến dạng đàn hồi tuyến tính σ = E · ε. Vượt quá giới hạn chảy σ_y, vật liệu hóa bền biến dạng "
            "cho tới ứng suất bền cực đại σ_UTS, sau đó xảy ra hiện tượng thắt cổ (necking) tập trung ứng suất và dẫn đến nứt gãy cơ học."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items


    @staticmethod
    def create_kepler_orbits_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HALLEY_COMET_EXTREME_ECCENTRICITY",
        title_vi: str = "Sao Chổi Halley (Độ Lệch Tâm Cực Lớn e = 0.967)",
        body_name: str = "Sao Chổi Halley (1P/Halley)",
        central_body: str = "Mặt Trời (Sun)",
        semi_major_axis_km: float = 2.668e9,
        eccentricity: float = 0.967,
        semi_minor_axis_km: float = 6.798e8,
        periapsis_km: float = 8.766e7,
        apoapsis_km: float = 5.248e9,
        orbital_period_s: float = 2.376e9,
        periapsis_velocity_km_s: float = 54.55,
        apoapsis_velocity_km_s: float = 0.91,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Gravitational Orbits,
        Kepler's 3 Laws, Elliptical Trajectories, Equal-Area Sectors, and Vis-Viva Speeds.
        """
        import math
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f1e")))
        bg.setPen(QPen(QColor("#28375a"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header
        hdr = QGraphicsTextItem("🪐 QUỸ ĐẠO HẤP DẪN & 3 ĐỊNH LUẬT KEPLER (ORBITAL MECHANICS)")
        hdr.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor("#38bdf8"))
        hdr.setPos(x0 + 16.0, y0 + 12.0)
        hdr.setZValue(10.1)
        items.append(hdr)

        sub = QGraphicsTextItem(f"Hệ thiên thể: {title_vi} | Bán trục lớn a = {semi_major_axis_km:,.0f} km | e = {eccentricity:.4f}")
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 18.0, y0 + 34.0)
        sub.setZValue(10.1)
        items.append(sub)

        # 3. Left Panel: Elliptical Orbit & Swept Area Visualization
        plot_rect = QGraphicsRectItem(x0 + 16.0, y0 + 58.0, 350.0, 260.0)
        plot_rect.setBrush(QBrush(QColor("#0f172a")))
        plot_rect.setPen(QPen(QColor("#334155"), 1.0))
        plot_rect.setZValue(10.1)
        items.append(plot_rect)

        cx = x0 + 16.0 + 175.0
        cy = y0 + 58.0 + 130.0

        a_px = 120.0
        b_px = a_px * math.sqrt(max(1e-4, 1.0 - eccentricity ** 2))
        c_px = a_px * eccentricity

        f1_x = cx + c_px * 0.4
        f1_y = cy
        ell_x = f1_x - c_px
        ell_y = f1_y

        # Axes
        ax_maj = QGraphicsLineItem(ell_x - a_px, ell_y, ell_x + a_px, ell_y)
        ax_maj.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
        ax_maj.setZValue(10.2)
        items.append(ax_maj)

        # Kepler II Equal-Area Swept Sectors
        delta_t = 0.35
        # Periapsis sector
        p1 = QPainterPath()
        p1.moveTo(f1_x, f1_y)
        for s in range(21):
            ang = -delta_t * 0.5 + (s / 20.0) * delta_t
            r_ang = a_px * (1.0 - eccentricity ** 2) / (1.0 + eccentricity * math.cos(ang))
            p1.lineTo(f1_x + r_ang * math.cos(ang), f1_y - r_ang * math.sin(ang))
        p1.closeSubpath()
        sec1 = QGraphicsPathItem(p1)
        sec1.setBrush(QBrush(QColor(56, 189, 248, 50)))
        sec1.setPen(QPen(QColor(56, 189, 248, 120), 1.0))
        sec1.setZValue(10.2)
        items.append(sec1)

        # Apoapsis sector
        p2 = QPainterPath()
        p2.moveTo(f1_x, f1_y)
        for s in range(21):
            ang = math.pi - delta_t * 0.5 + (s / 20.0) * delta_t
            r_ang = a_px * (1.0 - eccentricity ** 2) / (1.0 + eccentricity * math.cos(ang))
            p2.lineTo(f1_x + r_ang * math.cos(ang), f1_y - r_ang * math.sin(ang))
        p2.closeSubpath()
        sec2 = QGraphicsPathItem(p2)
        sec2.setBrush(QBrush(QColor(244, 63, 94, 50)))
        sec2.setPen(QPen(QColor(244, 63, 94, 120), 1.0))
        sec2.setZValue(10.2)
        items.append(sec2)

        # Orbit Path
        orb_path = QPainterPath()
        for i in range(101):
            th = (i / 100.0) * 2.0 * math.pi
            r_th = a_px * (1.0 - eccentricity ** 2) / (1.0 + eccentricity * math.cos(th))
            ox = f1_x + r_th * math.cos(th)
            oy = f1_y - r_th * math.sin(th)
            if i == 0:
                orb_path.moveTo(ox, oy)
            else:
                orb_path.lineTo(ox, oy)
        orb_item = QGraphicsPathItem(orb_path)
        orb_item.setPen(QPen(QColor("#38bdf8"), 1.8))
        orb_item.setZValue(10.3)
        items.append(orb_item)

        # Central Star at Focus F1
        sun = QGraphicsEllipseItem(f1_x - 10.0, f1_y - 10.0, 20.0, 20.0)
        sun.setBrush(QBrush(QColor("#fbbf24")))
        sun.setPen(QPen(QColor("#ffffff"), 1.2))
        sun.setZValue(10.4)
        items.append(sun)

        lbl_sun = QGraphicsTextItem("☀️ " + central_body.split()[0])
        lbl_sun.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_sun.setDefaultTextColor(QColor("#fbbf24"))
        lbl_sun.setPos(f1_x - 18.0, f1_y + 12.0)
        lbl_sun.setZValue(10.4)
        items.append(lbl_sun)

        # Labels for Periapsis & Apoapsis
        peri_pt_x = f1_x + (a_px * (1.0 - eccentricity))
        apo_pt_x = f1_x - (a_px * (1.0 + eccentricity))

        lbl_p = QGraphicsTextItem(f"Cận điểm P (v_p = {periapsis_velocity_km_s:.1f} km/s)")
        lbl_p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_p.setDefaultTextColor(QColor("#38bdf8"))
        lbl_p.setPos(peri_pt_x - 30.0, f1_y - 20.0)
        lbl_p.setZValue(10.4)
        items.append(lbl_p)

        lbl_a = QGraphicsTextItem(f"Viễn điểm A (v_a = {apoapsis_velocity_km_s:.1f} km/s)")
        lbl_a.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        lbl_a.setDefaultTextColor(QColor("#f43f5e"))
        lbl_a.setPos(apo_pt_x - 10.0, f1_y - 20.0)
        lbl_a.setZValue(10.4)
        items.append(lbl_a)

        # 4. Right Panel: Astrodynamic Parameters Card
        right_rect = QGraphicsRectItem(x0 + 380.0, y0 + 58.0, 284.0, 260.0)
        right_rect.setBrush(QBrush(QColor("#0f172a")))
        right_rect.setPen(QPen(QColor("#334155"), 1.0))
        right_rect.setZValue(10.1)
        items.append(right_rect)

        thdr_r = QGraphicsTextItem("📊 THÔNG SỐ CƠ HỌC THIÊN THỂ")
        thdr_r.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr_r.setDefaultTextColor(QColor("#38bdf8"))
        thdr_r.setPos(x0 + 388.0, y0 + 64.0)
        thdr_r.setZValue(10.2)
        items.append(thdr_r)

        t_yrs = orbital_period_s / (365.25 * 86400.0)
        p_text = (
            f"Vệ tinh / Hành tinh: {body_name}\n"
            f"Bán trục lớn (a): {semi_major_axis_km:,.0f} km\n"
            f"Độ lệch tâm (e): {eccentricity:.4f}\n"
            f"Bán trục nhỏ (b): {semi_minor_axis_km:,.0f} km\n"
            f"Cận điểm (r_p): {periapsis_km:,.0f} km\n"
            f"Viễn điểm (r_a): {apoapsis_km:,.0f} km\n"
            f"Chu kỳ Kepler (T): {t_yrs:.2f} năm ({orbital_period_s:,.0f} s)\n"
            f"Vận tốc cực đại v_p: {periapsis_velocity_km_s:.2f} km/s\n"
            f"Vận tốc cực tiểu v_a: {apoapsis_velocity_km_s:.2f} km/s\n"
            f"Tỉ số vận tốc v_p / v_a: {(1.0 + eccentricity) / max(1e-4, 1.0 - eccentricity):.2f} lần"
        )
        p_item = QGraphicsTextItem(p_text)
        p_item.setFont(QFont("Segoe UI", 7))
        p_item.setDefaultTextColor(QColor("#cbd5e1"))
        p_item.setPos(x0 + 388.0, y0 + 88.0)
        p_item.setZValue(10.2)
        items.append(p_item)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Định luật Kepler: (I) Các hành tinh chuyển động theo quỹ đạo elip với Mặt Trời nằm ở một tiêu điểm. "
            "(II) Đường nối hành tinh - Mặt Trời quét những diện tích bằng nhau trong các khoảng thời gian bằng nhau (vận tốc lớn nhất tại cận điểm, nhỏ nhất tại viễn điểm). "
            "(III) Tỉ số giữa lập phương bán trục lớn và bình phương chu kỳ a³/T² là hằng số."
        )
        foot.setFont(QFont("Segoe UI", 7))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items


class ChemistryTools:
    """Chemical equations, reversible reactions, bonds, and organic rings."""

    @staticmethod
    def create_reaction_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        reversible: bool = False,
        condition_top: str = "",
        condition_bottom: str = "",
        color: QColor = QColor("#202124"),
    ) -> list[QGraphicsItem]:
        """
        Creates forward (→) or equilibrium reversible (⇄) reaction arrow
        with top/bottom reaction conditions (e.g. t°, catalyst, pressure).
        """
        items: list[QGraphicsItem] = []
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 10.0:
            return items

        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        pen = QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0

        if reversible:
            # Reversible double half-arrows (⇄)
            sep = 4.0
            # Top half arrow (rightward)
            top_y0 = (x0 + px * sep, y0 + py * sep)
            top_y1 = (x1 + px * sep, y1 + py * sep)
            path_top = QPainterPath()
            path_top.moveTo(*top_y0)
            path_top.lineTo(*top_y1)
            # Upper barb
            path_top.lineTo(top_y1[0] - ux * 10.0 + px * 6.0, top_y1[1] - uy * 10.0 + py * 6.0)

            # Bottom half arrow (leftward)
            bot_y0 = (x1 - px * sep, y1 - py * sep)
            bot_y1 = (x0 - px * sep, y0 - py * sep)
            path_bot = QPainterPath()
            path_bot.moveTo(*bot_y0)
            path_bot.lineTo(*bot_y1)
            # Lower barb
            path_bot.lineTo(bot_y1[0] + ux * 10.0 - px * 6.0, bot_y1[1] + uy * 10.0 - py * 6.0)

            item_top = QGraphicsPathItem(path_top)
            item_top.setPen(pen)
            items.append(item_top)

            item_bot = QGraphicsPathItem(path_bot)
            item_bot.setPen(pen)
            items.append(item_bot)
        else:
            # Single direction arrow (→)
            path = QPainterPath()
            path.moveTo(x0, y0)
            path.lineTo(x1, y1)
            # Both barbs
            path.moveTo(x1, y1)
            path.lineTo(x1 - ux * 12.0 + px * 6.0, y1 - uy * 12.0 + py * 6.0)
            path.moveTo(x1, y1)
            path.lineTo(x1 - ux * 12.0 - px * 6.0, y1 - uy * 12.0 - py * 6.0)

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            items.append(item)

        # Condition labels
        if condition_top:
            txt_top = QGraphicsTextItem(condition_top)
            txt_top.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
            txt_top.setDefaultTextColor(color)
            txt_top.setPos(mid_x - 15 + px * 12.0, mid_y - 20 + py * 12.0)
            items.append(txt_top)

        if condition_bottom:
            txt_bot = QGraphicsTextItem(condition_bottom)
            txt_bot.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
            txt_bot.setDefaultTextColor(color)
            txt_bot.setPos(mid_x - 15 - px * 14.0, mid_y + 4 - py * 14.0)
            items.append(txt_bot)

        return items

    @staticmethod
    def create_benzene_ring(
        center: tuple[float, float],
        radius: float = 48.0,
        color: QColor = QColor("#202124"),
    ) -> list[QGraphicsItem]:
        """Creates a chemical hexagon ring (Benzene aromatic core)."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        pen = QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        # Hexagon outer vertices
        hex_poly = QPolygonF()
        for i in range(6):
            angle_rad = math.radians(60 * i - 30)
            px = cx + radius * math.cos(angle_rad)
            py = cy + radius * math.sin(angle_rad)
            hex_poly.append(QPointF(px, py))

        hex_item = QGraphicsPolygonItem(hex_poly)
        hex_item.setPen(pen)
        hex_item.setBrush(Qt.BrushStyle.NoBrush)
        hex_item.setZValue(11.0)
        items.append(hex_item)

        # Inner aromatic delocalized electron circle
        r_inner = radius * 0.58
        inner_circle = QGraphicsEllipseItem(cx - r_inner, cy - r_inner, 2 * r_inner, 2 * r_inner)
        inner_circle.setPen(pen)
        inner_circle.setBrush(Qt.BrushStyle.NoBrush)
        inner_circle.setZValue(11.0)
        items.append(inner_circle)

        return items


    @staticmethod
    def create_gcms_diagram(
        center: tuple[float, float],
        preset_code: str = "PETROCHEMICAL_GASOLINE_BTEX",
        selected_peak_idx: int = 0,
        ramp_rate: float = 10.0,
        flow_rate: float = 1.0,
        width: float = 640.0,
        height: float = 340.0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Gas Chromatography - Mass Spectrometry (GC-MS)."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#06b6d4'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HÓA PHÂN TÍCH: SẮC KÝ KHÍ - KHỐI PHỔ GC-MS – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 300.0
        vh = 195.0

        # 1. TIC Chromatogram Box
        tic_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        tic_box.setBrush(QBrush(QColor('#020617')))
        tic_box.setPen(QPen(QColor('#1e293b'), 1.0))
        tic_box.setZValue(10.2)
        items.append(tic_box)

        tic_hdr = QGraphicsTextItem('SẮC KÝ ĐỒ TỔNG DÒNG ION (TIC CHROMATOGRAM)')
        tic_hdr.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        tic_hdr.setDefaultTextColor(QColor('#38bdf8'))
        tic_hdr.setPos(vx0 + 10.0, vy0 + 8.0)
        tic_hdr.setZValue(10.4)
        items.append(tic_hdr)

        # Baseline
        base_y = vy0 + vh - 25.0
        base_line = QGraphicsLineItem(vx0 + 25.0, base_y, vx0 + vw - 15.0, base_y)
        base_line.setPen(QPen(QColor('#475569'), 1.0))
        base_line.setZValue(10.3)
        items.append(base_line)

        # Draw representative chromatographic Gaussian peaks
        peaks_x = [vx0 + 60.0, vx0 + 110.0, vx0 + 165.0, vx0 + 215.0, vx0 + 265.0]
        peaks_h = [45.0, 95.0, 70.0, 55.0, 35.0]
        peak_names = ["Benzen", "Toluen", "Etylbenzen", "m/p-Xylen", "o-Xylen"]

        for i, (px, ph) in enumerate(zip(peaks_x, peaks_h)):
            peak_path = QPainterPath()
            peak_path.moveTo(px - 18.0, base_y)
            peak_path.quadTo(px, base_y - ph * 1.2, px + 18.0, base_y)

            p_item = QGraphicsPathItem(peak_path)
            is_sel = (i == selected_peak_idx)
            p_item.setPen(QPen(QColor('#f59e0b') if is_sel else QColor('#38bdf8'), 1.8 if is_sel else 1.2))
            p_item.setBrush(QBrush(QColor(245, 158, 11, 40) if is_sel else QColor(56, 189, 248, 25)))
            p_item.setZValue(10.4)
            items.append(p_item)

            lbl = QGraphicsTextItem(f"t_R{i+1}")
            lbl.setFont(QFont('Segoe UI', 6))
            lbl.setDefaultTextColor(QColor('#f59e0b') if is_sel else QColor('#94a3b8'))
            lbl.setPos(px - 10.0, base_y + 2.0)
            lbl.setZValue(10.5)
            items.append(lbl)

        # 2. Right Box: Mass Spectrum & Readout
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        ms_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        ms_box.setBrush(QBrush(QColor('#020617')))
        ms_box.setPen(QPen(QColor('#1e293b'), 1.0))
        ms_box.setZValue(10.2)
        items.append(ms_box)

        ms_hdr = QGraphicsTextItem('KHỐI PHỔ ION HÓA ĐIỆN TỬ (EI-MS 70 eV)')
        ms_hdr.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        ms_hdr.setDefaultTextColor(QColor('#10b981'))
        ms_hdr.setPos(rx0 + 10.0, vy0 + 8.0)
        ms_hdr.setZValue(10.4)
        items.append(ms_hdr)

        # Sticks
        ms_base_y = vy0 + 95.0
        ms_baseline = QGraphicsLineItem(rx0 + 15.0, ms_base_y, rx0 + rw - 15.0, ms_base_y)
        ms_baseline.setPen(QPen(QColor('#475569'), 1.0))
        ms_baseline.setZValue(10.3)
        items.append(ms_baseline)

        # Draw standard stick peaks (e.g. Toluen: m/z 91 base peak tropylium ion, 92 M+, 65, 39)
        sticks = [(rx0 + 40.0, 25.0, "39"), (rx0 + 80.0, 40.0, "65"), (rx0 + 140.0, 75.0, "91 (100%)"), (rx0 + 160.0, 50.0, "92 (M⁺)")]
        for sx, sh, slbl in sticks:
            stk = QGraphicsLineItem(sx, ms_base_y, sx, ms_base_y - sh)
            stk.setPen(QPen(QColor('#10b981'), 2.2))
            stk.setZValue(10.4)
            items.append(stk)

            stk_lbl = QGraphicsTextItem(slbl)
            stk_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
            stk_lbl.setDefaultTextColor(QColor('#10b981'))
            stk_lbl.setPos(sx - 12.0, ms_base_y - sh - 14.0)
            stk_lbl.setZValue(10.5)
            items.append(stk_lbl)

        # Readout text
        sel_name = peak_names[selected_peak_idx] if selected_peak_idx < len(peak_names) else "Hợp chất"
        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ PHÂN TÍCH GC-MS:\n'
            f'• Hợp chất đang chọn: {sel_name} (Peak #{selected_peak_idx+1})\n'
            f'• Tốc độ gia nhiệt: {ramp_rate:.1f} °C/phút | Lưu lượng khí He: {flow_rate:.2f} mL/phút\n'
            f'• Ion phân tử M⁺•: m/z 92 | Ion gốc cơ sở (Base peak): m/z 91 (Tropylium C₇H₇⁺)\n'
            f'• Độ phân giải sắc ký: R_s ≥ 1.5 (Tách rời hoàn toàn về đường nền)'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, ms_base_y + 12.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Kỹ thuật GC-MS: Sắc ký khí tách các cấu tử bay hơi theo độ phân cực và điểm sôi trên cột mao quản DB-5MS. '
            'Khối phổ ion hóa điện tử 70 eV bắn phá phân tử thành ion phân tử M⁺• và các mảnh ion đặc trưng, '
            'cho phép nhận danh chính xác cấu trúc hóa học với độ tin cậy tuyệt đối.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


# ─── Vector Tools for Biology ─────────────────────────────────────────────────


    @staticmethod
    def create_reaction_kinetics_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'EXOTHERMIC_CATALYSIS_DECOMPOSITION',
        reaction_str: str = '2 H₂O₂ ➔ 2 H₂O + O₂',
        ea_uncat_kj: float = 75.0,
        ea_cat_kj: float = 28.0,
        delta_h_kj: float = -98.0,
        temp_k: float = 298.15,
        has_catalyst: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Chemical Reaction Kinetics and Energy Coordinate."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#f59e0b'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'ĐỘNG HÓA HỌC & ĐỊNH LUẬT ARRHENIUS: {reaction_str} – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        px0 = x0 + 30.0
        py0 = y0 + 60.0
        pw = 230.0
        ph = 140.0

        # Energy profile box
        coord_box = QGraphicsRectItem(px0, py0, pw, ph)
        coord_box.setBrush(QBrush(QColor('#020617')))
        coord_box.setPen(QPen(QColor('#1e293b'), 1.0))
        coord_box.setZValue(10.2)
        items.append(coord_box)

        # Baseline
        base_line = QGraphicsLineItem(px0 + 10.0, py0 + ph - 20.0, px0 + pw - 10.0, py0 + ph - 20.0)
        base_line.setPen(QPen(QColor('#475569'), 1.0))
        base_line.setZValue(10.3)
        items.append(base_line)

        # Uncatalyzed curve
        uncat_path = QPainterPath()
        uncat_path.moveTo(px0 + 20.0, py0 + ph - 50.0)
        uncat_path.quadTo(px0 + pw * 0.5, py0 + 20.0, px0 + pw - 20.0, py0 + ph - 30.0)
        uncat_item = QGraphicsPathItem(uncat_path)
        uncat_item.setPen(QPen(QColor('#ef4444'), 2.0))
        uncat_item.setZValue(10.4)
        items.append(uncat_item)

        # Catalyzed curve
        cat_path = QPainterPath()
        cat_path.moveTo(px0 + 20.0, py0 + ph - 50.0)
        cat_path.quadTo(px0 + pw * 0.5, py0 + 65.0, px0 + pw - 20.0, py0 + ph - 30.0)
        cat_item = QGraphicsPathItem(cat_path)
        cat_item.setPen(QPen(QColor('#22c55e'), 2.0, Qt.PenStyle.DashLine))
        cat_item.setZValue(10.4)
        items.append(cat_item)

        info = QGraphicsTextItem(
            f'THÔNG SỐ ĐỘNG HỌC:\n'
            f'• Phản ứng: {reaction_str}\n'
            f'• E_a không xúc tác: {ea_uncat_kj:.1f} kJ/mol | E_a có xúc tác: {ea_cat_kj:.1f} kJ/mol\n'
            f'• Biến thiên Entanpi ΔH: {delta_h_kj:.1f} kJ/mol ({"Tỏa nhiệt" if delta_h_kj < 0 else "Thu nhiệt"})\n'
            f'• Nhiệt độ: {temp_k:.1f} K'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 290.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_phase_diagram_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "WATER_H2O_ANOMALOUS_MELTING",
        substance_name: str = "Nước",
        formula: str = "H₂O",
        temp_c: float = 25.0,
        press_atm: float = 1.0,
        current_phase: str = "LỎNG (LIQUID)",
        t_triple: float = 0.01,
        p_triple: float = 0.006,
        t_crit: float = 373.95,
        p_crit: float = 217.75,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of thermodynamic P-T Phase Diagram and Clausius-Clapeyron curves."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'HÓA LÝ & NHIỆT ĐỘNG HỌC: GIẢN ĐỒ PHA P-T CỦA {substance_name.upper()} ({formula}) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 240.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Coordinate axes & 1 atm reference line
        y_1atm = vy0 + vh * 0.48
        line_1atm = QGraphicsLineItem(vx0 + 40.0, y_1atm, vx0 + vw - 10.0, y_1atm)
        line_1atm.setPen(QPen(QColor('#f59e0b'), 1.2, Qt.PenStyle.DotLine))
        line_1atm.setZValue(10.3)
        items.append(line_1atm)

        lbl_1atm = QGraphicsTextItem('1 atm')
        lbl_1atm.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        lbl_1atm.setDefaultTextColor(QColor('#f59e0b'))
        lbl_1atm.setPos(vx0 + 8.0, y_1atm - 10.0)
        lbl_1atm.setZValue(10.4)
        items.append(lbl_1atm)

        # Triple Point
        tx = vx0 + 110.0
        ty = vy0 + vh * 0.72
        trip_dot = QGraphicsEllipseItem(tx - 4.0, ty - 4.0, 8.0, 8.0)
        trip_dot.setBrush(QBrush(QColor('#f59e0b')))
        trip_dot.setPen(QPen(QColor('#ffffff'), 1.2))
        trip_dot.setZValue(10.5)
        items.append(trip_dot)

        trip_lbl = QGraphicsTextItem(f'T ({t_triple:g}°C, {p_triple:g} atm)')
        trip_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        trip_lbl.setDefaultTextColor(QColor('#f59e0b'))
        trip_lbl.setPos(tx + 4.0, ty + 2.0)
        trip_lbl.setZValue(10.6)
        items.append(trip_lbl)

        # Critical Point
        cx_pt = vx0 + vw - 40.0
        cy_pt = vy0 + 35.0
        crit_dot = QGraphicsEllipseItem(cx_pt - 5.0, cy_pt - 5.0, 10.0, 10.0)
        crit_dot.setBrush(QBrush(QColor('#ef4444')))
        crit_dot.setPen(QPen(QColor('#ffffff'), 1.5))
        crit_dot.setZValue(10.5)
        items.append(crit_dot)

        crit_lbl = QGraphicsTextItem(f'C ({t_crit:g}°C, {p_crit:g} atm)')
        crit_lbl.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        crit_lbl.setDefaultTextColor(QColor('#ef4444'))
        crit_lbl.setPos(cx_pt - 65.0, cy_pt - 14.0)
        crit_lbl.setZValue(10.6)
        items.append(crit_lbl)

        # Sublimation Curve (Solid - Gas)
        sub_path = QPainterPath()
        sub_path.moveTo(vx0 + 40.0, vy0 + vh - 20.0)
        sub_path.quadTo(vx0 + 70.0, vy0 + vh * 0.85, tx, ty)
        sub_item = QGraphicsPathItem(sub_path)
        sub_item.setPen(QPen(QColor('#f59e0b'), 2.0))
        sub_item.setZValue(10.4)
        items.append(sub_item)

        # Vaporization Curve (Liquid - Gas) Clausius-Clapeyron
        vap_path = QPainterPath()
        vap_path.moveTo(tx, ty)
        vap_path.quadTo((tx + cx_pt) / 2.0 + 15.0, (ty + cy_pt) / 2.0 + 25.0, cx_pt, cy_pt)
        vap_item = QGraphicsPathItem(vap_path)
        vap_item.setPen(QPen(QColor('#38bdf8'), 2.2))
        vap_item.setZValue(10.4)
        items.append(vap_item)

        # Melting Curve (Solid - Liquid)
        melt_path = QPainterPath()
        melt_path.moveTo(tx, ty)
        if 'WATER' in preset_code or 'H2O' in formula:
            # Anomalous negative melting slope (tilted left)
            melt_path.lineTo(tx - 25.0, vy0 + 25.0)
        else:
            # Positive melting slope (tilted right)
            melt_path.lineTo(tx + 30.0, vy0 + 25.0)
        melt_item = QGraphicsPathItem(melt_path)
        melt_item.setPen(QPen(QColor('#10b981'), 2.2))
        melt_item.setZValue(10.4)
        items.append(melt_item)

        # Labels for phases
        s_lbl = QGraphicsTextItem('RẮN')
        s_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor('#93c5fd'))
        s_lbl.setPos(vx0 + 55.0, vy0 + 60.0)
        s_lbl.setZValue(10.5)
        items.append(s_lbl)

        l_lbl = QGraphicsTextItem('LỎNG')
        l_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        l_lbl.setDefaultTextColor(QColor('#2dd4bf'))
        l_lbl.setPos(tx + 40.0, vy0 + 60.0)
        l_lbl.setZValue(10.5)
        items.append(l_lbl)

        g_lbl = QGraphicsTextItem('KHÍ / HƠI')
        g_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        g_lbl.setDefaultTextColor(QColor('#fdba74'))
        g_lbl.setPos(vx0 + vw * 0.65, vy0 + vh * 0.70)
        g_lbl.setZValue(10.5)
        items.append(g_lbl)

        # Right Panel: Metrics & Theory
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ CÂN BẰNG PHA ({formula}):\n'
            f'• Nhiệt độ T: {temp_c:.2f} °C ({temp_c + 273.15:.2f} K)\n'
            f'• Áp suất P: {press_atm:g} atm ({press_atm * 101.325:.1f} kPa)\n'
            f'• Trạng thái pha hiện tại: {current_phase}\n\n'
            f'ĐẶC TRƯNG NHIỆT ĐỘNG HỌC:\n'
            f'• Điểm tam giác T: ({t_triple:g}°C, {p_triple:g} atm)\n'
            f'• Điểm tới hạn C: ({t_crit:g}°C, {p_crit:g} atm)\n'
            f'• Phương trình Clausius-Clapeyron:\n'
            f'   dP/dT = ΔH / (T·ΔV)\n'
            f'• Nước: dP/dT < 0 (Băng nhẹ hơn nước lỏng)\n'
            f'• CO₂: P_triple > 1 atm (Băng khô thăng hoa)'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Giản đồ pha (Phase Diagram) biểu diễn các vùng cân bằng nhiệt động Solid-Liquid-Gas. '
            'Đường hóa hơi kết thúc tại Điểm tới hạn (Critical Point), mở ra trạng thái Chất lưu siêu tới hạn. '
            'Độ dốc nóng chảy âm của nước là hiện tượng bất thường duy nhất cho phép trượt băng dưới áp suất lớn.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items

    @staticmethod
    def create_hplc_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "REVERSE_PHASE_ANALGESICS",
        col_length_mm: float = 150.0,
        dp_um: float = 5.0,
        flow_rate: float = 1.0,
        plate_height_um: float = 15.0,
        plates_n: float = 10000.0,
        backpressure_bar: float = 120.0,
        resolution_rs: float = 2.2,
        void_time_s: float = 90.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of HPLC column elution & UV chromatogram.
        Includes stainless steel column cross-section, separated analyte migration bands,
        Gaussian absorbance peaks with resolution R_s, and Van Deemter efficiency metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"SẮC KÝ LỎNG HIỆU NĂNG CAO HPLC & HIỆU SUẤT CỘT: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Cột C18 L = {col_length_mm:.0f} mm | Hạt d_p = {dp_um:.1f} μm | Lưu lượng = {flow_rate:.2f} mL/min | "
            f"Số đĩa N = {plates_n:.0f} | Áp suất ΔP = {backpressure_bar:.0f} bar | R_s = {resolution_rs:.2f}"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Column & Analyte Bands
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("MÔ HÌNH CỘT SẮC KÝ & DẢI CHẤT PHÂN TÁCH:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # Column Tube
        col_w = cw - 60.0
        col_h = 32.0
        col_x = cx + 30.0
        col_y = cy + 40.0

        tube = QGraphicsRectItem(col_x, col_y, col_w, col_h)
        tube.setBrush(QBrush(QColor(30, 41, 59)))
        tube.setPen(QPen(QColor(148, 163, 184), 1.4))
        tube.setZValue(10.2)
        items.append(tube)

        # Analyte bands inside column
        colors = ["#38bdf8", "#eab308", "#f43f5e"]
        for i, pos_frac in enumerate([0.25, 0.55, 0.85]):
            band_x = col_x + pos_frac * (col_w - 20.0)
            band = QGraphicsRectItem(band_x, col_y + 2.0, 16.0, col_h - 4.0)
            band.setBrush(QBrush(QColor(colors[i % len(colors)])))
            band.setPen(Qt.PenStyle.NoPen)
            band.setZValue(10.3)
            items.append(band)

        # Tubing lines
        in_pipe = QGraphicsLineItem(cx + 8.0, col_y + col_h / 2.0, col_x, col_y + col_h / 2.0)
        in_pipe.setPen(QPen(QColor("#38bdf8"), 2.0))
        in_pipe.setZValue(10.3)
        items.append(in_pipe)

        out_pipe = QGraphicsLineItem(col_x + col_w, col_y + col_h / 2.0, cx + cw - 8.0, col_y + col_h / 2.0)
        out_pipe.setPen(QPen(QColor("#38bdf8"), 2.0))
        out_pipe.setZValue(10.3)
        items.append(out_pipe)

        # Column dimension label
        col_lbl = QGraphicsTextItem(f"Cột Pha Đảo C18 ({col_length_mm:.0f} mm × 4.6 mm, {dp_um:.1f} μm)")
        col_lbl.setFont(QFont("Segoe UI", 7))
        col_lbl.setDefaultTextColor(QColor("#94a3b8"))
        col_lbl.setPos(cx + 25.0, col_y + col_h + 10.0)
        col_lbl.setZValue(10.2)
        items.append(col_lbl)

        # 4. Right Panel: UV Absorbance Chromatogram
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("SẮC KÝ ĐỒ UV ABSORBANCE & ĐỘ PHÂN GIẢI R_s:")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Axes
        px = rx + 30.0
        py = ry + 25.0
        pw = rw - 45.0
        ph = rh - 85.0

        axis_x = QGraphicsLineItem(px, py + ph, px + pw, py + ph)
        axis_x.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_x.setZValue(10.2)
        items.append(axis_x)

        axis_y = QGraphicsLineItem(px, py, px, py + ph)
        axis_y.setPen(QPen(QColor(71, 85, 105), 1.0))
        axis_y.setZValue(10.2)
        items.append(axis_y)

        # Plot 3 Gaussian peaks on chromatogram
        peak_centers = [0.25, 0.55, 0.85]
        peak_heights = [0.85, 0.70, 0.95]
        chrom_path = QPainterPath()
        steps = int(pw)
        for s in range(steps + 1):
            x_frac = s / float(steps)
            tot_abs = 0.0
            for pc, ph_val in zip(peak_centers, peak_heights):
                sigma = 0.035
                tot_abs += ph_val * math.exp(-0.5 * ((x_frac - pc) / sigma) ** 2)
            tot_abs = min(1.0, tot_abs)
            gx = px + s
            gy = py + ph - tot_abs * ph
            if s == 0:
                chrom_path.moveTo(gx, gy)
            else:
                chrom_path.lineTo(gx, gy)

        chrom_item = QGraphicsPathItem(chrom_path)
        chrom_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        chrom_item.setZValue(10.4)
        items.append(chrom_item)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Chiều cao đĩa H = A + B/u + Cu: {plate_height_um:.2f} μm    • Số đĩa N = L/H: {plates_n:.0f}    • Áp suất ΔP: {backpressure_bar:.0f} bar\n"
            f"• Thời gian chết t_0: {void_time_s:.1f} s    • Độ phân giải đỉnh R_s: {resolution_rs:.2f} ({'Tách nền hoàn toàn' if resolution_rs >= 1.5 else 'Chưa đạt tách nền'})"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, py + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình Van Deemter H = A + B/u + C·u định lượng ba nguồn gốc gây giãn rộng pic: khuếch tán xoáy (A), khuếch tán dọc (B) và kháng truyền khối (C). "
            "Độ phân giải sắc ký R_s = 2(t_R2 - t_R1)/(W1 + W2) yêu cầu R_s ≥ 1.5 để đạt tiêu chuẩn tách nền hoàn toàn 99.7% giữa hai đỉnh chất phân tích."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_buffer_system_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "ACETIC_ACID_ACETATE_BUFFER",
        acid_name: str = "CH₃COOH",
        base_name: str = "CH₃COO⁻",
        pKa: float = 4.76,
        c_total_M: float = 0.20,
        ratio: float = 1.0,
        current_pH: float = 4.76,
        van_slyke_capacity: float = 0.115,
        beta_max: float = 0.115,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Acid-Base Buffer Systems & Van Slyke Capacity.
        Includes Van Slyke capacity curve beta(pH), effective buffering range, chemical speciation,
        laboratory beaker with Universal Indicator liquid color, and Henderson-Hasselbalch metrics.
        """
        import math
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"HỆ ĐỆM AXIT - BAZƠ & DUNG LƯỢNG VAN SLYKE: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Cặp đệm: {acid_name} / {base_name} | pKa = {pKa:.2f} | C_total = {c_total_M:.2f} M | "
            f"[A⁻]/[HA] = {ratio:.2f} | pH = {current_pH:.2f} | β = {van_slyke_capacity:.3f} M/pH"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Van Slyke Buffer Capacity Curve
        pw = (width - 40.0) / 2.0
        ph = height - 165.0
        p1_x = x0 + 16.0
        p_y = y0 + 58.0

        b_bg = QGraphicsRectItem(p1_x, p_y, pw, ph)
        b_bg.setBrush(QBrush(QColor(4, 7, 16)))
        b_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        b_bg.setZValue(10.1)
        items.append(b_bg)

        p1_lbl = QGraphicsTextItem("ĐỒ THỊ DUNG LƯỢNG ĐỆM β(pH) & VÙNG ĐỆM")
        p1_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # Plot frame inside Left Panel
        gx = p1_x + 30.0
        gy = p_y + 32.0
        gw = pw - 42.0
        gh = ph - 55.0

        plt_box = QGraphicsRectItem(gx, gy, gw, gh)
        plt_box.setBrush(QBrush(QColor(3, 7, 15)))
        plt_box.setPen(QPen(QColor(30, 41, 59), 1.0))
        plt_box.setZValue(10.2)
        items.append(plt_box)

        # Effective buffer zone [pKa - 1, pKa + 1]
        z_left = max(gx, gx + ((pKa - 1.0 - 1.0) / 12.0) * gw)
        z_right = min(gx + gw, gx + ((pKa + 1.0 - 1.0) / 12.0) * gw)
        if z_right > z_left:
            zone = QGraphicsRectItem(z_left, gy, z_right - z_left, gh)
            zone.setBrush(QBrush(QColor(16, 185, 129, 40)))
            zone.setPen(Qt.PenStyle.NoPen)
            zone.setZValue(10.2)
            items.append(zone)

        # pH grid ticks
        for p in (3, 5, 7, 9, 11):
            px_cur = gx + ((p - 1.0) / 12.0) * gw
            p_line = QGraphicsLineItem(px_cur, gy, px_cur, gy + gh)
            p_line.setPen(QPen(QColor(30, 41, 59, 120), 0.8, Qt.PenStyle.DotLine))
            p_line.setZValue(10.2)
            items.append(p_line)

            p_txt = QGraphicsTextItem(str(p))
            p_txt.setFont(QFont("Segoe UI", 6))
            p_txt.setDefaultTextColor(QColor("#64748b"))
            p_txt.setPos(px_cur - 6.0, gy + gh + 2.0)
            p_txt.setZValue(10.3)
            items.append(p_txt)

        # Van Slyke curve path
        ka = 10.0 ** (-pKa)
        max_b_vis = max(0.20, beta_max * 1.5)
        curve_path = QPainterPath()
        first = True
        for i in range(51):
            eval_p = 1.0 + (i / 50.0) * 12.0
            h = 10.0 ** (-eval_p)
            oh = 1.0e-14 / max(1e-15, h)
            denom = (ka + h) ** 2.0
            term = (c_total_M * ka * h) / denom if denom > 1e-30 else 0.0
            b_val = min(max_b_vis, 2.303 * (h + oh + term))

            px = gx + ((eval_p - 1.0) / 12.0) * gw
            py = gy + gh - (b_val / max_b_vis) * gh
            if first:
                curve_path.moveTo(px, py)
                first = False
            else:
                curve_path.lineTo(px, py)

        curve_item = QGraphicsPathItem(curve_path)
        curve_item.setPen(QPen(QColor("#38bdf8"), 2.0))
        curve_item.setZValue(10.3)
        items.append(curve_item)

        # Operating Point at current_pH
        cur_x = gx + ((current_pH - 1.0) / 12.0) * gw
        cur_y = gy + gh - (min(max_b_vis, van_slyke_capacity) / max_b_vis) * gh

        pt_dot = QGraphicsEllipseItem(cur_x - 3.5, cur_y - 3.5, 7.0, 7.0)
        pt_dot.setBrush(QBrush(QColor("#f59e0b")))
        pt_dot.setPen(Qt.PenStyle.NoPen)
        pt_dot.setZValue(10.4)
        items.append(pt_dot)

        pt_lbl = QGraphicsTextItem(f"pH={current_pH:.2f}")
        pt_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        pt_lbl.setDefaultTextColor(QColor("#f59e0b"))
        pt_lbl.setPos(cur_x - 18.0, cur_y - 14.0)
        pt_lbl.setZValue(10.4)
        items.append(pt_lbl)

        # 4. Right Panel: Beaker, Indicator & Chemical Equilibrium
        p2_x = p1_x + pw + 8.0
        d_bg = QGraphicsRectItem(p2_x, p_y, pw, ph)
        d_bg.setBrush(QBrush(QColor(4, 7, 16)))
        d_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        d_bg.setZValue(10.1)
        items.append(d_bg)

        p2_lbl = QGraphicsTextItem("CÂN BẰNG PHÂN LY & BÌNH DUNG DỊCH ĐỆM")
        p2_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p2_lbl.setPos(p2_x + 8.0, p_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        # Chemical Equation
        eq_txt = QGraphicsTextItem(f"{acid_name} + H₂O  ⇌  {base_name} + H₃O⁺")
        eq_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        eq_txt.setDefaultTextColor(QColor("#e2e8f0"))
        eq_txt.setPos(p2_x + 16.0, p_y + 28.0)
        eq_txt.setZValue(10.2)
        items.append(eq_txt)

        # Laboratory Beaker Outline
        bk_x = p2_x + pw / 2.0 - 55.0
        bk_y = p_y + 55.0
        bk_w = 110.0
        bk_h = 100.0

        bk_l = QGraphicsLineItem(bk_x, bk_y, bk_x, bk_y + bk_h)
        bk_l.setPen(QPen(QColor("#94a3b8"), 1.8))
        bk_l.setZValue(10.2)
        items.append(bk_l)

        bk_b = QGraphicsLineItem(bk_x, bk_y + bk_h, bk_x + bk_w, bk_y + bk_h)
        bk_b.setPen(QPen(QColor("#94a3b8"), 1.8))
        bk_b.setZValue(10.2)
        items.append(bk_b)

        bk_r = QGraphicsLineItem(bk_x + bk_w, bk_y + bk_h, bk_x + bk_w, bk_y)
        bk_r.setPen(QPen(QColor("#94a3b8"), 1.8))
        bk_r.setZValue(10.2)
        items.append(bk_r)

        # Liquid fill inside beaker (color matches universal pH indicator)
        if current_pH <= 3.5:
            liq_col = QColor("#ef4444")
        elif current_pH <= 5.5:
            liq_col = QColor("#f97316")
        elif current_pH <= 7.5:
            liq_col = QColor("#22c55e")
        elif current_pH <= 9.5:
            liq_col = QColor("#06b6d4")
        else:
            liq_col = QColor("#a855f7")

        liq_rect = QGraphicsRectItem(bk_x + 2.0, bk_y + 35.0, bk_w - 4.0, bk_h - 37.0)
        liq_rect.setBrush(QBrush(QColor(liq_col.red(), liq_col.green(), liq_col.blue(), 180)))
        liq_rect.setPen(Qt.PenStyle.NoPen)
        liq_rect.setZValue(10.1)
        items.append(liq_rect)

        # Digital pH Meter Badge
        m_box = QGraphicsRectItem(bk_x + 12.0, bk_y + bk_h + 10.0, bk_w - 24.0, 26.0)
        m_box.setBrush(QBrush(QColor("#0f172a")))
        m_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        m_box.setZValue(10.3)
        items.append(m_box)

        m_txt = QGraphicsTextItem(f"pH = {current_pH:.2f}")
        m_txt.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        m_txt.setDefaultTextColor(liq_col.lighter(130))
        m_txt.setPos(bk_x + 22.0, bk_y + bk_h + 12.0)
        m_txt.setZValue(10.4)
        items.append(m_txt)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Henderson-Hasselbalch: pH = pKa + log([A⁻]/[HA])    • Dung lượng Van Slyke: β = {van_slyke_capacity:.4f} M/pH\n"
            f"• Cực đại: β_max = 0.575·C_total = {beta_max:.4f} M/pH (tại pH = pKa)    • Khoảng đệm: pH ∈ [{pKa-1.0:.2f}, {pKa+1.0:.2f}]"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(p1_x + 8.0, p_y + ph + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình Henderson-Hasselbalch và dung lượng đệm Van Slyke xác định định lượng khả năng ổn định pH của dung dịch đệm khi có sự xâm nhập của axit hoặc bazơ mạnh. "
            "Dung lượng đệm cực đại đạt đỉnh tại pH = pKa khi nồng độ axit và bazơ liên hợp bằng nhau ([HA] = [A⁻])."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_spectrophotometry_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "POTASSIUM_PERMANGANATE_KMnO4",
        analyte_name: str = "KMnO₄ (Potassium Permanganate)",
        conc_mM: float = 0.40,
        path_cm: float = 1.0,
        wavelength_nm: float = 525.0,
        epsilon: float = 2400.0,
        absorbance: float = 0.96,
        transmittance_pct: float = 10.96,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of UV-Vis Spectrophotometry,
        cuvette light transmission, Beer-Lambert absorbance A = ε·c·l, and calibration curve.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#080c14")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 195.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("🧪 SPECTROSCOPY • BEER-LAMBERT")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("QUANG PHỔ HẤP THỤ & ĐỊNH LUẬT BEER-LAMBERT")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 220.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Dung dịch: {analyte_name} | c = {conc_mM:.3f} mM | l = {path_cm:.1f} cm | λ = {wavelength_nm:.0f} nm | ε = {epsilon:,.0f} L/(mol·cm)"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Optical Bench Transmission Path
        p1_x = x0 + 18.0
        p1_y = y0 + 64.0
        p1_w = 375.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#050811")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Đường Truyền Quang Học & Cu-vét Mẫu")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        cy = p1_y + p1_h / 2.0 + 8.0

        # Lamp
        lamp_x = p1_x + 18.0
        lamp_y = cy - 20.0
        lamp = QGraphicsRectItem(lamp_x, lamp_y, 28.0, 40.0)
        lamp.setBrush(QBrush(QColor("#eab308")))
        lamp.setPen(QPen(QColor("#ffffff"), 1.0))
        lamp.setZValue(10.2)
        items.append(lamp)

        # Prism
        mono_x = lamp_x + 50.0
        mono_y = cy - 25.0
        tri = QPainterPath()
        tri.moveTo(mono_x, mono_y + 50.0)
        tri.lineTo(mono_x + 16.0, mono_y)
        tri.lineTo(mono_x + 32.0, mono_y + 50.0)
        tri.closeSubpath()
        prism = QGraphicsPathItem(tri)
        prism.setBrush(QBrush(QColor("#1e293b")))
        prism.setPen(QPen(QColor("#38bdf8"), 1.2))
        prism.setZValue(10.2)
        items.append(prism)

        # Beam from Lamp to Prism
        b1 = QGraphicsLineItem(lamp_x + 28.0, cy, mono_x, cy)
        b1.setPen(QPen(QColor("#fef08a"), 2.5))
        b1.setZValue(10.15)
        items.append(b1)

        # Cuvette
        cuv_w = max(24.0, path_cm * 40.0)
        cuv_h = 85.0
        cuv_x = mono_x + 70.0
        cuv_y = cy - cuv_h / 2.0

        # Beam I0 from Prism to Cuvette
        b_i0 = QGraphicsLineItem(mono_x + 32.0, cy, cuv_x, cy)
        b_i0.setPen(QPen(QColor("#22c55e"), 3.0))
        b_i0.setZValue(10.15)
        items.append(b_i0)

        i0_tag = QGraphicsTextItem("I₀ (100%)")
        i0_tag.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        i0_tag.setDefaultTextColor(QColor("#22c55e"))
        i0_tag.setPos(mono_x + 35.0, cy - 14.0)
        i0_tag.setZValue(10.25)
        items.append(i0_tag)

        # Cuvette liquid
        cuv_liq = QGraphicsRectItem(cuv_x + 2.0, cuv_y + 3.0, cuv_w - 4.0, cuv_h - 6.0)
        cuv_liq.setBrush(QBrush(QColor(168, 85, 247, 180)))
        cuv_liq.setPen(Qt.PenStyle.NoPen)
        cuv_liq.setZValue(10.18)
        items.append(cuv_liq)

        # Cuvette frame
        cuv_box = QGraphicsRectItem(cuv_x, cuv_y, cuv_w, cuv_h)
        cuv_box.setBrush(Qt.BrushStyle.NoBrush)
        cuv_box.setPen(QPen(QColor("#e2e8f0"), 2.0))
        cuv_box.setZValue(10.2)
        items.append(cuv_box)

        c_tag = QGraphicsTextItem(f"l={path_cm:.1f}cm")
        c_tag.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        c_tag.setDefaultTextColor(QColor("#ffffff"))
        c_tag.setPos(cuv_x - 2.0, cuv_y - 12.0)
        c_tag.setZValue(10.25)
        items.append(c_tag)

        # Transmitted beam I to detector
        det_x = p1_x + p1_w - 40.0
        det_y = cy - 20.0
        b_it = QGraphicsLineItem(cuv_x + cuv_w, cy, det_x, cy)
        b_it.setPen(QPen(QColor(34, 197, 94, int(max(30, 2.55 * transmittance_pct))), 2.0))
        b_it.setZValue(10.15)
        items.append(b_it)

        it_tag = QGraphicsTextItem(f"I ({transmittance_pct:.1f}%)")
        it_tag.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        it_tag.setDefaultTextColor(QColor("#22c55e"))
        it_tag.setPos(cuv_x + cuv_w + 8.0, cy - 14.0)
        it_tag.setZValue(10.25)
        items.append(it_tag)

        # Detector
        det = QGraphicsRectItem(det_x, det_y, 22.0, 40.0)
        det.setBrush(QBrush(QColor("#1e293b")))
        det.setPen(QPen(QColor("#10b981"), 1.2))
        det.setZValue(10.2)
        items.append(det)

        # 4. Right Panel: Calibration Curve (A vs c)
        p2_x = x0 + 403.0
        p2_y = y0 + 64.0
        p2_w = 259.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#050811")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Đường Chuẩn Tuyến Tính (A = ε·l·c)")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        ox = p2_x + 36.0
        oy = p2_y + p2_h - 30.0
        ax_w = p2_w - 50.0
        ax_h = 125.0

        # Axes
        ax_c = QGraphicsLineItem(ox, oy, ox + ax_w, oy)
        ax_c.setPen(QPen(QColor("#64748b"), 1.2))
        ax_c.setZValue(10.2)
        items.append(ax_c)

        ax_a = QGraphicsLineItem(ox, oy, ox, oy - ax_h - 6.0)
        ax_a.setPen(QPen(QColor("#64748b"), 1.2))
        ax_a.setZValue(10.2)
        items.append(ax_a)

        c_axis_lbl = QGraphicsTextItem("c (mM)")
        c_axis_lbl.setFont(QFont("Segoe UI", 7))
        c_axis_lbl.setDefaultTextColor(QColor("#94a3b8"))
        c_axis_lbl.setPos(ox + ax_w - 32.0, oy + 4.0)
        c_axis_lbl.setZValue(10.2)
        items.append(c_axis_lbl)

        a_axis_lbl = QGraphicsTextItem("A")
        a_axis_lbl.setFont(QFont("Segoe UI", 7))
        a_axis_lbl.setDefaultTextColor(QColor("#94a3b8"))
        a_axis_lbl.setPos(ox - 18.0, oy - ax_h - 8.0)
        a_axis_lbl.setZValue(10.2)
        items.append(a_axis_lbl)

        # Calibration line
        cal_line = QGraphicsLineItem(ox, oy, ox + ax_w * 0.90, oy - ax_h * 0.85)
        cal_line.setPen(QPen(QColor("#38bdf8"), 1.8))
        cal_line.setZValue(10.25)
        items.append(cal_line)

        # Operating Point
        pt_x = ox + ax_w * 0.45
        pt_y = oy - ax_h * 0.45 * (0.85 / 0.90)
        pt_dot = QGraphicsEllipseItem(pt_x - 3.5, pt_y - 3.5, 7.0, 7.0)
        pt_dot.setBrush(QBrush(QColor("#ffffff")))
        pt_dot.setPen(QPen(QColor("#0284c7"), 1.2))
        pt_dot.setZValue(10.3)
        items.append(pt_dot)

        pt_tag = QGraphicsTextItem(f"A={absorbance:.3f}")
        pt_tag.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        pt_tag.setDefaultTextColor(QColor("#38bdf8"))
        pt_tag.setPos(pt_x + 6.0, pt_y - 8.0)
        pt_tag.setZValue(10.35)
        items.append(pt_tag)

        # 5. Bottom Quantitative Metrics Bar
        m_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 36.0, 52.0)
        m_box.setBrush(QBrush(QColor("#0d1526")))
        m_box.setPen(QPen(QColor("#1e293b"), 1.0))
        m_box.setZValue(10.15)
        items.append(m_box)

        m_txt = QGraphicsTextItem(
            f"• Định luật Beer-Lambert: A = -log₁₀(T) = ε·c·l = {absorbance:.4f}  ➔  Độ truyền quang %T = {transmittance_pct:.2f}%\n"
            f"• Nồng độ c = {conc_mM:.3f} mM    • Cu-vét l = {path_cm:.2f} cm    • Hệ số tắt mol ε = {epsilon:,.0f} L/(mol·cm)    • λ = {wavelength_nm:.0f} nm"
        )
        m_txt.setFont(QFont("Consolas", 8))
        m_txt.setDefaultTextColor(QColor("#e2e8f0"))
        m_txt.setPos(p1_x + 10.0, y0 + 274.0)
        m_txt.setZValue(10.25)
        items.append(m_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Định luật Beer-Lambert thiết lập mối quan hệ tuyến tính cơ bản giữa mật độ quang hấp thụ và nồng độ chất tan. "
            "Màu sắc quan sát được của dung dịch là màu bù (complementary color) của dải sóng bị hấp thụ mạnh nhất (λ_max). "
            "Khoảng đo tối ưu trong thực nghiệm nằm trong dải 0.2 ≤ A ≤ 0.8 để tối thiểu hóa sai số quang trắc."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_molecular_orbital_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HYDROGEN_MOLECULE_H2_SIGMA_BOND",
        molecule_name: str = "Hydrogen Molecule (H₂)",
        orbital_type: str = "mo_sigma",
        distance_angstrom: float = 0.74,
        overlap_integral_S: float = 0.753,
        bond_order: float = 1.0,
        is_antibonding: bool = False,
        hybridization_str: str = "1s + 1s (Không lai hóa)",
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Molecular Orbital Theory (LCAO-MO),
        Atomic Hybridization, Quantum Overlap Integral S(R), and Energy Splitting Levels.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0a0f18")))
        bg.setPen(QPen(QColor("#1e293b"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title & Subtitle
        title = QGraphicsTextItem("⚛ CƠ HỌC LƯỢNG TỬ: OBITAN NGUYÊN TỬ & PHÂN TỬ (MO-LCAO)")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#f8fafc"))
        title.setPos(x0 + 16.0, y0 + 12.0)
        title.setZValue(10.2)
        items.append(title)

        mode_str = "Phản liên kết σ*" if is_antibonding else "Liên kết σ"
        sub_str = f"Phân tử: {molecule_name} | Trạng thái: {mode_str} | Lai hóa: {hybridization_str}"
        sub = QGraphicsTextItem(sub_str)
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 16.0, y0 + 32.0)
        sub.setZValue(10.2)
        items.append(sub)

        # 3. Left Panel: Quantum Iso-Surface Wavefunction Box
        wf_box = QGraphicsRectItem(x0 + 20.0, y0 + 56.0, 310.0, 160.0)
        wf_box.setBrush(QBrush(QColor("#0b1329")))
        wf_box.setPen(QPen(QColor("#38bdf8"), 1.2))
        wf_box.setZValue(10.1)
        items.append(wf_box)

        lbl_wf = QGraphicsTextItem("HÀM SÓNG ELECTRON (WAVEFUNCTION ψ)")
        lbl_wf.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_wf.setDefaultTextColor(QColor("#38bdf8"))
        lbl_wf.setPos(x0 + 28.0, y0 + 64.0)
        lbl_wf.setZValue(10.2)
        items.append(lbl_wf)

        cx = x0 + 175.0
        cy = y0 + 140.0
        r_px = max(25.0, min(100.0, distance_angstrom * 45.0))
        nuc_A = QPointF(cx - r_px * 0.5, cy)
        nuc_B = QPointF(cx + r_px * 0.5, cy)

        # Orbital Iso-surface
        if not is_antibonding:
            # Bonding: ovate single lobe
            lobe_w = r_px + 55.0
            lobe_h = 50.0
            lobe = QGraphicsEllipseItem(cx - lobe_w * 0.5, cy - lobe_h * 0.5, lobe_w, lobe_h)
            lobe.setBrush(QBrush(QColor(0, 210, 255, 140)))
            lobe.setPen(QPen(QColor("#38bdf8"), 1.5))
            lobe.setZValue(10.15)
            items.append(lobe)

            lbl_ph = QGraphicsTextItem("+")
            lbl_ph.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            lbl_ph.setDefaultTextColor(QColor("#ffffff"))
            lbl_ph.setPos(cx - 6.0, cy - 14.0)
            lbl_ph.setZValue(10.2)
            items.append(lbl_ph)
        else:
            # Antibonding: opposed lobes with nodal line
            lobe_l = QGraphicsEllipseItem(nuc_A.x() - 30.0, cy - 24.0, 40.0, 48.0)
            lobe_l.setBrush(QBrush(QColor(0, 210, 255, 140)))
            lobe_l.setPen(QPen(QColor("#38bdf8"), 1.2))
            lobe_l.setZValue(10.15)
            items.append(lobe_l)

            lobe_r = QGraphicsEllipseItem(nuc_B.x() - 10.0, cy - 24.0, 40.0, 48.0)
            lobe_r.setBrush(QBrush(QColor(239, 68, 68, 140)))
            lobe_r.setPen(QPen(QColor("#ef4444"), 1.2))
            lobe_r.setZValue(10.15)
            items.append(lobe_r)

            nodal = QGraphicsLineItem(cx, cy - 35.0, cx, cy + 35.0)
            nodal.setPen(QPen(QColor("#ffffff"), 1.4, Qt.PenStyle.DashLine))
            nodal.setZValue(10.2)
            items.append(nodal)

        # Nuclei dots
        for nuc, name in [(nuc_A, "A"), (nuc_B, "B")]:
            dot = QGraphicsEllipseItem(nuc.x() - 4.0, nuc.y() - 4.0, 8.0, 8.0)
            dot.setBrush(QBrush(QColor("#ef4444")))
            dot.setPen(QPen(QColor("#ffffff"), 1.2))
            dot.setZValue(10.25)
            items.append(dot)

        # Telemetry Box below orbital
        t_box = QGraphicsRectItem(x0 + 20.0, y0 + 224.0, 310.0, 105.0)
        t_box.setBrush(QBrush(QColor("#0f172a")))
        t_box.setPen(QPen(QColor("#334155"), 1.0))
        t_box.setZValue(10.1)
        items.append(t_box)

        lines_q = [
            ("Khoảng cách liên hạt nhân R:", f"{distance_angstrom:.2f} Å ({distance_angstrom*100:.0f} pm)", QColor("#38bdf8")),
            ("Tích phân xen phủ S(R):", f"{overlap_integral_S:.4f}", QColor("#4ade80")),
            ("Bậc liên kết (Bond Order):", f"{bond_order:.1f}", QColor("#fbbf24")),
            ("Trạng thái obitan:", "Phản liên kết σ*" if is_antibonding else "Liên kết σ", QColor("#f87171") if is_antibonding else QColor("#38bdf8")),
        ]
        cy_t = y0 + 230.0
        for l_k, l_v, col_v in lines_q:
            txt_k = QGraphicsTextItem(l_k)
            txt_k.setFont(QFont("Segoe UI", 7))
            txt_k.setDefaultTextColor(QColor("#94a3b8"))
            txt_k.setPos(x0 + 28.0, cy_t)
            txt_k.setZValue(10.2)
            items.append(txt_k)

            txt_v = QGraphicsTextItem(l_v)
            txt_v.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            txt_v.setDefaultTextColor(col_v)
            txt_v.setPos(x0 + 205.0, cy_t)
            txt_v.setZValue(10.2)
            items.append(txt_v)
            cy_t += 22.0

        # 4. Right Panel: MO Energy Level Diagram
        plot_rect = QGraphicsRectItem(x0 + 345.0, y0 + 56.0, 315.0, 273.0)
        plot_rect.setBrush(QBrush(QColor("#0f172a")))
        plot_rect.setPen(QPen(QColor("#334155"), 1.0))
        plot_rect.setZValue(10.1)
        items.append(plot_rect)

        thdr = QGraphicsTextItem("📊 GIẢN ĐỒ MỨC NĂNG LƯỢNG OBITAN PHÂN TỬ")
        thdr.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr.setDefaultTextColor(QColor("#fbbf24"))
        thdr.setPos(x0 + 355.0, y0 + 64.0)
        thdr.setZValue(10.2)
        items.append(thdr)

        y_mid = y0 + 180.0
        x_ao_A = x0 + 375.0
        x_ao_B = x0 + 600.0
        x_mo = x0 + 485.0
        bar_w = 34.0

        # Atomic Orbitals levels
        ao_a = QGraphicsLineItem(x_ao_A, y_mid, x_ao_A + bar_w, y_mid)
        ao_a.setPen(QPen(QColor("#94a3b8"), 2.0))
        ao_a.setZValue(10.2)
        items.append(ao_a)

        ao_b = QGraphicsLineItem(x_ao_B, y_mid, x_ao_B + bar_w, y_mid)
        ao_b.setPen(QPen(QColor("#94a3b8"), 2.0))
        ao_b.setZValue(10.2)
        items.append(ao_b)

        # Splitting
        split_dy = 35.0
        y_bond = y_mid + split_dy
        y_anti = y_mid - split_dy

        mo_b = QGraphicsLineItem(x_mo, y_bond, x_mo + bar_w, y_bond)
        mo_b.setPen(QPen(QColor("#38bdf8"), 2.5))
        mo_b.setZValue(10.2)
        items.append(mo_b)

        mo_a = QGraphicsLineItem(x_mo, y_anti, x_mo + bar_w, y_anti)
        mo_a.setPen(QPen(QColor("#f43f5e"), 2.5))
        mo_a.setZValue(10.2)
        items.append(mo_a)

        # Dashed correlation lines
        cor_pen = QPen(QColor("#475569"), 1.0, Qt.PenStyle.DashLine)
        for p1, p2 in [
            (QPointF(x_ao_A + bar_w, y_mid), QPointF(x_mo, y_bond)),
            (QPointF(x_ao_A + bar_w, y_mid), QPointF(x_mo, y_anti)),
            (QPointF(x_ao_B, y_mid), QPointF(x_mo + bar_w, y_bond)),
            (QPointF(x_ao_B, y_mid), QPointF(x_mo + bar_w, y_anti)),
        ]:
            c_line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            c_line.setPen(cor_pen)
            c_line.setZValue(10.15)
            items.append(c_line)

        # Labels
        lbl_b = QGraphicsTextItem("σ (Bonding)")
        lbl_b.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_b.setDefaultTextColor(QColor("#38bdf8"))
        lbl_b.setPos(x_mo + bar_w + 6.0, y_bond - 8.0)
        lbl_b.setZValue(10.2)
        items.append(lbl_b)

        lbl_a = QGraphicsTextItem("σ* (Antibonding)")
        lbl_a.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_a.setDefaultTextColor(QColor("#f43f5e"))
        lbl_a.setPos(x_mo + bar_w + 6.0, y_anti - 8.0)
        lbl_a.setZValue(10.2)
        items.append(lbl_a)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Thuyết Obitan Phân tử (MO-LCAO) xem electron chuyển động trên toàn bộ phân tử dưới tác dụng của tất cả các hạt nhân. "
            "Sự giao thoa hàm sóng đồng pha (+/+) hạ thấp năng lượng tạo liên kết σ/π bền vững, trong khi giao thoa ngược pha (+/-) sinh mặt nút tạo obitan phản liên kết σ*/π*."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items


class BiologyTools:
    """Anatomical callout pins, cellular diagrams, and biological annotations."""

    @staticmethod
    def create_callout_pin(
        target_pt: tuple[float, float],
        label_pt: tuple[float, float],
        text: str,
        color: QColor = QColor("#1a73e8"),
    ) -> list[QGraphicsItem]:
        """
        Creates an anatomical pointer leader line with circular pin at target
        and callout badge text for biological diagrams.
        """
        items: list[QGraphicsItem] = []
        tx, ty = target_pt
        lx, ly = label_pt

        # Anchor Pin Dot
        r = 4.0
        pin = QGraphicsEllipseItem(tx - r, ty - r, 2 * r, 2 * r)
        pin.setBrush(QBrush(color))
        pin.setPen(QPen(QColor("#ffffff"), 1.5))
        pin.setZValue(12.5)
        items.append(pin)

        # Leader Line
        elbow_x = lx - (20.0 if lx > tx else -20.0)
        path = QPainterPath()
        path.moveTo(tx, ty)
        path.lineTo(elbow_x, ly)
        path.lineTo(lx, ly)

        line_item = QGraphicsPathItem(path)
        line_item.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine))
        line_item.setZValue(12.0)
        items.append(line_item)

        # Label Text
        txt_item = QGraphicsTextItem(text)
        txt_item.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        txt_item.setDefaultTextColor(color)
        txt_item.setPos(lx + 4, ly - 14)
        txt_item.setZValue(13.0)
        items.append(txt_item)

        return items


# ─── Vector Tools for Informatics ─────────────────────────────────────────────


    @staticmethod
    def create_punnett_genetics_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'MONOHYBRID_PEA_PLANT',
        parent_cross: str = 'Aa × Aa',
        phenotype_ratio: str = '3 Vàng : 1 Xanh',
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Mendelian Genetics & Punnett Square."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#22c55e'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'DI TRUYỀN HỌC MENDEL & KHUNG PUNNETT: {parent_cross} – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        # 2x2 Punnett Grid Cells
        gx0 = x0 + 40.0
        gy0 = y0 + 60.0
        cell_s = 60.0

        for r in range(2):
            for c in range(2):
                cell = QGraphicsRectItem(gx0 + c * cell_s, gy0 + r * cell_s, cell_s, cell_s)
                cell.setBrush(QBrush(QColor('#020617')))
                cell.setPen(QPen(QColor('#22c55e'), 1.2))
                cell.setZValue(10.3)
                items.append(cell)

        info = QGraphicsTextItem(
            f'THÔNG SỐ PHÉP LAI:\n'
            f'• Phép lai P: {parent_cross}\n'
            f'• Tỷ lệ phân ly kiểu hình F₁: {phenotype_ratio}\n'
            f'• Quy luật: Phân ly độc lập & Tương tác alen'
        )
        info.setFont(QFont('Segoe UI', 8))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(x0 + 200.0, y0 + 60.0)
        info.setZValue(10.4)
        items.append(info)

        return items

    @staticmethod
    def create_action_potential_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'STANDARD_ACTION_POTENTIAL',
        v_peak_mv: float = 32.4,
        i_inj_ua: float = 10.0,
        toxin: str = 'none',
        temp_c: float = 6.3,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hodgkin-Huxley Action Potential."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#f43f5e'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'SINH HỌC & ĐIỆN SINH LÝ: ĐIỆN THẾ HOẠT ĐỘNG HODGKIN-HUXLEY – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 310.0
        vh = 195.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Baseline & thresholds
        base_y = vy0 + vh - 55.0
        grid_rest = QGraphicsLineItem(vx0 + 10.0, base_y, vx0 + vw - 10.0, base_y)
        grid_rest.setPen(QPen(QColor('#38bdf8'), 1.0, Qt.PenStyle.DashLine))
        grid_rest.setZValue(10.3)
        items.append(grid_rest)

        # Action potential spike path
        spike_path = QPainterPath()
        spike_path.moveTo(vx0 + 20.0, base_y)
        spike_path.lineTo(vx0 + 60.0, base_y)
        # Depolarization up to peak (+35 mV)
        spike_path.lineTo(vx0 + 100.0, vy0 + 35.0)
        # Repolarization
        spike_path.lineTo(vx0 + 140.0, base_y + 20.0)  # Hyperpolarization undershoot
        # Recovery to rest
        spike_path.quadTo(vx0 + 180.0, base_y + 5.0, vx0 + 220.0, base_y)
        spike_path.lineTo(vx0 + vw - 20.0, base_y)

        spike_item = QGraphicsPathItem(spike_path)
        spike_item.setPen(QPen(QColor('#f43f5e'), 2.4))
        spike_item.setZValue(10.4)
        items.append(spike_item)

        # Peak badge
        peak_lbl = QGraphicsTextItem(f'Đỉnh: {v_peak_mv:+.1f} mV')
        peak_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        peak_lbl.setDefaultTextColor(QColor('#f43f5e'))
        peak_lbl.setPos(vx0 + 85.0, vy0 + 18.0)
        peak_lbl.setZValue(10.5)
        items.append(peak_lbl)

        # Right box: Equations & Bio-parameters
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        info = QGraphicsTextItem(
            f'THÔNG SỐ ĐIỆN SINH LÝ HỌC:\n'
            f'• Dòng kích thích: {i_inj_ua:.1f} μA/cm²\n'
            f'• Điện thế đỉnh: {v_peak_mv:+.1f} mV | Điện thế nghỉ: -65.0 mV\n'
            f'• Kênh Na⁺: g_Na = ḡ_Na·m³h (E_Na = +50.0 mV)\n'
            f'• Kênh K⁺: g_K = ḡ_K·n⁴ (E_K = -77.0 mV)\n'
            f'• Độc tố ức chế: {toxin.upper()}\n'
            f'• Nhiệt độ mô phỏng: {temp_c:.1f} °C'
        )
        info.setFont(QFont('Segoe UI', 7))
        info.setDefaultTextColor(QColor('#e2e8f0'))
        info.setPos(rx0 + 10.0, vy0 + 12.0)
        info.setZValue(10.4)
        items.append(info)

        cap = QGraphicsTextItem(
            'Mô hình Hodgkin-Huxley (Giải Nobel Y sinh 1963): Điện thế hoạt động phát sinh từ sự biến thiên độ dẫn '
            'của các kênh ion Na⁺ và K⁺ phụ thuộc điện thế trên màng sợi trục thần kinh mực ống khổng lồ, '
            'tuân thủ quy luật Tất cả hoặc Không (All-or-None Law).'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


    @staticmethod
    def create_hardy_weinberg_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CLASSIC_IDEAL_EQUILIBRIUM",
        generation: int = 0,
        p: float = 0.7,
        q: float = 0.3,
        f_aa: float = 0.49,
        f_het: float = 0.42,
        f_rec: float = 0.09,
        pop_size: int = 0,
        chi_square: float = 0.0,
        is_hwe: bool = True,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Hardy-Weinberg Population Genetics Equilibrium."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'SINH HỌC & DI TRUYỀN HỌC QUẦN THỂ: CÂN BẰNG HARDY-WEINBERG – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 240.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Plot axes
        px0 = vx0 + 40.0
        py0 = vy0 + vh - 30.0
        pw = vw - 60.0
        ph = vh - 55.0

        ax_line = QGraphicsLineItem(px0, py0, px0 + pw, py0)
        ax_line.setPen(QPen(QColor('#64748b'), 1.2))
        ax_line.setZValue(10.3)
        items.append(ax_line)

        ay_line = QGraphicsLineItem(px0, py0, px0, py0 - ph)
        ay_line.setPen(QPen(QColor('#64748b'), 1.2))
        ay_line.setZValue(10.3)
        items.append(ay_line)

        # Theoretical curves p², 2pq, q²
        p_aa = QPainterPath()
        p_het = QPainterPath()
        p_rec = QPainterPath()

        steps = 40
        for i in range(steps + 1):
            cur_p = i / float(steps)
            cur_q = 1.0 - cur_p
            cur_x = px0 + cur_p * pw

            y_aa = py0 - (cur_p**2) * ph
            y_het = py0 - (2.0 * cur_p * cur_q) * ph
            y_rec = py0 - (cur_q**2) * ph

            if i == 0:
                p_aa.moveTo(cur_x, y_aa)
                p_het.moveTo(cur_x, y_het)
                p_rec.moveTo(cur_x, y_rec)
            else:
                p_aa.lineTo(cur_x, y_aa)
                p_het.lineTo(cur_x, y_het)
                p_rec.lineTo(cur_x, y_rec)

        # p² (AA) - Red
        item_aa = QGraphicsPathItem(p_aa)
        item_aa.setPen(QPen(QColor('#ef4444'), 2.2))
        item_aa.setZValue(10.4)
        items.append(item_aa)

        # 2pq (Aa) - Purple
        item_het = QGraphicsPathItem(p_het)
        item_het.setPen(QPen(QColor('#a855f7'), 2.2))
        item_het.setZValue(10.4)
        items.append(item_het)

        # q² (aa) - Cyan
        item_rec = QGraphicsPathItem(p_rec)
        item_rec.setPen(QPen(QColor('#06b6d4'), 2.2))
        item_rec.setZValue(10.4)
        items.append(item_rec)

        # Current p indicator line
        curr_x = px0 + p * pw
        p_line = QGraphicsLineItem(curr_x, py0, curr_x, py0 - ph)
        p_line.setPen(QPen(QColor('#f59e0b'), 1.5, Qt.PenStyle.DashLine))
        p_line.setZValue(10.5)
        items.append(p_line)

        # Genotype dots at current p
        dot_aa = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_aa * ph - 3.5, 7.0, 7.0)
        dot_aa.setBrush(QBrush(QColor('#ef4444')))
        dot_aa.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_aa.setZValue(10.6)
        items.append(dot_aa)

        dot_het = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_het * ph - 3.5, 7.0, 7.0)
        dot_het.setBrush(QBrush(QColor('#a855f7')))
        dot_het.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_het.setZValue(10.6)
        items.append(dot_het)

        dot_rec = QGraphicsEllipseItem(curr_x - 3.5, py0 - f_rec * ph - 3.5, 7.0, 7.0)
        dot_rec.setBrush(QBrush(QColor('#06b6d4')))
        dot_rec.setPen(QPen(QColor('#ffffff'), 1.2))
        dot_rec.setZValue(10.6)
        items.append(dot_rec)

        # Legend
        l_aa = QGraphicsTextItem('■ p² (AA)')
        l_aa.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_aa.setDefaultTextColor(QColor('#ef4444'))
        l_aa.setPos(vx0 + 40.0, vy0 + 8.0)
        l_aa.setZValue(10.5)
        items.append(l_aa)

        l_het = QGraphicsTextItem('■ 2pq (Aa)')
        l_het.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_het.setDefaultTextColor(QColor('#a855f7'))
        l_het.setPos(vx0 + 115.0, vy0 + 8.0)
        l_het.setZValue(10.5)
        items.append(l_het)

        l_rec = QGraphicsTextItem('■ q² (aa)')
        l_rec.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l_rec.setDefaultTextColor(QColor('#06b6d4'))
        l_rec.setPos(vx0 + 195.0, vy0 + 8.0)
        l_rec.setZValue(10.5)
        items.append(l_rec)

        # Right Panel: Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        pop_str = "Vô hạn (N → ∞)" if pop_size == 0 else f"{pop_size}"
        hwe_status = "ĐẠT CÂN BẰNG HWE (p > 0.05)" if is_hwe else "LỆCH CÂN BẰNG HWE (p < 0.05)"

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ DI TRUYỀN QUẦN THỂ:\n'
            f'• Thế hệ: {generation} | Kích thước N: {pop_str}\n'
            f'• Tần số alen A (p): {p:.4f}\n'
            f'• Tần số alen a (q): {q:.4f} (p + q = 1.00)\n\n'
            f'TẦN SỐ KIỂU GEN (p² + 2pq + q² = 1.0):\n'
            f'• f(AA) = p² = {f_aa:.4f} ({f_aa*100:.1f}%)\n'
            f'• f(Aa) = 2pq = {f_het:.4f} ({f_het*100:.1f}%)\n'
            f'• f(aa) = q² = {f_rec:.4f} ({f_rec*100:.1f}%)\n\n'
            f'KIỂM ĐỊNH CHI-BÌNH PHƯƠNG (df=1):\n'
            f'• χ² = {chi_square:.3f} (Ngưỡng 3.841, α=0.05)\n'
            f'• Trạng thái: {hwe_status}'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Định luật Hardy-Weinberg (1908): Trong quần thể ngẫu phối vô hạn, không chọn lọc tự nhiên, đột biến hay di nhập gen, '
            'tần số tương đối của các alen và kiểu gen duy trì không đổi qua các thế hệ. '
            'Kiểm định Chi-bình phương so sánh tần số thực nghiệm với kỳ vọng để phát hiện các nhân tố tiến hóa tác động.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items

    @staticmethod
    def create_pcr_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "COVID19_RTPCR_DIAGNOSTICS",
        target_gene: str = "Gen N nucleocapsid",
        amplicon_bp: int = 150,
        cycle_n: int = 25,
        dna_copies: float = 2.5e8,
        ct_threshold: float = 18.5,
        efficiency: float = 0.98,
        annealing_temp_c: float = 58.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of PCR thermal cycling & gel electrophoresis.
        Includes 3-step temperature profiles, DNA strand denaturation/annealing,
        fluorescent agarose gel electrophoresis lanes, and exponential copy metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor(9, 14, 25, 245)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.6))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header Title
        title_txt = QGraphicsTextItem(f"PHẢN ỨNG CHUỖI POLYMERASE (PCR) & ĐIỆN DI GEL: {preset_code}")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#f8fafc"))
        title_txt.setPos(x0 + 16.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Mục tiêu: {target_gene} ({amplicon_bp} bp) | Chu kỳ n = {cycle_n} | "
            f"Số bản sao N = {dna_copies:.2e} | Ngưỡng C_t = {ct_threshold:.1f} | T_a = {annealing_temp_c:.1f}°C"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 16.0, y0 + 34.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: 3-Step Thermal Cycler
        cw = (width - 48.0) * 0.48
        ch = height - 130.0
        cx = x0 + 16.0
        cy = y0 + 58.0

        c_bg = QGraphicsRectItem(cx, cy, cw, ch)
        c_bg.setBrush(QBrush(QColor(4, 7, 16)))
        c_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        c_bg.setZValue(10.1)
        items.append(c_bg)

        c_title = QGraphicsTextItem("CHU TRÌNH NHIỆT 3 BƯỚC:")
        c_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        c_title.setDefaultTextColor(QColor("#38bdf8"))
        c_title.setPos(cx + 8.0, cy + 6.0)
        c_title.setZValue(10.2)
        items.append(c_title)

        # 3 Temperature Step Boxes
        step_w = (cw - 32.0) / 3.0
        step_h = 60.0
        steps_info = [
            ("Biến tính", "95°C", QColor("#ef4444")),
            ("Bắt cặp", f"{annealing_temp_c:.0f}°C", QColor("#38bdf8")),
            ("Kéo dài", "72°C", QColor("#22c55e")),
        ]
        for idx, (s_name, s_temp, s_col) in enumerate(steps_info):
            box_x = cx + 10.0 + idx * (step_w + 6.0)
            box_y = cy + 28.0
            box = QGraphicsRectItem(box_x, box_y, step_w, step_h)
            box.setBrush(QBrush(QColor(s_col.red(), s_col.green(), s_col.blue(), 50)))
            box.setPen(QPen(s_col, 1.2))
            box.setZValue(10.2)
            items.append(box)

            t1 = QGraphicsTextItem(s_name)
            t1.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            t1.setDefaultTextColor(s_col)
            t1.setPos(box_x + 4.0, box_y + 4.0)
            t1.setZValue(10.3)
            items.append(t1)

            t2 = QGraphicsTextItem(s_temp)
            t2.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            t2.setDefaultTextColor(QColor("#ffffff"))
            t2.setPos(box_x + 4.0, box_y + 24.0)
            t2.setZValue(10.3)
            items.append(t2)

        # DNA Strands diagram
        strand_y = cy + 115.0
        s_lbl = QGraphicsTextItem("Tách sợi đôi → Gắn mồi → Kéo dài:")
        s_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        s_lbl.setDefaultTextColor(QColor("#94a3b8"))
        s_lbl.setPos(cx + 10.0, strand_y)
        s_lbl.setZValue(10.2)
        items.append(s_lbl)

        line1 = QGraphicsLineItem(cx + 15.0, strand_y + 25.0, cx + cw - 15.0, strand_y + 25.0)
        line1.setPen(QPen(QColor("#ef4444"), 2.2))
        line1.setZValue(10.3)
        items.append(line1)

        line2 = QGraphicsLineItem(cx + 15.0, strand_y + 45.0, cx + cw - 15.0, strand_y + 45.0)
        line2.setPen(QPen(QColor("#38bdf8"), 2.2))
        line2.setZValue(10.3)
        items.append(line2)

        # Primers
        p1 = QGraphicsLineItem(cx + 25.0, strand_y + 32.0, cx + 85.0, strand_y + 32.0)
        p1.setPen(QPen(QColor("#22c55e"), 3.5))
        p1.setZValue(10.4)
        items.append(p1)

        p2 = QGraphicsLineItem(cx + cw - 85.0, strand_y + 38.0, cx + cw - 25.0, strand_y + 38.0)
        p2.setPen(QPen(QColor("#22c55e"), 3.5))
        p2.setZValue(10.4)
        items.append(p2)

        # 4. Right Panel: Agarose Gel Electrophoresis
        rw = (width - 48.0) * 0.52
        rh = ch
        rx = cx + cw + 16.0
        ry = cy

        r_bg = QGraphicsRectItem(rx, ry, rw, rh)
        r_bg.setBrush(QBrush(QColor(4, 7, 16)))
        r_bg.setPen(QPen(QColor(30, 41, 59), 1.2))
        r_bg.setZValue(10.1)
        items.append(r_bg)

        r_title = QGraphicsTextItem("BẢN ĐIỆN DI GEL AGAROSE (302 nm UV):")
        r_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        r_title.setDefaultTextColor(QColor("#38bdf8"))
        r_title.setPos(rx + 8.0, ry + 6.0)
        r_title.setZValue(10.2)
        items.append(r_title)

        # Gel Slab
        gel_w = rw - 50.0
        gel_h = rh - 55.0
        gel_x = rx + 25.0
        gel_y = ry + 25.0

        gel = QGraphicsRectItem(gel_x, gel_y, gel_w, gel_h)
        gel.setBrush(QBrush(QColor(10, 20, 45)))
        gel.setPen(QPen(QColor(30, 58, 138), 1.4))
        gel.setZValue(10.2)
        items.append(gel)

        # Wells
        lane_w = gel_w / 3.0
        for l in range(3):
            wx = gel_x + l * lane_w + lane_w * 0.2
            well = QGraphicsRectItem(wx, gel_y + 4.0, lane_w * 0.6, 6.0)
            well.setBrush(QBrush(QColor(2, 6, 23)))
            well.setPen(QPen(QColor(71, 85, 105), 0.8))
            well.setZValue(10.3)
            items.append(well)

        # DNA Ladder Bands in Lane 1
        l1_x = gel_x + 0.2 * lane_w
        for frac, bp in [(0.25, 1500), (0.40, 1000), (0.55, 750), (0.70, 500), (0.85, 250), (0.95, 100)]:
            by = gel_y + 15.0 + frac * (gel_h - 25.0)
            band = QGraphicsRectItem(l1_x, by, lane_w * 0.6, 3.0)
            band.setBrush(QBrush(QColor("#eab308")))
            band.setPen(Qt.PenStyle.NoPen)
            band.setZValue(10.4)
            items.append(band)

        # Target PCR band in Lane 2
        l2_x = gel_x + 1.2 * lane_w
        target_frac = 0.90 if amplicon_bp <= 150 else (0.70 if amplicon_bp <= 500 else 0.45)
        by_target = gel_y + 15.0 + target_frac * (gel_h - 25.0)
        t_band = QGraphicsRectItem(l2_x, by_target, lane_w * 0.6, 4.5)
        t_band.setBrush(QBrush(QColor("#38bdf8")))
        t_band.setPen(Qt.PenStyle.NoPen)
        t_band.setZValue(10.4)
        items.append(t_band)

        band_lbl = QGraphicsTextItem(f"{amplicon_bp} bp")
        band_lbl.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        band_lbl.setDefaultTextColor(QColor("#38bdf8"))
        band_lbl.setPos(l2_x + lane_w * 0.6 + 4.0, by_target - 6.0)
        band_lbl.setZValue(10.4)
        items.append(band_lbl)

        # 5. Bottom Quantitative Metrics
        metrics_txt = QGraphicsTextItem(
            f"• Khuếch đại N(n) = N_0(1+E)^n: {dna_copies:.2e} bản sao    • Ngưỡng chu kỳ C_t: {ct_threshold:.2f}    • Hiệu suất E: {efficiency*100:.1f}%\n"
            f"• Nhiệt độ bắt cặp T_a: {annealing_temp_c:.1f}°C    • Kích thước amplicons: {amplicon_bp} bp ({target_gene})"
        )
        metrics_txt.setFont(QFont("Segoe UI", 7))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(rx + 8.0, gel_y + gel_h + 8.0)
        metrics_txt.setZValue(10.3)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phản ứng chuỗi Polymerase (PCR) tạo ra hàng tỷ bản sao đoạn ADN đích qua 30-40 chu trình nhiệt lặp lại: biến tính 95°C, bắt cặp mồi 55-60°C và kéo dài 72°C. "
            "Kỹ thuật RT-qPCR đo cường độ huỳnh quang theo chu kỳ xác định ngưỡng C_t tỷ lệ nghịch với tải lượng virus khởi đầu."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 34.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items

    @staticmethod
    def create_enzyme_kinetics_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "CHYMOTRYPSIN_PEPTIDE_CLEAVAGE",
        enzyme_name: str = "Chymotrypsin",
        substrate_name: str = "ATEE",
        km_mM: float = 1.5,
        vmax_uM_per_s: float = 100.0,
        substrate_mM: float = 1.5,
        v0_uM_per_s: float = 50.0,
        inhibition_type: str = "NONE",
        alpha: float = 1.0,
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Michaelis-Menten Enzyme Kinetics
        and Lineweaver-Burk Double Reciprocal Plot.
        Shows hyperbolic curve v0 = Vmax*[S]/(Km+[S]), double reciprocal line 1/v0 vs 1/[S],
        apparent Km/Vmax shifts under competitive/non-competitive/uncompetitive inhibition,
        and saturation metrics.
        """
        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg_card = QGraphicsRectItem(x0, y0, width, height)
        bg_card.setBrush(QBrush(QColor("#0f172a")))
        bg_card.setPen(QPen(QColor("#1e293b"), 1.8))
        bg_card.setZValue(10.0)
        items.append(bg_card)

        # 2. Header & Badges
        badge_box = QGraphicsRectItem(x0 + 16.0, y0 + 14.0, 195.0, 22.0)
        badge_box.setBrush(QBrush(QColor("#0284c7")))
        badge_box.setPen(QPen(QColor("#38bdf8"), 1.0))
        badge_box.setZValue(10.1)
        items.append(badge_box)

        badge_txt = QGraphicsTextItem("🧬 BIOCHEMISTRY • ENZYME KINETICS")
        badge_txt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge_txt.setDefaultTextColor(QColor("#ffffff"))
        badge_txt.setPos(x0 + 18.0, y0 + 14.0)
        badge_txt.setZValue(10.2)
        items.append(badge_txt)

        title_txt = QGraphicsTextItem("ĐỘNG HỌC ENZYME MICHAELIS-MENTEN & ĐỒ THỊ LINEWEAVER-BURK")
        title_txt.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#38bdf8"))
        title_txt.setPos(x0 + 220.0, y0 + 12.0)
        title_txt.setZValue(10.2)
        items.append(title_txt)

        sub_txt = QGraphicsTextItem(
            f"Enzyme: {enzyme_name} | Cơ chất: {substrate_name} | Chế độ: {inhibition_type} (α = {alpha:.2f})"
        )
        sub_txt.setFont(QFont("Segoe UI", 8))
        sub_txt.setDefaultTextColor(QColor("#94a3b8"))
        sub_txt.setPos(x0 + 18.0, y0 + 38.0)
        sub_txt.setZValue(10.2)
        items.append(sub_txt)

        # 3. Left Panel: Michaelis-Menten Hyperbolic Curve (v0 vs [S])
        p1_x = x0 + 20.0
        p1_y = y0 + 64.0
        p1_w = 305.0
        p1_h = 195.0

        p1_box = QGraphicsRectItem(p1_x, p1_y, p1_w, p1_h)
        p1_box.setBrush(QBrush(QColor("#090d16")))
        p1_box.setPen(QPen(QColor("#334155"), 1.0))
        p1_box.setZValue(10.1)
        items.append(p1_box)

        p1_lbl = QGraphicsTextItem("1. Hyperbol Michaelis-Menten: v₀ = Vmax·[S] / (Km + [S])")
        p1_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p1_lbl.setDefaultTextColor(QColor("#38bdf8"))
        p1_lbl.setPos(p1_x + 8.0, p1_y + 6.0)
        p1_lbl.setZValue(10.2)
        items.append(p1_lbl)

        # Coordinate axes for Plot 1
        ox1 = p1_x + 38.0
        oy1 = p1_y + p1_h - 26.0
        pw1 = p1_w - 50.0
        ph1 = p1_h - 48.0

        # Grid lines
        for gi in range(1, 4):
            gy = oy1 - gi * (ph1 / 3.0)
            g_line = QGraphicsLineItem(ox1, gy, ox1 + pw1, gy)
            g_line.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
            g_line.setZValue(10.15)
            items.append(g_line)

        # Axes
        ax1_x = QGraphicsLineItem(ox1, oy1, ox1 + pw1 + 6.0, oy1)
        ax1_x.setPen(QPen(QColor("#64748b"), 1.2))
        ax1_x.setZValue(10.2)
        items.append(ax1_x)

        ax1_y = QGraphicsLineItem(ox1, oy1, ox1, oy1 - ph1 - 6.0)
        ax1_y.setPen(QPen(QColor("#64748b"), 1.2))
        ax1_y.setZValue(10.2)
        items.append(ax1_y)

        x1_label = QGraphicsTextItem("[S] (mM)")
        x1_label.setFont(QFont("Segoe UI", 7))
        x1_label.setDefaultTextColor(QColor("#94a3b8"))
        x1_label.setPos(ox1 + pw1 - 32.0, oy1 + 4.0)
        x1_label.setZValue(10.2)
        items.append(x1_label)

        y1_label = QGraphicsTextItem("v₀ (µM/s)")
        y1_label.setFont(QFont("Segoe UI", 7))
        y1_label.setDefaultTextColor(QColor("#94a3b8"))
        y1_label.setPos(ox1 - 34.0, oy1 - ph1 - 10.0)
        y1_label.setZValue(10.2)
        items.append(y1_label)

        # Plot 1 scaling
        s_max_plot = max(10.0, km_mM * 4.0, substrate_mM * 1.5)
        v_max_plot = max(vmax_uM_per_s * 1.2, 10.0)

        # Asymptote Vmax
        asymptote_y = oy1 - (vmax_uM_per_s / v_max_plot) * ph1
        asymptote = QGraphicsLineItem(ox1, asymptote_y, ox1 + pw1, asymptote_y)
        asymptote.setPen(QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DashLine))
        asymptote.setZValue(10.2)
        items.append(asymptote)

        vmax_tag = QGraphicsTextItem(f"Vmax = {vmax_uM_per_s:.0f}")
        vmax_tag.setFont(QFont("Consolas", 7))
        vmax_tag.setDefaultTextColor(QColor("#f43f5e"))
        vmax_tag.setPos(ox1 + pw1 - 56.0, asymptote_y - 14.0)
        vmax_tag.setZValue(10.25)
        items.append(vmax_tag)

        # Km reference
        km_x = ox1 + (km_mM / s_max_plot) * pw1
        half_vmax_y = oy1 - ((vmax_uM_per_s * 0.5) / v_max_plot) * ph1
        km_v_line = QGraphicsLineItem(km_x, oy1, km_x, half_vmax_y)
        km_v_line.setPen(QPen(QColor("#0ea5e9"), 1.0, Qt.PenStyle.DotLine))
        km_v_line.setZValue(10.2)
        items.append(km_v_line)

        km_h_line = QGraphicsLineItem(ox1, half_vmax_y, km_x, half_vmax_y)
        km_h_line.setPen(QPen(QColor("#0ea5e9"), 1.0, Qt.PenStyle.DotLine))
        km_h_line.setZValue(10.2)
        items.append(km_h_line)

        km_tag = QGraphicsTextItem(f"Km={km_mM:.1f}")
        km_tag.setFont(QFont("Consolas", 7))
        km_tag.setDefaultTextColor(QColor("#38bdf8"))
        km_tag.setPos(km_x - 18.0, oy1 + 2.0)
        km_tag.setZValue(10.25)
        items.append(km_tag)

        # Plot 1 Curve Path
        path1 = QPainterPath()
        num_pts = 60
        for i in range(num_pts + 1):
            s_val = (i / num_pts) * s_max_plot
            v_val = (vmax_uM_per_s * s_val) / (km_mM + s_val) if (km_mM + s_val) > 1e-9 else 0.0
            px = ox1 + (s_val / s_max_plot) * pw1
            py = oy1 - (v_val / v_max_plot) * ph1
            if i == 0:
                path1.moveTo(px, py)
            else:
                path1.lineTo(px, py)

        curve_item1 = QGraphicsPathItem(path1)
        curve_item1.setPen(QPen(QColor("#38bdf8"), 2.2))
        curve_item1.setZValue(10.3)
        items.append(curve_item1)

        # Operating Point on Curve 1
        curr_px1 = ox1 + (substrate_mM / s_max_plot) * pw1
        curr_py1 = oy1 - (v0_uM_per_s / v_max_plot) * ph1
        pt_halo1 = QGraphicsEllipseItem(curr_px1 - 6.0, curr_py1 - 6.0, 12.0, 12.0)
        pt_halo1.setBrush(QBrush(QColor(56, 189, 248, 80)))
        pt_halo1.setPen(Qt.PenStyle.NoPen)
        pt_halo1.setZValue(10.35)
        items.append(pt_halo1)

        pt_dot1 = QGraphicsEllipseItem(curr_px1 - 3.5, curr_py1 - 3.5, 7.0, 7.0)
        pt_dot1.setBrush(QBrush(QColor("#ffffff")))
        pt_dot1.setPen(QPen(QColor("#0284c7"), 1.5))
        pt_dot1.setZValue(10.4)
        items.append(pt_dot1)

        # 4. Right Panel: Lineweaver-Burk Double Reciprocal Plot (1/v0 vs 1/[S])
        p2_x = x0 + 345.0
        p2_y = y0 + 64.0
        p2_w = 315.0
        p2_h = 195.0

        p2_box = QGraphicsRectItem(p2_x, p2_y, p2_w, p2_h)
        p2_box.setBrush(QBrush(QColor("#090d16")))
        p2_box.setPen(QPen(QColor("#334155"), 1.0))
        p2_box.setZValue(10.1)
        items.append(p2_box)

        p2_lbl = QGraphicsTextItem("2. Đồ thị Nghịch đảo Kép: 1/v₀ = (Km/Vmax)·(1/[S]) + 1/Vmax")
        p2_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p2_lbl.setDefaultTextColor(QColor("#fbbf24"))
        p2_lbl.setPos(p2_x + 8.0, p2_y + 6.0)
        p2_lbl.setZValue(10.2)
        items.append(p2_lbl)

        # Origin for Lineweaver-Burk: allow negative 1/[S] to show -1/Km intercept
        ox2 = p2_x + 95.0   # shifted right to show negative axis
        oy2 = p2_y + p2_h - 26.0
        pw2_pos = p2_w - 110.0
        pw2_neg = 75.0
        ph2 = p2_h - 48.0

        # Axes for Plot 2
        ax2_x = QGraphicsLineItem(ox2 - pw2_neg, oy2, ox2 + pw2_pos + 6.0, oy2)
        ax2_x.setPen(QPen(QColor("#64748b"), 1.2))
        ax2_x.setZValue(10.2)
        items.append(ax2_x)

        ax2_y = QGraphicsLineItem(ox2, oy2, ox2, oy2 - ph2 - 6.0)
        ax2_y.setPen(QPen(QColor("#64748b"), 1.2))
        ax2_y.setZValue(10.2)
        items.append(ax2_y)

        x2_label = QGraphicsTextItem("1/[S] (mM⁻¹)")
        x2_label.setFont(QFont("Segoe UI", 7))
        x2_label.setDefaultTextColor(QColor("#94a3b8"))
        x2_label.setPos(ox2 + pw2_pos - 42.0, oy2 + 4.0)
        x2_label.setZValue(10.2)
        items.append(x2_label)

        y2_label = QGraphicsTextItem("1/v₀")
        y2_label.setFont(QFont("Segoe UI", 7))
        y2_label.setDefaultTextColor(QColor("#94a3b8"))
        y2_label.setPos(ox2 - 28.0, oy2 - ph2 - 10.0)
        y2_label.setZValue(10.2)
        items.append(y2_label)

        # Scaling for 1/S and 1/v0
        inv_s_max = max(2.5, 2.0 / km_mM)
        inv_v_max = max(0.04, 2.5 / vmax_uM_per_s)

        # Lineweaver-Burk line: y = m*x + c where c = 1/Vmax, m = Km/Vmax
        c_inv_vmax = 1.0 / vmax_uM_per_s if vmax_uM_per_s > 0 else 0.01
        m_slope = km_mM / vmax_uM_per_s if vmax_uM_per_s > 0 else 0.01
        x_int = -1.0 / km_mM if km_mM > 0 else -1.0

        # Pixel coords for x_int and positive extreme
        x_int_px = ox2 + (x_int / inv_s_max) * pw2_pos
        x_int_px = max(ox2 - pw2_neg, min(ox2, x_int_px))
        y_int_py = oy2 - (c_inv_vmax / inv_v_max) * ph2

        # Positive point
        x_pos_val = inv_s_max * 0.95
        y_pos_val = m_slope * x_pos_val + c_inv_vmax
        x_pos_px = ox2 + (x_pos_val / inv_s_max) * pw2_pos
        y_pos_py = oy2 - (y_pos_val / inv_v_max) * ph2

        lb_line = QGraphicsLineItem(x_int_px, oy2, x_pos_px, y_pos_py)
        lb_line.setPen(QPen(QColor("#f59e0b"), 2.0))
        lb_line.setZValue(10.3)
        items.append(lb_line)

        # Intercept markers
        # y-intercept (1/Vmax)
        y_int_dot = QGraphicsEllipseItem(ox2 - 3.0, y_int_py - 3.0, 6.0, 6.0)
        y_int_dot.setBrush(QBrush(QColor("#f43f5e")))
        y_int_dot.setPen(Qt.PenStyle.NoPen)
        y_int_dot.setZValue(10.4)
        items.append(y_int_dot)

        y_int_tag = QGraphicsTextItem("1/Vmax")
        y_int_tag.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        y_int_tag.setDefaultTextColor(QColor("#f43f5e"))
        y_int_tag.setPos(ox2 + 4.0, y_int_py - 9.0)
        y_int_tag.setZValue(10.4)
        items.append(y_int_tag)

        # x-intercept (-1/Km)
        x_int_dot = QGraphicsEllipseItem(x_int_px - 3.0, oy2 - 3.0, 6.0, 6.0)
        x_int_dot.setBrush(QBrush(QColor("#0ea5e9")))
        x_int_dot.setPen(Qt.PenStyle.NoPen)
        x_int_dot.setZValue(10.4)
        items.append(x_int_dot)

        x_int_tag = QGraphicsTextItem("-1/Km")
        x_int_tag.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        x_int_tag.setDefaultTextColor(QColor("#38bdf8"))
        x_int_tag.setPos(x_int_px - 14.0, oy2 + 4.0)
        x_int_tag.setZValue(10.4)
        items.append(x_int_tag)

        # 5. Bottom Quantitative Metrics Bar
        theta_sat = (substrate_mM / (km_mM + substrate_mM)) * 100.0 if (km_mM + substrate_mM) > 0 else 0.0
        metrics_box = QGraphicsRectItem(p1_x, y0 + 268.0, width - 40.0, 52.0)
        metrics_box.setBrush(QBrush(QColor("#0d1526")))
        metrics_box.setPen(QPen(QColor("#1e293b"), 1.0))
        metrics_box.setZValue(10.15)
        items.append(metrics_box)

        metrics_txt = QGraphicsTextItem(
            f"• Nồng độ [S] = {substrate_mM:.2f} mM  ➔  Vận tốc v₀ = {v0_uM_per_s:.2f} µM/s ({theta_sat:.1f}% Vmax)\n"
            f"• Km (hoặc Km_app) = {km_mM:.2f} mM    • Vmax (hoặc Vmax_app) = {vmax_uM_per_s:.1f} µM/s    • Hệ số ức chế α = {alpha:.2f}"
        )
        metrics_txt.setFont(QFont("Consolas", 8))
        metrics_txt.setDefaultTextColor(QColor("#e2e8f0"))
        metrics_txt.setPos(p1_x + 10.0, y0 + 274.0)
        metrics_txt.setZValue(10.25)
        items.append(metrics_txt)

        # 6. Pedagogical Summary Caption
        cap = QGraphicsTextItem(
            "Phương trình Michaelis-Menten mô tả động học bão hòa của phản ứng xúc tác enzyme đơn cơ chất. "
            "Biểu diễn Lineweaver-Burk tuyến tính hóa dữ liệu giúp chẩn đoán cơ chế ức chế: Cạnh tranh (quay quanh trục 1/v₀), "
            "Phi cạnh tranh (quay quanh trục 1/[S]), hoặc Bất cạnh tranh (đường song song dịch chuyển)."
        )
        cap.setFont(QFont("Segoe UI", 8))
        cap.setDefaultTextColor(QColor("#94a3b8"))
        cap.setPos(x0 + 16.0, y0 + height - 42.0)
        cap.setZValue(10.2)
        items.append(cap)

        return items


class InformaticsTools:
    """Algorithm flowcharts, decision diamonds, and code block components."""

    @staticmethod
    def create_flowchart_shape(
        shape_type: str,
        center: tuple[float, float],
        width: float = 140.0,
        height: float = 70.0,
        label: str = "",
        color: QColor = QColor("#1a73e8"),
        bg_color: QColor = QColor("#e8f0fe"),
    ) -> list[QGraphicsItem]:
        """
        Creates flowchart algorithm symbols:
          - TERMINATOR: Oval (Start / End)
          - PROCESS: Rectangle (Statements)
          - DECISION: Diamond (If / Switch)
          - IO: Parallelogram (Input / Output)
        """
        items: list[QGraphicsItem] = []
        cx, cy = center
        st = shape_type.upper()
        pen = QPen(color, 2.2, Qt.PenStyle.SolidLine)
        brush = QBrush(bg_color)

        if "DECISION" in st or "ĐIỀU KIỆN" in st:
            # Diamond
            poly = QPolygonF([
                QPointF(cx, cy - height / 2.0),
                QPointF(cx + width / 2.0, cy),
                QPointF(cx, cy + height / 2.0),
                QPointF(cx - width / 2.0, cy),
            ])
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            item.setBrush(brush)
            items.append(item)

        elif "IO" in st or "VÀO RA" in st:
            # Parallelogram
            offset = 18.0
            poly = QPolygonF([
                QPointF(cx - width / 2.0 + offset, cy - height / 2.0),
                QPointF(cx + width / 2.0, cy - height / 2.0),
                QPointF(cx + width / 2.0 - offset, cy + height / 2.0),
                QPointF(cx - width / 2.0, cy + height / 2.0),
            ])
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            item.setBrush(brush)
            items.append(item)

        elif "TERMINATOR" in st or "BẮT ĐẦU" in st or "KẾT THÚC" in st:
            # Rounded capsule
            path = QPainterPath()
            path.addRoundedRect(cx - width / 2.0, cy - height / 2.0, width, height, height / 2.0, height / 2.0)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(brush)
            items.append(item)

        else:
            # PROCESS rectangle
            item = QGraphicsRectItem(cx - width / 2.0, cy - height / 2.0, width, height)
            item.setPen(pen)
            item.setBrush(brush)
            items.append(item)

        # Centered label
        if label:
            txt = QGraphicsTextItem(label)
            txt.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            txt.setDefaultTextColor(QColor("#202124"))
            rect = txt.boundingRect()
            txt.setPos(cx - rect.width() / 2.0, cy - rect.height() / 2.0)
            txt.setZValue(12.0)
            items.append(txt)

        return items

    @staticmethod
    def create_code_block(
        top_left: tuple[float, float],
        width: float = 460.0,
        height: float = 240.0,
        code_text: str = "",
        title: str = "main.py",
    ) -> list[QGraphicsItem]:
        """Creates a professional styled dark IDE code block container."""
        items: list[QGraphicsItem] = []
        x, y = top_left

        # Container
        bg = QGraphicsRectItem(x, y, width, height)
        bg.setBrush(QBrush(QColor("#1e1e2e")))
        bg.setPen(QPen(QColor("#313244"), 1.5))
        bg.setZValue(11.0)
        items.append(bg)

        # Header Bar
        header = QGraphicsRectItem(x, y, width, 32.0)
        header.setBrush(QBrush(QColor("#181825")))
        header.setPen(Qt.PenStyle.NoPen)
        header.setZValue(11.2)
        items.append(header)

        # Window dots (macOS/Linux style red/yellow/green)
        for i, dot_color in enumerate(["#f38ba8", "#f9e2af", "#a6e3a1"]):
            dot = QGraphicsEllipseItem(x + 12 + i * 16, y + 10, 10, 10)
            dot.setBrush(QBrush(QColor(dot_color)))
            dot.setPen(Qt.PenStyle.NoPen)
            dot.setZValue(11.5)
            items.append(dot)

        # Header title
        title_txt = QGraphicsTextItem(title)
        title_txt.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        title_txt.setDefaultTextColor(QColor("#cdd6f4"))
        title_txt.setPos(x + width / 2.0 - 30, y + 5)
        title_txt.setZValue(11.5)
        items.append(title_txt)

        # Code content
        code_item = QGraphicsTextItem(code_text or "def solve():\n    print('Hello, AirTeach AI!')")
        code_item.setFont(QFont("Consolas", 10))
        code_item.setDefaultTextColor(QColor("#cdd6f4"))
        code_item.setPos(x + 14, y + 38)
        code_item.setZValue(12.0)
        items.append(code_item)

        return items


    @staticmethod
    def create_huffman_coding_diagram(
        center: tuple[float, float],
        width: float = 640.0,
        height: float = 340.0,
        preset_code: str = 'CLASSIC_ABRACADABRA',
        text_sample: str = 'ABRACADABRA',
        entropy: float = 2.04,
        avg_code_len: float = 2.09,
        efficiency_pct: float = 97.6,
        saved_storage_pct: float = 73.9,
        codes: Optional[dict[str, str]] = None,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Huffman Prefix-Free Tree and Shannon Entropy analysis."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#10b981'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & LÝ THUYẾT THÔNG TIN: CÂY TIỀN TỐ HUFFMAN & ENTROPY SHANNON – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 42.0
        vw = 320.0
        vh = 200.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Draw representative Huffman Tree
        root_x = vx0 + vw / 2.0
        root_y = vy0 + 30.0

        # Root
        root_circle = QGraphicsEllipseItem(root_x - 14.0, root_y - 14.0, 28.0, 28.0)
        root_circle.setBrush(QBrush(QColor('#1e293b')))
        root_circle.setPen(QPen(QColor('#38bdf8'), 1.5))
        root_circle.setZValue(10.4)
        items.append(root_circle)

        root_lbl = QGraphicsTextItem('11')
        root_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        root_lbl.setDefaultTextColor(QColor('#f8fafc'))
        root_lbl.setPos(root_x - 7.0, root_y - 10.0)
        root_lbl.setZValue(10.5)
        items.append(root_lbl)

        # Left Child: Leaf 'A' (bit 0)
        leaf_a_x = root_x - 70.0
        leaf_a_y = root_y + 60.0
        line_l = QGraphicsLineItem(root_x, root_y, leaf_a_x, leaf_a_y)
        line_l.setPen(QPen(QColor('#38bdf8'), 2.0))
        line_l.setZValue(10.3)
        items.append(line_l)

        b0_lbl = QGraphicsTextItem('0')
        b0_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        b0_lbl.setDefaultTextColor(QColor('#38bdf8'))
        b0_lbl.setPos((root_x + leaf_a_x) / 2.0 - 15.0, (root_y + leaf_a_y) / 2.0 - 12.0)
        b0_lbl.setZValue(10.5)
        items.append(b0_lbl)

        box_a = QGraphicsRectItem(leaf_a_x - 16.0, leaf_a_y - 14.0, 32.0, 28.0)
        box_a.setBrush(QBrush(QColor('#10b981')))
        box_a.setPen(QPen(QColor('#ffffff'), 1.5))
        box_a.setZValue(10.4)
        items.append(box_a)

        lbl_a = QGraphicsTextItem("'A'\n5")
        lbl_a.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        lbl_a.setDefaultTextColor(QColor('#020617'))
        lbl_a.setPos(leaf_a_x - 10.0, leaf_a_y - 14.0)
        lbl_a.setZValue(10.5)
        items.append(lbl_a)

        # Right Subtree: Internal Node (bit 1)
        sub_r_x = root_x + 70.0
        sub_r_y = root_y + 50.0
        line_r = QGraphicsLineItem(root_x, root_y, sub_r_x, sub_r_y)
        line_r.setPen(QPen(QColor('#f59e0b'), 2.0))
        line_r.setZValue(10.3)
        items.append(line_r)

        b1_lbl = QGraphicsTextItem('1')
        b1_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        b1_lbl.setDefaultTextColor(QColor('#f59e0b'))
        b1_lbl.setPos((root_x + sub_r_x) / 2.0 + 4.0, (root_y + sub_r_y) / 2.0 - 12.0)
        b1_lbl.setZValue(10.5)
        items.append(b1_lbl)

        node_r = QGraphicsEllipseItem(sub_r_x - 12.0, sub_r_y - 12.0, 24.0, 24.0)
        node_r.setBrush(QBrush(QColor('#1e293b')))
        node_r.setPen(QPen(QColor('#f59e0b'), 1.5))
        node_r.setZValue(10.4)
        items.append(node_r)

        # Children of Right Subtree
        # Leaf 'R'
        leaf_r_x = sub_r_x - 35.0
        leaf_r_y = sub_r_y + 55.0
        line_r_l = QGraphicsLineItem(sub_r_x, sub_r_y, leaf_r_x, leaf_r_y)
        line_r_l.setPen(QPen(QColor('#38bdf8'), 1.5))
        line_r_l.setZValue(10.3)
        items.append(line_r_l)

        box_r = QGraphicsRectItem(leaf_r_x - 14.0, leaf_r_y - 12.0, 28.0, 24.0)
        box_r.setBrush(QBrush(QColor('#10b981')))
        box_r.setPen(QPen(QColor('#ffffff'), 1.2))
        box_r.setZValue(10.4)
        items.append(box_r)

        lbl_r = QGraphicsTextItem("'R'\n2")
        lbl_r.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        lbl_r.setDefaultTextColor(QColor('#020617'))
        lbl_r.setPos(leaf_r_x - 9.0, leaf_r_y - 13.0)
        lbl_r.setZValue(10.5)
        items.append(lbl_r)

        # Leaf 'B'
        leaf_b_x = sub_r_x + 35.0
        leaf_b_y = sub_r_y + 55.0
        line_r_r = QGraphicsLineItem(sub_r_x, sub_r_y, leaf_b_x, leaf_b_y)
        line_r_r.setPen(QPen(QColor('#f59e0b'), 1.5))
        line_r_r.setZValue(10.3)
        items.append(line_r_r)

        box_b = QGraphicsRectItem(leaf_b_x - 14.0, leaf_b_y - 12.0, 28.0, 24.0)
        box_b.setBrush(QBrush(QColor('#10b981')))
        box_b.setPen(QPen(QColor('#ffffff'), 1.2))
        box_b.setZValue(10.4)
        items.append(box_b)

        lbl_b = QGraphicsTextItem("'B'\n2")
        lbl_b.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        lbl_b.setDefaultTextColor(QColor('#020617'))
        lbl_b.setPos(leaf_b_x - 9.0, leaf_b_y - 13.0)
        lbl_b.setZValue(10.5)
        items.append(lbl_b)

        # Right Panel: Prefix Codes & Shannon Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        sample_codes = codes or {'A': '0', 'B': '10', 'R': '11', 'C': '100', 'D': '101'}
        codes_str = '\n'.join([f"• '{c}': {code} ({len(code)} bit)" for c, code in list(sample_codes.items())[:5]])

        info_txt = QGraphicsTextItem(
            f'BẢNG MÃ TIỀN TỐ HUFFMAN:\n'
            f'{codes_str}\n\n'
            f'THÔNG SỐ LÝ THUYẾT THÔNG TIN:\n'
            f'• Chuỗi mẫu: "{text_sample}"\n'
            f'• Entropy Shannon H(X): {entropy:.3f} bits/ký tự\n'
            f'• Độ dài mã trung bình L̄: {avg_code_len:.3f} bits/ký tự\n'
            f'• Hiệu suất mã hóa: {efficiency_pct:.1f}% (Cận Shannon)\n'
            f'• Tiết kiệm dung lượng: {saved_storage_pct:.1f}% vs ASCII'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Huffman: Xây dựng cây nhị phân tiền tố tối ưu bằng hàng đợi ưu tiên Min-Heap. '
            'Ký tự xuất hiện càng nhiều có mã càng ngắn. Định lý Mã hóa nguồn Shannon khẳng định độ dài trung bình '
            'luôn đạt cận H(X) ≤ L̄ < H(X) + 1, mang lại hiệu suất nén dữ liệu tối đa.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 12.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items

    @staticmethod
    def create_aho_corasick_diagram(
        center: tuple[float, float],
        width: float = 660.0,
        height: float = 360.0,
        preset_code: str = 'CLASSIC_BELL_LABS_HE_SHE_HIS_HERS',
        patterns: Optional[list[str]] = None,
        sample_text: str = 'ushers',
        num_nodes: int = 10,
        num_matches: int = 3,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Aho-Corasick Multi-Pattern String Matching Automaton."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#080c14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & LÝ THUYẾT OTOMAT: AHO-CORASICK MULTI-PATTERN DFA – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 40.0
        vw = 340.0
        vh = 220.0

        v_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        v_box.setBrush(QBrush(QColor('#020617')))
        v_box.setPen(QPen(QColor('#1e293b'), 1.0))
        v_box.setZValue(10.2)
        items.append(v_box)

        # Root (0)
        r_x, r_y = vx0 + 40.0, vy0 + 110.0
        root_c = QGraphicsEllipseItem(r_x - 14.0, r_y - 14.0, 28.0, 28.0)
        root_c.setBrush(QBrush(QColor('#1e293b')))
        root_c.setPen(QPen(QColor('#38bdf8'), 1.5))
        root_c.setZValue(10.4)
        items.append(root_c)

        r_lbl = QGraphicsTextItem('0')
        r_lbl.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        r_lbl.setDefaultTextColor(QColor('#f8fafc'))
        r_lbl.setPos(r_x - 6.0, r_y - 10.0)
        r_lbl.setZValue(10.5)
        items.append(r_lbl)

        # Node 1 ('h')
        n1_x, n1_y = vx0 + 120.0, vy0 + 60.0
        line_0_1 = QGraphicsLineItem(r_x, r_y, n1_x, n1_y)
        line_0_1.setPen(QPen(QColor('#38bdf8'), 1.8))
        line_0_1.setZValue(10.3)
        items.append(line_0_1)

        c1 = QGraphicsEllipseItem(n1_x - 12.0, n1_y - 12.0, 24.0, 24.0)
        c1.setBrush(QBrush(QColor('#0f172a')))
        c1.setPen(QPen(QColor('#38bdf8'), 1.5))
        c1.setZValue(10.4)
        items.append(c1)

        l1 = QGraphicsTextItem('1:h')
        l1.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l1.setDefaultTextColor(QColor('#38bdf8'))
        l1.setPos(n1_x - 9.0, n1_y - 9.0)
        l1.setZValue(10.5)
        items.append(l1)

        # Node 2 ('e') -> Terminal 'he'
        n2_x, n2_y = vx0 + 200.0, vy0 + 40.0
        line_1_2 = QGraphicsLineItem(n1_x, n1_y, n2_x, n2_y)
        line_1_2.setPen(QPen(QColor('#38bdf8'), 1.8))
        line_1_2.setZValue(10.3)
        items.append(line_1_2)

        c2 = QGraphicsEllipseItem(n2_x - 14.0, n2_y - 14.0, 28.0, 28.0)
        c2.setBrush(QBrush(QColor('#10b981')))
        c2.setPen(QPen(QColor('#ffffff'), 1.8))
        c2.setZValue(10.4)
        items.append(c2)

        l2 = QGraphicsTextItem('2:e★')
        l2.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l2.setDefaultTextColor(QColor('#020617'))
        l2.setPos(n2_x - 12.0, n2_y - 9.0)
        l2.setZValue(10.5)
        items.append(l2)

        # Node 3 ('s')
        n3_x, n3_y = vx0 + 120.0, vy0 + 160.0
        line_0_3 = QGraphicsLineItem(r_x, r_y, n3_x, n3_y)
        line_0_3.setPen(QPen(QColor('#a855f7'), 1.8))
        line_0_3.setZValue(10.3)
        items.append(line_0_3)

        c3 = QGraphicsEllipseItem(n3_x - 12.0, n3_y - 12.0, 24.0, 24.0)
        c3.setBrush(QBrush(QColor('#0f172a')))
        c3.setPen(QPen(QColor('#a855f7'), 1.5))
        c3.setZValue(10.4)
        items.append(c3)

        l3 = QGraphicsTextItem('3:s')
        l3.setFont(QFont('Segoe UI', 6, QFont.Weight.Bold))
        l3.setDefaultTextColor(QColor('#a855f7'))
        l3.setPos(n3_x - 9.0, n3_y - 9.0)
        l3.setZValue(10.5)
        items.append(l3)

        # Failure Link: e.g. Node 6 ('sh') -> Node 1 ('h') (Dashed Amber Arc)
        f_line = QGraphicsLineItem(vx0 + 200.0, vy0 + 160.0, n1_x, n1_y + 12.0)
        pen_fail = QPen(QColor('#f59e0b'), 1.5, Qt.PenStyle.DashLine)
        f_line.setPen(pen_fail)
        f_line.setZValue(10.3)
        items.append(f_line)

        # Right Panel: Patterns & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        p_list = patterns or ['he', 'she', 'his', 'hers']
        p_str = ', '.join([f'"{p}"' for p in p_list])

        info_txt = QGraphicsTextItem(
            f'CẤU TRÚC OTOMAT AHO-CORASICK:\n'
            f'• Mẫu từ khóa: [{p_str}]\n'
            f'• Văn bản kiểm tra: "{sample_text}"\n'
            f'• Tổng số trạng thái (Trie nodes): {num_nodes}\n'
            f'• Số khớp từ khóa phát hiện: {num_matches}\n\n'
            f'ĐẶC TRƯNG ĐỘ PHỨC TẠP:\n'
            f'• Xây dựng cây tiền tố & failure links: O(m)\n'
            f'• Quét luồng văn bản độ dài n: O(n + z)\n'
            f'• z: Tổng số lần xuất hiện từ khóa khớp\n'
            f'• Đảm bảo thời gian tuyến tính tuyệt đối'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 8.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Aho-Corasick (1975): Kết hợp cây tiền tố (Trie) với các hàm chuyển tiếp thất bại (Failure function) '
            'và lan truyền từ khóa hậu tố (Suffix outputs). Cho phép tìm kiếm đồng thời k mẫu từ khóa trên luồng văn bản '
            'chỉ trong một lần duyệt tuyến tính duy nhất, ứng dụng cốt lõi trong Snort IDS, Grep và Bio-informatics.'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + vh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


    @staticmethod
    def create_fft_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "AUDIO_A440_CONCERT_PITCH",
        f0_hz: float = 125.0,
        sampling_rate_hz: float = 1000.0,
        peak_freq_hz: float = 125.0,
        resolution_hz: float = 15.625,
        nyquist_hz: float = 500.0,
        time_energy: float = 42.0,
        freq_energy: float = 42.0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        """Creates styled vector diagram of Cooley-Tukey Fast Fourier Transform (FFT) & Frequency Spectrum."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        x0 = cx - width / 2.0
        y0 = cy - height / 2.0

        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor('#070b14')))
        bg.setPen(QPen(pen_color or QColor('#38bdf8'), 1.8))
        bg.setZValue(10.0)
        items.append(bg)

        hdr = QGraphicsTextItem(f'TIN HỌC & XỬ LÝ TÍN HIỆU SỐ: THUẬT TOÁN COOLEY-TUKEY FFT O(N log N) – {preset_code}')
        hdr.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor('#f8fafc'))
        hdr.setPos(x0 + 14.0, y0 + 10.0)
        hdr.setZValue(10.5)
        items.append(hdr)

        vx0 = x0 + 16.0
        vy0 = y0 + 38.0
        vw = 360.0
        vh = 120.0

        # 1. Time Domain Waveform Box
        t_box = QGraphicsRectItem(vx0, vy0, vw, vh)
        t_box.setBrush(QBrush(QColor('#020617')))
        t_box.setPen(QPen(QColor('#1e293b'), 1.0))
        t_box.setZValue(10.2)
        items.append(t_box)

        t_lbl = QGraphicsTextItem('MIỀN THỜI GIAN x[n] (Dao động ký):')
        t_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        t_lbl.setDefaultTextColor(QColor('#38bdf8'))
        t_lbl.setPos(vx0 + 8.0, vy0 + 4.0)
        t_lbl.setZValue(10.4)
        items.append(t_lbl)

        # Draw sine wave
        t_mid = vy0 + vh * 0.58
        w_path = QPainterPath()
        w_path.moveTo(vx0 + 30.0, t_mid)
        n_pts = 40
        for i in range(n_pts + 1):
            px = vx0 + 30.0 + (i / float(n_pts)) * (vw - 50.0)
            py = t_mid - 32.0 * math.sin((i / float(n_pts)) * 4.0 * math.pi)
            w_path.lineTo(px, py)
        w_item = QGraphicsPathItem(w_path)
        w_item.setPen(QPen(QColor('#38bdf8'), 1.8))
        w_item.setZValue(10.3)
        items.append(w_item)

        # 2. Frequency Domain Spectrum Box
        fy0 = vy0 + vh + 10.0
        fh = 110.0
        f_box = QGraphicsRectItem(vx0, fy0, vw, fh)
        f_box.setBrush(QBrush(QColor('#020617')))
        f_box.setPen(QPen(QColor('#1e293b'), 1.0))
        f_box.setZValue(10.2)
        items.append(f_box)

        f_lbl = QGraphicsTextItem(f'MIỀN TẦN SỐ |X[k]| (Đỉnh: {peak_freq_hz:.1f} Hz):')
        f_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        f_lbl.setDefaultTextColor(QColor('#f59e0b'))
        f_lbl.setPos(vx0 + 8.0, fy0 + 4.0)
        f_lbl.setZValue(10.4)
        items.append(f_lbl)

        # Draw spectral bars
        f_base = fy0 + fh - 15.0
        bars = [
            (30.0, 10.0, '#f59e0b'),
            (60.0, 15.0, '#f59e0b'),
            (90.0, 75.0, '#ef4444'),   # Peak at f0
            (120.0, 12.0, '#f59e0b'),
            (150.0, 8.0, '#f59e0b'),
            (180.0, 35.0, '#f59e0b'),  # 2nd harmonic
            (210.0, 5.0, '#f59e0b'),
            (240.0, 18.0, '#f59e0b'),  # 3rd harmonic
        ]
        for bx_rel, b_h, col in bars:
            bar_item = QGraphicsRectItem(vx0 + bx_rel + 30.0, f_base - b_h, 12.0, b_h)
            bar_item.setBrush(QBrush(QColor(col)))
            bar_item.setPen(Qt.PenStyle.NoPen)
            bar_item.setZValue(10.4)
            items.append(bar_item)

        # 3. Right Panel: Butterfly Flow & Metrics
        rx0 = vx0 + vw + 14.0
        rw = width - (rx0 - x0) - 16.0
        rh = vh + fh + 10.0

        r_box = QGraphicsRectItem(rx0, vy0, rw, rh)
        r_box.setBrush(QBrush(QColor('#020617')))
        r_box.setPen(QPen(QColor('#1e293b'), 1.0))
        r_box.setZValue(10.2)
        items.append(r_box)

        # Butterfly miniature graph
        bf_lbl = QGraphicsTextItem('SƠ ĐỒ BƯỚM RADIX-2 DIT:')
        bf_lbl.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        bf_lbl.setDefaultTextColor(QColor('#10b981'))
        bf_lbl.setPos(rx0 + 10.0, vy0 + 6.0)
        bf_lbl.setZValue(10.4)
        items.append(bf_lbl)

        # Draw crossed butterfly lines
        for r_idx in range(4):
            y_a = vy0 + 30.0 + r_idx * 16.0
            y_b = y_a + 8.0
            l1 = QGraphicsLineItem(rx0 + 15.0, y_a, rx0 + 90.0, y_b)
            l1.setPen(QPen(QColor('#10b981'), 1.2))
            l1.setZValue(10.3)
            items.append(l1)

            l2 = QGraphicsLineItem(rx0 + 15.0, y_b, rx0 + 90.0, y_a)
            l2.setPen(QPen(QColor('#a855f7'), 1.2))
            l2.setZValue(10.3)
            items.append(l2)

        info_txt = QGraphicsTextItem(
            f'THÔNG SỐ PHÂN TÍCH PHỔ FFT:\n'
            f'• Tần số cơ bản f₀: {f0_hz:.1f} Hz\n'
            f'• Tốc độ lấy mẫu f_s: {sampling_rate_hz:.0f} Hz\n'
            f'• Tần số Nyquist: {nyquist_hz:.1f} Hz\n'
            f'• Độ phân giải phổ Δf: {resolution_hz:.2f} Hz/bin\n'
            f'• Đỉnh phổ cực đại: {peak_freq_hz:.1f} Hz\n\n'
            f'ĐỊNH LÝ BẢO TOÀN PARSEVAL:\n'
            f'• Năng lượng thời gian Σ|x[n]|² = {time_energy:.2f}\n'
            f'• Năng lượng tần số (1/N)Σ|X[k]|² = {freq_energy:.2f}\n'
            f'• Sai số bảo toàn: 0.00e+00\n'
            f'• Độ phức tạp: O(N log₂ N) = 384 phép tính'
        )
        info_txt.setFont(QFont('Segoe UI', 7))
        info_txt.setDefaultTextColor(QColor('#e2e8f0'))
        info_txt.setPos(rx0 + 10.0, vy0 + 95.0)
        info_txt.setZValue(10.4)
        items.append(info_txt)

        cap = QGraphicsTextItem(
            'Thuật toán Biến đổi Fourier Nhanh (Cooley-Tukey FFT, 1965): Giảm độ phức tạp từ O(N²) xuống O(N log N) '
            'bằng kỹ thuật chia để trị Decimation-in-Time và nhân thừa số quay W_N^k. '
            'Định lý Shannon-Nyquist khẳng định f_s ≥ 2 f_max để ngăn chặn hiện tượng chồng phổ (aliasing).'
        )
        cap.setFont(QFont('Segoe UI', 8))
        cap.setDefaultTextColor(QColor('#94a3b8'))
        cap.setPos(x0 + 16.0, vy0 + rh + 10.0)
        cap.setZValue(10.5)
        items.append(cap)

        return items


    @staticmethod
    def create_logic_circuits_diagram(
        center: tuple[float, float],
        width: float = 680.0,
        height: float = 380.0,
        preset_code: str = "HALF_ADDER_CIRCUIT",
        title_vi: str = "Mạch Nửa Cộng (Half Adder - XOR + AND)",
        circuit_name: str = "Binary Half Adder",
        boolean_formula: str = "Sum = A ⊕ B  |  C_out = A · B",
        inputs: tuple[int, ...] = (1, 0),
        outputs: tuple[int, ...] = (1, 0),
        input_names: list[str] = None,
        output_names: list[str] = None,
        description: str = "",
    ) -> list[QGraphicsItem]:
        """
        Creates a high-precision pedagogical vector diagram of Digital Logic Gates,
        Truth Tables, Boolean Algebra, and Binary Arithmetic Adders.
        """
        if input_names is None:
            input_names = ["A", "B"]
        if output_names is None:
            output_names = ["Sum", "C_out"]

        items: list[QGraphicsItem] = []
        x0 = center[0] - width / 2.0
        y0 = center[1] - height / 2.0

        # 1. Main Background Card
        bg = QGraphicsRectItem(x0, y0, width, height)
        bg.setBrush(QBrush(QColor("#0f172a")))
        bg.setPen(QPen(QColor("#334155"), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # 2. Header
        hdr = QGraphicsTextItem("🔌 ĐIỆN TỬ SỐ & CỔNG LOGIC (DIGITAL LOGIC & BOOLEAN ALGEBRA)")
        hdr.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hdr.setDefaultTextColor(QColor("#38bdf8"))
        hdr.setPos(x0 + 16.0, y0 + 12.0)
        hdr.setZValue(10.1)
        items.append(hdr)

        sub = QGraphicsTextItem(f"{title_vi} | Biểu thức Boole: {boolean_formula}")
        sub.setFont(QFont("Segoe UI", 8))
        sub.setDefaultTextColor(QColor("#94a3b8"))
        sub.setPos(x0 + 18.0, y0 + 34.0)
        sub.setZValue(10.1)
        items.append(sub)

        # 3. Left Panel: Logic Gate Schematic
        sch_rect = QGraphicsRectItem(x0 + 16.0, y0 + 58.0, 365.0, 260.0)
        sch_rect.setBrush(QBrush(QColor("#0a0f1e")))
        sch_rect.setPen(QPen(QColor("#1e293b"), 1.0))
        sch_rect.setZValue(10.1)
        items.append(sch_rect)

        thdr_l = QGraphicsTextItem("🔬 SƠ ĐỒ NGUYÊN LÝ CỔNG LOGIC")
        thdr_l.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr_l.setDefaultTextColor(QColor("#38bdf8"))
        thdr_l.setPos(x0 + 22.0, y0 + 64.0)
        thdr_l.setZValue(10.2)
        items.append(thdr_l)

        # Input switches & wires
        sw_x = x0 + 40.0
        start_y = y0 + 115.0
        y_gap = 50.0

        for i, val in enumerate(inputs):
            iy = start_y + i * y_gap
            val_col = QColor("#22c55e") if val == 1 else QColor("#64748b")

            # Switch rect
            sw_box = QGraphicsRectItem(sw_x - 12.0, iy - 12.0, 24.0, 24.0)
            sw_box.setBrush(QBrush(QColor("#1e293b")))
            sw_box.setPen(QPen(val_col, 1.5))
            sw_box.setZValue(10.3)
            items.append(sw_box)

            lbl_v = QGraphicsTextItem(str(val))
            lbl_v.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            lbl_v.setDefaultTextColor(val_col)
            lbl_v.setPos(sw_x - 6.0, iy - 12.0)
            lbl_v.setZValue(10.4)
            items.append(lbl_v)

            # In name
            name_str = input_names[i] if i < len(input_names) else f"In{i}"
            lbl_n = QGraphicsTextItem(name_str)
            lbl_n.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            lbl_n.setDefaultTextColor(QColor("#e2e8f0"))
            lbl_n.setPos(sw_x - 16.0, iy - 28.0)
            lbl_n.setZValue(10.4)
            items.append(lbl_n)

            # Wire to gate
            wire = QGraphicsLineItem(sw_x + 12.0, iy, x0 + 160.0, iy)
            wire.setPen(QPen(val_col, 2.0))
            wire.setZValue(10.2)
            items.append(wire)

        # Gate symbol box
        gate_box = QGraphicsRectItem(x0 + 160.0, y0 + 95.0, 95.0, 130.0)
        gate_box.setBrush(QBrush(QColor("#1e293b")))
        gate_box.setPen(QPen(QColor("#38bdf8"), 1.8))
        gate_box.setZValue(10.2)
        items.append(gate_box)

        lbl_gt = QGraphicsTextItem(circuit_name.split()[0])
        lbl_gt.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_gt.setDefaultTextColor(QColor("#38bdf8"))
        lbl_gt.setPos(x0 + 172.0, y0 + 145.0)
        lbl_gt.setZValue(10.3)
        items.append(lbl_gt)

        # Output wires & LEDs
        out_x = x0 + 330.0
        out_start_y = y0 + 125.0
        out_gap = 60.0

        for j, out_val in enumerate(outputs):
            oy = out_start_y + j * out_gap
            out_col = QColor("#22c55e") if out_val == 1 else QColor("#64748b")

            # Wire
            owire = QGraphicsLineItem(x0 + 255.0, oy, out_x - 12.0, oy)
            owire.setPen(QPen(out_col, 2.0))
            owire.setZValue(10.2)
            items.append(owire)

            # LED
            led = QGraphicsEllipseItem(out_x - 10.0, oy - 10.0, 20.0, 20.0)
            led.setBrush(QBrush(out_col))
            led.setPen(QPen(QColor("#ffffff"), 1.2 if out_val == 1 else 0.5))
            led.setZValue(10.3)
            items.append(led)

            out_n = output_names[j] if j < len(output_names) else f"Out{j}"
            lbl_on = QGraphicsTextItem(f"{out_n} = {out_val}")
            lbl_on.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            lbl_on.setDefaultTextColor(out_col)
            lbl_on.setPos(out_x - 20.0, oy + 12.0)
            lbl_on.setZValue(10.4)
            items.append(lbl_on)

        # 4. Right Panel: Truth Table
        tt_rect = QGraphicsRectItem(x0 + 395.0, y0 + 58.0, 269.0, 260.0)
        tt_rect.setBrush(QBrush(QColor("#0f172a")))
        tt_rect.setPen(QPen(QColor("#334155"), 1.0))
        tt_rect.setZValue(10.1)
        items.append(tt_rect)

        thdr_r = QGraphicsTextItem("📋 BẢNG CHÂN TRỊ (TRUTH TABLE)")
        thdr_r.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        thdr_r.setDefaultTextColor(QColor("#38bdf8"))
        thdr_r.setPos(x0 + 402.0, y0 + 64.0)
        thdr_r.setZValue(10.2)
        items.append(thdr_r)

        in_h = " | ".join(input_names)
        out_h = " | ".join(output_names)
        lbl_head = QGraphicsTextItem(f"{in_h}  ⟹  {out_h}")
        lbl_head.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_head.setDefaultTextColor(QColor("#94a3b8"))
        lbl_head.setPos(x0 + 405.0, y0 + 86.0)
        lbl_head.setZValue(10.2)
        items.append(lbl_head)

        line_sep = QGraphicsLineItem(x0 + 405.0, y0 + 104.0, x0 + 650.0, y0 + 104.0)
        line_sep.setPen(QPen(QColor("#334155"), 1.0))
        line_sep.setZValue(10.2)
        items.append(line_sep)

        # Summary text
        foot_info = QGraphicsTextItem(
            f"Trạng thái hiện tại: Input = {inputs} ⟹ Output = {outputs}\n"
            f"{description}"
        )
        foot_info.setFont(QFont("Segoe UI", 7))
        foot_info.setDefaultTextColor(QColor("#cbd5e1"))
        foot_info.setPos(x0 + 405.0, y0 + 115.0)
        foot_info.setZValue(10.2)
        items.append(foot_info)

        # 5. Pedagogical Summary Footer
        foot = QGraphicsTextItem(
            "Đại số Boole và cổng logic là nền tảng của toàn bộ kiến trúc máy tính số (Digital Computer Architecture). "
            "Mọi phép tính số học (ALU) và xử lý thông tin nhị phân trong CPU hiện đại đều cấu thành từ các liên kết cổng logic cơ bản."
        )
        foot.setFont(QFont("Segoe UI", 8))
        foot.setDefaultTextColor(QColor("#94a3b8"))
        foot.setPos(x0 + 16.0, y0 + height - 42.0)
        foot.setZValue(10.2)
        items.append(foot)

        return items


# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────

class StemToolDialog(QDialog):
    """
    Modal configuration dialog allowing teachers to enter parameters for
    STEM vector tools (Cartesian Oxy, function formula, chemical formula, etc.).
    """

    def __init__(self, title: str, tool_type: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.tool_type = tool_type
        self.resize(400, 260)
        self.setModal(True)

        layout = QVBoxLayout(self)
        self.form = QFormLayout()

        if tool_type == "math_oxy":
            self.inp_x_range = QLineEdit("8")
            self.inp_y_range = QLineEdit("5")
            self.form.addRow("Phạm vi trục X (-N đến +N):", self.inp_x_range)
            self.form.addRow("Phạm vi trục Y (-N đến +N):", self.inp_y_range)
        elif tool_type == "math_func":
            self.inp_expr = QLineEdit("x**2 - 2")
            self.form.addRow("Hàm số f(x):", self.inp_expr)
            hint = QLabel("Hỗ trợ: x**2, sin(x), cos(x), exp(x), sqrt(x), 2*x + 1")
            hint.setStyleSheet("color: #5f6368; font-size: 9pt;")
            self.form.addRow("", hint)
        elif tool_type == "chem_formula":
            self.inp_chem = QLineEdit("H2SO4 + 2NaOH -> Na2SO4 + 2H2O")
            self.form.addRow("Phương trình / Công thức:", self.inp_chem)
            hint = QLabel("Tự động định dạng chỉ số dưới: H2SO4 → H₂SO₄, Fe3+ → Fe³⁺")
            hint.setStyleSheet("color: #5f6368; font-size: 9pt;")
            self.form.addRow("", hint)
        elif tool_type == "physics_force":
            self.inp_label = QLineEdit("F⃗")
            self.inp_mag = QLineEdit("100")
            self.inp_angle = QLineEdit("45")
            self.form.addRow("Tên vectơ lực:", self.inp_label)
            self.form.addRow("Độ dài (px):", self.inp_mag)
            self.form.addRow("Góc hợp với phương ngang (°):", self.inp_angle)
        elif tool_type == "info_flowchart":
            self.combo_type = QComboBox()
            self.combo_type.addItems(["PROCESS (Thực thi)", "DECISION (Điều kiện)", "IO (Nhập / Xuất)", "TERMINATOR (Bắt đầu / Kết thúc)"])
            self.inp_text = QLineEdit("i = i + 1")
            self.form.addRow("Loại khối thuật toán:", self.combo_type)
            self.form.addRow("Nội dung khối:", self.inp_text)

        layout.addLayout(self.form)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Chèn vào bài giảng")
        btn_ok.setStyleSheet("background: #1a73e8; color: white; font-weight: bold; padding: 6px 14px;")
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)
