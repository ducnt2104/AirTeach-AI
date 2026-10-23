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


# ─── Vector Tools for Chemistry ───────────────────────────────────────────────

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


# ─── Vector Tools for Biology ─────────────────────────────────────────────────

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
