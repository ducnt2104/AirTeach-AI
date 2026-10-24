"""
AirTeach AI - Interactive Geometric Triangle Theorem Solver & Euler Line HUD
=============================================================================
Provides a live, dynamic 3-vertex triangle geometry item:
  - Exact metric calculations: side lengths (a, b, c), perimeter, Heron's area (S).
  - Inradius (r = S/p) and circumradius (R = abc / 4S).
  - Triangle taxonomy: Đều (Equilateral), Cân (Isosceles), Vuông (Right), Thường (Scalene).
  - Special center points: Centroid G, Circumcenter O, Orthocenter H, Incenter I.
  - Famous Euler Line (Đường thẳng Euler) visualization: collinear line passing through O, G, H with HG = 2*GO.
  - Circumcircle and Incircle toggles.
  - Real-time dragging of vertices A, B, C with mouse and touchless air gestures.
  - Vector stamping to LessonBoard canvas.
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


class InteractiveTriangleGeometryItem(QGraphicsObject):
    """
    Interactive plane geometry triangle with 3 movable vertices (A, B, C).
    Computes real-time metric properties, centers (G, O, H, I), and the Euler Line.
    """

    geometry_changed = Signal(float, float)  # (area, perimeter)
    vertex_moved = Signal(str, float, float)  # (vertex_name: "A"|"B"|"C", x, y)

    def __init__(
        self,
        vert_a: tuple[float, float] = (750.0, 320.0),
        vert_b: tuple[float, float] = (550.0, 620.0),
        vert_c: tuple[float, float] = (1050.0, 620.0),
        labels: tuple[str, str, str] = ("A", "B", "C"),
        color: QColor | str = "#3b82f6",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._a = QPointF(float(vert_a[0]), float(vert_a[1]))
        self._b = QPointF(float(vert_b[0]), float(vert_b[1]))
        self._c = QPointF(float(vert_c[0]), float(vert_c[1]))
        self._labels = (labels[0], labels[1], labels[2])
        self._color = QColor(color) if isinstance(color, str) else color

        # Toggles for geometric overlays
        self._show_euler_line: bool = True
        self._show_centroid: bool = True
        self._show_circumcircle: bool = False
        self._show_incircle: bool = False
        self._show_hud_card: bool = True

        self._handle_radius: float = 12.0
        self._active_handle: Optional[str] = None

        self.setZValue(14.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    # ── Vertex Coordinates & Geometry ──────────────────────────────────────────

    def vertex_pos(self, name: str) -> QPointF:
        name = name.upper()
        if name == "A":
            return self._a
        elif name == "B":
            return self._b
        elif name == "C":
            return self._c
        return self._a

    def set_vertex_pos(self, name: str, x: float, y: float) -> None:
        """Set position of vertex A, B, or C."""
        name = name.upper()
        self.prepareGeometryChange()
        pt = QPointF(float(x), float(y))
        if name == "A":
            self._a = pt
        elif name == "B":
            self._b = pt
        elif name == "C":
            self._c = pt

        self.update()
        s, p = self.area(), self.perimeter()
        self.geometry_changed.emit(s, p)
        self.vertex_moved.emit(name, pt.x(), pt.y())

    def set_vertices(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> None:
        """Set all three vertices simultaneously."""
        self.prepareGeometryChange()
        self._a = QPointF(float(a[0]), float(a[1]))
        self._b = QPointF(float(b[0]), float(b[1]))
        self._c = QPointF(float(c[0]), float(c[1]))
        self.update()
        self.geometry_changed.emit(self.area(), self.perimeter())

    # ── Metric Properties ──────────────────────────────────────────────────────

    def side_lengths(self) -> tuple[float, float, float]:
        """Return side lengths (a=BC, b=CA, c=AB)."""
        a = math.hypot(self._c.x() - self._b.x(), self._c.y() - self._b.y())
        b = math.hypot(self._a.x() - self._c.x(), self._a.y() - self._c.y())
        c = math.hypot(self._b.x() - self._a.x(), self._b.y() - self._a.y())
        return (a, b, c)

    def perimeter(self) -> float:
        """Return perimeter 2p = a + b + c."""
        a, b, c = self.side_lengths()
        return a + b + c

    def area(self) -> float:
        """Return non-negative area S using cross-product / Heron's formula."""
        ax, ay = self._a.x(), self._a.y()
        bx, by = self._b.x(), self._b.y()
        cx, cy = self._c.x(), self._c.y()
        cross = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        return 0.5 * abs(cross)

    def inradius(self) -> float:
        """Return inradius r = S / p."""
        s = self.area()
        p = self.perimeter() / 2.0
        return s / p if p > 1e-4 else 0.0

    def circumradius(self) -> float:
        """Return circumradius R = abc / 4S."""
        a, b, c = self.side_lengths()
        s = self.area()
        return (a * b * c) / (4.0 * s) if s > 1e-4 else 0.0

    # ── Special Triangle Centers ───────────────────────────────────────────────

    def centroid(self) -> QPointF:
        """Centroid G = (A + B + C) / 3."""
        gx = (self._a.x() + self._b.x() + self._c.x()) / 3.0
        gy = (self._a.y() + self._b.y() + self._c.y()) / 3.0
        return QPointF(gx, gy)

    def circumcenter(self) -> QPointF:
        """Circumcenter O (intersection of perpendicular bisectors)."""
        ax, ay = self._a.x(), self._a.y()
        bx, by = self._b.x(), self._b.y()
        cx, cy = self._c.x(), self._c.y()

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-4:
            return self.centroid()

        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy

        ox = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        oy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        return QPointF(ox, oy)

    def orthocenter(self) -> QPointF:
        """Orthocenter H = 3*G - 2*O (via Euler's theorem)."""
        g = self.centroid()
        o = self.circumcenter()
        hx = 3.0 * g.x() - 2.0 * o.x()
        hy = 3.0 * g.y() - 2.0 * o.y()
        return QPointF(hx, hy)

    def incenter(self) -> QPointF:
        """Incenter I = (a*A + b*B + c*C) / (a + b + c)."""
        a, b, c = self.side_lengths()
        peri = a + b + c
        if peri < 1e-4:
            return self.centroid()
        ix = (a * self._a.x() + b * self._b.x() + c * self._c.x()) / peri
        iy = (a * self._a.y() + b * self._b.y() + c * self._c.y()) / peri
        return QPointF(ix, iy)

    # ── Triangle Classification ────────────────────────────────────────────────

    def classification(self) -> str:
        """Classify triangle: 'equilateral', 'right', 'isosceles', 'scalene'."""
        a, b, c = sorted(self.side_lengths())
        if a < 1e-4 or b < 1e-4 or c < 1e-4:
            return "degenerate"

        tol = 3.0  # pixel tolerance
        is_equilateral = (abs(a - b) <= tol and abs(b - c) <= tol)
        if is_equilateral:
            return "equilateral"

        is_right = abs(a * a + b * b - c * c) <= (2.0 * c * tol)
        is_isosceles = (abs(a - b) <= tol or abs(b - c) <= tol or abs(a - c) <= tol)

        if is_right and is_isosceles:
            return "right_isosceles"
        elif is_right:
            return "right"
        elif is_isosceles:
            return "isosceles"
        return "scalene"

    def classification_vietnamese(self) -> str:
        t = self.classification()
        mapping = {
            "equilateral": "Tam giác đều",
            "right_isosceles": "Tam giác vuông cân",
            "right": "Tam giác vuông",
            "isosceles": "Tam giác cân",
            "scalene": "Tam giác thường",
            "degenerate": "Tam giác suy biến",
        }
        return mapping.get(t, "Tam giác thường")

    # ── Overlays Toggles ───────────────────────────────────────────────────────

    def toggle_euler_line(self) -> bool:
        self.prepareGeometryChange()
        self._show_euler_line = not self._show_euler_line
        self.update()
        return self._show_euler_line

    def toggle_hud_card(self) -> bool:
        self.prepareGeometryChange()
        self._show_hud_card = not self._show_hud_card
        self.update()
        return self._show_hud_card

    def toggle_circumcircle(self) -> bool:
        self.prepareGeometryChange()
        self._show_circumcircle = not self._show_circumcircle
        self.update()
        return self._show_circumcircle

    def toggle_incircle(self) -> bool:
        self.prepareGeometryChange()
        self._show_incircle = not self._show_incircle
        self.update()
        return self._show_incircle

    # ── Hit-testing & Interaction ──────────────────────────────────────────────

    def get_handle_at(self, sx: float, sy: float) -> Optional[str]:
        for name, pt in (("A", self._a), ("B", self._b), ("C", self._c)):
            if math.hypot(sx - pt.x(), sy - pt.y()) <= self._handle_radius * 1.5:
                return name
        return None

    def air_drag_vertex(self, name: str, sx: float, sy: float) -> bool:
        name = name.upper()
        if name in ("A", "B", "C"):
            self.set_vertex_pos(name, sx, sy)
            return True
        return False

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        xs = [self._a.x(), self._b.x(), self._c.x()]
        ys = [self._a.y(), self._b.y(), self._c.y()]

        if self._show_circumcircle:
            o = self.circumcenter()
            r = self.circumradius()
            xs.extend([o.x() - r, o.x() + r])
            ys.extend([o.y() - r, o.y() + r])

        margin = 120.0
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Triangle Body & Edges
        tri_path = QPainterPath()
        tri_path.moveTo(self._a)
        tri_path.lineTo(self._b)
        tri_path.lineTo(self._c)
        tri_path.closeSubpath()

        # Translucent body fill
        fill_color = QColor(self._color.red(), self._color.green(), self._color.blue(), 25)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(self._color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(tri_path)

        # 2. Incircle (if enabled)
        if self._show_incircle:
            in_center = self.incenter()
            in_r = self.inradius()
            if in_r > 1.0:
                painter.setBrush(QBrush(QColor(16, 185, 129, 20)))
                painter.setPen(QPen(QColor("#10b981"), 1.2, Qt.PenStyle.DashDotLine))
                painter.drawEllipse(in_center, in_r, in_r)

        # 3. Circumcircle (if enabled)
        if self._show_circumcircle:
            c_center = self.circumcenter()
            c_r = self.circumradius()
            if c_r > 1.0 and c_r < 1000.0:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#06b6d4"), 1.2, Qt.PenStyle.DashLine))
                painter.drawEllipse(c_center, c_r, c_r)

        # 4. Medians & Centroid G
        g = self.centroid()
        if self._show_centroid:
            # Medians (faint dashed lines from vertices to opposite midpoints)
            mid_bc = QPointF((self._b.x() + self._c.x()) / 2.0, (self._b.y() + self._c.y()) / 2.0)
            painter.setPen(QPen(QColor(148, 163, 184, 120), 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(self._a, mid_bc)

            # Centroid G dot
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(g, 4.5, 4.5)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(int(g.x() + 8.0), int(g.y() - 4.0), "G")

        # 5. Euler Line (Collinear through O, G, H)
        if self._show_euler_line:
            o = self.circumcenter()
            h = self.orthocenter()

            # Extend line slightly past O and H
            dx = h.x() - o.x()
            dy = h.y() - o.y()
            len_line = math.hypot(dx, dy)
            if len_line > 2.0:
                scale = 1.3
                p_start = QPointF(g.x() - dx * scale, g.y() - dy * scale)
                p_end = QPointF(g.x() + dx * scale, g.y() + dy * scale)

                # Glowing Euler Line
                painter.setPen(QPen(QColor(244, 63, 94, 70), 5.0, Qt.PenStyle.SolidLine))
                painter.drawLine(p_start, p_end)
                painter.setPen(QPen(QColor("#f43f5e"), 1.8, Qt.PenStyle.DashLine))
                painter.drawLine(p_start, p_end)

                # Point O (Circumcenter)
                painter.setBrush(QBrush(QColor("#06b6d4")))
                painter.setPen(QPen(QColor("#ffffff"), 1.0))
                painter.drawEllipse(o, 4.0, 4.0)
                painter.setPen(QPen(QColor("#06b6d4")))
                painter.drawText(int(o.x() + 6.0), int(o.y() - 4.0), "O")

                # Point H (Orthocenter)
                painter.setBrush(QBrush(QColor("#ef4444")))
                painter.setPen(QPen(QColor("#ffffff"), 1.0))
                painter.drawEllipse(h, 4.0, 4.0)
                painter.setPen(QPen(QColor("#ef4444")))
                painter.drawText(int(h.x() + 6.0), int(h.y() - 4.0), "H")

        # 6. Vertex Handles (A, B, C)
        for name, pt in (("A", self._a), ("B", self._b), ("C", self._c)):
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(self._color, 2.2))
            painter.drawEllipse(pt, self._handle_radius, self._handle_radius)

            lbl = self._labels[{"A": 0, "B": 1, "C": 2}[name]]
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f8fafc")))
            # Offset letter away from centroid
            lx = pt.x() + (12.0 if pt.x() >= g.x() else -24.0)
            ly = pt.y() + (18.0 if pt.y() >= g.y() else -10.0)
            painter.drawText(int(lx), int(ly), lbl)

        # 7. Geometric Theorem HUD Card
        if self._show_hud_card:
            card_w, card_h = 170.0, 78.0
            # Anchor card above highest vertex
            min_y = min(self._a.y(), self._b.y(), self._c.y())
            cx = (self._a.x() + self._b.x() + self._c.x()) / 3.0
            card_x = max(20.0, min(1920.0 - card_w - 20.0, cx - card_w / 2.0))
            card_y = max(20.0, min_y - card_h - 22.0)
            card_rect = QRectF(card_x, card_y, card_w, card_h)

            card_bg = QColor(15, 23, 42, 230)
            painter.setBrush(QBrush(card_bg))
            painter.setPen(QPen(self._color, 1.4))
            painter.drawRoundedRect(card_rect, 6.0, 6.0)

            # Title: Classification
            cls_txt = self.classification_vietnamese()
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#38bdf8")))
            t_rect = QRectF(card_x + 8.0, card_y + 4.0, card_w - 16.0, 16.0)
            painter.drawText(t_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, cls_txt)

            # Metrics
            a, b, c = self.side_lengths()
            s = self.area()
            p = self.perimeter()
            r = self.inradius()
            R = self.circumradius()

            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QPen(QColor("#e2e8f0")))
            painter.drawText(int(card_x + 8.0), int(card_y + 36.0), f"a={a:.0f} b={b:.0f} c={c:.0f}")
            painter.drawText(int(card_x + 8.0), int(card_y + 52.0), f"Chu vi P: {p:.1f}")
            painter.drawText(int(card_x + 8.0), int(card_y + 68.0), f"Diện tích S: {s:.1f}")

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        handle = self.get_handle_at(pos.x(), pos.y())
        if handle:
            self._active_handle = handle
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._active_handle:
            pos = event.pos()
            self.set_vertex_pos(self._active_handle, pos.x(), pos.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._active_handle:
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Permanently stamp triangle ABC onto LessonBoard as vector items."""
        from src.ui.stem_tools import MathTools
        is_right = ("right" in self.classification())
        items = MathTools.create_geometric_triangle(
            v0=(self._a.x(), self._a.y()),
            v1=(self._b.x(), self._b.y()),
            v2=(self._c.x(), self._c.y()),
            labels=self._labels,
            show_right_angle=is_right,
            right_angle_vertex_idx=0 if is_right else 0,
            pen_color=self._color,
        )
        for itm in items:
            board.addItem(itm)
        return items
