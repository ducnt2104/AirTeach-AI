"""
AirTeach AI - Interactive Dynamic Angle Measure & Plane Geometry Vertex HUD
===========================================================================
Provides an interactive 3-vertex angle measurement tool (A, B, C):
  - Dynamic interior angle computation: angle(ABC) = arccos((BA . BC) / (|BA| * |BC|)).
  - Real-time right-angle geometric snapping (90° square marker when theta ~= 90°).
  - Classification: Nhọn (Acute, <90°), Vuông (Right, =90°), Tù (Obtuse, >90°), Bẹt (Straight, 180°).
  - Floating translucent degree badge at angle bisector.
  - Draggable control handles for vertices A, B (apex), and C.
  - Air gesture manipulation and vector stamping to LessonBoard canvas.
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


class InteractiveAngleMeasureItem(QGraphicsObject):
    """
    Interactive plane geometry angle measure item with 3 movable vertices (A, B, C).
    B is the angle apex (vertex). Dynamically calculates and displays angle ABC.
    """

    angle_changed = Signal(float)          # (angle_degrees: 0.0 - 180.0)
    vertex_moved = Signal(str, float, float)  # (vertex_name: "A"|"B"|"C", x, y)

    def __init__(
        self,
        apex_b: tuple[float, float] = (800.0, 500.0),
        arm_a: tuple[float, float] = (620.0, 360.0),
        arm_c: tuple[float, float] = (1020.0, 500.0),
        labels: tuple[str, str, str] = ("A", "B", "C"),
        arc_radius: float = 48.0,
        snap_right_angle: bool = True,
        color: QColor | str = "#3b82f6",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._a = QPointF(float(arm_a[0]), float(arm_a[1]))
        self._b = QPointF(float(apex_b[0]), float(apex_b[1]))
        self._c = QPointF(float(arm_c[0]), float(arm_c[1]))
        self._labels = (labels[0], labels[1], labels[2])
        self._arc_radius = max(20.0, float(arc_radius))
        self._snap_right_angle = bool(snap_right_angle)
        self._color = QColor(color) if isinstance(color, str) else color

        self._handle_radius: float = 14.0
        self._active_handle: Optional[str] = None
        self._drag_offset: QPointF = QPointF()

        self.setZValue(14.0)  # Floating above standard strokes
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
        return self._b

    def set_vertex_pos(self, name: str, x: float, y: float) -> None:
        """Set position of vertex A, B (apex), or C in scene coordinates."""
        name = name.upper()
        self.prepareGeometryChange()
        pt = QPointF(float(x), float(y))
        if name == "A":
            self._a = pt
        elif name == "B":
            # Moving apex B moves arms A and C relatively
            dx = pt.x() - self._b.x()
            dy = pt.y() - self._b.y()
            self._b = pt
            self._a = QPointF(self._a.x() + dx, self._a.y() + dy)
            self._c = QPointF(self._c.x() + dx, self._c.y() + dy)
        elif name == "C":
            self._c = pt

        self.update()
        deg = self.angle_degrees()
        self.angle_changed.emit(deg)
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
        self.angle_changed.emit(self.angle_degrees())

    # ── Angle Mathematical Calculations ───────────────────────────────────────

    def angle_degrees(self) -> float:
        """
        Compute interior angle ABC at vertex B in degrees [0.0, 180.0].
        Auto-snaps to exactly 90.0° if within right-angle snap tolerance (1.8°).
        """
        ux = self._a.x() - self._b.x()
        uy = self._a.y() - self._b.y()
        vx = self._c.x() - self._b.x()
        vy = self._c.y() - self._b.y()

        len_u = math.hypot(ux, uy)
        len_v = math.hypot(vx, vy)
        if len_u < 1e-4 or len_v < 1e-4:
            return 0.0

        cos_theta = (ux * vx + uy * vy) / (len_u * len_v)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        rad = math.acos(cos_theta)
        deg = math.degrees(rad)

        if self._snap_right_angle and abs(deg - 90.0) <= 1.8:
            return 90.0
        return deg

    def is_right_angle(self) -> bool:
        """Return True if angle is a right angle (90°)."""
        return abs(self.angle_degrees() - 90.0) < 1e-3

    def angle_type(self) -> str:
        """Classify angle: 'right', 'acute', 'obtuse', 'straight'."""
        deg = self.angle_degrees()
        if abs(deg - 90.0) < 1e-3:
            return "right"
        elif deg < 90.0:
            return "acute"
        elif abs(deg - 180.0) < 1e-3:
            return "straight"
        else:
            return "obtuse"

    def angle_type_vietnamese(self) -> str:
        """Return Vietnamese classification label."""
        t = self.angle_type()
        if t == "right":
            return "Góc vuông"
        elif t == "acute":
            return "Góc nhọn"
        elif t == "straight":
            return "Góc bẹt"
        else:
            return "Góc tù"

    # ── Control Handles & Hit-testing ──────────────────────────────────────────

    def get_handle_at(self, sx: float, sy: float) -> Optional[str]:
        """Check if (sx, sy) is within hit radius of handle A, B, or C."""
        for name, pt in (("A", self._a), ("B", self._b), ("C", self._c)):
            if math.hypot(sx - pt.x(), sy - pt.y()) <= self._handle_radius * 1.5:
                return name
        return None

    def air_drag_vertex(self, name: str, sx: float, sy: float) -> bool:
        """Touchless air gesture support for moving vertices."""
        name = name.upper()
        if name in ("A", "B", "C"):
            self.set_vertex_pos(name, sx, sy)
            return True
        return False

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        xs = [self._a.x(), self._b.x(), self._c.x()]
        ys = [self._a.y(), self._b.y(), self._c.y()]
        margin = self._arc_radius + 40.0
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

        bx, by = self._b.x(), self._b.y()
        ax, ay = self._a.x(), self._a.y()
        cx, cy = self._c.x(), self._c.y()

        deg = self.angle_degrees()
        is_right = self.is_right_angle()

        # 1. Draw Ray BA and Ray BC
        ray_pen = QPen(self._color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(ray_pen)
        painter.drawLine(self._b, self._a)
        painter.drawLine(self._b, self._c)

        # 2. Angle Arc / Right-Angle Square Symbol
        ang_a = math.atan2(ay - by, ax - bx)
        ang_c = math.atan2(cy - by, cx - bx)

        # Normalize sweep from ray BA to ray BC
        diff = (ang_c - ang_a) % (2.0 * math.pi)
        if diff > math.pi:
            # Sweep the other way to always span interior angle <= 180°
            start_ang = ang_c
            sweep_rad = (2.0 * math.pi - diff)
            sweep_dir = -1.0
        else:
            start_ang = ang_a
            sweep_rad = diff
            sweep_dir = 1.0

        if is_right:
            # Draw standard geometry right-angle square marker (L)
            sq_size = min(self._arc_radius * 0.65, 24.0)
            u_unit = QPointF(math.cos(ang_a), math.sin(ang_a))
            v_unit = QPointF(math.cos(ang_c), math.sin(ang_c))

            p1 = QPointF(bx + u_unit.x() * sq_size, by + u_unit.y() * sq_size)
            p3 = QPointF(bx + v_unit.x() * sq_size, by + v_unit.y() * sq_size)
            p2 = QPointF(p1.x() + v_unit.x() * sq_size, p1.y() + v_unit.y() * sq_size)

            sq_path = QPainterPath()
            sq_path.moveTo(p1)
            sq_path.lineTo(p2)
            sq_path.lineTo(p3)

            emerald = QColor("#10b981")
            painter.setPen(QPen(emerald, 2.0))
            painter.setBrush(QBrush(QColor(16, 185, 129, 40)))
            painter.drawPath(sq_path)
        else:
            # Draw smooth circular sector arc
            r = self._arc_radius
            rect = QRectF(bx - r, by - r, 2 * r, 2 * r)

            arc_path = QPainterPath()
            deg_start = -math.degrees(start_ang)
            deg_sweep = -sweep_dir * math.degrees(sweep_rad)

            arc_path.arcMoveTo(rect, deg_start)
            arc_path.arcTo(rect, deg_start, deg_sweep)

            # Arc stroke and light fill
            painter.setPen(QPen(self._color, 1.8, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 30)))
            painter.drawPath(arc_path)

        # 3. Floating Degree Badge at Angle Bisector
        mid_rad = start_ang + (sweep_dir * sweep_rad / 2.0)
        badge_dist = self._arc_radius + 24.0
        bdg_cx = bx + badge_dist * math.cos(mid_rad)
        bdg_cy = by + badge_dist * math.sin(mid_rad)

        badge_w, badge_h = 76.0, 24.0
        badge_rect = QRectF(bdg_cx - badge_w / 2.0, bdg_cy - badge_h / 2.0, badge_w, badge_h)

        badge_bg = QColor(15, 23, 42, 230)
        border_c = QColor("#10b981") if is_right else self._color
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(border_c, 1.2))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        # Badge text
        deg_str = f"{deg:.0f}°" if is_right or deg.is_integer() else f"{deg:.1f}°"
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, deg_str)

        # 4. Draw Vertex Handles (A, B, C)
        for name, pt in (("A", self._a), ("B", self._b), ("C", self._c)):
            is_apex = (name == "B")
            h_rad = 10.0 if is_apex else 8.0

            # Ring halo
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(border_c if is_apex else self._color, 2.0))
            painter.drawEllipse(pt, h_rad, h_rad)

            # Vertex label letter (A, B, C)
            lbl = self._labels[{"A": 0, "B": 1, "C": 2}[name]]
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f8fafc")))
            # Offset label slightly away from apex
            lx = pt.x() + (12.0 if pt.x() >= bx else -22.0)
            ly = pt.y() + (16.0 if pt.y() >= by else -10.0)
            painter.drawText(int(lx), int(ly), lbl)

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
        """
        Permanently render and stamp the current geometric angle onto LessonBoard.
        Returns the list of added scene items.
        """
        from src.ui.stem_tools import MathTools
        deg = self.angle_degrees()
        label_text = f"{self._labels[1]} = {deg:.1f}°" if not self.is_right_angle() else f"{self._labels[1]} = 90°"

        items = MathTools.create_angle_arc(
            vertex=(self._b.x(), self._b.y()),
            p1=(self._a.x(), self._a.y()),
            p2=(self._c.x(), self._c.y()),
            radius=self._arc_radius,
            label=label_text,
            color=self._color,
        )
        for itm in items:
            board.addItem(itm)
        return items
