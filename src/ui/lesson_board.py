"""
AirTeach AI - Lesson Board (QGraphicsScene Canvas)
====================================================
Renders:
  - Document page images as background QGraphicsPixmapItem
  - Live ink strokes as QPainterPath (cubic Bézier, variable width)
  - Recognized text labels and LaTeX PNG overlays
  - Geometric snapped shapes (line, arrow, circle, rectangle, triangle)

Thread safety: all mutating calls must be made from the UI thread.
Use Qt Signals from worker threads to enqueue updates.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
    Signal,
    QObject,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
)

from src.core.stroke_beautifier import BezierSegment, SnappedShape as SnapResult, ShapeType

logger = logging.getLogger(__name__)

# ── Color palette ─────────────────────────────────────────────────────────────
_SNAP_COLOR: dict[ShapeType, QColor] = {
    ShapeType.LINE:      QColor("#1a73e8"),
    ShapeType.ARROW:     QColor("#1a73e8"),
    ShapeType.CIRCLE:    QColor("#34a853"),
    ShapeType.ELLIPSE:   QColor("#34a853"),
    ShapeType.RECTANGLE: QColor("#fbbc04"),
    ShapeType.TRIANGLE:  QColor("#ea4335"),
    ShapeType.FREEFORM:  QColor("#202124"),
}

BOARD_WIDTH  = 1920
BOARD_HEIGHT = 1080


# ── Data class for a completed ink stroke ─────────────────────────────────────

@dataclass
class InkStroke:
    """A fully processed ink stroke ready for rendering."""
    stroke_id: str
    segments: list[BezierSegment]           # Cubic Bézier segments (C¹)
    widths: list[float]                     # per-segment line width
    snap: Optional[SnapResult] = None       # None = render as freehand Bézier
    color: QColor = field(default_factory=lambda: QColor("#202124"))
    timestamps_ms: list[float] = field(default_factory=list)

    # Graphics item references (populated after addToScene)
    _path_item: Optional[QGraphicsPathItem] = field(default=None, repr=False, compare=False)
    _shape_item: Optional[QGraphicsItem]    = field(default=None, repr=False, compare=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _numpy_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Convert a BGR uint8 numpy array to QPixmap."""
    import cv2  # noqa: PLC0415
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


def _numpy_rgba_to_qpixmap(img_rgba: np.ndarray) -> QPixmap:
    """Convert an RGBA uint8 numpy array to QPixmap with transparency."""
    h, w, ch = img_rgba.shape
    fmt = QImage.Format.Format_RGBA8888
    qi = QImage(img_rgba.data, w, h, w * ch, fmt).copy()
    return QPixmap.fromImage(qi)


def _build_bezier_path(segments: list[BezierSegment]) -> QPainterPath:
    """Assemble a QPainterPath from a list of cubic Bézier segments.
    BezierSegment fields: p0 (start), p1 (control 1), p2 (control 2), p3 (end).
    """
    path = QPainterPath()
    if not segments:
        return path
    p0 = segments[0].p0
    path.moveTo(p0[0], p0[1])
    for seg in segments:
        path.cubicTo(
            seg.p1[0], seg.p1[1],   # control point 1
            seg.p2[0], seg.p2[1],   # control point 2
            seg.p3[0], seg.p3[1],   # end point
        )
    return path


# ── Main Scene ────────────────────────────────────────────────────────────────

class LessonBoard(QGraphicsScene):
    """
    QGraphicsScene that manages one lesson page:
      - Background document page (pixmap)
      - Ink strokes (QPainterPath items)
      - Snapped geometric shapes
      - OCR text / LaTeX overlays
      - Live drawing preview (current stroke being drawn)

    Signals:
        stroke_committed(str): emitted when a finalized stroke is added (stroke_id).
        selection_rect_changed(QRectF): emitted when SELECT gesture defines a region.
    """

    stroke_committed     = Signal(str)
    selection_rect_changed = Signal(QRectF)

    # Z-layer constants
    Z_BACKGROUND  = 0
    Z_INK         = 10
    Z_SHAPE       = 11
    Z_OCR_OVERLAY = 20
    Z_PREVIEW     = 30
    Z_LASER       = 40

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setSceneRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        self.setBackgroundBrush(QBrush(QColor("#f8f9fa")))

        # State
        self._strokes: dict[str, InkStroke] = {}
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._preview_item: Optional[QGraphicsPathItem] = None
        self._laser_item: Optional[QGraphicsEllipseItem] = None
        self._select_rect_item: Optional[QGraphicsRectItem] = None
        self._live_points: list[tuple[float, float]] = []

        # Laser pointer dot
        self._init_laser()

    # ── Laser pointer ─────────────────────────────────────────────────────────

    def _init_laser(self) -> None:
        r = 12.0
        self._laser_item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        pen = QPen(QColor("#ff0000"), 2)
        self._laser_item.setPen(pen)
        self._laser_item.setBrush(QBrush(QColor(255, 0, 0, 80)))
        self._laser_item.setZValue(self.Z_LASER)
        self._laser_item.setVisible(False)
        self.addItem(self._laser_item)

    def move_laser(self, x: float, y: float, visible: bool = True) -> None:
        """Update laser pointer position (call from UI thread via signal)."""
        self._laser_item.setPos(x, y)
        self._laser_item.setVisible(visible)

    # ── Background page ───────────────────────────────────────────────────────

    def set_background_page(self, img_bgr: np.ndarray) -> None:
        """
        Set or replace the background document page image.
        Scales to fill the board while preserving aspect ratio.
        """
        if self._bg_item is not None:
            self.removeItem(self._bg_item)
            self._bg_item = None

        pixmap = _numpy_bgr_to_qpixmap(img_bgr)
        scaled = pixmap.scaled(
            BOARD_WIDTH, BOARD_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._bg_item = QGraphicsPixmapItem(scaled)
        # Center on board
        ox = (BOARD_WIDTH  - scaled.width())  / 2
        oy = (BOARD_HEIGHT - scaled.height()) / 2
        self._bg_item.setPos(ox, oy)
        self._bg_item.setZValue(self.Z_BACKGROUND)
        self.addItem(self._bg_item)

    def clear_background(self) -> None:
        if self._bg_item is not None:
            self.removeItem(self._bg_item)
            self._bg_item = None

    # ── Live drawing preview ──────────────────────────────────────────────────

    def begin_stroke_preview(self, x: float, y: float, color: QColor) -> None:
        """Start a new in-progress stroke preview (first point)."""
        self._live_points = [(x, y)]
        if self._preview_item is not None:
            self.removeItem(self._preview_item)
        path = QPainterPath()
        path.moveTo(x, y)
        pen = QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        self._preview_item = QGraphicsPathItem(path)
        self._preview_item.setPen(pen)
        self._preview_item.setZValue(self.Z_PREVIEW)
        self.addItem(self._preview_item)

    def update_stroke_preview(self, x: float, y: float) -> None:
        """Extend the live preview stroke with a new filtered point."""
        if self._preview_item is None or not self._live_points:
            return
        self._live_points.append((x, y))
        path = QPainterPath()
        path.moveTo(*self._live_points[0])
        for px, py in self._live_points[1:]:
            path.lineTo(px, py)
        self._preview_item.setPath(path)

    def end_stroke_preview(self) -> None:
        """Remove the preview stroke (called after commit_stroke or cancel)."""
        if self._preview_item is not None:
            self.removeItem(self._preview_item)
            self._preview_item = None
        self._live_points.clear()

    # ── Committed strokes ─────────────────────────────────────────────────────

    def commit_stroke(self, stroke: InkStroke) -> None:
        """
        Add a fully processed InkStroke to the scene.
        If stroke.snap is set and not FREEHAND, draw the geometric shape.
        Otherwise render as a variable-width Bézier path.
        """
        self.end_stroke_preview()

        if stroke.snap is not None and stroke.snap.shape_type != ShapeType.FREEFORM:
            item = self._render_snapped_shape(stroke)
            stroke._shape_item = item
        else:
            item = self._render_bezier_stroke(stroke)
            stroke._path_item = item

        self._strokes[stroke.stroke_id] = stroke
        self.stroke_committed.emit(stroke.stroke_id)

    def _render_bezier_stroke(self, stroke: InkStroke) -> QGraphicsPathItem:
        """Render a freehand stroke as a single QPainterPath."""
        path = _build_bezier_path(stroke.segments)
        avg_width = float(np.mean(stroke.widths)) if stroke.widths else 2.5
        pen = QPen(stroke.color, avg_width, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self.Z_INK)
        item.setData(0, stroke.stroke_id)
        self.addItem(item)
        return item

    def _render_snapped_shape(self, stroke: InkStroke) -> QGraphicsItem:
        """Render a snapped geometric shape."""
        snap = stroke.snap
        color = _SNAP_COLOR.get(snap.shape_type, QColor("#1a73e8"))
        pen = QPen(color, 2.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item: QGraphicsItem

        st = snap.shape_type
        p = snap.params

        if st == ShapeType.LINE:
            pts = p.get("points", [])
            if len(pts) >= 2:
                item = QGraphicsLineItem(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
                item.setPen(pen)
            else:
                item = self._render_bezier_stroke(stroke)

        elif st == ShapeType.ARROW:
            item = self._build_arrow_item(p, pen)

        elif st in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
            cx = float(p.get("center_x", 0))
            cy = float(p.get("center_y", 0))
            rx = float(p.get("radius_x", p.get("radius", 40)))
            ry = float(p.get("radius_y", p.get("radius", 40)))
            angle = float(p.get("angle", 0))
            item = QGraphicsEllipseItem(cx - rx, cy - ry, 2 * rx, 2 * ry)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            if abs(angle) > 0.5:
                item.setTransformOriginPoint(cx, cy)
                item.setRotation(angle)

        elif st == ShapeType.RECTANGLE:
            verts = p.get("vertices", [])
            if len(verts) == 4:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                item = QGraphicsRectItem(min(xs), min(ys),
                                         max(xs) - min(xs), max(ys) - min(ys))
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            else:
                item = self._render_bezier_stroke(stroke)

        elif st == ShapeType.TRIANGLE:
            verts = p.get("vertices", [])
            path = QPainterPath()
            if len(verts) >= 3:
                path.moveTo(verts[0][0], verts[0][1])
                for v in verts[1:]:
                    path.lineTo(v[0], v[1])
                path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)

        else:
            item = self._render_bezier_stroke(stroke)

        item.setZValue(self.Z_SHAPE)
        item.setData(0, stroke.stroke_id)
        self.addItem(item)
        return item

    def _build_arrow_item(self, params: dict, pen: QPen) -> QGraphicsPathItem:
        """Build an arrow path: shaft + two arrowhead lines."""
        pts = params.get("points", [])
        if len(pts) < 2:
            return QGraphicsPathItem()

        start = pts[0]
        end   = pts[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return QGraphicsPathItem()

        ux, uy = dx / length, dy / length
        arrow_len = min(20.0, length * 0.25)
        angle_rad = math.radians(25)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # Two arrowhead directions
        def _rotate(vx: float, vy: float, c: float, s: float) -> tuple[float, float]:
            return vx * c - vy * s, vx * s + vy * c

        l1x, l1y = _rotate(-ux, -uy,  cos_a, sin_a)
        l2x, l2y = _rotate(-ux, -uy,  cos_a, -sin_a)

        path = QPainterPath()
        path.moveTo(start[0], start[1])
        path.lineTo(end[0], end[1])
        path.moveTo(end[0], end[1])
        path.lineTo(end[0] + l1x * arrow_len, end[1] + l1y * arrow_len)
        path.moveTo(end[0], end[1])
        path.lineTo(end[0] + l2x * arrow_len, end[1] + l2y * arrow_len)

        item = QGraphicsPathItem(path)
        item.setPen(pen)
        return item

    def remove_stroke(self, stroke_id: str) -> bool:
        """Remove a stroke by ID. Returns True if found and removed."""
        stroke = self._strokes.pop(stroke_id, None)
        if stroke is None:
            return False
        for attr in ("_path_item", "_shape_item"):
            item = getattr(stroke, attr, None)
            if item is not None and item.scene() is self:
                self.removeItem(item)
        return True

    def clear_all_strokes(self) -> None:
        """Remove all ink strokes from the scene (keep background)."""
        for sid in list(self._strokes.keys()):
            self.remove_stroke(sid)

    # ── OCR overlays ─────────────────────────────────────────────────────────

    def add_text_overlay(
        self,
        text: str,
        x: float,
        y: float,
        font_size: int = 18,
        color: QColor | None = None,
    ) -> QGraphicsTextItem:
        """Place a text label on the board (replaces raw strokes after OCR)."""
        if color is None:
            color = QColor("#1a73e8")
        item = QGraphicsTextItem(text)
        item.setDefaultTextColor(color)
        font = QFont("Segoe UI", font_size)
        item.setFont(font)
        item.setPos(x, y)
        item.setZValue(self.Z_OCR_OVERLAY)
        self.addItem(item)
        return item

    def add_latex_overlay(
        self,
        rgba_img: np.ndarray,
        x: float,
        y: float,
        max_width: float = 600.0,
    ) -> QGraphicsPixmapItem:
        """Render a LaTeX RGBA image onto the board at (x, y)."""
        pixmap = _numpy_rgba_to_qpixmap(rgba_img)
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(
                int(max_width), Qt.TransformationMode.SmoothTransformation
            )
        item = QGraphicsPixmapItem(pixmap)
        item.setPos(x, y)
        item.setZValue(self.Z_OCR_OVERLAY)
        self.addItem(item)
        return item

    # ── Selection rectangle ───────────────────────────────────────────────────

    def show_selection_rect(self, rect: QRectF) -> None:
        """Draw a dashed selection rectangle (SELECT gesture)."""
        if self._select_rect_item is not None:
            self.removeItem(self._select_rect_item)
        pen = QPen(QColor("#1a73e8"), 1.5, Qt.PenStyle.DashLine)
        self._select_rect_item = QGraphicsRectItem(rect)
        self._select_rect_item.setPen(pen)
        self._select_rect_item.setBrush(QBrush(QColor(26, 115, 232, 30)))
        self._select_rect_item.setZValue(self.Z_LASER - 1)
        self.addItem(self._select_rect_item)
        self.selection_rect_changed.emit(rect)

    def hide_selection_rect(self) -> None:
        if self._select_rect_item is not None:
            self.removeItem(self._select_rect_item)
            self._select_rect_item = None

    # ── Serialization helpers ─────────────────────────────────────────────────

    def get_strokes_for_export(self) -> list[dict]:
        """
        Serialize all strokes to a list of dicts for .airteach package export.
        Each dict matches the ink/strokes.json schema.
        """
        result = []
        for stroke in self._strokes.values():
            segs = []
            for seg in stroke.segments:
                segs.append({
                    "p0":  [float(seg.p0[0]),  float(seg.p0[1])],
                    "cp1": [float(seg.p1[0]), float(seg.p1[1])],
                    "cp2": [float(seg.p2[0]), float(seg.p2[1])],
                    "p1":  [float(seg.p3[0]),  float(seg.p3[1])],
                })
            entry: dict = {
                "stroke_id":     stroke.stroke_id,
                "segments":      segs,
                "widths":        [float(w) for w in stroke.widths],
                "color_hex":     stroke.color.name(),
                "timestamps_ms": stroke.timestamps_ms,
            }
            if stroke.snap is not None:
                entry["snap"] = {
                    "shape_type": stroke.snap.shape_type.name,
                    "parameters": stroke.snap.params,
                }
            result.append(entry)
        return result

    def snapshot_as_pixmap(self) -> QPixmap:
        """Render the entire board to a QPixmap (for export / thumbnail)."""
        pixmap = QPixmap(BOARD_WIDTH, BOARD_HEIGHT)
        pixmap.fill(QColor("#f8f9fa"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.render(painter)
        painter.end()
        return pixmap
