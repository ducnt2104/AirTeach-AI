"""
AirTeach AI - Interactive Touchless Lesson Board Canvas
========================================================
Combines QGraphicsView and QGraphicsScene into a high-performance vector
canvas with strict 3-tier Z-layering:

  - Layer 0 (Z = 0):  Document Page Background (locked, never cleared by eraser)
  - Layer 1 (Z = 10): Vector Ink Strokes (Cubic Bézier, variable width),
                      Geometric Snapped Shapes (Line, Arrow, Circle, Ellipse, Rect, Tri),
                      and Smart Ink OCR Text/LaTeX Overlays
  - Layer 2 (Z = 20): Virtual Laser Pointer with soft outer glow (POINT mode),
                      Pen Tip Dot (DRAW mode),
                      Dashed Selection Bounding Box (SELECT mode)

Features:
  - Color palette & variable stroke widths
  - Undo / Redo history stack (with Ctrl+Z / Ctrl+Y shortcuts)
  - Eraser mode (stroke-by-stroke hit testing) and Clear Ink (preserves page)
  - Async offline Smart Ink OCR (Handwriting + LaTeX-OCR replacement)
  - High-DPI antialiased rendering (QPainter.Antialiasing | SmoothPixmapTransform)
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable

import cv2
import numpy as np
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThreadPool,
    QRunnable,
    Signal,
    Slot,
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
    QRadialGradient,
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
    QGraphicsView,
    QSizePolicy,
    QWidget,
)

from src.core.stroke_beautifier import (
    BezierSegment,
    GeometricShapeSnapper,
    OneEuroFilter2D,
    SnappedShape as SnapResult,
    ShapeType,
    fit_cubic_bezier_spline,
    ramer_douglas_peucker,
)
from src.services.smart_ink_ocr import SmartInkRecognizer, Stroke, StrokePoint

logger = logging.getLogger(__name__)

BOARD_WIDTH  = 1920
BOARD_HEIGHT = 1080

# Shape colors
_SNAP_COLOR: dict[ShapeType, QColor] = {
    ShapeType.LINE:      QColor("#1a73e8"),
    ShapeType.ARROW:     QColor("#1a73e8"),
    ShapeType.CIRCLE:    QColor("#34a853"),
    ShapeType.ELLIPSE:   QColor("#34a853"),
    ShapeType.RECTANGLE: QColor("#fbbc04"),
    ShapeType.TRIANGLE:  QColor("#ea4335"),
    ShapeType.FREEFORM:  QColor("#202124"),
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class InkStroke:
    """A fully rendered vector stroke."""
    stroke_id: str
    segments: list[BezierSegment]
    widths: list[float]
    snap: Optional[SnapResult] = None
    color: QColor = field(default_factory=lambda: QColor("#202124"))
    timestamps_ms: list[float] = field(default_factory=list)

    # Item references (not compared in equality checks)
    _path_item: Optional[QGraphicsPathItem] = field(default=None, repr=False, compare=False)
    _shape_item: Optional[QGraphicsItem]    = field(default=None, repr=False, compare=False)


# ─── OCR Async Worker ─────────────────────────────────────────────────────────

class _OcrTask(QRunnable):
    """Runs SmartInkRecognizer in the background to avoid UI blocking."""

    class Signals(QObject):
        completed = Signal(object, list, float, float)  # (OcrResult, stroke_ids, x, y)
        error = Signal(str)

    def __init__(
        self,
        recognizer: SmartInkRecognizer,
        strokes: list[Stroke],
        stroke_ids: list[str],
        top_left_x: float,
        top_left_y: float,
        force_formula: bool = False,
    ) -> None:
        super().__init__()
        self.signals = _OcrTask.Signals()
        self._recognizer = recognizer
        self._strokes = strokes
        self._stroke_ids = stroke_ids
        self._x = top_left_x
        self._y = top_left_y
        self._force_formula = force_formula

    def run(self) -> None:
        try:
            result = self._recognizer.recognize_strokes(
                self._strokes, force_formula=self._force_formula
            )
            self.signals.completed.emit(result, self._stroke_ids, self._x, self._y)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ─── Interactive Multi-Layer Canvas ──────────────────────────────────────────

class LessonBoardCanvas(QGraphicsView):
    """
    Touchless Interactive Teaching Canvas with QGraphicsView + QGraphicsScene.
    
    Signals:
        stroke_added(str): Emitted when a stroke is finalized.
        stroke_removed(str): Emitted when a stroke is undone or erased.
        selection_changed(QRectF): Emitted when the selection box changes.
        ocr_finished(str, bool): Emitted when OCR replaces strokes with text/latex.
    """

    stroke_added      = Signal(str)
    stroke_removed    = Signal(str)
    selection_changed = Signal(QRectF)
    ocr_finished      = Signal(str, bool)  # (recognized_text, is_formula)

    # Z-Layers
    Z_BACKGROUND = 0.0
    Z_INK        = 10.0
    Z_SHAPE      = 11.0
    Z_OVERLAY    = 15.0
    Z_PREVIEW    = 18.0
    Z_CURSOR     = 20.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(0, 0, BOARD_WIDTH, BOARD_HEIGHT, self)
        self.setScene(self._scene)

        # High quality rendering
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setBackgroundBrush(QBrush(QColor("#f8f9fa")))

        # Core tools
        self._filter = OneEuroFilter2D()
        self._snapper = GeometricShapeSnapper()
        self._recognizer = SmartInkRecognizer()
        self._thread_pool = QThreadPool.globalInstance()

        # Canvas settings
        self._ink_color = QColor("#202124")
        self._min_width = 1.5
        self._max_width = 6.0
        self._rdp_epsilon = 2.5
        self._beautify_enabled = True
        self._snapping_enabled = True

        # Layer 0: Background
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._bg_image_bgr: Optional[np.ndarray] = None

        # Layer 1: Strokes, Shapes, Overlays & History
        self._strokes: dict[str, InkStroke] = {}
        self._undo_stack: list[str] = []
        self._redo_stack: list[InkStroke] = []
        self._overlays: list[QGraphicsItem] = []

        # Layer 2: Cursors & Overlays
        self._laser_item: Optional[QGraphicsEllipseItem] = None
        self._laser_glow: Optional[QGraphicsEllipseItem] = None
        self._pen_dot_item: Optional[QGraphicsEllipseItem] = None
        self._preview_item: Optional[QGraphicsPathItem] = None
        self._selection_item: Optional[QGraphicsRectItem] = None

        # Live drawing buffer
        self._live_raw: list[tuple[float, float, float]] = []  # (x, y, t_ms)
        self._current_stroke_id: Optional[str] = None
        self._select_origin: Optional[QPointF] = None

        self._init_cursors()

    # ─── Cursors & Laser Glow ─────────────────────────────────────────────────

    def _init_cursors(self) -> None:
        """Create high-visibility laser cursor with radial gradient glow."""
        # Outer glow
        r_glow = 24.0
        self._laser_glow = QGraphicsEllipseItem(-r_glow, -r_glow, 2 * r_glow, 2 * r_glow)
        grad = QRadialGradient(0, 0, r_glow)
        grad.setColorAt(0.0, QColor(255, 50, 50, 160))
        grad.setColorAt(0.5, QColor(255, 0, 0, 80))
        grad.setColorAt(1.0, QColor(255, 0, 0, 0))
        self._laser_glow.setBrush(QBrush(grad))
        self._laser_glow.setPen(Qt.PenStyle.NoPen)
        self._laser_glow.setZValue(self.Z_CURSOR - 0.1)
        self._laser_glow.setVisible(False)
        self._scene.addItem(self._laser_glow)

        # Core laser dot
        r_core = 6.0
        self._laser_item = QGraphicsEllipseItem(-r_core, -r_core, 2 * r_core, 2 * r_core)
        self._laser_item.setBrush(QBrush(QColor(255, 255, 255)))
        self._laser_item.setPen(QPen(QColor(230, 0, 0), 2))
        self._laser_item.setZValue(self.Z_CURSOR)
        self._laser_item.setVisible(False)
        self._scene.addItem(self._laser_item)

        # Pen tip dot (when DRAW mode is hovering)
        r_pen = 4.0
        self._pen_dot_item = QGraphicsEllipseItem(-r_pen, -r_pen, 2 * r_pen, 2 * r_pen)
        self._pen_dot_item.setBrush(QBrush(self._ink_color))
        self._pen_dot_item.setPen(QPen(QColor(255, 255, 255), 1.5))
        self._pen_dot_item.setZValue(self.Z_CURSOR)
        self._pen_dot_item.setVisible(False)
        self._scene.addItem(self._pen_dot_item)

    def set_laser_position(self, x: float, y: float, visible: bool = True) -> None:
        """Move the glowing virtual laser pointer."""
        if self._laser_item is not None and self._laser_glow is not None:
            self._laser_item.setPos(x, y)
            self._laser_glow.setPos(x, y)
            self._laser_item.setVisible(visible)
            self._laser_glow.setVisible(visible)
        if visible and self._pen_dot_item:
            self._pen_dot_item.setVisible(False)

    def set_pen_dot_position(self, x: float, y: float, visible: bool = True) -> None:
        """Move the pen tip cursor dot."""
        if self._pen_dot_item is not None:
            self._pen_dot_item.setPos(x, y)
            self._pen_dot_item.setBrush(QBrush(self._ink_color))
            self._pen_dot_item.setVisible(visible)
        if visible and self._laser_item and self._laser_glow:
            self._laser_item.setVisible(False)
            self._laser_glow.setVisible(False)

    def hide_cursors(self) -> None:
        """Hide all cursors."""
        if self._laser_item:
            self._laser_item.setVisible(False)
        if self._laser_glow:
            self._laser_glow.setVisible(False)
        if self._pen_dot_item:
            self._pen_dot_item.setVisible(False)

    # ─── Layer 0: Background Management ───────────────────────────────────────

    def set_background_page(self, image_bgr: np.ndarray) -> None:
        """Set page background image (locked on Layer 0)."""
        self._bg_image_bgr = image_bgr.copy()
        if self._bg_item is not None:
            self._scene.removeItem(self._bg_item)
            self._bg_item = None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        scaled = pix.scaled(
            BOARD_WIDTH, BOARD_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._bg_item = QGraphicsPixmapItem(scaled)
        ox = (BOARD_WIDTH - scaled.width()) / 2.0
        oy = (BOARD_HEIGHT - scaled.height()) / 2.0
        self._bg_item.setPos(ox, oy)
        self._bg_item.setZValue(self.Z_BACKGROUND)
        self._scene.addItem(self._bg_item)

    def clear_background(self) -> None:
        """Clear background image."""
        if self._bg_item is not None:
            self._scene.removeItem(self._bg_item)
            self._bg_item = None
            self._bg_image_bgr = None

    # ─── Live Ink Drawing ─────────────────────────────────────────────────────

    def start_stroke(self, x: float, y: float, t_ms: float) -> None:
        """Initialize a new ink stroke preview."""
        self._current_stroke_id = str(uuid.uuid4())
        self._live_raw = [(x, y, t_ms)]
        self._filter.reset()
        fx, fy = self._filter.filter(x, y, t_ms)

        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)

        path = QPainterPath()
        path.moveTo(fx, fy)
        pen = QPen(self._ink_color, 2.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self._preview_item = QGraphicsPathItem(path)
        self._preview_item.setPen(pen)
        self._preview_item.setZValue(self.Z_PREVIEW)
        self._scene.addItem(self._preview_item)

    def update_stroke(self, x: float, y: float, t_ms: float) -> None:
        """Add point to active preview stroke."""
        if self._current_stroke_id is None or self._preview_item is None:
            return
        self._live_raw.append((x, y, t_ms))
        fx, fy = self._filter.filter(x, y, t_ms)

        path = self._preview_item.path()
        path.lineTo(fx, fy)
        self._preview_item.setPath(path)

    def finish_stroke(self) -> Optional[InkStroke]:
        """Finalize stroke, run beautifier/snapper, and commit to Layer 1."""
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

        if self._current_stroke_id is None or len(self._live_raw) < 2:
            self._current_stroke_id = None
            self._live_raw.clear()
            return None

        raw_pts = np.array([(p[0], p[1]) for p in self._live_raw], dtype=np.float64)
        t_ms = [p[2] for p in self._live_raw]

        # Tier 2: RDP simplification + Bezier Spline
        if self._beautify_enabled:
            simplified = ramer_douglas_peucker(raw_pts, epsilon=self._rdp_epsilon)
        else:
            simplified = raw_pts

        dt = np.diff(t_ms) if len(t_ms) > 1 else np.array([1.0])
        dt = np.where(dt < 1.0, 1.0, dt)
        d = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        vels = d / dt
        avg_vel = float(vels.mean()) if len(vels) > 0 else 1.0

        segments = fit_cubic_bezier_spline(
            simplified,
            velocities=vels if len(vels) == len(simplified) else None,
            width_min=self._min_width,
            width_max=self._max_width,
        )
        widths = [(seg.width_start + seg.width_end) / 2.0 for seg in segments]

        # Tier 3: Shape Snapping
        snap: Optional[SnapResult] = None
        if self._snapping_enabled and len(simplified) >= 2:
            try:
                candidate = self._snapper.snap(simplified)
                if candidate.shape_type != ShapeType.FREEFORM:
                    snap = candidate
            except Exception as exc:
                logger.debug("Shape snapping error: %s", exc)

        stroke = InkStroke(
            stroke_id=self._current_stroke_id,
            segments=segments,
            widths=widths,
            snap=snap,
            color=self._ink_color,
            timestamps_ms=t_ms,
        )

        self._render_and_commit(stroke)
        self._current_stroke_id = None
        self._live_raw.clear()
        return stroke

    def _render_and_commit(self, stroke: InkStroke) -> None:
        """Render stroke onto scene and push to Undo stack."""
        if stroke.snap is not None and stroke.snap.shape_type != ShapeType.FREEFORM:
            item = self._render_shape(stroke)
            stroke._shape_item = item
        else:
            item = self._render_bezier(stroke)
            stroke._path_item = item

        self._strokes[stroke.stroke_id] = stroke
        self._undo_stack.append(stroke.stroke_id)
        self._redo_stack.clear()
        self.stroke_added.emit(stroke.stroke_id)

    def _render_bezier(self, stroke: InkStroke) -> QGraphicsPathItem:
        """Render Cubic Bézier segments."""
        path = QPainterPath()
        if stroke.segments:
            path.moveTo(stroke.segments[0].p0[0], stroke.segments[0].p0[1])
            for seg in stroke.segments:
                path.cubicTo(
                    seg.p1[0], seg.p1[1],
                    seg.p2[0], seg.p2[1],
                    seg.p3[0], seg.p3[1],
                )
        avg_width = float(np.mean(stroke.widths)) if stroke.widths else 2.5
        pen = QPen(stroke.color, avg_width, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self.Z_INK)
        item.setData(0, stroke.stroke_id)
        self._scene.addItem(item)
        return item

    def _render_shape(self, stroke: InkStroke) -> QGraphicsItem:
        """Render snapped geometric shapes."""
        snap = stroke.snap
        color = _SNAP_COLOR.get(snap.shape_type, self._ink_color)
        pen = QPen(color, 2.8, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        st = snap.shape_type
        p = snap.params
        item: QGraphicsItem

        if st == ShapeType.LINE:
            pts = p.get("points", [])
            if len(pts) >= 2:
                item = QGraphicsLineItem(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
                item.setPen(pen)
            else:
                item = self._render_bezier(stroke)

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
                item = QGraphicsRectItem(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                item.setPen(pen)
                item.setBrush(Qt.BrushStyle.NoBrush)
            else:
                item = self._render_bezier(stroke)

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
            item = self._render_bezier(stroke)

        item.setZValue(self.Z_SHAPE)
        item.setData(0, stroke.stroke_id)
        self._scene.addItem(item)
        return item

    def _build_arrow_item(self, params: dict, pen: QPen) -> QGraphicsPathItem:
        """Create arrow path with shaft and arrowhead."""
        pts = params.get("points", [])
        if len(pts) < 2:
            return QGraphicsPathItem()

        start = pts[0]
        end = pts[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return QGraphicsPathItem()

        ux, uy = dx / length, dy / length
        arrow_len = min(24.0, length * 0.3)
        angle_rad = math.radians(25)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        l1x = (-ux) * cos_a - (-uy) * sin_a
        l1y = (-ux) * sin_a + (-uy) * cos_a
        l2x = (-ux) * cos_a - (-uy) * (-sin_a)
        l2y = (-ux) * (-sin_a) + (-uy) * cos_a

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

    # ─── Undo / Redo & Eraser ─────────────────────────────────────────────────

    def undo(self) -> bool:
        """Undo last committed stroke."""
        if not self._undo_stack:
            return False
        last_id = self._undo_stack.pop()
        stroke = self._strokes.pop(last_id, None)
        if stroke is None:
            return False

        for attr in ("_path_item", "_shape_item"):
            item = getattr(stroke, attr, None)
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)

        self._redo_stack.append(stroke)
        self.stroke_removed.emit(last_id)
        return True

    def redo(self) -> bool:
        """Redo previously undone stroke."""
        if not self._redo_stack:
            return False
        stroke = self._redo_stack.pop()
        self._render_and_commit(stroke)
        return True

    def erase_at(self, x: float, y: float, radius: float = 16.0) -> bool:
        """
        Stroke-level eraser: removes any stroke intersecting circle at (x, y).
        Background layer is NEVER affected.
        """
        pt = QPointF(x, y)
        erased_any = False

        for sid in list(self._strokes.keys()):
            stroke = self._strokes[sid]
            # Check bounding box first
            hit = False
            for seg in stroke.segments:
                dist = math.hypot(seg.p0[0] - x, seg.p0[1] - y)
                if dist <= radius:
                    hit = True
                    break
            if hit:
                self.remove_stroke(sid)
                erased_any = True

        return erased_any

    def remove_stroke(self, stroke_id: str) -> bool:
        """Remove a stroke by ID (preserves background)."""
        stroke = self._strokes.pop(stroke_id, None)
        if stroke is None:
            return False

        if stroke_id in self._undo_stack:
            self._undo_stack.remove(stroke_id)

        for attr in ("_path_item", "_shape_item"):
            item = getattr(stroke, attr, None)
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)

        self.stroke_removed.emit(stroke_id)
        return True

    def clear_all_ink(self) -> None:
        """Clear all strokes and overlays from Layer 1 while preserving Layer 0 (background)."""
        for sid in list(self._strokes.keys()):
            self.remove_stroke(sid)
        for ov in self._overlays:
            if ov.scene() is self._scene:
                self._scene.removeItem(ov)
        self._overlays.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()

    # ─── Selection Region (SELECT Gesture) ────────────────────────────────────

    def update_selection(self, origin_x: float, origin_y: float, curr_x: float, curr_y: float) -> None:
        """Update dashed selection bounding box on Layer 2."""
        rect = QRectF(QPointF(origin_x, origin_y), QPointF(curr_x, curr_y)).normalized()
        if self._selection_item is None:
            self._selection_item = QGraphicsRectItem(rect)
            pen = QPen(QColor("#1a73e8"), 2, Qt.PenStyle.DashLine)
            self._selection_item.setPen(pen)
            self._selection_item.setBrush(QBrush(QColor(26, 115, 232, 28)))
            self._selection_item.setZValue(self.Z_CURSOR - 0.5)
            self._scene.addItem(self._selection_item)
        else:
            self._selection_item.setRect(rect)
        self.selection_changed.emit(rect)

    def hide_selection(self) -> None:
        """Hide selection bounding box."""
        if self._selection_item is not None:
            self._scene.removeItem(self._selection_item)
            self._selection_item = None

    def get_selected_rect(self) -> Optional[QRectF]:
        """Return the current selection rectangle if visible."""
        if self._selection_item is not None:
            return self._selection_item.rect()
        return None

    # ─── Smart Ink OCR Replacement ───────────────────────────────────────────

    def trigger_ocr_on_selection(self, force_formula: bool = False) -> None:
        """
        Perform offline Smart Ink recognition on strokes inside selection.
        Replaces raw strokes with crisp text or LaTeX RGBA image overlay.
        """
        rect = self.get_selected_rect()
        if rect is None or rect.isEmpty():
            return

        # Find strokes inside selection rect
        target_ids: list[str] = []
        target_strokes: list[Stroke] = []

        for sid, ink in self._strokes.items():
            contained = False
            for seg in ink.segments:
                if rect.contains(QPointF(seg.p0[0], seg.p0[1])):
                    contained = True
                    break
            if contained:
                target_ids.append(sid)
                ocr_pts = [StrokePoint(s.p0[0], s.p0[1], 0.0) for s in ink.segments]
                target_strokes.append(
                    Stroke(points=ocr_pts, color_bgr=(0, 0, 0), width=2.5, stroke_id=sid)
                )

        if not target_strokes:
            self.hide_selection()
            return

        # Launch async task
        task = _OcrTask(
            recognizer=self._recognizer,
            strokes=target_strokes,
            stroke_ids=target_ids,
            top_left_x=rect.x(),
            top_left_y=rect.y(),
            force_formula=force_formula,
        )
        task.signals.completed.connect(self._on_ocr_completed)
        task.signals.error.connect(lambda err: logger.warning("OCR error: %s", err))
        self._thread_pool.start(task)
        self.hide_selection()

    @Slot(object, list, float, float)
    def _on_ocr_completed(self, result, stroke_ids: list[str], x: float, y: float) -> None:
        """Replace ink strokes with text/latex item."""
        # Remove old raw ink strokes
        for sid in stroke_ids:
            self.remove_stroke(sid)

        # Insert overlay
        if result.is_formula and result.render_image_rgba is not None:
            # Rendered LaTeX PNG
            rgb = cv2.cvtColor(result.render_image_rgba, cv2.COLOR_RGBA2BGRA)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_ARGB32).copy()
            pix = QPixmap.fromImage(qimg)
            pix_item = QGraphicsPixmapItem(pix)
            pix_item.setPos(x, y)
            pix_item.setZValue(self.Z_OVERLAY)
            self._scene.addItem(pix_item)
            self._overlays.append(pix_item)
            self.ocr_finished.emit(result.latex, True)

        elif result.text:
            text_item = QGraphicsTextItem(result.text)
            text_item.setDefaultTextColor(QColor("#1a73e8"))
            text_item.setFont(QFont("Segoe UI", 20, QFont.Weight.DemiBold))
            text_item.setPos(x, y)
            text_item.setZValue(self.Z_OVERLAY)
            self._scene.addItem(text_item)
            self._overlays.append(text_item)
            self.ocr_finished.emit(result.text, False)

    # ─── Settings Accessors ───────────────────────────────────────────────────

    @property
    def ink_color(self) -> QColor:
        return self._ink_color

    @ink_color.setter
    def ink_color(self, val: QColor) -> None:
        self._ink_color = val
        if self._pen_dot_item:
            self._pen_dot_item.setBrush(QBrush(val))

    @property
    def stroke_width_range(self) -> tuple[float, float]:
        return self._min_width, self._max_width

    def set_stroke_width(self, min_w: float, max_w: float) -> None:
        self._min_width = min_w
        self._max_width = max_w

    @property
    def beautify_enabled(self) -> bool:
        return self._beautify_enabled

    @beautify_enabled.setter
    def beautify_enabled(self, val: bool) -> None:
        self._beautify_enabled = val

    @property
    def snapping_enabled(self) -> bool:
        return self._snapping_enabled

    @snapping_enabled.setter
    def snapping_enabled(self, val: bool) -> None:
        self._snapping_enabled = val

    # ─── Snapshot & Serialization ─────────────────────────────────────────────

    def snapshot_as_pixmap(self) -> QPixmap:
        """Render board to high-res QPixmap."""
        pixmap = QPixmap(BOARD_WIDTH, BOARD_HEIGHT)
        pixmap.fill(QColor("#ffffff"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._scene.render(painter)
        painter.end()
        return pixmap

    def get_strokes_for_export(self) -> list[dict]:
        """Serialize strokes for .airteach packages."""
        result = []
        for stroke in self._strokes.values():
            segs = []
            for seg in stroke.segments:
                segs.append({
                    "p0": [float(seg.p0[0]), float(seg.p0[1])],
                    "p1": [float(seg.p1[0]), float(seg.p1[1])],
                    "p2": [float(seg.p2[0]), float(seg.p2[1])],
                    "p3": [float(seg.p3[0]), float(seg.p3[1])],
                })
            entry = {
                "stroke_id": stroke.stroke_id,
                "segments": segs,
                "widths": [float(w) for w in stroke.widths],
                "color_hex": stroke.color.name(),
                "timestamps_ms": stroke.timestamps_ms,
            }
            if stroke.snap is not None:
                entry["snap"] = {
                    "shape_type": stroke.snap.shape_type.name,
                    "parameters": stroke.snap.params,
                }
            result.append(entry)
        return result

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
