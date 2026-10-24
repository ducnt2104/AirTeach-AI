"""
AirTeach AI - Whiteboard Canvas Mini-Map & Bird's Eye View Navigator HUD
=======================================================================
Provides an interactive bird's eye view thumbnail of the 1920x1080 LessonBoard:
  - Scaled miniature representation (exact 1:12 scale, 160x90 px canvas).
  - Glowing translucent viewport indicator showing active camera window.
  - Interactive navigation: click or drag on the mini-map to instantly pan the main canvas.
  - Touchless air pointer support for gesture navigation.
  - Frosted dark acrylic HUD styled to match CanvasZoomWidget.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsView, QWidget

logger = logging.getLogger(__name__)

BOARD_WIDTH: float = 1920.0
BOARD_HEIGHT: float = 1080.0

MINI_MAP_W: float = 160.0
MINI_MAP_H: float = 90.0
OFFSET_X: float = 8.0
OFFSET_Y: float = 22.0
SCALE_FACTOR: float = 1.0 / 12.0  # 1920 * (1/12) = 160, 1080 * (1/12) = 90


class CanvasMiniMapWidget(QWidget):
    """
    Floating acrylic HUD widget showing a bird's eye view mini-map of the LessonBoard.
    Displays active content and a movable viewport rectangle reflecting current zoom/pan.
    """

    navigated = Signal(float, float)  # (scene_x, scene_y)
    visibility_toggled = Signal(bool)

    def __init__(
        self,
        view: Optional[QGraphicsView] = None,
        scene: Optional[QGraphicsScene] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent or view)
        self.setObjectName("CanvasMiniMapWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self._view: Optional[QGraphicsView] = view
        self._scene: Optional[QGraphicsScene] = scene
        self._is_dragging: bool = False

        self.widget_width: int = int(MINI_MAP_W + 16.0)
        self.widget_height: int = int(MINI_MAP_H + 30.0)
        self.setFixedSize(self.widget_width, self.widget_height)

    # ── View & Scene Connection ────────────────────────────────────────────────

    def set_view_and_scene(self, view: QGraphicsView, scene: QGraphicsScene) -> None:
        """Bind graphics view and scene to the mini-map."""
        self._view = view
        self._scene = scene
        self.setParent(self._view)
        self.update_viewport()

    def update_viewport(self) -> None:
        """Request a repaint to update the viewport indicator and content."""
        self.update()

    # ── Coordinate Transformations ─────────────────────────────────────────────

    def scene_to_map(self, sx: float, sy: float) -> QPointF:
        """Transform canvas scene coordinates to widget local map coordinates."""
        mx = OFFSET_X + sx * SCALE_FACTOR
        my = OFFSET_Y + sy * SCALE_FACTOR
        return QPointF(mx, my)

    def map_to_scene(self, mx: float, my: float) -> tuple[float, float]:
        """Transform widget local map coordinates to canvas scene coordinates."""
        rel_x = max(0.0, min(MINI_MAP_W, mx - OFFSET_X))
        rel_y = max(0.0, min(MINI_MAP_H, my - OFFSET_Y))
        sx = rel_x / SCALE_FACTOR
        sy = rel_y / SCALE_FACTOR
        return sx, sy

    def get_viewport_rect(self) -> QRectF:
        """Calculate the visible viewport boundary in local mini-map coordinates."""
        if not self._view or not self._scene:
            return QRectF(OFFSET_X, OFFSET_Y, MINI_MAP_W, MINI_MAP_H)

        viewport_scene = self._view.mapToScene(self._view.viewport().rect()).boundingRect()

        # Map top-left and bottom-right to mini-map coordinates
        tl = self.scene_to_map(viewport_scene.left(), viewport_scene.top())
        br = self.scene_to_map(viewport_scene.right(), viewport_scene.bottom())

        # Clamp to mini-map inner canvas rectangle
        left = max(OFFSET_X, min(OFFSET_X + MINI_MAP_W, tl.x()))
        top = max(OFFSET_Y, min(OFFSET_Y + MINI_MAP_H, tl.y()))
        right = max(OFFSET_X, min(OFFSET_X + MINI_MAP_W, br.x()))
        bottom = max(OFFSET_Y, min(OFFSET_Y + MINI_MAP_H, br.y()))

        w = max(4.0, right - left)
        h = max(3.0, bottom - top)
        return QRectF(left, top, w, h)

    def center_view_on(self, sx: float, sy: float) -> None:
        """Center the parent QGraphicsView on specified scene coordinates."""
        if self._view:
            clamped_x = max(0.0, min(BOARD_WIDTH, sx))
            clamped_y = max(0.0, min(BOARD_HEIGHT, sy))
            self._view.centerOn(clamped_x, clamped_y)
            self.navigated.emit(clamped_x, clamped_y)
            self.update()

    # ── Geometry Positioning in View ───────────────────────────────────────────

    def update_position(self) -> None:
        """Position mini-map widget at bottom-right, directly above CanvasZoomWidget."""
        if not self._view:
            return
        vw = self._view.viewport().width()
        vh = self._view.viewport().height()
        margin_x = 24
        # Position stacked 12px above CanvasZoomWidget (38px + 20px margin + 12px gap)
        margin_y = 20 + 38 + 12 + self.widget_height
        self.move(vw - self.widget_width - margin_x, vh - margin_y)

    # ── Painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer acrylic frosted card
        outer_rect = QRectF(0.0, 0.0, float(self.widget_width), float(self.widget_height))
        outer_path = QPainterPath()
        outer_path.addRoundedRect(outer_rect, 12.0, 12.0)
        painter.fillPath(outer_path, QBrush(QColor(22, 24, 34, 230)))
        painter.strokePath(outer_path, QPen(QColor(255, 255, 255, 30), 1.0))

        # 2. Header: Title & Close Button
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(10, 15, "🗺️ Bản đồ (Mini-Map)")

        # Close button [✕]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#64748b")))
        painter.drawText(self.widget_width - 18, 15, "✕")

        # 3. Inner canvas area (16:9 ratio, 160x90)
        inner_rect = QRectF(OFFSET_X, OFFSET_Y, MINI_MAP_W, MINI_MAP_H)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner_rect, 4.0, 4.0)
        painter.fillPath(inner_path, QBrush(QColor(15, 17, 23, 220)))
        painter.strokePath(inner_path, QPen(QColor(255, 255, 255, 20), 1.0))

        # 4. Thumbnail rendering of scene content
        if self._scene is not None:
            painter.save()
            painter.setClipPath(inner_path)

            # Draw strokes and item thumbnails
            for item in self._scene.items():
                if not item.isVisible():
                    continue

                # Draw stroke paths scaled down
                if isinstance(item, QGraphicsPathItem):
                    pen = item.pen()
                    path = item.path()
                    if not path.isEmpty():
                        mini_pen = QPen(pen.color(), max(1.0, pen.widthF() * SCALE_FACTOR))
                        painter.setPen(mini_pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)

                        # Scale and translate path to mini-map coordinates
                        # Transformation: (x * scale + offset_x, y * scale + offset_y)
                        t_path = QPainterPath()
                        for i in range(path.elementCount()):
                            el = path.elementAt(i)
                            mx = OFFSET_X + (el.x + item.x()) * SCALE_FACTOR
                            my = OFFSET_Y + (el.y + item.y()) * SCALE_FACTOR
                            if el.isMoveTo():
                                t_path.moveTo(mx, my)
                            elif el.isLineTo():
                                t_path.lineTo(mx, my)
                        painter.drawPath(t_path)
                elif hasattr(item, "boundingRect") and item.parentItem() is None:
                    # Draw subtle bounding box for sticky notes, logic gates, stem tools
                    b = item.boundingRect()
                    ix, iy = item.x(), item.y()
                    tl = self.scene_to_map(ix + b.left(), iy + b.top())
                    br = self.scene_to_map(ix + b.right(), iy + b.bottom())
                    mini_item_rect = QRectF(tl, br)
                    if mini_item_rect.width() > 1.0 and mini_item_rect.height() > 1.0:
                        painter.setPen(QPen(QColor(99, 102, 241, 120), 0.8))
                        painter.setBrush(QBrush(QColor(99, 102, 241, 40)))
                        painter.drawRoundedRect(mini_item_rect, 1.5, 1.5)

            painter.restore()

        # 5. Glowing Viewport Box
        vp_rect = self.get_viewport_rect()
        painter.setBrush(QBrush(QColor(59, 130, 246, 45)))
        painter.setPen(QPen(QColor(96, 165, 250, 220), 1.5))
        painter.drawRoundedRect(vp_rect, 2.0, 2.0)

        # Center indicator crosshair dot
        cx, cy = vp_rect.center().x(), vp_rect.center().y()
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 1.5, 1.5)

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        px, py = pos.x(), pos.y()

        # 1. Close button hit test
        if px >= self.widget_width - 24.0 and py <= 20.0:
            self.hide()
            self.visibility_toggled.emit(False)
            event.accept()
            return

        # 2. Mini-map canvas hit test
        if OFFSET_X <= px <= OFFSET_X + MINI_MAP_W and OFFSET_Y <= py <= OFFSET_Y + MINI_MAP_H:
            self._is_dragging = True
            sx, sy = self.map_to_scene(px, py)
            self.center_view_on(sx, sy)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging:
            pos = event.position()
            sx, sy = self.map_to_scene(pos.x(), pos.y())
            self.center_view_on(sx, sy)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging:
            self._is_dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Touchless Air Gesture Tap / Drag Support ───────────────────────────────

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Allows touchless air tapping on the mini-map to center viewport.
        canvas_x, canvas_y are scene coordinates.
        Returns True if the air tap was within the mini-map widget.
        """
        if self.isHidden() or not self._view:
            return False

        # Convert scene pos to view viewport pos, then to widget local pos
        viewport_pt = self._view.mapFromScene(QPointF(canvas_x, canvas_y))
        widget_pt = self.mapFrom(self._view.viewport(), viewport_pt)
        wx, wy = widget_pt.x(), widget_pt.y()

        # Check if inside mini-map widget bounds
        if 0.0 <= wx <= self.widget_width and 0.0 <= wy <= self.widget_height:
            # Check if close button
            if wx >= self.widget_width - 24.0 and wy <= 20.0:
                self.hide()
                self.visibility_toggled.emit(False)
                return True

            # If inside map canvas, navigate to clicked location
            if OFFSET_X <= wx <= OFFSET_X + MINI_MAP_W and OFFSET_Y <= wy <= OFFSET_Y + MINI_MAP_H:
                sx, sy = self.map_to_scene(wx, wy)
                self.center_view_on(sx, sy)
                return True

            return True

        return False
