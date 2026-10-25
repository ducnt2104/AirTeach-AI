"""
AirTeach AI - Spotlight Presentation Overlay
============================================
Provides a presentation spotlight mode on LessonBoard:
  - Dims the entire board with a semi-transparent dark vignette mask.
  - Leaves a bright, clear circular aperture around the teacher's air pointer.
  - Adds a subtle glowing highlight border around the aperture to focus audience attention.
  - High-DPI and multi-resolution responsive canvas boundary support.
  - Touchless air-gesture delta adjustment for dynamic aperture tuning.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)


class SpotlightOverlayItem(QGraphicsObject):
    """
    Presentation Spotlight overlay item spanning the entire LessonBoard canvas.
    Follows touchless air pointer coordinates to illuminate key teaching areas.
    """

    def __init__(
        self,
        board_width: int = 1920,
        board_height: int = 1080,
        initial_radius: float = 135.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board_width = board_width
        self.board_height = board_height
        self._radius = max(40.0, float(initial_radius))
        self._cx: float = board_width / 2.0
        self._cy: float = board_height / 2.0

        # Semi-transparent dark vignette (65% black)
        self._dim_color = QColor(0, 0, 0, 165)
        self._border_color = QColor(255, 255, 255, 210)
        self._border_width = 2.5

        # Ignore mouse/pointer clicks so underlying canvas drawing & items pass through
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)

        self.setZValue(38.0)
        self.setVisible(False)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, float(self.board_width), float(self.board_height))

    def set_board_size(self, width: int, height: int) -> None:
        """Update canvas bounds for full-canvas vignette coverage on resize/High-DPI."""
        self.prepareGeometryChange()
        self.board_width = max(100, int(width))
        self.board_height = max(100, int(height))
        self.update()

    def set_dim_opacity(self, opacity: float) -> None:
        """Set vignette dim opacity (0.10 to 0.95)."""
        clamped = max(0.10, min(0.95, float(opacity)))
        self._dim_color.setAlpha(int(clamped * 255.0))
        self.update()

    def dim_opacity(self) -> float:
        """Return current vignette dim opacity as a normalized float [0.0, 1.0]."""
        return self._dim_color.alpha() / 255.0

    def adjust_radius_delta(self, delta: float, sensitivity: float = 1.0) -> float:
        """
        Smoothly adjust spotlight aperture radius by pinch/gesture delta.
        Returns the new clamped aperture radius [40.0, 450.0].
        """
        new_r = self._radius + delta * sensitivity
        self.set_radius(new_r)
        return self._radius

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if not self.isVisible():
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        full_rect = QRectF(0.0, 0.0, float(self.board_width), float(self.board_height))
        base_path = QPainterPath()
        base_path.addRect(full_rect)

        # Circular cutout at spotlight aperture
        cutout_path = QPainterPath()
        cutout_path.addEllipse(QPointF(self._cx, self._cy), self._radius, self._radius)

        # Subtractive mask
        vignette_path = base_path.subtracted(cutout_path)

        # Draw darkened background
        painter.setBrush(QBrush(self._dim_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(vignette_path)

        # Glowing white border around aperture
        glow_pen = QPen(self._border_color, self._border_width, Qt.PenStyle.SolidLine)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self._cx, self._cy), self._radius, self._radius)

    def set_spotlight_pos(self, x: float, y: float) -> None:
        """Update spotlight aperture center coordinates."""
        self._cx = float(x)
        self._cy = float(y)
        self.update()

    def set_radius(self, radius: float) -> None:
        """Update spotlight aperture radius (clamped between 40px and 450px)."""
        self._radius = max(40.0, min(450.0, float(radius)))
        self.update()

    def radius(self) -> float:
        """Return current aperture radius."""
        return self._radius

    def spotlight_pos(self) -> tuple[float, float]:
        """Return current aperture center (cx, cy)."""
        return self._cx, self._cy
