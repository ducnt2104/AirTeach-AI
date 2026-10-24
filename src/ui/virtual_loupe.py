"""
AirTeach AI - Virtual Loupe (Interactive Magnifying Glass)
=========================================================
Provides an optical zoom loupe overlay on LessonBoard:
  - Magnifies the scene content directly underneath the air pointer by 2x-4x.
  - Features an anti-aliased circular aperture with metallic lens frame.
  - Optical glare highlights and dynamic zoom multiplier readout badge.
  - Safe recursion-free rendering via offscreen scene snapshot.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
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


class VirtualLoupeItem(QGraphicsObject):
    """
    Interactive Whiteboard Magnifying Glass (Virtual Loupe).
    Renders an optically enlarged 2.0x-4.0x circular view of the underlying
    LessonBoard canvas directly at the touchless pointer coordinates.
    """

    def __init__(
        self,
        board_width: int = 1920,
        board_height: int = 1080,
        radius: float = 115.0,
        zoom_factor: float = 2.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board_width = board_width
        self.board_height = board_height
        self._radius = max(50.0, min(300.0, float(radius)))
        self._zoom_factor = max(1.2, min(5.0, float(zoom_factor)))
        self._cx: float = board_width / 2.0
        self._cy: float = board_height / 2.0

        # Layering: sits at Z=37 (just under spotlight Z=38 and laser Z=40)
        self.setZValue(37.0)
        self.setVisible(False)

    def boundingRect(self) -> QRectF:
        r = self._radius + 15.0
        return QRectF(self._cx - r, self._cy - r, 2.0 * r, 2.0 * r)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if not self.isVisible():
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        r = self._radius
        m = self._zoom_factor
        cx = self._cx
        cy = self._cy

        scene = self.scene()

        # 1. Render magnified snapshot of the scene underneath
        if scene is not None:
            src_w = (2.0 * r) / m
            src_h = (2.0 * r) / m
            src_rect = QRectF(cx - src_w / 2.0, cy - src_h / 2.0, src_w, src_h)

            img_w = max(10, int(2.0 * r))
            img_h = max(10, int(2.0 * r))
            img = QImage(img_w, img_h, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(0)

            # Temporarily hide loupe to prevent infinite recursive rendering
            self.setVisible(False)
            sub_painter = QPainter(img)
            sub_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            sub_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            scene.render(sub_painter, QRectF(0.0, 0.0, float(img_w), float(img_h)), src_rect)
            sub_painter.end()
            self.setVisible(True)

            # Clip snapshot to circular aperture
            painter.save()
            clip_path = QPainterPath()
            clip_path.addEllipse(QPointF(cx, cy), r, r)
            painter.setClipPath(clip_path)
            painter.drawImage(QRectF(cx - r, cy - r, 2.0 * r, 2.0 * r), img)

            # Slight optical glass tint (translucent sky blue)
            painter.setBrush(QBrush(QColor(230, 245, 255, 30)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            painter.restore()

        # 2. Metallic / Acrylic Lens Frame
        frame_pen = QPen(QColor(90, 105, 120, 230), 4.5, Qt.PenStyle.SolidLine)
        painter.setPen(frame_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Inner highlight ring
        inner_pen = QPen(QColor(255, 255, 255, 190), 1.8, Qt.PenStyle.SolidLine)
        painter.setPen(inner_pen)
        painter.drawEllipse(QPointF(cx, cy), r - 2.5, r - 2.5)

        # 3. Specular Glare Arc on top-left
        glare_rect = QRectF(cx - r + 7.0, cy - r + 7.0, 2.0 * (r - 7.0), 2.0 * (r - 7.0))
        glare_pen = QPen(QColor(255, 255, 255, 150), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(glare_pen)
        painter.drawArc(glare_rect, 105 * 16, 55 * 16)

        # 4. Center Crosshairs Dot
        cross_pen = QPen(QColor(26, 115, 232, 140), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(cross_pen)
        painter.drawLine(cx - 15.0, cy, cx + 15.0, cy)
        painter.drawLine(cx, cy - 15.0, cx, cy + 15.0)

        # 5. Telemetry Badge on bottom right of loupe: "2.0×"
        badge_rect = QRectF(cx + r * 0.42, cy + r * 0.42, 48.0, 22.0)
        painter.setBrush(QBrush(QColor(32, 33, 36, 215)))
        painter.setPen(QPen(QColor(255, 255, 255, 210), 1.0))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"{m:.1f}×")

    def set_loupe_pos(self, x: float, y: float) -> None:
        """Update loupe center focus coordinates."""
        self.prepareGeometryChange()
        self._cx = float(x)
        self._cy = float(y)
        self.update()

    def set_zoom_factor(self, zoom: float) -> None:
        """Update loupe magnification factor (1.2x to 5.0x)."""
        self._zoom_factor = max(1.2, min(5.0, float(zoom)))
        self.update()

    def set_radius(self, radius: float) -> None:
        """Update loupe lens radius (50px to 300px)."""
        self.prepareGeometryChange()
        self._radius = max(50.0, min(300.0, float(radius)))
        self.update()

    def zoom_factor(self) -> float:
        """Return current magnification factor."""
        return self._zoom_factor

    def radius(self) -> float:
        """Return current lens radius."""
        return self._radius

    def loupe_pos(self) -> tuple[float, float]:
        """Return current loupe center (cx, cy)."""
        return self._cx, self._cy
