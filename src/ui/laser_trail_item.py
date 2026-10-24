"""
Dynamic Fading Laser Trail (Sparkler Effect) for AirTeach AI.

Provides glowing neon temporary ink trail behind the laser pointer
that automatically dissolves after 1.0 to 2.5 seconds, allowing teachers
to circle, underline, or highlight content without altering permanent slide ink.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem


class LaserTrailItem(QGraphicsItem):
    """
    QGraphicsItem rendering a fading neon particle trail behind the laser pointer.

    Points are automatically pruned when older than `lifetime_sec`.
    A lightweight internal QTimer drives repainting only while points exist,
    stopping automatically when the trail fades away (0% idle CPU overhead).
    """

    def __init__(
        self,
        color: QColor | str = "#ff1744",
        lifetime_sec: float = 1.5,
        base_width: float = 6.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._color = QColor(color) if isinstance(color, str) else color
        self._lifetime_sec = max(0.2, float(lifetime_sec))
        self._base_width = max(2.0, float(base_width))

        # Points deque: (x, y, timestamp_s)
        self._points: deque[tuple[float, float, float]] = deque()
        self._bounds = QRectF(0, 0, 1920, 1080)

        # Decay timer: 30 FPS (33ms)
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_timer_tick)

        self.setZValue(39)  # Just below Z_LASER (40)
        self.setAcceptHoverEvents(False)

    def set_color(self, color: QColor | str) -> None:
        """Set base glowing color of the trail."""
        self._color = QColor(color) if isinstance(color, str) else color
        self.update()

    def color(self) -> QColor:
        """Get trail color."""
        return self._color

    def set_lifetime(self, lifetime_sec: float) -> None:
        """Set dissolution lifetime in seconds."""
        self._lifetime_sec = max(0.2, float(lifetime_sec))

    def lifetime(self) -> float:
        """Get dissolution lifetime in seconds."""
        return self._lifetime_sec

    def add_point(self, x: float, y: float, timestamp: Optional[float] = None) -> None:
        """Add a new position sample to the trail."""
        now = time.perf_counter() if timestamp is None else timestamp
        # Filter out micro-jitter if distance < 2px from last point
        if self._points:
            last_x, last_y, _ = self._points[-1]
            dx = x - last_x
            dy = y - last_y
            if dx * dx + dy * dy < 4.0:
                return

        self._points.append((x, y, now))
        self._update_bounds()
        self.update()

        if not self._timer.isActive():
            self._timer.start()

    def clear(self) -> None:
        """Immediately clear all trail points."""
        self.prepareGeometryChange()
        self._points.clear()
        if self._timer.isActive():
            self._timer.stop()
        self.update()

    def _on_timer_tick(self) -> None:
        """Prune expired points and trigger repaint; stop timer when empty."""
        now = time.perf_counter()
        cutoff = now - self._lifetime_sec

        pruned = False
        while self._points and self._points[0][2] < cutoff:
            self._points.popleft()
            pruned = True

        if pruned:
            self._update_bounds()

        if not self._points:
            self._timer.stop()

        self.update()

    def _update_bounds(self) -> None:
        """Recompute item bounding rectangle based on active points."""
        if not self._points:
            self._bounds = QRectF(0, 0, 10, 10)
            return

        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        pad = self._base_width * 2.5 + 4.0

        min_x = min(xs) - pad
        min_y = min(ys) - pad
        max_x = max(xs) + pad
        max_y = max(ys) + pad

        self.prepareGeometryChange()
        self._bounds = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if len(self._points) < 2:
            return

        now = time.perf_counter()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        pts = list(self._points)
        num_pts = len(pts)

        # Draw line segments with fading alpha and diminishing width
        for i in range(num_pts - 1):
            x1, y1, t1 = pts[i]
            x2, y2, t2 = pts[i + 1]

            age = now - ((t1 + t2) / 2.0)
            if age >= self._lifetime_sec:
                continue

            # Fractional opacity: 1.0 (newest) -> 0.0 (expired)
            fraction = max(0.0, min(1.0, 1.0 - (age / self._lifetime_sec)))

            r, g, b = self._color.red(), self._color.green(), self._color.blue()

            # 1. Outer Neon Glow Arc (wide, semi-transparent)
            glow_pen = QPen(
                QColor(r, g, b, int(75 * fraction)),
                self._base_width * (1.2 + 0.8 * fraction),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(glow_pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # 2. Inner Hot Core (narrow, bright white-tinted)
            core_r = min(255, int(r * 0.4 + 255 * 0.6))
            core_g = min(255, int(g * 0.4 + 255 * 0.6))
            core_b = min(255, int(b * 0.4 + 255 * 0.6))

            core_pen = QPen(
                QColor(core_r, core_g, core_b, int(230 * fraction)),
                max(1.5, self._base_width * 0.45 * fraction),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(core_pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        painter.restore()
