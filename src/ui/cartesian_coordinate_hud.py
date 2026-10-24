"""
AirTeach AI - Real-Time Cartesian Oxy Grid Coordinate HUD & Precision Pointer Readout
===================================================================================
Provides a floating, high-precision mathematical coordinate readout HUD:
  - Real-time conversion between scene pixels (1920x1080) and Cartesian Oxy units.
  - Mathematical orientation: screen Y points down, Cartesian Y points up.
  - Dynamic grid snapping (Snap to Grid) with crosshair reticle.
  - Special landmark detection: Origin O(0,0), Ox axis, and Oy axis.
  - Non-intrusive acrylic translucent glass card following mouse or touchless air pointer.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)

logger = logging.getLogger(__name__)


class CartesianCoordinateHudItem(QGraphicsObject):
    """
    Floating HUD item displaying real-time Cartesian Oxy coordinates (X, Y)
    calibrated to the whiteboard grid origin and spacing.
    """

    coordinate_changed = Signal(float, float)  # (math_x, math_y)
    snap_toggled = Signal(bool)

    def __init__(
        self,
        origin: tuple[float, float] = (960.0, 540.0),
        grid_spacing: float = 40.0,
        decimals: int = 1,
        snap_to_grid: bool = False,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._origin_x: float = float(origin[0])
        self._origin_y: float = float(origin[1])
        self._grid_spacing: float = max(10.0, float(grid_spacing))
        self._decimals: int = max(0, min(3, int(decimals)))
        self._snap_to_grid: bool = bool(snap_to_grid)
        self._show_pixels: bool = False

        # Current pointer scene position
        self._pointer_x: float = self._origin_x
        self._pointer_y: float = self._origin_y

        # Badge dimensions & placement
        self._badge_w: float = 148.0
        self._badge_h: float = 38.0
        self._badge_x: float = self._origin_x + 18.0
        self._badge_y: float = self._origin_y - 42.0

        self.setZValue(25.0)  # Floating above strokes/grids, below menus
        self.setAcceptHoverEvents(False)

    # ── Geometry & Mathematical Conversions ───────────────────────────────────

    def set_origin(self, ox: float, oy: float) -> None:
        """Update Cartesian origin point in scene coordinates (default 960, 540)."""
        self.prepareGeometryChange()
        self._origin_x = float(ox)
        self._origin_y = float(oy)
        self.update()

    def origin(self) -> tuple[float, float]:
        return (self._origin_x, self._origin_y)

    def set_grid_spacing(self, spacing: float) -> None:
        """Update grid spacing in pixels (e.g. 40px per unit)."""
        self.prepareGeometryChange()
        self._grid_spacing = max(10.0, float(spacing))
        self.update()

    def grid_spacing(self) -> float:
        return self._grid_spacing

    def set_decimals(self, decimals: int) -> None:
        """Set decimal precision for non-integer coordinates (0 to 3)."""
        self.prepareGeometryChange()
        self._decimals = max(0, min(3, int(decimals)))
        self.update()

    def decimals(self) -> int:
        return self._decimals

    def set_snap_to_grid(self, enabled: bool) -> None:
        """Enable or disable magnetic snapping to integer grid intersections."""
        if self._snap_to_grid != enabled:
            self.prepareGeometryChange()
            self._snap_to_grid = bool(enabled)
            self.snap_toggled.emit(self._snap_to_grid)
            self.update()

    def snap_to_grid(self) -> bool:
        return self._snap_to_grid

    def toggle_snap_to_grid(self) -> bool:
        """Toggle grid snapping on/off; returns new state."""
        self.set_snap_to_grid(not self._snap_to_grid)
        return self._snap_to_grid

    def set_show_pixels(self, show: bool) -> None:
        """Toggle pixel coordinate secondary readout."""
        self.prepareGeometryChange()
        self._show_pixels = bool(show)
        self._badge_h = 48.0 if self._show_pixels else 38.0
        self.update()

    def show_pixels(self) -> bool:
        return self._show_pixels

    def scene_to_math(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Convert scene pixels (sx, sy) to Cartesian coordinates (X, Y).
        Note: Screen Y points down, Cartesian Y points up.
        """
        mx = (sx - self._origin_x) / self._grid_spacing
        my = -(sy - self._origin_y) / self._grid_spacing
        return (mx, my)

    def math_to_scene(self, mx: float, my: float) -> tuple[float, float]:
        """Convert Cartesian coordinates (X, Y) back to scene pixels."""
        sx = self._origin_x + mx * self._grid_spacing
        sy = self._origin_y - my * self._grid_spacing
        return (sx, sy)

    def get_snapped_math_pos(self, sx: float, sy: float) -> tuple[int, int]:
        """Return nearest integer Cartesian coordinates."""
        mx, my = self.scene_to_math(sx, sy)
        return (int(round(mx)), int(round(my)))

    def get_snapped_scene_pos(self, sx: float, sy: float) -> tuple[float, float]:
        """Return nearest grid intersection in scene pixel coordinates."""
        snap_mx, snap_my = self.get_snapped_math_pos(sx, sy)
        return self.math_to_scene(float(snap_mx), float(snap_my))

    # ── Pointer Tracking & Positioning ────────────────────────────────────────

    def set_pointer_pos(self, px: float, py: float) -> None:
        """Update active pointer position (mouse or air gesture)."""
        self.prepareGeometryChange()
        self._pointer_x = float(px)
        self._pointer_y = float(py)

        # Reposition badge with offset, clamped within 1920x1080 scene margins
        margin = 12.0
        target_bx = self._pointer_x + 18.0
        target_by = self._pointer_y - self._badge_h - 10.0

        # Flip horizontally if near right boundary
        if target_bx + self._badge_w > 1920.0 - margin:
            target_bx = self._pointer_x - self._badge_w - 18.0

        # Flip vertically if near top boundary
        if target_by < margin:
            target_by = self._pointer_y + 22.0

        # Clamp to canvas boundaries
        self._badge_x = max(margin, min(1920.0 - self._badge_w - margin, target_bx))
        self._badge_y = max(margin, min(1080.0 - self._badge_h - margin, target_by))

        mx, my = self.scene_to_math(self._pointer_x, self._pointer_y)
        self.coordinate_changed.emit(mx, my)
        self.update()

    def pointer_pos(self) -> tuple[float, float]:
        return (self._pointer_x, self._pointer_y)

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        # Enclose badge and optional snap reticle
        margin = 30.0
        xs = [self._pointer_x, self._badge_x, self._badge_x + self._badge_w]
        ys = [self._pointer_y, self._badge_y, self._badge_y + self._badge_h]

        if self._snap_to_grid:
            snap_sx, snap_sy = self.get_snapped_scene_pos(self._pointer_x, self._pointer_y)
            xs.append(snap_sx)
            ys.append(snap_sy)

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

        px, py = self._pointer_x, self._pointer_y
        mx, my = self.scene_to_math(px, py)
        snap_sx, snap_sy = self.get_snapped_scene_pos(px, py)
        snap_mx, snap_my = self.get_snapped_math_pos(px, py)

        # 1. Landmark proximity checks
        is_origin = (math.hypot(px - self._origin_x, py - self._origin_y) <= 12.0)
        is_ox = (abs(py - self._origin_y) <= 6.0)
        is_oy = (abs(px - self._origin_x) <= 6.0)

        # 2. Draw Snap Reticle (if snap-to-grid is active)
        if self._snap_to_grid:
            reticle_color = QColor("#10b981")  # Emerald Snap
            painter.setPen(QPen(reticle_color, 1.5, Qt.PenStyle.DashLine))
            # Connecting guideline from pointer to snap intersection
            painter.drawLine(int(px), int(py), int(snap_sx), int(snap_sy))

            # Snap crosshair
            c_pen = QPen(reticle_color, 2.0, Qt.PenStyle.SolidLine)
            painter.setPen(c_pen)
            r_arm = 8.0
            painter.drawLine(int(snap_sx - r_arm), int(snap_sy), int(snap_sx + r_arm), int(snap_sy))
            painter.drawLine(int(snap_sx), int(snap_sy - r_arm), int(snap_sx), int(snap_sy + r_arm))

            # Inner snap dot
            painter.setBrush(QBrush(reticle_color))
            painter.drawEllipse(QPointF(snap_sx, snap_sy), 3.0, 3.0)

        # 3. Draw Acrylic Glass Coordinate Badge
        badge_rect = QRectF(self._badge_x, self._badge_y, self._badge_w, self._badge_h)

        # Theme color: gold if origin, emerald if snapped, cyan if standard
        if is_origin:
            accent_color = QColor("#f59e0b")  # Amber gold
        elif self._snap_to_grid:
            accent_color = QColor("#10b981")  # Emerald
        else:
            accent_color = QColor("#38bdf8")  # Electric cyan

        # Frosted glass card background
        card_bg = QColor(15, 23, 42, 235)  # Dark slate glass
        painter.setBrush(QBrush(card_bg))
        painter.setPen(QPen(accent_color, 1.5))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        # Accent status strip on left
        strip_rect = QRectF(self._badge_x, self._badge_y + 4.0, 3.5, self._badge_h - 8.0)
        painter.setBrush(QBrush(accent_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(strip_rect, 1.5, 1.5)

        # 4. Format Coordinate Text
        if is_origin:
            text_line = "O (0, 0)"
            sub_tag = "[Gốc O]"
        elif self._snap_to_grid:
            text_line = f"({snap_mx}, {snap_my})"
            sub_tag = "🧲 SNAP"
        else:
            # Format according to precision
            if self._decimals == 0:
                text_line = f"({int(round(mx))}, {int(round(my))})"
            else:
                fmt = f"{{:.{self._decimals}f}}"
                text_line = f"({fmt.format(mx)}, {fmt.format(my)})"

            if is_ox:
                sub_tag = "[Ox]"
            elif is_oy:
                sub_tag = "[Oy]"
            else:
                sub_tag = ""

        # Draw primary text line
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        text_rect = QRectF(self._badge_x + 10.0, self._badge_y + 3.0, self._badge_w - 20.0, 18.0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text_line)

        # Draw sub-tag if present
        if sub_tag:
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(accent_color))
            tag_rect = QRectF(self._badge_x + 10.0, self._badge_y + 3.0, self._badge_w - 20.0, 18.0)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, sub_tag)

        # 5. Secondary line: pixel coords or hint
        if self._show_pixels:
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QPen(QColor("#94a3b8")))
            px_rect = QRectF(self._badge_x + 10.0, self._badge_y + 24.0, self._badge_w - 20.0, 16.0)
            painter.drawText(px_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"px: {int(px)}, {int(py)}")
        else:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor("#64748b")))
            hint_rect = QRectF(self._badge_x + 10.0, self._badge_y + 20.0, self._badge_w - 20.0, 14.0)
            hint_txt = "Lưới Oxy 1:1" if not self._snap_to_grid else f"Bắt dính ({self._grid_spacing:.0f}px)"
            painter.drawText(hint_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, hint_txt)
