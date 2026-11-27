#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glassmorphic Right-Click Radial / Pie Menu (`cv_engine/radial_pie_menu.py`)
for AirTeach AI SmartboardCanvas (Phase 3 Upgrade).

Interaction:
  Hold Right-Click -> Drag towards any of 8 directional sectors -> Release to activate.
  Drag into center deadzone (< 22px) -> Release to cancel without changing tool.
  Drag beyond outer ring (> 142px) on Pen / Shapes / Clear-or-New -> Pick quick Sub-Ring option
  (ink colors, Highlighter, Smart Shape, specific Shape types, Sticky Note, New Page, Clear Board, AI Lasso, Laser).

8 Main Sectors clockwise from Top (North):
  0 (N,   -90°): PEN          — Bút vẽ
  1 (NE,  -45°): ERASER       — Tẩy
  2 (E,     0°): SELECT       — Chọn / Di chuyển Object
  3 (SE,  +45°): PAN          — Di chuyển bảng
  4 (S,   +90°): TEXT         — Thêm chữ
  5 (SW, +135°): SHAPES       — Hình khối
  6 (W,  ±180°): UNDO         — Hoàn tác
  7 (NW, -135°): CLEAR_OR_NEW — Xóa bảng / Trang mới
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QVariantAnimation, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget


DEFAULT_PIE_DEADZONE_PX = 22.0
DEFAULT_PIE_INNER_RADIUS_PX = 42.0
DEFAULT_PIE_OUTER_RADIUS_PX = 136.0
DEFAULT_SUB_RING_INNER_PX = 144.0
DEFAULT_SUB_RING_OUTER_PX = 198.0


@dataclass(frozen=True)
class PieSliceSpec:
    index: int
    action_id: str
    direction: str
    short_label: str
    full_label_vi: str
    glyph: str
    accent_hex: str


@dataclass(frozen=True)
class PieSubItemSpec:
    action_id: str
    short_label: str
    full_label_vi: str
    glyph: str
    accent_hex: str


PIE_MENU_SLICES: Tuple[PieSliceSpec, ...] = (
    PieSliceSpec(0, "PEN", "N", "Bút Vẽ", "Bút Vẽ (Pen)", "✎", "#00F5D4"),
    PieSliceSpec(1, "ERASER", "NE", "Tẩy", "Tẩy Nét (Eraser)", "⌫", "#F43F5E"),
    PieSliceSpec(2, "SELECT", "E", "Chọn & Kéo", "Chọn & Kéo Object (Select)", "⬉", "#38BDF8"),
    PieSliceSpec(3, "PAN", "SE", "Kéo Bảng", "Di Chuyển Bảng (Pan)", "✥", "#10B981"),
    PieSliceSpec(4, "TEXT", "S", "Thêm Chữ", "Thêm Hộp Văn Bản (Text)", "T", "#FACC15"),
    PieSliceSpec(5, "SHAPES", "SW", "Hình Khối", "Vẽ Hình Khối (Shapes)", "◇", "#A855F7"),
    PieSliceSpec(6, "UNDO", "W", "Hoàn Tác", "Hoàn Tác (Undo)", "↶", "#FB923C"),
    PieSliceSpec(7, "CLEAR_OR_NEW", "NW", "Cử Chỉ / AI", "Sơ Đồ Cử Chỉ / AI & Bảng", "✦", "#06B6D4"),
)

PIE_SUB_ACTIONS: Dict[str, Tuple[PieSubItemSpec, ...]] = {
    "PEN": (
        PieSubItemSpec("COLOR:#00F5D4", "Cyan", "Mực Xanh Ngọc (Cyan)", "●", "#00F5D4"),
        PieSubItemSpec("COLOR:#FEE440", "Vàng", "Mực Vàng Nổi Bật", "●", "#FEE440"),
        PieSubItemSpec("COLOR:#F43F5E", "Đỏ", "Mực Đỏ Chấm Bài", "●", "#F43F5E"),
        PieSubItemSpec("HIGHLIGHTER", "Dạ Quang", "Bút Dạ Quang (Highlighter)", "🖍", "#FACC15"),
        PieSubItemSpec("TOGGLE_SMART_SHAPE", "Nắn Hình", "Bật / Tắt Nắn Hình Thông Minh", "◇", "#10B981"),
    ),
    "SHAPES": (
        PieSubItemSpec("SHAPE:rect", "Chữ Nhật", "Hình Chữ Nhật", "▭", "#38BDF8"),
        PieSubItemSpec("SHAPE:circle", "Tròn", "Hình Tròn", "◯", "#00F5D4"),
        PieSubItemSpec("SHAPE:triangle", "Tam Giác", "Hình Tam Giác", "△", "#FACC15"),
        PieSubItemSpec("SHAPE:arrow", "Mũi Tên", "Mũi Tên Vector", "➜", "#A855F7"),
        PieSubItemSpec("STICKY", "Ghi Chú", "Giấy Ghi Chú (Sticky Note)", "🗒", "#FEF08A"),
    ),
    "CLEAR_OR_NEW": (
        PieSubItemSpec("NEW_PAGE", "Trang Mới", "Thêm Trang Trắng Mới", "📄", "#10B981"),
        PieSubItemSpec("CLEAR_BOARD", "Xóa Bảng", "Xóa Sạch Trang Hiện Tại", "🗑", "#F43F5E"),
        PieSubItemSpec("ASK_AI_REGION", "Hỏi AI", "Khoanh Vùng Hỏi Trợ Lý AI", "✦", "#A855F7"),
        PieSubItemSpec("OPEN_GESTURE_MAP", "Sơ Đồ Cử Chỉ", "Mở Sơ Đồ Cử Chỉ 1 & 2 Tay", "🖐", "#FEE440"),
        PieSubItemSpec("OPEN_AI_CHAT", "Mở AI Chat", "Mở Ngăn Trợ Lý AI Chat", "🤖", "#06B6D4"),
    ),
}


def resolve_pie_slice(
    dx: float,
    dy: float,
    deadzone_px: float = DEFAULT_PIE_DEADZONE_PX,
) -> Optional[int]:
    """
    Map cursor offset `(dx, dy)` relative to pie center onto slice index `0..7`
    starting clockwise from Top (`0 = N, 1 = NE, 2 = E, ..., 7 = NW`).
    Returns `None` when inside `deadzone_px` (Cancel).
    """
    dist = math.hypot(float(dx), float(dy))
    if dist < max(1.0, float(deadzone_px)):
        return None
    angle_deg = math.degrees(math.atan2(float(dy), float(dx)))
    shifted = (angle_deg + 90.0 + 360.0) % 360.0
    idx = int(((shifted + 22.5) % 360.0) // 45.0)
    return idx % 8


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def resolve_pie_sub_item(
    dx: float,
    dy: float,
    parent_idx: int,
    *,
    sub_inner_px: float = DEFAULT_SUB_RING_INNER_PX,
) -> Optional[PieSubItemSpec]:
    """
    Resolve an outer sub-ring item when the pointer moves beyond `sub_inner_px`
    for a parent slice that defines `PIE_SUB_ACTIONS`.
    """
    if not (0 <= parent_idx < len(PIE_MENU_SLICES)):
        return None
    parent_spec = PIE_MENU_SLICES[parent_idx]
    sub_items = PIE_SUB_ACTIONS.get(parent_spec.action_id)
    if not sub_items:
        return None
    dist = math.hypot(float(dx), float(dy))
    if dist < float(sub_inner_px):
        return None

    # Pointer angle in clockwise-from-North degrees [0, 360)
    angle_cw = (math.degrees(math.atan2(float(dy), float(dx))) + 90.0 + 360.0) % 360.0
    parent_center_cw = parent_idx * 45.0
    n = len(sub_items)
    step_deg = 22.0
    total_span = step_deg * n
    rel = _angle_diff_deg(angle_cw, parent_center_cw)
    if abs(rel) > total_span * 0.5 + 14.0:
        return None
    offset = rel + 0.5 * total_span
    sub_idx = max(0, min(n - 1, int(offset // step_deg)))
    return sub_items[sub_idx]


class RadialPieMenuOverlay(QWidget):
    """Glassmorphic 8-slice Radial Pie Menu with Center Info Hub & Outer Sub-Ring."""

    action_triggered = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.deadzone_px: float = DEFAULT_PIE_DEADZONE_PX
        self.inner_radius: float = DEFAULT_PIE_INNER_RADIUS_PX
        self.outer_radius: float = DEFAULT_PIE_OUTER_RADIUS_PX
        self.sub_inner_radius: float = DEFAULT_SUB_RING_INNER_PX
        self.sub_outer_radius: float = DEFAULT_SUB_RING_OUTER_PX
        self._widget_size: int = int((self.sub_outer_radius + 64.0) * 2.0)
        self.setFixedSize(self._widget_size, self._widget_size)

        self.center_in_parent: QPointF = QPointF(0.0, 0.0)
        self.active_slice_idx: Optional[int] = None
        self.active_sub_item: Optional[PieSubItemSpec] = None
        self.smart_shape_active: bool = True
        self.dark_mode: bool = True
        self._anim_scale: float = 1.0

        self._scale_anim = QVariantAnimation(self)
        self._scale_anim.setDuration(125)
        self._scale_anim.setStartValue(0.82)
        self._scale_anim.setEndValue(1.0)
        self._scale_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._scale_anim.valueChanged.connect(self._on_anim_value)
        self.hide()

    def _on_anim_value(self, val: object) -> None:
        try:
            self._anim_scale = float(val)  # type: ignore[arg-type]
        except Exception:
            self._anim_scale = 1.0
        self.update()

    def set_dark_mode(self, dark: bool) -> None:
        self.dark_mode = bool(dark)
        if self.isVisible():
            self.update()

    def open_at(self, parent_pos: QPointF, *, smart_shape_enabled: bool = True) -> None:
        cx = float(parent_pos.x())
        cy = float(parent_pos.y())
        par = self.parentWidget()
        if par is not None:
            pw = float(par.width())
            ph = float(par.height())
            margin = self.outer_radius + 18.0
            if pw >= margin * 2.2:
                cx = max(margin, min(pw - margin, cx))
            if ph >= margin * 2.2:
                cy = max(margin, min(ph - margin, cy))
        self.center_in_parent = QPointF(cx, cy)
        self.smart_shape_active = bool(smart_shape_enabled)
        self.active_slice_idx = None
        self.active_sub_item = None
        half = self._widget_size * 0.5
        self.move(int(round(cx - half)), int(round(cy - half)))
        self.show()
        self.raise_()
        self._scale_anim.stop()
        self._anim_scale = 0.82
        self._scale_anim.start()
        self.update()

    def update_pointer(self, parent_pos: QPointF) -> Optional[int]:
        dx = float(parent_pos.x() - self.center_in_parent.x())
        dy = float(parent_pos.y() - self.center_in_parent.y())
        dist = math.hypot(dx, dy)

        if dist < self.deadzone_px:
            changed = (self.active_slice_idx is not None) or (self.active_sub_item is not None)
            self.active_slice_idx = None
            self.active_sub_item = None
            if changed:
                self.update()
            return None

        # Keep locked parent slice while exploring its outer sub-ring
        if (
            dist >= self.sub_inner_radius
            and self.active_slice_idx is not None
            and PIE_MENU_SLICES[self.active_slice_idx].action_id in PIE_SUB_ACTIONS
        ):
            sub_hit = resolve_pie_sub_item(dx, dy, self.active_slice_idx, sub_inner_px=self.sub_inner_radius)
            if sub_hit is not None:
                if sub_hit != self.active_sub_item:
                    self.active_sub_item = sub_hit
                    self.update()
                return self.active_slice_idx

        new_idx = resolve_pie_slice(dx, dy, deadzone_px=self.deadzone_px)
        new_sub = (
            resolve_pie_sub_item(dx, dy, new_idx, sub_inner_px=self.sub_inner_radius)
            if new_idx is not None
            else None
        )
        if new_idx != self.active_slice_idx or new_sub != self.active_sub_item:
            self.active_slice_idx = new_idx
            self.active_sub_item = new_sub
            self.update()
        return new_idx

    def release_and_dismiss(self, parent_pos: Optional[QPointF] = None) -> Optional[PieSliceSpec]:
        if parent_pos is not None:
            self.update_pointer(parent_pos)
        chosen: Optional[PieSliceSpec] = None
        chosen_sub: Optional[PieSubItemSpec] = self.active_sub_item
        if self.active_slice_idx is not None and 0 <= self.active_slice_idx < len(PIE_MENU_SLICES):
            chosen = PIE_MENU_SLICES[self.active_slice_idx]
        self._scale_anim.stop()
        self.hide()
        self.active_slice_idx = None
        self.active_sub_item = None
        if chosen_sub is not None:
            self.action_triggered.emit(chosen_sub.action_id)
        elif chosen is not None:
            self.action_triggered.emit(chosen.action_id)
        return chosen

    def cancel(self) -> None:
        self._scale_anim.stop()
        self.active_slice_idx = None
        self.active_sub_item = None
        self.hide()

    def _draw_vector_icon(
        self,
        painter: QPainter,
        action_id: str,
        cx: float,
        cy: float,
        r: float,
        color: QColor,
    ) -> None:
        """Render crisp anti-aliased vector glyphs via QPainter (zero missing-font tofu boxes)."""
        painter.save()
        painter.translate(cx, cy)
        pen = QPen(color, max(1.6, r * 0.22), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        aid = str(action_id or "").upper()
        if aid == "PEN":
            # Diagonal pencil body + tip
            painter.drawLine(QPointF(-r * 0.75, r * 0.75), QPointF(r * 0.55, -r * 0.55))
            painter.drawLine(QPointF(r * 0.25, -r * 0.85), QPointF(r * 0.85, -r * 0.25))
            painter.drawLine(QPointF(-r * 0.75, r * 0.75), QPointF(-r * 0.95, r * 0.95))
        elif aid == "ERASER":
            # Angled eraser block + horizontal erase baseline
            path = QPainterPath()
            path.moveTo(-r * 0.55, r * 0.15)
            path.lineTo(r * 0.10, -r * 0.55)
            path.lineTo(r * 0.75, r * 0.05)
            path.lineTo(r * 0.15, r * 0.65)
            path.lineTo(-r * 0.35, r * 0.65)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawLine(QPointF(-r * 0.85, r * 0.72), QPointF(r * 0.85, r * 0.72))
        elif aid == "SELECT":
            # Crisp pointer arrow polygon
            path = QPainterPath()
            path.moveTo(-r * 0.55, -r * 0.80)
            path.lineTo(-r * 0.55, r * 0.65)
            path.lineTo(-r * 0.12, r * 0.28)
            path.lineTo(r * 0.22, r * 0.85)
            path.lineTo(r * 0.45, r * 0.72)
            path.lineTo(r * 0.12, r * 0.18)
            path.lineTo(r * 0.65, r * 0.14)
            path.closeSubpath()
            painter.setBrush(color)
            painter.drawPath(path)
        elif aid == "PAN":
            # 4-way pan cross arrows
            painter.drawLine(QPointF(-r * 0.80, 0.0), QPointF(r * 0.80, 0.0))
            painter.drawLine(QPointF(0.0, -r * 0.80), QPointF(0.0, r * 0.80))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                tip = QPointF(dx * r * 0.82, dy * r * 0.82)
                if dx != 0:
                    painter.drawLine(tip, QPointF(tip.x() - dx * r * 0.30, -r * 0.26))
                    painter.drawLine(tip, QPointF(tip.x() - dx * r * 0.30, r * 0.26))
                else:
                    painter.drawLine(tip, QPointF(-r * 0.26, tip.y() - dy * r * 0.30))
                    painter.drawLine(tip, QPointF(r * 0.26, tip.y() - dy * r * 0.30))
        elif aid == "TEXT":
            painter.setFont(QFont("Segoe UI", max(9, int(round(r * 1.45))), QFont.ExtraBold))
            painter.drawText(QRectF(-r * 1.2, -r * 1.1, r * 2.4, r * 2.2), Qt.AlignCenter, "T")
        elif aid in ("SHAPES", "TOGGLE_SMART_SHAPE"):
            path = QPainterPath()
            path.moveTo(0.0, -r * 0.82)
            path.lineTo(r * 0.82, 0.0)
            path.lineTo(0.0, r * 0.82)
            path.lineTo(-r * 0.82, 0.0)
            path.closeSubpath()
            painter.drawPath(path)
        elif aid == "UNDO":
            # Curved undo arrow
            arc_rect = QRectF(-r * 0.65, -r * 0.35, r * 1.35, r * 1.15)
            painter.drawArc(arc_rect, -35 * 16, 195 * 16)
            painter.drawLine(QPointF(-r * 0.72, -r * 0.05), QPointF(-r * 0.72, -r * 0.62))
            painter.drawLine(QPointF(-r * 0.72, -r * 0.05), QPointF(-r * 0.16, -r * 0.08))
        elif aid.startswith("COLOR:"):
            c_hex = action_id.split(":", 1)[1].strip()
            painter.setBrush(QColor(c_hex))
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
            painter.drawEllipse(QPointF(0.0, 0.0), r * 0.68, r * 0.68)
        elif aid == "HIGHLIGHTER":
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 110))
            painter.drawRoundedRect(QRectF(-r * 0.75, -r * 0.40, r * 1.50, r * 0.80), 2.5, 2.5)
        elif aid == "SHAPE:RECT":
            painter.drawRect(QRectF(-r * 0.75, -r * 0.52, r * 1.50, r * 1.04))
        elif aid == "SHAPE:CIRCLE":
            painter.drawEllipse(QPointF(0.0, 0.0), r * 0.68, r * 0.68)
        elif aid == "SHAPE:TRIANGLE":
            path = QPainterPath()
            path.moveTo(0.0, -r * 0.72)
            path.lineTo(r * 0.78, r * 0.62)
            path.lineTo(-r * 0.78, r * 0.62)
            path.closeSubpath()
            painter.drawPath(path)
        elif aid == "SHAPE:ARROW":
            painter.drawLine(QPointF(-r * 0.80, 0.0), QPointF(r * 0.78, 0.0))
            painter.drawLine(QPointF(r * 0.25, -r * 0.45), QPointF(r * 0.78, 0.0))
            painter.drawLine(QPointF(r * 0.25, r * 0.45), QPointF(r * 0.78, 0.0))
        elif aid == "STICKY":
            painter.drawRoundedRect(QRectF(-r * 0.65, -r * 0.65, r * 1.30, r * 1.30), 2.0, 2.0)
            painter.drawLine(QPointF(-r * 0.40, -r * 0.15), QPointF(r * 0.40, -r * 0.15))
            painter.drawLine(QPointF(-r * 0.40, r * 0.25), QPointF(r * 0.20, r * 0.25))
        elif aid == "NEW_PAGE":
            painter.drawRoundedRect(QRectF(-r * 0.58, -r * 0.72, r * 1.16, r * 1.44), 2.0, 2.0)
            painter.drawLine(QPointF(-r * 0.30, 0.0), QPointF(r * 0.30, 0.0))
            painter.drawLine(QPointF(0.0, -r * 0.30), QPointF(0.0, r * 0.30))
        elif aid == "CLEAR_BOARD":
            painter.drawLine(QPointF(-r * 0.65, -r * 0.45), QPointF(r * 0.65, -r * 0.45))
            painter.drawRect(QRectF(-r * 0.48, -r * 0.45, r * 0.96, r * 1.12))
            painter.drawLine(QPointF(-r * 0.20, -r * 0.68), QPointF(r * 0.20, -r * 0.68))
        elif aid == "OPEN_GESTURE_MAP":
            # Stylised open hand silhouette (palm + 5 fingers)
            painter.drawRoundedRect(QRectF(-r * 0.48, -r * 0.05, r * 0.96, r * 0.76), 3.0, 3.0)
            for fx, fy in ((-0.42, -0.52), (-0.16, -0.78), (0.12, -0.82), (0.38, -0.62)):
                painter.drawLine(QPointF(fx * r, -r * 0.05), QPointF(fx * r, fy * r))
            painter.drawLine(QPointF(-r * 0.48, r * 0.35), QPointF(-r * 0.80, r * 0.02))
        elif aid == "OPEN_AI_CHAT":
            painter.drawRoundedRect(QRectF(-r * 0.72, -r * 0.56, r * 1.44, r * 1.12), 4.0, 4.0)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(-r * 0.26, -r * 0.06), r * 0.14, r * 0.14)
            painter.drawEllipse(QPointF(r * 0.26, -r * 0.06), r * 0.14, r * 0.14)
            painter.drawLine(QPointF(-r * 0.28, r * 0.28), QPointF(r * 0.28, r * 0.28))
        elif aid == "CANCEL":
            painter.drawLine(QPointF(-r * 0.55, -r * 0.55), QPointF(r * 0.55, r * 0.55))
            painter.drawLine(QPointF(r * 0.55, -r * 0.55), QPointF(-r * 0.55, r * 0.55))
        else:
            # 4-point AI / Gesture sparkle star (CLEAR_OR_NEW / ASK_AI_REGION)
            path = QPainterPath()
            path.moveTo(0.0, -r * 0.85)
            path.quadTo(0.0, 0.0, r * 0.85, 0.0)
            path.quadTo(0.0, 0.0, 0.0, r * 0.85)
            path.quadTo(0.0, 0.0, -r * 0.85, 0.0)
            path.quadTo(0.0, 0.0, 0.0, -r * 0.85)
            path.closeSubpath()
            painter.setBrush(color)
            painter.drawPath(path)

        painter.restore()

    def _draw_annular_wedge(
        self,
        painter: QPainter,
        r_in: float,
        r_out: float,
        start_deg: float,
        span_deg: float,
        fill: QColor,
        border: QColor,
        pen_w: float,
    ) -> None:
        outer_rect = QRectF(-r_out, -r_out, 2.0 * r_out, 2.0 * r_out)
        inner_rect = QRectF(-r_in, -r_in, 2.0 * r_in, 2.0 * r_in)
        path = QPainterPath()
        start_rad = math.radians(start_deg)
        path.moveTo(r_in * math.cos(start_rad), -r_in * math.sin(start_rad))
        path.arcTo(outer_rect, start_deg, span_deg)
        end_rad = math.radians(start_deg + span_deg)
        path.lineTo(r_in * math.cos(end_rad), -r_in * math.sin(end_rad))
        path.arcTo(inner_rect, start_deg + span_deg, -span_deg)
        path.closeSubpath()
        painter.setBrush(fill)
        painter.setPen(QPen(border, pen_w))
        painter.drawPath(path)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() * 0.5
        cy = self.height() * 0.5
        painter.translate(cx, cy)
        painter.scale(self._anim_scale, self._anim_scale)

        r_out = self.outer_radius
        r_in = self.inner_radius

        # Glassmorphic frosted halo backdrop
        halo = QRadialGradient(QPointF(0.0, 0.0), r_out + 26.0)
        if self.dark_mode:
            halo.setColorAt(0.0, QColor(15, 23, 42, 224))
            halo.setColorAt(0.84, QColor(15, 23, 42, 188))
            halo.setColorAt(1.0, QColor(15, 23, 42, 0))
        else:
            halo.setColorAt(0.0, QColor(248, 250, 252, 235))
            halo.setColorAt(0.84, QColor(226, 232, 240, 200))
            halo.setColorAt(1.0, QColor(226, 232, 240, 0))
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0.0, 0.0), r_out + 26.0, r_out + 26.0)

        # Draw 8 main annular wedge sectors
        for spec in PIE_MENU_SLICES:
            is_sel = self.active_slice_idx == spec.index
            cw_deg = spec.index * 45.0
            qt_center_deg = 90.0 - cw_deg
            start_deg = qt_center_deg - 20.5
            span_deg = 41.0

            accent = QColor(spec.accent_hex)
            if is_sel:
                fill = QColor(accent.red(), accent.green(), accent.blue(), 208)
                border = QColor("#FFFFFF") if self.dark_mode else QColor("#0F172A")
                pen_w = 2.2
            else:
                fill = QColor(30, 41, 59, 208) if self.dark_mode else QColor(255, 255, 255, 226)
                border = QColor(71, 85, 105, 180) if self.dark_mode else QColor(148, 163, 184, 210)
                pen_w = 1.1

            self._draw_annular_wedge(painter, r_in, r_out, start_deg, span_deg, fill, border, pen_w)

            # Icon + crisp short Vietnamese label inside sector midpoint
            mid_rad = math.radians(qt_center_deg)
            label_r = 0.53 * (r_in + r_out)
            lx = label_r * math.cos(mid_rad)
            ly = -label_r * math.sin(mid_rad)

            has_sub = spec.action_id in PIE_SUB_ACTIONS
            icon_col = QColor("#0F172A") if is_sel else accent
            self._draw_vector_icon(painter, spec.action_id, lx, ly - 10.0, 9.5, icon_col)

            short_txt = spec.short_label + (" ›" if has_sub else "")
            painter.setPen(
                QColor("#0F172A") if is_sel else (QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(
                QRectF(lx - 44.0, ly + 2.0, 88.0, 20.0),
                Qt.AlignCenter,
                short_txt,
            )

        # Outer Sub-Ring when hovering a sector with sub-actions (Pen / Shapes / Gestures-AI)
        if self.active_slice_idx is not None:
            parent_spec = PIE_MENU_SLICES[self.active_slice_idx]
            sub_items = PIE_SUB_ACTIONS.get(parent_spec.action_id)
            if sub_items:
                n = len(sub_items)
                step_deg = 22.0
                total_span = step_deg * n
                parent_cw = parent_spec.index * 45.0
                for s_i, sub in enumerate(sub_items):
                    item_cw = parent_cw - 0.5 * total_span + (s_i + 0.5) * step_deg
                    qt_c = 90.0 - item_cw
                    is_sub_sel = self.active_sub_item == sub
                    sub_acc = QColor(sub.accent_hex)
                    if is_sub_sel:
                        s_fill = QColor(sub_acc.red(), sub_acc.green(), sub_acc.blue(), 225)
                        s_border = QColor("#FFFFFF")
                        s_pw = 2.0
                    else:
                        s_fill = QColor(15, 23, 42, 224) if self.dark_mode else QColor(241, 245, 249, 236)
                        s_border = QColor(sub_acc.red(), sub_acc.green(), sub_acc.blue(), 170)
                        s_pw = 1.2
                    self._draw_annular_wedge(
                        painter,
                        self.sub_inner_radius,
                        self.sub_outer_radius,
                        qt_c - (step_deg * 0.46),
                        step_deg * 0.92,
                        s_fill,
                        s_border,
                        s_pw,
                    )
                    rad_s = math.radians(qt_c)
                    sr_mid = 0.5 * (self.sub_inner_radius + self.sub_outer_radius)
                    sx = sr_mid * math.cos(rad_s)
                    sy = -sr_mid * math.sin(rad_s)
                    self._draw_vector_icon(
                        painter,
                        sub.action_id,
                        sx,
                        sy - 7.5,
                        6.8,
                        QColor("#0F172A") if is_sub_sel else sub_acc,
                    )
                    painter.setPen(
                        QColor("#0F172A") if is_sub_sel else (QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
                    )
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(QRectF(sx - 34.0, sy + 2.0, 68.0, 15.0), Qt.AlignCenter, sub.short_label)

        # Center Hub (Deadzone Cancel / Active Function Name Display)
        hub_r = self.inner_radius - 3.0
        active_accent = (
            QColor(self.active_sub_item.accent_hex)
            if self.active_sub_item is not None
            else (QColor(PIE_MENU_SLICES[self.active_slice_idx].accent_hex) if self.active_slice_idx is not None else QColor("#94A3B8"))
        )
        hub_fill = (
            QColor(15, 23, 42, 244) if self.dark_mode else QColor(255, 255, 255, 248)
        )
        painter.setBrush(hub_fill)
        painter.setPen(QPen(active_accent, 2.2))
        painter.drawEllipse(QPointF(0.0, 0.0), hub_r, hub_r)

        if self.active_sub_item is not None:
            hub_action = self.active_sub_item.action_id
            hub_lbl = self.active_sub_item.short_label
            full_banner = self.active_sub_item.full_label_vi
        elif self.active_slice_idx is not None:
            spec = PIE_MENU_SLICES[self.active_slice_idx]
            hub_action = spec.action_id
            hub_lbl = spec.short_label
            full_banner = spec.full_label_vi
            if spec.action_id in PIE_SUB_ACTIONS:
                full_banner += "  •  Kéo ra ngoài chọn mục con"
        else:
            hub_action = "CANCEL"
            hub_lbl = "Hủy"
            full_banner = ""

        self._draw_vector_icon(painter, hub_action, 0.0, -10.5, 8.2, active_accent)

        painter.setPen(QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.drawText(QRectF(-hub_r + 3.0, 1.0, 2.0 * hub_r - 6.0, 18.0), Qt.AlignCenter, hub_lbl)

        # Bottom glassmorphic description pill
        if full_banner:
            pill_w = min(float(self._widget_size) - 16.0, max(180.0, len(full_banner) * 6.9 + 28.0))
            pill_y = self.sub_outer_radius + 10.0 if (self.active_slice_idx is not None and PIE_MENU_SLICES[self.active_slice_idx].action_id in PIE_SUB_ACTIONS) else (r_out + 14.0)
            pill_rect = QRectF(-0.5 * pill_w, pill_y, pill_w, 26.0)
            painter.setBrush(QColor(15, 23, 42, 240) if self.dark_mode else QColor(255, 255, 255, 244))
            painter.setPen(QPen(active_accent, 1.5))
            painter.drawRoundedRect(pill_rect, 13.0, 13.0)
            painter.setPen(QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
            painter.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
            painter.drawText(pill_rect, Qt.AlignCenter, full_banner)
