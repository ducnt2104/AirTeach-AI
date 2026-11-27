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
    PieSliceSpec(2, "SELECT", "E", "Chọn", "Chọn / Di Chuyển Object", "⬉", "#38BDF8"),
    PieSliceSpec(3, "PAN", "SE", "Kéo Bảng", "Di Chuyển Bảng (Pan)", "✥", "#10B981"),
    PieSliceSpec(4, "TEXT", "S", "Thêm Chữ", "Thêm Hộp Văn Bản (Text)", "T", "#FACC15"),
    PieSliceSpec(5, "SHAPES", "SW", "Hình Khối", "Vẽ Hình Khối (Shapes)", "◇", "#A855F7"),
    PieSliceSpec(6, "UNDO", "W", "Hoàn Tác", "Hoàn Tác (Undo)", "↶", "#FB923C"),
    PieSliceSpec(7, "CLEAR_OR_NEW", "NW", "Xóa / Mới", "Xóa Bảng / Trang Mới", "✦", "#EF4444"),
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
        PieSubItemSpec("LASER", "Laser", "Con Trỏ Laser Thuyết Trình", "◉", "#EF4444"),
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
        self.center_in_parent = QPointF(parent_pos)
        self.smart_shape_active = bool(smart_shape_enabled)
        self.active_slice_idx = None
        self.active_sub_item = None
        half = self._widget_size * 0.5
        self.move(int(round(parent_pos.x() - half)), int(round(parent_pos.y() - half)))
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
            painter.setPen(QColor("#0F172A") if is_sel else accent)
            painter.setFont(QFont("Segoe UI Symbol", 14, QFont.Bold))
            painter.drawText(
                QRectF(lx - 28.0, ly - 22.0, 56.0, 24.0),
                Qt.AlignCenter,
                spec.glyph,
            )

            short_txt = spec.short_label + (" ▸" if has_sub else "")
            painter.setPen(
                QColor("#0F172A") if is_sel else (QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(
                QRectF(lx - 44.0, ly + 2.0, 88.0, 20.0),
                Qt.AlignCenter,
                short_txt,
            )

        # Outer Sub-Ring when hovering a sector with sub-actions (Pen / Shapes / Clear-or-New)
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
                    painter.setPen(QColor("#0F172A") if is_sub_sel else sub_acc)
                    painter.setFont(QFont("Segoe UI Symbol", 11, QFont.Bold))
                    painter.drawText(QRectF(sx - 24.0, sy - 16.0, 48.0, 18.0), Qt.AlignCenter, sub.glyph)
                    painter.setPen(
                        QColor("#0F172A") if is_sub_sel else (QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
                    )
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(QRectF(sx - 34.0, sy + 1.0, 68.0, 15.0), Qt.AlignCenter, sub.short_label)

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
            hub_glyph = self.active_sub_item.glyph
            hub_lbl = self.active_sub_item.short_label
            full_banner = self.active_sub_item.full_label_vi
        elif self.active_slice_idx is not None:
            spec = PIE_MENU_SLICES[self.active_slice_idx]
            hub_glyph = spec.glyph
            hub_lbl = spec.short_label
            full_banner = spec.full_label_vi
            if spec.action_id in PIE_SUB_ACTIONS:
                full_banner += "  •  Kéo ra ngoài chọn mục con"
        else:
            hub_glyph = "✕"
            hub_lbl = "Hủy"
            full_banner = ""

        painter.setPen(active_accent)
        painter.setFont(QFont("Segoe UI Symbol", 12, QFont.Bold))
        painter.drawText(QRectF(-hub_r + 4.0, -21.0, 2.0 * hub_r - 8.0, 20.0), Qt.AlignCenter, hub_glyph)

        painter.setPen(QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.drawText(QRectF(-hub_r + 3.0, 0.0, 2.0 * hub_r - 6.0, 18.0), Qt.AlignCenter, hub_lbl)

        # Bottom glassmorphic description pill
        if full_banner:
            pill_w = max(180.0, len(full_banner) * 6.9 + 28.0)
            pill_y = self.sub_outer_radius + 10.0 if (self.active_slice_idx is not None and PIE_MENU_SLICES[self.active_slice_idx].action_id in PIE_SUB_ACTIONS) else (r_out + 14.0)
            pill_rect = QRectF(-0.5 * pill_w, pill_y, pill_w, 26.0)
            painter.setBrush(QColor(15, 23, 42, 240) if self.dark_mode else QColor(255, 255, 255, 244))
            painter.setPen(QPen(active_accent, 1.5))
            painter.drawRoundedRect(pill_rect, 13.0, 13.0)
            painter.setPen(QColor("#F8FAFC") if self.dark_mode else QColor("#0F172A"))
            painter.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
            painter.drawText(pill_rect, Qt.AlignCenter, full_banner)
