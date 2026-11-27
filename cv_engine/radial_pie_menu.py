#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender/Maya-Style Right-Click Radial Pie Menu (`cv_engine/radial_pie_menu.py`)
for AirTeach AI SmartboardCanvas.

Ergonomic 8-slice layout clockwise starting from Top (North):
  0 (N,   -90°): Bút Vẽ (Pen)
  1 (NE,  -45°): Bút Nhớ (Highlighter)
  2 (E,     0°): Di Chuyển Bảng (Pan)
  3 (SE,  +45°): Hỏi Trợ Lý AI (Chụp Vùng)
  4 (S,   +90°): Tẩy (Eraser)
  5 (SW, +135°): Hoàn Tác (Undo)
  6 (W,  ±180°): Con Trỏ Laser
  7 (NW, -135°): Nắn Hình Thông Minh (Smart Shape)

Center deadzone radius (default 20px) cancels selection cleanly on release.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QVariantAnimation, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget


DEFAULT_PIE_DEADZONE_PX = 20.0
DEFAULT_PIE_OUTER_RADIUS_PX = 136.0
DEFAULT_PIE_INNER_RADIUS_PX = 34.0


@dataclass(frozen=True)
class PieSliceSpec:
    index: int
    action_id: str
    direction: str
    short_label: str
    full_label_vi: str
    glyph: str
    accent_hex: str


PIE_MENU_SLICES: Tuple[PieSliceSpec, ...] = (
    PieSliceSpec(0, "PEN", "N", "Bút Vẽ", "Bút Vẽ (Pen)", "✎", "#00F5D4"),
    PieSliceSpec(1, "HIGHLIGHTER", "NE", "Bút Nhớ", "Bút Nhớ (Highlighter)", "🖍", "#FEE440"),
    PieSliceSpec(2, "PAN", "E", "Di Chuyển", "Di Chuyển Bảng (Pan)", "✥", "#38BDF8"),
    PieSliceSpec(3, "ASK_AI_REGION", "SE", "Hỏi AI", "Hỏi Trợ Lý AI (Chụp Vùng)", "✦", "#A855F7"),
    PieSliceSpec(4, "ERASER", "S", "Tẩy", "Tẩy (Eraser)", "⌫", "#F43F5E"),
    PieSliceSpec(5, "UNDO", "SW", "Hoàn Tác", "Hoàn Tác (Undo)", "↶", "#FB923C"),
    PieSliceSpec(6, "LASER", "W", "Laser", "Con Trỏ Laser", "◉", "#EF4444"),
    PieSliceSpec(7, "TOGGLE_SMART_SHAPE", "NW", "Nắn Hình", "Nắn Hình Thông Minh", "◇", "#10B981"),
)


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


class RadialPieMenuOverlay(QWidget):
    """Glassmorphic 8-slice Radial Pie Menu centered over the right-click press point."""

    action_triggered = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.deadzone_px: float = DEFAULT_PIE_DEADZONE_PX
        self.outer_radius: float = DEFAULT_PIE_OUTER_RADIUS_PX
        self.inner_radius: float = DEFAULT_PIE_INNER_RADIUS_PX
        self._widget_size: int = int((self.outer_radius + 78.0) * 2.0)
        self.setFixedSize(self._widget_size, self._widget_size)

        self.center_in_parent: QPointF = QPointF(0.0, 0.0)
        self.active_slice_idx: Optional[int] = None
        self.smart_shape_active: bool = True
        self._anim_scale: float = 1.0

        self._scale_anim = QVariantAnimation(self)
        self._scale_anim.setDuration(110)
        self._scale_anim.setStartValue(0.86)
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

    def open_at(self, parent_pos: QPointF, *, smart_shape_enabled: bool = True) -> None:
        self.center_in_parent = QPointF(parent_pos)
        self.smart_shape_active = bool(smart_shape_enabled)
        self.active_slice_idx = None
        half = self._widget_size * 0.5
        self.move(int(round(parent_pos.x() - half)), int(round(parent_pos.y() - half)))
        self.show()
        self.raise_()
        self._scale_anim.stop()
        self._anim_scale = 0.86
        self._scale_anim.start()
        self.update()

    def update_pointer(self, parent_pos: QPointF) -> Optional[int]:
        dx = float(parent_pos.x() - self.center_in_parent.x())
        dy = float(parent_pos.y() - self.center_in_parent.y())
        new_idx = resolve_pie_slice(dx, dy, deadzone_px=self.deadzone_px)
        if new_idx != self.active_slice_idx:
            self.active_slice_idx = new_idx
            self.update()
        return new_idx

    def release_and_dismiss(self, parent_pos: Optional[QPointF] = None) -> Optional[PieSliceSpec]:
        if parent_pos is not None:
            self.update_pointer(parent_pos)
        chosen: Optional[PieSliceSpec] = None
        if self.active_slice_idx is not None and 0 <= self.active_slice_idx < len(PIE_MENU_SLICES):
            chosen = PIE_MENU_SLICES[self.active_slice_idx]
        self._scale_anim.stop()
        self.hide()
        self.active_slice_idx = None
        if chosen is not None:
            self.action_triggered.emit(chosen.action_id)
        return chosen

    def cancel(self) -> None:
        self._scale_anim.stop()
        self.active_slice_idx = None
        self.hide()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() * 0.5
        cy = self.height() * 0.5
        painter.translate(cx, cy)
        painter.scale(self._anim_scale, self._anim_scale)

        r_out = self.outer_radius
        r_in = self.inner_radius

        # Subtle backdrop halo
        halo = QRadialGradient(QPointF(0.0, 0.0), r_out + 24.0)
        halo.setColorAt(0.0, QColor(15, 23, 42, 230))
        halo.setColorAt(0.85, QColor(15, 23, 42, 200))
        halo.setColorAt(1.0, QColor(15, 23, 42, 0))
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0.0, 0.0), r_out + 24.0, r_out + 24.0)

        outer_rect = QRectF(-r_out, -r_out, 2.0 * r_out, 2.0 * r_out)
        inner_rect = QRectF(-r_in, -r_in, 2.0 * r_in, 2.0 * r_in)

        # Draw 8 annular wedge sectors
        for spec in PIE_MENU_SLICES:
            is_sel = self.active_slice_idx == spec.index
            cw_deg = spec.index * 45.0
            qt_center_deg = 90.0 - cw_deg
            start_deg = qt_center_deg - 20.5
            span_deg = 41.0

            path = QPainterPath()
            start_rad = math.radians(start_deg)
            path.moveTo(r_in * math.cos(start_rad), -r_in * math.sin(start_rad))
            path.arcTo(outer_rect, start_deg, span_deg)
            end_rad = math.radians(start_deg + span_deg)
            path.lineTo(r_in * math.cos(end_rad), -r_in * math.sin(end_rad))
            path.arcTo(inner_rect, start_deg + span_deg, -span_deg)
            path.closeSubpath()

            accent = QColor(spec.accent_hex)
            if is_sel:
                fill = QColor(accent.red(), accent.green(), accent.blue(), 195)
                border = QColor("#FFFFFF")
                pen_w = 2.2
            else:
                fill = QColor(30, 41, 59, 218)
                border = QColor(71, 85, 105, 190)
                pen_w = 1.1

            painter.setBrush(fill)
            painter.setPen(QPen(border, pen_w))
            painter.drawPath(path)

            # Icon + crisp short Vietnamese label inside sector midpoint
            mid_rad = math.radians(qt_center_deg)
            label_r = 0.52 * (r_in + r_out)
            lx = label_r * math.cos(mid_rad)
            ly = -label_r * math.sin(mid_rad)

            painter.setPen(QColor("#0F172A") if is_sel else accent)
            painter.setFont(QFont("Segoe UI Symbol", 14, QFont.Bold))
            painter.drawText(
                QRectF(lx - 28.0, ly - 22.0, 56.0, 24.0),
                Qt.AlignCenter,
                spec.glyph,
            )

            short_txt = spec.short_label
            if spec.action_id == "TOGGLE_SMART_SHAPE":
                short_txt = "Nắn: BẬT" if self.smart_shape_active else "Nắn: TẮT"
            painter.setPen(QColor("#0F172A") if is_sel else QColor("#F8FAFC"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(
                QRectF(lx - 42.0, ly + 2.0, 84.0, 20.0),
                Qt.AlignCenter,
                short_txt,
            )

        # Center deadzone hub (Cancel indicator)
        hub_col = QColor(51, 65, 85, 235) if self.active_slice_idx is None else QColor(15, 23, 42, 245)
        hub_border = QColor("#94A3B8") if self.active_slice_idx is None else QColor(PIE_MENU_SLICES[self.active_slice_idx].accent_hex)
        painter.setBrush(hub_col)
        painter.setPen(QPen(hub_border, 2.0))
        painter.drawEllipse(QPointF(0.0, 0.0), self.deadzone_px + 8.0, self.deadzone_px + 8.0)

        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(
            QRectF(-26.0, -14.0, 52.0, 28.0),
            Qt.AlignCenter,
            "Hủy" if self.active_slice_idx is None else "✓",
        )

        # Top/Bottom floating full Vietnamese title banner for highlighted slice
        if self.active_slice_idx is not None:
            spec = PIE_MENU_SLICES[self.active_slice_idx]
            title_txt = spec.full_label_vi
            if spec.action_id == "TOGGLE_SMART_SHAPE":
                title_txt += f" ({'Đang BẬT' if self.smart_shape_active else 'Đang TẮT'})"
            pill_w = max(172.0, len(title_txt) * 7.6 + 26.0)
            pill_y = r_out + 12.0
            pill_rect = QRectF(-0.5 * pill_w, pill_y, pill_w, 28.0)
            painter.setBrush(QColor(15, 23, 42, 245))
            painter.setPen(QPen(QColor(spec.accent_hex), 1.6))
            painter.drawRoundedRect(pill_rect, 14.0, 14.0)
            painter.setPen(QColor("#F8FAFC"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(pill_rect, Qt.AlignCenter, title_txt)
