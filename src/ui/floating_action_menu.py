"""
AirTeach AI - Floating Action Menu
====================================
In-air gesture-driven action menu rendered directly on the QGraphicsScene.
Supports:
  - Radial Wheel (Pie Menu) for quick muscle-memory selection
  - Floating Quick Panel (Vertical/Grid card)
  - Hover tracking, Dwell progress visualization (0.0 to 1.0)
  - Air Tap & Pinch Click instant activation
  - Submenus (Pens, Colors, Widths, Eraser tools, Shapes)
  - Automatic boundary clamping to keep menu within canvas bounds
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsObject

logger = logging.getLogger(__name__)


class MenuLayoutMode(Enum):
    RADIAL_WHEEL = "radial_wheel"
    FLOATING_PANEL = "floating_panel"


@dataclass
class MenuItem:
    """An item in the action menu."""
    item_id: str
    label: str
    icon_text: str = ""
    command: str = ""                     # e.g. "tool", "color", "width", "eraser", "action", "submenu"
    value: Any = None                     # payload (e.g. "pen", "#1a73e8", 4.0, "stroke", "undo")
    submenu_id: Optional[str] = None      # If set, clicking opens submenu
    color: Optional[str] = None           # Optional accent color hex
    enabled: bool = True


# Predefined Menu Definitions
ROOT_MENU_ITEMS = [
    MenuItem("pen", "Bút vẽ", "✏️", command="submenu", submenu_id="pen_menu"),
    MenuItem("eraser", "Tẩy xóa", "🧹", command="submenu", submenu_id="eraser_menu"),
    MenuItem("laser", "Laser", "🔴", command="tool", value="laser"),
    MenuItem("shapes", "Hình học", "📐", command="submenu", submenu_id="shapes_menu"),
    MenuItem("stem", "STEM", "🎓", command="submenu", submenu_id="stem_menu"),
    MenuItem("select", "Chọn & AI", "🔍", command="tool", value="select"),
    MenuItem("undo", "Hoàn tác", "↩️", command="action", value="undo"),
    MenuItem("redo", "Làm lại", "↪️", command="action", value="redo"),
    MenuItem("close", "Đóng", "✕", command="action", value="close"),
]

PEN_MENU_ITEMS = [
    MenuItem("pen_black", "Đen", "●", command="color", value="#202124", color="#202124"),
    MenuItem("pen_blue", "Xanh lam", "●", command="color", value="#1a73e8", color="#1a73e8"),
    MenuItem("pen_red", "Đỏ", "●", command="color", value="#ea4335", color="#ea4335"),
    MenuItem("pen_green", "Xanh lá", "●", command="color", value="#34a853", color="#34a853"),
    MenuItem("pen_yellow", "Vàng", "●", command="color", value="#f9ab00", color="#f9ab00"),
    MenuItem("pen_chalk_white", "Phấn trắng", "●", command="color", value="#ffffff", color="#ffffff"),
    MenuItem("pen_chalk_yellow", "Phấn vàng", "●", command="color", value="#fff275", color="#fff275"),
    MenuItem("pen_chalk_cyan", "Phấn xanh", "●", command="color", value="#70d6ff", color="#70d6ff"),
    MenuItem("width_thin", "Nét mảnh", "─", command="width", value=2.0),
    MenuItem("width_med", "Nét vừa", "━", command="width", value=4.0),
    MenuItem("width_thick", "Nét đậm", "█", command="width", value=8.0),
    MenuItem("back", "Quay lại", "⬅", command="submenu", submenu_id="root"),
]

ERASER_MENU_ITEMS = [
    MenuItem("eraser_stroke", "Tẩy nét", "✏️✕", command="eraser", value="stroke"),
    MenuItem("eraser_object", "Xóa vật thể", "🎯✕", command="eraser", value="object"),
    MenuItem("eraser_area", "Xóa vùng", "⬚✕", command="eraser", value="area"),
    MenuItem("eraser_clear", "Xóa toàn bộ", "🗑️", command="eraser", value="clear_page"),
    MenuItem("back", "Quay lại", "⬅", command="submenu", submenu_id="root"),
]

SHAPES_MENU_ITEMS = [
    MenuItem("shape_line", "Đường thẳng", "📏", command="shape", value="line"),
    MenuItem("shape_arrow", "Mũi tên", "➡️", command="shape", value="arrow"),
    MenuItem("shape_circle", "Hình tròn", "⭕", command="shape", value="circle"),
    MenuItem("shape_rect", "Chữ nhật", "▭", command="shape", value="rectangle"),
    MenuItem("shape_triangle", "Tam giác", "🔺", command="shape", value="triangle"),
    MenuItem("shape_free", "Vẽ tự do", "〰️", command="shape", value="freeform"),
    MenuItem("back", "Quay lại", "⬅", command="submenu", submenu_id="root"),
]

STEM_MENU_ITEMS = [
    MenuItem("stem_oxy", "Hệ Oxy", "📐", command="stem", value="oxy"),
    MenuItem("stem_func", "Đồ thị f(x)", "📈", command="stem", value="function"),
    MenuItem("stem_force", "Vectơ Lực", "➡️", command="stem", value="force"),
    MenuItem("stem_circuit", "Mạch điện", "⚡", command="stem", value="circuit"),
    MenuItem("stem_optics", "Quang hình", "🔍", command="stem", value="optics"),
    MenuItem("stem_chem", "Công thức Hóa", "🧪", command="stem", value="chem_formula"),
    MenuItem("stem_benzene", "Vòng Benzene", "⌬", command="stem", value="benzene"),
    MenuItem("stem_molecule", "Phân tử 3D", "⚛️", command="stem", value="molecule"),
    MenuItem("stem_bio", "Chú thích Sinh", "📍", command="stem", value="biology_callout"),
    MenuItem("stem_dna", "Xoắn kép DNA", "🧬", command="stem", value="dna"),
    MenuItem("stem_punnett", "Khung Punnett", "▦", command="stem", value="punnett"),
    MenuItem("stem_flow", "Thuật toán", "🔄", command="stem", value="flowchart"),
    MenuItem("stem_code", "Khối Code", "💻", command="stem", value="code_block"),
    MenuItem("stem_logic", "Cổng Logic", "🔲", command="stem", value="logic_gate"),
    MenuItem("stem_arrow", "Nối Luồng", "➡️", command="stem", value="flow_arrow"),
    MenuItem("stem_ruler", "Thước kẻ", "📏", command="instrument", value="ruler"),
    MenuItem("stem_protractor", "Thước đo độ", "🧭", command="instrument", value="protractor"),
    MenuItem("stem_compass", "Compa", "📐", command="instrument", value="compass"),
    MenuItem("stem_angle_tool", "Đo góc động", "📐", command="instrument", value="angle_tool"),
    MenuItem("stem_triangle_tool", "Tam giác động", "📐", command="instrument", value="triangle_tool"),
    MenuItem("stem_function_plotter", "Khảo sát f(x)", "📈", command="instrument", value="function_plotter"),
    MenuItem("stem_inclined_plane", "Hệ lực dốc", "⚡", command="instrument", value="inclined_plane"),
    MenuItem("stem_spotlight", "Đèn rọi", "🔦", command="tool", value="spotlight"),
    MenuItem("stem_loupe", "Kính lúp", "🔍", command="tool", value="loupe"),
    MenuItem("stem_timer", "Đồng hồ hẹn giờ", "⏱️", command="instrument", value="timer"),
    MenuItem("stem_metronome", "Máy đếm nhịp", "🎵", command="instrument", value="metronome"),
    MenuItem("stem_coord_hud", "Tọa độ Oxy", "📍", command="action", value="toggle_coord_hud"),
    MenuItem("stem_minimap", "Bản đồ", "🗺️", command="action", value="toggle_minimap"),
    MenuItem("stem_laser_color", "Màu Laser", "🔴", command="action", value="cycle_laser_color"),
    MenuItem("stem_desktop", "Vẽ Desktop", "🖥️", command="action", value="toggle_desktop"),
    MenuItem("stem_board_theme", "Đổi Bảng", "🎨", command="action", value="toggle_board_theme"),
    MenuItem("stem_export_img", "Xuất Ảnh", "🖼️", command="action", value="export_image"),
    MenuItem("back", "Quay lại", "⬅", command="submenu", submenu_id="root"),
]


class FloatingActionMenu(QGraphicsObject):
    """
    QGraphicsObject representing a non-modal in-air floating menu.
    Positioned directly on QGraphicsScene (Z=50).

    Signals:
        action_triggered(str, object): (command, value) emitted when an item is chosen.
        menu_closed(): emitted when the menu is closed.
    """

    action_triggered = Signal(str, object)
    menu_closed = Signal()

    def __init__(
        self,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        layout_mode: MenuLayoutMode = MenuLayoutMode.RADIAL_WHEEL,
    ) -> None:
        super().__init__()
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.layout_mode = layout_mode

        # Menus dictionary
        self._menus: dict[str, list[MenuItem]] = {
            "root": list(ROOT_MENU_ITEMS),
            "pen_menu": list(PEN_MENU_ITEMS),
            "eraser_menu": list(ERASER_MENU_ITEMS),
            "shapes_menu": list(SHAPES_MENU_ITEMS),
            "stem_menu": list(STEM_MENU_ITEMS),
        }
        self._current_menu_id = "root"
        self._items = self._menus[self._current_menu_id]

        # Geometry dimensions
        self.r_inner = 45.0
        self.r_outer = 155.0
        self.panel_width = 240.0
        self.panel_item_height = 40.0

        # State
        self._hovered_index: Optional[int] = None
        self._hovered_center: bool = False
        self._dwell_progress: float = 0.0  # 0.0 to 1.0
        self._is_active: bool = False

        self.setZValue(50)
        self.setVisible(False)

    # ── QGraphicsItem overrides ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        if self.layout_mode == MenuLayoutMode.RADIAL_WHEEL:
            margin = 35.0
            r = self.r_outer + margin
            return QRectF(-r, -r, 2 * r, 2 * r)
        else:
            h = len(self._items) * self.panel_item_height + 40.0
            w = self.panel_width
            return QRectF(-w / 2, -h / 2, w, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self.isVisible():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.layout_mode == MenuLayoutMode.RADIAL_WHEEL:
            self._paint_radial_wheel(painter)
        else:
            self._paint_floating_panel(painter)

        painter.restore()

    # ── Radial Wheel Rendering ────────────────────────────────────────────────

    def _paint_radial_wheel(self, painter: QPainter) -> None:
        n = len(self._items)
        if n == 0:
            return

        angle_step = 360.0 / n
        start_offset = -90.0 - (angle_step / 2.0)  # Center first item at top

        # Outer background glow / halo
        halo_rect = QRectF(-self.r_outer - 8, -self.r_outer - 8, 2 * (self.r_outer + 8), 2 * (self.r_outer + 8))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(15, 23, 42, 60)))
        painter.drawEllipse(halo_rect)

        # Draw wedge sectors
        for i, item in enumerate(self._items):
            start_ang = start_offset + i * angle_step
            is_hovered = (self._hovered_index == i)

            path = QPainterPath()
            outer_r = self.r_outer + (10.0 if is_hovered else 0.0)

            # Build annular sector path
            # Outer arc
            outer_rect = QRectF(-outer_r, -outer_r, 2 * outer_r, 2 * outer_r)
            inner_rect = QRectF(-self.r_inner, -self.r_inner, 2 * self.r_inner, 2 * self.r_inner)
            path.arcMoveTo(outer_rect, start_ang)
            path.arcTo(outer_rect, start_ang, angle_step)
            path.arcTo(inner_rect, start_ang + angle_step, -angle_step)
            path.closeSubpath()

            # Fill color
            if is_hovered:
                bg_col = QColor("#2563eb")  # Accent blue
                border_pen = QPen(QColor("#60a5fa"), 2.5)
            else:
                bg_col = QColor(30, 41, 59, 230)  # Slate dark glass
                border_pen = QPen(QColor(71, 85, 105, 180), 1.2)

            painter.setPen(border_pen)
            painter.setBrush(QBrush(bg_col))
            painter.drawPath(path)

            # Draw Dwell progress arc if hovered
            if is_hovered and self._dwell_progress > 0.01:
                dwell_ang = angle_step * min(1.0, max(0.0, self._dwell_progress))
                dwell_pen = QPen(QColor("#38bdf8"), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                painter.setPen(dwell_pen)
                dwell_path = QPainterPath()
                dwell_path.arcMoveTo(outer_rect, start_ang)
                dwell_path.arcTo(outer_rect, start_ang, dwell_ang)
                painter.drawPath(dwell_path)

            # Item text & icon
            mid_ang_deg = start_ang + angle_step / 2.0
            mid_rad = math.radians(mid_ang_deg)
            text_r = (self.r_inner + outer_r) / 2.0
            tx = text_r * math.cos(mid_rad)
            ty = -text_r * math.sin(mid_rad)  # Screen coordinates y inverted

            # Draw icon & label
            painter.setPen(QPen(QColor("#ffffff" if is_hovered else "#e2e8f0")))
            icon_font = QFont("Segoe UI Emoji", 14, QFont.Weight.Bold)
            painter.setFont(icon_font)
            painter.drawText(QRectF(tx - 35, ty - 20, 70, 24), Qt.AlignmentFlag.AlignCenter, item.icon_text)

            label_font = QFont("Segoe UI", 9, QFont.Weight.Medium)
            painter.setFont(label_font)
            painter.drawText(QRectF(tx - 40, ty + 2, 80, 18), Qt.AlignmentFlag.AlignCenter, item.label)

        # Center Button (Back / Close / Title)
        center_rect = QRectF(-self.r_inner + 4, -self.r_inner + 4, 2 * (self.r_inner - 4), 2 * (self.r_inner - 4))
        if self._hovered_center:
            c_bg = QColor("#dc2626") if self._current_menu_id == "root" else QColor("#3b82f6")
            c_pen = QPen(QColor("#ffffff"), 2.0)
        else:
            c_bg = QColor(15, 23, 42, 245)
            c_pen = QPen(QColor(100, 116, 139, 200), 1.5)

        painter.setPen(c_pen)
        painter.setBrush(QBrush(c_bg))
        painter.drawEllipse(center_rect)

        # Center dwell progress ring
        if self._hovered_center and self._dwell_progress > 0.01:
            ring_pen = QPen(QColor("#f87171"), 3.5)
            painter.setPen(ring_pen)
            ring_rect = QRectF(-self.r_inner + 1, -self.r_inner + 1, 2 * (self.r_inner - 1), 2 * (self.r_inner - 1))
            painter.drawArc(ring_rect, int(90 * 16), int(-360 * self._dwell_progress * 16))

        # Center label
        center_text = "✕ Đóng" if self._current_menu_id == "root" else "⬅ Về"
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(center_rect, Qt.AlignmentFlag.AlignCenter, center_text)

    # ── Floating Panel Rendering ──────────────────────────────────────────────

    def _paint_floating_panel(self, painter: QPainter) -> None:
        n = len(self._items)
        h = n * self.panel_item_height + 40.0
        w = self.panel_width
        rect = QRectF(-w / 2, -h / 2, w, h)

        # Panel Card Background
        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))
        painter.drawRoundedRect(rect, 14, 14)

        # Header Title
        title = "MENU ĐIỀU KHIỂN" if self._current_menu_id == "root" else self._current_menu_id.upper()
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(QRectF(-w / 2 + 12, -h / 2 + 8, w - 24, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        # Item List
        start_y = -h / 2 + 32.0
        for i, item in enumerate(self._items):
            iy = start_y + i * self.panel_item_height
            item_rect = QRectF(-w / 2 + 10, iy, w - 20, self.panel_item_height - 4)
            is_hovered = (self._hovered_index == i)

            # Item Background
            if is_hovered:
                bg = QColor("#2563eb")
                painter.setPen(QPen(QColor("#60a5fa"), 1.5))
                painter.setBrush(QBrush(bg))
                painter.drawRoundedRect(item_rect, 8, 8)

                # Dwell progress fill
                if self._dwell_progress > 0.01:
                    prog_w = item_rect.width() * min(1.0, max(0.0, self._dwell_progress))
                    prog_rect = QRectF(item_rect.x(), item_rect.y(), prog_w, item_rect.height())
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(56, 189, 248, 120)))
                    painter.drawRoundedRect(prog_rect, 8, 8)
            else:
                bg = QColor(30, 41, 59, 180)
                painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0))
                painter.setBrush(QBrush(bg))
                painter.drawRoundedRect(item_rect, 8, 8)

            # Icon
            painter.setPen(QPen(QColor("#ffffff" if is_hovered else "#e2e8f0")))
            painter.setFont(QFont("Segoe UI Emoji", 12))
            painter.drawText(QRectF(item_rect.x() + 8, item_rect.y(), 28, item_rect.height()), Qt.AlignmentFlag.AlignCenter, item.icon_text)

            # Label
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            painter.drawText(QRectF(item_rect.x() + 40, item_rect.y(), item_rect.width() - 48, item_rect.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, item.label)

    # ── Interaction & Hit Testing ─────────────────────────────────────────────

    def show_at(self, x: float, y: float) -> None:
        """
        Open menu centered at (x, y) with boundary clamping to keep within canvas.
        """
        margin = self.r_outer + 20.0 if self.layout_mode == MenuLayoutMode.RADIAL_WHEEL else self.panel_width / 2.0 + 20.0
        clamped_x = max(margin, min(float(self.canvas_width) - margin, x))
        clamped_y = max(margin, min(float(self.canvas_height) - margin, y))

        self.setPos(clamped_x, clamped_y)
        self._current_menu_id = "root"
        self._items = self._menus[self._current_menu_id]
        self._hovered_index = None
        self._hovered_center = False
        self._dwell_progress = 0.0
        self._is_active = True
        self.setVisible(True)
        self.update()
        logger.info("FloatingActionMenu opened at (%.1f, %.1f) clamped to (%.1f, %.1f)", x, y, clamped_x, clamped_y)

    def close_menu(self) -> None:
        """Close and hide menu."""
        if not self._is_active:
            return
        self._is_active = False
        self.setVisible(False)
        self._hovered_index = None
        self._hovered_center = False
        self._dwell_progress = 0.0
        self.menu_closed.emit()
        self.update()
        logger.info("FloatingActionMenu closed.")

    def set_layout_mode(self, mode: MenuLayoutMode | str) -> None:
        """Switch between RADIAL_WHEEL and FLOATING_PANEL."""
        if isinstance(mode, str):
            self.layout_mode = MenuLayoutMode.FLOATING_PANEL if "panel" in mode.lower() else MenuLayoutMode.RADIAL_WHEEL
        else:
            self.layout_mode = mode
        self.prepareGeometryChange()
        self.update()

    def update_pointer(
        self,
        scene_x: float,
        scene_y: float,
        dwell_progress: float = 0.0,
    ) -> Optional[tuple[str, Any]]:
        """
        Update cursor pointer position relative to menu and update hover states.
        If dwell_progress >= 1.0 on a hovered item, triggers activation automatically.
        Returns:
            (command, value) if an item was triggered, else None.
        """
        if not self.isVisible():
            return None

        # Convert scene coordinates to item local coordinates
        local_pt = self.mapFromScene(QPointF(scene_x, scene_y))
        lx = local_pt.x()
        ly = local_pt.y()

        prev_idx = self._hovered_index
        prev_cnt = self._hovered_center

        if self.layout_mode == MenuLayoutMode.RADIAL_WHEEL:
            dist = math.hypot(lx, ly)
            if dist < self.r_inner:
                self._hovered_center = True
                self._hovered_index = None
            elif dist <= self.r_outer + 35.0:
                self._hovered_center = False
                n = len(self._items)
                if n > 0:
                    # In Qt screen coordinates, ly is positive downwards
                    # Angle in standard polar: atan2(-ly, lx)
                    ang_rad = math.atan2(-ly, lx)
                    ang_deg = math.degrees(ang_rad) % 360.0

                    angle_step = 360.0 / n
                    start_offset = -90.0 - (angle_step / 2.0)
                    norm_ang = (ang_deg - start_offset) % 360.0
                    idx = int(norm_ang // angle_step) % n
                    self._hovered_index = idx
                else:
                    self._hovered_index = None
            else:
                self._hovered_index = None
                self._hovered_center = False
        else:
            # Panel layout hit testing
            n = len(self._items)
            h = n * self.panel_item_height + 40.0
            w = self.panel_width
            if -w / 2 <= lx <= w / 2 and -h / 2 <= ly <= h / 2:
                start_y = -h / 2 + 32.0
                idx = int((ly - start_y) // self.panel_item_height)
                if 0 <= idx < n:
                    self._hovered_index = idx
                    self._hovered_center = False
                else:
                    self._hovered_index = None
                    self._hovered_center = False
            else:
                self._hovered_index = None
                self._hovered_center = False

        # Update dwell progress with jitter-resistant decay
        if self._hovered_index is None and not self._hovered_center:
            self._dwell_progress = 0.0
        elif prev_idx != self._hovered_index or prev_cnt != self._hovered_center:
            # Target changed: reset dwell progress
            self._dwell_progress = max(0.0, min(1.0, dwell_progress))
        else:
            # Same target: decay smoothly if tracking momentarily drops
            if dwell_progress < self._dwell_progress:
                self._dwell_progress = max(0.0, self._dwell_progress * 0.85)
            else:
                self._dwell_progress = max(0.0, min(1.0, dwell_progress))

        if prev_idx != self._hovered_index or prev_cnt != self._hovered_center:
            self.update()

        # Check dwell activation trigger
        if self._dwell_progress >= 0.99:
            return self.trigger_click()

        return None

    def trigger_click(self) -> Optional[tuple[str, Any]]:
        """
        Execute click on the currently hovered item (Air Tap / Pinch Click).
        Returns:
            (command, value) if triggered, else None.
        """
        if not self.isVisible():
            return None

        # Center click (back / close)
        if self._hovered_center:
            if self._current_menu_id == "root":
                self.close_menu()
                return ("action", "close")
            else:
                self._current_menu_id = "root"
                self._items = self._menus[self._current_menu_id]
                self._hovered_index = None
                self._hovered_center = False
                self._dwell_progress = 0.0
                self.update()
                return ("nav", "root")

        # Item click
        if self._hovered_index is not None and 0 <= self._hovered_index < len(self._items):
            item = self._items[self._hovered_index]
            logger.info("Menu item activated: %s (cmd=%s, val=%s)", item.item_id, item.command, item.value)

            # Submenu navigation
            if item.command == "submenu" and item.submenu_id:
                if item.submenu_id in self._menus:
                    self._current_menu_id = item.submenu_id
                    self._items = self._menus[self._current_menu_id]
                    self._hovered_index = None
                    self._hovered_center = False
                    self._dwell_progress = 0.0
                    self.update()
                    return ("nav", item.submenu_id)

            # Back button
            if item.item_id == "back":
                self._current_menu_id = "root"
                self._items = self._menus[self._current_menu_id]
                self._hovered_index = None
                self._hovered_center = False
                self._dwell_progress = 0.0
                self.update()
                return ("nav", "root")

            # Close action
            if item.item_id == "close" or (item.command == "action" and item.value == "close"):
                self.close_menu()
                return ("action", "close")

            # Emit action triggered
            cmd = item.command
            val = item.value
            self.action_triggered.emit(cmd, val)
            try:
                from src.utils.audio_feedback import AudioFeedbackManager
                AudioFeedbackManager.instance().play_click()
            except Exception:
                pass

            # For terminal actions (colors, eraser tools, widths), auto-close or keep open?
            # Keeping menu open for quick color/width switching or auto-closing on tool select
            if cmd in ("tool", "eraser", "action", "stem"):
                self.close_menu()

            self._dwell_progress = 0.0
            self.update()
            return (cmd, val)

        return None
