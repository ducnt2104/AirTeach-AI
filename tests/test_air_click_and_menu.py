"""
AirTeach AI - Air Click & Floating Action Menu Unit Tests
=========================================================
Tests:
  - Boundary clamping on canvas
  - Radial wheel & Floating panel hit testing
  - Dwell selection progress and activation
  - Submenu navigation (root -> pen_menu -> back)
  - Close action and signal emissions
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from src.ui.floating_action_menu import (
    FloatingActionMenu,
    MenuLayoutMode,
    ROOT_MENU_ITEMS,
    PEN_MENU_ITEMS,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_menu_boundary_clamping(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    
    # Negative position should be clamped
    menu.show_at(-100, -100)
    pos = menu.pos()
    assert pos.x() >= 155.0
    assert pos.y() >= 155.0

    # Over-canvas position should be clamped
    menu.show_at(2500, 1500)
    pos = menu.pos()
    assert pos.x() <= 1920 - 155.0
    assert pos.y() <= 1080 - 155.0
    menu.close_menu()
    assert not menu.isVisible()


def test_radial_wheel_hit_test_and_dwell(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080, layout_mode=MenuLayoutMode.RADIAL_WHEEL)
    menu.show_at(500, 500)
    assert menu.isVisible()

    triggered_actions = []
    menu.action_triggered.connect(lambda cmd, val: triggered_actions.append((cmd, val)))

    # Point directly at center (dist < r_inner=45)
    menu.update_pointer(500, 500, dwell_progress=0.0)
    assert menu._hovered_center is True
    assert menu._hovered_index is None

    # Point at top sector (scene_x=500, scene_y=400 -> local_x=0, local_y=-100)
    menu.update_pointer(500, 400, dwell_progress=0.5)
    assert menu._hovered_center is False
    assert menu._hovered_index is not None
    assert menu._dwell_progress == 0.5

    # Dwell reaches 1.0 -> should trigger activation
    result = menu.update_pointer(500, 400, dwell_progress=1.0)
    assert result is not None
    cmd, val = result
    assert cmd in ("submenu", "tool", "action", "eraser", "shape", "color", "width")


def test_submenu_navigation(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(500, 500)

    # Hover over item 0 ("pen") which has command="submenu", submenu_id="pen_menu"
    menu._hovered_index = 0
    menu._hovered_center = False

    res = menu.trigger_click()
    assert res == ("nav", "pen_menu")
    assert menu._current_menu_id == "pen_menu"
    assert len(menu._items) == len(PEN_MENU_ITEMS)

    # Hover over "back" item (last item in pen_menu)
    menu._hovered_index = len(menu._items) - 1
    res_back = menu.trigger_click()
    assert res_back == ("nav", "root")
    assert menu._current_menu_id == "root"


def test_menu_close_action(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(500, 500)

    closed_signals = []
    menu.menu_closed.connect(lambda: closed_signals.append(True))

    # Center click in root closes menu
    menu._hovered_center = True
    res = menu.trigger_click()
    assert res == ("action", "close")
    assert not menu.isVisible()
    assert len(closed_signals) == 1


def test_floating_panel_mode(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080, layout_mode=MenuLayoutMode.FLOATING_PANEL)
    menu.show_at(500, 500)

    # Update pointer to item 1
    # Panel center is at (500, 500). First item starts around -h/2 + 32
    # Local y = 0 is roughly in the middle of items
    menu.update_pointer(500, 500, dwell_progress=0.2)
    assert menu._hovered_index is not None

    menu.set_layout_mode("radial_wheel")
    assert menu.layout_mode == MenuLayoutMode.RADIAL_WHEEL


def test_menu_properties_and_state(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    assert not menu.is_active
    assert menu.current_menu_id == "root"
    assert len(menu.items) == len(ROOT_MENU_ITEMS)

    menu.show_at(600, 600)
    assert menu.is_active
    assert menu.isVisible()

    menu.close_menu()
    assert not menu.is_active
    assert not menu.isVisible()


def test_programmatic_navigation_and_custom_submenu(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(600, 600)

    # Valid navigation
    assert menu.navigate_to("shapes_menu") is True
    assert menu.current_menu_id == "shapes_menu"
    assert len(menu.items) > 0

    # Invalid navigation
    assert menu.navigate_to("nonexistent_submenu_id") is False
    assert menu.current_menu_id == "shapes_menu"

    # Register custom submenu
    from src.ui.floating_action_menu import MenuItem
    custom_items = [
        MenuItem("custom_opt1", "Tùy chọn 1", command="tool", value="opt1"),
        MenuItem("back", "Quay lại", command="submenu", submenu_id="root"),
    ]
    menu.register_submenu("custom_menu", custom_items)
    assert menu.navigate_to("custom_menu") is True
    assert menu.current_menu_id == "custom_menu"
    assert len(menu.items) == 2


def test_terminal_action_auto_closes_vs_palette_stays_open(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(600, 600)

    emitted_actions = []
    menu.action_triggered.connect(lambda cmd, val: emitted_actions.append((cmd, val)))

    # Navigate to pen menu
    menu.navigate_to("pen_menu")
    assert menu.isVisible()

    # Click a color item (palette command: "color" -> menu should stay open)
    # Pen menu item 1 is "pen_blue" (command="color", value="#1a73e8")
    menu._hovered_index = 1
    menu._hovered_center = False
    res_color = menu.trigger_click()
    assert res_color == ("color", "#1a73e8")
    assert menu.isVisible(), "Color selection should keep menu open for multi-attribute tuning"
    assert ("color", "#1a73e8") in emitted_actions

    # Navigate to root and trigger a tool command -> auto closes
    menu.navigate_to("root")
    # Item 2 in root is "laser" (command="tool", value="laser")
    menu._hovered_index = 2
    menu._hovered_center = False
    res_tool = menu.trigger_click()
    assert res_tool == ("tool", "laser")
    assert not menu.isVisible(), "Terminal tool selection must auto-close menu"
    assert ("tool", "laser") in emitted_actions


def test_center_click_submenu_navigates_back_to_root(qapp):
    menu = FloatingActionMenu(canvas_width=1920, canvas_height=1080)
    menu.show_at(600, 600)

    # In root: center click closes
    menu._hovered_center = True
    res_close = menu.trigger_click()
    assert res_close == ("action", "close")
    assert not menu.isVisible()

    # Reopen and navigate to submenu
    menu.show_at(600, 600)
    menu.navigate_to("eraser_menu")
    assert menu.current_menu_id == "eraser_menu"

    # In submenu: center click navigates back to root
    menu._hovered_center = True
    res_back = menu.trigger_click()
    assert res_back == ("nav", "root")
    assert menu.current_menu_id == "root"
    assert menu.isVisible(), "Submenu center-click back navigation must keep menu open"
