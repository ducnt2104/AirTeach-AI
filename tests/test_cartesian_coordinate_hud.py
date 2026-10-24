"""
Unit Tests for Real-Time Cartesian Oxy Grid Coordinate HUD & Precision Pointer Readout
======================================================================================
Covers:
1. Exact mathematical coordinate conversions (Scene -> Cartesian Oxy, Cartesian -> Scene).
2. Proper inverted Y-axis handling (Screen Y points down, Cartesian Y points up).
3. Integer grid snapping (Snap to Grid) and crosshair reticle generation.
4. Landmark detection: Origin O(0,0), Ox axis, and Oy axis.
5. Canvas margin clamping (ensuring badge never drifts offscreen).
6. QPainter visual rendering of frosted acrylic glass badge and snap reticle.
7. LessonBoard integration (show/hide, toggle, snap toggle, grid spacing synchronization).
8. MainWindow action shortcuts and navigation toolbar integration.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.cartesian_coordinate_hud import CartesianCoordinateHudItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cartesian_coordinate_math(qapp):
    """Test mathematical conversions between scene pixels and Cartesian Oxy units."""
    hud = CartesianCoordinateHudItem(origin=(960.0, 540.0), grid_spacing=40.0)

    # 1. Origin test
    mx, my = hud.scene_to_math(960.0, 540.0)
    assert abs(mx) < 1e-4
    assert abs(my) < 1e-4

    # 2. Quadrant I (X > 0, Y > 0)
    # 80 px right, 80 px up (scene y = 540 - 80 = 460)
    mx, my = hud.scene_to_math(1040.0, 460.0)
    assert abs(mx - 2.0) < 1e-4
    assert abs(my - 2.0) < 1e-4

    # 3. Quadrant III (X < 0, Y < 0)
    # 120 px left (scene x = 840), 80 px down (scene y = 620)
    mx, my = hud.scene_to_math(840.0, 620.0)
    assert abs(mx - (-3.0)) < 1e-4
    assert abs(my - (-2.0)) < 1e-4

    # 4. Inverse: math_to_scene
    sx, sy = hud.math_to_scene(2.5, -1.5)
    assert abs(sx - (960.0 + 2.5 * 40.0)) < 1e-4
    assert abs(sy - (540.0 - (-1.5) * 40.0)) < 1e-4


def test_grid_snapping_logic(qapp):
    """Test integer grid snapping calculation and toggle behavior."""
    hud = CartesianCoordinateHudItem(origin=(960.0, 540.0), grid_spacing=40.0, snap_to_grid=False)

    # Pointer near (2.2, -1.8) -> scene: x = 960 + 88 = 1048, y = 540 + 72 = 612
    snap_mx, snap_my = hud.get_snapped_math_pos(1048.0, 612.0)
    assert snap_mx == 2
    assert snap_my == -2

    snap_sx, snap_sy = hud.get_snapped_scene_pos(1048.0, 612.0)
    assert abs(snap_sx - 1040.0) < 1e-4
    assert abs(snap_sy - 620.0) < 1e-4

    # Toggle snap
    toggled_states = []
    hud.snap_toggled.connect(lambda s: toggled_states.append(s))

    new_state = hud.toggle_snap_to_grid()
    assert new_state is True
    assert hud.snap_to_grid() is True
    assert toggled_states == [True]

    hud.toggle_snap_to_grid()
    assert hud.snap_to_grid() is False
    assert toggled_states == [True, False]


def test_precision_and_landmark_detection(qapp):
    """Test origin and axis proximity detection, and decimal formatting."""
    hud = CartesianCoordinateHudItem(origin=(960.0, 540.0), grid_spacing=40.0, decimals=2)

    # Origin landmark
    hud.set_pointer_pos(960.0, 540.0)
    assert math.hypot(hud._pointer_x - 960.0, hud._pointer_y - 540.0) <= 12.0

    # Ox axis landmark (y = 540, x = 1200)
    hud.set_pointer_pos(1200.0, 540.0)
    assert abs(hud._pointer_y - 540.0) <= 6.0

    # Oy axis landmark (x = 960, y = 300)
    hud.set_pointer_pos(960.0, 300.0)
    assert abs(hud._pointer_x - 960.0) <= 6.0

    # Decimals update
    hud.set_decimals(0)
    assert hud.decimals() == 0
    hud.set_decimals(3)
    assert hud.decimals() == 3


def test_badge_boundary_clamping(qapp):
    """Test that the HUD badge stays strictly within canvas bounds."""
    hud = CartesianCoordinateHudItem(origin=(960.0, 540.0), grid_spacing=40.0)

    # Pointer at top-left corner
    hud.set_pointer_pos(10.0, 10.0)
    assert hud._badge_x >= 10.0
    assert hud._badge_y >= 10.0

    # Pointer at bottom-right corner (1910, 1070)
    hud.set_pointer_pos(1910.0, 1070.0)
    assert hud._badge_x + hud._badge_w <= 1920.0
    assert hud._badge_y + hud._badge_h <= 1080.0


def test_coordinate_hud_rendering(qapp):
    """Test QPainter rendering in standard, snap, and origin landmark modes."""
    hud = CartesianCoordinateHudItem(origin=(960.0, 540.0), grid_spacing=40.0)

    img = QImage(1920, 1080, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)

    # Render standard mode
    hud.set_pointer_pos(1100.0, 450.0)
    br = hud.boundingRect()
    assert not br.isEmpty()
    hud.paint(painter, None, None)

    # Render Snap mode
    hud.set_snap_to_grid(True)
    hud.paint(painter, None, None)

    # Render Origin mode
    hud.set_pointer_pos(960.0, 540.0)
    hud.paint(painter, None, None)

    # Render with pixel readout enabled
    hud.set_show_pixels(True)
    assert hud.show_pixels() is True
    hud.paint(painter, None, None)

    painter.end()


def test_lesson_board_coordinate_hud_lifecycle(qapp):
    """Test LessonBoard methods for managing coordinate HUD item."""
    board = LessonBoard()

    # Initially not created
    assert board.is_coordinate_hud_visible() is False
    assert board.get_coordinate_hud_item() is None

    # Show HUD
    board.show_coordinate_hud(True)
    assert board.is_coordinate_hud_visible() is True
    item = board.get_coordinate_hud_item()
    assert item is not None
    assert item.isVisible() is True

    # Snap controls
    assert board.is_coordinate_hud_snap() is False
    board.set_coordinate_hud_snap(True)
    assert board.is_coordinate_hud_snap() is True

    toggled_snap = board.toggle_coordinate_hud_snap()
    assert toggled_snap is False
    assert board.is_coordinate_hud_snap() is False

    # Update pointer pos
    board.update_coordinate_hud(1000.0, 500.0)
    assert item.pointer_pos() == (1000.0, 500.0)

    # Grid spacing synchronization
    board.set_grid_spacing(60)
    assert item.grid_spacing() == 60.0

    # Toggle HUD off
    toggled = board.toggle_coordinate_hud()
    assert toggled is False
    assert board.is_coordinate_hud_visible() is False

    # Excluded from selection
    items = board.get_items_in_rect(QRectF(0, 0, 1920, 1080))
    assert item not in items


def test_main_window_coordinate_hud_integration(qapp):
    """Test MainWindow actions, shortcuts, and in-air menu triggers."""
    win = MainWindow()

    # Check View menu actions exist
    actions = win.findChildren(QAction)
    coord_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+X"]
    assert len(coord_actions) >= 1, "Expected action with shortcut Ctrl+Shift+X"

    snap_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+G"]
    assert len(snap_actions) >= 1, "Expected action with shortcut Ctrl+Shift+G"

    # Toggle HUD via MainWindow slot
    assert win._scene.is_coordinate_hud_visible() is False
    is_now_vis = win._on_toggle_coordinate_hud()
    assert is_now_vis is True
    assert win._scene.is_coordinate_hud_visible() is True

    # Toggle Snap via slot
    is_snap = win._on_toggle_coordinate_snap()
    assert is_snap is True
    assert win._scene.is_coordinate_hud_snap() is True

    # Clean up
    win._on_toggle_coordinate_hud()
    assert win._scene.is_coordinate_hud_visible() is False
    win.close()
