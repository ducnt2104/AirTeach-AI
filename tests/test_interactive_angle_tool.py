"""
Unit Tests for Interactive Dynamic Angle Measure & Plane Geometry Vertex Tool
=============================================================================
Covers:
1. Exact mathematical interior angle computation (cosine rule / dot product).
2. Right-angle geometric auto-snapping (90.0° tolerance check).
3. Angle taxonomy classification (Acute, Right, Obtuse, Straight).
4. Vertex manipulation and relative translation on apex B movement.
5. Control handle hit-testing and touchless air gesture interaction.
6. QPainter visual rendering of arc, right-angle square marker, and bisector degree badge.
7. Canvas stamping (vectorizing permanent angle onto LessonBoard).
8. LessonBoard and MainWindow integration (View menu action Ctrl+Shift+A, toolbar, slots).
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_angle_tool import InteractiveAngleMeasureItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_angle_mathematical_calculations(qapp):
    """Test interior angle degree calculations across various vertex configurations."""
    # Apex B at (0, 0), Arm A at (100, 0), Arm C at (0, 100) -> 90 degrees
    tool = InteractiveAngleMeasureItem(
        apex_b=(0.0, 0.0),
        arm_a=(100.0, 0.0),
        arm_c=(0.0, 100.0),
        snap_right_angle=True,
    )
    assert abs(tool.angle_degrees() - 90.0) < 1e-4
    assert tool.is_right_angle() is True
    assert tool.angle_type() == "right"
    assert tool.angle_type_vietnamese() == "Góc vuông"

    # Acute angle: 45 degrees
    tool.set_vertices(a=(100.0, 100.0), b=(0.0, 0.0), c=(100.0, 0.0))
    assert abs(tool.angle_degrees() - 45.0) < 1e-3
    assert tool.is_right_angle() is False
    assert tool.angle_type() == "acute"
    assert tool.angle_type_vietnamese() == "Góc nhọn"

    # Obtuse angle: 120 degrees
    # Arm C along x-axis (100, 0), Arm A at cos(120), sin(120) = (-50, 86.6025)
    tool.set_vertices(a=(-50.0, 86.60254), b=(0.0, 0.0), c=(100.0, 0.0))
    assert abs(tool.angle_degrees() - 120.0) < 1e-2
    assert tool.angle_type() == "obtuse"
    assert tool.angle_type_vietnamese() == "Góc tù"

    # Straight angle: 180 degrees
    tool.set_vertices(a=(-100.0, 0.0), b=(0.0, 0.0), c=(100.0, 0.0))
    assert abs(tool.angle_degrees() - 180.0) < 1e-3
    assert tool.angle_type() == "straight"
    assert tool.angle_type_vietnamese() == "Góc bẹt"


def test_right_angle_auto_snapping(qapp):
    """Test automatic geometric snapping to exactly 90.0° when near perpendicular."""
    # 89.0 degrees (within 1.8° snap tolerance)
    rad_89 = math.radians(89.0)
    tool = InteractiveAngleMeasureItem(
        apex_b=(0.0, 0.0),
        arm_a=(100.0 * math.cos(rad_89), 100.0 * math.sin(rad_89)),
        arm_c=(100.0, 0.0),
        snap_right_angle=True,
    )
    assert tool.angle_degrees() == 90.0
    assert tool.is_right_angle() is True

    # When snap is disabled, returns genuine ~89.0 degrees
    tool._snap_right_angle = False
    assert abs(tool.angle_degrees() - 89.0) < 0.1
    assert tool.is_right_angle() is False


def test_vertex_manipulation_and_apex_translation(qapp):
    """Test modifying individual vertices and relative apex translation."""
    tool = InteractiveAngleMeasureItem(
        apex_b=(500.0, 500.0),
        arm_a=(400.0, 400.0),
        arm_c=(600.0, 500.0),
    )

    angles = []
    tool.angle_changed.connect(lambda deg: angles.append(deg))

    # Move vertex A
    tool.set_vertex_pos("A", 500.0, 400.0)
    assert tool.vertex_pos("A") == QPointF(500.0, 400.0)
    assert len(angles) > 0

    # Moving apex B by (+50, +30) translates arms A and C relatively
    orig_a = QPointF(tool.vertex_pos("A"))
    orig_c = QPointF(tool.vertex_pos("C"))
    tool.set_vertex_pos("B", 550.0, 530.0)

    assert tool.vertex_pos("B") == QPointF(550.0, 530.0)
    assert tool.vertex_pos("A") == QPointF(orig_a.x() + 50.0, orig_a.y() + 30.0)
    assert tool.vertex_pos("C") == QPointF(orig_c.x() + 50.0, orig_c.y() + 30.0)


def test_handle_hit_testing_and_air_gestures(qapp):
    """Test hit testing for control handles and air drag gestures."""
    tool = InteractiveAngleMeasureItem(
        apex_b=(500.0, 500.0),
        arm_a=(400.0, 400.0),
        arm_c=(600.0, 500.0),
    )

    # Hit test apex B
    assert tool.get_handle_at(500.0, 500.0) == "B"
    assert tool.get_handle_at(505.0, 498.0) == "B"

    # Hit test arm A
    assert tool.get_handle_at(400.0, 400.0) == "A"

    # Hit test arm C
    assert tool.get_handle_at(600.0, 500.0) == "C"

    # Far away -> None
    assert tool.get_handle_at(200.0, 200.0) is None

    # Air drag vertex
    assert tool.air_drag_vertex("A", 420.0, 410.0) is True
    assert tool.vertex_pos("A") == QPointF(420.0, 410.0)
    assert tool.air_drag_vertex("INVALID", 100.0, 100.0) is False


def test_interactive_angle_rendering(qapp):
    """Test visual rendering of angle arcs, right-angle markers, and degree badges."""
    tool = InteractiveAngleMeasureItem(
        apex_b=(500.0, 500.0),
        arm_a=(400.0, 400.0),
        arm_c=(600.0, 500.0),
    )

    br = tool.boundingRect()
    assert not br.isEmpty()

    img = QImage(1200, 900, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)

    # 1. Paint general acute angle
    tool.paint(painter, None, None)

    # 2. Paint right angle (triggers square corner marker L)
    tool.set_vertices(a=(500.0, 400.0), b=(500.0, 500.0), c=(600.0, 500.0))
    assert tool.is_right_angle() is True
    tool.paint(painter, None, None)

    painter.end()


def test_canvas_stamping(qapp):
    """Test permanently stamping the interactive angle onto LessonBoard as vector items."""
    board = LessonBoard()
    tool = InteractiveAngleMeasureItem(
        apex_b=(500.0, 500.0),
        arm_a=(400.0, 400.0),
        arm_c=(600.0, 500.0),
    )

    stamped_items = tool.stamp_to_canvas(board)
    assert len(stamped_items) >= 1
    for itm in stamped_items:
        assert itm in board.items()


def test_lesson_board_angle_tool_lifecycle(qapp):
    """Test LessonBoard management of interactive angle tool."""
    board = LessonBoard()

    assert board.is_angle_tool_visible() is False
    assert board.get_angle_tool_item() is None

    # Show angle tool
    board.show_angle_tool(True)
    assert board.is_angle_tool_visible() is True
    item = board.get_angle_tool_item()
    assert item is not None
    assert item.isVisible() is True

    # Toggle angle tool
    toggled = board.toggle_angle_tool()
    assert toggled is False
    assert board.is_angle_tool_visible() is False

    # Excluded from selection
    items = board.get_items_in_rect(QRectF(0, 0, 1920, 1080))
    assert item not in items


def test_main_window_angle_tool_integration(qapp):
    """Test MainWindow actions, shortcuts, and slot integration for angle tool."""
    win = MainWindow()

    actions = win.findChildren(QAction)
    angle_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+A"]
    assert len(angle_actions) >= 1, "Expected action with shortcut Ctrl+Shift+A"

    assert win._scene.is_angle_tool_visible() is False

    is_now_vis = win._on_toggle_angle_tool()
    assert is_now_vis is True
    assert win._scene.is_angle_tool_visible() is True

    win._on_toggle_angle_tool()
    assert win._scene.is_angle_tool_visible() is False
    win.close()
