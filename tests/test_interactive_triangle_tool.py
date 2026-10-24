"""
Unit Tests for Interactive Geometric Triangle Theorem Solver & Euler Line HUD
=============================================================================
Covers:
1. Exact metric computations: Side lengths, Perimeter, Heron's Area, Inradius, Circumradius.
2. Triangle classification: Equilateral, Isosceles, Right, Scalene.
3. Special centers (G, O, H, I) and verified Euler Line theorem: HG = 2*GO.
4. Vertex manipulation, control handle hit-testing, and air gestures.
5. Overlays toggles: Euler line, Incircle, Circumcircle, and HUD theorem card.
6. QPainter visual rendering of triangle body, medians, Euler line, and HUD card.
7. Canvas stamping (vectorizing triangle onto LessonBoard).
8. LessonBoard and MainWindow integration (View menu action Ctrl+Shift+T, toolbar, slots).
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_triangle_tool import InteractiveTriangleGeometryItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_triangle_metric_calculations(qapp):
    """Test side lengths, perimeter, area, inradius, and circumradius on a standard 3-4-5 right triangle."""
    # Vertices A(0, 80), B(0, 0), C(60, 0)
    tri = InteractiveTriangleGeometryItem(
        vert_a=(0.0, 80.0),
        vert_b=(0.0, 0.0),
        vert_c=(60.0, 0.0),
    )

    a, b, c = tri.side_lengths()
    # a = BC = 60, b = CA = 100, c = AB = 80
    assert abs(a - 60.0) < 1e-4
    assert abs(b - 100.0) < 1e-4
    assert abs(c - 80.0) < 1e-4

    # Perimeter: 60 + 100 + 80 = 240
    assert abs(tri.perimeter() - 240.0) < 1e-4

    # Area: 1/2 * 60 * 80 = 2400
    assert abs(tri.area() - 2400.0) < 1e-4

    # Inradius: r = S / p = 2400 / 120 = 20.0
    assert abs(tri.inradius() - 20.0) < 1e-4

    # Circumradius: R = abc / 4S = (60 * 100 * 80) / (4 * 2400) = 50.0
    assert abs(tri.circumradius() - 50.0) < 1e-4


def test_triangle_classification(qapp):
    """Test geometric classification of various triangle types."""
    # 1. Right triangle
    tri_right = InteractiveTriangleGeometryItem(vert_a=(0.0, 80.0), vert_b=(0.0, 0.0), vert_c=(60.0, 0.0))
    assert "right" in tri_right.classification()
    assert "vuông" in tri_right.classification_vietnamese().lower()

    # 2. Equilateral triangle: side = 100
    h = 100.0 * math.sqrt(3) / 2.0
    tri_eq = InteractiveTriangleGeometryItem(vert_a=(50.0, h), vert_b=(0.0, 0.0), vert_c=(100.0, 0.0))
    assert tri_eq.classification() == "equilateral"
    assert tri_eq.classification_vietnamese() == "Tam giác đều"

    # 3. Isosceles triangle
    tri_iso = InteractiveTriangleGeometryItem(vert_a=(50.0, 120.0), vert_b=(0.0, 0.0), vert_c=(100.0, 0.0))
    assert "isosceles" in tri_iso.classification()

    # 4. Scalene triangle
    tri_scalene = InteractiveTriangleGeometryItem(vert_a=(30.0, 110.0), vert_b=(0.0, 0.0), vert_c=(120.0, 0.0))
    assert tri_scalene.classification() == "scalene"
    assert tri_scalene.classification_vietnamese() == "Tam giác thường"


def test_special_centers_and_euler_line_theorem(qapp):
    """Verify centroid G, circumcenter O, orthocenter H, and the Euler relation HG = 2*GO."""
    # Use non-degenerate scalene triangle A(100, 300), B(20, 50), C(250, 40)
    tri = InteractiveTriangleGeometryItem(
        vert_a=(100.0, 300.0),
        vert_b=(20.0, 50.0),
        vert_c=(250.0, 40.0),
    )

    g = tri.centroid()
    o = tri.circumcenter()
    h = tri.orthocenter()

    # Centroid definition check: (A + B + C) / 3
    assert abs(g.x() - (100.0 + 20.0 + 250.0) / 3.0) < 1e-4
    assert abs(g.y() - (300.0 + 50.0 + 40.0) / 3.0) < 1e-4

    # Distance from O to all 3 vertices must equal circumradius R
    r = tri.circumradius()
    d_oa = math.hypot(tri._a.x() - o.x(), tri._a.y() - o.y())
    d_ob = math.hypot(tri._b.x() - o.x(), tri._b.y() - o.y())
    d_oc = math.hypot(tri._c.x() - o.x(), tri._c.y() - o.y())
    assert abs(d_oa - r) < 1e-3
    assert abs(d_ob - r) < 1e-3
    assert abs(d_oc - r) < 1e-3

    # Euler Theorem Invariant: Vector HG = 2 * Vector GO
    vec_hg = QPointF(g.x() - h.x(), g.y() - h.y())
    vec_go = QPointF(o.x() - g.x(), o.y() - g.y())

    # HG_x == 2 * GO_x  and  HG_y == 2 * GO_y
    assert abs(vec_hg.x() - 2.0 * vec_go.x()) < 1e-3
    assert abs(vec_hg.y() - 2.0 * vec_go.y()) < 1e-3


def test_vertex_manipulation_and_air_gestures(qapp):
    """Test modifying vertices, hit-testing control handles, and air gestures."""
    tri = InteractiveTriangleGeometryItem(
        vert_a=(500.0, 200.0),
        vert_b=(300.0, 600.0),
        vert_c=(700.0, 600.0),
    )

    signals = []
    tri.geometry_changed.connect(lambda s, p: signals.append((s, p)))

    # Move vertex A
    tri.set_vertex_pos("A", 520.0, 180.0)
    assert tri.vertex_pos("A") == QPointF(520.0, 180.0)
    assert len(signals) > 0

    # Handle hit-testing
    assert tri.get_handle_at(520.0, 180.0) == "A"
    assert tri.get_handle_at(300.0, 600.0) == "B"
    assert tri.get_handle_at(700.0, 600.0) == "C"
    assert tri.get_handle_at(100.0, 100.0) is None

    # Air drag vertex
    assert tri.air_drag_vertex("B", 310.0, 590.0) is True
    assert tri.vertex_pos("B") == QPointF(310.0, 590.0)


def test_overlay_toggles_and_visual_rendering(qapp):
    """Test toggling geometric overlays and executing QPainter rendering."""
    tri = InteractiveTriangleGeometryItem(
        vert_a=(500.0, 200.0),
        vert_b=(300.0, 600.0),
        vert_c=(700.0, 600.0),
    )

    # Toggle overlays
    tri.toggle_circumcircle()
    assert tri._show_circumcircle is True
    tri.toggle_incircle()
    assert tri._show_incircle is True
    tri.toggle_euler_line()
    assert tri._show_euler_line is False
    tri.toggle_hud_card()
    assert tri._show_hud_card is False

    # Turn them back on for rendering
    tri.toggle_euler_line()
    tri.toggle_hud_card()

    # Bounding rect
    br = tri.boundingRect()
    assert not br.isEmpty()

    # Paint on QImage
    img = QImage(1200, 900, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)
    tri.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Test permanently stamping the triangle onto LessonBoard as vector items."""
    board = LessonBoard()
    tri = InteractiveTriangleGeometryItem(
        vert_a=(500.0, 200.0),
        vert_b=(300.0, 600.0),
        vert_c=(700.0, 600.0),
    )

    stamped_items = tri.stamp_to_canvas(board)
    assert len(stamped_items) >= 1
    for itm in stamped_items:
        assert itm in board.items()


def test_lesson_board_triangle_tool_lifecycle(qapp):
    """Test LessonBoard management of triangle theorem solver item."""
    board = LessonBoard()

    assert board.is_triangle_tool_visible() is False
    assert board.get_triangle_tool_item() is None

    # Show tool
    board.show_triangle_tool(True)
    assert board.is_triangle_tool_visible() is True
    item = board.get_triangle_tool_item()
    assert item is not None
    assert item.isVisible() is True

    # Toggle tool
    toggled = board.toggle_triangle_tool()
    assert toggled is False
    assert board.is_triangle_tool_visible() is False

    # Excluded from selection
    items = board.get_items_in_rect(QRectF(0, 0, 1920, 1080))
    assert item not in items


def test_main_window_triangle_tool_integration(qapp):
    """Test MainWindow actions, shortcuts, and slot integration for triangle tool."""
    win = MainWindow()

    actions = win.findChildren(QAction)
    tri_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+T"]
    assert len(tri_actions) >= 1, "Expected action with shortcut Ctrl+Shift+T"

    assert win._scene.is_triangle_tool_visible() is False

    is_now_vis = win._on_toggle_triangle_tool()
    assert is_now_vis is True
    assert win._scene.is_triangle_tool_visible() is True

    win._on_toggle_triangle_tool()
    assert win._scene.is_triangle_tool_visible() is False
    win.close()
