"""
Unit Tests for Interactive Function Graph Plotter & Tangent Line Inspector Suite
================================================================================
Covers:
1. Safe mathematical expression evaluation and numerical derivative computation.
2. Parabola, cubic, and trigonometric function curve sampling.
3. Tracer probe M(x0, y0) gliding and horizontal tangent at extrema (f'(x) = 0).
4. Dynamic tangent line equation formatting: y - y0 = k*(x - x0).
5. Touchless air gesture dragging of calculus probe.
6. QPainter visual rendering of neon halo, tangent line, probe reticle, and calculus badge.
7. Canvas stamping (vectorizing curve and tangent onto LessonBoard).
8. LessonBoard and MainWindow integration (View menu action Ctrl+Shift+F, toolbar, slots).
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_function_plotter import InteractiveFunctionPlotterItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_function_evaluation_and_derivatives(qapp):
    """Test safe evaluation of f(x) and numerical derivatives on parabola f(x) = x**2 - 4."""
    plotter = InteractiveFunctionPlotterItem(
        expr="x**2 - 4",
        origin=(960.0, 540.0),
        grid_step=50.0,
    )

    # Values check
    assert abs(plotter.evaluate_f(0.0) - (-4.0)) < 1e-4
    assert abs(plotter.evaluate_f(2.0) - 0.0) < 1e-4
    assert abs(plotter.evaluate_f(-2.0) - 0.0) < 1e-4

    # Derivative check: f'(x) = 2x
    # At vertex x=0: f'(0) = 0 (horizontal tangent extrema)
    assert abs(plotter.evaluate_derivative(0.0)) < 1e-3

    # At x=2: f'(2) = 4
    assert abs(plotter.evaluate_derivative(2.0) - 4.0) < 1e-2

    # At x=-3: f'(-3) = -6
    assert abs(plotter.evaluate_derivative(-3.0) - (-6.0)) < 1e-2


def test_cubic_polynomial_and_extrema(qapp):
    """Test cubic function f(x) = 0.25*x**3 - x and its local extrema."""
    plotter = InteractiveFunctionPlotterItem(expr="0.25 * x**3 - x")

    # At x=0: f(0) = 0, f'(0) = -1.0
    assert abs(plotter.evaluate_f(0.0) - 0.0) < 1e-4
    assert abs(plotter.evaluate_derivative(0.0) - (-1.0)) < 1e-2

    # Extrema at x = sqrt(4/3) ~= 1.1547: f'(x) = 0.75*x^2 - 1 = 0
    x_ext = math.sqrt(4.0 / 3.0)
    slope_at_ext = plotter.evaluate_derivative(x_ext)
    assert abs(slope_at_ext) < 0.01


def test_probe_tracking_and_signals(qapp):
    """Test moving tracer probe along curve and emitting probe_moved signals."""
    plotter = InteractiveFunctionPlotterItem(expr="x**2", origin=(960.0, 540.0), grid_step=40.0)

    signals = []
    plotter.probe_moved.connect(lambda x0, y0, k: signals.append((x0, y0, k)))

    plotter.set_probe_x(2.0)
    assert plotter.probe_x() == 2.0
    assert len(signals) > 0
    x0, y0, k = signals[-1]
    assert abs(x0 - 2.0) < 1e-4
    assert abs(y0 - 4.0) < 1e-4
    assert abs(k - 4.0) < 1e-2

    # Scene position: origin_x + 2*40 = 1040, origin_y - 4*40 = 380
    pos = plotter.probe_scene_pos()
    assert pos is not None
    assert abs(pos.x() - 1040.0) < 1e-4
    assert abs(pos.y() - 380.0) < 1e-4


def test_air_gesture_probe_drag(qapp):
    """Test touchless air gesture mapping canvas x to probe x."""
    plotter = InteractiveFunctionPlotterItem(origin=(960.0, 540.0), grid_step=50.0)

    # Air drag to canvas x = 1060 -> (1060 - 960)/50 = 2.0
    res = plotter.air_drag_probe(1060.0, 540.0)
    assert res is True
    assert abs(plotter.probe_x() - 2.0) < 1e-4


def test_overlays_and_visual_rendering(qapp):
    """Test toggling tangent line, HUD badge, and executing QPainter rendering."""
    plotter = InteractiveFunctionPlotterItem(expr="sin(x)", origin=(960.0, 540.0), grid_step=60.0)

    # Toggles
    plotter.toggle_tangent()
    assert plotter._show_tangent is False
    plotter.toggle_tangent()
    assert plotter._show_tangent is True

    plotter.toggle_hud_badge()
    assert plotter._show_hud_badge is False
    plotter.toggle_hud_badge()
    assert plotter._show_hud_badge is True

    # Bounding rect
    br = plotter.boundingRect()
    assert not br.isEmpty()

    # Paint on QImage
    img = QImage(1920, 1080, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)
    plotter.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Test stamping function curve and tangent line onto LessonBoard."""
    board = LessonBoard()
    plotter = InteractiveFunctionPlotterItem(expr="x**2 - 2", probe_x=1.5)

    stamped_items = plotter.stamp_to_canvas(board)
    assert len(stamped_items) >= 1
    for itm in stamped_items:
        assert itm in board.items()


def test_lesson_board_function_plotter_lifecycle(qapp):
    """Test LessonBoard management of interactive function plotter item."""
    board = LessonBoard()

    assert board.is_function_plotter_visible() is False
    assert board.get_function_plotter_item() is None

    # Show plotter
    board.show_function_plotter(True, expr="x**2")
    assert board.is_function_plotter_visible() is True
    item = board.get_function_plotter_item()
    assert item is not None
    assert item.isVisible() is True

    # Grid spacing synchronization
    board.set_grid_spacing(80)
    assert item._grid_step == 80.0

    # Toggle plotter off
    toggled = board.toggle_function_plotter()
    assert toggled is False
    assert board.is_function_plotter_visible() is False

    # Excluded from selection
    items = board.get_items_in_rect(QRectF(0, 0, 1920, 1080))
    assert item not in items


def test_main_window_function_plotter_integration(qapp):
    """Test MainWindow actions, shortcuts, and slot integration for function plotter."""
    win = MainWindow()

    actions = win.findChildren(QAction)
    func_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+F"]
    assert len(func_actions) >= 1, "Expected action with shortcut Ctrl+Shift+F"

    assert win._scene.is_function_plotter_visible() is False

    is_now_vis = win._on_toggle_function_plotter()
    assert is_now_vis is True
    assert win._scene.is_function_plotter_visible() is True

    win._on_toggle_function_plotter()
    assert win._scene.is_function_plotter_visible() is False
    win.close()
