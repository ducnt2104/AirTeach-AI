"""
Unit Tests for Interactive Physics Vector Force Decomposer & Inclined Plane Suite
================================================================================
Covers:
1. Exact physical calculations: Gravity P, Normal force N, Components P_parallel & P_perp.
2. Friction force F_ms and kinetic sliding vs static equilibrium condition (tan(alpha) > mu).
3. Net force F_net and acceleration a = F_net / m.
4. Parameter modifications (mass m, angle alpha, friction coefficient mu) and signals.
5. Touchless air gesture dragging and air tapping to cycle slope angle.
6. QPainter visual rendering of wedge, rotated block, vector force arrows, and physics HUD card.
7. Canvas stamping (vectorizing wedge and force vectors onto LessonBoard).
8. LessonBoard and MainWindow integration (View menu action Ctrl+Shift+I, toolbar, slots).
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_physics_incline import InteractiveInclinedPlaneItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_physics_force_calculations_sliding(qapp):
    """Test physical quantities for a sliding block at 30° incline with mu = 0.25."""
    incline = InteractiveInclinedPlaneItem(
        alpha_deg=30.0,
        mass_kg=2.0,
        mu_friction=0.25,
    )

    # Gravity P = 2.0 * 9.8 = 19.6 N
    p = incline.gravity_force()
    assert abs(p - 19.6) < 1e-4

    # P_parallel = 19.6 * sin(30°) = 9.8 N
    p_par = incline.parallel_force()
    assert abs(p_par - 9.8) < 1e-3

    # P_perp = N = 19.6 * cos(30°) ~= 16.974 N
    n = incline.normal_force()
    assert abs(n - 19.6 * math.cos(math.radians(30.0))) < 1e-3

    # Friction F_ms = 0.25 * N ~= 4.2435 N
    f_ms = incline.friction_force()
    assert abs(f_ms - 0.25 * n) < 1e-3

    # Motion condition: tan(30°) ~= 0.577 > mu=0.25 -> sliding!
    assert incline.is_sliding() is True

    # Net force F_net = P_parallel - F_ms ~= 9.8 - 4.2435 = 5.5565 N
    f_net = incline.net_force()
    assert abs(f_net - (p_par - f_ms)) < 1e-3

    # Acceleration a = F_net / m ~= 2.778 m/s^2
    a = incline.acceleration()
    assert abs(a - (f_net / 2.0)) < 1e-3


def test_physics_static_equilibrium(qapp):
    """Test static equilibrium condition when slope is gentle: tan(alpha) <= mu."""
    # alpha = 10°, mu = 0.30 -> tan(10°) ~= 0.1763 <= 0.30 -> no slide!
    incline = InteractiveInclinedPlaneItem(
        alpha_deg=10.0,
        mass_kg=2.0,
        mu_friction=0.30,
    )

    assert incline.is_sliding() is False
    assert incline.net_force() == 0.0
    assert incline.acceleration() == 0.0


def test_parameter_updates_and_signals(qapp):
    """Test updating slope angle, mass, and friction coefficient."""
    incline = InteractiveInclinedPlaneItem(alpha_deg=30.0, mass_kg=2.0, mu_friction=0.25)

    angle_signals = []
    physics_signals = []
    incline.angle_changed.connect(lambda a: angle_signals.append(a))
    incline.physics_updated.connect(lambda f, a: physics_signals.append((f, a)))

    # Update angle
    incline.set_alpha(45.0)
    assert incline.alpha() == 45.0
    assert len(angle_signals) == 1
    assert angle_signals[0] == 45.0

    # Update mass
    incline.set_mass(5.0)
    assert incline.mass() == 5.0
    assert abs(incline.gravity_force() - 5.0 * 9.8) < 1e-3

    # Update friction
    incline.set_friction_coeff(0.4)
    assert incline.friction_coeff() == 0.4


def test_touchless_air_gestures(qapp):
    """Test air drag slope adjustment and air tap angle cycling."""
    incline = InteractiveInclinedPlaneItem(
        base_pos=(700.0, 700.0),
        base_length=400.0,
        alpha_deg=30.0,
    )

    # Air drag top handle: base_y is 700, dragging to y=300 -> height = 400
    # tan(alpha) = 400 / 400 = 1.0 -> alpha = 45°
    res = incline.air_drag_slope(300.0)
    assert res is True
    assert abs(incline.alpha() - 45.0) < 1e-2

    # Air click near top vertex to cycle (+5°)
    _, _, w3 = incline.wedge_vertices()
    old_a = incline.alpha()
    click_res = incline.air_click_at(w3.x(), w3.y())
    assert click_res is True
    assert abs(incline.alpha() - (old_a + 5.0)) < 1e-2


def test_visual_rendering(qapp):
    """Test QPainter rendering of inclined plane, forces, and HUD card."""
    incline = InteractiveInclinedPlaneItem(
        base_pos=(600.0, 600.0),
        base_length=400.0,
        alpha_deg=35.0,
        mass_kg=3.0,
        mu_friction=0.2,
    )

    br = incline.boundingRect()
    assert not br.isEmpty()

    img = QImage(1600, 1000, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)
    incline.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Test stamping inclined plane and force vectors onto LessonBoard."""
    board = LessonBoard()
    incline = InteractiveInclinedPlaneItem(
        base_pos=(600.0, 600.0),
        base_length=400.0,
        alpha_deg=30.0,
    )

    stamped_items = incline.stamp_to_canvas(board)
    assert len(stamped_items) >= 2  # Wedge + force vectors
    for itm in stamped_items:
        assert itm in board.items()


def test_lesson_board_inclined_plane_lifecycle(qapp):
    """Test LessonBoard management of interactive inclined plane item."""
    board = LessonBoard()

    assert board.is_inclined_plane_visible() is False
    assert board.get_inclined_plane_item() is None

    # Show item
    board.show_inclined_plane(True, alpha=35.0, mass=2.5, mu=0.3)
    assert board.is_inclined_plane_visible() is True
    item = board.get_inclined_plane_item()
    assert item is not None
    assert item.isVisible() is True
    assert item.alpha() == 35.0

    # Toggle item
    toggled = board.toggle_inclined_plane()
    assert toggled is False
    assert board.is_inclined_plane_visible() is False

    # Excluded from selection
    items = board.get_items_in_rect(QRectF(0, 0, 1920, 1080))
    assert item not in items


def test_main_window_inclined_plane_integration(qapp):
    """Test MainWindow actions, shortcuts, and slot integration for inclined plane."""
    win = MainWindow()

    actions = win.findChildren(QAction)
    incline_actions = [a for a in actions if a.shortcut().toString() == "Ctrl+Shift+I"]
    assert len(incline_actions) >= 1, "Expected action with shortcut Ctrl+Shift+I"

    assert win._scene.is_inclined_plane_visible() is False

    is_now_vis = win._on_toggle_inclined_plane()
    assert is_now_vis is True
    assert win._scene.is_inclined_plane_visible() is True

    win._on_toggle_inclined_plane()
    assert win._scene.is_inclined_plane_visible() is False
    win.close()
