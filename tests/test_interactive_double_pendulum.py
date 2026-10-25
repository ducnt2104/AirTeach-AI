"""
Unit tests for InteractiveDoublePendulumItem - Nonlinear Dynamics & Chaos Simulation.
=====================================================================================
Covers Lagrangian equations of motion, RK4 energy conservation, Lyapunov exponent,
twin pendulum butterfly effect divergence, UI sliders/buttons, and serialization.
"""
import math
import pytest
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_double_pendulum import (
    InteractiveDoublePendulumItem,
    PRESETS,
    PendulumPreset,
    GRAVITY_G,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    item = InteractiveDoublePendulumItem()
    assert item.WIDTH == 920.0
    assert item.HEIGHT == 600.0
    assert item.current_preset.key == "small_oscillations"
    assert item._l1 == 1.0
    assert item._l2 == 1.0
    assert item._m1 == 1.0
    assert item._m2 == 1.0
    assert item._damping == 0.0
    assert not item._is_paused
    assert len(item._state_a) == 4
    assert len(item._state_b) == 4
    assert math.isclose(item._state_b[0] - item._state_a[0], 1e-4, abs_tol=1e-8)


def test_preset_switching(qapp):
    item = InteractiveDoublePendulumItem()
    keys = ["small_oscillations", "developed_chaos", "flip_over", "asymmetric_whip", "damped_attractor"]

    for i, expected_key in enumerate(keys):
        item.set_preset(i)
        assert item.current_preset.key == expected_key
        assert item._sim_time == 0.0
        assert len(item._trail_a) == 0
        assert len(item._phase_history) == 0

    # Bounds check: out of range index ignored
    curr_idx = item._preset_index
    item.set_preset(99)
    assert item._preset_index == curr_idx


def test_lagrangian_equations_of_motion(qapp):
    item = InteractiveDoublePendulumItem()
    # At rest at downward equilibrium theta1=0, theta2=0, w1=0, w2=0
    s_eq = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    derivs = item._derivatives(s_eq)
    # Both angular accelerations should be 0
    assert math.isclose(derivs[0], 0.0, abs_tol=1e-6)
    assert math.isclose(derivs[1], 0.0, abs_tol=1e-6)
    assert math.isclose(derivs[2], 0.0, abs_tol=1e-6)
    assert math.isclose(derivs[3], 0.0, abs_tol=1e-6)

    # Deflected theta1=pi/2, theta2=0, rest
    s_def = np.array([math.pi / 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    derivs_def = item._derivatives(s_def)
    # Gravitational torque accelerates theta1 downwards (negative ddtheta1)
    assert derivs_def[1] < -1.0


def test_rk4_energy_conservation(qapp):
    item = InteractiveDoublePendulumItem()
    item.set_preset(0)  # small oscillations, zero damping
    item._damping = 0.0

    initial_energy = item.compute_energy(item._state_a)

    # Step simulation for 3 seconds of physical time
    dt = 0.02
    for _ in range(150):
        item.step_simulation(dt)

    final_energy = item.compute_energy(item._state_a)
    delta_e = abs(final_energy - initial_energy)
    relative_error = delta_e / abs(initial_energy)

    # Symplectic sub-stepped RK4 maintains energy within 0.1%
    assert relative_error < 0.001


def test_butterfly_effect_divergence(qapp):
    item = InteractiveDoublePendulumItem()
    # Developed chaos preset: theta1=90°, theta2=90°
    item.set_preset(1)
    d0 = math.sqrt(
        (item._state_a[0] - item._state_b[0])**2
        + (item._state_a[2] - item._state_b[2])**2
    )
    assert math.isclose(d0, 1e-4, abs_tol=1e-6)

    # Step simulation for 12 seconds
    dt = 0.02
    for _ in range(600):
        item.step_simulation(dt)

    # In chaotic regime, twin trajectories must diverge significantly
    d_final = math.sqrt(
        (item._state_a[0] - item._state_b[0])**2
        + (item._state_a[2] - item._state_b[2])**2
    )
    assert d_final > 5.0 * d0  # Diverged significantly
    assert len(item._lyapunov_history) > 0


def test_parameter_setters(qapp):
    item = InteractiveDoublePendulumItem()
    # Length ratio
    item.set_length_ratio(1.5)
    assert math.isclose(item._l2, 1.5 * item._l1)
    item.set_length_ratio(10.0)  # Clamped to 2.5
    assert math.isclose(item._l2, 2.5 * item._l1)

    # Mass ratio
    item.set_mass_ratio(2.0)
    assert math.isclose(item._m2, 2.0 * item._m1)
    item.set_mass_ratio(10.0)  # Clamped to 5.0
    assert math.isclose(item._m2, 5.0 * item._m1)

    # Damping
    item.set_damping(0.12)
    assert math.isclose(item._damping, 0.12)
    item.set_damping(1.0)  # Clamped to 0.25
    assert math.isclose(item._damping, 0.25)


def test_pause_and_reset(qapp):
    item = InteractiveDoublePendulumItem()
    assert not item._is_paused
    item.toggle_pause()
    assert item._is_paused

    time_before = item._sim_time
    item.step_simulation(0.05)
    # Time must not advance when paused
    assert item._sim_time == time_before

    item.toggle_pause()
    assert not item._is_paused
    item.step_simulation(0.05)
    assert item._sim_time > time_before

    # Reset
    item.reset_pendulum()
    assert item._sim_time == 0.0


def test_phase_space_recording(qapp):
    item = InteractiveDoublePendulumItem()
    item.set_preset(1)
    for _ in range(25):
        item.step_simulation(0.02)

    assert len(item._phase_history) == 25
    assert len(item._trail_a) == 25
    assert len(item._trail_b) == 25
    th2, w2 = item._phase_history[-1]
    assert isinstance(th2, float)
    assert isinstance(w2, float)


def test_paint_rendering(qapp):
    item = InteractiveDoublePendulumItem()
    item.set_preset(1)
    for _ in range(20):
        item.step_simulation(0.02)

    img = QImage(920, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()

    assert not img.isNull()
    assert img.width() == 920


def test_mouse_interactions(qapp):
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    item = InteractiveDoublePendulumItem()

    def make_event(event_type, pos_f):
        e = QGraphicsSceneMouseEvent(event_type)
        e.setPos(pos_f)
        e.setScenePos(pos_f)
        e.setButton(Qt.LeftButton)
        return e

    # Click Preset 2 (flip_over)
    p_rects = item._preset_button_rects()
    target_rect, _ = p_rects[2]
    click_ev = make_event(QEvent.GraphicsSceneMousePress, target_rect.center())
    item.mousePressEvent(click_ev)
    assert item.current_preset.key == "flip_over"

    # Click Pause badge
    badge_rect = QRectF(item.WIDTH - 150.0, 22.0, 122.0, 26.0)
    pause_ev = make_event(QEvent.GraphicsSceneMousePress, badge_rect.center())
    item.mousePressEvent(pause_ev)
    assert item._is_paused

    # Click Reset button
    ctrl_rect = item._controls_rect()
    col_w = (ctrl_rect.width() - 40.0) / 4.0
    btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
    btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
    reset_ev = make_event(QEvent.GraphicsSceneMousePress, btn_rect.center())
    item.mousePressEvent(reset_ev)
    assert item._sim_time == 0.0

    # Drag slider (l_ratio)
    track_rect = QRectF(ctrl_rect.left() + 20.0, ctrl_rect.top() + 36.0, col_w - 25.0, 10.0)
    drag_start = make_event(QEvent.GraphicsSceneMousePress, track_rect.center())
    item.mousePressEvent(drag_start)
    assert item._dragging_slider == "l_ratio"

    drag_move = make_event(QEvent.GraphicsSceneMouseMove, track_rect.topRight())
    item.mouseMoveEvent(drag_move)

    drag_rel = make_event(QEvent.GraphicsSceneMouseRelease, track_rect.topRight())
    item.mouseReleaseEvent(drag_rel)
    assert item._dragging_slider is None


def test_serialization_roundtrip(qapp):
    item = InteractiveDoublePendulumItem()
    item.set_preset(3)  # asymmetric_whip
    item.set_length_ratio(1.8)
    item.set_mass_ratio(0.5)
    item.set_damping(0.05)
    item.setPos(120.0, 240.0)
    for _ in range(10):
        item.step_simulation(0.02)

    data = item.to_dict()
    assert data["type"] == "InteractiveDoublePendulumItem"
    assert data["preset_key"] == "asymmetric_whip"
    assert math.isclose(data["l2"] / data["l1"], 1.8, abs_tol=1e-4)
    assert math.isclose(data["m2"] / data["m1"], 0.5, abs_tol=1e-4)
    assert math.isclose(data["damping"], 0.05, abs_tol=1e-4)

    restored = InteractiveDoublePendulumItem.from_dict(data)
    assert restored.current_preset.key == "asymmetric_whip"
    assert math.isclose(restored._l2 / restored._l1, 1.8, abs_tol=1e-4)
    assert math.isclose(restored._m2 / restored._m1, 0.5, abs_tol=1e-4)
    assert math.isclose(restored._damping, 0.05, abs_tol=1e-4)
    assert restored.pos() == QPointF(120.0, 240.0)


def test_damped_attractor_energy_decay(qapp):
    item = InteractiveDoublePendulumItem()
    # Damped attractor preset with damping = 0.08
    item.set_preset(4)
    assert item._damping > 0.0

    e_initial = item.compute_energy(item._state_a)
    for _ in range(250):
        item.step_simulation(0.02)
    e_later = item.compute_energy(item._state_a)

    # In damped regime, energy must monotonically dissipate
    assert e_later < e_initial


def test_bounding_rect_and_shape(qapp):
    item = InteractiveDoublePendulumItem()
    rect = item.boundingRect()
    assert rect.width() == 920.0
    assert rect.height() == 600.0

