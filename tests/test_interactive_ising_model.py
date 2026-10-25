"""
Unit tests for InteractiveIsingModelItem - 2D Ising Model & Statistical Mechanics.
==================================================================================
Covers Lars Onsager exact solution, Metropolis Monte Carlo algorithm, spontaneous
magnetization, Curie temperature phase transition, UI sliders, and serialization.
"""
import math
import pytest
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_ising_model import (
    InteractiveIsingModelItem,
    PRESETS,
    IsingPreset,
    ONSAGER_TC,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    item = InteractiveIsingModelItem()
    assert item.WIDTH == 940.0
    assert item.HEIGHT == 600.0
    assert item.current_preset.key == "critical_point"
    assert item.LATTICE_SIZE == 50
    assert item._spins.shape == (50, 50)
    assert not item._is_paused


def test_presets(qapp):
    item = InteractiveIsingModelItem()
    keys = ["ferro_ordered", "critical_point", "paramagnetic_gas", "field_hysteresis", "antiferromagnet"]

    for i, expected_key in enumerate(keys):
        item.set_preset(i)
        assert item.current_preset.key == expected_key
        assert item._sim_time == 0.0

    curr_idx = item._preset_index
    item.set_preset(99)
    assert item._preset_index == curr_idx


def test_onsager_analytical_solution(qapp):
    item = InteractiveIsingModelItem()
    # At low temperature T = 0.5 << Tc
    m_low = item.compute_onsager_magnetization(0.5)
    assert math.isclose(m_low, 1.0, abs_tol=1e-3)

    # At T = 2.0 < Tc ≈ 2.269
    m_mid = item.compute_onsager_magnetization(2.0)
    assert 0.90 < m_mid < 0.93

    # Near Tc (T = 2.25)
    m_near_crit = item.compute_onsager_magnetization(2.25)
    assert 0.40 < m_near_crit < 0.75

    # At T >= Tc, exact spontaneous magnetization is identically zero
    m_crit = item.compute_onsager_magnetization(ONSAGER_TC)
    assert m_crit == 0.0

    m_high = item.compute_onsager_magnetization(3.5)
    assert m_high == 0.0


def test_metropolis_thermalization(qapp):
    item = InteractiveIsingModelItem()
    item.set_preset(0)  # ferro_ordered preset with T = 1.2
    # Start with all spins = 1
    m_init, e_init = item.compute_observables()
    assert math.isclose(m_init, 1.0)
    # Energy per spin in uniform 2D square lattice with J=1, h=0 is -2.0
    assert math.isclose(e_init, -2.0)

    # Run 10 Monte Carlo sweeps at low T = 1.0
    item.set_temperature(1.0)
    for _ in range(10):
        item.monte_carlo_sweep()

    m_after, e_after = item.compute_observables()
    # Should remain highly magnetized
    assert m_after > 0.85
    assert e_after < -1.5


def test_paramagnetic_fluctuation(qapp):
    item = InteractiveIsingModelItem()
    item.set_preset(2)  # paramagnetic_gas, T = 4.2 >> Tc
    # Run 15 sweeps to randomize
    for _ in range(15):
        item.monte_carlo_sweep()

    m_para, _ = item.compute_observables()
    # At high temperature, mean magnetization should fluctuate near zero
    assert m_para < 0.25


def test_external_field_alignment(qapp):
    item = InteractiveIsingModelItem()
    item.set_preset(3)  # field_hysteresis
    # Apply strong positive field h = +1.5 at T = 1.5
    item.set_temperature(1.5)
    item.set_external_field(1.5)

    for _ in range(15):
        item.monte_carlo_sweep()

    m_aligned, _ = item.compute_observables()
    assert m_aligned > 0.80


def test_parameter_setters(qapp):
    item = InteractiveIsingModelItem()
    # Temperature
    item.set_temperature(2.5)
    assert math.isclose(item._temp, 2.5)
    item.set_temperature(10.0)  # Clamped to 6.0
    assert math.isclose(item._temp, 6.0)

    # Coupling J
    item.set_coupling_j(-1.0)
    assert math.isclose(item._j, -1.0)
    item.set_coupling_j(5.0)  # Clamped to 2.0
    assert math.isclose(item._j, 2.0)

    # External field
    item.set_external_field(1.2)
    assert math.isclose(item._h, 1.2)
    item.set_external_field(-5.0)  # Clamped to -2.0
    assert math.isclose(item._h, -2.0)


def test_pause_and_reset(qapp):
    item = InteractiveIsingModelItem()
    assert not item._is_paused
    item.toggle_pause()
    assert item._is_paused

    time_before = item._sim_time
    item.step_physics(0.05)
    assert item._sim_time == time_before

    item.toggle_pause()
    assert not item._is_paused
    item.step_physics(0.05)
    assert item._sim_time > time_before

    item.reset_simulation()
    assert item._sim_time == 0.0


def test_paint_rendering(qapp):
    item = InteractiveIsingModelItem()
    for _ in range(5):
        item.step_physics(0.033)

    img = QImage(940, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()

    assert not img.isNull()
    assert img.width() == 940


def test_mouse_interactions(qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    item = InteractiveIsingModelItem()

    def make_event(event_type, pos_f):
        e = QGraphicsSceneMouseEvent(event_type)
        e.setPos(pos_f)
        e.setScenePos(pos_f)
        e.setButton(Qt.LeftButton)
        return e

    # Click Preset 0 (ferro_ordered)
    p_rects = item._preset_button_rects()
    target_rect, _ = p_rects[0]
    click_ev = make_event(QEvent.GraphicsSceneMousePress, target_rect.center())
    item.mousePressEvent(click_ev)
    assert item.current_preset.key == "ferro_ordered"

    # Click Pause badge
    badge_rect = QRectF(item.WIDTH - 150.0, 14.0, 130.0, 28.0)
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

    # Drag slider (temp)
    track_rect = QRectF(ctrl_rect.left() + 20.0, ctrl_rect.top() + 36.0, col_w - 25.0, 10.0)
    drag_start = make_event(QEvent.GraphicsSceneMousePress, track_rect.center())
    item.mousePressEvent(drag_start)
    assert item._dragging_slider == "temp"

    drag_move = make_event(QEvent.GraphicsSceneMouseMove, track_rect.topRight())
    item.mouseMoveEvent(drag_move)

    drag_rel = make_event(QEvent.GraphicsSceneMouseRelease, track_rect.topRight())
    item.mouseReleaseEvent(drag_rel)
    assert item._dragging_slider is None


def test_serialization_roundtrip(qapp):
    item = InteractiveIsingModelItem()
    item.set_preset(4)  # antiferromagnet
    item.set_temperature(1.6)
    item.set_coupling_j(-1.2)
    item.set_external_field(0.3)
    item.setPos(140.0, 230.0)
    for _ in range(5):
        item.step_physics(0.033)

    data = item.to_dict()
    assert data["type"] == "InteractiveIsingModelItem"
    assert data["preset_key"] == "antiferromagnet"
    assert math.isclose(data["temp"], 1.6, abs_tol=1e-3)
    assert math.isclose(data["j"], -1.2, abs_tol=1e-3)
    assert math.isclose(data["h"], 0.3, abs_tol=1e-3)

    restored = InteractiveIsingModelItem.from_dict(data)
    assert restored.current_preset.key == "antiferromagnet"
    assert math.isclose(restored._temp, 1.6, abs_tol=1e-3)
    assert math.isclose(restored._j, -1.2, abs_tol=1e-3)
    assert math.isclose(restored._h, 0.3, abs_tol=1e-3)
    assert restored.pos() == QPointF(140.0, 230.0)
