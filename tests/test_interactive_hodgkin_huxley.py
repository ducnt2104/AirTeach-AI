"""
AirTeach AI - Tests for Interactive Hodgkin-Huxley Neuron ODE Simulator
========================================================================
Validates 4-variable biophysical differential equations, ion channel gating
kinetics, action potential spike generation, and pharmacological blockers.
"""
import math
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_hodgkin_huxley import (
    HODGKIN_PRESETS,
    V_REST,
    InteractiveHodgkinHuxleyItem,
)


@pytest.fixture
def hh_item(qapp):
    return InteractiveHodgkinHuxleyItem(preset_key="SINGLE_ACTION_POTENTIAL", stim_amp=10.0)


def test_rate_function_singularities(hh_item):
    """Verify alpha_m and alpha_n handle 0/0 singularities safely via L'Hopital limits."""
    # alpha_m at v = 25.0
    am_at_25 = hh_item.alpha_m(25.0)
    assert math.isclose(am_at_25, 1.0, rel_tol=1e-3)

    # alpha_n at v = 10.0
    an_at_10 = hh_item.alpha_n(10.0)
    assert math.isclose(an_at_10, 0.1, rel_tol=1e-3)


def test_action_potential_spike_generation(hh_item):
    """A 10 μA/cm² stimulus must generate an action potential with peak overshoot > 0 mV."""
    hh_item.set_stim_amp(10.0)
    sim = hh_item.compute_simulation()

    assert sim["has_spike"] is True
    assert sim["v_max"] > 15.0  # Typically reaches +30 to +40 mV
    assert sim["v_min"] < -65.0  # Displays afterhyperpolarization undershoot


def test_subthreshold_stimulus_no_spike(hh_item):
    """A weak stimulus (3 μA/cm²) must fail to trigger a full action potential."""
    hh_item.apply_preset("SUBTHRESHOLD_STIMULUS")
    sim = hh_item.compute_simulation()

    assert sim["has_spike"] is False
    assert sim["v_max"] < -45.0


def test_ttx_sodium_block_abolishes_spike(hh_item):
    """Tetrodotoxin blocks g_Na, abolishing the spike even under strong stimulation."""
    hh_item.set_stim_amp(15.0)

    # Normal state
    sim_normal = hh_item.compute_simulation()
    assert sim_normal["has_spike"] is True

    # Blocked with TTX
    hh_item.toggle_ttx()
    sim_ttx = hh_item.compute_simulation()
    assert sim_ttx["has_spike"] is False
    assert sim_ttx["v_max"] < -40.0


def test_tea_potassium_block_prolongs_depolarization(hh_item):
    """Tetraethylammonium blocks g_K, preventing fast repolarization."""
    hh_item.apply_preset("TEA_POTASSIUM_BLOCK")
    sim = hh_item.compute_simulation()

    # Without K+ channels, membrane remains depolarized near peak
    final_v = sim["v_pts"][-1]
    assert final_v > -30.0


def test_repetitive_train_firing(hh_item):
    """Continuous current generates multiple spike oscillations."""
    hh_item.apply_preset("REPETITIVE_TRAIN_FIRING")
    sim = hh_item.compute_simulation()

    # Count number of times voltage crosses 0 mV going upwards
    v_pts = sim["v_pts"]
    spike_crossings = 0
    for i in range(1, len(v_pts)):
        if v_pts[i - 1] < 0.0 and v_pts[i] >= 0.0:
            spike_crossings += 1

    assert spike_crossings >= 3


def test_paint_and_render_zero_crash(qapp, hh_item):
    """Verify paint rendering executes without crashing on a QImage."""
    scene = QGraphicsScene()
    scene.addItem(hh_item)

    image = QImage(720, 450, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.black)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    hh_item.paint(painter, option, None)
    painter.end()

    assert not image.isNull()
