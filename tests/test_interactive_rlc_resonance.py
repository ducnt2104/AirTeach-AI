"""
AirTeach AI — Unit Tests for Interactive Electrodynamics & AC Circuit Analysis:
RLC Resonant Circuit, Phasor Diagram, Quality Factor Q & Frequency Response (Milestone 151 / Task I.138).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_rlc_resonance import (
    InteractiveRLCResonanceItem,
    RLCPreset,
    RLC_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_rlc_presets_and_physical_parameters():
    """Verifies that all 5 curriculum presets are properly defined with authentic electrical values."""
    expected_codes = [
        "AM_RADIO_TUNER_540KHZ_HIGH_Q",
        "AUDIO_CROSSOVER_LOWPASS_SUBWOOFER",
        "TESLA_COIL_PRIMARY_TANK_250KHZ",
        "MAINS_POWER_FACTOR_CORRECTION_50HZ",
        "CRITICALLY_DAMPED_PULSE_CIRCUIT",
    ]
    for code in expected_codes:
        assert code in RLC_PRESETS
        preset = RLC_PRESETS[code]
        assert preset.r_ohms > 0.0
        assert preset.l_henries > 0.0
        assert preset.c_farads > 0.0
        assert preset.v0_volts > 0.0
        assert preset.accent_color.isValid()
        assert len(preset.circuit_name) > 0


def test_resonance_condition_and_impedance_minimum(qapp):
    """
    Verifies that at exact resonance f = f_0:
    - X_L == X_C
    - Net reactance X == 0
    - Impedance |Z| reaches minimum |Z| == R
    - Phase angle phi == 0 deg (in-phase)
    - Current I_0 reaches maximum V_0 / R
    """
    item = InteractiveRLCResonanceItem()
    item.set_preset("AM_RADIO_TUNER_540KHZ_HIGH_Q")

    f0 = item.resonant_frequency_hz
    item.tune_to_resonance()

    assert math.isclose(item._freq_hz, f0, rel_tol=1e-5)
    assert math.isclose(item.inductive_reactance_ohms, item.capacitive_reactance_ohms, rel_tol=1e-3)
    assert math.isclose(item.net_reactance_ohms, 0.0, abs_tol=1e-3)
    assert math.isclose(item.impedance_magnitude_ohms, item._r_ohms, rel_tol=1e-4)
    assert math.isclose(item.phase_angle_deg, 0.0, abs_tol=1e-2)

    expected_i_max = item._v0_volts / item._r_ohms
    assert math.isclose(item.current_amplitude_amps, expected_i_max, rel_tol=1e-4)


def test_sub_and_super_resonant_phase_shift(qapp):
    """
    Verifies phase shift:
    - Below resonance (f < f0): XC > XL -> phi < 0 (capacitive, current leads)
    - Above resonance (f > f0): XL > XC -> phi > 0 (inductive, voltage leads)
    """
    item = InteractiveRLCResonanceItem()
    item.set_preset("AM_RADIO_TUNER_540KHZ_HIGH_Q")
    f0 = item.resonant_frequency_hz

    # Below resonance
    item.set_frequency_hz(f0 * 0.90)
    assert item.capacitive_reactance_ohms > item.inductive_reactance_ohms
    assert item.phase_angle_deg < 0.0
    assert item.impedance_magnitude_ohms > item._r_ohms

    # Above resonance
    item.set_frequency_hz(f0 * 1.10)
    assert item.inductive_reactance_ohms > item.capacitive_reactance_ohms
    assert item.phase_angle_deg > 0.0
    assert item.impedance_magnitude_ohms > item._r_ohms


def test_quality_factor_and_bandwidth(qapp):
    """
    Verifies Q = (1 / R) * sqrt(L / C) and Bandwidth Delta f = f0 / Q.
    Halving R must double Q and cut Bandwidth in half.
    """
    item = InteractiveRLCResonanceItem()
    item.set_preset("AM_RADIO_TUNER_540KHZ_HIGH_Q")

    q_initial = item.quality_factor_Q
    bw_initial = item.bandwidth_hz

    # Halve R
    item.set_resistance_ohms(item._r_ohms * 0.50)
    q_doubled = item.quality_factor_Q
    bw_halved = item.bandwidth_hz

    assert math.isclose(q_doubled, 2.0 * q_initial, rel_tol=1e-4)
    assert math.isclose(bw_halved, 0.5 * bw_initial, rel_tol=1e-4)


def test_frequency_and_resistance_adjustments(qapp):
    """Verifies parameter setting and reset functionality."""
    item = InteractiveRLCResonanceItem()
    item.set_preset("TESLA_COIL_PRIMARY_TANK_250KHZ")

    # Set frequency
    item.set_frequency_hz(200000.0)
    assert math.isclose(item._freq_hz, 200000.0)

    # Tune to resonance
    item.tune_to_resonance()
    assert math.isclose(item._freq_hz, item.resonant_frequency_hz, rel_tol=1e-4)

    # Reset
    item.reset_defaults()
    assert math.isclose(item._r_ohms, item._preset.r_ohms)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button handlers and interactive action triggers."""
    item = InteractiveRLCResonanceItem()
    item.set_preset("AM_RADIO_TUNER_540KHZ_HIGH_Q")

    # Frequency adjustments
    orig_f = item._freq_hz
    item._handle_button_click("f_inc")
    assert math.isclose(item._freq_hz, orig_f * 1.10, rel_tol=1e-3)

    item._handle_button_click("f_dec")
    assert math.isclose(item._freq_hz, orig_f * 1.10 * 0.90, rel_tol=1e-3)

    # Tune to resonance
    item._handle_button_click("tune_res")
    assert math.isclose(item._freq_hz, item.resonant_frequency_hz, rel_tol=1e-4)

    # Q boost (halves R)
    orig_r = item._r_ohms
    item._handle_button_click("q_high")
    assert math.isclose(item._r_ohms, orig_r * 0.50, rel_tol=1e-3)

    # Preset cycler
    item._handle_button_click("preset_next")
    assert item._active_preset_code != "AM_RADIO_TUNER_540KHZ_HIGH_Q"


def test_stem_tools_rlc_diagram():
    """Verifies vector diagram creation in PhysicsTools for RLC Resonance."""
    items = PhysicsTools.create_rlc_resonance_diagram(
        center=(400.0, 300.0),
        preset_code="AM_RADIO_TUNER_540KHZ_HIGH_Q",
        circuit_name="AM Receiver Tuner (540 kHz)",
        f0_hz=540000.0,
        freq_hz=540000.0,
        r_ohms=10.0,
        l_henries=220.0e-6,
        c_farads=395.7e-12,
        z_ohms=10.0,
        i0_amps=0.10,
        phi_deg=0.0,
        quality_factor_Q=74.6,
        bandwidth_hz=7238.0,
    )
    assert len(items) >= 15
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsLineItem" in types


def test_lesson_board_rlc_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_rlc_resonance_visible()

    # Show
    item = board.show_rlc_resonance(True)
    assert item is not None
    assert board.is_rlc_resonance_visible()
    assert board.get_rlc_resonance() is item

    # Toggle off
    state = board.toggle_rlc_resonance()
    assert not state
    assert not board.is_rlc_resonance_visible()
    assert board.get_rlc_resonance() is None

    # Toggle on
    state = board.toggle_rlc_resonance()
    assert state
    assert board.is_rlc_resonance_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_rlc_resonance({
        "preset_code": "AM_RADIO_TUNER_540KHZ_HIGH_Q",
        "circuit_name": "AM Receiver",
        "f0_hz": 540000.0,
        "freq_hz": 540000.0,
        "r_ohms": 10.0,
        "l_henries": 220.0e-6,
        "c_farads": 395.7e-12,
        "z_ohms": 10.0,
        "i0_amps": 0.10,
        "phi_deg": 0.0,
        "quality_factor_Q": 74.6,
        "bandwidth_hz": 7238.0,
    })
    after_count = len(board.items())
    assert after_count > initial_count + 10


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveRLCResonanceItem()
    image = QImage(700, 430, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not image.isNull()
