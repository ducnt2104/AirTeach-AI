"""
Unit and integration tests for Interactive AC Series RLC Circuit & Fresnel Phasor Diagram Simulator.
Verifies AC impedance formula Z = sqrt(R² + (Z_L - Z_C)²), resonance condition f₀ = 1/(2π*sqrt(LC)),
Fresnel vector voltage decomposition, power factor cos(φ), air gestures, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_rlc_circuit import InteractiveRlcCircuitItem
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveRlcCircuit:
    """Test suite for InteractiveRlcCircuitItem."""

    def test_rlc_initialization(self, qapp):
        """Verify default 50Hz resonance parameters, impedance, current, and phase."""
        item = InteractiveRlcCircuitItem(pos=(200.0, 150.0), preset="RESONANCE_50HZ")
        assert item.x() == 200.0
        assert item.y() == 150.0
        assert item.boundingRect().width() == 840.0
        assert item.boundingRect().height() == 560.0
        assert item.preset == "RESONANCE_50HZ"
        assert item.frequency_f == 50.0
        assert item.resistor_R == 50.0
        assert pytest.approx(item.inductive_reactance_ZL, rel=1e-2) == 100.0
        assert pytest.approx(item.capacitive_reactance_ZC, rel=1e-2) == 100.0

        # At resonance Z = R = 50 Ω
        assert pytest.approx(item.total_impedance_Z, rel=1e-2) == 50.0
        # Current I = 220 / 50 = 4.4 A
        assert pytest.approx(item.rms_current_I, rel=1e-2) == 4.40
        # Phase difference φ ≈ 0.0 deg
        assert pytest.approx(item.phase_deg, abs=0.5) == 0.0
        # Power factor cos(φ) ≈ 1.0
        assert pytest.approx(item.power_factor, abs=0.01) == 1.0
        item.close()

    def test_curriculum_presets_and_electrical_regimes(self, qapp):
        """Verify switching through all 4 authentic AC electrical regimes."""
        item = InteractiveRlcCircuitItem()

        # Preset 1: Inductive Circuit (Z_L > Z_C => φ > 0)
        item.set_preset("INDUCTIVE_CIRCUIT")
        assert item.preset == "INDUCTIVE_CIRCUIT"
        assert item.inductive_reactance_ZL > item.capacitive_reactance_ZC
        assert item.phase_deg > 0.0  # Voltage leads current
        assert item.power_factor < 1.0

        # Preset 2: Capacitive Circuit (Z_C > Z_L => φ < 0)
        item.set_preset("CAPACITIVE_CIRCUIT")
        assert item.preset == "CAPACITIVE_CIRCUIT"
        assert item.capacitive_reactance_ZC > item.inductive_reactance_ZL
        assert item.phase_deg < 0.0  # Current leads voltage

        # Preset 3: High-Q Resonance (R = 10 Ω => Q = 10.0)
        item.set_preset("HIGH_Q_RESONANCE")
        assert item.preset == "HIGH_Q_RESONANCE"
        assert item.resistor_R == 10.0
        assert pytest.approx(item.quality_factor_Q, rel=1e-2) == 10.0
        # Higher current at resonance when R is small (220 / 10 = 22 A)
        assert pytest.approx(item.rms_current_I, rel=1e-2) == 22.0

        # Preset 4: Standard 50Hz Resonance
        item.set_preset("RESONANCE_50HZ")
        assert item.preset == "RESONANCE_50HZ"
        assert pytest.approx(item.resonance_frequency_f0, abs=0.5) == 50.0
        item.close()

    def test_resonance_condition_and_snap(self, qapp):
        """Verify manual detuning and snapping back to resonant frequency f₀."""
        item = InteractiveRlcCircuitItem()

        # Detune frequency to 75 Hz
        item.set_frequency(75.0)
        assert item.frequency_f == 75.0
        assert item.inductive_reactance_ZL != item.capacitive_reactance_ZC
        assert abs(item.phase_deg) > 5.0

        # Snap to resonance
        item.snap_to_resonance()
        assert pytest.approx(item.frequency_f, abs=0.5) == item.resonance_frequency_f0
        assert pytest.approx(item.inductive_reactance_ZL, rel=1e-2) == item.capacitive_reactance_ZC
        assert pytest.approx(item.phase_deg, abs=0.5) == 0.0
        item.close()

    def test_power_and_quality_factor_math(self, qapp):
        """Verify active power P = I²*R = U*I*cos(φ) and quality factor."""
        item = InteractiveRlcCircuitItem()
        item.set_preset("INDUCTIVE_CIRCUIT")

        i_val = item.rms_current_I
        r_val = item.resistor_R
        u_val = 220.0
        cos_phi = item.power_factor

        # P = I² * R
        p_expected = i_val * i_val * r_val
        p_formula = u_val * i_val * cos_phi

        assert pytest.approx(item.active_power_P, rel=1e-4) == p_expected
        assert pytest.approx(item.active_power_P, rel=1e-4) == p_formula
        item.close()

    def test_sliders_and_parameter_clamping(self, qapp):
        """Verify parameter clamping and setters."""
        item = InteractiveRlcCircuitItem()

        # Frequency clamp [10, 200]
        item.set_frequency(350.0)
        assert item.frequency_f == 200.0
        item.set_frequency(2.0)
        assert item.frequency_f == 10.0
        item.set_frequency(60.0)
        assert item.frequency_f == 60.0

        # Resistor clamp [10, 500]
        item.set_resistance(800.0)
        assert item.resistor_R == 500.0
        item.set_resistance(2.0)
        assert item.resistor_R == 10.0

        # Inductor clamp [0.05, 1.00]
        item.set_inductance(5.0)
        assert item.inductor_L == 1.00
        item.set_inductance(0.01)
        assert item.inductor_L == 0.05

        # Capacitor clamp [5, 100]
        item.set_capacitance(200.0)
        assert item.capacitor_C_uF == 100.0
        item.set_capacitance(1.0)
        assert item.capacitor_C_uF == 5.0
        item.close()

    def test_air_gestures_and_pinch_interaction(self, qapp):
        """Verify touchless air pinch and air tap gestures on presets, buttons and sliders."""
        item = InteractiveRlcCircuitItem(pos=(100.0, 100.0))

        # Air tap on Inductive Circuit preset button
        ind_btn = item._get_preset_button_rect(0, 1)
        sc_pt = item.mapToScene(ind_btn.center())
        handled = item.handle_air_pinch(sc_pt)
        assert handled is True
        assert item.preset == "INDUCTIVE_CIRCUIT"

        # Air tap on Resonance Snap button
        snap_btn = item._get_resonance_snap_button_rect()
        sc_snap = item.mapToScene(snap_btn.center())
        handled_snap = item.handle_air_tap(sc_snap)
        assert handled_snap is True
        assert pytest.approx(item.frequency_f, abs=0.5) == item.resonance_frequency_f0

        # Air tap close button
        close_btn = item._get_close_button_rect()
        sc_close = item.mapToScene(close_btn.center())
        handled_close = item.handle_air_tap(sc_close)
        assert handled_close is True
        assert item.isVisible() is False

    def test_vector_stamping_to_lesson_board(self, qapp):
        """Verify stamping vector diagram to LessonBoard produces valid QGraphicsItems."""
        board = LessonBoard()
        item = InteractiveRlcCircuitItem(pos=(100.0, 100.0))
        board.addItem(item)

        initial_count = len(board.items())
        item.stamp_to_board()
        final_count = len(board.items())

        # Stamped vector diagram contains banner, schematic, phasor vectors, and metrics (> 15 items)
        assert final_count > initial_count + 12
        item.close()

    def test_lesson_board_lifecycle_and_selection_exclusion(self, qapp):
        """Verify LessonBoard show/toggle/get methods and selection exclusion."""
        board = LessonBoard()

        # Show
        rlc_item = board.show_rlc_circuit(True, pos=(350.0, 180.0))
        assert rlc_item is not None
        assert board.is_rlc_circuit_visible() is True
        assert board.get_rlc_circuit() is rlc_item
        assert board.get_rlc_circuit_item() is rlc_item

        # Selection exclusion: Ensure get_items_in_rect ignores interactive RLC circuit item
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        items_in_sel = board.get_items_in_rect(selection_rect)
        assert rlc_item not in items_in_sel

        # Toggle
        new_state = board.toggle_rlc_circuit()
        assert new_state is False
        assert board.is_rlc_circuit_visible() is False
        assert board.get_rlc_circuit() is None
        assert board.get_rlc_circuit_item() is rlc_item

        new_state_2 = board.toggle_rlc_circuit()
        assert new_state_2 is True
        assert board.is_rlc_circuit_visible() is True
        board.show_rlc_circuit(False)
