"""
AirTeach AI - Unit Tests for Interactive Physics Simple Harmonic Motion & Resonance Simulator
==============================================================================================
Comprehensive automated unit tests verifying:
  1. Authentic mechanical oscillations (ideal SHM, energy conservation W = W_t + W_d).
  2. Damping regimes (underdamped, critically damped automotive shock, overdamped honey).
  3. Forced resonance and peak Lorentzian amplitude response A(ω).
  4. 5 Curriculum Presets (Ideal SHM, Underdamped, Critical, Overdamped, Resonance).
  5. Touchless air pinch gesture interactions, parameter tuning, and simulation progression.
  6. Vector diagram generation via PhysicsTools.create_harmonic_oscillator_diagram.
  7. LessonBoard lifecycle integration and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_harmonic_oscillator import (
    InteractiveHarmonicOscillatorItem,
    PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure single persistent QApplication for QGraphics testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveHarmonicOscillator:
    """Test suite for InteractiveHarmonicOscillatorItem and mechanical oscillations STEM integration."""

    def test_initialization(self, qapp):
        """Verifies initial configuration, geometry, and default physical values."""
        item = InteractiveHarmonicOscillatorItem(pos=QPointF(150.0, 100.0))
        assert item.pos() == QPointF(150.0, 100.0)
        assert item.boundingRect().width() == InteractiveHarmonicOscillatorItem.WIDTH
        assert item.boundingRect().height() == InteractiveHarmonicOscillatorItem.HEIGHT
        assert item.active_preset_code == "IDEAL_SPRING_CONSERVATIVE"
        assert item.spring_k == 50.0
        assert item.mass_kg == 0.5
        assert item.damping_b == 0.0
        assert abs(item.natural_frequency_rad_s - 10.0) < 1e-4  # sqrt(50/0.5) = 10.0
        assert item.is_animating is True
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies loading of all 5 curriculum presets and signal emissions."""
        item = InteractiveHarmonicOscillatorItem()
        emitted_presets = []
        item.preset_changed.connect(lambda p: emitted_presets.append(p))

        for code, expected_preset in PRESETS.items():
            loaded = item.load_preset(code)
            assert loaded is True
            assert item.active_preset_code == code
            assert item.preset.title_vi == expected_preset.title_vi
            assert item.preset.regime == expected_preset.regime
            assert item.mass_kg == expected_preset.mass_kg
            assert item.spring_k == expected_preset.spring_k_n_m
            assert item.damping_b == expected_preset.damping_b_ns_m

        assert len(emitted_presets) == 5
        assert item.load_preset("NON_EXISTENT_PRESET") is False
        item.cleanup()

    def test_ideal_shm_energy_conservation(self, qapp):
        """Verifies total mechanical energy conservation W = W_t + W_d in ideal SHM."""
        item = InteractiveHarmonicOscillatorItem()
        item.load_preset("IDEAL_SPRING_CONSERVATIVE")  # b = 0.0, k = 50, A = 0.15
        
        expected_w = 0.5 * 50.0 * (0.15 ** 2)  # 0.5 * 50 * 0.0225 = 0.5625 J
        assert abs(item.total_energy_j - expected_w) < 1e-4

        # Simulate several cycles of oscillation
        for _ in range(30):
            item._on_animation_tick()
            # Energy must remain conserved (within numerical integration tolerance)
            assert abs(item.total_energy_j - expected_w) < 0.08
        item.cleanup()

    def test_damping_regimes_classification(self, qapp):
        """Verifies analytical damping ratio zeta classifications."""
        item = InteractiveHarmonicOscillatorItem()

        # 1. Underdamped (zeta < 1)
        item.load_preset("UNDERDAMPED_AIR_FLUID")
        assert item.damping_ratio_zeta < 1.0
        assert item.damping_ratio_zeta > 0.0

        # 2. Critical damping (zeta == 1.0)
        item.load_preset("CRITICAL_AUTOMOTIVE_SHOCK")
        assert abs(item.damping_ratio_zeta - 1.00) < 1e-3

        # 3. Overdamped (zeta > 1.0)
        item.load_preset("OVERDAMPED_VISCOUS_HONEY")
        assert item.damping_ratio_zeta > 1.5
        item.cleanup()

    def test_resonance_amplitude_peak(self, qapp):
        """Verifies resonant frequency peak near natural frequency omega0."""
        item = InteractiveHarmonicOscillatorItem()
        item.load_preset("DRIVEN_ACOUSTIC_RESONANCE")

        omega0 = item.natural_frequency_rad_s
        omega_res = item.resonant_frequency_rad_s
        # Resonant peak is slightly below omega0 due to light damping
        assert 0.0 < omega_res <= omega0
        assert abs(omega_res - omega0) < 0.2
        # Quality factor Q should be high for light damping
        assert item.quality_factor_q > 5.0
        item.cleanup()

    def test_frequency_and_period_formulas(self, qapp):
        """Verifies mathematical consistency of natural angular frequency and period."""
        item = InteractiveHarmonicOscillatorItem()
        item.set_spring_constant(100.0)
        item.set_mass(1.0)

        # omega0 = sqrt(100 / 1) = 10 rad/s
        assert abs(item.natural_frequency_rad_s - 10.0) < 1e-4
        # f0 = 10 / (2*pi) ≈ 1.5915 Hz
        assert abs(item.natural_frequency_hz - (10.0 / (2.0 * math.pi))) < 1e-4
        item.cleanup()

    def test_parameter_setters_and_clamping(self, qapp):
        """Verifies parameter adjustments and boundary clamping."""
        item = InteractiveHarmonicOscillatorItem()

        item.set_spring_constant(1000.0)
        assert item.spring_k == 200.0
        item.set_spring_constant(1.0)
        assert item.spring_k == 5.0

        item.set_mass(20.0)
        assert item.mass_kg == 5.0
        item.set_mass(0.01)
        assert item.mass_kg == 0.1

        item.set_damping(100.0)
        assert item.damping_b == 30.0
        item.set_damping(-5.0)
        assert item.damping_b == 0.0
        item.cleanup()

    def test_simulation_tick_and_reset(self, qapp):
        """Verifies animation advancement, waveform recording, and state reset."""
        item = InteractiveHarmonicOscillatorItem()
        assert item.is_animating is True

        # Toggle to pause
        item.toggle_animation()
        assert not item.is_animating

        # Toggle to resume
        item.toggle_animation()
        assert item.is_animating

        # Simulate ticks
        for _ in range(15):
            item._on_animation_tick()
        assert item._sim_time_s > 0.0
        assert len(item._waveform_history) > 0

        # Reset
        item.reset_simulation()
        assert item._sim_time_s == 0.0
        assert len(item._waveform_history) == 0
        assert item._pos_x_m == item._initial_a
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch gesture dispatch on buttons and steppers."""
        item = InteractiveHarmonicOscillatorItem(pos=QPointF(0.0, 0.0))

        # 1. Pinch on Preset button (preset_CRITICAL_AUTOMOTIVE_SHOCK)
        rect_crit = item._btn_rects["preset_CRITICAL_AUTOMOTIVE_SHOCK"]
        pinch_pt = QPointF(rect_crit.center().x(), rect_crit.center().y())
        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "CRITICAL_AUTOMOTIVE_SHOCK"

        # 2. Pinch on Play / Pause button
        rect_play = item._btn_rects["play_pause"]
        assert item.is_animating is True
        handled_play = item.handle_air_pinch(rect_play.center())
        assert handled_play is True
        assert item.is_animating is False

        # 3. Pinch on Inc Spring k button
        rect_inc_k = item._btn_rects["inc_k"]
        init_k = item.spring_k
        handled_k = item.handle_air_pinch(rect_inc_k.center())
        assert handled_k is True
        assert item.spring_k > init_k

        # 4. Pinch on Inc Mass m button
        rect_inc_m = item._btn_rects["inc_m"]
        init_m = item.mass_kg
        handled_m = item.handle_air_pinch(rect_inc_m.center())
        assert handled_m is True
        assert item.mass_kg > init_m

        # 5. Pinch outside bounding box
        outside_pt = QPointF(1000.0, 1000.0)
        assert item.handle_air_pinch(outside_pt) is False
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies static vector diagram generation via PhysicsTools and scene stamping."""
        board = LessonBoard()
        item = InteractiveHarmonicOscillatorItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct diagram creation via PhysicsTools
        items = PhysicsTools.create_harmonic_oscillator_diagram(
            center=(400.0, 300.0),
            mass_kg=1.0,
            spring_k=100.0,
            damping_b=20.0,
            mode="DAMPED",
        )
        assert len(items) >= 8
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("DAO ĐỘNG CƠ HỌC" in t for t in text_contents)
        assert any("Phương Trình Vi Phân" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/visibility lifecycle and lasso exclusion."""
        board = LessonBoard()
        assert not board.is_harmonic_oscillator_visible()
        assert board.get_harmonic_oscillator() is None

        # Show
        sim = board.show_harmonic_oscillator(True, pos=(200.0, 100.0))
        assert sim is not None
        assert board.is_harmonic_oscillator_visible()
        assert board.get_harmonic_oscillator() is sim
        assert board.get_harmonic_oscillator_item() is sim

        # Lasso exclusion check
        items_in_box = board.get_items_in_rect(QRectF(150.0, 80.0, 800.0, 600.0))
        assert board.get_harmonic_oscillator_item() not in items_in_box

        # Toggle to hide
        visible = board.toggle_harmonic_oscillator()
        assert not visible
        assert not board.is_harmonic_oscillator_visible()
        assert board.get_harmonic_oscillator() is None
        assert board.get_harmonic_oscillator_item() is sim

        # Toggle to show
        visible2 = board.toggle_harmonic_oscillator()
        assert visible2
        assert board.is_harmonic_oscillator_visible()
