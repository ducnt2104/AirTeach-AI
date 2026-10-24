"""
Tests for Interactive Biophysics & Physiology: Hodgkin-Huxley Action Potential Simulator.
Verifies authentic numerical integration, All-or-None threshold law, overshoot peak,
repetitive spike trains, pharmacological TTX/TEA blocks, paired-pulse refractory period,
touchless air gestures, rendering, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_action_potential import (
    E_K,
    E_NA,
    PRESETS,
    V_REST,
    ActionPotentialPreset,
    InteractiveActionPotentialItem,
    Neurotoxin,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import BiologyTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestHodgkinHuxleyBiophysics:
    """Authentic mathematical biophysics and electrophysiology verification."""

    def test_resting_membrane_potential(self, qapp):
        item = InteractiveActionPotentialItem()
        item.reset_state()
        # Initial resting potential must be close to -65 mV
        assert math.isclose(item.vm, V_REST, abs_tol=1.0)
        # Steady-state gating variables in valid [0, 1] bounds
        assert 0.0 < item._m < 0.2
        assert 0.4 < item._h < 0.8
        assert 0.2 < item._n < 0.5

    def test_suprathreshold_all_or_none_action_potential(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("STANDARD_ACTION_POTENTIAL")

        # Must fire full spike with peak overshoot > +10 mV
        v_peak = max(item._vm_history)
        assert v_peak > 15.0, f"Expected action potential overshoot peak > +15 mV, got {v_peak} mV"
        assert v_peak < E_NA + 5.0, f"Peak exceeded Sodium Nernst limit: {v_peak} mV"

        # Check repolarization and hyperpolarization undershoot below resting
        v_min_after_peak = min(item._vm_history)
        assert v_min_after_peak < V_REST - 2.0  # Hyperpolarization undershoot approaching E_K

    def test_subthreshold_stimulation_failure(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("SUBTHRESHOLD_STIMULATION")

        # 3 μA stimulus must NOT fire action potential (peak stays well below 0 mV)
        v_peak = max(item._vm_history)
        assert v_peak < -45.0, f"Subthreshold stimulus should not fire action potential, reached {v_peak} mV"

    def test_repetitive_spike_train(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("REPETITIVE_SPIKE_TRAIN")

        # Sustained 12 μA current must produce multiple rhythmic action potentials
        # Count peaks above 0 mV
        peaks = 0
        for i in range(1, len(item._vm_history) - 1):
            if item._vm_history[i] > 0.0:
                if item._vm_history[i] > item._vm_history[i - 1] and item._vm_history[i] > item._vm_history[i + 1]:
                    peaks += 1
        assert peaks >= 2, f"Expected multiple spikes in train, found {peaks}"

    def test_tetrodotoxin_sodium_channel_block(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("TETRODOTOXIN_TTX_BLOCK")

        assert item.toxin == Neurotoxin.TTX
        # Even with strong 15 μA stimulus, TTX blocks Na⁺ channels, preventing spike
        v_peak = max(item._vm_history)
        assert v_peak < -20.0, f"TTX should completely abolish action potential spike, reached {v_peak} mV"

    def test_paired_pulse_refractory_period(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("REFRACTORY_PERIOD_PROBING")

        # In paired pulse with short 4 ms delay, second pulse arrives during refractory period
        # Count peaks above 0 mV (only the first pulse fires)
        peaks = 0
        for i in range(1, len(item._vm_history) - 1):
            if item._vm_history[i] > 0.0:
                if item._vm_history[i] > item._vm_history[i - 1] and item._vm_history[i] > item._vm_history[i + 1]:
                    peaks += 1
        assert peaks == 1, f"Expected only 1 spike due to refractory period, got {peaks}"


class TestActionPotentialPresetsAndControls:
    """Verifies presets, toxin controls, steppers, and air gestures."""

    def test_all_presets_load(self, qapp):
        item = InteractiveActionPotentialItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            assert item.active_preset.code == code
            assert len(item._vm_history) > 10

    def test_stimulus_and_toxin_controls(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("STANDARD_ACTION_POTENTIAL")

        init_i = item._i_inj_ua
        item.step_stimulus_current(4.0)
        assert math.isclose(item._i_inj_ua, init_i + 4.0)

        item.set_toxin(Neurotoxin.TEA)
        assert item.toxin == Neurotoxin.TEA
        item.set_toxin(Neurotoxin.NONE)
        assert item.toxin == Neurotoxin.NONE

        # Animation toggle
        assert item._is_running is True
        item.toggle_animation()
        assert item._is_running is False
        item.toggle_animation()
        assert item._is_running is True

    def test_air_pinch_button_interactions(self, qapp):
        item = InteractiveActionPotentialItem()
        item.load_preset("STANDARD_ACTION_POTENTIAL")

        # Paint once to populate buttons dict
        img = QImage(960, 650, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()

        assert "btn_inject" in item._buttons
        assert "btn_stamp" in item._buttons

        # Air pinch on inject button
        inject_rect = item._buttons["btn_inject"]
        handled = item.handle_air_pinch(item.mapToScene(inject_rect.center()))
        assert handled is True

        # Air pinch on stamp button
        stamp_handled = item.handle_air_pinch(item.mapToScene(item._buttons["btn_stamp"].center()))
        assert stamp_handled is True

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveActionPotentialItem()
        img = QImage(960, 650, QImage.Format.Format_ARGB32)

        for code in PRESETS.keys():
            item.load_preset(code)
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle."""

    def test_biology_tools_action_potential_diagram(self, qapp):
        items = BiologyTools.create_action_potential_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="STANDARD_ACTION_POTENTIAL",
            v_peak_mv=32.4,
            i_inj_ua=10.0,
            toxin="none",
            temp_c=6.3,
        )
        assert len(items) >= 5
        assert any("HODGKIN-HUXLEY" in getattr(item, "toPlainText", lambda: "")() for item in items)

    def test_lesson_board_action_potential_lifecycle(self, qapp):
        board = LessonBoard()

        # Initially not visible
        assert board.is_action_potential_visible() is False
        assert board.get_action_potential() is None

        # Show
        item = board.show_action_potential(True)
        assert item is not None
        assert board.is_action_potential_visible() is True
        assert board.get_action_potential() is item

        # Toggle hide
        new_state = board.toggle_action_potential()
        assert new_state is False
        assert board.is_action_potential_visible() is False

        # Toggle show
        new_state = board.toggle_action_potential()
        assert new_state is True
        assert board.is_action_potential_visible() is True

        # Ensure item is excluded from selection box in get_items_in_rect
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        selected_items = board.get_items_in_rect(selection_rect)
        assert item not in selected_items
