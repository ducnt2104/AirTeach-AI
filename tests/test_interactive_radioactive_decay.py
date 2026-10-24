"""
Unit tests for Interactive Nuclear Physics & Radioactive Decay Chain Law Simulator.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_radioactive_decay import (
    InteractiveRadioactiveDecayItem,
    ISOTOPE_SPECS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def decay_sim(qapp):
    item = InteractiveRadioactiveDecayItem(
        pos=(200.0, 150.0),
        width=780.0,
        height=500.0,
        isotope="CARBON_14",
    )
    yield item
    item.close()


class TestInteractiveRadioactiveDecay:
    """Test suite for InteractiveRadioactiveDecayItem."""

    def test_decay_initialization(self, decay_sim: InteractiveRadioactiveDecayItem):
        assert decay_sim.isotope == "CARBON_14"
        assert decay_sim.total_nuclei == 100
        assert decay_sim.undecayed_count == 100
        assert decay_sim.decayed_count == 0
        assert decay_sim.elapsed_half_lives == pytest.approx(0.0, abs=1e-5)
        assert decay_sim.theoretical_remaining_count == pytest.approx(100.0, abs=1e-5)
        assert decay_sim.theoretical_fraction == pytest.approx(1.0, abs=1e-5)
        assert decay_sim.is_running is True

    def test_isotope_presets(self, decay_sim: InteractiveRadioactiveDecayItem):
        expected_isotopes = ["CARBON_14", "RADON_222", "IODINE_131", "URANIUM_238", "COBALT_60"]

        for iso in expected_isotopes:
            decay_sim.set_isotope(iso)
            assert decay_sim.isotope == iso
            assert decay_sim.undecayed_count == 100
            assert decay_sim.elapsed_half_lives == pytest.approx(0.0, abs=1e-5)
            spec = ISOTOPE_SPECS[iso]
            assert len(spec[0]) > 0  # Name
            assert len(spec[1]) > 0  # Reaction
            assert spec[4] > 0.0     # Half-life in seconds

    def test_advance_one_half_life(self, decay_sim: InteractiveRadioactiveDecayItem):
        decay_sim.reset_simulation()
        decay_sim.advance_one_half_life()

        assert decay_sim.elapsed_half_lives == pytest.approx(1.0, abs=1e-5)
        # In a sample of N0 = 100 with p = 0.5, mean is 50, std dev is sqrt(100*0.5*0.5) = 5.
        # 4-sigma interval is [30, 70]
        assert 25 <= decay_sim.undecayed_count <= 75
        assert decay_sim.decayed_count == 100 - decay_sim.undecayed_count

    def test_multi_half_life_decay_trend(self, decay_sim: InteractiveRadioactiveDecayItem):
        decay_sim.reset_simulation()
        counts: list[int] = [decay_sim.undecayed_count]

        for _ in range(4):
            decay_sim.advance_one_half_life()
            counts.append(decay_sim.undecayed_count)

        # Monotonically non-increasing
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1]

        # After 4 half-lives, expected is 100 * (1/16) ≈ 6.25 nuclei.
        # Should be <= 20
        assert counts[-1] <= 22

    def test_monte_carlo_time_step(self, decay_sim: InteractiveRadioactiveDecayItem):
        decay_sim.reset_simulation()
        init_undec = decay_sim.undecayed_count

        # Step by 0.5 half-life
        decay_sim._monte_carlo_step(0.50)

        assert decay_sim.elapsed_half_lives == pytest.approx(0.50, abs=1e-4)
        # Should have had some decays
        assert decay_sim.undecayed_count < init_undec
        assert decay_sim.decayed_count > 0

    def test_speed_and_reset_controls(self, decay_sim: InteractiveRadioactiveDecayItem):
        decay_sim.set_speed(2.0)
        assert decay_sim.simulation_speed == pytest.approx(2.0, abs=1e-3)

        decay_sim.set_speed(15.0)  # Above max
        assert decay_sim.simulation_speed == pytest.approx(10.0, abs=1e-3)

        decay_sim.set_speed(0.1)   # Below min
        assert decay_sim.simulation_speed == pytest.approx(0.5, abs=1e-3)

        # Play/Pause toggle
        running_init = decay_sim.is_running
        decay_sim.toggle_play_pause()
        assert decay_sim.is_running != running_init
        decay_sim.toggle_play_pause()
        assert decay_sim.is_running == running_init

        # Advance and reset
        decay_sim.advance_one_half_life()
        assert decay_sim.undecayed_count < 100
        decay_sim.reset_simulation()
        assert decay_sim.undecayed_count == 100
        assert decay_sim.elapsed_half_lives == pytest.approx(0.0, abs=1e-5)

    def test_air_gestures_and_mouse_interactions(self, decay_sim: InteractiveRadioactiveDecayItem):
        # Air Tap on Radon-222
        pt_rn = decay_sim.mapToScene(decay_sim._rect_btn_rn222.center())
        assert decay_sim.handle_air_tap(pt_rn) is True
        assert decay_sim.isotope == "RADON_222"

        # Air Tap on 1 T1/2 step
        pt_step = decay_sim.mapToScene(decay_sim._rect_btn_step_thalf.center())
        assert decay_sim.handle_air_tap(pt_step) is True
        assert decay_sim.elapsed_half_lives == pytest.approx(1.0, abs=1e-5)

        # Air Tap on Reset
        pt_reset = decay_sim.mapToScene(decay_sim._rect_btn_reset.center())
        assert decay_sim.handle_air_tap(pt_reset) is True
        assert decay_sim.elapsed_half_lives == pytest.approx(0.0, abs=1e-5)

        # Air Pinch on Plot area (scrubbing time)
        # Midpoint of plot corresponds to ~2 T1/2
        plot_center = QPointF(decay_sim._plot_x + decay_sim._plot_w * 0.5, decay_sim._plot_y + decay_sim._plot_h * 0.5)
        scene_pt = decay_sim.mapToScene(plot_center)
        consumed = decay_sim.handle_air_pinch(scene_pt)
        assert consumed is True
        assert decay_sim.elapsed_half_lives > 1.5

    def test_stamping_and_lesson_board_integration(self, qapp):
        board = LessonBoard()

        # Check visibility toggle
        assert board.is_radioactive_decay_visible() is False
        item = board.show_radioactive_decay(True)
        assert item is not None
        assert board.is_radioactive_decay_visible() is True
        assert board.get_radioactive_decay() is item

        # Toggle visibility
        is_now_vis = board.toggle_radioactive_decay()
        assert is_now_vis is False
        assert board.is_radioactive_decay_visible() is False

        # Toggle back on
        board.toggle_radioactive_decay()
        assert board.is_radioactive_decay_visible() is True

        # Eraser exclusion check
        sim_item = board.get_radioactive_decay_item()
        sim_rect = sim_item.sceneBoundingRect()
        items_in_rect = board.get_items_in_rect(sim_rect)
        assert sim_item not in items_in_rect

        # Stamping vector diagram
        stamped_items = sim_item.stamp_to_canvas(board)
        assert len(stamped_items) > 0

        # Clean up
        sim_item.close()
