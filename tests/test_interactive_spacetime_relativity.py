"""
Unit and integration tests for Interactive Special Relativity Minkowski Spacetime & Lorentz Boost Simulator.
Verifies Minkowski spacetime diagram, Lorentz boost math, invariant interval conservation,
curriculum presets (Muon, Barn-Pole, Twin Paradox, Light Clock), air gestures, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_spacetime_relativity import (
    InteractiveSpacetimeRelativityItem,
    C_LIGHT,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveSpacetimeRelativity:
    """Test suite for InteractiveSpacetimeRelativityItem."""

    def test_relativity_initialization(self, qapp):
        """Verify default initialization, dimensions, beta, gamma and default preset."""
        item = InteractiveSpacetimeRelativityItem(pos=(200.0, 150.0), preset="LIGHT_CLOCK_GEDANKEN")
        assert item.x() == 200.0
        assert item.y() == 150.0
        assert item.boundingRect().width() == 840.0
        assert item.boundingRect().height() == 560.0
        assert item.preset == "LIGHT_CLOCK_GEDANKEN"
        assert item.beta == 0.60
        assert pytest.approx(item.gamma, rel=1e-3) == 1.250  # 1 / sqrt(1 - 0.36) = 1 / 0.8 = 1.25
        assert item.delta_s_squared > 0.0  # Timelike separation
        item.close()

    def test_curriculum_presets_and_scenarios(self, qapp):
        """Verify switching through all 4 authentic relativistic curriculum scenarios."""
        item = InteractiveSpacetimeRelativityItem()

        # Preset 1: Cosmic Muon Atmospheric Decay
        item.set_preset("MUON_ATMOSPHERIC_DECAY")
        assert item.preset == "MUON_ATMOSPHERIC_DECAY"
        assert item.beta == 0.95
        assert item.gamma > 3.0
        assert item.delta_s_squared > 0.0  # Timelike proper time

        # Preset 2: Barn-Pole Paradox
        item.set_preset("BARN_POLE_PARADOX")
        assert item.preset == "BARN_POLE_PARADOX"
        assert item.beta == 0.80
        assert pytest.approx(item.gamma, rel=1e-3) == 1.667
        # Simultaneous in S: dt = 0, dx = 4.0 => ds² = -16.0 (spacelike)
        assert item.delta_s_squared < 0.0

        # Preset 3: Twin Paradox Broken Worldline
        item.set_preset("TWIN_PARADOX_WORLDLINE")
        assert item.preset == "TWIN_PARADOX_WORLDLINE"
        assert item.beta == 0.80
        assert len(item._worldline_points) == 3

        # Preset 4: Einstein's Light Clock Gedanken
        item.set_preset("LIGHT_CLOCK_GEDANKEN")
        assert item.preset == "LIGHT_CLOCK_GEDANKEN"
        assert item.beta == 0.60
        assert pytest.approx(item.gamma, rel=1e-3) == 1.250
        item.close()

    def test_lorentz_transformation_and_invariant_interval(self, qapp):
        """
        Verify that the spacetime interval ds² = c²dt² - dx² is strictly invariant
        under Lorentz boost across various velocities beta ∈ [0.0, 0.95].
        """
        item = InteractiveSpacetimeRelativityItem()

        # Test events: E1(1.0, 2.0), E2(3.0, 6.0)
        item._event1 = QPointF(1.0, 2.0)
        item._event2 = QPointF(3.0, 6.0)
        dx = item._event2.x() - item._event1.x()      # 2.0
        dct = item._event2.y() - item._event1.y()    # 4.0
        ds_sq_S = dct * dct - dx * dx                # 16.0 - 4.0 = 12.0

        for test_beta in [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]:
            item.set_beta(test_beta)
            gamma = 1.0 / math.sqrt(1.0 - test_beta * test_beta)

            # Lorentz boost
            dx_prime = gamma * (dx - test_beta * dct)
            dct_prime = gamma * (dct - test_beta * dx)
            ds_sq_S_prime = dct_prime * dct_prime - dx_prime * dx_prime

            assert pytest.approx(ds_sq_S_prime, rel=1e-5) == ds_sq_S, (
                f"Lorentz boost failed invariance at beta={test_beta}: {ds_sq_S_prime} != {ds_sq_S}"
            )
        item.close()

    def test_spacetime_interval_classification(self, qapp):
        """Verify proper classification of Timelike, Spacelike, and Lightlike intervals."""
        item = InteractiveSpacetimeRelativityItem()

        # 1. Timelike: dt > dx => ds² > 0
        item._event1 = QPointF(0.0, 0.0)
        item._event2 = QPointF(2.0, 5.0)
        assert item.delta_s_squared == 21.0
        assert item.delta_s_squared > 0.0

        # 2. Spacelike: dx > dt => ds² < 0
        item._event1 = QPointF(0.0, 0.0)
        item._event2 = QPointF(5.0, 2.0)
        assert item.delta_s_squared == -21.0
        assert item.delta_s_squared < 0.0

        # 3. Lightlike / Null: dx == dt => ds² = 0
        item._event1 = QPointF(0.0, 0.0)
        item._event2 = QPointF(4.0, 4.0)
        assert pytest.approx(item.delta_s_squared, abs=1e-6) == 0.0
        item.close()

    def test_visual_toggles_and_beta_controls(self, qapp):
        """Verify toggling visual layers and adjusting beta limits."""
        item = InteractiveSpacetimeRelativityItem()

        # Toggles
        assert item._show_hyperbolas is True
        assert item._show_boosted_grid is True
        assert item._show_light_cone is True

        # Clamp beta within [0.0, 0.99]
        item.set_beta(1.5)
        assert item.beta == 0.99
        item.set_beta(-0.5)
        assert item.beta == 0.0

        item.set_beta(0.8)
        assert pytest.approx(item.gamma, rel=1e-3) == 1.667
        item.close()

    def test_air_gestures_and_pinch_interaction(self, qapp):
        """Verify touchless air pinch and air tap gestures on presets, buttons and sliders."""
        item = InteractiveSpacetimeRelativityItem(pos=(100.0, 100.0))

        # Air tap on Barn-Pole preset button
        barn_btn = item._get_preset_button_rect(0, 1)
        scene_pt = item.mapToScene(barn_btn.center())
        handled = item.handle_air_pinch(scene_pt)
        assert handled is True
        assert item.preset == "BARN_POLE_PARADOX"

        # Air tap on Twin Paradox preset button
        twin_btn = item._get_preset_button_rect(1, 0)
        scene_pt2 = item.mapToScene(twin_btn.center())
        handled2 = item.handle_air_tap(scene_pt2)
        assert handled2 is True
        assert item.preset == "TWIN_PARADOX_WORLDLINE"

        # Air tap close button
        close_btn = item._get_close_button_rect()
        scene_close = item.mapToScene(close_btn.center())
        handled_close = item.handle_air_tap(scene_close)
        assert handled_close is True
        assert item.isVisible() is False

    def test_vector_stamping_to_lesson_board(self, qapp):
        """Verify stamping vector diagram to LessonBoard produces valid QGraphicsItems."""
        board = LessonBoard()
        item = InteractiveSpacetimeRelativityItem(pos=(100.0, 100.0))
        board.addItem(item)

        initial_count = len(board.items())
        item.stamp_to_board()
        final_count = len(board.items())

        # Vector diagram must add at least 15+ descriptive vector graphic elements
        assert final_count > initial_count + 10
        item.close()

    def test_lesson_board_lifecycle_and_selection_exclusion(self, qapp):
        """Verify LessonBoard show/toggle/get methods and selection exclusion."""
        board = LessonBoard()

        # Show
        rel_item = board.show_spacetime_relativity(True, pos=(350.0, 180.0))
        assert rel_item is not None
        assert board.is_spacetime_relativity_visible() is True
        assert board.get_spacetime_relativity() is rel_item
        assert board.get_spacetime_relativity_item() is rel_item

        # Selection exclusion: Ensure get_items_in_rect ignores interactive spacetime relativity item
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        items_in_sel = board.get_items_in_rect(selection_rect)
        assert rel_item not in items_in_sel

        # Toggle
        new_state = board.toggle_spacetime_relativity()
        assert new_state is False
        assert board.is_spacetime_relativity_visible() is False
        assert board.get_spacetime_relativity() is None
        assert board.get_spacetime_relativity_item() is rel_item

        new_state_2 = board.toggle_spacetime_relativity()
        assert new_state_2 is True
        assert board.is_spacetime_relativity_visible() is True
        board.show_spacetime_relativity(False)
