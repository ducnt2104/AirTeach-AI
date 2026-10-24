"""
Unit and integration tests for Interactive Geometric Optics Thin Lens & Optical Bench Simulator.
Verifies Gaussian lens formula 1/f = 1/d + 1/d', lateral magnification k = -d'/d,
real vs virtual image classification, 3 principal rays, presets, air gestures, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_thin_lens import InteractiveThinLensBenchItem
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveThinLens:
    """Test suite for InteractiveThinLensBenchItem."""

    def test_thin_lens_initialization(self, qapp):
        """Verify default projector lens parameters, real inverted image distance and magnification."""
        item = InteractiveThinLensBenchItem(pos=(200.0, 150.0), preset="PROJECTOR_LENS")
        assert item.x() == 200.0
        assert item.y() == 150.0
        assert item.boundingRect().width() == 840.0
        assert item.boundingRect().height() == 560.0
        assert item.preset == "PROJECTOR_LENS"
        assert item.lens_type == "CONVERGING"
        assert item.focal_length_cm == 10.0
        assert item.object_d_cm == 15.0
        assert item.object_h_cm == 5.0

        # Projector lens (f < d < 2f): d' = (15 * 10) / (15 - 10) = 30.0 cm (Real image)
        assert pytest.approx(item.image_d_prime_cm, rel=1e-3) == 30.0
        # Magnification k = -d' / d = -30 / 15 = -2.00 (Inverted, 2x magnified)
        assert pytest.approx(item.magnification_k, rel=1e-3) == -2.00
        # Image height h' = k * h = -2.0 * 5.0 = -10.0 cm
        assert pytest.approx(item.image_h_prime_cm, rel=1e-3) == -10.0
        item.close()

    def test_curriculum_presets_and_imaging_modes(self, qapp):
        """Verify switching through all 4 authentic geometric optics presets."""
        item = InteractiveThinLensBenchItem()

        # Preset 1: Magnifying Glass (d < f => virtual upright magnified image)
        item.set_preset("MAGNIFYING_GLASS")
        assert item.preset == "MAGNIFYING_GLASS"
        assert item.lens_type == "CONVERGING"
        assert item.object_d_cm < item.focal_length_cm
        # d' must be negative (virtual image)
        assert item.image_d_prime_cm < 0.0
        # k must be positive and > 1.0 (upright magnified)
        assert item.magnification_k > 1.0

        # Preset 2: Camera Objective (d > 2f => real inverted diminished image)
        item.set_preset("CAMERA_OBJECTIVE")
        assert item.preset == "CAMERA_OBJECTIVE"
        assert item.object_d_cm > 2.0 * item.focal_length_cm
        # d' must be positive (real image)
        assert item.image_d_prime_cm > 0.0
        # k must be negative with magnitude < 1.0 (diminished)
        assert -1.0 < item.magnification_k < 0.0

        # Preset 3: Diverging Eyeglass (kính cận: f < 0 => virtual upright diminished image)
        item.set_preset("DIVERGING_EYEGLASS")
        assert item.preset == "DIVERGING_EYEGLASS"
        assert item.lens_type == "DIVERGING"
        assert item.focal_length_cm < 0.0
        # d' must be negative
        assert item.image_d_prime_cm < 0.0
        # k must be between 0 and 1 (upright diminished)
        assert 0.0 < item.magnification_k < 1.0

        # Preset 4: Projector Lens
        item.set_preset("PROJECTOR_LENS")
        assert item.preset == "PROJECTOR_LENS"
        assert item.lens_type == "CONVERGING"
        item.close()

    def test_gaussian_equation_consistency(self, qapp):
        """Verify 1/f = 1/d + 1/d' across multiple parameter sets."""
        item = InteractiveThinLensBenchItem()

        for d_test in [14.0, 20.0, 25.0, 35.0]:
            item.set_object_distance(d_test)
            f_val = item.focal_length_cm
            d_val = item.object_d_cm
            dp_val = item.image_d_prime_cm

            # 1/f ≈ 1/d + 1/d'
            inv_f = 1.0 / f_val
            inv_sum = (1.0 / d_val) + (1.0 / dp_val)
            assert pytest.approx(inv_sum, rel=1e-4) == inv_f
        item.close()

    def test_lens_mode_switching(self, qapp):
        """Verify switching between converging and diverging lens modes."""
        item = InteractiveThinLensBenchItem()

        item.set_lens_type("DIVERGING")
        assert item.lens_type == "DIVERGING"
        assert item.focal_length_cm < 0.0

        item.set_lens_type("CONVERGING")
        assert item.lens_type == "CONVERGING"
        assert item.focal_length_cm > 0.0
        item.close()

    def test_sliders_and_parameter_clamping(self, qapp):
        """Verify parameter setters and clamping."""
        item = InteractiveThinLensBenchItem()

        # Focal length clamp [5, 25]
        item.set_focal_length(50.0)
        assert abs(item.focal_length_cm) == 25.0
        item.set_focal_length(1.0)
        assert abs(item.focal_length_cm) == 5.0
        item.set_focal_length(14.0)
        assert abs(item.focal_length_cm) == 14.0

        # Object distance clamp [4, 40]
        item.set_object_distance(80.0)
        assert item.object_d_cm == 40.0
        item.set_object_distance(1.0)
        assert item.object_d_cm == 4.0

        # Object height clamp [2, 12]
        item.set_object_height(20.0)
        assert item.object_h_cm == 12.0
        item.set_object_height(0.5)
        assert item.object_h_cm == 2.0
        item.close()

    def test_air_gestures_and_pinch_interaction(self, qapp):
        """Verify touchless air pinch and air tap gestures on presets, buttons and sliders."""
        item = InteractiveThinLensBenchItem(pos=(100.0, 100.0))

        # Air tap on Magnifying Glass preset button
        mag_btn = item._get_preset_button_rect(0, 0)
        sc_pt = item.mapToScene(mag_btn.center())
        handled = item.handle_air_pinch(sc_pt)
        assert handled is True
        assert item.preset == "MAGNIFYING_GLASS"

        # Air tap on Diverging mode button
        div_btn = item._get_mode_button_rect("DIVERGING")
        sc_div = item.mapToScene(div_btn.center())
        handled_div = item.handle_air_tap(sc_div)
        assert handled_div is True
        assert item.lens_type == "DIVERGING"

        # Air tap close button
        close_btn = item._get_close_button_rect()
        sc_close = item.mapToScene(close_btn.center())
        handled_close = item.handle_air_tap(sc_close)
        assert handled_close is True
        assert item.isVisible() is False

    def test_vector_stamping_to_lesson_board(self, qapp):
        """Verify stamping vector diagram to LessonBoard produces valid QGraphicsItems."""
        board = LessonBoard()
        item = InteractiveThinLensBenchItem(pos=(100.0, 100.0))
        board.addItem(item)

        initial_count = len(board.items())
        item.stamp_to_board()
        final_count = len(board.items())

        # Stamped vector diagram contains banner, axis, lens, focal points, object, image, rays, and metrics (> 15 items)
        assert final_count > initial_count + 12
        item.close()

    def test_lesson_board_lifecycle_and_selection_exclusion(self, qapp):
        """Verify LessonBoard show/toggle/get methods and selection exclusion."""
        board = LessonBoard()

        # Show
        lens_item = board.show_thin_lens(True, pos=(350.0, 180.0))
        assert lens_item is not None
        assert board.is_thin_lens_visible() is True
        assert board.get_thin_lens() is lens_item
        assert board.get_thin_lens_item() is lens_item

        # Selection exclusion: Ensure get_items_in_rect ignores interactive thin lens item
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        items_in_sel = board.get_items_in_rect(selection_rect)
        assert lens_item not in items_in_sel

        # Toggle
        new_state = board.toggle_thin_lens()
        assert new_state is False
        assert board.is_thin_lens_visible() is False
        assert board.get_thin_lens() is None
        assert board.get_thin_lens_item() is lens_item

        new_state_2 = board.toggle_thin_lens()
        assert new_state_2 is True
        assert board.is_thin_lens_visible() is True
        board.show_thin_lens(False)
