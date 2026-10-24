"""
Unit and integration tests for Interactive Wave Optics & Young's Double-Slit Interference Simulator.
Verifies fringe spacing formula i = lambda*D/a, intensity distribution with single-slit diffraction envelope,
wavelength-to-sRGB color physics, curriculum presets, air gestures, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_young_interference import (
    InteractiveYoungInterferenceItem,
    wavelength_to_rgb,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveYoungInterference:
    """Test suite for InteractiveYoungInterferenceItem."""

    def test_young_interference_initialization(self, qapp):
        """Verify default parameters, fringe spacing calculation, and central maximum intensity."""
        item = InteractiveYoungInterferenceItem(pos=(200.0, 150.0), preset="RED_HE_NE_LASER")
        assert item.x() == 200.0
        assert item.y() == 150.0
        assert item.boundingRect().width() == 840.0
        assert item.boundingRect().height() == 560.0
        assert item.preset == "RED_HE_NE_LASER"
        assert item.wavelength_nm == 632.8
        assert item.slit_a_mm == 0.50
        assert item.distance_D_m == 1.50
        assert item.num_slits == 2

        # Expected fringe spacing i = (632.8e-6 * 1.50e3) / 0.50 = 1.8984 mm
        expected_i = (632.8 * 1.50 * 1e-3) / 0.50
        assert pytest.approx(item.fringe_spacing_mm, rel=1e-4) == expected_i

        # Central maximum x = 0 has maximum normalized intensity I = 1.0
        assert pytest.approx(item.calculate_intensity(0.0), rel=1e-4) == 1.0
        item.close()

    def test_curriculum_presets_and_optics_configs(self, qapp):
        """Verify switching through all 5 authentic wave optics curriculum presets."""
        item = InteractiveYoungInterferenceItem()

        # Preset 1: Green DPSS Laser (532 nm)
        item.set_preset("GREEN_DPSS_LASER")
        assert item.preset == "GREEN_DPSS_LASER"
        assert item.wavelength_nm == 532.0
        assert pytest.approx(item.fringe_spacing_mm, rel=1e-4) == (532.0 * 1.50 * 1e-3) / 0.50  # ~1.596 mm

        # Preset 2: Violet Laser (405 nm)
        item.set_preset("VIOLET_LASER")
        assert item.preset == "VIOLET_LASER"
        assert item.wavelength_nm == 405.0
        assert pytest.approx(item.fringe_spacing_mm, rel=1e-4) == (405.0 * 1.50 * 1e-3) / 0.50  # ~1.215 mm
        # Violet fringe spacing is narrower than green/red
        assert item.fringe_spacing_mm < 1.5

        # Preset 3: White Light Dispersion
        item.set_preset("WHITE_LIGHT_RAINBOW")
        assert item.preset == "WHITE_LIGHT_RAINBOW"
        assert item.is_white_light is True

        # Preset 4: Diffraction Grating (N = 5 slits)
        item.set_preset("DIFFRACTION_GRATING")
        assert item.preset == "DIFFRACTION_GRATING"
        assert item.num_slits == 5
        assert item.slit_a_mm == 0.25

        # Preset 5: Red He-Ne Laser
        item.set_preset("RED_HE_NE_LASER")
        assert item.preset == "RED_HE_NE_LASER"
        assert item.wavelength_nm == 632.8
        item.close()

    def test_physical_intensity_and_diffraction_envelope(self, qapp):
        """
        Verify that bright fringes occur at x = k*i with intensity peaks,
        and dark fringes occur at x = (k + 0.5)*i with intensity near zero.
        """
        item = InteractiveYoungInterferenceItem()
        item.set_preset("GREEN_DPSS_LASER")
        i_fringe = item.fringe_spacing_mm

        # Central Bright Fringe (k = 0)
        assert pytest.approx(item.calculate_intensity(0.0), abs=1e-3) == 1.0

        # First Dark Fringe (k = 0 => x = +0.5 * i)
        x_dark_1 = 0.5 * i_fringe
        i_dark_1 = item.calculate_intensity(x_dark_1)
        assert i_dark_1 < 0.05, f"Dark fringe intensity should be near zero, got {i_dark_1}"

        # First Bright Fringe (k = 1 => x = +1.0 * i)
        x_bright_1 = 1.0 * i_fringe
        i_bright_1 = item.calculate_intensity(x_bright_1)
        assert i_bright_1 > 0.85, f"First bright fringe intensity should be high, got {i_bright_1}"

        # Second Dark Fringe (k = 1 => x = +1.5 * i)
        x_dark_2 = 1.5 * i_fringe
        i_dark_2 = item.calculate_intensity(x_dark_2)
        assert i_dark_2 < 0.05, f"Second dark fringe intensity should be near zero, got {i_dark_2}"

        item.close()

    def test_wavelength_to_rgb_spectrum(self, qapp):
        """Verify physical spectral wavelength to sRGB color conversion."""
        # Violet (~405 nm): strong blue and red component, low green
        r_v, g_v, b_v = wavelength_to_rgb(405.0)
        assert b_v > 100
        assert r_v > 0
        assert g_v == 0

        # Cyan / Blue (~480 nm): high blue and green
        r_b, g_b, b_b = wavelength_to_rgb(480.0)
        assert b_b > 200
        assert g_b > 100

        # Green (~532 nm): high green, low red/blue
        r_g, g_g, b_g = wavelength_to_rgb(532.0)
        assert g_g > 200
        assert b_g == 0

        # Yellow (~589 nm Sodium D-line): high red and green
        r_y, g_y, b_y = wavelength_to_rgb(589.0)
        assert r_y > 200
        assert g_y > 150
        assert b_y == 0

        # Red (~650 nm): high red, zero green/blue
        r_r, g_r, b_r = wavelength_to_rgb(650.0)
        assert r_r > 200
        assert g_r == 0
        assert b_r == 0

    def test_sliders_and_parameter_clamping(self, qapp):
        """Verify parameter setters, range clamping, and probe position tracking."""
        item = InteractiveYoungInterferenceItem()

        # Wavelength clamp [380, 780]
        item.set_wavelength(900.0)
        assert item.wavelength_nm == 780.0
        item.set_wavelength(200.0)
        assert item.wavelength_nm == 380.0
        item.set_wavelength(550.0)
        assert item.wavelength_nm == 550.0

        # Slit separation clamp [0.10, 2.00]
        item.set_slit_separation(5.0)
        assert item.slit_a_mm == 2.00
        item.set_slit_separation(0.01)
        assert item.slit_a_mm == 0.10

        # Distance D clamp [0.50, 3.00]
        item.set_distance_D(10.0)
        assert item.distance_D_m == 3.00
        item.set_distance_D(0.10)
        assert item.distance_D_m == 0.50

        # Probe position
        item.set_probe_x(2.50)
        assert item.probe_x_mm == 2.50
        item.close()

    def test_air_gestures_and_pinch_interaction(self, qapp):
        """Verify touchless air pinch and air tap gestures on presets and sliders."""
        item = InteractiveYoungInterferenceItem(pos=(100.0, 100.0))

        # Air tap on Green DPSS preset button
        green_btn = item._get_preset_button_rect(0, 1)
        sc_pt = item.mapToScene(green_btn.center())
        handled = item.handle_air_pinch(sc_pt)
        assert handled is True
        assert item.preset == "GREEN_DPSS_LASER"
        assert item.wavelength_nm == 532.0

        # Air tap on Violet preset button
        violet_btn = item._get_preset_button_rect(1, 0)
        sc_pt2 = item.mapToScene(violet_btn.center())
        handled2 = item.handle_air_tap(sc_pt2)
        assert handled2 is True
        assert item.preset == "VIOLET_LASER"
        assert item.wavelength_nm == 405.0

        # Air tap close button
        close_btn = item._get_close_button_rect()
        sc_close = item.mapToScene(close_btn.center())
        handled_close = item.handle_air_tap(sc_close)
        assert handled_close is True
        assert item.isVisible() is False

    def test_vector_stamping_to_lesson_board(self, qapp):
        """Verify stamping vector diagram to LessonBoard produces valid QGraphicsItems."""
        board = LessonBoard()
        item = InteractiveYoungInterferenceItem(pos=(100.0, 100.0))
        board.addItem(item)

        initial_count = len(board.items())
        item.stamp_to_board()
        final_count = len(board.items())

        # Stamped vector diagram contains banner, apparatus, screen, intensity curve, and metrics (> 15 items)
        assert final_count > initial_count + 12
        item.close()

    def test_lesson_board_lifecycle_and_selection_exclusion(self, qapp):
        """Verify LessonBoard show/toggle/get methods and selection exclusion."""
        board = LessonBoard()

        # Show
        young_item = board.show_young_interference(True, pos=(350.0, 180.0))
        assert young_item is not None
        assert board.is_young_interference_visible() is True
        assert board.get_young_interference() is young_item
        assert board.get_young_interference_item() is young_item

        # Selection exclusion: Ensure get_items_in_rect ignores interactive wave optics item
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        items_in_sel = board.get_items_in_rect(selection_rect)
        assert young_item not in items_in_sel

        # Toggle
        new_state = board.toggle_young_interference()
        assert new_state is False
        assert board.is_young_interference_visible() is False
        assert board.get_young_interference() is None
        assert board.get_young_interference_item() is young_item

        new_state_2 = board.toggle_young_interference()
        assert new_state_2 is True
        assert board.is_young_interference_visible() is True
        board.show_young_interference(False)
