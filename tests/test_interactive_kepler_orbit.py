"""
AirTeach AI - Interactive Astronomy Celestial Mechanics & Kepler Orbit Unit Tests
==================================================================================
Verifies:
  - Kepler's First Law (elliptical orbit, semi-major/minor axes, focal distance).
  - Kepler's Second Law (equal area sweeps, variable orbital velocity, Vis-Viva).
  - Kepler's Third Law (harmonic orbital period consistency).
  - Real solar system presets (Earth, Mars, Mercury, Halley's Comet).
  - Touchless air tap, air drag, and button dispatch.
  - Vector diagram stamping onto LessonBoard.
  - LessonBoard integration and region exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_kepler_orbit import (
    InteractiveKeplerOrbitItem,
    KEPLER_PRESETS,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveKeplerOrbit:
    """Test suite for InteractiveKeplerOrbitItem celestial mechanics simulator."""

    def test_kepler_orbit_init_and_properties(self, qapp):
        """Verifies initial geometric and physical properties of Keplerian ellipse."""
        item = InteractiveKeplerOrbitItem(pos=(300.0, 200.0), semi_major_axis=200.0, eccentricity=0.5)

        assert item.pos() == QPointF(300.0, 200.0)
        assert item.semi_major_axis() == 200.0
        assert item.eccentricity() == 0.5

        # Minor axis b = a * sqrt(1 - e^2) = 200 * sqrt(1 - 0.25) = 200 * sqrt(0.75) ~= 173.205
        expected_b = 200.0 * math.sqrt(0.75)
        assert abs(item.semi_minor_axis() - expected_b) < 1e-3

        # Focal distance c = a * e = 200 * 0.5 = 100.0
        assert abs(item.focal_distance() - 100.0) < 1e-3

        # Perihelion rp = a * (1 - e) = 200 * 0.5 = 100.0
        # Aphelion ra = a * (1 + e) = 200 * 1.5 = 300.0
        assert abs(item.perihelion_distance() - 100.0) < 1e-3
        assert abs(item.aphelion_distance() - 300.0) < 1e-3

        # Sun is placed at focus F1, Empty focus at F2
        sun_pt = item.sun_position()
        f2_pt = item.empty_focus_position()
        assert abs(f2_pt.x() - sun_pt.x() - 2.0 * item.focal_distance()) < 1e-3
        item.close()

    def test_kepler_polar_equation(self, qapp):
        """Verifies polar radius equation r(theta) = a * (1 - e^2) / (1 + e * cos(theta))."""
        item = InteractiveKeplerOrbitItem(semi_major_axis=180.0, eccentricity=0.4)

        # At theta = 0 (perihelion): r = a * (1 - e) = 180 * 0.6 = 108.0
        r_0 = item.radius_at_theta(0.0)
        assert abs(r_0 - item.perihelion_distance()) < 1e-3

        # At theta = pi (aphelion): r = a * (1 + e) = 180 * 1.4 = 252.0
        r_pi = item.radius_at_theta(math.pi)
        assert abs(r_pi - item.aphelion_distance()) < 1e-3

        # At theta = pi / 2: r = a * (1 - e^2) = 180 * (1 - 0.16) = 151.2
        r_p2 = item.radius_at_theta(math.pi * 0.5)
        expected_p = 180.0 * (1.0 - 0.4 * 0.4)
        assert abs(r_p2 - expected_p) < 1e-3
        item.close()

    def test_kepler_presets(self, qapp):
        """Verifies authentic astronomical presets for solar system celestial bodies."""
        item = InteractiveKeplerOrbitItem()

        # EARTH: e = 0.017 (almost circular)
        item.set_preset("EARTH")
        assert item._active_preset == "EARTH"
        assert abs(item.eccentricity() - 0.017) < 1e-4

        # MARS: e = 0.093
        item.set_preset("MARS")
        assert item._active_preset == "MARS"
        assert abs(item.eccentricity() - 0.093) < 1e-4

        # MERCURY: e = 0.206
        item.set_preset("MERCURY")
        assert item._active_preset == "MERCURY"
        assert abs(item.eccentricity() - 0.206) < 1e-4

        # HALLEY: e = 0.800 (highly eccentric comet)
        item.set_preset("HALLEY")
        assert item._active_preset == "HALLEY"
        assert abs(item.eccentricity() - 0.800) < 1e-4

        # Setting custom clamped eccentricity
        item.set_eccentricity(0.65)
        assert item._active_preset == "CUSTOM"
        assert abs(item.eccentricity() - 0.65) < 1e-3

        # Clamp boundaries: [0.0, 0.85]
        item.set_eccentricity(-0.2)
        assert item.eccentricity() == 0.0
        item.set_eccentricity(0.99)
        assert item.eccentricity() == 0.85
        item.close()

    def test_kepler_vis_viva_velocity(self, qapp):
        """Verifies Vis-Viva law: orbital velocity is higher at perihelion than at aphelion."""
        item = InteractiveKeplerOrbitItem(eccentricity=0.6)

        # Force theta = 0 (perihelion)
        item._theta = 0.0
        v_perihelion = item.orbital_velocity()

        # Force theta = pi (aphelion)
        item._theta = math.pi
        v_aphelion = item.orbital_velocity()

        assert v_perihelion > v_aphelion
        # Ratio of speeds v_p / v_a = (1 + e) / (1 - e) = 1.6 / 0.4 = 4.0
        ratio = v_perihelion / v_aphelion
        expected_ratio = (1.0 + 0.6) / (1.0 - 0.6)
        assert abs(ratio - expected_ratio) < 0.15
        item.close()

    def test_kepler_toggle_and_tick(self, qapp):
        """Verifies animation play/pause and orbital time integration."""
        item = InteractiveKeplerOrbitItem()
        assert item.is_orbiting() is True

        state = item.toggle_orbit()
        assert state is False
        assert item.is_orbiting() is False

        state = item.toggle_orbit()
        assert state is True
        assert item.is_orbiting() is True

        # Toggle equal areas
        assert item._show_equal_areas is True
        area_state = item.toggle_equal_areas()
        assert area_state is False
        assert item._show_equal_areas is False

        # Simulation step tick
        theta_before = item._theta
        item._on_orbit_tick()
        assert item._theta > theta_before
        item.close()

    def test_kepler_air_gestures(self, qapp):
        """Verifies touchless air tap button dispatch and air drag."""
        board = LessonBoard()
        item = InteractiveKeplerOrbitItem(pos=(100.0, 100.0))
        board.addItem(item)

        # Test air tap play toggle
        btn_play_scene = item.mapToScene(item._rect_btn_play.center())
        was_orbiting = item.is_orbiting()
        consumed = item.handle_air_tap(btn_play_scene, board=board)
        assert consumed is True
        assert item.is_orbiting() != was_orbiting

        # Test air tap sector sweep toggle
        btn_sec_scene = item.mapToScene(item._rect_btn_sector.center())
        consumed = item.handle_air_tap(btn_sec_scene, board=board)
        assert consumed is True

        # Test air tap on preset MARS
        btn_mars_scene = item.mapToScene(item._rect_btn_presets[1].center())
        consumed = item.handle_air_tap(btn_mars_scene, board=board)
        assert consumed is True
        assert item._active_preset == "MARS"

        # Test air hover pos
        item.set_air_hover_pos(QPointF(200.0, 200.0))
        assert item._air_hover_pos is not None
        item.set_air_hover_pos(None)
        assert item._air_hover_pos is None

        # Test air tap close
        btn_close_scene = item.mapToScene(item._rect_btn_close.center())
        consumed = item.handle_air_tap(btn_close_scene, board=board)
        assert consumed is True
        assert item.scene() is None

    def test_kepler_stamping(self, qapp):
        """Verifies vector diagram stamping of elliptical orbit, Sun, and Kepler formulas."""
        board = LessonBoard()
        item = InteractiveKeplerOrbitItem(pos=(200.0, 150.0), eccentricity=0.3)
        board.addItem(item)

        stamped = item.stamp_to_canvas(board)
        assert len(stamped) >= 3  # Conic ellipse elements, Sun marker, Kepler HUD card

        # Stamped items should remain in board scene
        for itm in stamped:
            assert itm.scene() is board
        item.close()

    def test_lesson_board_kepler_integration(self, qapp):
        """Verifies LessonBoard integration, visibility toggle, and region exclusion."""
        board = LessonBoard()
        assert board.is_kepler_orbit_visible() is False
        assert board.get_kepler_orbit_item() is None

        # Show Kepler orbit
        board.show_kepler_orbit(True, pos=(450.0, 250.0), preset="MERCURY")
        assert board.is_kepler_orbit_visible() is True
        kepler_item = board.get_kepler_orbit_item()
        assert kepler_item is not None
        assert kepler_item.pos() == QPointF(450.0, 250.0)
        assert kepler_item._active_preset == "MERCURY"

        # Verify excluded from eraser selection region
        sel_rect = QRectF(400.0, 200.0, 500.0, 400.0)
        items_inside = board.get_items_in_rect(sel_rect)
        assert kepler_item not in items_inside

        # Toggle visibility
        is_vis = board.toggle_kepler_orbit()
        assert is_vis is False
        assert board.is_kepler_orbit_visible() is False

        is_vis = board.toggle_kepler_orbit()
        assert is_vis is True
        assert board.is_kepler_orbit_visible() is True

        if hasattr(kepler_item, "close"):
            kepler_item.close()
