# tests/test_interactive_kepler_orbits.py
"""
Unit and integration tests for InteractiveKeplerOrbitsItem,
PhysicsTools.create_kepler_orbits_diagram, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF

from src.ui.interactive_kepler_orbits import (
    InteractiveKeplerOrbitsItem,
    OrbitPreset,
    ORBIT_PRESETS,
    MU_SUN,
    MU_EARTH,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveKeplerOrbits:
    def test_kepler_presets_physical_validity(self, qapp):
        """Verify all celestial presets conform to orbital mechanics laws."""
        assert len(ORBIT_PRESETS) == 5
        assert "EARTH_AROUND_SUN" in ORBIT_PRESETS
        assert "HALLEY_COMET_EXTREME_ECCENTRICITY" in ORBIT_PRESETS
        assert "ISS_LOW_EARTH_ORBIT_LEO" in ORBIT_PRESETS
        assert "GEOSTATIONARY_ORBIT_GEO" in ORBIT_PRESETS
        assert "MOLNIYA_HIGH_INCLINATION_COMM" in ORBIT_PRESETS

        for code, preset in ORBIT_PRESETS.items():
            assert preset.semi_major_axis_km > 0.0
            assert 0.0 <= preset.eccentricity < 1.0
            assert preset.periapsis_km > 0.0
            assert preset.apoapsis_km >= preset.semi_major_axis_km >= preset.periapsis_km
            assert preset.period_seconds > 0.0
            assert preset.periapsis_velocity_km_s >= preset.apoapsis_velocity_km_s
            assert preset.accent_color.isValid()
            assert preset.central_color.isValid()

    def test_kepler_first_law_ellipse_geometry(self, qapp):
        """Verify ellipse relation a^2 = b^2 + c^2 where c = a * e."""
        item = InteractiveKeplerOrbitsItem()
        item.set_orbit_preset("HALLEY_COMET_EXTREME_ECCENTRICITY")

        a = item.semi_major_axis_km
        e = item.eccentricity
        b = item.semi_minor_axis_km
        c = a * e

        assert math.isclose(a ** 2, b ** 2 + c ** 2, rel_tol=1e-3)
        assert math.isclose(item.periapsis_km, a * (1.0 - e), rel_tol=1e-4)
        assert math.isclose(item.apoapsis_km, a * (1.0 + e), rel_tol=1e-4)

    def test_kepler_second_law_and_vis_viva(self, qapp):
        """Verify conservation of angular momentum h = r_p * v_p = r_a * v_a."""
        item = InteractiveKeplerOrbitsItem()
        item.set_orbit_preset("HALLEY_COMET_EXTREME_ECCENTRICITY")
        preset = item._preset

        rp = preset.periapsis_km
        ra = preset.apoapsis_km
        vp = preset.periapsis_velocity_km_s
        va = preset.apoapsis_velocity_km_s

        h_periapsis = rp * vp
        h_apoapsis = ra * va

        # Angular momentum must be conserved across orbit
        assert math.isclose(h_periapsis, h_apoapsis, rel_tol=1e-3)

        # Vis-Viva calculation at periapsis and apoapsis
        vp_calc = item.orbital_velocity_vis_viva_km_s(rp)
        va_calc = item.orbital_velocity_vis_viva_km_s(ra)
        assert math.isclose(vp_calc, vp, rel_tol=1e-3)
        assert math.isclose(va_calc, va, rel_tol=1e-3)

    def test_kepler_third_law_harmonic_ratio(self, qapp):
        """Verify Kepler's Harmonic Law: a^3 / T^2 = mu / (4 * pi^2)."""
        item = InteractiveKeplerOrbitsItem()

        # Check Earth around Sun
        item.set_orbit_preset("EARTH_AROUND_SUN")
        a_earth = item.semi_major_axis_km
        t_earth = item.orbital_period_s
        ratio_sun = (a_earth ** 3) / (t_earth ** 2)
        expected_sun = MU_SUN / (4.0 * (math.pi ** 2))
        assert math.isclose(ratio_sun, expected_sun, rel_tol=1e-3)

        # Check ISS around Earth
        item.set_orbit_preset("ISS_LOW_EARTH_ORBIT_LEO")
        a_iss = item.semi_major_axis_km
        t_iss = item.orbital_period_s
        ratio_earth = (a_iss ** 3) / (t_iss ** 2)
        expected_earth = MU_EARTH / (4.0 * (math.pi ** 2))
        assert math.isclose(ratio_earth, expected_earth, rel_tol=1e-3)

    def test_kepler_equation_numerical_solver(self, qapp):
        """Verify Newton-Raphson solver for M = E - e*sin(E)."""
        item = InteractiveKeplerOrbitsItem()
        item.set_eccentricity(0.50)

        test_mean_anomalies = [0.0, 0.5, 1.2, math.pi / 2.0, math.pi, 2.0 * math.pi * 0.75]
        for m in test_mean_anomalies:
            e_sol = item.solve_kepler_eccentric_anomaly(m)
            # Reconstruct M = E - e*sin(E)
            m_reconstructed = e_sol - 0.50 * math.sin(e_sol)
            assert math.isclose(m % (2.0 * math.pi), m_reconstructed % (2.0 * math.pi), abs_tol=1e-7)

    def test_interactive_controls_and_buttons(self, qapp):
        """Verify buttons: eccentricity step, semi-major axis scale, circularize, reset."""
        item = InteractiveKeplerOrbitsItem()
        item.set_orbit_preset("EARTH_AROUND_SUN")

        init_e = item._eccentricity
        item.step_eccentricity(0.05)
        assert math.isclose(item._eccentricity, init_e + 0.05, rel_tol=1e-3)

        item.set_eccentricity(0.0)
        assert item._eccentricity == 0.0
        assert math.isclose(item.semi_major_axis_km, item.semi_minor_axis_km, rel_tol=1e-3)

        orig_a = item._semi_major_axis_km
        item.step_semi_major_axis(1.10)
        assert math.isclose(item._semi_major_axis_km, orig_a * 1.10, rel_tol=1e-3)

        # Toggle equal areas
        assert item._show_equal_areas
        item.toggle_equal_areas()
        assert not item._show_equal_areas

        # Toggle animation
        assert item._is_animating
        item.toggle_animation()
        assert not item._is_animating

        # Reset
        item.reset_orbit()
        assert math.isclose(item._eccentricity, 0.0167, rel_tol=1e-3)

    def test_stamping_and_vector_diagram(self, qapp):
        """Verify stamp request produces complete vector diagram items."""
        item = InteractiveKeplerOrbitsItem()
        stamped_data = {}

        def on_stamp(data):
            stamped_data.update(data)

        item.stamp_requested.connect(on_stamp)
        item.request_stamp()

        assert "preset_code" in stamped_data
        assert "semi_major_axis_km" in stamped_data
        assert "eccentricity" in stamped_data
        assert "orbital_period_s" in stamped_data

        items = PhysicsTools.create_kepler_orbits_diagram(
            center=(400.0, 300.0),
            preset_code=stamped_data["preset_code"],
            title_vi=stamped_data["title_vi"],
            body_name=stamped_data["body_name"],
            central_body=stamped_data["central_body"],
            semi_major_axis_km=stamped_data["semi_major_axis_km"],
            eccentricity=stamped_data["eccentricity"],
            semi_minor_axis_km=stamped_data["semi_minor_axis_km"],
            periapsis_km=stamped_data["periapsis_km"],
            apoapsis_km=stamped_data["apoapsis_km"],
            orbital_period_s=stamped_data["orbital_period_s"],
            periapsis_velocity_km_s=stamped_data["periapsis_velocity_km_s"],
            apoapsis_velocity_km_s=stamped_data["apoapsis_velocity_km_s"],
        )

        assert len(items) >= 15
        has_title = any(
            getattr(itm, "toPlainText", lambda: "")().startswith("🪐 QUỸ ĐẠO HẤP DẪN")
            for itm in items
        )
        assert has_title

    def test_lesson_board_integration(self, qapp):
        """Verify LessonBoard visibility toggling and stamping lifecycle."""
        board = LessonBoard()

        assert not board.is_kepler_orbits_visible()
        item = board.show_kepler_orbits(True, pos=(150.0, 80.0))
        assert item is not None
        assert board.is_kepler_orbits_visible()
        assert board.get_kepler_orbits() is item

        # Toggle visibility
        is_vis = board.toggle_kepler_orbits()
        assert not is_vis
        assert not board.is_kepler_orbits_visible()

        # Re-show
        is_vis = board.toggle_kepler_orbits()
        assert is_vis
        assert board.is_kepler_orbits_visible()

        # Stamp onto board
        initial_count = len(board.items())
        item.request_stamp()
        assert len(board.items()) > initial_count
