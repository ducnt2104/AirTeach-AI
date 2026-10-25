"""
Unit Tests for Interactive Astrodynamics & Orbital Mechanics: Hohmann Transfer & Gravitational Slingshot
========================================================================================================
Validates authentic astrodynamics, Keplerian orbital mechanics, Hohmann delta-v,
J2 oblateness nodal precession, Molniya critical inclination, touchless air gestures,
canvas stamping, and .airteach serialization.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_orbital_mechanics import (
    PRESETS,
    InteractiveOrbitalMechanicsItem,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def orbital_item(qapp):
    item = InteractiveOrbitalMechanicsItem(pos=(100.0, 100.0))
    scene = QGraphicsScene()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_initial_state(orbital_item):
    """Verify default initial state of Astrodynamics item."""
    assert orbital_item._width == 940.0
    assert orbital_item._height == 650.0
    assert orbital_item._current_preset == "HOHMANN_EARTH_MARS"
    assert orbital_item._scenario_type == "hohmann"
    assert math.isclose(orbital_item._r1, 1.00, rel_tol=1e-3)
    assert math.isclose(orbital_item._r2, 1.524, rel_tol=1e-3)
    assert orbital_item._is_playing is True
    assert orbital_item.pos() == QPointF(100.0, 100.0)


def test_all_five_presets_exist():
    """Verify all 5 curriculum presets exist with valid parameters."""
    expected_keys = [
        "HOHMANN_EARTH_MARS",
        "GRAVITATIONAL_SLINGSHOT",
        "LOW_EARTH_TO_GEO",
        "SUN_SYNCHRONOUS_J2",
        "MOLNIYA_CRITICAL_INCLINATION",
    ]
    for key in expected_keys:
        assert key in PRESETS
        p = PRESETS[key]
        assert p.code == key
        assert len(p.name_vi) > 0
        assert p.r1 > 0
        assert p.r2 > 0
        assert 0.0 <= p.inclination_deg <= 180.0


def test_hohmann_transfer_parameters(orbital_item):
    """Verify Hohmann transfer semi-major axis, eccentricity, delta-v, and transfer time."""
    orbital_item.set_preset("HOHMANN_EARTH_MARS")

    # a_tx = (1.0 + 1.524) / 2 = 1.262 AU
    a_tx = orbital_item.transfer_semi_major_axis
    assert math.isclose(a_tx, 1.262, rel_tol=1e-3)

    # e_tx = (1.524 - 1.0) / (1.524 + 1.0) = 0.524 / 2.524 ≈ 0.2076
    e_tx = orbital_item.transfer_eccentricity
    assert math.isclose(e_tx, 0.2076, rel_tol=1e-3)

    # Departure & arrival burns must be positive
    dv1 = orbital_item.calculate_delta_v1()
    dv2 = orbital_item.calculate_delta_v2()
    assert dv1 > 0.0
    assert dv2 > 0.0
    assert orbital_item.calculate_total_delta_v() == dv1 + dv2

    # Transfer time: ~258.9 days
    t_days = orbital_item.calculate_transfer_time_days()
    assert 240.0 < t_days < 280.0


def test_low_earth_to_geo_preset(orbital_item):
    """Verify LEO to GEO transfer eccentricity and parameters."""
    orbital_item.set_preset("LOW_EARTH_TO_GEO")
    assert orbital_item._scenario_type == "geo"
    assert orbital_item.transfer_eccentricity > 0.65
    assert orbital_item.calculate_total_delta_v() > 0.0


def test_molniya_critical_inclination(orbital_item):
    """Verify Molniya orbit critical inclination 5*cos^2(i) - 1 = 0 (i = 63.43°)."""
    orbital_item.set_preset("MOLNIYA_CRITICAL_INCLINATION")
    assert math.isclose(orbital_item._inclination_deg, 63.43, abs_tol=0.1)

    cos_i = math.cos(math.radians(orbital_item._inclination_deg))
    crit_val = 5.0 * (cos_i ** 2) - 1.0
    assert math.isclose(crit_val, 0.0, abs_tol=0.02)


def test_sun_synchronous_nodal_precession(orbital_item):
    """Verify Sun-synchronous orbit retrograde inclination produces positive nodal regression."""
    orbital_item.set_preset("SUN_SYNCHRONOUS_J2")
    assert orbital_item._inclination_deg > 90.0  # Retrograde orbit
    rate_deg_day = orbital_item.calculate_j2_nodal_precession_deg_day()
    assert rate_deg_day > 0.0, f"Expected positive nodal regression, got {rate_deg_day}"


def test_step_simulation_and_trail(orbital_item):
    """Verify step_simulation advances spacecraft anomaly and logs trail points."""
    orbital_item.set_preset("HOHMANN_EARTH_MARS")
    init_anomaly = orbital_item._spacecraft_true_anomaly_rad

    for _ in range(5):
        orbital_item.step_simulation(0.05)

    assert orbital_item._spacecraft_true_anomaly_rad > init_anomaly
    assert len(orbital_item._trail_points) > 0


def test_play_pause_and_reset(orbital_item):
    """Verify play/pause halts execution and reset clears orbit anomaly."""
    orbital_item.toggle_play_pause()
    assert orbital_item._is_playing is False

    t_before = orbital_item._sim_time_sec
    orbital_item.step_simulation(0.1)
    assert orbital_item._sim_time_sec == t_before

    orbital_item.toggle_play_pause()
    assert orbital_item._is_playing is True

    orbital_item.step_simulation(0.5)
    assert orbital_item._spacecraft_true_anomaly_rad > 0.0
    orbital_item.reset_orbit()
    assert orbital_item._spacecraft_true_anomaly_rad == 0.0
    assert len(orbital_item._trail_points) == 0


def test_sliders_clamping(orbital_item):
    """Verify sliders clamp to allowed astrodynamic ranges."""
    # Target radius r2 [0.5, 6.0]
    orbital_item.set_target_radius(10.0)
    assert orbital_item._r2 == 6.0
    orbital_item.set_target_radius(0.1)
    assert orbital_item._r2 == 0.5

    # Inclination [0, 180]
    orbital_item.set_inclination(200.0)
    assert orbital_item._inclination_deg == 180.0
    orbital_item.set_inclination(-20.0)
    assert orbital_item._inclination_deg == 0.0


def test_touchless_air_tap_gesture(orbital_item):
    """Verify air tap triggers presets, playback, reset, and close."""
    # Air tap on Preset 2: GRAVITATIONAL_SLINGSHOT
    # Button rect: start_x + 1 * (176 + 8) = 20 + 184 = 204, y = 58..86
    p_tap = orbital_item.mapToScene(QPointF(220.0, 70.0))
    handled = orbital_item.handle_air_tap(p_tap)
    assert handled is True
    assert orbital_item._current_preset == "GRAVITATIONAL_SLINGSHOT"

    # Air tap on Play/Pause button
    canvas_bottom = 96 + 470
    p_play_tap = orbital_item.mapToScene(QPointF(40.0, canvas_bottom + 20.0))
    was_playing = orbital_item._is_playing
    orbital_item.handle_air_tap(p_play_tap)
    assert orbital_item._is_playing != was_playing


def test_serialization_and_canvas_stamping(orbital_item):
    """Verify .airteach serialization and canvas cloning."""
    orbital_item.set_preset("LOW_EARTH_TO_GEO")
    orbital_item.set_target_radius(4.5)
    orbital_item.set_inclination(28.5)

    data = orbital_item.to_dict()
    assert data["type"] == "InteractiveOrbitalMechanicsItem"
    assert data["preset"] == "LOW_EARTH_TO_GEO"
    assert data["r2"] == 4.5
    assert data["inclination_deg"] == 28.5

    # Reconstruct from dict
    restored = InteractiveOrbitalMechanicsItem.from_dict(data)
    assert restored._current_preset == "LOW_EARTH_TO_GEO"
    assert restored._r2 == 4.5
    assert restored._inclination_deg == 28.5

    # Stamp to canvas
    stamped = orbital_item.stamp_to_canvas()
    assert stamped is not None
    assert isinstance(stamped, InteractiveOrbitalMechanicsItem)
    assert stamped._current_preset == "LOW_EARTH_TO_GEO"
    assert stamped.pos() == QPointF(orbital_item.x() + 30.0, orbital_item.y() + 30.0)


def test_paint_rendering_all_modes(orbital_item):
    """Verify QPainter renders without error across all 5 simulation presets."""
    img = QImage(940, 650, QImage.Format_ARGB32)
    painter = QPainter(img)

    preset_keys = list(PRESETS.keys())
    for key in preset_keys:
        orbital_item.set_preset(key)
        orbital_item.step_simulation(0.04)
        orbital_item.paint(painter, None, None)

    painter.end()
    assert not img.isNull()


def test_lesson_board_orbital_mechanics_integration(qapp):
    """Verify LessonBoard lifecycle methods for Orbital Mechanics visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_orbital_mechanics_visible() is False
    assert board.get_orbital_mechanics_item() is None

    # Show item
    item = board.show_orbital_mechanics(True, pos=(120.0, 90.0))
    assert item is not None
    assert board.is_orbital_mechanics_visible() is True
    assert board.get_orbital_mechanics_item() is item
    assert item.pos() == QPointF(120.0, 90.0)

    # Toggle off
    is_vis = board.toggle_orbital_mechanics()
    assert is_vis is False
    assert board.is_orbital_mechanics_visible() is False

    # Toggle on
    is_vis = board.toggle_orbital_mechanics()
    assert is_vis is True
    assert board.is_orbital_mechanics_visible() is True


def test_main_window_orbital_mechanics_toggle(qapp):
    """Verify MainWindow menu action and slot for Orbital Mechanics visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_orbital_mechanics")
        assert win._scene.is_orbital_mechanics_visible() is False

        # Toggle on
        is_vis = win._on_toggle_orbital_mechanics()
        assert is_vis is True
        assert win._scene.is_orbital_mechanics_visible() is True

        # Toggle off
        is_vis = win._on_toggle_orbital_mechanics()
        assert is_vis is False
        assert win._scene.is_orbital_mechanics_visible() is False
    finally:
        win.close()

