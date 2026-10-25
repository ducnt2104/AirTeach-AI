"""
Unit tests for InteractiveCoriolisEarthItem (Coriolis Effect, Geostrophic Flow & Cyclonic Vorticity).
"""
import pytest
import math
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_coriolis_earth import (
    InteractiveCoriolisEarthItem,
    PRESETS,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_initial_state(qapp):
    """Verify default initial state of Coriolis simulator."""
    item = InteractiveCoriolisEarthItem()
    assert item._preset_code == "NORTH_CYCLONE"
    assert item._latitude_deg == pytest.approx(21.0)
    assert item._rotation_speed == pytest.approx(1.0)
    assert item._pressure_gradient == pytest.approx(3.5)
    assert item._is_playing is True
    assert item._show_isobars is True
    assert item._show_force_vectors is True


def test_presets_loading(qapp):
    """Verify all 5 curriculum presets load with authentic atmospheric physics parameters."""
    item = InteractiveCoriolisEarthItem()

    for code, preset in PRESETS.items():
        item.load_preset(code)
        assert item._preset_code == code
        assert item._preset.scenario_type == preset.scenario_type
        assert item._latitude_deg == pytest.approx(preset.latitude_deg)
        assert item._rotation_speed == pytest.approx(preset.rotation_speed)
        assert item._pressure_gradient == pytest.approx(preset.pressure_gradient)


def test_coriolis_parameter_calculation(qapp):
    """Verify authentic f = 2 Ω sin(φ) parameter calculation across hemispheres and equator."""
    item = InteractiveCoriolisEarthItem()

    # 1. North Pole (φ = +90°): f is maximum positive
    item.set_latitude(90.0)
    item.set_rotation_speed(1.0)
    f_north_pole = item.get_coriolis_parameter()
    assert f_north_pole == pytest.approx(2.0, abs=1e-4)

    # 2. South Pole (φ = -90°): f is maximum negative
    item.set_latitude(-90.0)
    f_south_pole = item.get_coriolis_parameter()
    assert f_south_pole == pytest.approx(-2.0, abs=1e-4)

    # 3. Equator (φ = 0°): f is strictly 0.0
    item.set_latitude(0.0)
    f_equator = item.get_coriolis_parameter()
    assert f_equator == pytest.approx(0.0, abs=1e-6)

    # 4. Hanoi latitude (φ = +21.0°): f = 2 * sin(21°) ≈ 0.7167
    item.set_latitude(21.0)
    assert item.get_coriolis_parameter() == pytest.approx(2.0 * math.sin(math.radians(21.0)), abs=1e-4)


def test_cyclonic_vorticity_deflection(qapp):
    """Verify Northern hemisphere deflects right (counterclockwise) and Southern deflects left (clockwise)."""
    item = InteractiveCoriolisEarthItem()

    # Test point on positive x-axis (1.5, 0.0)
    # 1. Northern hemisphere (f > 0): inward (-x) + tangential (+y)
    item.set_latitude(25.0)
    vx_n, vy_n = item.compute_velocity_field(1.5, 0.0)
    assert vx_n < 0.0  # inward towards center
    assert vy_n > 0.0  # counterclockwise tangential deflection

    # 2. Southern hemisphere (f < 0): inward (-x) + tangential (-y)
    item.set_latitude(-25.0)
    vx_s, vy_s = item.compute_velocity_field(1.5, 0.0)
    assert vx_s < 0.0  # inward
    assert vy_s < 0.0  # clockwise tangential deflection


def test_geostrophic_wind_balance(qapp):
    """Verify geostrophic wind flows parallel to isobars along x-axis."""
    item = InteractiveCoriolisEarthItem()
    item.load_preset("GEOSTROPHIC_WIND")

    # In northern hemisphere mid-latitudes, geostrophic wind flows eastward (+u_g)
    u_g, v_ageostrophic = item.compute_velocity_field(0.0, 0.0)
    assert u_g > 0.0
    assert abs(u_g) > abs(v_ageostrophic)


def test_ballistic_trajectory_simulation(qapp):
    """Verify artillery flight simulation in rotating frame."""
    item = InteractiveCoriolisEarthItem()
    item.load_preset("BALLISTIC_CANNON")
    assert item._ballistic_t == 0.0
    assert len(item._ballistic_traj_rot) >= 1

    # Advance simulation tick
    for _ in range(10):
        item._on_sim_tick()

    assert item._ballistic_t > 0.0
    assert len(item._ballistic_traj_rot) > 1
    # Projectile drifts eastward (positive x) in Northern hemisphere
    last_pt = item._ballistic_traj_rot[-1]
    assert last_pt[0] > 0.0
    assert last_pt[1] > -2.2


def test_foucault_pendulum_precession(qapp):
    """Verify Foucault pendulum plane of oscillation precesses proportionally to sin(φ)."""
    item = InteractiveCoriolisEarthItem()
    item.load_preset("FOUCAULT_PENDULUM")
    item.set_latitude(48.8)  # Paris

    initial_angle = item._foucault_angle
    for _ in range(15):
        item._on_sim_tick()

    # Precession angle should have changed
    assert item._foucault_angle != initial_angle

    # At equator (latitude = 0), precession rate is 0
    item.set_latitude(0.0)
    item._foucault_angle = 0.0
    for _ in range(15):
        item._on_sim_tick()
    assert item._foucault_angle == pytest.approx(0.0)


def test_bounds_and_clamping(qapp):
    """Verify latitude and rotation speed adhere to physical clamping boundaries."""
    item = InteractiveCoriolisEarthItem()
    item.set_latitude(-200.0)
    assert item._latitude_deg == pytest.approx(-90.0)

    item.set_latitude(180.0)
    assert item._latitude_deg == pytest.approx(90.0)

    item.set_rotation_speed(-5.0)
    assert item._rotation_speed == pytest.approx(0.0)

    item.set_rotation_speed(10.0)
    assert item._rotation_speed == pytest.approx(3.0)


def test_serialization_roundtrip(qapp):
    """Verify to_dict and from_dict preserve complete simulation state."""
    item = InteractiveCoriolisEarthItem(pos=(140.0, 75.0))
    item.load_preset("SOUTH_CYCLONE")
    item.set_latitude(-30.0)
    item.set_rotation_speed(1.5)
    item.toggle_isobars()

    data = item.to_dict()
    assert data["type"] == "interactive_coriolis_earth"
    assert data["preset_code"] == "SOUTH_CYCLONE"
    assert data["latitude_deg"] == pytest.approx(-30.0)
    assert data["rotation_speed"] == pytest.approx(1.5)
    assert data["show_isobars"] is False

    restored = InteractiveCoriolisEarthItem.from_dict(data)
    assert restored._preset_code == "SOUTH_CYCLONE"
    assert restored._latitude_deg == pytest.approx(-30.0)
    assert restored._rotation_speed == pytest.approx(1.5)
    assert restored._show_isobars is False


def test_handle_air_tap_preset_and_buttons(qapp):
    """Verify touchless air-tap interaction on presets, buttons, and sliders."""
    item = InteractiveCoriolisEarthItem()

    # Tap on SOUTH_CYCLONE preset
    p_rect = item._btn_rects["preset_SOUTH_CYCLONE"]
    tap_pt = item.mapToScene(p_rect.center())
    handled = item.handle_air_tap(tap_pt)
    assert handled is True
    assert item._preset_code == "SOUTH_CYCLONE"

    # Tap on set_equator (φ = 0°)
    eq_rect = item._btn_rects["set_equator"]
    tap_eq = item.mapToScene(eq_rect.center())
    assert item.handle_air_tap(tap_eq) is True
    assert item._latitude_deg == pytest.approx(0.0)

    # Tap on lat_plus (+15°)
    lp_rect = item._btn_rects["lat_plus"]
    tap_lp = item.mapToScene(lp_rect.center())
    assert item.handle_air_tap(tap_lp) is True
    assert item._latitude_deg == pytest.approx(15.0)

    # Tap on toggle_isobars
    iso_rect = item._btn_rects["toggle_isobars"]
    tap_iso = item.mapToScene(iso_rect.center())
    prev_iso = item._show_isobars
    assert item.handle_air_tap(tap_iso) is True
    assert item._show_isobars != prev_iso

    # Tap on latitude slider
    lat_rect = item._slider_rect_lat
    tap_lat = item.mapToScene(lat_rect.center())
    assert item.handle_air_tap(tap_lat) is True
    assert abs(item._latitude_deg) < 5.0  # Center of [-90, +90] is 0°

    # Tap outside returns False
    out_pt = item.mapToScene(QPointF(-100.0, -100.0))
    assert item.handle_air_tap(out_pt) is False


def test_close_signal_emission(qapp):
    """Verify closed signal is emitted upon tapping close button."""
    item = InteractiveCoriolisEarthItem()
    emitted = False

    def on_closed():
        nonlocal emitted
        emitted = True

    item.closed.connect(on_closed)
    c_rect = item._btn_rects["btn_close"]
    tap_c = item.mapToScene(c_rect.center())
    assert item.handle_air_tap(tap_c) is True
    assert emitted is True


def test_qpainter_rendering(qapp):
    """Verify QPainter renders without exceptions."""
    item = InteractiveCoriolisEarthItem()
    img = QImage(800, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Verify canvas stamping clones the Coriolis simulation onto scene."""
    scene = QGraphicsScene()
    item = InteractiveCoriolisEarthItem(pos=(140.0, 70.0))
    item.load_preset("BALLISTIC_CANNON")
    item.set_latitude(38.0)

    stamped = item.stamp_to_canvas(scene)
    assert len(stamped) == 1
    clone = stamped[0]
    assert isinstance(clone, InteractiveCoriolisEarthItem)
    assert clone._preset_code == "BALLISTIC_CANNON"
    assert clone._latitude_deg == pytest.approx(38.0)


def test_lesson_board_coriolis_integration(qapp):
    """Verify LessonBoard lifecycle methods for Coriolis visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_coriolis_earth_visible() is False
    assert board.get_coriolis_earth_item() is None

    # Show item
    item = board.show_coriolis_earth(True, pos=(190.0, 110.0))
    assert item is not None
    assert board.is_coriolis_earth_visible() is True
    assert board.get_coriolis_earth_item() is item
    assert item.pos() == QPointF(190.0, 110.0)

    # Toggle off
    is_vis = board.toggle_coriolis_earth()
    assert is_vis is False
    assert board.is_coriolis_earth_visible() is False

    # Toggle on
    is_vis = board.toggle_coriolis_earth()
    assert is_vis is True
    assert board.is_coriolis_earth_visible() is True


def test_main_window_coriolis_toggle(qapp):
    """Verify MainWindow menu action and slot for Coriolis visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_coriolis_earth")
        assert win._scene.is_coriolis_earth_visible() is False

        # Toggle on
        is_vis = win._on_toggle_coriolis_earth()
        assert is_vis is True
        assert win._scene.is_coriolis_earth_visible() is True

        # Toggle off
        is_vis = win._on_toggle_coriolis_earth()
        assert is_vis is False
        assert win._scene.is_coriolis_earth_visible() is False
    finally:
        win.close()

