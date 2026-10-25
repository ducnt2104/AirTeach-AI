"""
Unit tests for InteractiveSynchrotronRadiationItem - Synchrotron Radiation & Undulators.
========================================================================================
Covers relativistic Lorentz factor, Lienard-Wiechert beaming cone, critical photon energy,
undulator deflection parameter K, harmonic wavelengths, UI sliders, and serialization.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_synchrotron_radiation import (
    InteractiveSynchrotronRadiationItem,
    PRESETS,
    SynchrotronPreset,
    ELECTRON_MASS_EV,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    item = InteractiveSynchrotronRadiationItem()
    assert item.WIDTH == 940.0
    assert item.HEIGHT == 600.0
    assert item.current_preset.key == "planar_undulator"
    assert math.isclose(item._beam_energy_gev, 3.0)
    assert not item._is_paused


def test_presets(qapp):
    item = InteractiveSynchrotronRadiationItem()
    keys = ["bending_magnet", "planar_undulator", "multipole_wiggler", "free_electron_laser", "low_energy_synchrotron"]

    for i, expected_key in enumerate(keys):
        item.set_preset(i)
        assert item.current_preset.key == expected_key
        assert item._sim_time == 0.0

    curr_idx = item._preset_index
    item.set_preset(99)
    assert item._preset_index == curr_idx


def test_lorentz_factor_and_beaming_angle(qapp):
    item = InteractiveSynchrotronRadiationItem()
    item.set_beam_energy(3.0)  # 3.0 GeV

    gamma = item.compute_gamma()
    expected_gamma = 3e9 / ELECTRON_MASS_EV
    assert math.isclose(gamma, expected_gamma, rel_tol=1e-3)
    assert 5800 < gamma < 5900

    theta_mrad = item.compute_beaming_angle()
    expected_theta = (1.0 / expected_gamma) * 1000.0
    assert math.isclose(theta_mrad, expected_theta, rel_tol=1e-3)
    assert 0.16 < theta_mrad < 0.18  # ~0.17 mrad


def test_bending_radius_and_critical_energy(qapp):
    item = InteractiveSynchrotronRadiationItem()
    item.set_preset(0)  # Bending magnet: 3.0 GeV, 1.2 T
    assert item._b0_tesla == 1.2

    rho = item.compute_bending_radius()
    # rho ≈ E / (0.3 * B) = 3.0 / (0.3 * 1.2) = 3.0 / 0.36 ≈ 8.34 m
    assert 8.0 < rho < 8.7

    ec_kev = item.compute_critical_energy()
    # Ec ≈ 0.665 * E^2 * B = 0.665 * 9 * 1.2 = 7.182 keV
    assert 7.0 < ec_kev < 7.3


def test_undulator_deflection_parameter_k(qapp):
    item = InteractiveSynchrotronRadiationItem()
    item.set_preset(1)  # Planar undulator: B0 = 0.45 T, period = 2.8 cm
    k = item.compute_undulator_k()
    # K ≈ 0.9337 * B0 * lambda_u = 0.9337 * 0.45 * 2.8 ≈ 1.176
    assert 1.1 < k < 1.25


def test_fundamental_harmonic_energy_and_redshift(qapp):
    item = InteractiveSynchrotronRadiationItem()
    item.set_preset(1)  # Planar undulator
    item.set_observation_angle(0.0)  # on-axis

    e1_on_axis = item.compute_fundamental_harmonic_energy()
    assert e1_on_axis > 0.5  # keV range

    # Off-axis observation (theta = 2.0 mrad)
    item.set_observation_angle(2.0)
    e1_off_axis = item.compute_fundamental_harmonic_energy()

    # Red-shift: off-axis emission has lower photon energy
    assert e1_off_axis < e1_on_axis


def test_parameter_setters(qapp):
    item = InteractiveSynchrotronRadiationItem()
    # Energy
    item.set_beam_energy(6.0)
    assert math.isclose(item._beam_energy_gev, 6.0)
    item.set_beam_energy(20.0)  # Clamped to 10.0 GeV
    assert math.isclose(item._beam_energy_gev, 10.0)

    # Magnetic field
    item.set_magnetic_field(2.1)
    assert math.isclose(item._b0_tesla, 2.1)
    item.set_magnetic_field(10.0)  # Clamped to 3.0 T
    assert math.isclose(item._b0_tesla, 3.0)

    # Observation angle
    item.set_observation_angle(3.5)
    assert math.isclose(item._obs_angle_mrad, 3.5)
    item.set_observation_angle(50.0)  # Clamped to 10.0 mrad
    assert math.isclose(item._obs_angle_mrad, 10.0)


def test_pause_and_reset(qapp):
    item = InteractiveSynchrotronRadiationItem()
    assert not item._is_paused
    item.toggle_pause()
    assert item._is_paused

    time_before = item._sim_time
    item.step_physics(0.05)
    assert item._sim_time == time_before

    item.toggle_pause()
    assert not item._is_paused
    item.step_physics(0.05)
    assert item._sim_time > time_before

    item.reset_simulation()
    assert item._sim_time == 0.0


def test_paint_rendering(qapp):
    item = InteractiveSynchrotronRadiationItem()
    # Test painting undulator preset
    item.set_preset(1)
    for _ in range(5):
        item.step_physics(0.033)

    img = QImage(940, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()

    assert not img.isNull()
    assert img.width() == 940

    # Test painting bending magnet preset
    item.set_preset(0)
    img2 = QImage(940, 600, QImage.Format_ARGB32)
    img2.fill(Qt.transparent)
    painter2 = QPainter(img2)
    item.paint(painter2, None, None)
    painter2.end()
    assert not img2.isNull()


def test_mouse_interactions(qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    item = InteractiveSynchrotronRadiationItem()

    def make_event(event_type, pos_f):
        e = QGraphicsSceneMouseEvent(event_type)
        e.setPos(pos_f)
        e.setScenePos(pos_f)
        e.setButton(Qt.LeftButton)
        return e

    # Click Preset 2 (multipole_wiggler)
    p_rects = item._preset_button_rects()
    target_rect, _ = p_rects[2]
    click_ev = make_event(QEvent.GraphicsSceneMousePress, target_rect.center())
    item.mousePressEvent(click_ev)
    assert item.current_preset.key == "multipole_wiggler"

    # Click Pause badge
    badge_rect = QRectF(item.WIDTH - 150.0, 14.0, 130.0, 28.0)
    pause_ev = make_event(QEvent.GraphicsSceneMousePress, badge_rect.center())
    item.mousePressEvent(pause_ev)
    assert item._is_paused

    # Click Reset button
    ctrl_rect = item._controls_rect()
    col_w = (ctrl_rect.width() - 40.0) / 4.0
    btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
    btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
    reset_ev = make_event(QEvent.GraphicsSceneMousePress, btn_rect.center())
    item.mousePressEvent(reset_ev)
    assert item._sim_time == 0.0

    # Drag slider (energy)
    track_rect = QRectF(ctrl_rect.left() + 20.0, ctrl_rect.top() + 36.0, col_w - 25.0, 10.0)
    drag_start = make_event(QEvent.GraphicsSceneMousePress, track_rect.center())
    item.mousePressEvent(drag_start)
    assert item._dragging_slider == "energy"

    drag_move = make_event(QEvent.GraphicsSceneMouseMove, track_rect.topRight())
    item.mouseMoveEvent(drag_move)

    drag_rel = make_event(QEvent.GraphicsSceneMouseRelease, track_rect.topRight())
    item.mouseReleaseEvent(drag_rel)
    assert item._dragging_slider is None


def test_serialization_roundtrip(qapp):
    item = InteractiveSynchrotronRadiationItem()
    item.set_preset(2)  # multipole_wiggler
    item.set_beam_energy(4.5)
    item.set_magnetic_field(2.0)
    item.set_observation_angle(1.5)
    item.setPos(160.0, 210.0)
    for _ in range(10):
        item.step_physics(0.033)

    data = item.to_dict()
    assert data["type"] == "InteractiveSynchrotronRadiationItem"
    assert data["preset_key"] == "multipole_wiggler"
    assert math.isclose(data["beam_energy_gev"], 4.5, abs_tol=1e-3)
    assert math.isclose(data["b0_tesla"], 2.0, abs_tol=1e-3)
    assert math.isclose(data["obs_angle_mrad"], 1.5, abs_tol=1e-3)

    restored = InteractiveSynchrotronRadiationItem.from_dict(data)
    assert restored.current_preset.key == "multipole_wiggler"
    assert math.isclose(restored._beam_energy_gev, 4.5, abs_tol=1e-3)
    assert math.isclose(restored._b0_tesla, 2.0, abs_tol=1e-3)
    assert math.isclose(restored._obs_angle_mrad, 1.5, abs_tol=1e-3)
    assert restored.pos() == QPointF(160.0, 210.0)
