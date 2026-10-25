"""
AirTeach AI - Gravitational Waves & Laser Interferometry Tests
=============================================================
Authentic verification of compact binary coalescence (CBC), chirp mass,
ISCO cutoff frequency, QNM ringdown, laser interferometry strain,
historic presets, touchless controls, and serialization.
"""
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_gravitational_waves import (
    InteractiveGravitationalWavesItem,
    PRESETS,
    GravitationalWavePreset,
    SOLAR_MASS,
    SPEED_OF_LIGHT,
    GRAVITATIONAL_CONST,
    LIGO_ARM_LENGTH,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def gw_item(qapp):
    item = InteractiveGravitationalWavesItem()
    yield item
    item.stop()


def test_chirp_mass_calculation(gw_item):
    """
    Test authentic chirp mass calculation:
    M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5).
    For GW150914: m1 = 36.2, m2 = 29.1 => M_chirp ~= 28.1 M_sun.
    For GW170817: m1 = 1.46, m2 = 1.27 => M_chirp ~= 1.188 M_sun.
    """
    gw_item.set_preset(0)  # GW150914
    m_chirp = gw_item.compute_chirp_mass_msun()
    assert 28.0 < m_chirp < 28.3, f"Expected ~28.1 M_sun, got {m_chirp}"

    gw_item.set_preset(1)  # GW170817
    m_chirp_bns = gw_item.compute_chirp_mass_msun()
    assert 1.17 < m_chirp_bns < 1.21, f"Expected ~1.188 M_sun, got {m_chirp_bns}"


def test_isco_frequency(gw_item):
    """
    Test ISCO frequency: f_ISCO = c^3 / (6^(3/2) * pi * G * M_tot).
    For GW150914: M_tot = 65.3 M_sun => f_ISCO ~= 4396 / 65.3 ~= 67.3 Hz.
    """
    gw_item.set_preset(0)
    f_isco = gw_item.compute_isco_frequency_hz()
    assert 60.0 < f_isco < 75.0, f"Expected ~67 Hz, got {f_isco}"


def test_ringdown_frequency_and_damping(gw_item):
    """
    Test Quasi-Normal Mode (QNM) ringdown frequency and damping time:
    For GW150914 remnant (~62.3 M_sun, spin a ~ 0.69):
    f_ring ~= 200 - 300 Hz.
    tau_damp ~= 0.003 - 0.006 s (several milliseconds).
    """
    gw_item.set_preset(0)
    f_ring = gw_item.compute_ringdown_frequency_hz()
    tau_damp = gw_item.compute_ringdown_damping_time_sec()

    assert 180.0 < f_ring < 320.0, f"Expected ringdown ~250 Hz, got {f_ring}"
    assert 0.001 < tau_damp < 0.02, f"Expected damping time ~4 ms, got {tau_damp}"


def test_strain_computation_inspiral_and_ringdown(gw_item):
    """
    Test strain and frequency evolution during inspiral and ringdown.
    GW frequency must increase monotonically during inspiral (the chirp).
    """
    gw_item.set_preset(0)
    t_total = gw_item._disp_duration
    t_merger = t_total * 0.8

    # Early inspiral
    h_early, f_early = gw_item.compute_strain_at_time(0.05)
    # Late inspiral (closer to merger)
    h_late, f_late = gw_item.compute_strain_at_time(t_merger - 0.02)

    assert f_late > f_early, f"GW frequency must chirp upward: {f_late} > {f_early}"
    assert abs(h_early) > 0.0
    assert abs(h_late) > 0.0

    # Post-merger ringdown
    h_ring, f_ring = gw_item.compute_strain_at_time(t_merger + 0.02)
    assert abs(h_ring) > 0.0
    assert f_ring > 150.0


def test_arm_displacement_and_optical_phase(gw_item):
    """
    Test Michelson interferometer differential displacement:
    Delta L = L * h(t). For L = 4000 m and h = 1e-21 => Delta L = 4e-18 m.
    Phase shift: Delta Phi = 4*pi*Delta L / lambda.
    """
    h_test = 1.0e-21
    dl = gw_item.compute_arm_displacement_meters(h_test)
    assert abs(dl - 4.0e-18) < 1e-20, f"Expected 4e-18 m, got {dl}"

    dphi = gw_item.compute_optical_phase_shift_rad(h_test)
    assert dphi > 0.0


def test_preset_switching(gw_item):
    """Test switching across all 5 presets updates masses and emits signal."""
    emitted = []
    gw_item.preset_changed.connect(lambda k: emitted.append(k))

    gw_item.set_preset(2)  # GW190521
    assert gw_item.current_preset.key == "gw190521"
    assert gw_item._mass1_msun == 85.0
    assert gw_item._mass2_msun == 66.0
    assert "gw190521" in emitted

    gw_item.set_preset(4)  # LISA SMBH
    assert gw_item.current_preset.key == "lisa_smbh"
    assert gw_item._mass1_msun == 1.0e6


def test_sliders_and_clamping(gw_item):
    """Test parameter setters clamp out-of-range values and emit property_changed."""
    props = {}
    gw_item.property_changed.connect(lambda k, v: props.update({k: v}))

    gw_item.set_mass1(3.0e6)  # max 2e6
    assert gw_item._mass1_msun == 2.0e6
    assert props.get("mass1_msun") == 2.0e6

    gw_item.set_mass1(0.2)  # min 1.0
    assert gw_item._mass1_msun == 1.0

    gw_item.set_distance_mpc(30000.0)  # max 20000.0
    assert gw_item._dist_mpc == 20000.0

    gw_item.set_distance_mpc(1.0)  # min 5.0
    assert gw_item._dist_mpc == 5.0


def test_step_physics_and_pause(gw_item):
    """Test step_physics advances time when running and freezes when paused."""
    gw_item._sim_time = 0.0
    gw_item.step_physics(0.05)
    assert gw_item._sim_time == pytest.approx(0.05, 1e-4)

    gw_item.toggle_pause()
    assert gw_item._is_paused is True
    gw_item.step_physics(0.1)
    assert gw_item._sim_time == pytest.approx(0.05, 1e-4)

    gw_item.toggle_pause()
    assert gw_item._is_paused is False
    gw_item.step_physics(0.05)
    assert gw_item._sim_time == pytest.approx(0.1, 1e-4)


def test_serialization_roundtrip(gw_item):
    """Test serialization to dict and reconstruction from dict."""
    gw_item.set_preset(1)  # GW170817
    gw_item.set_mass1(1.5)
    gw_item.set_mass2(1.3)
    gw_item.set_distance_mpc(42.0)
    gw_item.setPos(150.0, 300.0)

    data = gw_item.to_dict()
    assert data["type"] == "InteractiveGravitationalWavesItem"
    assert data["preset_key"] == "gw170817"
    assert data["mass1_msun"] == 1.5
    assert data["mass2_msun"] == 1.3
    assert data["dist_mpc"] == 42.0
    assert data["pos"] == [150.0, 300.0]

    reconstructed = InteractiveGravitationalWavesItem.from_dict(data)
    try:
        assert reconstructed.current_preset.key == "gw170817"
        assert reconstructed._mass1_msun == 1.5
        assert reconstructed._mass2_msun == 1.3
        assert reconstructed._dist_mpc == 42.0
        assert reconstructed.pos().x() == 150.0
        assert reconstructed.pos().y() == 300.0
    finally:
        reconstructed.stop()


def test_mouse_interactions(gw_item):
    """Test clicking preset pills and pause/reset buttons."""
    scene = QGraphicsScene()
    scene.addItem(gw_item)
    try:
        # Click Preset 1 (GW170817) using its exact geometry center
        btn_rects = gw_item._preset_button_rects()
        assert len(btn_rects) >= 2
        p1_center = btn_rects[1][0].center()

        event_p1 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_p1.setPos(p1_center)
        event_p1.setButton(Qt.MouseButton.LeftButton)
        gw_item.mousePressEvent(event_p1)
        assert gw_item.current_preset.key == "gw170817"

        # Click Status badge (pause / resume)
        badge_rect = QRectF(gw_item.WIDTH - 150.0, 14.0, 130.0, 28.0)
        init_pause = gw_item._is_paused
        event_pause = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_pause.setPos(badge_rect.center())
        event_pause.setButton(Qt.MouseButton.LeftButton)
        gw_item.mousePressEvent(event_pause)
        assert gw_item._is_paused != init_pause

        # Click Reset button
        ctrl_rect = gw_item._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        c4_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_w = (col_w - 30.0) / 2.0
        reset_rect = QRectF(c4_x, ctrl_rect.top() + 22.0, btn_w, 38.0)
        event_reset = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_reset.setPos(reset_rect.center())
        event_reset.setButton(Qt.MouseButton.LeftButton)
        gw_item.mousePressEvent(event_reset)
    finally:
        scene.removeItem(gw_item)


def test_paint_render(gw_item):
    """Verify that paint renders into QPainter/QImage without crashing."""
    img = QImage(940, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    try:
        gw_item.paint(painter, None, None)
    finally:
        painter.end()
    assert img.pixelColor(20, 20).isValid()


def test_stop_cleanup(gw_item):
    """Verify stopping the item halts the animation timer."""
    assert gw_item._anim_timer.isActive()
    gw_item.stop()
    assert not gw_item._anim_timer.isActive()
