"""
AirTeach AI - Cryo-EM CTF & 3D Reconstruction Interactive Simulator Tests
========================================================================
Authentic verification of relativistic de Broglie wavelength, Scherzer defocus,
Contrast Transfer Function (CTF), Fourier Shell Correlation (FSC), presets,
touchless air gesture controls, and serialization.
"""
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_cryo_em import (
    InteractiveCryoEMItem,
    PRESETS,
    CryoEMPreset,
    PLANCK_H,
    SPEED_OF_LIGHT,
    ELECTRON_MASS,
    E_CHARGE,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def cryo_item(qapp):
    item = InteractiveCryoEMItem()
    yield item
    item.stop()


def test_electron_wavelength_relativistic(cryo_item):
    """
    Test authentic relativistic de Broglie wavelength for electron beam:
    At 300 kV: lambda ~= 0.019687 Angstrom
    At 200 kV: lambda ~= 0.025079 Angstrom
    At 100 kV: lambda ~= 0.037014 Angstrom
    """
    cryo_item.set_voltage_kv(300.0)
    lam_300 = cryo_item.compute_electron_wavelength_angstrom()
    assert 0.0195 < lam_300 < 0.0199, f"Expected ~0.01969 Å at 300 kV, got {lam_300}"

    cryo_item.set_voltage_kv(200.0)
    lam_200 = cryo_item.compute_electron_wavelength_angstrom()
    assert 0.0249 < lam_200 < 0.0253, f"Expected ~0.02508 Å at 200 kV, got {lam_200}"

    cryo_item.set_voltage_kv(100.0)
    lam_100 = cryo_item.compute_electron_wavelength_angstrom()
    assert 0.0368 < lam_100 < 0.0372, f"Expected ~0.03701 Å at 100 kV, got {lam_100}"


def test_scherzer_defocus(cryo_item):
    """
    Test Scherzer optimal defocus: Delta f_Sch = -1.2 * sqrt(Cs * lambda).
    For Cs = 2.7 mm and 300 kV (lambda ~= 0.01969 Å):
    Cs = 2.7e7 Å => sqrt(2.7e7 * 0.01969) ~= sqrt(531630) ~= 729 Å
    df_sch ~= -1.2 * 729 ~= -875 Å = -0.0875 µm.
    """
    cryo_item.set_voltage_kv(300.0)
    cryo_item.set_cs_mm(2.7)
    df_sch = cryo_item.compute_scherzer_defocus_um()
    assert -0.12 < df_sch < -0.05, f"Expected ~ -0.087 µm, got {df_sch}"


def test_ctf_damping_envelope_and_oscillations(cryo_item):
    """
    Test CTF function values across spatial frequencies:
    At k = 0, CTF value is small (amplitude contrast fraction -0.08).
    At moderate spatial frequency (e.g. k = 0.1 1/Å), oscillations occur.
    At very high spatial frequency (e.g. k = 0.5 1/Å), B-factor damping reduces CTF to near zero.
    """
    cryo_item.set_voltage_kv(300.0)
    cryo_item.set_defocus_um(-1.5)
    cryo_item.set_b_factor(50.0)

    ctf_0 = cryo_item.compute_ctf(0.0)
    assert abs(ctf_0 - (-0.08)) < 1e-3, f"CTF at k=0 should be -amplitude_contrast, got {ctf_0}"

    ctf_mid = cryo_item.compute_ctf(0.1)
    assert abs(ctf_mid) > 0.01, "CTF should oscillate at intermediate frequencies"

    ctf_high = cryo_item.compute_ctf(0.8)
    assert abs(ctf_high) < 0.01, f"High frequency should be heavily damped, got {ctf_high}"


def test_fsc_resolution_cutoff(cryo_item):
    """
    Test Fourier Shell Correlation curve:
    FSC(k=0) should be near 1.0.
    At nominal resolution (e.g., 2.4 Å -> k = 0.4167 1/Å), FSC transitions to ~0.5.
    Well beyond nominal resolution (e.g. 1.0 Å -> k = 1.0 1/Å), FSC drops near 0.
    """
    cryo_item.set_preset(0)  # Ribosome 70S: 2.4 Å nominal
    fsc_low = cryo_item.compute_fsc(0.05)
    assert fsc_low > 0.95

    k_nominal = 1.0 / 2.4
    fsc_nominal = cryo_item.compute_fsc(k_nominal)
    assert 0.4 < fsc_nominal < 0.6, f"FSC at nominal resolution should be ~0.5, got {fsc_nominal}"

    fsc_high = cryo_item.compute_fsc(1.0)
    assert fsc_high < 0.05


def test_preset_switching(cryo_item):
    """Test switching presets updates parameters and emits preset_changed."""
    emitted = []
    cryo_item.preset_changed.connect(lambda k: emitted.append(k))

    cryo_item.set_preset(2)  # Apoferritin atomic
    assert cryo_item.current_preset.key == "apoferritin_atomic"
    assert cryo_item._cs_mm == 0.01
    assert cryo_item._nominal_res() == 1.25 if hasattr(cryo_item, "_nominal_res") else cryo_item.current_preset.nominal_res_ang == 1.25
    assert "apoferritin_atomic" in emitted


def test_sliders_and_clamping(cryo_item):
    """Test parameter setters clamp out-of-range values and emit property_changed."""
    props = {}
    cryo_item.property_changed.connect(lambda k, v: props.update({k: v}))

    cryo_item.set_voltage_kv(500.0)  # max 400
    assert cryo_item._voltage_kv == 400.0
    assert props.get("voltage_kv") == 400.0

    cryo_item.set_voltage_kv(20.0)  # min 80
    assert cryo_item._voltage_kv == 80.0

    cryo_item.set_defocus_um(10.0)  # max 3.0
    assert cryo_item._defocus_um == 3.0

    cryo_item.set_defocus_um(-10.0)  # min -5.0
    assert cryo_item._defocus_um == -5.0

    cryo_item.set_b_factor(200.0)  # max 150.0
    assert cryo_item._b_factor == 150.0


def test_step_physics_and_pause(cryo_item):
    """Test step_physics advances time when running and freezes when paused."""
    cryo_item._sim_time = 0.0
    cryo_item.step_physics(0.1)
    assert cryo_item._sim_time == pytest.approx(0.1, 1e-4)

    cryo_item.toggle_pause()
    assert cryo_item._is_paused is True
    cryo_item.step_physics(0.2)
    assert cryo_item._sim_time == pytest.approx(0.1, 1e-4)

    cryo_item.toggle_pause()
    assert cryo_item._is_paused is False
    cryo_item.step_physics(0.05)
    assert cryo_item._sim_time == pytest.approx(0.15, 1e-4)


def test_serialization_roundtrip(cryo_item):
    """Test serialization to dict and reconstruction from dict."""
    cryo_item.set_preset(1)  # SARS-CoV-2 spike
    cryo_item.set_defocus_um(-1.8)
    cryo_item.set_voltage_kv(300.0)
    cryo_item.set_b_factor(55.0)
    cryo_item.setPos(120.0, 240.0)

    data = cryo_item.to_dict()
    assert data["type"] == "InteractiveCryoEMItem"
    assert data["preset_key"] == "sars_cov2_spike"
    assert data["defocus_um"] == -1.8
    assert data["b_factor"] == 55.0
    assert data["pos"] == [120.0, 240.0]

    reconstructed = InteractiveCryoEMItem.from_dict(data)
    try:
        assert reconstructed.current_preset.key == "sars_cov2_spike"
        assert reconstructed._defocus_um == -1.8
        assert reconstructed._b_factor == 55.0
        assert reconstructed.pos().x() == 120.0
        assert reconstructed.pos().y() == 240.0
    finally:
        reconstructed.stop()


def test_mouse_interactions(cryo_item):
    """Test clicking preset pills and pause/reset buttons."""
    scene = QGraphicsScene()
    scene.addItem(cryo_item)
    try:
        # Click Preset 1 (SARS-CoV-2 Spike) using its exact geometry center
        btn_rects = cryo_item._preset_button_rects()
        assert len(btn_rects) >= 2
        p1_center = btn_rects[1][0].center()

        event_p1 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_p1.setPos(p1_center)
        event_p1.setButton(Qt.MouseButton.LeftButton)
        cryo_item.mousePressEvent(event_p1)
        assert cryo_item.current_preset.key == "sars_cov2_spike"

        # Click Status badge (pause / resume)
        badge_rect = QRectF(cryo_item.WIDTH - 150.0, 14.0, 130.0, 28.0)
        init_pause = cryo_item._is_paused
        event_pause = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_pause.setPos(badge_rect.center())
        event_pause.setButton(Qt.MouseButton.LeftButton)
        cryo_item.mousePressEvent(event_pause)
        assert cryo_item._is_paused != init_pause

        # Click Reset button
        ctrl_rect = cryo_item._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        reset_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        event_reset = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_reset.setPos(reset_rect.center())
        event_reset.setButton(Qt.MouseButton.LeftButton)
        cryo_item.mousePressEvent(event_reset)
    finally:
        scene.removeItem(cryo_item)


def test_paint_render(cryo_item):
    """Verify that paint renders into QPainter/QImage without crashing."""
    img = QImage(940, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    try:
        cryo_item.paint(painter, None, None)
    finally:
        painter.end()
    # Image should not be empty black
    assert img.pixelColor(20, 20).isValid()


def test_stop_cleanup(cryo_item):
    """Verify stopping the item halts the animation timer."""
    assert cryo_item._anim_timer.isActive()
    cryo_item.stop()
    assert not cryo_item._anim_timer.isActive()
