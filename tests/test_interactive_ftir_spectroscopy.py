"""
AirTeach AI - Interactive FTIR Spectroscopy Tests
=================================================
Authentic verification of molecular vibrational normal modes, Beer-Lambert
absorbance linearity, %Transmittance conversion, Michelson interferogram
centerburst at ZPD, presets, touchless controls, and serialization.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_ftir_spectroscopy import (
    InteractiveFTIRSpectroscopyItem,
    PRESETS,
    FTIRPreset,
    IRAbsorptionPeak,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def ftir_item(qapp):
    item = InteractiveFTIRSpectroscopyItem()
    yield item
    item.stop()


def test_beer_lambert_absorbance_linearity(ftir_item):
    """
    Test authentic Beer-Lambert law linearity: A(c) = c * epsilon * l.
    Doubling concentration must exactly double absorbance.
    """
    ftir_item.set_preset(0)  # Ethanol
    nu = 3350.0  # O-H stretch center

    ftir_item.set_concentration(1.0)
    abs_1 = ftir_item.compute_absorbance(nu)

    ftir_item.set_concentration(2.0)
    abs_2 = ftir_item.compute_absorbance(nu)

    assert abs_1 > 0.5, f"O-H band should have strong absorbance, got {abs_1}"
    assert abs(abs_2 - 2.0 * abs_1) < 1e-4, f"Absorbance should double: {abs_2} vs 2 * {abs_1}"


def test_ethanol_characteristic_peaks(ftir_item):
    """
    Test ethanol signature peaks:
    O-H stretch at 3350 cm^-1, C-H sp3 stretch at 2970 cm^-1, C-O at 1050 cm^-1.
    """
    ftir_item.set_preset(0)  # Ethanol
    ftir_item.set_concentration(1.0)

    a_oh = ftir_item.compute_absorbance(3350.0)
    a_ch = ftir_item.compute_absorbance(2970.0)
    a_co = ftir_item.compute_absorbance(1050.0)
    a_baseline = ftir_item.compute_absorbance(2000.0)  # Clean transparent window

    assert a_oh > 0.8
    assert a_ch > 0.6
    assert a_co > 0.7
    assert a_baseline < 0.05, f"Baseline should be transparent, got {a_baseline}"


def test_acetone_carbonyl_peak(ftir_item):
    """
    Test acetone signature carbonyl C=O peak at 1715 cm^-1.
    Must be the dominant peak in the acetone spectrum.
    """
    ftir_item.set_preset(1)  # Acetone
    a_co = ftir_item.compute_absorbance(1715.0)
    a_other = ftir_item.compute_absorbance(2500.0)

    assert a_co > 0.9, f"C=O peak should be very intense, got {a_co}"
    assert a_other < 0.05


def test_co2_greenhouse_absorption(ftir_item):
    """
    Test carbon dioxide asymmetric stretch nu3 at 2349 cm^-1.
    """
    ftir_item.set_preset(3)  # CO2
    a_nu3 = ftir_item.compute_absorbance(2349.0)
    a_nu2 = ftir_item.compute_absorbance(667.0)
    a_inactive = ftir_item.compute_absorbance(1388.0)  # Raman active only

    assert a_nu3 > 0.9, f"nu3 should be extremely strong, got {a_nu3}"
    assert a_nu2 > 0.7
    assert a_inactive < 0.05, f"nu1 is IR inactive, should have ~0 absorbance, got {a_inactive}"


def test_transmittance_conversion(ftir_item):
    """
    Test %Transmittance conversion: %T = 100 * 10^(-A).
    A = 0 => 100% T, A = 1 => 10% T, A = 2 => 1% T.
    """
    ftir_item.set_preset(0)
    # Transparent region (A ~ 0)
    t_trans = ftir_item.compute_transmittance(2000.0)
    assert 95.0 < t_trans <= 100.0

    # Strong absorption (A ~ 0.9)
    t_abs = ftir_item.compute_transmittance(3350.0)
    expected_t = 100.0 * (10.0 ** (-ftir_item.compute_absorbance(3350.0)))
    assert abs(t_abs - expected_t) < 0.1
    assert t_abs < 25.0


def test_interferogram_centerburst_zpd(ftir_item):
    """
    Test Michelson interferogram I_AC(delta):
    At ZPD (delta = 0), constructive interference produces the intense centerburst.
    At large OPD (e.g. delta = 0.005 cm), amplitude is much smaller.
    """
    ftir_item.set_preset(0)
    sig_zpd = ftir_item.compute_interferogram_at_opd(0.0)
    sig_wing = ftir_item.compute_interferogram_at_opd(0.006)

    assert sig_zpd > 1.5, f"Centerburst should be prominent, got {sig_zpd}"
    assert abs(sig_wing) < abs(sig_zpd) * 0.4, f"Wing signal should be significantly lower: {sig_wing} vs {sig_zpd}"


def test_preset_switching(ftir_item):
    """Test switching across all 5 presets updates chemical metadata."""
    emitted = []
    ftir_item.preset_changed.connect(lambda k: emitted.append(k))

    for idx, p in enumerate(PRESETS):
        ftir_item.set_preset(idx)
        assert ftir_item.current_preset.key == p.key
        assert len(ftir_item.current_preset.peaks) > 0

    assert len(emitted) == len(PRESETS)


def test_mode_toggles_and_clamping(ftir_item):
    """Test toggling display mode, apodization, and concentration clamping."""
    init_mode = ftir_item._is_absorbance_mode
    ftir_item.toggle_display_mode()
    assert ftir_item._is_absorbance_mode != init_mode

    init_apod = ftir_item._use_hamming_apodization
    ftir_item.toggle_apodization()
    assert ftir_item._use_hamming_apodization != init_apod

    # Concentration clamping
    ftir_item.set_concentration(10.0)  # max 3.0
    assert ftir_item._concentration == 3.0

    ftir_item.set_concentration(0.01)  # min 0.1
    assert ftir_item._concentration == 0.1


def test_step_physics_and_pause(ftir_item):
    """Test step_physics advances simulation time and pause freezes time."""
    ftir_item._sim_time = 0.0
    ftir_item.step_physics(0.05)
    assert ftir_item._sim_time == pytest.approx(0.05, 1e-4)

    ftir_item.toggle_pause()
    assert ftir_item._is_paused is True
    ftir_item.step_physics(0.1)
    assert ftir_item._sim_time == pytest.approx(0.05, 1e-4)

    ftir_item.toggle_pause()
    assert ftir_item._is_paused is False
    ftir_item.step_physics(0.02)
    assert ftir_item._sim_time == pytest.approx(0.07, 1e-4)


def test_mouse_interactions(ftir_item):
    """Test clicking preset pills, mode buttons, and reset button."""
    scene = QGraphicsScene()
    scene.addItem(ftir_item)
    try:
        # Click Preset 1 (Acetone)
        rects = ftir_item._preset_button_rects()
        assert len(rects) >= 2
        p1_center = rects[1][0].center()

        event_p1 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_p1.setPos(p1_center)
        event_p1.setButton(Qt.MouseButton.LeftButton)
        ftir_item.mousePressEvent(event_p1)
        assert ftir_item.current_preset.key == "acetone"

        # Click Mode button (Col 1 in controls)
        ctrl_rect = ftir_item._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_w = col_w - 20.0
        b2_rect = QRectF(ctrl_rect.left() + 20.0 + col_w, ctrl_rect.top() + 22.0, btn_w, 38.0)

        prev_mode = ftir_item._is_absorbance_mode
        event_b2 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_b2.setPos(b2_rect.center())
        event_b2.setButton(Qt.MouseButton.LeftButton)
        ftir_item.mousePressEvent(event_b2)
        assert ftir_item._is_absorbance_mode != prev_mode

        # Click Reset button (Col 3 left sub-button)
        b4_x = ctrl_rect.left() + 20.0 + 3 * col_w
        b4_w = (btn_w - 10.0) / 2.0
        rst_rect = QRectF(b4_x, ctrl_rect.top() + 22.0, b4_w, 38.0)
        event_rst = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_rst.setPos(rst_rect.center())
        event_rst.setButton(Qt.MouseButton.LeftButton)
        ftir_item.mousePressEvent(event_rst)
        assert ftir_item._sim_time == 0.0
    finally:
        scene.removeItem(ftir_item)


def test_serialization_and_cleanup(ftir_item):
    """Test serialization roundtrip and defensive stop cleanup."""
    ftir_item.set_preset(2)  # Benzene
    ftir_item.set_concentration(1.8)
    ftir_item.setPos(140.0, 260.0)

    data = ftir_item.to_dict()
    assert data["type"] == "InteractiveFTIRSpectroscopyItem"
    assert data["preset_key"] == "benzene"
    assert data["concentration"] == 1.8
    assert data["pos"] == [140.0, 260.0]

    reconstructed = InteractiveFTIRSpectroscopyItem.from_dict(data)
    try:
        assert reconstructed.current_preset.key == "benzene"
        assert reconstructed._concentration == 1.8
        assert reconstructed.pos().x() == 140.0
        assert reconstructed.pos().y() == 260.0
    finally:
        reconstructed.stop()

    # Paint render test
    img = QImage(940, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    try:
        ftir_item.paint(painter, None, None)
    finally:
        painter.end()
    assert img.pixelColor(20, 20).isValid()

    # Stop cleanup
    assert ftir_item._anim_timer.isActive()
    ftir_item.stop()
    assert not ftir_item._anim_timer.isActive()
