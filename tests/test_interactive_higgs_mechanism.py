"""
AirTeach AI - Interactive Higgs Boson Mechanism Tests
=====================================================
Authentic verification of spontaneous symmetry breaking (SSB),
Mexican hat potential, VEV vacuum expectation value, W/Z gauge boson
mass generation, invariant mass resonance at 125.1 GeV, 5-sigma discovery
significance, presets, touchless controls, and serialization.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_higgs_mechanism import (
    InteractiveHiggsMechanismItem,
    PRESETS,
    HiggsPreset,
    HIGGS_MASS_GEV,
    HIGGS_VEV_GEV,
    W_BOSON_MASS_GEV,
    Z_BOSON_MASS_GEV,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def higgs_item(qapp):
    item = InteractiveHiggsMechanismItem()
    yield item
    item.stop()


def test_higgs_mass_and_resonance_peak(higgs_item):
    """
    Test invariant mass resonance peak at m_H = 125.10 GeV:
    Signal must be strictly maximal at m = 125.1 GeV, and fall rapidly off-peak.
    """
    higgs_item.set_preset(0)  # Diphoton
    tot_peak, sig_peak, bkg_peak = higgs_item.compute_invariant_mass_spectrum(HIGGS_MASS_GEV)
    tot_off, sig_off, bkg_off = higgs_item.compute_invariant_mass_spectrum(115.0)

    assert sig_peak > 50.0, f"Signal at peak should be strong, got {sig_peak}"
    assert sig_off < 1.0, f"Signal at 115 GeV should be virtually zero, got {sig_off}"
    assert tot_peak > bkg_peak, "Total events must exceed background at peak"


def test_discovery_significance_5_sigma(higgs_item):
    """
    Test authentic CERN discovery significance:
    For nominal Run 2 luminosity (140 fb^-1), significance Z >= 5.0 sigma.
    """
    higgs_item.set_preset(0)  # Diphoton
    higgs_item.set_luminosity_fb(140.0)
    z_sig = higgs_item.compute_statistical_significance_sigma()

    assert z_sig >= 5.0, f"Expected discovery threshold >= 5.0 sigma, got {z_sig:.2f} sigma"


def test_mexican_hat_potential_ssb(higgs_item):
    """
    Test temperature-dependent Mexican hat potential:
    At T < Tc (e.g. T/Tc = 0.2): V(r=1.0) is minimum < V(0.0) (Broken Phase).
    At T >= Tc (e.g. T/Tc = 1.2): V(0.0) is minimum < V(r=1.0) (Symmetric Phase).
    """
    # 1. Broken phase
    higgs_item.set_temperature_ratio(0.2)
    v_center_broken = higgs_item.compute_potential_value(0.0)
    v_valley_broken = higgs_item.compute_potential_value(1.0)
    assert v_valley_broken < v_center_broken, "At T < Tc, vacuum valley must be deeper than center"

    # 2. Symmetric phase
    higgs_item.set_temperature_ratio(1.2)
    v_center_sym = higgs_item.compute_potential_value(0.0)
    v_rim_sym = higgs_item.compute_potential_value(1.0)
    assert v_center_sym < v_rim_sym, "At T >= Tc, center must be the stable ground state"


def test_vev_and_gauge_boson_masses(higgs_item):
    """
    Test electroweak mass generation via Higgs VEV:
    At T = 0: v = 246.22 GeV, M_W ~= 80.4 GeV, M_Z ~= 91.2 GeV, M_gamma = 0.
    At T >= Tc: v = 0, M_W = 0, M_Z = 0, M_gamma = 0 (massless gauge bosons).
    """
    higgs_item.set_temperature_ratio(0.0)
    vev_0 = higgs_item.compute_current_vev_gev()
    assert abs(vev_0 - HIGGS_VEV_GEV) < 1e-2

    m_w, m_z, m_gamma = higgs_item.compute_gauge_boson_masses()
    assert abs(m_w - W_BOSON_MASS_GEV) < 0.5, f"Expected M_W ~ 80.4, got {m_w}"
    assert abs(m_z - Z_BOSON_MASS_GEV) < 0.5, f"Expected M_Z ~ 91.2, got {m_z}"
    assert m_gamma == 0.0

    # High temperature
    higgs_item.set_temperature_ratio(1.5)
    vev_high = higgs_item.compute_current_vev_gev()
    m_w_h, m_z_h, _ = higgs_item.compute_gauge_boson_masses()
    assert vev_high == 0.0
    assert m_w_h == 0.0 and m_z_h == 0.0


def test_luminosity_scaling(higgs_item):
    """
    Test luminosity scaling:
    Increasing luminosity from 50 to 200 fb^-1 increases signal and significance.
    """
    higgs_item.set_preset(0)
    higgs_item.set_luminosity_fb(50.0)
    z_low = higgs_item.compute_statistical_significance_sigma()

    higgs_item.set_luminosity_fb(200.0)
    z_high = higgs_item.compute_statistical_significance_sigma()

    assert z_high > z_low, f"Significance must scale with sqrt(L): {z_high} > {z_low}"


def test_preset_switching(higgs_item):
    """Test switching across all 5 particle physics presets."""
    emitted = []
    higgs_item.preset_changed.connect(lambda k: emitted.append(k))

    for idx, p in enumerate(PRESETS):
        higgs_item.set_preset(idx)
        assert higgs_item.current_preset.key == p.key
        assert len(higgs_item.current_preset.channel) > 0

    assert len(emitted) == len(PRESETS)


def test_sliders_and_clamping(higgs_item):
    """Test parameter setters clamp out-of-range values and emit property_changed."""
    props = {}
    higgs_item.property_changed.connect(lambda k, v: props.update({k: v}))

    higgs_item.set_luminosity_fb(500.0)  # max 300.0
    assert higgs_item._luminosity_fb == 300.0
    assert props.get("luminosity_fb") == 300.0

    higgs_item.set_luminosity_fb(2.0)  # min 10.0
    assert higgs_item._luminosity_fb == 10.0

    higgs_item.set_temperature_ratio(3.5)  # max 2.0
    assert higgs_item._temperature_ratio == 2.0

    higgs_item.set_temperature_ratio(-0.5)  # min 0.0
    assert higgs_item._temperature_ratio == 0.0


def test_step_physics_and_pause(higgs_item):
    """Test step_physics advances time when running and freezes when paused."""
    higgs_item._sim_time = 0.0
    higgs_item.step_physics(0.05)
    assert higgs_item._sim_time == pytest.approx(0.05, 1e-4)

    higgs_item.toggle_pause()
    assert higgs_item._is_paused is True
    higgs_item.step_physics(0.1)
    assert higgs_item._sim_time == pytest.approx(0.05, 1e-4)

    higgs_item.toggle_pause()
    assert higgs_item._is_paused is False
    higgs_item.step_physics(0.05)
    assert higgs_item._sim_time == pytest.approx(0.10, 1e-4)


def test_mouse_interactions(higgs_item):
    """Test clicking preset pills, accumulate events, and reset buttons."""
    scene = QGraphicsScene()
    scene.addItem(higgs_item)
    try:
        # Click Preset 1 (Four Lepton)
        rects = higgs_item._preset_button_rects()
        assert len(rects) >= 2
        p1_center = rects[1][0].center()

        event_p1 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_p1.setPos(p1_center)
        event_p1.setButton(Qt.MouseButton.LeftButton)
        higgs_item.mousePressEvent(event_p1)
        assert higgs_item.current_preset.key == "higgs_four_lepton"

        # Click Accumulate Events button (Col 2 in controls)
        ctrl_rect = higgs_item._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_w = col_w - 20.0
        b3_rect = QRectF(ctrl_rect.left() + 20.0 + 2 * col_w, ctrl_rect.top() + 22.0, btn_w, 38.0)

        init_lumi = higgs_item._luminosity_fb
        event_b3 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_b3.setPos(b3_rect.center())
        event_b3.setButton(Qt.MouseButton.LeftButton)
        higgs_item.mousePressEvent(event_b3)
        assert higgs_item._luminosity_fb > init_lumi

        # Click Reset button (Col 3 left sub-button)
        c4_x = ctrl_rect.left() + 20.0 + 3 * col_w
        b4_w = (btn_w - 10.0) / 2.0
        rst_rect = QRectF(c4_x, ctrl_rect.top() + 22.0, b4_w, 38.0)
        event_rst = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_rst.setPos(rst_rect.center())
        event_rst.setButton(Qt.MouseButton.LeftButton)
        higgs_item.mousePressEvent(event_rst)
        assert higgs_item._sim_time == 0.0
    finally:
        scene.removeItem(higgs_item)


def test_serialization_roundtrip(higgs_item):
    """Test serialization to dict and reconstruction from dict."""
    higgs_item.set_preset(1)  # Four Lepton
    higgs_item.set_luminosity_fb(220.0)
    higgs_item.set_temperature_ratio(0.4)
    higgs_item.setPos(160.0, 240.0)

    data = higgs_item.to_dict()
    assert data["type"] == "InteractiveHiggsMechanismItem"
    assert data["preset_key"] == "higgs_four_lepton"
    assert data["luminosity_fb"] == 220.0
    assert data["temperature_ratio"] == 0.4
    assert data["pos"] == [160.0, 240.0]

    reconstructed = InteractiveHiggsMechanismItem.from_dict(data)
    try:
        assert reconstructed.current_preset.key == "higgs_four_lepton"
        assert reconstructed._luminosity_fb == 220.0
        assert reconstructed._temperature_ratio == 0.4
        assert reconstructed.pos().x() == 160.0
        assert reconstructed.pos().y() == 240.0
    finally:
        reconstructed.stop()


def test_paint_and_cleanup(higgs_item):
    """Verify paint rendering without crash and timer stop cleanup."""
    img = QImage(940, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    try:
        higgs_item.paint(painter, None, None)
    finally:
        painter.end()
    assert img.pixelColor(20, 20).isValid()

    assert higgs_item._anim_timer.isActive()
    higgs_item.stop()
    assert not higgs_item._anim_timer.isActive()
