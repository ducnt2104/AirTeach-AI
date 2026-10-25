"""
Unit tests for InteractiveBandTheoryItem - Solid-State Band Theory & Kronig-Penney.
===================================================================================
Covers Kronig-Penney dispersion, energy bands, Fermi-Dirac statistics, intrinsic/extrinsic
doping, carrier densities, UI interaction sliders, and serialization.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_band_theory import (
    InteractiveBandTheoryItem,
    PRESETS,
    BandTheoryPreset,
    BOLTZMANN_K_EV,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    item = InteractiveBandTheoryItem()
    assert item.WIDTH == 940.0
    assert item.HEIGHT == 600.0
    assert item.current_preset.key == "intrinsic_silicon"
    assert math.isclose(item._eg_ev, 1.12)
    assert math.isclose(item._temp_k, 300.0)
    assert not item._is_paused


def test_presets(qapp):
    item = InteractiveBandTheoryItem()
    keys = ["intrinsic_silicon", "n_type_doped", "p_type_doped", "direct_gap_gaas", "wide_gap_insulator"]

    for i, expected_key in enumerate(keys):
        item.set_preset(i)
        assert item.current_preset.key == expected_key
        assert item._sim_time == 0.0

    curr_idx = item._preset_index
    item.set_preset(99)
    assert item._preset_index == curr_idx


def test_fermi_level_calculation(qapp):
    item = InteractiveBandTheoryItem()

    # Intrinsic silicon: EF at midgap (0.0 eV)
    item.set_preset(0)
    assert math.isclose(item.compute_fermi_level(), 0.0, abs_tol=1e-4)

    # n-type silicon: EF shifted upwards toward Ec = +0.56 eV
    item.set_preset(1)
    ef_n = item.compute_fermi_level()
    assert ef_n > 0.3
    assert ef_n < 0.56

    # p-type silicon: EF shifted downwards toward Ev = -0.56 eV
    item.set_preset(2)
    ef_p = item.compute_fermi_level()
    assert ef_p < -0.3
    assert ef_p > -0.56


def test_fermi_dirac_distribution(qapp):
    item = InteractiveBandTheoryItem()
    item.set_preset(0)  # EF = 0.0
    ef = item.compute_fermi_level()

    # Exactly at EF, probability must be 0.5 (50%)
    f_ef = item.fermi_dirac(ef)
    assert math.isclose(f_ef, 0.5, abs_tol=1e-3)

    # Far below EF (in valence band, e.g. -2.0 eV), probability ~ 1.0 (fully occupied)
    f_vb = item.fermi_dirac(-2.0)
    assert f_vb > 0.999

    # Far above EF (in conduction band, e.g. +2.0 eV), probability ~ 0.0 (empty)
    f_cb = item.fermi_dirac(+2.0)
    assert f_cb < 0.001


def test_carrier_densities_mass_action(qapp):
    item = InteractiveBandTheoryItem()

    # Intrinsic silicon at 300K: ni ≈ 1e10 cm^-3, n = p = ni
    item.set_preset(0)
    ni, n, p = item.compute_carrier_densities()
    assert 1e9 < ni < 1e11
    assert math.isclose(n, ni, rel_tol=1e-3)
    assert math.isclose(p, ni, rel_tol=1e-3)

    # n-type silicon: n ≈ Nd = 1e16 cm^-3 >> p
    item.set_preset(1)
    ni_n, n_n, p_n = item.compute_carrier_densities()
    assert math.isclose(n_n, 1e16, rel_tol=0.1)
    assert p_n < 1e6

    # Wide bandgap insulator (Diamond, Eg = 5.47 eV): ni ≈ 0 at 300K
    item.set_preset(4)
    ni_dia, n_dia, p_dia = item.compute_carrier_densities()
    assert ni_dia < 1e-10
    assert n_dia < 1e-10


def test_parameter_setters(qapp):
    item = InteractiveBandTheoryItem()
    # Temperature
    item.set_temperature(450.0)
    assert math.isclose(item._temp_k, 450.0)
    item.set_temperature(1000.0)  # Clamped to 600 K
    assert math.isclose(item._temp_k, 600.0)

    # Bandgap
    item.set_bandgap(2.5)
    assert math.isclose(item._eg_ev, 2.5)
    item.set_bandgap(15.0)  # Clamped to 6.5 eV
    assert math.isclose(item._eg_ev, 6.5)

    # Barrier strength
    item.set_barrier_strength(12.0)
    assert math.isclose(item._p_barrier, 12.0)
    item.set_barrier_strength(50.0)  # Clamped to 25.0
    assert math.isclose(item._p_barrier, 25.0)


def test_pause_and_reset(qapp):
    item = InteractiveBandTheoryItem()
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
    item = InteractiveBandTheoryItem()
    for _ in range(10):
        item.step_physics(0.033)

    img = QImage(940, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()

    assert not img.isNull()
    assert img.width() == 940


def test_mouse_interactions(qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    item = InteractiveBandTheoryItem()

    def make_event(event_type, pos_f):
        e = QGraphicsSceneMouseEvent(event_type)
        e.setPos(pos_f)
        e.setScenePos(pos_f)
        e.setButton(Qt.LeftButton)
        return e

    # Click Preset 3 (direct_gap_gaas)
    p_rects = item._preset_button_rects()
    target_rect, _ = p_rects[3]
    click_ev = make_event(QEvent.GraphicsSceneMousePress, target_rect.center())
    item.mousePressEvent(click_ev)
    assert item.current_preset.key == "direct_gap_gaas"

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

    # Drag slider (temp)
    track_rect = QRectF(ctrl_rect.left() + 20.0, ctrl_rect.top() + 36.0, col_w - 25.0, 10.0)
    drag_start = make_event(QEvent.GraphicsSceneMousePress, track_rect.center())
    item.mousePressEvent(drag_start)
    assert item._dragging_slider == "temp"

    drag_move = make_event(QEvent.GraphicsSceneMouseMove, track_rect.topRight())
    item.mouseMoveEvent(drag_move)

    drag_rel = make_event(QEvent.GraphicsSceneMouseRelease, track_rect.topRight())
    item.mouseReleaseEvent(drag_rel)
    assert item._dragging_slider is None


def test_serialization_roundtrip(qapp):
    item = InteractiveBandTheoryItem()
    item.set_preset(2)  # p_type_doped
    item.set_temperature(350.0)
    item.set_bandgap(1.25)
    item.set_barrier_strength(8.0)
    item.setPos(180.0, 220.0)
    for _ in range(10):
        item.step_physics(0.033)

    data = item.to_dict()
    assert data["type"] == "InteractiveBandTheoryItem"
    assert data["preset_key"] == "p_type_doped"
    assert math.isclose(data["temp_k"], 350.0, abs_tol=1e-3)
    assert math.isclose(data["eg_ev"], 1.25, abs_tol=1e-3)
    assert math.isclose(data["p_barrier"], 8.0, abs_tol=1e-3)

    restored = InteractiveBandTheoryItem.from_dict(data)
    assert restored.current_preset.key == "p_type_doped"
    assert math.isclose(restored._temp_k, 350.0, abs_tol=1e-3)
    assert math.isclose(restored._eg_ev, 1.25, abs_tol=1e-3)
    assert math.isclose(restored._p_barrier, 8.0, abs_tol=1e-3)
    assert restored.pos() == QPointF(180.0, 220.0)
