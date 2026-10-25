"""
Unit tests for InteractiveSuperconductivityItem - Superconductivity & Quantum Levitation.
========================================================================================
Covers BCS critical fields, London penetration depth, Ginzburg-Landau classification,
vortex flux pinning, quantum levitation dynamics, thermal quench, and serialization.
"""
import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_superconductivity import (
    InteractiveSuperconductivityItem,
    PRESETS,
    SuperconductorPreset,
    FLUX_QUANTUM_PHI0,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_initialization(qapp):
    item = InteractiveSuperconductivityItem()
    assert item.WIDTH == 940.0
    assert item.HEIGHT == 600.0
    assert item.current_preset.key == "type2_flux_pinning"
    assert math.isclose(item._temp_k, 77.0)
    assert item.current_preset.is_type2
    assert not item._is_paused
    assert item._magnet_z == 38.0


def test_presets(qapp):
    item = InteractiveSuperconductivityItem()
    keys = ["type1_meissner", "type2_flux_pinning", "thermal_quench", "magnetic_breakdown", "quantum_track_loop"]

    for i, expected_key in enumerate(keys):
        item.set_preset(i)
        assert item.current_preset.key == expected_key
        assert item._sim_time == 0.0

    # Bounds check
    curr_idx = item._preset_index
    item.set_preset(99)
    assert item._preset_index == curr_idx


def test_bcs_critical_field_temperature_dependence(qapp):
    item = InteractiveSuperconductivityItem()
    item.set_preset(0)  # Type I Lead (Pb), Tc = 7.2 K, Bc0 = 0.08 T
    p = item.current_preset

    # T = 0 K limit (approx 0.1 K)
    item.set_temperature(0.1)
    bc, _, _ = item.compute_critical_fields()
    assert math.isclose(bc, p.bc0_t, rel_tol=0.01)

    # T = Tc / 2 = 3.6 K -> Bc = Bc0 * (1 - 0.25) = 0.75 * Bc0
    item.set_temperature(3.6)
    bc_half, _, _ = item.compute_critical_fields()
    expected_bc = p.bc0_t * 0.75
    assert math.isclose(bc_half, expected_bc, rel_tol=0.01)

    # T >= Tc -> Bc = 0
    item.set_temperature(7.5)
    bc_supra, _, _ = item.compute_critical_fields()
    assert bc_supra == 0.0


def test_london_penetration_depth_divergence(qapp):
    item = InteractiveSuperconductivityItem()
    item.set_preset(1)  # YBCO, Tc = 93 K, lambda0 = 150 nm
    p = item.current_preset

    # At low temperature T = 10 K << Tc
    item.set_temperature(10.0)
    lam_low = item.compute_london_penetration_depth()
    assert math.isclose(lam_low, p.lambda0_nm, rel_tol=0.01)

    # As T approaches Tc = 93 K, lambda_L increases significantly
    item.set_temperature(88.0)
    lam_high = item.compute_london_penetration_depth()
    assert lam_high > 1.8 * p.lambda0_nm

    # At T >= Tc, penetration depth diverges
    item.set_temperature(95.0)
    lam_normal = item.compute_london_penetration_depth()
    assert lam_normal >= 1e5


def test_ginzburg_landau_type_classification(qapp):
    item = InteractiveSuperconductivityItem()

    # Preset 0: Lead Pb (Type I)
    item.set_preset(0)
    p1 = item.current_preset
    kappa1 = p1.lambda0_nm / p1.xi0_nm
    assert kappa1 < 1.0 / math.sqrt(2.0)  # kappa < 0.707 -> Type I
    assert not p1.is_type2

    # Preset 1: YBCO (Type II)
    item.set_preset(1)
    p2 = item.current_preset
    kappa2 = p2.lambda0_nm / p2.xi0_nm
    assert kappa2 > 1.0 / math.sqrt(2.0)  # kappa = 100 >> 0.707 -> Type II
    assert p2.is_type2


def test_superconducting_state_transitions(qapp):
    item = InteractiveSuperconductivityItem()
    item.set_preset(1)  # YBCO: Tc = 93 K, Bc1 ≈ 1.0 mT at 77K, Bc2 ≈ 38 T

    # Low field B = 0.5 mT < Bc1 -> Meissner state
    item.set_temperature(77.0)
    item.set_magnetic_field(0.0005)
    assert item.get_superconducting_state() == "meissner"

    # Moderate field Bc1 < B = 1.0 T < Bc2 -> Mixed Vortex state
    item.set_magnetic_field(1.0)
    assert item.get_superconducting_state() == "vortex"

    # High temperature T = 100 K > Tc -> Normal state
    item.set_temperature(100.0)
    assert item.get_superconducting_state() == "normal"


def test_quantum_levitation_restoring_force(qapp):
    item = InteractiveSuperconductivityItem()
    item.set_preset(1)  # YBCO with strong pinning
    assert item.get_superconducting_state() == "vortex"

    # Displace magnet upwards to 65 mm (equilibrium is ~42 mm)
    item._magnet_z = 65.0
    item._magnet_dz = 0.0

    # Step physics for 0.5s
    dt = 0.033
    for _ in range(15):
        item.step_physics(dt)

    # Restoring force must accelerate magnet back downwards
    assert item._magnet_z < 65.0


def test_thermal_quench_dynamics(qapp):
    item = InteractiveSuperconductivityItem()
    item.set_preset(2)  # thermal quench preset starting at 88 K
    assert item._temp_k == 88.0

    # Step simulation for warming (0.35 K/s)
    dt = 0.1
    for _ in range(200):  # 20 seconds of simulation
        item.step_physics(dt)

    # Temperature must have crossed Tc = 93 K
    assert item._temp_k > 93.0
    assert item.get_superconducting_state() == "normal"
    # In normal state, levitation collapses (z drops close to surface)
    assert item._magnet_z < 10.0


def test_parameter_setters(qapp):
    item = InteractiveSuperconductivityItem()
    # Temperature
    item.set_temperature(50.0)
    assert math.isclose(item._temp_k, 50.0)
    item.set_temperature(200.0)  # Clamped to 150 K
    assert item._temp_k <= 150.0

    # Magnetic field
    item.set_magnetic_field(5.5)
    assert math.isclose(item._b_ext, 5.5)
    item.set_magnetic_field(50.0)  # Clamped to 25 T
    assert math.isclose(item._b_ext, 25.0)

    # Pinning density
    item.set_pinning_density(0.65)
    assert math.isclose(item._pinning, 0.65)
    item.set_pinning_density(2.0)  # Clamped to 1.0
    assert math.isclose(item._pinning, 1.0)


def test_pause_and_reset(qapp):
    item = InteractiveSuperconductivityItem()
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
    item = InteractiveSuperconductivityItem()
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

    item = InteractiveSuperconductivityItem()

    def make_event(event_type, pos_f):
        e = QGraphicsSceneMouseEvent(event_type)
        e.setPos(pos_f)
        e.setScenePos(pos_f)
        e.setButton(Qt.LeftButton)
        return e

    # Click Preset 0 (type1_meissner)
    p_rects = item._preset_button_rects()
    target_rect, _ = p_rects[0]
    click_ev = make_event(QEvent.GraphicsSceneMousePress, target_rect.center())
    item.mousePressEvent(click_ev)
    assert item.current_preset.key == "type1_meissner"

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
    item = InteractiveSuperconductivityItem()
    item.set_preset(4)  # quantum_track_loop
    item.set_temperature(65.0)
    item.set_magnetic_field(1.2)
    item.set_pinning_density(0.88)
    item.setPos(150.0, 250.0)
    for _ in range(10):
        item.step_physics(0.033)

    data = item.to_dict()
    assert data["type"] == "InteractiveSuperconductivityItem"
    assert data["preset_key"] == "quantum_track_loop"
    assert math.isclose(data["temp_k"], 65.0, abs_tol=1e-3)
    assert math.isclose(data["b_ext"], 1.2, abs_tol=1e-3)
    assert math.isclose(data["pinning"], 0.88, abs_tol=1e-3)

    restored = InteractiveSuperconductivityItem.from_dict(data)
    assert restored.current_preset.key == "quantum_track_loop"
    assert math.isclose(restored._temp_k, 65.0, abs_tol=1e-3)
    assert math.isclose(restored._b_ext, 1.2, abs_tol=1e-3)
    assert math.isclose(restored._pinning, 0.88, abs_tol=1e-3)
    assert restored.pos() == QPointF(150.0, 250.0)
