"""
Unit tests for InteractiveElectromagneticWaveItem and PhysicsTools.create_electromagnetic_wave_diagram.
Verifies Maxwell's transverse wave solutions (E ⊥ B ⊥ k), Poynting energy flux vector S = (E x B) / μ₀,
radiation pressure, polarization classifications (Linear, Circular RCP/LCP, Elliptical),
Stokes parameter invariants (S₀² = S₁² + S₂² + S₃²), touchless air gestures,
and lesson board lifecycle integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_em_wave import (
    InteractiveElectromagneticWaveItem,
    EM_PRESETS,
    C_LIGHT,
    EPSILON_0,
    MU_0,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_em_wave_initialization(qapp):
    item = InteractiveElectromagneticWaveItem()
    assert item.active_preset.code == "LINEAR_POLARIZATION_45_DEG"
    assert item.boundingRect().width() == InteractiveElectromagneticWaveItem.CARD_W
    assert item.boundingRect().height() == InteractiveElectromagneticWaveItem.CARD_H

    # Wavelength: 532 nm (Green laser)
    assert item._wavelength_nm == 532.0
    f_expected = C_LIGHT / (532.0 * 1e-9)
    assert math.isclose(item.frequency_hz, f_expected, rel_tol=1e-4)

    # Stokes parameter S3 should be zero for linear polarization
    s0, s1, s2, s3 = item.stokes_parameters
    assert math.isclose(s3, 0.0, abs_tol=1e-5)
    assert "Tuyến tính" in item.get_polarization_classification()


def test_electrodynamics_maxwell_relationships(qapp):
    item = InteractiveElectromagneticWaveItem()

    # Speed of light in vacuum relation c = 1 / sqrt(eps0 * mu0)
    c_calc = 1.0 / math.sqrt(EPSILON_0 * MU_0)
    assert math.isclose(c_calc, C_LIGHT, rel_tol=1e-3)

    # Magnetic field amplitude B0 = E0 / c
    e0_total = math.hypot(item._e0x, item._e0y)
    b0_calc = e0_total / C_LIGHT
    assert math.isclose(item.b0_tesla, b0_calc, rel_tol=1e-5)

    # Poynting intensity <S> = 1/2 * c * eps0 * E0^2
    s_calc = 0.5 * C_LIGHT * EPSILON_0 * (e0_total**2)
    assert math.isclose(item.poynting_intensity_w_m2, s_calc, rel_tol=1e-5)

    # Radiation pressure P_rad = <S> / c
    p_rad_calc = (s_calc / C_LIGHT) * 1e6  # in μPa
    assert math.isclose(item.radiation_pressure_micro_pa, p_rad_calc, rel_tol=1e-5)


def test_circular_polarization_rcp_and_lcp(qapp):
    item = InteractiveElectromagneticWaveItem()

    # 1. RCP: delta = -pi/2 (-90°)
    item.load_preset("CIRCULAR_RIGHT_HANDED_RCP")
    assert math.isclose(item._delta, -math.pi / 2.0, rel_tol=1e-4)
    s0, s1, s2, s3 = item.stokes_parameters
    assert math.isclose(s1, 0.0, abs_tol=1e-4)
    assert math.isclose(s2, 0.0, abs_tol=1e-4)
    assert math.isclose(s3 / s0, -1.0, rel_tol=1e-4)
    assert "Tròn Phải (RCP" in item.get_polarization_classification()

    # 2. LCP: delta = +pi/2 (+90°)
    item.load_preset("CIRCULAR_LEFT_HANDED_LCP")
    assert math.isclose(item._delta, math.pi / 2.0, rel_tol=1e-4)
    s0, s1, s2, s3 = item.stokes_parameters
    assert math.isclose(s1, 0.0, abs_tol=1e-4)
    assert math.isclose(s2, 0.0, abs_tol=1e-4)
    assert math.isclose(s3 / s0, 1.0, rel_tol=1e-4)
    assert "Tròn Trái (LCP" in item.get_polarization_classification()


def test_stokes_parameter_invariant_for_polarized_light(qapp):
    item = InteractiveElectromagneticWaveItem()
    # For any 100% polarized wave: S0^2 = S1^2 + S2^2 + S3^2
    for preset_code in EM_PRESETS:
        item.load_preset(preset_code)
        s0, s1, s2, s3 = item.stokes_parameters
        sum_sq = s1**2 + s2**2 + s3**2
        assert math.isclose(s0**2, sum_sq, rel_tol=1e-4), f"Stokes invariant violated in {preset_code}"


def test_elliptical_polarization_general(qapp):
    item = InteractiveElectromagneticWaveItem()
    item.load_preset("ELLIPTICAL_GENERAL_POLARIZATION")

    assert item._e0x != item._e0y
    assert item._delta != 0.0
    pol_name = item.get_polarization_classification()
    assert "Elip" in pol_name


def test_all_five_presets_load(qapp):
    item = InteractiveElectromagneticWaveItem()
    preset_keys = [
        "LINEAR_POLARIZATION_45_DEG",
        "CIRCULAR_RIGHT_HANDED_RCP",
        "CIRCULAR_LEFT_HANDED_LCP",
        "ELLIPTICAL_GENERAL_POLARIZATION",
        "QUARTER_WAVE_RETARDER_CRYSTAL",
    ]
    for key in preset_keys:
        item.load_preset(key)
        assert item.active_preset.code == key
        assert len(item.active_preset.title_vi) > 0
        assert item.frequency_hz > 1e14
        assert item.poynting_intensity_w_m2 > 0.0
        assert len(item.get_polarization_classification()) > 0


def test_parameter_steppers(qapp):
    item = InteractiveElectromagneticWaveItem()
    init_ex = item._e0x
    item.step_amplitude(20.0)
    assert item._e0x == init_ex + 20.0
    item.step_amplitude(-40.0)
    assert item._e0x == init_ex - 20.0

    init_delta = item._delta
    item.step_phase_delta(math.pi / 4.0)
    assert math.isclose(item._delta, (init_delta + math.pi / 4.0), rel_tol=1e-4)

    init_lam = item._wavelength_nm
    item.step_wavelength(50.0)
    assert item._wavelength_nm == init_lam + 50.0


def test_toggle_controls(qapp):
    item = InteractiveElectromagneticWaveItem()

    # Animation
    init_run = item._is_running
    item.toggle_animation()
    assert item._is_running != init_run

    # B-field
    init_b = item._show_b_field
    item.toggle_b_field()
    assert item._show_b_field != init_b

    # Poynting
    init_s = item._show_poynting
    item.toggle_poynting()
    assert item._show_poynting != init_s


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractiveElectromagneticWaveItem(pos=(100.0, 80.0))
    init_pos = item.pos()
    item.handle_air_drag(QPointF(25.0, -15.0))
    assert item.pos().x() == init_pos.x() + 25.0
    assert item.pos().y() == init_pos.y() - 15.0

    # Pinch on btn_e_up
    btn_e_up = item._buttons.get("btn_e_up")
    assert btn_e_up is not None
    init_ex = item._e0x
    scene_pt = item.mapToScene(btn_e_up.center())
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item._e0x == init_ex + 20.0

    # Pinch on btn_phase_rcp
    btn_rcp = item._buttons.get("btn_phase_rcp")
    assert btn_rcp is not None
    scene_rcp = item.mapToScene(btn_rcp.center())
    item.handle_air_pinch(scene_rcp)
    assert math.isclose(item._delta, -math.pi / 2.0, rel_tol=1e-4)


def test_vector_diagram_generation_and_stamping(qapp):
    items = PhysicsTools.create_electromagnetic_wave_diagram(
        center=(340.0, 190.0),
        width=680.0,
        height=380.0,
        preset_code="LINEAR_POLARIZATION_45_DEG",
        e0x=100.0,
        e0y=100.0,
        delta_rad=0.0,
        wavelength_nm=532.0,
        intensity_w_m2=26.54,
        b0_ut=0.472,
        rad_pressure_upa=0.089,
        polarization_name="Tuyến tính (Linear 45°)",
    )
    assert len(items) >= 12
    bg = items[0]
    assert bg.rect().width() == 680.0
    assert bg.rect().height() == 380.0

    # Test stamp signal emission
    item = InteractiveElectromagneticWaveItem()
    stamped = []
    item.stamp_requested.connect(lambda data: stamped.append(data))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert stamped[0]["preset_code"] == "LINEAR_POLARIZATION_45_DEG"
    assert "intensity_w_m2" in stamped[0]


def test_lesson_board_em_wave_lifecycle(qapp):
    board = LessonBoard()
    assert not board.is_em_wave_visible()
    assert board.get_em_wave() is None

    # Show item
    item = board.show_em_wave(True, pos=(120.0, 80.0))
    assert item is not None
    assert board.is_em_wave_visible()
    assert board.get_em_wave() == item
    assert board.get_em_wave_item() == item

    # Toggle off
    res = board.toggle_em_wave()
    assert not res
    assert not board.is_em_wave_visible()
    assert board.get_em_wave() is None
    assert board.get_em_wave_item() is not None

    # Toggle on
    res2 = board.toggle_em_wave()
    assert res2
    assert board.is_em_wave_visible()
