"""
Unit tests for InteractivePhotoelectricEffectItem and PhysicsTools.create_photoelectric_effect_diagram.
Verifies Einstein's photoelectric equation E_k = hν - Φ = e·U_h, work functions of cathode metals,
red threshold wavelength λ₀ = hc/Φ, stopping potential U_h(ν), Millikan's slope h/e,
independence of stopping potential on light intensity, current-voltage I-U_AK curve,
touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_photoelectric_effect import (
    InteractivePhotoelectricEffectItem,
    PHOTOELECTRIC_PRESETS,
    METALS,
    H_PLANCK_EV_S,
    C_LIGHT,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_photoelectric_initialization(qapp):
    item = InteractivePhotoelectricEffectItem()
    assert item.active_preset.code == "CESIUM_VISIBLE_LIGHT_THRESHOLD"
    assert item.metal.symbol == "Cs"
    assert math.isclose(item.work_function_ev, 2.14, rel_tol=1e-4)
    assert math.isclose(item.wavelength_nm, 532.0, rel_tol=1e-4)
    assert item.is_photoelectric_active is True
    assert item.boundingRect().width() == InteractivePhotoelectricEffectItem.CARD_W
    assert item.boundingRect().height() == InteractivePhotoelectricEffectItem.CARD_H


def test_einstein_photoelectric_equation_exact_math(qapp):
    """
    Verifies Einstein's photoelectric equation:
    E = hc / λ
    E_k = max(0, E - Φ)
    U_h = E_k / e
    """
    item = InteractivePhotoelectricEffectItem()
    # 1. Above threshold: Green laser on Cesium (532 nm, Φ = 2.14 eV)
    # E = 1239.84 / 532 ≈ 2.3305 eV
    # E_k = 2.3305 - 2.14 = 0.1905 eV
    # U_h = 0.1905 V
    e_photon = item.photon_energy_ev
    expected_e = 1239.8419 / 532.0
    assert math.isclose(e_photon, expected_e, rel_tol=1e-4)

    expected_ek = expected_e - 2.14
    assert math.isclose(item.max_kinetic_energy_ev, expected_ek, rel_tol=1e-4)
    assert math.isclose(item.stopping_potential_uh_v, expected_ek, rel_tol=1e-4)

    # 2. Below threshold: Red light (700 nm) on Cesium (E = 1.77 eV < 2.14 eV)
    item.step_wavelength_nm(168.0)  # 532 + 168 = 700 nm
    assert item.is_photoelectric_active is False
    assert item.max_kinetic_energy_ev == 0.0
    assert item.stopping_potential_uh_v == 0.0
    assert item.measured_photocurrent_ua == 0.0


def test_red_threshold_wavelength_and_frequency(qapp):
    """
    Verifies red threshold wavelength λ_0 = hc / Φ across metals.
    """
    item = InteractivePhotoelectricEffectItem()

    # Cesium: Φ = 2.14 eV -> λ_0 ≈ 579.4 nm
    item.set_metal("CESIUM")
    assert math.isclose(item.red_threshold_wavelength_nm, 1239.8419 / 2.14, rel_tol=1e-4)

    # Zinc: Φ = 4.31 eV -> λ_0 ≈ 287.7 nm (UV only)
    item.set_metal("ZINC")
    assert math.isclose(item.red_threshold_wavelength_nm, 1239.8419 / 4.31, rel_tol=1e-4)

    # Platinum: Φ = 6.35 eV -> λ_0 ≈ 195.3 nm (VUV only)
    item.set_metal("PLATINUM")
    assert math.isclose(item.red_threshold_wavelength_nm, 1239.8419 / 6.35, rel_tol=1e-4)


def test_stopping_potential_and_max_electron_velocity(qapp):
    """
    Verifies maximum electron velocity:
    v_max = √(2 E_k / m_e)
    """
    item = InteractivePhotoelectricEffectItem()
    item.load_preset("SODIUM_MILLIKAN_PLANCK_MEASUREMENT")

    ek = item.max_kinetic_energy_ev
    v_max = item.max_velocity_m_s

    assert ek > 0.5
    # Velocity should be around ~ 5.7 x 10^5 m/s
    assert 1e5 < v_max < 3e6


def test_millikan_linear_slope_determination(qapp):
    """
    Millikan (1916) experimental verification:
    The slope of U_h vs frequency ν is universally h / e:
    ΔU_h / Δν = h / e = 4.13567 x 10⁻¹⁵ V·s
    """
    item = InteractivePhotoelectricEffectItem()
    item.set_metal("SODIUM")

    # Wavelength 1: 300 nm
    item._lambda_nm = 300.0
    nu1 = item.light_frequency_hz
    uh1 = item.stopping_potential_uh_v

    # Wavelength 2: 400 nm
    item._lambda_nm = 400.0
    nu2 = item.light_frequency_hz
    uh2 = item.stopping_potential_uh_v

    slope = (uh1 - uh2) / (nu1 - nu2)
    assert math.isclose(slope, H_PLANCK_EV_S, rel_tol=1e-4)


def test_light_intensity_independence_of_stopping_potential(qapp):
    """
    Core quantum milestone: Light intensity affects ONLY the saturation current I_s,
    while stopping potential U_h and max kinetic energy E_k remain completely invariant!
    """
    item = InteractivePhotoelectricEffectItem()
    item.load_preset("LIGHT_INTENSITY_INDEPENDENCE")

    init_uh = item.stopping_potential_uh_v
    init_ek = item.max_kinetic_energy_ev
    init_current = item.measured_photocurrent_ua

    # Double the intensity
    item.step_intensity_mw(100.0)
    assert item.stopping_potential_uh_v == init_uh
    assert item.max_kinetic_energy_ev == init_ek
    assert item.measured_photocurrent_ua > init_current


def test_current_voltage_retarding_characteristic(qapp):
    """
    Verifies I-U_AK curve behavior:
    - Retarding potential U_AK <= -U_h -> I = 0
    - Intermediate -U_h < U_AK < 0 -> 0 < I < I_s
    - Forward accelerating U_AK >= 0 -> I = I_s (saturation)
    """
    item = InteractivePhotoelectricEffectItem()
    item.load_preset("CESIUM_VISIBLE_LIGHT_THRESHOLD")
    uh = item.stopping_potential_uh_v
    i_sat = item.saturation_current_ua

    # 1. Strong retarding potential: U_AK = -2.0 V <= -U_h (-0.19 V)
    item._voltage_uak_v = -2.0
    assert item.measured_photocurrent_ua == 0.0

    # 2. Moderate retarding potential: U_AK = -0.10 V
    item._voltage_uak_v = -0.10
    i_ret = item.measured_photocurrent_ua
    assert 0.0 < i_ret < i_sat

    # 3. Forward voltage: U_AK = +2.0 V -> Saturation
    item._voltage_uak_v = 2.0
    assert math.isclose(item.measured_photocurrent_ua, i_sat, rel_tol=1e-5)


def test_all_five_presets_load(qapp):
    item = InteractivePhotoelectricEffectItem()
    for preset_code, preset in PHOTOELECTRIC_PRESETS.items():
        item.load_preset(preset_code)
        assert item.active_preset.code == preset_code
        assert item.metal == preset.metal
        assert item.wavelength_nm == preset.wavelength_nm
        assert item.photon_energy_ev > 0.0
        assert item.work_function_ev > 0.0


def test_parameter_steppers_and_metal_selection(qapp):
    item = InteractivePhotoelectricEffectItem()
    init_w = item.wavelength_nm
    item.step_wavelength_nm(40.0)
    assert math.isclose(item.wavelength_nm, init_w + 40.0, rel_tol=1e-4)

    init_v = item._voltage_uak_v
    item.step_voltage_v(0.4)
    assert math.isclose(item._voltage_uak_v, init_v + 0.4, rel_tol=1e-4)

    item.set_metal("PLATINUM")
    assert item.metal.symbol == "Pt"

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractivePhotoelectricEffectItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(20.0, 35.0))
    assert item.pos() == start_pos + QPointF(20.0, 35.0)

    # Air pinch on btn_m_zn (switch to Zinc)
    rect_m_zn = item._buttons["btn_m_zn"]
    scene_pt = item.mapToScene(rect_m_zn.center())
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item.metal.symbol == "Zn"


def test_vector_diagram_generation_and_stamping(qapp):
    item = InteractivePhotoelectricEffectItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "work_function_ev" in stamped[0]
    assert "stopping_potential_v" in stamped[0]
    assert "photocurrent_ua" in stamped[0]

    items = PhysicsTools.create_photoelectric_effect_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="CESIUM_VISIBLE_LIGHT_THRESHOLD",
        metal_name="Xesi (Cs)",
        work_function_ev=2.14,
        wavelength_nm=532.0,
        photon_energy_ev=2.33,
        stopping_potential_v=0.19,
        voltage_uak_v=1.0,
        photocurrent_ua=5.0,
        red_threshold_nm=579.4,
    )
    assert len(items) >= 15
    assert any("QUANG ĐIỆN" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_photoelectric_lifecycle(qapp):
    board = LessonBoard()
    assert board.is_photoelectric_visible() is False
    assert board.get_photoelectric() is None

    # Show
    item = board.show_photoelectric(True)
    assert item is not None
    assert board.is_photoelectric_visible() is True
    assert board.get_photoelectric() is item
    assert board.get_photoelectric_item() is item

    # Toggle to hide
    visible = board.toggle_photoelectric()
    assert visible is False
    assert board.is_photoelectric_visible() is False
    assert board.get_photoelectric() is None
    assert board.get_photoelectric_item() is not None

    # Toggle to show
    visible = board.toggle_photoelectric()
    assert visible is True
    assert board.is_photoelectric_visible() is True
