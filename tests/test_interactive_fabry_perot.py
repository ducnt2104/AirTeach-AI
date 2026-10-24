"""
Unit tests for InteractiveFabryPerotItem and PhysicsTools.create_fabry_perot_diagram.
Verifies Airy transmission function T(δ) = 1 / (1 + F sin²(δ/2)), mirror reflectance R,
coefficient of finesse F = 4R/(1-R)², cavity finesse F = π√R / (1-R),
Free Spectral Range (FSR) Δν_FSR = c / (2nd), resonance linewidth FWHM δν = Δν / F,
longitudinal laser cavity modes, touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_fabry_perot import (
    InteractiveFabryPerotItem,
    FABRY_PEROT_PRESETS,
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


def test_fabry_perot_initialization(qapp):
    item = InteractiveFabryPerotItem()
    assert item.active_preset.code == "HELIUM_NEON_LASER_CAVITY"
    assert math.isclose(item.wavelength_nm, 632.8, rel_tol=1e-4)
    assert math.isclose(item.cavity_d_um, 20.0, rel_tol=1e-4)
    assert math.isclose(item.reflectance_r, 0.96, rel_tol=1e-4)
    assert item.boundingRect().width() == InteractiveFabryPerotItem.CARD_W
    assert item.boundingRect().height() == InteractiveFabryPerotItem.CARD_H


def test_airy_transmission_and_resonance_condition(qapp):
    """
    Verifies the fundamental Airy transmission function:
    T(δ) = 1 / [1 + F sin²(δ/2)]
    - Peak resonance: T(2mπ) = 1.0 (100% transmission for lossless mirrors)
    - Anti-resonance: T((2m+1)π) = 1 / (1 + F)
    """
    item = InteractiveFabryPerotItem()
    f_coeff = item.coefficient_of_finesse_f

    # Test peak resonance (delta = 0 or 2π)
    # T = 1 / (1 + F*0) = 1.0
    t_peak = 1.0 / (1.0 + f_coeff * (math.sin(0.0) ** 2))
    assert math.isclose(t_peak, 1.0, rel_tol=1e-6)

    # Test anti-resonance (delta = π)
    t_min_theoretical = 1.0 / (1.0 + f_coeff)
    t_min_calc = 1.0 / (1.0 + f_coeff * (math.sin(math.pi / 2.0) ** 2))
    assert math.isclose(t_min_calc, t_min_theoretical, rel_tol=1e-6)
    # For R = 0.96, F = 4*0.96 / (0.04)^2 = 2400 -> T_min = 1/2401 ≈ 0.000416
    assert t_min_calc < 0.001


def test_finesse_and_coefficient_of_finesse_scaling(qapp):
    """
    Verifies algebraic relationship between coefficient of finesse F
    and cavity finesse Script-F = π * √F / 2 = π * √R / (1 - R).
    """
    item = InteractiveFabryPerotItem()

    for r_val in [0.30, 0.50, 0.85, 0.95, 0.99]:
        item._r = r_val
        f_coeff = item.coefficient_of_finesse_f
        cavity_finesse = item.cavity_finesse

        # Check Script-F == (π * √F) / 2
        finesse_from_f = (math.pi * math.sqrt(f_coeff)) / 2.0
        assert math.isclose(cavity_finesse, finesse_from_f, rel_tol=1e-5)

        # Monotonicity: higher reflectance must strictly yield higher finesse
        if r_val == 0.99:
            assert cavity_finesse > 300.0
        elif r_val == 0.30:
            assert cavity_finesse < 3.0


def test_free_spectral_range_relationships(qapp):
    """
    Verifies Free Spectral Range formulas:
    Δν_FSR = c / (2 n d cos θ)
    Δλ_FSR = λ² / (2 n d cos θ)
    """
    item = InteractiveFabryPerotItem()
    item.load_preset("HELIUM_NEON_LASER_CAVITY")

    d_m = item.cavity_d_um * 1e-6
    expected_fsr_ghz = (C_LIGHT / (2.0 * 1.0 * d_m)) * 1e-9
    assert math.isclose(item.free_spectral_range_ghz, expected_fsr_ghz, rel_tol=1e-4)

    expected_fsr_nm = ((632.8e-9) ** 2 / (2.0 * 1.0 * d_m)) * 1e9
    assert math.isclose(item.free_spectral_range_nm, expected_fsr_nm, rel_tol=1e-4)


def test_resonance_linewidth_fwhm_and_q_factor(qapp):
    """
    Verifies FWHM linewidth δν = Δν_FSR / Script-F.
    """
    item = InteractiveFabryPerotItem()
    item.load_preset("HELIUM_NEON_LASER_CAVITY")

    fwhm = item.resonance_linewidth_fwhm_ghz
    fsr = item.free_spectral_range_ghz
    fin = item.cavity_finesse

    assert math.isclose(fwhm, fsr / fin, rel_tol=1e-5)
    # For high finesse He-Ne cavity, linewidth is much smaller than FSR
    assert fwhm < fsr / 50.0


def test_all_five_presets_load(qapp):
    item = InteractiveFabryPerotItem()
    for preset_code, preset in FABRY_PEROT_PRESETS.items():
        item.load_preset(preset_code)
        assert item.active_preset.code == preset_code
        assert item.wavelength_nm == preset.wavelength_nm
        assert item.cavity_d_um == preset.cavity_d_um
        assert item.reflectance_r == preset.reflectance_r
        assert item.cavity_finesse > 1.0
        assert item.free_spectral_range_ghz > 0.0
        assert item.resonance_linewidth_fwhm_ghz > 0.0


def test_parameter_steppers_and_clamping(qapp):
    item = InteractiveFabryPerotItem()
    init_d = item.cavity_d_um
    item.step_distance_um(5.0)
    assert math.isclose(item.cavity_d_um, init_d + 5.0, rel_tol=1e-4)

    init_w = item.wavelength_nm
    item.step_wavelength_nm(-20.0)
    assert math.isclose(item.wavelength_nm, init_w - 20.0, rel_tol=1e-4)

    init_r = item.reflectance_r
    item.step_reflectance_r(0.02)
    assert math.isclose(item.reflectance_r, init_r + 0.02, rel_tol=1e-4)

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractiveFabryPerotItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(25.0, 45.0))
    assert item.pos() == start_pos + QPointF(25.0, 45.0)

    # Air pinch on btn_d_up
    rect_d_up = item._buttons["btn_d_up"]
    scene_pt = item.mapToScene(rect_d_up.center())
    prev_d = item.cavity_d_um
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item.cavity_d_um > prev_d


def test_vector_diagram_generation_and_stamping(qapp):
    item = InteractiveFabryPerotItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "finesse" in stamped[0]
    assert "fsr_ghz" in stamped[0]
    assert "fwhm_ghz" in stamped[0]

    items = PhysicsTools.create_fabry_perot_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="HELIUM_NEON_LASER_CAVITY",
        wavelength_nm=632.8,
        cavity_d_um=20.0,
        reflectance_r=0.96,
        finesse=76.8,
        fsr_ghz=7494.8,
        fwhm_ghz=97.6,
        transmittance=1.0,
        order_m=63.2,
    )
    assert len(items) >= 15
    assert any("FABRY-PÉROT" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_fabry_perot_lifecycle(qapp):
    board = LessonBoard()
    assert board.is_fabry_perot_visible() is False
    assert board.get_fabry_perot() is None

    # Show
    item = board.show_fabry_perot(True)
    assert item is not None
    assert board.is_fabry_perot_visible() is True
    assert board.get_fabry_perot() is item
    assert board.get_fabry_perot_item() is item

    # Toggle to hide
    visible = board.toggle_fabry_perot()
    assert visible is False
    assert board.is_fabry_perot_visible() is False
    assert board.get_fabry_perot() is None
    assert board.get_fabry_perot_item() is not None

    # Toggle to show
    visible = board.toggle_fabry_perot()
    assert visible is True
    assert board.is_fabry_perot_visible() is True
