"""
AirTeach AI — Unit Tests for Interactive Nuclear Physics & Quantum Mechanics:
Radiation Penetration, Exponential Attenuation Law & Half-Value Layer (HVL) Simulator (Milestone 148 / Task I.135).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_radiation_shielding import (
    InteractiveRadiationShieldingItem,
    RadiationPreset,
    RADIATION_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_radiation_presets_and_physical_parameters():
    """Verifies that all 5 curriculum presets are properly defined with authentic nuclear data."""
    expected_codes = [
        "COBALT_60_GAMMA_LEAD_SHIELDING",
        "CESIUM_137_GAMMA_CONCRETE_BARRIER",
        "STRONTIUM_90_BETA_ALUMINUM_ABSORBER",
        "AMERICIUM_241_ALPHA_PAPER_AIR",
        "IODINE_131_THYROID_WATER_TISSUE",
    ]
    for code in expected_codes:
        assert code in RADIATION_PRESETS
        preset = RADIATION_PRESETS[code]
        assert preset.energy_MeV > 0.0
        assert preset.density_g_cm3 > 0.0
        assert preset.linear_mu_cm_inv > 0.0
        assert preset.default_thickness_cm >= 0.0
        assert preset.default_intensity_cps > 0.0
        assert preset.particle_symbol in ("α", "β⁻", "γ")
        assert preset.accent_color.isValid()


def test_exponential_attenuation_and_hvl_exact_math(qapp):
    """
    Verifies Beer-Lambert-Bouguer exponential attenuation:
    I(x) = I_0 * exp(-mu * x)
    HVL = ln(2) / mu
    TVL = ln(10) / mu
    """
    item = InteractiveRadiationShieldingItem()
    item.set_preset("COBALT_60_GAMMA_LEAD_SHIELDING")

    mu = item.linear_mu  # 0.655 cm^-1
    expected_hvl = math.log(2.0) / mu
    expected_tvl = math.log(10.0) / mu

    assert math.isclose(item.half_value_layer_cm, expected_hvl, rel_tol=1e-4)
    assert math.isclose(item.tenth_value_layer_cm, expected_tvl, rel_tol=1e-4)

    # 1. Exactly 1 HVL thickness -> 50% transmitted
    item.set_thickness_cm(expected_hvl)
    assert math.isclose(item.transmission_fraction, 0.50, rel_tol=1e-3)
    assert math.isclose(item.transmitted_intensity_cps, item._i0_cps * 0.50, rel_tol=1e-3)

    # 2. Exactly 2 HVL thickness -> 25% transmitted
    item.set_thickness_cm(2.0 * expected_hvl)
    assert math.isclose(item.transmission_fraction, 0.25, rel_tol=1e-3)

    # 3. Exactly 1 TVL thickness -> 10% transmitted
    item.set_thickness_cm(expected_tvl)
    assert math.isclose(item.transmission_fraction, 0.10, rel_tol=1e-3)


def test_alpha_radiation_cutoff_in_paper(qapp):
    """
    Verifies Bragg peak complete absorption of Alpha radiation in paper / air.
    """
    item = InteractiveRadiationShieldingItem()
    item.set_preset("AMERICIUM_241_ALPHA_PAPER_AIR")

    # With zero barrier (x=0)
    item.set_thickness_cm(0.0)
    assert math.isclose(item.transmission_percent, 100.0, rel_tol=1e-3)

    # With paper barrier (x = 0.010 cm = 0.1 mm)
    item.set_thickness_cm(0.010)
    assert math.isclose(item.transmission_fraction, 0.0, abs_tol=1e-5)
    assert math.isclose(item.transmitted_intensity_cps, 0.0, abs_tol=1e-5)
    assert math.isclose(item.absorption_percent, 100.0, abs_tol=1e-5)


def test_thickness_and_intensity_adjustments(qapp):
    """Verifies parameter updates and bounds clamping."""
    item = InteractiveRadiationShieldingItem()
    item.set_preset("CESIUM_137_GAMMA_CONCRETE_BARRIER")

    # Change intensity
    item.set_intensity_cps(2000.0)
    assert math.isclose(item._i0_cps, 2000.0, rel_tol=1e-4)

    # Change thickness
    item.set_thickness_cm(5.0)
    assert math.isclose(item._thickness_cm, 5.0, rel_tol=1e-4)

    # Reset
    item.reset_defaults()
    assert math.isclose(item._thickness_cm, item._preset.default_thickness_cm, rel_tol=1e-4)
    assert math.isclose(item._i0_cps, item._preset.default_intensity_cps, rel_tol=1e-4)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button handlers and interactive action triggers."""
    item = InteractiveRadiationShieldingItem()
    item.set_preset("COBALT_60_GAMMA_LEAD_SHIELDING")

    # Test thickness increments
    orig_x = item._thickness_cm
    item._handle_button_click("x_inc")
    assert math.isclose(item._thickness_cm, orig_x + 0.50, abs_tol=1e-3)

    # Test removing shield (x=0)
    item._handle_button_click("x_zero")
    assert math.isclose(item._thickness_cm, 0.0, abs_tol=1e-5)
    assert math.isclose(item.transmission_percent, 100.0, rel_tol=1e-3)

    # Test add +1 HVL
    item._handle_button_click("add_hvl")
    assert math.isclose(item._thickness_cm, item.half_value_layer_cm, abs_tol=1e-3)
    assert math.isclose(item.transmission_percent, 50.0, rel_tol=1e-2)

    # Test preset cycler
    item._handle_button_click("preset_next")
    assert item._active_preset_code != "COBALT_60_GAMMA_LEAD_SHIELDING"


def test_stem_tools_radiation_diagram():
    """Verifies vector diagram creation in PhysicsTools for radiation shielding."""
    items = PhysicsTools.create_radiation_shielding_diagram(
        center=(400.0, 300.0),
        preset_code="COBALT_60_GAMMA_LEAD_SHIELDING",
        isotope_name="Cobalt-60 (⁶⁰Co, γ = 1.25 MeV)",
        radiation_type="gamma",
        absorber_material="Chì (Lead - Pb)",
        thickness_cm=2.50,
        i0_cps=1000.0,
        i_trans_cps=194.5,
        transmission_pct=19.45,
        hvl_cm=1.058,
        tvl_cm=3.515,
        linear_mu_cm_inv=0.655,
    )
    assert len(items) >= 15
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsLineItem" in types


def test_lesson_board_radiation_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_radiation_shielding_visible()

    # Show
    item = board.show_radiation_shielding(True)
    assert item is not None
    assert board.is_radiation_shielding_visible()
    assert board.get_radiation_shielding() is item

    # Toggle off
    state = board.toggle_radiation_shielding()
    assert not state
    assert not board.is_radiation_shielding_visible()
    assert board.get_radiation_shielding() is None

    # Toggle on
    state = board.toggle_radiation_shielding()
    assert state
    assert board.is_radiation_shielding_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_radiation_shielding({
        "preset_code": "COBALT_60_GAMMA_LEAD_SHIELDING",
        "isotope_name": "⁶⁰Co",
        "radiation_type": "gamma",
        "absorber_material": "Lead",
        "thickness_cm": 2.50,
        "i0_cps": 1000.0,
        "i_trans_cps": 194.5,
        "transmission_pct": 19.45,
        "hvl_cm": 1.058,
        "tvl_cm": 3.515,
        "linear_mu_cm_inv": 0.655,
    })
    after_count = len(board.items())
    assert after_count > initial_count + 10


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveRadiationShieldingItem()
    image = QImage(700, 430, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not image.isNull()
