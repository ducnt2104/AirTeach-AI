"""
AirTeach AI — Unit Tests for Interactive Thermodynamics & Statistical Mechanics:
Maxwell-Boltzmann Molecular Speed Distribution, Mean Free Path & Kinetic Theory (Milestone 149 / Task I.136).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_maxwell_boltzmann import (
    InteractiveMaxwellBoltzmannItem,
    GasPreset,
    GAS_PRESETS,
    R_GAS_CONST,
    K_BOLTZMANN,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gas_presets_and_physical_parameters():
    """Verifies that all 5 curriculum presets are properly defined with authentic thermodynamic constants."""
    expected_codes = [
        "HELIUM_GAS_ROOM_TEMP_300K",
        "NITROGEN_AIR_EARTH_ATMOSPHERE",
        "HYDROGEN_HIGH_TEMP_FURNACE_1000K",
        "XENON_HEAVY_NOBLE_GAS_COLD",
        "OXYGEN_SURFACE_VENUS_737K",
    ]
    for code in expected_codes:
        assert code in GAS_PRESETS
        preset = GAS_PRESETS[code]
        assert preset.molar_mass_kg_mol > 0.0
        assert preset.molecular_diameter_m > 0.0
        assert preset.default_temp_K > 0.0
        assert preset.default_pressure_atm > 0.0
        assert preset.accent_color.isValid()
        assert len(preset.chemical_formula) > 0


def test_characteristic_speeds_ordering_invariant(qapp):
    """
    Verifies exact analytical equations for characteristic speeds:
    v_p = sqrt(2 * R * T / M)
    <v> = sqrt(8 * R * T / (pi * M))
    v_rms = sqrt(3 * R * T / M)
    and fundamental invariant: v_p < <v> < v_rms (1 : 1.128 : 1.225).
    """
    item = InteractiveMaxwellBoltzmannItem()
    item.set_preset("HELIUM_GAS_ROOM_TEMP_300K")

    t = 300.0
    m = 0.0040026

    exp_vp = math.sqrt(2.0 * R_GAS_CONST * t / m)      # ~1115.8 m/s
    exp_vavg = math.sqrt(8.0 * R_GAS_CONST * t / (math.pi * m)) # ~1259.0 m/s
    exp_vrms = math.sqrt(3.0 * R_GAS_CONST * t / m)    # ~1366.6 m/s

    assert math.isclose(item.most_probable_speed_m_s, exp_vp, rel_tol=1e-4)
    assert math.isclose(item.average_speed_m_s, exp_vavg, rel_tol=1e-4)
    assert math.isclose(item.rms_speed_m_s, exp_vrms, rel_tol=1e-4)

    # Invariant ordering
    assert item.most_probable_speed_m_s < item.average_speed_m_s < item.rms_speed_m_s

    # Exact mathematical ratios
    assert math.isclose(item.average_speed_m_s / item.most_probable_speed_m_s, math.sqrt(4.0 / math.pi), rel_tol=1e-4)
    assert math.isclose(item.rms_speed_m_s / item.most_probable_speed_m_s, math.sqrt(1.5), rel_tol=1e-4)


def test_maxwell_boltzmann_pdf_integral_and_peak(qapp):
    """
    Verifies that the PDF maximum strictly occurs at v = v_p
    and that the integral of f(v) over all speeds evaluates to 1.0.
    """
    item = InteractiveMaxwellBoltzmannItem()
    item.set_preset("NITROGEN_AIR_EARTH_ATMOSPHERE")

    vp = item.most_probable_speed_m_s
    f_vp = item.pdf_f_v(vp)

    # Test peak behavior: points to the left and right must have strictly lower probability
    assert f_vp > item.pdf_f_v(vp - 40.0)
    assert f_vp > item.pdf_f_v(vp + 40.0)

    # Test analytical peak formula: f(vp) = 4 / (e * sqrt(pi) * vp)
    analytical_f_peak = 4.0 / (math.e * math.sqrt(math.pi) * vp)
    assert math.isclose(f_vp, analytical_f_peak, rel_tol=1e-3)

    # Numerical integration (trapezoidal rule)
    max_v = item.rms_speed_m_s * 4.0
    n_bins = 200
    dv = max_v / n_bins
    total_prob = sum(0.5 * (item.pdf_f_v(i * dv) + item.pdf_f_v((i + 1) * dv)) * dv for i in range(n_bins))
    assert math.isclose(total_prob, 1.0, rel_tol=0.01)


def test_temperature_and_mass_scaling(qapp):
    """
    Verifies physical scaling laws:
    v_rms proportional to sqrt(T)
    v_rms inversely proportional to sqrt(M)
    """
    item = InteractiveMaxwellBoltzmannItem()
    item.set_preset("HELIUM_GAS_ROOM_TEMP_300K")
    item.set_temperature_K(300.0)
    v_300 = item.rms_speed_m_s

    # Quadruple T -> double speeds
    item.set_temperature_K(1200.0)
    v_1200 = item.rms_speed_m_s
    assert math.isclose(v_1200, 2.0 * v_300, rel_tol=1e-3)

    # Hydrogen (lightest) vs Xenon (heaviest) at same temperature
    item.set_temperature_K(300.0)
    item.set_preset("HYDROGEN_HIGH_TEMP_FURNACE_1000K")
    item.set_temperature_K(300.0)
    v_h2 = item.rms_speed_m_s

    item.set_preset("XENON_HEAVY_NOBLE_GAS_COLD")
    item.set_temperature_K(300.0)
    v_xe = item.rms_speed_m_s

    # Ratio should match sqrt(M_xe / M_h2) = sqrt(0.131293 / 0.002016) ~ 8.07
    ratio = v_h2 / v_xe
    expected_ratio = math.sqrt(0.131293 / 0.002016)
    assert math.isclose(ratio, expected_ratio, rel_tol=1e-3)


def test_mean_free_path_and_pressure_inversion(qapp):
    """
    Verifies mean free path lambda = (k_B * T) / (sqrt(2) * pi * d^2 * P).
    Doubling pressure cuts mean free path in half.
    """
    item = InteractiveMaxwellBoltzmannItem()
    item.set_preset("HELIUM_GAS_ROOM_TEMP_300K")
    item.set_pressure_atm(1.0)
    mfp_1atm = item.mean_free_path_nm
    assert mfp_1atm > 50.0  # Around 197 nm for helium at 1 atm

    # Double pressure to 2.0 atm -> mfp should halve
    item.set_pressure_atm(2.0)
    mfp_2atm = item.mean_free_path_nm
    assert math.isclose(mfp_2atm, 0.5 * mfp_1atm, rel_tol=1e-3)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button handlers and interactive action triggers."""
    item = InteractiveMaxwellBoltzmannItem()
    item.set_preset("HELIUM_GAS_ROOM_TEMP_300K")

    # Temperature adjustment
    orig_t = item._temperature_K
    item._handle_button_click("t_inc")
    assert math.isclose(item._temperature_K, orig_t + 50.0, abs_tol=1e-3)

    item._handle_button_click("t_room")
    assert math.isclose(item._temperature_K, 300.0, abs_tol=1e-3)

    item._handle_button_click("t_hot")
    assert math.isclose(item._temperature_K, 1000.0, abs_tol=1e-3)

    # Pressure adjustment
    orig_p = item._pressure_atm
    item._handle_button_click("p_inc")
    assert math.isclose(item._pressure_atm, orig_p + 0.50, abs_tol=1e-3)

    # Preset cycler
    item._handle_button_click("preset_next")
    assert item._active_preset_code != "HELIUM_GAS_ROOM_TEMP_300K"


def test_stem_tools_maxwell_diagram():
    """Verifies vector diagram creation in PhysicsTools for Maxwell-Boltzmann."""
    items = PhysicsTools.create_maxwell_boltzmann_diagram(
        center=(400.0, 300.0),
        preset_code="HELIUM_GAS_ROOM_TEMP_300K",
        gas_name="Helium (He)",
        formula="⁴He",
        temperature_K=300.0,
        pressure_atm=1.0,
        molar_mass_kg_mol=0.0040026,
        vp_m_s=1115.8,
        v_avg_m_s=1259.0,
        v_rms_m_s=1366.6,
        mean_free_path_nm=197.5,
        collision_freq_GHz=6.37,
    )
    assert len(items) >= 15
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsLineItem" in types
    assert "QGraphicsEllipseItem" in types


def test_lesson_board_maxwell_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_maxwell_boltzmann_visible()

    # Show
    item = board.show_maxwell_boltzmann(True)
    assert item is not None
    assert board.is_maxwell_boltzmann_visible()
    assert board.get_maxwell_boltzmann() is item

    # Toggle off
    state = board.toggle_maxwell_boltzmann()
    assert not state
    assert not board.is_maxwell_boltzmann_visible()
    assert board.get_maxwell_boltzmann() is None

    # Toggle on
    state = board.toggle_maxwell_boltzmann()
    assert state
    assert board.is_maxwell_boltzmann_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_maxwell_boltzmann({
        "preset_code": "HELIUM_GAS_ROOM_TEMP_300K",
        "gas_name": "Helium",
        "formula": "⁴He",
        "temperature_K": 300.0,
        "pressure_atm": 1.0,
        "molar_mass_kg_mol": 0.0040026,
        "vp_m_s": 1115.8,
        "v_avg_m_s": 1259.0,
        "v_rms_m_s": 1366.6,
        "mean_free_path_nm": 197.5,
        "collision_freq_GHz": 6.37,
    })
    after_count = len(board.items())
    assert after_count > initial_count + 10


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveMaxwellBoltzmannItem()
    image = QImage(700, 430, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not image.isNull()
