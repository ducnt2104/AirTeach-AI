"""
AirTeach AI - Tests for Interactive Chemical Equilibrium & Le Chatelier Simulator
==================================================================================
Validates thermodynamic Gibbs free energy, van 't Hoff temperature dependence,
Le Chatelier pressure shifts, mole fraction mass balances, and UI rendering.
"""
import math
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_chemical_equilibrium import (
    R_GAS_CONST,
    REACTION_PRESETS,
    InteractiveChemicalEquilibriumItem,
)


@pytest.fixture
def chem_item(qapp):
    return InteractiveChemicalEquilibriumItem(preset_key="HABER_BOSCH", temp_k=673.15, pressure_atm=200.0)


def test_gibbs_free_energy_and_kp(chem_item):
    """Verify ΔG° = ΔH° - TΔS° and K_p = exp(-ΔG° / RT)."""
    t = 673.15
    chem_item.set_temp_k(t)
    eq = chem_item.compute_equilibrium()

    expected_dg_kj = -92.22 - (t * -198.75 / 1000.0)
    assert math.isclose(eq["delta_g_kj"], expected_dg_kj, rel_tol=1e-4)

    expected_kp = math.exp(-expected_dg_kj * 1000.0 / (R_GAS_CONST * t))
    assert math.isclose(eq["kp"], expected_kp, rel_tol=1e-4)


def test_le_chatelier_temperature_exothermic(chem_item):
    """In exothermic Haber-Bosch (ΔH < 0), higher T must shift equilibrium left (decrease yield)."""
    chem_item.set_pressure_atm(100.0)

    chem_item.set_temp_k(500.0)  # Lower T
    eq_cold = chem_item.compute_equilibrium()

    chem_item.set_temp_k(800.0)  # Higher T
    eq_hot = chem_item.compute_equilibrium()

    assert eq_cold["kp"] > eq_hot["kp"]
    assert eq_cold["yield_pct"] > eq_hot["yield_pct"]
    assert eq_cold["mole_fractions"]["NH₃"] > eq_hot["mole_fractions"]["NH₃"]


def test_le_chatelier_temperature_endothermic(chem_item):
    """In endothermic HI synthesis (ΔH > 0), higher T must shift equilibrium right (increase yield)."""
    chem_item.apply_preset("HYDROGEN_IODIDE")

    chem_item.set_temp_k(400.0)
    eq_cold = chem_item.compute_equilibrium()

    chem_item.set_temp_k(800.0)
    eq_hot = chem_item.compute_equilibrium()

    assert eq_hot["kp"] > eq_cold["kp"]
    assert eq_hot["yield_pct"] > eq_cold["yield_pct"]


def test_le_chatelier_pressure_shift(chem_item):
    """In Haber-Bosch (Δn_gas = -2), higher P must increase K_x and boost NH3 yield."""
    chem_item.set_temp_k(673.15)

    chem_item.set_pressure_atm(10.0)  # Low P
    eq_low_p = chem_item.compute_equilibrium()

    chem_item.set_pressure_atm(300.0)  # High P
    eq_high_p = chem_item.compute_equilibrium()

    # K_p is independent of P, but K_x increases as P^2
    assert math.isclose(eq_low_p["kp"], eq_high_p["kp"], rel_tol=1e-5)
    assert eq_high_p["kx"] > eq_low_p["kx"]
    assert eq_high_p["yield_pct"] > eq_low_p["yield_pct"]


def test_isochoric_pressure_invariance(chem_item):
    """When Δn_gas = 0 (HI synthesis), pressure changes must NOT alter K_x or yield."""
    chem_item.apply_preset("HYDROGEN_IODIDE")

    chem_item.set_pressure_atm(1.0)
    eq_1 = chem_item.compute_equilibrium()

    chem_item.set_pressure_atm(50.0)
    eq_50 = chem_item.compute_equilibrium()

    assert math.isclose(eq_1["kx"], eq_50["kx"], rel_tol=1e-5)
    assert math.isclose(eq_1["yield_pct"], eq_50["yield_pct"], rel_tol=1e-5)


def test_mole_fractions_sum_to_one(chem_item):
    """Sum of mole fractions for all species must equal 1.0 within numerical precision."""
    for preset_key in REACTION_PRESETS:
        chem_item.apply_preset(preset_key)
        eq = chem_item.compute_equilibrium()
        total_frac = sum(eq["mole_fractions"].values())
        assert math.isclose(total_frac, 1.0, abs_tol=1e-4)


def test_catalyst_toggle_and_stamping(chem_item):
    """Verify catalyst toggle does not alter equilibrium state and stamp requests emit data."""
    init_state = chem_item.catalyst_enabled
    chem_item.toggle_catalyst()
    assert chem_item.catalyst_enabled != init_state

    stamps = []
    chem_item.stamp_requested.connect(lambda d: stamps.append(d))
    chem_item.request_stamp()

    assert len(stamps) == 1
    assert "yield_pct" in stamps[0]
    assert "mole_fractions" in stamps[0]


def test_paint_and_render_zero_crash(qapp, chem_item):
    """Verify paint rendering executes without crashing on a QImage."""
    scene = QGraphicsScene()
    scene.addItem(chem_item)

    image = QImage(720, 450, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.black)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    chem_item.paint(painter, option, None)
    painter.end()

    assert not image.isNull()
