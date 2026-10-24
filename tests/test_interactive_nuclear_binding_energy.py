"""
AirTeach AI - Tests for Interactive Nuclear Binding Energy & SEMF Simulator
===========================================================================
Validates Bethe-Weizsäcker Semi-Empirical Mass Formula (SEMF), liquid-drop model,
mass defect, binding energy per nucleon, and UI lifecycle.
"""
import math
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_nuclear_binding_energy import (
    A_V,
    A_S,
    A_C,
    A_A,
    A_P,
    AMU_TO_MEV,
    NUCLIDE_PRESETS,
    InteractiveNuclearBindingEnergyItem,
)


@pytest.fixture
def binding_item(qapp):
    return InteractiveNuclearBindingEnergyItem(A=56, Z=26)


def test_iron_56_binding_energy_near_peak(binding_item):
    """Iron-56 (A=56, Z=26) should have BE/A around 8.79 MeV/nucleon."""
    binding_item.set_A(56)
    binding_item.set_Z(26)
    phys = binding_item.compute_physics()

    assert 8.5 < phys["be_per_nucleon_mev"] < 9.0
    assert phys["total_be_mev"] > 450.0  # ~490 MeV
    assert phys["mass_defect_u"] > 0.5  # ~0.528 u


def test_semf_individual_terms_signs(binding_item):
    """Validate sign conventions of the 5 SEMF terms."""
    phys = binding_item.calculate_semf(A=56, Z=26)

    assert phys["e_vol"] > 0.0  # Volume term binds nucleons (+)
    assert phys["e_surf"] < 0.0  # Surface tension reduces binding (-)
    assert phys["e_coul"] < 0.0  # Proton repulsion reduces binding (-)
    assert phys["e_asym"] <= 0.0  # Neutron-proton asymmetry penalty (-)
    assert phys["e_pair"] > 0.0  # Even-even nucleus adds pairing stability (+)


def test_pairing_term_rules():
    """Verify pairing term δ(A, Z) follows even-even (+), odd-A (0), odd-odd (-) rules."""
    # Even-Even: A=12, Z=6, N=6
    ee = InteractiveNuclearBindingEnergyItem.calculate_semf(12, 6)
    assert ee["e_pair"] > 0.0

    # Odd-A: A=13, Z=6, N=7
    odd = InteractiveNuclearBindingEnergyItem.calculate_semf(13, 6)
    assert odd["e_pair"] == 0.0

    # Odd-Odd: A=14, Z=7, N=7
    oo = InteractiveNuclearBindingEnergyItem.calculate_semf(14, 7)
    assert oo["e_pair"] < 0.0


def test_mass_defect_relation(binding_item):
    """Mass defect must satisfy Δm = B(A, Z) / (931.494 MeV/u)."""
    for a_val, z_val in [(4, 2), (12, 6), (56, 26), (235, 92)]:
        binding_item.set_A(a_val)
        binding_item.set_Z(z_val)
        phys = binding_item.compute_physics()
        expected_dm = phys["total_be_mev"] / AMU_TO_MEV
        assert math.isclose(phys["mass_defect_u"], expected_dm, rel_tol=1e-5)


def test_bounds_enforcement(binding_item):
    """Z must never exceed A, and A must be clamped within physical bounds."""
    binding_item.set_A(20)
    binding_item.set_Z(50)  # Try setting Z > A
    assert binding_item.Z <= binding_item.A

    binding_item.set_A(300)  # Exceeds max 260
    assert binding_item.A <= 260


def test_presets_loading(binding_item):
    """Test applying all nuclide presets."""
    for key, preset in NUCLIDE_PRESETS.items():
        binding_item.apply_preset(key)
        assert binding_item.A == preset.A
        assert binding_item.Z == preset.Z


def test_paint_and_render_zero_crash(qapp, binding_item):
    """Verify item renders cleanly to QImage without exception."""
    scene = QGraphicsScene()
    scene.addItem(binding_item)

    image = QImage(700, 450, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        opt = QStyleOptionGraphicsItem()
        binding_item.paint(painter, opt, None)
    finally:
        painter.end()

    assert not image.isNull()


def test_stamp_request_signal(binding_item):
    """Verify request_stamp emits valid dictionary payload."""
    emitted_data = {}

    def on_stamp(data):
        nonlocal emitted_data
        emitted_data = data

    binding_item.stamp_requested.connect(on_stamp)
    binding_item.request_stamp()

    assert "A" in emitted_data
    assert "Z" in emitted_data
    assert "total_be_mev" in emitted_data
    assert emitted_data["A"] == binding_item.A
