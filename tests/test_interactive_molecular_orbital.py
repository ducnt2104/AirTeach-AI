"""
AirTeach AI — Unit Tests for Interactive Quantum Chemistry & Molecular Orbital Theory:
Atomic Orbitals, Hybridization (sp, sp2, sp3) & Molecular Orbital LCAO Simulator (Milestone 150 / Task I.137).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_molecular_orbital import (
    InteractiveMolecularOrbitalItem,
    OrbitalPreset,
    ORBITAL_PRESETS,
    BOHR_RADIUS_ANGSTROM,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_orbital_presets_and_quantum_parameters():
    """Verifies that all 5 curriculum presets are properly defined with quantum chemical data."""
    expected_codes = [
        "HYDROGEN_MOLECULE_H2_SIGMA_BOND",
        "CARBON_SP3_METHANE_CH4_TETRAHEDRAL",
        "ETHYLENE_SP2_PI_BONDING_C2H4",
        "HELIUM_DIMER_HE2_UNSTABLE",
        "PARAMAGNETIC_OXYGEN_O2_DIRADICAL",
    ]
    for code in expected_codes:
        assert code in ORBITAL_PRESETS
        preset = ORBITAL_PRESETS[code]
        assert preset.default_distance_angstrom > 0.0
        assert preset.n_bonding_electrons >= 0
        assert preset.n_antibonding_electrons >= 0
        assert preset.bond_order >= 0.0
        assert preset.accent_color.isValid()
        assert len(preset.hybridization_str) > 0


def test_overlap_integral_analytical_function(qapp):
    """
    Verifies analytical Overlap Integral S(rho) = exp(-rho) * (1 + rho + rho^2 / 3).
    - At equilibrium distance of H2 (R = 0.74 Å), S is approximately 0.753.
    - S strictly decreases as internuclear distance increases.
    """
    item = InteractiveMolecularOrbitalItem()
    item.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")
    item.set_distance_angstrom(0.74)

    rho = 0.74 / BOHR_RADIUS_ANGSTROM
    expected_s = math.exp(-rho) * (1.0 + rho + (rho * rho) / 3.0)

    assert math.isclose(item.overlap_integral_S, expected_s, rel_tol=1e-4)
    assert 0.70 < item.overlap_integral_S < 0.80

    # Test monotonic decrease
    item.set_distance_angstrom(1.50)
    s_far = item.overlap_integral_S
    assert s_far < expected_s

    item.set_distance_angstrom(3.00)
    s_very_far = item.overlap_integral_S
    assert s_very_far < s_far


def test_bond_order_and_stability(qapp):
    """
    Verifies Bond Order BO = (N_bonding - N_antibonding) / 2:
    - H2 single bond: BO = 1.0
    - He2 non-existent: BO = 0.0
    - O2 double bond: BO = 2.0
    """
    item = InteractiveMolecularOrbitalItem()

    # H2
    item.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")
    assert math.isclose(item.bond_order, 1.0)

    # He2
    item.set_preset("HELIUM_DIMER_HE2_UNSTABLE")
    assert math.isclose(item.bond_order, 0.0)

    # O2
    item.set_preset("PARAMAGNETIC_OXYGEN_O2_DIRADICAL")
    assert math.isclose(item.bond_order, 2.0)


def test_bonding_vs_antibonding_toggle(qapp):
    """Verifies state toggling between Bonding (sigma/pi) and Antibonding (sigma*/pi*)."""
    item = InteractiveMolecularOrbitalItem()
    item.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")

    assert not item._is_antibonding_view

    # Toggle to antibonding
    item.toggle_bonding_mode()
    assert item._is_antibonding_view

    # Toggle back to bonding
    item.toggle_bonding_mode()
    assert not item._is_antibonding_view


def test_distance_adjustments_and_bounds(qapp):
    """Verifies internuclear distance changes and clamping bounds."""
    item = InteractiveMolecularOrbitalItem()
    item.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")

    # Change distance
    item.set_distance_angstrom(1.20)
    assert math.isclose(item._distance_angstrom, 1.20, abs_tol=1e-3)

    # Lower bound clamping
    item.set_distance_angstrom(0.10)
    assert item._distance_angstrom >= 0.40

    # Upper bound clamping
    item.set_distance_angstrom(10.0)
    assert item._distance_angstrom <= 4.0

    # Reset
    item.reset_defaults()
    assert math.isclose(item._distance_angstrom, item._preset.default_distance_angstrom, abs_tol=1e-3)


def test_interactive_controls_and_buttons(qapp):
    """Verifies button handlers and interactive action triggers."""
    item = InteractiveMolecularOrbitalItem()
    item.set_preset("HYDROGEN_MOLECULE_H2_SIGMA_BOND")

    # Step distance
    orig_r = item._distance_angstrom
    item._handle_button_click("r_inc")
    assert math.isclose(item._distance_angstrom, orig_r + 0.10, abs_tol=1e-3)

    item._handle_button_click("r_eq")
    assert math.isclose(item._distance_angstrom, item._preset.default_distance_angstrom, abs_tol=1e-3)

    # Mode toggle
    item._handle_button_click("mode_toggle")
    assert item._is_antibonding_view

    # Preset switcher
    item._handle_button_click("set_sp3")
    assert item._active_preset_code == "CARBON_SP3_METHANE_CH4_TETRAHEDRAL"


def test_stem_tools_mo_diagram():
    """Verifies vector diagram creation in ChemistryTools for Molecular Orbital."""
    items = ChemistryTools.create_molecular_orbital_diagram(
        center=(400.0, 300.0),
        preset_code="HYDROGEN_MOLECULE_H2_SIGMA_BOND",
        molecule_name="Hydrogen Molecule (H₂)",
        orbital_type="mo_sigma",
        distance_angstrom=0.74,
        overlap_integral_S=0.753,
        bond_order=1.0,
        is_antibonding=False,
        hybridization_str="1s + 1s (Không lai hóa)",
    )
    assert len(items) >= 15
    types = [type(itm).__name__ for itm in items]
    assert "QGraphicsRectItem" in types
    assert "QGraphicsTextItem" in types
    assert "QGraphicsLineItem" in types
    assert "QGraphicsEllipseItem" in types


def test_lesson_board_mo_lifecycle(qapp):
    """Verifies integration, toggling, and stamping on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_molecular_orbital_visible()

    # Show
    item = board.show_molecular_orbital(True)
    assert item is not None
    assert board.is_molecular_orbital_visible()
    assert board.get_molecular_orbital() is item

    # Toggle off
    state = board.toggle_molecular_orbital()
    assert not state
    assert not board.is_molecular_orbital_visible()
    assert board.get_molecular_orbital() is None

    # Toggle on
    state = board.toggle_molecular_orbital()
    assert state
    assert board.is_molecular_orbital_visible()

    # Test stamp
    initial_count = len(board.items())
    board._on_stamp_molecular_orbital({
        "preset_code": "HYDROGEN_MOLECULE_H2_SIGMA_BOND",
        "molecule_name": "H₂",
        "orbital_type": "mo_sigma",
        "distance_angstrom": 0.74,
        "overlap_integral_S": 0.753,
        "bond_order": 1.0,
        "is_antibonding": False,
        "hybridization_str": "1s + 1s",
    })
    after_count = len(board.items())
    assert after_count > initial_count + 10


def test_paint_rendering_zero_crash(qapp):
    """Ensures QPainter rendering executes without raising any exception."""
    item = InteractiveMolecularOrbitalItem()
    image = QImage(700, 430, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not image.isNull()
