"""
AirTeach AI – Test Suite: Interactive Quantum Mechanics & Particle in a Box Simulation
========================================================================================
Verifies:
  - Exact Time-Independent Schrödinger Equation (TISE) physics
  - Quantized energy levels E_n = n^2 * E_1 (Zero-point energy E_1 > 0)
  - Boundary conditions: ψ(0) = 0, ψ(L) = 0
  - Interior nodes count (n - 1)
  - Heisenberg uncertainty principle: Δx · Δp >= hbar / 2
  - Time-dependent quantum superposition beats at Bohr transition frequency
  - Presets, gestures, steppers, and rendering
  - STEMTools vector diagram generation
  - LessonBoard integration & lifecycle
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter, QImage, QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_quantum_well import (
    InteractiveQuantumWellItem,
    PRESETS,
    H_PLANCK,
    H_BAR,
    M_ELECTRON,
    M_PROTON,
    EV_TO_JOULE,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestQuantumPhysicsCalculations:
    """Verifies physical invariants and mathematics without mock data."""

    def test_ground_state_energy_analytic(self, qapp):
        item = InteractiveQuantumWellItem()
        item.set_well_width_nm(1.0)
        item.set_quantum_number(1)

        # L = 1.0 nm = 1e-9 m
        # E_1 = h^2 / (8 * m * L^2)
        expected_joules = (H_PLANCK ** 2) / (8.0 * M_ELECTRON * (1e-9 ** 2))
        expected_ev = expected_joules / EV_TO_JOULE

        assert math.isclose(item.ground_state_energy_joules, expected_joules, rel_tol=1e-5)
        assert math.isclose(item.ground_state_energy_ev, expected_ev, rel_tol=1e-5)
        # For electron in 1 nm well, E_1 is approximately 0.376 eV
        assert 0.370 < item.ground_state_energy_ev < 0.385
        item.cleanup()

    def test_energy_quantization_levels(self, qapp):
        item = InteractiveQuantumWellItem()
        e1 = item.ground_state_energy_ev

        for n in range(1, 6):
            item.set_quantum_number(n)
            assert item.quantum_number_n == n
            assert math.isclose(item.current_energy_ev, (n ** 2) * e1, rel_tol=1e-5)
            assert item.node_count == n - 1
            assert math.isclose(item.de_broglie_wavelength_nm, (2.0 * item.well_width_nm) / n, rel_tol=1e-5)
        item.cleanup()

    def test_boundary_conditions_and_nodes(self, qapp):
        item = InteractiveQuantumWellItem()

        for n in [1, 2, 3, 4]:
            # Boundary conditions: psi(0) = 0 and psi(L) = 0
            assert math.isclose(item.wavefunction_psi(0.0, n), 0.0, abs_tol=1e-12)
            assert math.isclose(item.wavefunction_psi(1.0, n), 0.0, abs_tol=1e-12)

            # Center for n=1 is antinode (maximum)
            if n == 1:
                assert item.wavefunction_psi(0.5, 1) > 0.0
                assert item.probability_density(0.5, 1) > 0.0
            # Center for n=2 is a node (zero)
            elif n == 2:
                assert math.isclose(item.wavefunction_psi(0.5, 2), 0.0, abs_tol=1e-12)
                assert math.isclose(item.probability_density(0.5, 2), 0.0, abs_tol=1e-12)
        item.cleanup()

    def test_heisenberg_uncertainty_bound(self, qapp):
        item = InteractiveQuantumWellItem()

        for n in range(1, 6):
            item.set_quantum_number(n)
            # Heisenberg product ratio: (Δx * Δp) / (hbar / 2) >= 1.0
            ratio = item.heisenberg_uncertainty_product
            assert ratio >= 1.0
            # For n=1, ratio = sqrt(pi^2 / 3 - 2) ≈ 1.136
            if n == 1:
                assert 1.10 < ratio < 1.18
        item.cleanup()

    def test_bohr_transition_frequency_beats(self, qapp):
        item = InteractiveQuantumWellItem()
        # Delta E = E_2 - E_1 = 3 * E_1
        # omega_21 = Delta E / hbar
        omega_fs = item.bohr_transition_freq_rad_per_fs
        assert omega_fs > 0.0

        # Density oscillates between left and right in well
        p_t0_left = item.superposition_density(0.25, 0.0)
        # Half period t = pi / omega
        t_half = math.pi / omega_fs
        p_thalf_left = item.superposition_density(0.25, t_half)
        # At half period, wavefunction has shifted
        assert not math.isclose(p_t0_left, p_thalf_left, rel_tol=1e-2)
        item.cleanup()


class TestQuantumPresetsAndControls:
    """Verifies presets, particle switching, display modes, and air pinches."""

    def test_all_presets_load(self, qapp):
        item = InteractiveQuantumWellItem()

        for code in PRESETS:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.boundingRect().width() == 700.0
            assert item.boundingRect().height() == 440.0
        item.cleanup()

    def test_particle_mass_toggle(self, qapp):
        item = InteractiveQuantumWellItem()
        e_electron = item.ground_state_energy_ev

        item.set_particle_type("proton")
        e_proton = item.ground_state_energy_ev
        # Proton is ~1836 times heavier, so energy is ~1836 times smaller
        ratio = e_electron / e_proton
        assert 1800 < ratio < 1850

        item.set_particle_type("electron")
        assert math.isclose(item.ground_state_energy_ev, e_electron, rel_tol=1e-5)
        item.cleanup()

    def test_display_mode_cycling(self, qapp):
        item = InteractiveQuantumWellItem()
        assert item.display_mode == 0  # Both
        item.cycle_display_mode()
        assert item.display_mode == 1  # psi only
        item.cycle_display_mode()
        assert item.display_mode == 2  # prob only
        item.cycle_display_mode()
        assert item.display_mode == 0
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        item = InteractiveQuantumWellItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        # 1. Preset pinch
        p_rect = item._btn_rects.get("preset_EXCITED_STATES_N2_N3_N4")
        assert p_rect is not None
        center = item.mapToScene(p_rect.center())
        assert item.handle_air_pinch(center) is True
        assert item.active_preset.code == "EXCITED_STATES_N2_N3_N4"

        # 2. Stepper n=4 pinch
        n4_rect = item._btn_rects.get("set_n_4")
        assert n4_rect is not None
        center_n4 = item.mapToScene(n4_rect.center())
        assert item.handle_air_pinch(center_n4) is True
        assert item.quantum_number_n == 4

        # 3. Outside pinch
        outside = item.mapToScene(QPointF(9999.0, 9999.0))
        assert item.handle_air_pinch(outside) is False
        item.cleanup()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle integration."""

    def test_stem_tools_diagram_generation(self, qapp):
        items = PhysicsTools.create_quantum_well_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="GROUND_STATE_N1",
            n_quantum=1,
            well_width_nm=1.0,
            energy_ev=0.376,
        )

        assert len(items) >= 5
        # First item is background
        assert items[0].boundingRect().width() >= 640.0
        assert items[0].boundingRect().height() >= 340.0

    def test_lesson_board_quantum_well_lifecycle(self, qapp):
        board = LessonBoard()

        assert not board.is_quantum_well_visible()
        assert board.get_quantum_well() is None
        assert board.get_quantum_well_item() is None

        # Toggle ON
        res = board.toggle_quantum_well()
        assert res is True
        assert board.is_quantum_well_visible()
        qw_item = board.get_quantum_well()
        assert qw_item is not None
        assert board.get_quantum_well_item() is qw_item

        # Toggle OFF
        res = board.toggle_quantum_well()
        assert res is False
        assert not board.is_quantum_well_visible()
        assert board.get_quantum_well() is None
        assert board.get_quantum_well_item() is qw_item

        # Verify exclusion from get_items_in_rect
        board.show_quantum_well(True, pos=(100.0, 100.0))
        items_in_rect = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert qw_item not in items_in_rect

        qw_item.cleanup()
