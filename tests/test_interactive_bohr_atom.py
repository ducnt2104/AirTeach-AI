"""
Unit Tests for Interactive Quantum Physics Bohr Atomic Model & Hydrogen Spectral Line Series Simulator
========================================================================================================
Verifies:
  1. Quantum physical constants (Rydberg energy E_0 = 13.6057 eV, Bohr radius r_0 = 0.529 Å, hc = 1239.84 eV·nm).
  2. Bohr stationary orbits: r_n = n² * r_0, orbital velocity v_n = v_1 / n, orbital letters K, L, M, N, O, P.
  3. Discrete energy levels: E_n = -13.6057 / n² eV and ionization energy.
  4. Authentic Hydrogen spectral transitions:
       - Balmer H_α (3 → 2, 656.3 nm Red)
       - Balmer H_β (4 → 2, 486.1 nm Cyan)
       - Lyman L_α (2 → 1, 121.6 nm UV)
       - Paschen P_α (4 → 3, 1875.1 nm IR)
  5. Quantum jump triggering, photon wave packet emission, and parameter signals.
  6. Presets, level chips, and diagram modes (ORBITS_AND_LEVELS vs EMISSION_SPECTRUM).
  7. Air gestures (air tap / air pinch) and mouse interaction.
  8. LessonBoard lifecycle, visibility toggle, selection exclusion, and vector diagram stamping.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_bohr_atom import (
    InteractiveBohrAtomItem,
    RYDBERG_ENERGY_EV,
    BOHR_RADIUS_ANGSTROM,
    HC_EV_NM,
    VELOCITY_V1_KMS,
    ORBIT_NAMES,
    PRESETS,
    wavelength_to_spectral_color,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def item(board):
    it = InteractiveBohrAtomItem(pos=(320.0, 160.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestBohrAtomPhysics:
    """Verifies quantum mechanics postulates, orbit math, and spectral formulas."""

    def test_quantum_constants(self):
        assert abs(RYDBERG_ENERGY_EV - 13.605693) < 1e-4
        assert abs(BOHR_RADIUS_ANGSTROM - 0.529177) < 1e-4
        assert abs(HC_EV_NM - 1239.841984) < 1e-3
        assert abs(VELOCITY_V1_KMS - 2187.69) < 1e-1
        assert ORBIT_NAMES[1] == "K"
        assert ORBIT_NAMES[2] == "L"
        assert ORBIT_NAMES[6] == "P"

    def test_bohr_orbits_and_energy_levels(self, item):
        """r_n = n² * r_0, E_n = -E_0 / n², v_n = v_1 / n."""
        # Ground state n = 1
        r1 = item.get_orbit_radius_angstrom(1)
        e1 = item.get_energy_level_ev(1)
        v1 = item.get_electron_velocity_kms(1)
        assert abs(r1 - BOHR_RADIUS_ANGSTROM) < 1e-4
        assert abs(e1 - (-13.605693)) < 1e-4
        assert abs(v1 - VELOCITY_V1_KMS) < 1e-2

        # Level n = 2 (L orbit)
        r2 = item.get_orbit_radius_angstrom(2)
        e2 = item.get_energy_level_ev(2)
        v2 = item.get_electron_velocity_kms(2)
        assert abs(r2 - (4.0 * BOHR_RADIUS_ANGSTROM)) < 1e-4
        assert abs(e2 - (-13.605693 / 4.0)) < 1e-4
        assert abs(v2 - (VELOCITY_V1_KMS / 2.0)) < 1e-2

        # Level n = 3 (M orbit)
        r3 = item.get_orbit_radius_angstrom(3)
        e3 = item.get_energy_level_ev(3)
        assert abs(r3 - (9.0 * BOHR_RADIUS_ANGSTROM)) < 1e-4
        assert abs(e3 - (-13.605693 / 9.0)) < 1e-4

    def test_balmer_alpha_transition(self, item):
        """m = 3 -> n = 2: ΔE = 1.8897 eV, λ ≈ 656.3 nm (Red H_α)."""
        item.set_levels(3, 2)
        delta_e = item.get_transition_energy_ev()
        expected_de = abs((-13.605693 / 9.0) - (-13.605693 / 4.0))
        assert abs(delta_e - expected_de) < 1e-4
        assert abs(delta_e - 1.88968) < 1e-3

        wl = item.get_transition_wavelength_nm()
        assert abs(wl - 656.28) < 0.5  # Authentic Balmer H_alpha

        freq = item.get_transition_frequency_hz()
        assert 4.5e14 < freq < 4.6e14  # Red light frequency ~ 4.57e14 Hz

        series = item.get_series_name()
        assert "Balmer" in series

        color, band, line_nm = wavelength_to_spectral_color(wl)
        assert "Đỏ" in band or "Đỏ" in line_nm
        assert color.red() > 200

    def test_balmer_beta_and_lyman_transitions(self, item):
        """Balmer H_β (4 -> 2, 486.1 nm) and Lyman L_α (2 -> 1, 121.6 nm)."""
        # Balmer H_beta (4 -> 2)
        item.set_levels(4, 2)
        wl_beta = item.get_transition_wavelength_nm()
        assert abs(wl_beta - 486.13) < 0.5
        _, band_beta, _ = wavelength_to_spectral_color(wl_beta)
        assert "Lam" in band_beta

        # Lyman L_alpha (2 -> 1)
        item.set_levels(2, 1)
        wl_lyman = item.get_transition_wavelength_nm()
        assert abs(wl_lyman - 121.57) < 0.5
        assert "Lyman" in item.get_series_name()
        _, band_lyman, _ = wavelength_to_spectral_color(wl_lyman)
        assert "Tử ngoại" in band_lyman

        # Paschen P_alpha (4 -> 3)
        item.set_levels(4, 3)
        wl_paschen = item.get_transition_wavelength_nm()
        assert abs(wl_paschen - 1875.1) < 2.0
        assert "Paschen" in item.get_series_name()
        _, band_paschen, _ = wavelength_to_spectral_color(wl_paschen)
        assert "Hồng ngoại" in band_paschen

    def test_quantum_jump_and_presets(self, item):
        """Verifies quantum jump emission, preset application, and photon packet creation."""
        signals_received = []
        item.quantum_jump_triggered.connect(lambda m, n: signals_received.append((m, n)))

        item.apply_preset("BALMER_ALPHA")
        assert item.level_m == 3
        assert item.level_n == 2
        assert len(signals_received) == 1
        assert signals_received[0] == (3, 2)
        assert len(item._photon_packets) >= 1

        item.apply_preset("LYMAN_ALPHA")
        assert item.level_m == 2
        assert item.level_n == 1
        assert len(signals_received) == 2
        assert signals_received[1] == (2, 1)

    def test_level_clamping_and_modes(self, item):
        """Levels must clamp to [1..6] and modes switch correctly."""
        item.set_level_m(10)
        assert item.level_m == 6

        item.set_level_m(-2)
        assert item.level_m == 1

        item.set_level_n(0)
        assert item.level_n == 1

        item.set_level_n(99)
        assert item.level_n == 6

        item.set_diagram_mode("EMISSION_SPECTRUM")
        assert item._diagram_mode == "EMISSION_SPECTRUM"

        item.set_diagram_mode("ORBITS_AND_LEVELS")
        assert item._diagram_mode == "ORBITS_AND_LEVELS"

    def test_air_gestures_and_interactions(self, item):
        """Air pinch/tap triggers presets, jump button, mode toggle, and close."""
        # 1. Preset button air pinch
        preset_rects = item._get_preset_rects()
        p0_key, p0_rect = preset_rects[0]
        scene_pt = item.mapToScene(p0_rect.center())
        res = item.handle_air_pinch(scene_pt)
        assert res is True
        assert item._current_preset == p0_key

        # 2. Mode toggle air pinch
        _, tab_spec = item._get_mode_toggle_rects()
        scene_tab = item.mapToScene(tab_spec.center())
        res_tab = item.handle_air_pinch(scene_tab)
        assert res_tab is True
        assert item._diagram_mode == "EMISSION_SPECTRUM"

        # 3. Jump button air tap
        jump_rect = item._get_jump_button_rect()
        scene_jump = item.mapToScene(jump_rect.center())
        packets_before = len(item._photon_packets)
        res_jump = item.handle_air_tap(scene_jump)
        assert res_jump is True
        assert len(item._photon_packets) == packets_before + 1

        # 4. Close button air pinch
        c_rect = item._get_close_button_rect()
        scene_c = item.mapToScene(c_rect.center())
        assert item.isVisible() is True
        res_close = item.handle_air_pinch(scene_c)
        assert res_close is True
        assert item.isVisible() is False

    def test_lesson_board_lifecycle_and_stamping(self, board):
        """Verifies LessonBoard show/toggle/hide methods and vector diagram stamping."""
        assert board.is_bohr_atom_visible() is False
        assert board.get_bohr_atom() is None

        # Show
        it = board.show_bohr_atom(pos=(400.0, 200.0))
        assert it is not None
        assert board.is_bohr_atom_visible() is True
        assert board.get_bohr_atom() == it

        # Exclusion in get_items_in_rect (selection exclusion invariant)
        rect_all = board.sceneRect()
        selected = board.get_items_in_rect(rect_all)
        assert it not in selected

        # Stamping vector diagram
        items_before = len(board.items())
        it.stamp_to_board()
        items_after = len(board.items())
        assert items_after > items_before

        # Toggle hide
        vis = board.toggle_bohr_atom()
        assert vis is False
        assert board.is_bohr_atom_visible() is False

        # Toggle show
        vis2 = board.toggle_bohr_atom()
        assert vis2 is True
        assert board.is_bohr_atom_visible() is True

        # Clean cleanup
        it.cleanup()
        board.removeItem(it)
