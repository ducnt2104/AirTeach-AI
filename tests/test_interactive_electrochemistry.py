"""
AirTeach AI - Unit Tests for Interactive Chemistry Electrochemistry & Galvanic / Electrolysis Cell Simulator
========================================================================================================
Comprehensive automated tests verifying:
  1. Standard electrode potentials E°, Daniell cell EMF, Gibbs free energy ΔG°.
  2. Nernst equation concentration dependence and concentration cell thermodynamics.
  3. Faraday's laws of electrolysis: mass deposition m = (A·I·t)/(n·F) and gas volume generation.
  4. 5 Authentic Curriculum Presets (Daniell, Lead-Acid, CuSO4 Electrolysis, Water Splitting PEM, Concentration Cell).
  5. Touchless air pinch gestures, parameter adjustments, and animation ticks.
  6. Permanent vector diagram stamping via ChemistryTools.create_electrochemistry_diagram.
  7. LessonBoard lifecycle and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_electrochemistry import (
    FARADAY_F,
    GAS_CONST_R,
    TEMP_STANDARD_K,
    InteractiveElectrochemistryItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import ChemistryTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestElectrochemistryThermodynamics:
    """Tests authentic physical and chemical thermodynamic calculations."""

    def test_daniell_cell_emf_and_gibbs(self, qapp):
        preset = PRESETS["DANIELL_CELL_ZN_CU"]
        # E°_cell = E°_cathode (Cu: +0.342) - E°_anode (Zn: -0.763) = 1.105 V
        e0_cell = preset.e0_cathode - preset.e0_anode
        assert abs(e0_cell - 1.105) < 1e-3

        # ΔG° = -n * F * E°_cell
        n = preset.n_electrons
        delta_g_j = -n * FARADAY_F * e0_cell
        delta_g_kj = delta_g_j / 1000.0
        # Expected ΔG° ≈ -213.2 kJ/mol (spontaneous galvanic reaction)
        assert abs(delta_g_kj - (-213.23)) < 0.5
        assert delta_g_kj < 0.0

    def test_nernst_equation_behavior(self, qapp):
        item = InteractiveElectrochemistryItem()
        item.load_preset("DANIELL_CELL_ZN_CU")

        # Standard conditions: [Zn²⁺] = 1.0M, [Cu²⁺] = 1.0M -> log10(1) = 0 -> E = E°
        item.set_concentrations(1.0, 1.0)
        e_std = item.calculate_cell_potential()
        assert abs(e_std - 1.105) < 1e-3

        # Le Chatelier / Nernst: Higher [Zn²⁺] (product) shifts equilibrium back, lowers E
        item.set_concentrations(2.0, 1.0)
        e_high_product = item.calculate_cell_potential()
        assert e_high_product < e_std

        # Higher [Cu²⁺] (reactant) increases E
        item.set_concentrations(1.0, 2.0)
        e_high_reactant = item.calculate_cell_potential()
        assert e_high_reactant > e_std

        item.cleanup()

    def test_concentration_cell_nernst(self, qapp):
        item = InteractiveElectrochemistryItem()
        item.load_preset("CONCENTRATION_CELL")

        # Symmetrical concentrations -> E = 0
        item.set_concentrations(1.0, 1.0)
        assert abs(item.calculate_cell_potential() - 0.0) < 1e-4

        # 0.01M Anode || 1.0M Cathode: E = (0.05916 / 2) * log10(1.0 / 0.01) = 0.05916 * 1.0 = 0.05916 V
        item.set_concentrations(0.01, 1.0)
        e_conc = item.calculate_cell_potential()
        assert abs(e_conc - 0.05916) < 1e-3

        item.cleanup()

    def test_faraday_electrolysis_mass_and_gas(self, qapp):
        item = InteractiveElectrochemistryItem()
        item.load_preset("ELECTROLYSIS_CUSO4")
        item.set_current(2.0)  # 2.0 Amperes
        item._elapsed_time_s = 964.8533  # t = 964.85 s

        # Faraday mass: m = (A * I * t) / (n * F)
        # For Cu (n=2, A=63.546): m = (63.546 * 2.0 * 964.8533) / (2 * 96485.33) = 63.546 * 0.01 = 0.63546 g
        dep_mass_g = item.calculate_deposited_mass_g()
        assert abs(dep_mass_g - 0.63546) < 1e-3

        # Water splitting ratio
        item.load_preset("WATER_SPLITTING_PEM")
        v_h2, v_o2 = item.calculate_gas_volumes_stp_liters()
        # V(H2) / V(O2) should be authentic 2:1 ratio
        assert v_h2 > 0.0
        assert v_o2 > 0.0
        assert abs((v_h2 / v_o2) - 2.0) < 1e-3

        item.cleanup()


class TestElectrochemistryItemInteraction:
    """Tests interactive item UI controls, air pinch gestures, and presets."""

    def test_preset_switching(self, qapp):
        item = InteractiveElectrochemistryItem()
        for code in ["DANIELL_CELL_ZN_CU", "LEAD_ACID_ACCUMULATOR", "ELECTROLYSIS_CUSO4", "WATER_SPLITTING_PEM", "CONCENTRATION_CELL"]:
            ok = item.load_preset(code)
            assert ok is True
            assert item.active_preset_code == code
            assert item.preset is not None
        item.cleanup()

    def test_air_pinch_preset_buttons(self, qapp):
        item = InteractiveElectrochemistryItem()
        btn_rect = item._btn_rects["preset_LEAD_ACID_ACCUMULATOR"]
        pinch_pt = item.mapToScene(btn_rect.center())

        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "LEAD_ACID_ACCUMULATOR"

        item.cleanup()

    def test_air_pinch_parameter_controls(self, qapp):
        item = InteractiveElectrochemistryItem()
        item.load_preset("DANIELL_CELL_ZN_CU")
        curr_before = item.current_amperes

        # Increase current
        inc_curr_pt = item.mapToScene(item._btn_rects["inc_curr"].center())
        assert item.handle_air_pinch(inc_curr_pt) is True
        assert item.current_amperes > curr_before

        # Decrease current
        dec_curr_pt = item.mapToScene(item._btn_rects["dec_curr"].center())
        assert item.handle_air_pinch(dec_curr_pt) is True
        assert abs(item.current_amperes - curr_before) < 1e-3

        item.cleanup()

    def test_animation_tick_advances_particles(self, qapp):
        item = InteractiveElectrochemistryItem()
        item.load_preset("DANIELL_CELL_ZN_CU")
        e_pos_before = list(item._electrons)

        item._on_animation_tick()
        assert item._elapsed_time_s > 0.0
        # Electron fractions should advance
        assert item._electrons != e_pos_before

        # Toggle pause
        item.toggle_animation()
        assert item.is_animating is False
        t_paused = item._elapsed_time_s
        item._on_animation_tick()
        assert item._elapsed_time_s == t_paused  # No advance when paused

        item.cleanup()

    def test_stamp_creates_vector_diagram(self, qapp):
        board = LessonBoard()
        item = InteractiveElectrochemistryItem(pos=(200.0, 100.0))
        board.addItem(item)

        items_before = len(board.items())
        item.stamp_to_board()
        items_after = len(board.items())
        assert items_after > items_before

        item.cleanup()

    def test_chemistry_tools_vector_diagram(self, qapp):
        items = ChemistryTools.create_electrochemistry_diagram(
            center=(400.0, 300.0),
            preset_code="DANIELL_CELL_ZN_CU",
            anode_conc=1.0,
            cathode_conc=1.0,
            current_a=2.0,
        )
        assert len(items) >= 10
        # Check text items contain thermodynamic constants
        texts = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("DANIELL" in t.upper() or "PIN GALVANI" in t.upper() for t in texts)
        assert any("NERNST" in t.upper() or "E°" in t for t in texts)


class TestElectrochemistryBoardIntegration:
    """Tests LessonBoard lifecycle methods and lasso selection exclusion."""

    def test_board_lifecycle(self, qapp):
        board = LessonBoard()
        assert board.is_electrochemistry_visible() is False
        assert board.get_electrochemistry() is None

        # Show
        item = board.show_electrochemistry(True)
        assert item is not None
        assert board.is_electrochemistry_visible() is True
        assert board.get_electrochemistry() is item
        assert board.get_electrochemistry_item() is item

        # Toggle off
        vis = board.toggle_electrochemistry()
        assert vis is False
        assert board.is_electrochemistry_visible() is False

        # Toggle on
        vis = board.toggle_electrochemistry()
        assert vis is True
        assert board.is_electrochemistry_visible() is True

        item.cleanup()

    def test_lasso_exclusion(self, qapp):
        board = LessonBoard()
        item = board.show_electrochemistry(True, pos=(200.0, 200.0))
        rect = QRectF(150.0, 150.0, 850.0, 600.0)

        # Interactive item itself should be excluded from selection/lasso
        selected_items = board.get_items_in_rect(rect)
        assert item not in selected_items

        item.cleanup()
