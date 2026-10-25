"""
AirTeach AI - Test Suite for Chemical Equation Balancer & Stoichiometry Engine
=============================================================================
Comprehensive unit tests for:
- Chemical formula parsing (simple, hydrates, polyatomic ions, brackets)
- Equation balancing across synthesis, decomposition, combustion, and complex redox
- Law of Conservation of Mass verification (Lomonosov - Lavoisier)
- Pedagogical derivation steps generation
- Stoichiometric yields calculation (mass, moles, STP volume)
- Interactive ChemicalReactionCardItem graphics item and JSON serialization
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.services.chemical_equation_balancer import (
    ChemicalEquationBalancer,
    ChemicalReactionResult,
    ChemicalSpecies,
    parse_chemical_formula,
)
from src.ui.chemical_reaction_card import ChemicalReactionCardItem


# ═══════════════════════════════════════════════════════════════════════════════
# Chemical Formula Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChemicalFormulaParser:
    """Verify parsing of chemical formulas and molar mass calculation."""

    def test_parse_simple_formulas(self):
        c_h2o, m_h2o = parse_chemical_formula("H2O")
        assert c_h2o == {"H": 2, "O": 1}
        assert abs(m_h2o - (2 * 1.008 + 15.999)) < 1e-3

        c_co2, m_co2 = parse_chemical_formula("CO2")
        assert c_co2 == {"C": 1, "O": 2}

    def test_parse_nested_groups_and_brackets(self):
        c_ca, m_ca = parse_chemical_formula("Ca3(PO4)2")
        assert c_ca == {"Ca": 3, "P": 2, "O": 8}
        assert m_ca > 300.0

        c_al, _ = parse_chemical_formula("Al2(SO4)3")
        assert c_al == {"Al": 2, "S": 3, "O": 12}

        c_fe, _ = parse_chemical_formula("[Fe(CN)6]4-")
        assert c_fe == {"Fe": 1, "C": 6, "N": 6}

    def test_parse_hydrates(self):
        c_cu, m_cu = parse_chemical_formula("CuSO4.5H2O")
        assert c_cu == {"Cu": 1, "S": 1, "O": 9, "H": 10}
        assert m_cu > 240.0


# ═══════════════════════════════════════════════════════════════════════════════
# Chemical Equation Balancing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChemicalEquationBalancing:
    """Verify stoichiometric integer balancing and conservation of mass."""

    @pytest.fixture(autouse=True)
    def setup_balancer(self):
        self.balancer = ChemicalEquationBalancer()

    def test_balance_synthesis_and_decomposition(self):
        res1 = self.balancer.balance("H2 + O2 -> H2O")
        assert res1.success is True
        assert res1.balanced_equation_text == "2 H2 + O2 -> 2 H2O"
        assert res1.mass_conservation_verified is True

        res2 = self.balancer.balance("KClO3 -> KCl + O2")
        assert res2.success is True
        assert res2.balanced_equation_text == "2 KClO3 -> 2 KCl + 3 O2"
        assert res2.mass_conservation_verified is True

    def test_balance_combustion(self):
        res1 = self.balancer.balance("CH4 + O2 -> CO2 + H2O")
        assert res1.success is True
        assert res1.balanced_equation_text == "CH4 + 2 O2 -> CO2 + 2 H2O"

        res2 = self.balancer.balance("C2H5OH + O2 -> CO2 + H2O")
        assert res2.success is True
        assert res2.balanced_equation_text == "C2H5OH + 3 O2 -> 2 CO2 + 3 H2O"
        assert res2.mass_conservation_verified is True

    def test_balance_single_and_double_displacement(self):
        res1 = self.balancer.balance("Al + HCl -> AlCl3 + H2")
        assert res1.success is True
        assert res1.balanced_equation_text == "2 Al + 6 HCl -> 2 AlCl3 + 3 H2"

        res2 = self.balancer.balance("NaOH + H3PO4 -> Na3PO4 + H2O")
        assert res2.success is True
        assert res2.balanced_equation_text == "3 NaOH + H3PO4 -> Na3PO4 + 3 H2O"

    def test_balance_complex_redox(self):
        # Permanganate redox
        res1 = self.balancer.balance("KMnO4 + HCl -> KCl + MnCl2 + Cl2 + H2O")
        assert res1.success is True
        assert res1.balanced_equation_text == "2 KMnO4 + 16 HCl -> 2 KCl + 2 MnCl2 + 5 Cl2 + 8 H2O"
        assert res1.mass_conservation_verified is True

        # Nitric acid oxidation
        res2 = self.balancer.balance("Fe + HNO3 -> Fe(NO3)3 + NO + H2O")
        assert res2.success is True
        assert res2.balanced_equation_text == "Fe + 4 HNO3 -> Fe(NO3)3 + NO + 2 H2O"
        assert res2.mass_conservation_verified is True

        # Sulfuric acid oxidation
        res3 = self.balancer.balance("Cu + H2SO4 -> CuSO4 + SO2 + H2O")
        assert res3.success is True
        assert res3.balanced_equation_text == "Cu + 2 H2SO4 -> CuSO4 + SO2 + 2 H2O"
        assert res3.mass_conservation_verified is True

    def test_pedagogical_derivation_steps(self):
        res = self.balancer.balance("Al + O2 -> Al2O3")
        assert res.success is True
        assert len(res.steps) == 4
        assert "Kiểm đếm nguyên tố" in res.steps[0].title
        assert "ma trận" in res.steps[1].title
        assert "tối giản" in res.steps[2].title
        assert "bảo toàn khối lượng" in res.steps[3].title

    def test_invalid_reaction_error_handling(self):
        # Missing arrow
        res1 = self.balancer.balance("H2 + O2")
        assert res1.success is False
        assert "mũi tên" in res1.error_message

        # Non-conserved elements (element on left not on right)
        res2 = self.balancer.balance("Na + Cl2 -> H2O")
        assert res2.success is False
        assert "không có ở vế" in res2.error_message


# ═══════════════════════════════════════════════════════════════════════════════
# Stoichiometry Calculation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoichiometryCalculations:
    """Verify molar conversions, product yields, and conservation."""

    @pytest.fixture(autouse=True)
    def setup_balancer(self):
        self.balancer = ChemicalEquationBalancer()

    def test_stoichiometry_for_combustion(self):
        reaction = self.balancer.balance("C2H5OH + O2 -> CO2 + H2O")
        st = self.balancer.compute_stoichiometry(
            reaction,
            given_species_index=0,  # C2H5OH
            given_quantity=46.068,  # ~1 mol C2H5OH
            quantity_type="mass_g",
        )
        assert "error" not in st
        assert abs(st["reaction_events_moles"] - 1.0) < 0.01

        data = {d["formula"]: d for d in st["species_data"]}
        # 1 mol C2H5OH requires 3 mol O2 -> forms 2 mol CO2 + 3 mol H2O
        assert abs(data["O2"]["moles"] - 3.0) < 0.02
        assert abs(data["CO2"]["moles"] - 2.0) < 0.02
        assert abs(data["H2O"]["moles"] - 3.0) < 0.02

        # Check total mass in vs total mass out
        mass_in = data["C2H5OH"]["mass_g"] + data["O2"]["mass_g"]
        mass_out = data["CO2"]["mass_g"] + data["H2O"]["mass_g"]
        assert abs(mass_in - mass_out) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# ChemicalReactionCardItem Graphics & Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChemicalReactionCardItem:
    """Verify interactive graphics behavior, toggle collapse, and serialization."""

    @pytest.fixture(autouse=True)
    def setup_card(self, qapp):
        balancer = ChemicalEquationBalancer()
        self.result = balancer.balance("Fe + HNO3 -> Fe(NO3)3 + NO + H2O")
        self.card = ChemicalReactionCardItem(self.result, is_expanded=True)

    def test_initial_card_properties(self):
        assert self.card.result.success is True
        assert self.card.is_expanded is True
        rect = self.card.boundingRect()
        assert rect.width() == 500.0
        assert rect.height() > 200.0

    def test_toggle_expansion(self):
        h_expanded = self.card.boundingRect().height()
        self.card.set_expanded(False)
        assert self.card.is_expanded is False
        h_collapsed = self.card.boundingRect().height()
        assert h_collapsed < h_expanded
        assert h_collapsed <= 180.0

        self.card.toggle_expanded()
        assert self.card.is_expanded is True
        assert self.card.boundingRect().height() == h_expanded

    def test_serialization_and_deserialization(self):
        self.card.setPos(200.0, 300.0)
        self.card.set_expanded(False)
        data = self.card.to_dict()

        assert data["type"] == "chemical_reaction_card"
        assert data["pos"] == [200.0, 300.0]
        assert data["expanded"] is False
        assert data["result"]["success"] is True

        # Reconstruct
        restored = ChemicalReactionCardItem.from_dict(data)
        assert restored.pos() == QPointF(200.0, 300.0)
        assert restored.is_expanded is False
        assert restored.result.balanced_equation_text == self.result.balanced_equation_text
        assert len(restored.result.steps) == len(self.result.steps)

    def test_rendering_on_qimage(self):
        img = QImage(600, 600, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(img)
        self.card.paint(painter, None, None)
        painter.end()

    def test_handle_air_tap(self):
        # 1. Tap on toggle expand button (x: 440, y: 20)
        initial_expanded = self.card.is_expanded
        consumed = self.card.handle_air_tap(QPointF(self.card.x() + 450.0, self.card.y() + 20.0))
        assert consumed is True
        assert self.card.is_expanded != initial_expanded

        # 2. Tap inside card
        consumed = self.card.handle_air_tap(QPointF(self.card.x() + 100.0, self.card.y() + 50.0))
        assert consumed is True

        # 3. Tap outside card
        consumed = self.card.handle_air_tap(QPointF(self.card.x() - 100.0, self.card.y() - 100.0))
        assert consumed is False


class TestSmartInkReviewDialogChemicalIntegration:
    """Verify dialog auto-balancing and reaction card confirmation."""

    def test_dialog_auto_balances_chemical_equation(self, qapp):
        from src.services.smart_ink_ocr import OcrResult
        from src.ui.smart_ink_review_dialog import SmartInkReviewDialog

        ocr_res = OcrResult(
            text="Fe + HNO3 -> Fe(NO3)3 + NO + H2O",
            latex=r"\text{Fe} + \text{HNO}_3 \rightarrow \text{Fe(NO}_3\text{)}_3 + \text{NO} + \text{H}_2\text{O}",
            confidence=0.98,
            is_formula=True,
        )
        dlg = SmartInkReviewDialog(ocr_res)
        assert dlg._reaction_result is not None
        assert dlg._reaction_result.is_balanced is True
        assert "4 HNO3" in dlg._reaction_result.balanced_equation

        # Check signals
        emitted_reaction = []
        emitted_text = []
        dlg.reaction_confirmed.connect(lambda r, rep: emitted_reaction.append((r, rep)))
        dlg.result_confirmed.connect(lambda t, rep: emitted_text.append((t, rep)))

        dlg._on_accept()
        assert len(emitted_reaction) == 1
        assert emitted_reaction[0][0].is_balanced is True
        assert emitted_reaction[0][1] is True
        assert len(emitted_text) == 1

