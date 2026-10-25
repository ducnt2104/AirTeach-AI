"""
AirTeach AI - Test Suite for Symbolic Math Engine & Derivation Cards
=====================================================================
Comprehensive unit tests for:
- LaTeX to Python expression conversion (fractions, radicals, powers, trig, implicit mult)
- Linear and quadratic equation solving with pedagogical step-by-step breakdown
- Symbolic differentiation with sum/product rules breakdown
- Indefinite and definite integration with Newton-Leibniz breakdown
- Algebraic and trigonometric simplification
- Interactive MathDerivationCardItem graphics item, toggle expansion, and serialization
- SmartInkReviewDialog integration with symbolic solver
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.services.smart_ink_ocr import OcrResult
from src.services.symbolic_math import (
    DerivationStep,
    MathDerivationResult,
    MathOperationType,
    SymbolicMathEngine,
    convert_latex_to_python_expr,
    detect_primary_variable,
)
from src.ui.math_derivation_card import MathDerivationCardItem
from src.ui.smart_ink_review_dialog import SmartInkReviewDialog


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX to Python Converter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatexToPythonConverter:
    """Verify LaTeX math string parsing and normalization."""

    def test_convert_fractions(self):
        s1 = convert_latex_to_python_expr(r"\frac{2x + 1}{3}")
        assert s1 == "((2*x + 1)/(3))"

        s2 = convert_latex_to_python_expr(r"\dfrac{a}{b}")
        assert s2 == "((a)/(b))"

    def test_convert_radicals(self):
        s1 = convert_latex_to_python_expr(r"\sqrt{x + 4}")
        assert s1 == "sqrt(x + 4)"

        s2 = convert_latex_to_python_expr(r"\sqrt[3]{x}")
        assert s2 == "((x)**(1/(3)))"

    def test_convert_powers_and_trig(self):
        s1 = convert_latex_to_python_expr(r"x^2 + 5x + 6")
        assert s1 == "x**2 + 5*x + 6"

        s2 = convert_latex_to_python_expr(r"\sin^2(x) + \cos^2(x)")
        assert s2 == "(sin(x))**2 + (cos(x))**2"

    def test_convert_implicit_multiplication(self):
        s1 = convert_latex_to_python_expr(r"2(x + 3)(x - 1)")
        assert s1 == "2*(x + 3)*(x - 1)"

        s2 = convert_latex_to_python_expr(r"3.5x + 4y")
        assert s2 == "3.5*x + 4*y"

    def test_detect_primary_variable(self):
        assert detect_primary_variable("3*x**2 + 2*x - 1") == "x"
        assert detect_primary_variable("9.8*t**2 + 5*t") == "t"
        assert detect_primary_variable("2*y + 10") == "y"


# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic Equation Solving Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolicEquationSolving:
    """Verify equation solving with pedagogical derivation steps."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = SymbolicMathEngine()

    def test_solve_linear_equation(self):
        res = self.engine.solve_equation(r"2x + 10 = 26")
        assert res.success is True
        assert res.operation == "solve"
        assert "8" in res.solution_latex
        assert len(res.steps) >= 3
        # Check step titles
        titles = [s.title for s in res.steps]
        assert any("bậc nhất" in t for t in titles)
        assert any("Chuyển vế" in t for t in titles)

    def test_solve_quadratic_real_roots(self):
        res = self.engine.solve_equation(r"x^2 - 5x + 6 = 0")
        assert res.success is True
        assert res.operation == "solve"
        assert "3" in res.solution_latex and "2" in res.solution_latex
        assert len(res.steps) == 4
        assert "Delta" in res.steps[1].title
        assert res.approximate_values == [3.0, 2.0]

    def test_solve_quadratic_double_root(self):
        res = self.engine.solve_equation(r"x^2 - 4x + 4 = 0")
        assert res.success is True
        assert "2" in res.solution_latex
        titles = [s.title for s in res.steps]
        assert any("nghiệm kép" in t for t in titles)

    def test_solve_quadratic_complex_roots(self):
        res = self.engine.solve_equation(r"x^2 + 4 = 0")
        assert res.success is True
        assert "i" in res.solution_latex or "I" in res.solution_raw
        titles = [s.title for s in res.steps]
        assert any("vô nghiệm thực" in t for t in titles)


# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic Calculus & Simplification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolicCalculusAndSimplification:
    """Verify symbolic differentiation, integration, and algebraic simplification."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = SymbolicMathEngine()

    def test_differentiate_polynomial(self):
        res = self.engine.differentiate(r"\frac{d}{dx}(x^3 + 2x^2 - 5x + 1)")
        assert res.success is True
        assert res.operation == "derivative"
        assert "3 x^{2} + 4 x - 5" in res.solution_latex
        assert len(res.steps) >= 2

    def test_differentiate_raw_expression(self):
        res = self.engine.differentiate(r"3x^2 + 6x")
        assert res.success is True
        assert "6 x + 6" in res.solution_latex

    def test_integrate_indefinite(self):
        res = self.engine.integrate_expression(r"\int (3x^2 + 2x) dx")
        assert res.success is True
        assert res.operation == "integral"
        assert "x^{3} + x^{2} + C" in res.solution_latex
        assert len(res.steps) == 3

    def test_integrate_definite(self):
        res = self.engine.integrate_expression(r"\int_0^2 (3x^2 + 2x) dx")
        assert res.success is True
        assert res.operation == "integral"
        assert "12" in res.solution_latex
        assert res.approximate_values == [12.0]

    def test_simplify_trigonometric(self):
        res = self.engine.simplify_expression(r"\sin^2(x) + \cos^2(x)")
        assert res.success is True
        assert res.operation == "simplify"
        assert res.solution_latex == "1"

    def test_simplify_algebraic(self):
        res = self.engine.simplify_expression(r"(x + 1)^2 - (x - 1)^2")
        assert res.success is True
        assert "4 x" in res.solution_latex

    def test_evaluate_auto_heuristic(self):
        # Auto solve
        r_eq = self.engine.evaluate_auto(r"3x - 9 = 0")
        assert r_eq.operation == "solve"
        assert "3" in r_eq.solution_latex

        # Auto diff
        r_diff = self.engine.evaluate_auto(r"\frac{d}{dx}(x^2)")
        assert r_diff.operation == "derivative"
        assert "2 x" in r_diff.solution_latex

        # Auto int
        r_int = self.engine.evaluate_auto(r"\int 2x dx")
        assert r_int.operation == "integral"
        assert "x^{2} + C" in r_int.solution_latex


# ═══════════════════════════════════════════════════════════════════════════════
# MathDerivationCardItem Graphics & Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathDerivationCardItem:
    """Verify MathDerivationCardItem interactive graphics behavior and serialization."""

    @pytest.fixture(autouse=True)
    def setup_card(self, qapp):
        engine = SymbolicMathEngine()
        self.result = engine.solve_equation(r"x^2 - 5x + 6 = 0")
        self.card = MathDerivationCardItem(self.result, is_expanded=True)

    def test_initial_card_properties(self):
        assert self.card.result.success is True
        assert self.card.is_expanded is True
        rect = self.card.boundingRect()
        assert rect.width() == 460.0
        assert rect.height() > 200.0  # Expanded height includes steps

    def test_toggle_expansion(self):
        expanded_h = self.card.boundingRect().height()
        self.card.set_expanded(False)
        assert self.card.is_expanded is False
        collapsed_h = self.card.boundingRect().height()
        assert collapsed_h < expanded_h
        assert collapsed_h <= 140.0

        self.card.toggle_expanded()
        assert self.card.is_expanded is True
        assert self.card.boundingRect().height() == expanded_h

    def test_serialization_and_deserialization(self):
        self.card.setPos(150.0, 250.0)
        self.card.set_expanded(False)
        data = self.card.to_dict()

        assert data["type"] == "math_derivation_card"
        assert data["pos"] == [150.0, 250.0]
        assert data["expanded"] is False
        assert data["result"]["success"] is True

        # Reconstruct
        restored = MathDerivationCardItem.from_dict(data)
        assert restored.pos() == QPointF(150.0, 250.0)
        assert restored.is_expanded is False
        assert restored.result.solution_latex == self.result.solution_latex
        assert len(restored.result.steps) == len(self.result.steps)


# ═══════════════════════════════════════════════════════════════════════════════
# SmartInkReviewDialog Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmartInkReviewDialogIntegration:
    """Verify dialog interaction with symbolic math derivation."""

    def test_dialog_auto_derives_formula(self, qapp):
        ocr_res = OcrResult(
            latex=r"x^2 - 5x + 6 = 0",
            confidence=0.95,
            is_formula=True,
        )
        dlg = SmartInkReviewDialog(ocr_res)
        assert dlg._derivation_result is not None
        assert dlg._derivation_result.success is True
        assert "3" in dlg._derivation_result.solution_latex

        # Check signals
        emitted_deriv = []
        emitted_text = []
        dlg.derivation_confirmed.connect(lambda r, rep: emitted_deriv.append((r, rep)))
        dlg.result_confirmed.connect(lambda t, rep: emitted_text.append((t, rep)))

        dlg._on_accept()
        assert len(emitted_deriv) == 1
        assert emitted_deriv[0][0].operation == "solve"
        assert emitted_deriv[0][1] is True  # replace original ink
        assert len(emitted_text) == 1
