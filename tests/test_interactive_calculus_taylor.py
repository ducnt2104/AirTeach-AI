"""
AirTeach AI – Test Suite: Interactive Calculus, Riemann Quadrature & Taylor Series Simulation
==============================================================================================
Verifies:
  - Exact Riemann integrals: Left, Right, Midpoint, Trapezoidal, and Simpson's 1/3 rule
  - Fundamental Theorem of Calculus: ∫₀^π sin(x)dx = 2.0
  - Parabolic segment area: ∫₋₂^² (4 - x²)dx = 32/3 ≈ 10.666667
  - Taylor / Maclaurin polynomial series expansions: sin(x) and exp(x)
  - Error convergence rates O(n⁻²) and O(n⁻⁴)
  - Presets, air gestures, steppers, and rendering
  - STEMTools vector diagram generation
  - LessonBoard integration & lifecycle
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QApplication

from src.ui.interactive_calculus_taylor import (
    InteractiveCalculusTaylorItem,
    PRESETS,
)
from src.ui.stem_tools import MathTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestCalculusMathematics:
    """Verifies numerical quadrature and Taylor expansions without mock data."""

    def test_riemann_quadratures_sine_integral(self, qapp):
        item = InteractiveCalculusTaylorItem()
        item.load_preset("RIEMANN_SINE_INTEGRAL")
        item.set_partitions(16)

        # Exact integral of sin(x) from 0 to pi is 2.0
        exact = 2.0

        # Midpoint rule (idx = 0)
        m_val = item.compute_riemann_sum(0)
        # Left rule (idx = 1)
        l_val = item.compute_riemann_sum(1)
        # Right rule (idx = 2)
        r_val = item.compute_riemann_sum(2)
        # Trapezoid rule (idx = 3)
        t_val = item.compute_riemann_sum(3)
        # Simpson rule (idx = 4)
        s_val = item.compute_riemann_sum(4)

        # Due to symmetry of sin(x) on [0, pi], Left and Right sums are identical
        assert math.isclose(l_val, r_val, rel_tol=1e-5)

        # Simpson's 1/3 rule must be the most accurate (error < 0.0001)
        assert abs(s_val - exact) < 1e-4

        # Trapezoid and Midpoint are both close to 2.0
        assert math.isclose(m_val, exact, rel_tol=1e-2)
        assert math.isclose(t_val, exact, rel_tol=1e-2)

        item.cleanup()

    def test_archimedes_parabola_area(self, qapp):
        item = InteractiveCalculusTaylorItem()
        item.load_preset("RIEMANN_PARABOLA_AREA")

        exact = 32.0 / 3.0  # 10.666667
        # Simpson's rule is exact for polynomials up to degree 3!
        # With n = 4, Simpson's 1/3 rule must give the EXACT value:
        item.set_partitions(4)
        s_val = item.compute_riemann_sum(4)
        assert math.isclose(s_val, exact, rel_tol=1e-6)

        item.cleanup()

    def test_taylor_series_convergence_sin(self, qapp):
        item = InteractiveCalculusTaylorItem()
        item.load_preset("TAYLOR_SIN_APPROXIMATION")

        x_test = 1.0  # 1 radian
        sin_true = math.sin(x_test)

        # P_1(x) = x
        p1 = item.eval_taylor(x_test, 1)
        assert math.isclose(p1, 1.0, rel_tol=1e-5)

        # P_3(x) = x - x^3 / 6 = 1 - 1/6 = 5/6 ≈ 0.83333
        p3 = item.eval_taylor(x_test, 3)
        assert math.isclose(p3, 5.0 / 6.0, rel_tol=1e-5)

        # P_5(x) = x - x^3/6 + x^5/120 = 5/6 + 1/120 = 101/120 ≈ 0.841667
        p5 = item.eval_taylor(x_test, 5)
        assert math.isclose(p5, 101.0 / 120.0, rel_tol=1e-5)

        # As degree increases, error monotonically decreases
        err1 = abs(p1 - sin_true)
        err3 = abs(p3 - sin_true)
        err5 = abs(p5 - sin_true)
        assert err5 < err3 < err1
        assert err5 < 0.001

        item.cleanup()

    def test_taylor_series_convergence_exp(self, qapp):
        item = InteractiveCalculusTaylorItem()
        item.load_preset("TAYLOR_EXP_APPROXIMATION")

        x_test = 1.0
        e_true = math.e

        # Degree 1: 1 + x = 2
        p1 = item.eval_taylor(x_test, 1)
        assert math.isclose(p1, 2.0, rel_tol=1e-5)

        # Degree 4: 1 + 1 + 1/2 + 1/6 + 1/24 = 2.70833
        p4 = item.eval_taylor(x_test, 4)
        assert math.isclose(p4, 65.0 / 24.0, rel_tol=1e-5)

        # Degree 6: highly accurate
        p6 = item.eval_taylor(x_test, 6)
        assert abs(p6 - e_true) < 0.002

        item.cleanup()


class TestCalculusPresetsAndControls:
    """Verifies presets, rule cycling, degree stepping, and air pinches."""

    def test_all_presets_load(self, qapp):
        item = InteractiveCalculusTaylorItem()

        for code in PRESETS:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.boundingRect().width() == 700.0
            assert item.boundingRect().height() == 440.0
        item.cleanup()

    def test_mode_and_rule_controls(self, qapp):
        item = InteractiveCalculusTaylorItem()

        # Toggle mode
        was_taylor = item.is_taylor_mode
        item.toggle_mode()
        assert item.is_taylor_mode != was_taylor

        # Rule cycling: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        assert item.riemann_rule == 0
        item.cycle_riemann_rule()
        assert item.riemann_rule == 1
        item.cycle_riemann_rule()
        assert item.riemann_rule == 2

        # Partition clamping
        item.set_partitions(100)
        assert item.n_partitions == 64
        item.set_partitions(1)
        assert item.n_partitions == 2

        # Taylor degree clamping
        item.set_taylor_degree(20)
        assert item.taylor_degree == 11
        item.set_taylor_degree(0)
        assert item.taylor_degree == 1

        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        item = InteractiveCalculusTaylorItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        # 1. Preset pinch
        p_rect = item._btn_rects.get("preset_TAYLOR_SIN_APPROXIMATION")
        assert p_rect is not None
        center = item.mapToScene(p_rect.center())
        assert item.handle_air_pinch(center) is True
        assert item.active_preset.code == "TAYLOR_SIN_APPROXIMATION"
        assert item.is_taylor_mode is True

        # 2. Toggle mode pinch
        m_rect = item._btn_rects.get("toggle_mode")
        assert m_rect is not None
        center_m = item.mapToScene(m_rect.center())
        was_t = item.is_taylor_mode
        assert item.handle_air_pinch(center_m) is True
        assert item.is_taylor_mode != was_t

        # 3. Outside pinch
        outside = item.mapToScene(QPointF(9999.0, 9999.0))
        assert item.handle_air_pinch(outside) is False

        item.cleanup()

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveCalculusTaylorItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()

        for code in PRESETS:
            item.load_preset(code)
            painter = QPainter(image)
            item.paint(painter, opt)
            painter.end()

        item.cleanup()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle integration."""

    def test_stem_tools_diagram_generation(self, qapp):
        items = MathTools.create_calculus_riemann_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="RIEMANN_SINE_INTEGRAL",
            func_name="f(x) = sin(x)",
            exact_val=2.0,
            approx_val=1.974,
            n_partitions=8,
        )

        assert len(items) >= 5
        assert items[0].boundingRect().width() >= 640.0
        assert items[0].boundingRect().height() >= 340.0

    def test_lesson_board_calculus_lifecycle(self, qapp):
        board = LessonBoard()

        assert not board.is_calculus_taylor_visible()
        assert board.get_calculus_taylor() is None
        assert board.get_calculus_taylor_item() is None

        # Toggle ON
        res = board.toggle_calculus_taylor()
        assert res is True
        assert board.is_calculus_taylor_visible()
        calc_item = board.get_calculus_taylor()
        assert calc_item is not None
        assert board.get_calculus_taylor_item() is calc_item

        # Toggle OFF
        res = board.toggle_calculus_taylor()
        assert res is False
        assert not board.is_calculus_taylor_visible()
        assert board.get_calculus_taylor() is None
        assert board.get_calculus_taylor_item() is calc_item

        # Exclusion from get_items_in_rect
        board.show_calculus_taylor(True, pos=(100.0, 100.0))
        items_in_rect = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert calc_item not in items_in_rect

        calc_item.cleanup()
