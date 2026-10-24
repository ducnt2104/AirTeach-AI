"""
AirTeach AI – Test Suite: Interactive Chemical Kinetics, Arrhenius Law & Catalysis Simulation
==============================================================================================
Verifies:
  - Exact Arrhenius equation: k = A * exp(-E_a / (R * T))
  - Catalysis rate acceleration ratio: Ratio = exp((E_a,uncat - E_a,cat) / (R * T))
  - Arrhenius plot linear slope: -E_a / R
  - Potential energy profiles E(q) and reaction enthalpy ΔH (Exothermic vs Endothermic)
  - Temperature effects and Van't Hoff scaling
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

from src.ui.interactive_reaction_kinetics import (
    InteractiveReactionKineticsItem,
    PRESETS,
    R_GAS,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestKineticsMathematics:
    """Verifies Arrhenius equations, catalysis speedup factors, and potential energy curves."""

    def test_arrhenius_rate_constant_analytic(self, qapp):
        item = InteractiveReactionKineticsItem()
        item.load_preset("EXOTHERMIC_CATALYSIS_DECOMPOSITION")
        item._has_catalyst = False  # Uncatalyzed

        # Ea = 75 kJ/mol = 75,000 J/mol, T = 298.15 K
        ea_j = 75000.0
        expected_k = 1.0e11 * math.exp(-ea_j / (R_GAS * 298.15))
        assert math.isclose(item.current_rate_constant, expected_k, rel_tol=1e-5)

        # Now enable catalyst: Ea drops to 28 kJ/mol = 28,000 J/mol
        item._has_catalyst = True
        ea_cat_j = 28000.0
        expected_k_cat = 1.0e11 * math.exp(-ea_cat_j / (R_GAS * 298.15))
        assert math.isclose(item.current_rate_constant, expected_k_cat, rel_tol=1e-5)

        # Rate with catalyst must be strictly greater
        assert item.current_rate_constant > expected_k

        item.cleanup()

    def test_catalysis_acceleration_ratio(self, qapp):
        item = InteractiveReactionKineticsItem()
        item.load_preset("EXOTHERMIC_CATALYSIS_DECOMPOSITION")

        # Delta Ea = 75 - 28 = 47 kJ/mol = 47,000 J/mol
        # Speedup = exp(47000 / (8.31446 * 298.15)) ≈ exp(18.96) ≈ 1.7e8 times!
        expected_speedup = math.exp(47000.0 / (R_GAS * 298.15))
        assert math.isclose(item.catalysis_speedup_ratio, expected_speedup, rel_tol=1e-4)
        assert item.catalysis_speedup_ratio > 1e7

        item.cleanup()

    def test_arrhenius_slope(self, qapp):
        item = InteractiveReactionKineticsItem()
        item.load_preset("ARRHENIUS_PLOT_LINEAR_FIT")

        # Uncatalyzed Ea = 85 kJ/mol -> slope = -85000 / 8.31446 ≈ -10223.1
        item._has_catalyst = False
        assert math.isclose(item.arrhenius_slope, -85000.0 / R_GAS, rel_tol=1e-4)

        # Catalyzed Ea = 45 kJ/mol -> slope = -45000 / 8.31446 ≈ -5412.3
        item._has_catalyst = True
        assert math.isclose(item.arrhenius_slope, -45000.0 / R_GAS, rel_tol=1e-4)

        item.cleanup()

    def test_potential_energy_profile_continuity(self, qapp):
        item = InteractiveReactionKineticsItem()
        item.load_preset("EXOTHERMIC_CATALYSIS_DECOMPOSITION")

        # Energy at reactants q = 0.0
        e_0 = item.energy_profile_val(0.0, is_cat=False)
        # Energy at products q = 1.0
        e_1 = item.energy_profile_val(1.0, is_cat=False)
        # For exothermic reaction, e_1 < e_0
        assert e_1 < e_0
        # The difference must closely match ΔH (-98 kJ/mol)
        assert math.isclose(e_1 - e_0, -98.0, abs_tol=2.0)

        # Peak at transition state q = 0.5 must be the maximum
        e_mid_uncat = item.energy_profile_val(0.5, is_cat=False)
        e_mid_cat = item.energy_profile_val(0.5, is_cat=True)

        assert e_mid_uncat > e_0
        assert e_mid_cat > e_0
        # Catalyzed peak must be lower than uncatalyzed peak
        assert e_mid_cat < e_mid_uncat

        item.cleanup()


class TestKineticsPresetsAndControls:
    """Verifies presets, catalyst toggle, temperature steppers, and air pinches."""

    def test_all_presets_load(self, qapp):
        item = InteractiveReactionKineticsItem()

        for code in PRESETS:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.boundingRect().width() == 700.0
            assert item.boundingRect().height() == 440.0
        item.cleanup()

    def test_temperature_and_catalyst_toggles(self, qapp):
        item = InteractiveReactionKineticsItem()

        # Catalyst toggle
        was_cat = item.has_catalyst
        item.toggle_catalyst()
        assert item.has_catalyst != was_cat

        # Temperature setting
        item.set_temperature_k(500.0)
        assert item.temp_k == 500.0
        assert math.isclose(item.temp_c, 500.0 - 273.15, rel_tol=1e-4)

        # Clamping
        item.set_temperature_k(100.0)
        assert item.temp_k == 200.0
        item.set_temperature_k(2000.0)
        assert item.temp_k == 900.0

        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        item = InteractiveReactionKineticsItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        # 1. Preset pinch
        p_rect = item._btn_rects.get("preset_ENDOTHERMIC_SYNTHESIS_HI")
        assert p_rect is not None
        center = item.mapToScene(p_rect.center())
        assert item.handle_air_pinch(center) is True
        assert item.active_preset.code == "ENDOTHERMIC_SYNTHESIS_HI"

        # 2. Toggle catalyst pinch
        cat_rect = item._btn_rects.get("toggle_catalyst")
        assert cat_rect is not None
        center_cat = item.mapToScene(cat_rect.center())
        was_cat = item.has_catalyst
        assert item.handle_air_pinch(center_cat) is True
        assert item.has_catalyst != was_cat

        # 3. Outside pinch
        outside = item.mapToScene(QPointF(9999.0, 9999.0))
        assert item.handle_air_pinch(outside) is False

        item.cleanup()

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveReactionKineticsItem()
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
        items = ChemistryTools.create_reaction_kinetics_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="EXOTHERMIC_CATALYSIS_DECOMPOSITION",
            reaction_str="2 H₂O₂ ➔ 2 H₂O + O₂",
            ea_uncat_kj=75.0,
            ea_cat_kj=28.0,
            delta_h_kj=-98.0,
            temp_k=298.15,
            has_catalyst=True,
        )

        assert len(items) >= 5
        assert items[0].boundingRect().width() >= 640.0
        assert items[0].boundingRect().height() >= 340.0

    def test_lesson_board_kinetics_lifecycle(self, qapp):
        board = LessonBoard()

        assert not board.is_reaction_kinetics_visible()
        assert board.get_reaction_kinetics() is None
        assert board.get_reaction_kinetics_item() is None

        # Toggle ON
        res = board.toggle_reaction_kinetics()
        assert res is True
        assert board.is_reaction_kinetics_visible()
        kin_item = board.get_reaction_kinetics()
        assert kin_item is not None
        assert board.get_reaction_kinetics_item() is kin_item

        # Toggle OFF
        res = board.toggle_reaction_kinetics()
        assert res is False
        assert not board.is_reaction_kinetics_visible()
        assert board.get_reaction_kinetics() is None
        assert board.get_reaction_kinetics_item() is kin_item

        # Exclusion from get_items_in_rect
        board.show_reaction_kinetics(True, pos=(100.0, 100.0))
        items_in_rect = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert kin_item not in items_in_rect

        kin_item.cleanup()
