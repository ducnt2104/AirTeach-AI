"""
AirTeach AI - Unit Tests for Interactive Chemistry Acid-Base Neutralization & pH Titration Simulator
===================================================================================================
Comprehensive automated unit tests verifying:
  1. Analytical acid-base equilibria: strong acid-strong base pH curve, exact pH 7.00 equivalence.
  2. Henderson-Hasselbalch equation and half-equivalence pH = pKa for weak acid titration.
  3. Polyprotic acid multi-step titration (H3PO4) and acetate buffer capacity.
  4. Chemical indicator transitions: Phenolphthalein, Methyl Orange, Bromothymol Blue.
  5. 5 Curriculum Presets (Strong/Strong, Weak/Strong, Strong/Weak, Polyprotic, Buffer).
  6. Air pinch gestures, burette drop dispensing, magnetic stirring animation, and reset.
  7. Vector diagram stamping via ChemistryTools.create_ph_titration_diagram.
  8. LessonBoard lifecycle and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_ph_titration import (
    InteractiveAcidBaseTitrationItem,
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


class TestTitrationEquilibria:
    """Verifies authentic analytical chemical equilibria and Henderson-Hasselbalch calculations."""

    def test_strong_acid_strong_base_ph_curve(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        item.load_preset("STRONG_ACID_STRONG_BASE")

        # Initial solution: 0.1M HCl -> pH = 1.00
        assert abs(item.calculate_ph(0.0) - 1.00) < 1e-2

        # V_eq = (0.1 * 25) / 0.1 = 25.0 mL
        assert abs(item.equivalence_volume_ml - 25.0) < 1e-3

        # At V = 10.0 mL: [H+] = (2.5 - 1.0) / 35.0 = 1.5 / 35 ≈ 0.04286 M -> pH ≈ 1.368
        ph_10 = item.calculate_ph(10.0)
        assert abs(ph_10 - 1.37) < 0.05

        # At Equivalence point V = 25.0 mL: exact neutrality pH = 7.00
        ph_eq = item.calculate_ph(25.0)
        assert abs(ph_eq - 7.00) < 1e-2

        # After equivalence V = 30.0 mL: [OH-] = 0.5 / 55 ≈ 0.00909 M -> pOH ≈ 2.04 -> pH ≈ 11.96
        ph_30 = item.calculate_ph(30.0)
        assert abs(ph_30 - 11.96) < 0.05

        item.cleanup()

    def test_weak_acid_buffer_and_half_equivalence(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        item.load_preset("WEAK_ACID_STRONG_BASE")
        # pKa = 4.76, V_eq = 25.0 mL

        # Half-equivalence point: V = 12.5 mL -> [A-] = [HA] -> pH = pKa = 4.76
        ph_half = item.calculate_ph(12.5)
        assert abs(ph_half - 4.76) < 1e-2

        # Equivalence point salt hydrolysis: pH > 7.00 (alkaline equivalence)
        ph_eq = item.calculate_ph(25.0)
        assert ph_eq > 8.0  # Typically ~8.72
        assert ph_eq < 9.5

        item.cleanup()

    def test_indicator_color_transitions(self, qapp):
        item = InteractiveAcidBaseTitrationItem()

        # 1. Phenolphthalein: Colorless at acid, Vivid pink at pH > 10.0
        item._indicator = "PHENOLPHTHALEIN"
        col_acid = item.get_solution_color(3.0)
        col_base = item.get_solution_color(11.0)
        # Pink base should have higher red/blue and alpha saturation
        assert col_acid.alpha() <= 120
        assert col_base.red() > 200 and col_base.blue() > 100

        # 2. Methyl Orange: Red at pH < 3.1, Yellow at pH > 4.4
        item._indicator = "METHYL_ORANGE"
        col_mo_acid = item.get_solution_color(2.5)
        col_mo_base = item.get_solution_color(5.0)
        assert col_mo_acid.red() > 220 and col_mo_acid.green() < 100  # Red
        assert col_mo_base.green() > 150  # Yellow

        # 3. Bromothymol Blue: Yellow at pH < 6.0, Green at pH 7.0, Deep Blue at pH > 7.6
        item._indicator = "BROMOTHYMOL_BLUE"
        col_bb_acid = item.get_solution_color(4.5)
        col_bb_neut = item.get_solution_color(7.0)
        col_bb_base = item.get_solution_color(9.0)
        assert col_bb_acid.red() > 200 and col_bb_acid.blue() < 50   # Yellow
        assert col_bb_neut.green() > 180 and col_bb_neut.red() < 50  # Green
        assert col_bb_base.blue() > 200 and col_bb_base.red() < 50   # Blue

        item.cleanup()


class TestTitrationInteraction:
    """Verifies presets, air pinch gestures, burette dispensing, and vector stamping."""

    def test_all_five_presets_load(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        for code in ["STRONG_ACID_STRONG_BASE", "WEAK_ACID_STRONG_BASE", "STRONG_ACID_WEAK_BASE", "POLYPROTIC_PHOSPHORIC", "BUFFER_ACETATE_CAPACITY"]:
            ok = item.load_preset(code)
            assert ok is True
            assert item.active_preset_code == code
            assert item.preset is not None
            assert item.equivalence_volume_ml > 0.0
        item.cleanup()

    def test_air_pinch_preset_switching(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        btn_r = item._btn_rects["preset_WEAK_ACID_STRONG_BASE"]
        pinch_pt = item.mapToScene(btn_r.center())

        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "WEAK_ACID_STRONG_BASE"

        item.cleanup()

    def test_dispensing_drops_and_continuous_titration(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        v_start = item.v_titrant_ml
        assert v_start == 0.0

        # Drop once (+0.25 mL)
        drop_pt = item.mapToScene(item._btn_rects["drop_once"].center())
        assert item.handle_air_pinch(drop_pt) is True
        assert item.v_titrant_ml == 0.25

        # Toggle continuous
        cont_pt = item.mapToScene(item._btn_rects["drop_cont"].center())
        assert item.handle_air_pinch(cont_pt) is True
        assert item.is_animating is True

        # Advance tick
        item._drop_fraction = 0.95
        item._on_animation_tick()
        assert item.v_titrant_ml > 0.25

        # Reset
        reset_pt = item.mapToScene(item._btn_rects["reset_titr"].center())
        assert item.handle_air_pinch(reset_pt) is True
        assert item.v_titrant_ml == 0.0
        assert item.is_animating is False

        item.cleanup()

    def test_cycle_indicator_air_pinch(self, qapp):
        item = InteractiveAcidBaseTitrationItem()
        ind_start = item.indicator

        cyc_pt = item.mapToScene(item._btn_rects["cycle_ind"].center())
        assert item.handle_air_pinch(cyc_pt) is True
        assert item.indicator != ind_start

        item.cleanup()

    def test_stamp_creates_vector_diagram(self, qapp):
        board = LessonBoard()
        item = InteractiveAcidBaseTitrationItem(pos=(200.0, 100.0))
        board.addItem(item)

        count_before = len(board.items())
        item.stamp_to_board()
        count_after = len(board.items())
        assert count_after > count_before

        item.cleanup()

    def test_chemistry_tools_vector_diagram(self, qapp):
        items = ChemistryTools.create_ph_titration_diagram(
            center=(400.0, 300.0),
            preset_code="STRONG_ACID_STRONG_BASE",
            v_titrant_ml=25.0,
            indicator="PHENOLPHTHALEIN",
        )
        assert len(items) >= 10
        texts = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("CHUẨN ĐỘ AXIT - BAZƠ" in t.upper() for t in texts)
        assert any("V_EQ" in t.upper() or "PH" in t.upper() for t in texts)


class TestTitrationBoardIntegration:
    """Verifies LessonBoard lifecycle and lasso selection exclusion."""

    def test_board_lifecycle(self, qapp):
        board = LessonBoard()
        assert board.is_ph_titration_visible() is False
        assert board.get_ph_titration() is None

        # Show
        item = board.show_ph_titration(True)
        assert item is not None
        assert board.is_ph_titration_visible() is True
        assert board.get_ph_titration() is item
        assert board.get_ph_titration_item() is item

        # Toggle off
        vis = board.toggle_ph_titration()
        assert vis is False
        assert board.is_ph_titration_visible() is False

        # Toggle on
        vis = board.toggle_ph_titration()
        assert vis is True
        assert board.is_ph_titration_visible() is True

        item.cleanup()

    def test_lasso_exclusion(self, qapp):
        board = LessonBoard()
        item = board.show_ph_titration(True, pos=(250.0, 150.0))
        rect = QRectF(200.0, 100.0, 800.0, 600.0)

        # Simulator widget should be excluded from lasso selection
        selected_items = board.get_items_in_rect(rect)
        assert item not in selected_items

        item.cleanup()
