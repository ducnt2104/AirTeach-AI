"""
AirTeach AI - Unit Tests for Interactive Informatics Dynamic Programming Simulator
==================================================================================
Comprehensive automated unit tests verifying:
  1. Authentic dynamic programming algorithms: 0/1 Knapsack, Longest Common Subsequence (LCS),
     Coin Change optimal vs greedy, Levenshtein Edit Distance, and Fibonacci memoization.
  2. Optimal solution values and backtracking recovery paths.
  3. 5 Curriculum Presets (Knapsack 0/1, LCS DNA, Coin Change, Edit Distance, Fibonacci).
  4. Air pinch gesture interactions, stepping forward/backward, auto-play, and reset.
  5. Vector diagram stamping via InformaticsTools.create_dynamic_programming_diagram.
  6. LessonBoard lifecycle and lasso selection exclusion.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_dynamic_programming import (
    InteractiveDynamicProgrammingItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import InformaticsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestDPSolversAndAlgorithms:
    """Verifies authentic recurrence calculations and optimal results across DP algorithms."""

    def test_knapsack_01_optimal_value_and_traceback(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.load_preset("KNAPSACK_01_STANDARD")

        # Table dimensions: (4+1) rows x (10+1) cols
        assert len(item._table) == 5
        assert len(item._table[0]) == 11

        # Items: (w=2, v=3), (w=3, v=4), (w=4, v=5), (w=5, v=8), W=10
        # Best combination: Items 1, 2, 4 -> weights 2+3+5 = 10, value = 3+4+8 = 15
        max_val = item._table[-1][-1]
        assert max_val == 15

        # Backtrack path must connect (4, 10) back to base (0, 0)
        assert (4, 10) in item._backtrack_cells
        assert (0, 0) in item._backtrack_cells
        assert len(item._backtrack_cells) >= 4

        item.cleanup()

    def test_lcs_dna_sequence(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.load_preset("LCS_DNA_SEQUENCE")

        # "AGGTAB" vs "GXTXAYB" -> LCS is "GTAB", length 4
        assert item._table[-1][-1] == 4
        assert (len("AGGTAB"), len("GXTXAYB")) in item._backtrack_cells

        item.cleanup()

    def test_coin_change_optimal_vs_greedy(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.load_preset("COIN_CHANGE_GREEDY_FAIL")

        # Coins {1, 3, 4}
        # For target 6:
        # Greedy would pick 4 + 1 + 1 (3 coins)
        # Optimal DP picks 3 + 3 (2 coins)
        min_coins_for_6 = item._table[0][6]
        assert min_coins_for_6 == 2

        # For target 10: 4 + 3 + 3 = 10 (3 coins)
        assert item._table[0][10] == 3

        item.cleanup()

    def test_edit_distance_spellcheck(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.load_preset("EDIT_DISTANCE_SPELLCHECK")

        # "KITTEN" -> "SITTING": distance is 3
        # K->S (replace), E->I (replace), +G (insert)
        assert item._table[-1][-1] == 3

        item.cleanup()

    def test_fibonacci_memoization(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.load_preset("FIBONACCI_MEMOIZATION")

        # F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13, F(8)=21, F(9)=34, F(10)=55, F(11)=89, F(12)=144
        fib_table = item._table[0]
        assert fib_table[0] == 0
        assert fib_table[1] == 1
        assert fib_table[6] == 8
        assert fib_table[12] == 144

        item.cleanup()


class TestDPInteraction:
    """Verifies presets, air pinch gestures, stepping execution, and vector stamping."""

    def test_all_five_presets_load(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        for code in ["KNAPSACK_01_STANDARD", "LCS_DNA_SEQUENCE", "COIN_CHANGE_GREEDY_FAIL", "EDIT_DISTANCE_SPELLCHECK", "FIBONACCI_MEMOIZATION"]:
            ok = item.load_preset(code)
            assert ok is True
            assert item.active_preset_code == code
            assert item.preset is not None
            assert len(item._table) > 0
            assert item.max_steps > 0
        item.cleanup()

    def test_air_pinch_preset_switch(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        btn_r = item._btn_rects["preset_LCS_DNA_SEQUENCE"]
        pinch_pt = item.mapToScene(btn_r.center())

        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "LCS_DNA_SEQUENCE"

        item.cleanup()

    def test_stepping_and_reset(self, qapp):
        item = InteractiveDynamicProgrammingItem()
        item.reset_dp()
        assert item.current_step == 0

        # Step forward
        item.step_forward()
        assert item.current_step == 1

        # Step backward
        item.step_backward()
        assert item.current_step == 0

        # Solve all
        item.solve_all()
        assert item.current_step == item.max_steps

        # Auto run toggle
        is_anim = item.toggle_animation()
        assert is_anim is True
        item._on_animation_tick()
        assert item.current_step >= 1

        item.cleanup()

    def test_stamp_creates_vector_diagram(self, qapp):
        board = LessonBoard()
        item = InteractiveDynamicProgrammingItem(pos=(200.0, 100.0))
        board.addItem(item)

        count_before = len(board.items())
        item.stamp_to_board()
        count_after = len(board.items())
        assert count_after > count_before

        item.cleanup()

    def test_informatics_tools_vector_diagram(self, qapp):
        items = InformaticsTools.create_dynamic_programming_diagram(
            center=(400.0, 300.0),
            preset_code="KNAPSACK_01_STANDARD",
        )
        assert len(items) >= 10
        texts = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("QUY HOẠCH ĐỘNG" in t.upper() or "DP" in t.upper() for t in texts)
        assert any("TRUY HỒI" in t.upper() or "KNAPSACK" in t.upper() for t in texts)


class TestDPBoardIntegration:
    """Verifies LessonBoard lifecycle and lasso selection exclusion."""

    def test_board_lifecycle(self, qapp):
        board = LessonBoard()
        assert board.is_dynamic_programming_visible() is False
        assert board.get_dynamic_programming() is None

        # Show
        item = board.show_dynamic_programming(True)
        assert item is not None
        assert board.is_dynamic_programming_visible() is True
        assert board.get_dynamic_programming() is item
        assert board.get_dynamic_programming_item() is item

        # Toggle off
        vis = board.toggle_dynamic_programming()
        assert vis is False
        assert board.is_dynamic_programming_visible() is False

        # Toggle on
        vis = board.toggle_dynamic_programming()
        assert vis is True
        assert board.is_dynamic_programming_visible() is True

        item.cleanup()

    def test_lasso_exclusion(self, qapp):
        board = LessonBoard()
        item = board.show_dynamic_programming(True, pos=(250.0, 150.0))
        rect = QRectF(200.0, 100.0, 800.0, 600.0)

        # Simulator widget should be excluded from lasso selection
        selected_items = board.get_items_in_rect(rect)
        assert item not in selected_items

        item.cleanup()
