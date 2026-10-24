"""
Unit Tests for Interactive Computer Science Finite State Machine (FSM) & Automata (DFA/NFA) Simulator
========================================================================================================
Verifies:
  1. Formal 5-tuple M = <Q, Σ, δ, q₀, F> definition and state model consistency.
  2. Four curriculum automata presets:
       - EVEN_ZEROS (recognizes even number of '0' bits).
       - MOD_3 (binary modulo 3 counter: accepts strings whose binary value mod 3 == 0).
       - ENDS_WITH_ABB (lexical pattern matching for suffix 'abb').
       - ODD_PARITY (parity bit error detection: accepts odd number of '1' bits).
  3. Step-by-step execution: step_forward, step_backward, history stack, tape reading head.
  4. Batch string testing (test_string) matching formal language specifications.
  5. Tape editing: append_symbol, backspace_symbol, clear_tape, set_input_string.
  6. Play, pause, auto-run timer, and reset simulation.
  7. Touchless air gestures (air pinch / air tap) on presets, tape chips, controls, and close.
  8. LessonBoard lifecycle, visibility toggle, selection exclusion, and vector diagram stamping.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_automata import (
    InteractiveAutomataItem,
    AutomatonState,
    AutomatonTransition,
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


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def item(board):
    it = InteractiveAutomataItem(pos=(300.0, 150.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestAutomataFormalLanguages:
    """Verifies formal language theory, transition functions, and automata execution."""

    def test_presets_exist_and_valid(self):
        assert "EVEN_ZEROS" in PRESETS
        assert "MOD_3" in PRESETS
        assert "ENDS_WITH_ABB" in PRESETS
        assert "ODD_PARITY" in PRESETS

        for key, p in PRESETS.items():
            assert "states" in p
            assert "transitions" in p
            assert "alphabet" in p
            assert len(p["states"]) >= 2
            # Must have at least one start state and one accept state
            assert any(s.is_start for s in p["states"])
            assert any(s.is_accept for s in p["states"])

    def test_even_zeros_dfa(self, item):
        """L = {w ∈ {0,1}* | |w|_0 is even}."""
        item.load_preset("EVEN_ZEROS")

        # Accepted test cases: even count of 0s
        assert item.test_string("") is True       # 0 zeros (even)
        assert item.test_string("111") is True    # 0 zeros (even)
        assert item.test_string("00") is True     # 2 zeros
        assert item.test_string("1001") is True   # 2 zeros
        assert item.test_string("1001100") is True # 4 zeros

        # Rejected test cases: odd count of 0s
        assert item.test_string("0") is False      # 1 zero
        assert item.test_string("101") is False    # 1 zero
        assert item.test_string("000") is False    # 3 zeros
        assert item.test_string("101001") is False # 3 zeros

    def test_mod_3_binary_counter(self, item):
        """L = {w ∈ {0,1}* | value(w) mod 3 == 0}."""
        item.load_preset("MOD_3")

        # Binary 0 -> 0 mod 3 == 0 (Accept)
        assert item.test_string("0") is True
        # Binary 3 ("11") -> 3 mod 3 == 0 (Accept)
        assert item.test_string("11") is True
        # Binary 6 ("110") -> 6 mod 3 == 0 (Accept)
        assert item.test_string("110") is True
        # Binary 9 ("1001") -> 9 mod 3 == 0 (Accept)
        assert item.test_string("1001") is True
        # Binary 12 ("1100") -> 12 mod 3 == 0 (Accept)
        assert item.test_string("1100") is True

        # Rejected test cases
        assert item.test_string("1") is False     # 1 mod 3 == 1
        assert item.test_string("10") is False    # 2 mod 3 == 2
        assert item.test_string("100") is False   # 4 mod 3 == 1
        assert item.test_string("101") is False   # 5 mod 3 == 2
        assert item.test_string("111") is False   # 7 mod 3 == 1

    def test_ends_with_abb_lexer(self, item):
        """L = {w ∈ {a,b}* | w ends with 'abb'}."""
        item.load_preset("ENDS_WITH_ABB")

        assert item.test_string("abb") is True
        assert item.test_string("aabb") is True
        assert item.test_string("babb") is True
        assert item.test_string("ababb") is True
        assert item.test_string("abbabb") is True

        # Rejected test cases
        assert item.test_string("ab") is False
        assert item.test_string("abba") is False
        assert item.test_string("abbab") is False
        assert item.test_string("b") is False
        assert item.test_string("a") is False

    def test_odd_parity_checker(self, item):
        """L = {w ∈ {0,1}* | |w|_1 is odd}."""
        item.load_preset("ODD_PARITY")

        assert item.test_string("1") is True
        assert item.test_string("010") is True
        assert item.test_string("10110") is True   # 3 ones (odd)
        assert item.test_string("11111") is True   # 5 ones (odd)

        assert item.test_string("0") is False      # 0 ones (even)
        assert item.test_string("11") is False     # 2 ones (even)
        assert item.test_string("1010") is False   # 2 ones (even)

    def test_step_by_step_execution_and_rewind(self, item):
        """Tests step_forward(), step_backward(), tape head position, and signals."""
        item.load_preset("EVEN_ZEROS")
        item.set_input_string("010")  # 3 symbols: '0', '1', '0'

        assert item._current_state_id == "q0"
        assert item._tape_index == 0
        assert item.is_finished() is False

        step_events = []
        item.step_executed.connect(lambda s, sym, t: step_events.append((s, sym, t)))

        # Step 1: Read '0' -> q0 to q1
        res1 = item.step_forward()
        assert res1 is True
        assert item._current_state_id == "q1"
        assert item._tape_index == 1
        assert len(step_events) == 1
        assert step_events[-1] == ("q0", "0", "q1")

        # Step 2: Read '1' -> q1 to q1 (self-loop)
        res2 = item.step_forward()
        assert res2 is True
        assert item._current_state_id == "q1"
        assert item._tape_index == 2

        # Rewind Step: step_backward()
        rewind_res = item.step_backward()
        assert rewind_res is True
        assert item._current_state_id == "q1"
        assert item._tape_index == 1

        # Step forward again
        item.step_forward()
        assert item._tape_index == 2

        # Step 3: Read '0' -> q1 to q0 (final symbol)
        res3 = item.step_forward()
        assert res3 is True
        assert item._current_state_id == "q0"
        assert item._tape_index == 3
        assert item.is_finished() is True
        assert item.is_accepted() is True  # 2 zeros -> Accept!

        # Step at end returns False
        assert item.step_forward() is False

    def test_tape_editing_and_reset(self, item):
        """Tests appending, backspacing, clearing, and reset."""
        item.load_preset("EVEN_ZEROS")
        item.clear_tape()
        assert item._input_string == ""

        item.append_symbol("0")
        item.append_symbol("1")
        item.append_symbol("0")
        assert item._input_string == "010"

        item.backspace_symbol()
        assert item._input_string == "01"

        item.set_input_string("001100")
        assert item._input_string == "001100"
        assert item._tape_index == 0

        # Step through partially, then reset
        item.step_forward()
        item.step_forward()
        assert item._tape_index == 2

        item.reset_simulation()
        assert item._tape_index == 0
        assert item._current_state_id == item._start_state

    def test_air_gestures_and_interactions(self, item):
        """Verifies air gestures for step, play/pause, reset, presets, stamp, and close."""
        # 1. Preset button air pinch
        preset_rects = item._get_preset_rects()
        p_key, p_rect = preset_rects[1]  # MOD_3
        scene_pt = item.mapToScene(p_rect.center())
        res_preset = item.handle_air_pinch(scene_pt)
        assert res_preset is True
        assert item._current_preset_key == p_key

        # 2. Step button air tap
        step_rect = item._get_step_btn_rect()
        scene_step = item.mapToScene(step_rect.center())
        idx_before = item._tape_index
        res_step = item.handle_air_tap(scene_step)
        assert res_step is True
        assert item._tape_index == idx_before + 1

        # 3. Play / Pause button air tap
        play_rect = item._get_play_btn_rect()
        scene_play = item.mapToScene(play_rect.center())
        res_play = item.handle_air_tap(scene_play)
        assert res_play is True
        assert item._is_running is True

        res_pause = item.handle_air_tap(scene_play)
        assert res_pause is True
        assert item._is_running is False

        # 4. Reset button air tap
        reset_rect = item._get_reset_btn_rect()
        scene_reset = item.mapToScene(reset_rect.center())
        res_reset = item.handle_air_tap(scene_reset)
        assert res_reset is True
        assert item._tape_index == 0

        # 5. Alphabet chip air tap
        chips = item._get_alphabet_btn_rects()
        chip_sym, chip_rect = chips[0]
        scene_chip = item.mapToScene(chip_rect.center())
        len_before = len(item._input_string)
        res_chip = item.handle_air_tap(scene_chip)
        assert res_chip is True
        assert len(item._input_string) == len_before + 1

        # 6. Close button air pinch
        c_rect = item._get_close_btn_rect()
        scene_c = item.mapToScene(c_rect.center())
        assert item.isVisible() is True
        res_close = item.handle_air_pinch(scene_c)
        assert res_close is True
        assert item.isVisible() is False

    def test_lesson_board_lifecycle_and_stamping(self, board):
        """Verifies LessonBoard show/toggle/hide methods and vector diagram stamping."""
        assert board.is_automata_visible() is False
        assert board.get_automata() is None

        # Show
        it = board.show_automata(pos=(350.0, 180.0))
        assert it is not None
        assert board.is_automata_visible() is True
        assert board.get_automata() == it

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
        vis = board.toggle_automata()
        assert vis is False
        assert board.is_automata_visible() is False

        # Toggle show
        vis2 = board.toggle_automata()
        assert vis2 is True
        assert board.is_automata_visible() is True

        # Clean cleanup
        it.cleanup()
        board.removeItem(it)
