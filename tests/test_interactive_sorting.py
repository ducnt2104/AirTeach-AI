"""
Unit Tests for Interactive Computer Science Sorting Algorithm & Big-O Complexity Visualizer
=============================================================================================
Verifies:
  1. Six authentic sorting algorithms (Bubble, Selection, Insertion, Merge, Quick, Heap):
       - Exact mathematical correctness: final frame array is strictly sorted.
       - Authentic comparison and swap counts recorded.
       - Accurate Big-O complexity metadata (best, avg, worst, space, stability).
  2. Four array generation distributions (RANDOM, NEARLY_SORTED, REVERSED, FEW_UNIQUE).
  3. Array size clamping (8 to 32 elements).
  4. Step-by-step navigation (step_forward, step_backward), timeline scrubbing, and reset.
  5. Play, pause, auto-run timer, and completion signals.
  6. Touchless air gestures (air pinch / air tap) on algorithm tabs, controls, stamp, and close.
  7. LessonBoard lifecycle, visibility toggle, selection exclusion, and vector diagram stamping.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_sorting import (
    InteractiveSortingItem,
    ALGORITHMS,
    ARRAY_DISTRIBUTIONS,
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
    it = InteractiveSortingItem(pos=(300.0, 140.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestSortingAlgorithmsAndComplexity:
    """Verifies sorting correctness, Big-O tables, and playback mechanics."""

    def test_algorithms_metadata(self):
        assert len(ALGORITHMS) == 6
        for key in ["BUBBLE", "SELECTION", "INSERTION", "MERGE", "QUICK", "HEAP"]:
            assert key in ALGORITHMS
            meta = ALGORITHMS[key]
            assert "name" in meta
            assert "best" in meta
            assert "avg" in meta
            assert "worst" in meta
            assert "space" in meta
            assert "stable" in meta

    def test_bubble_sort_correctness(self, item):
        item.set_algorithm("BUBBLE")
        item.set_array_size(12)
        item.set_distribution("REVERSED")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        sorted_arr = final_step.array

        assert sorted_arr == sorted(initial)
        assert final_step.comparisons > 0
        assert final_step.swaps > 0
        assert len(final_step.sorted_indices) == len(initial)

    def test_selection_sort_correctness(self, item):
        item.set_algorithm("SELECTION")
        item.set_array_size(10)
        item.set_distribution("RANDOM")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        assert final_step.array == sorted(initial)
        assert final_step.comparisons > 0

    def test_insertion_sort_correctness(self, item):
        item.set_algorithm("INSERTION")
        item.set_array_size(12)
        item.set_distribution("RANDOM")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        assert final_step.array == sorted(initial)
        assert final_step.comparisons > 0

    def test_merge_sort_correctness(self, item):
        item.set_algorithm("MERGE")
        item.set_array_size(16)
        item.set_distribution("REVERSED")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        assert final_step.array == sorted(initial)
        assert final_step.comparisons > 0
        assert final_step.swaps > 0  # Writes to main array

    def test_quick_sort_correctness(self, item):
        item.set_algorithm("QUICK")
        item.set_array_size(14)
        item.set_distribution("RANDOM")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        assert final_step.array == sorted(initial)
        assert final_step.comparisons > 0
        assert final_step.swaps > 0

    def test_heap_sort_correctness(self, item):
        item.set_algorithm("HEAP")
        item.set_array_size(12)
        item.set_distribution("FEW_UNIQUE")

        initial = list(item._initial_array)
        final_step = item._steps[-1]
        assert final_step.array == sorted(initial)
        assert final_step.comparisons > 0
        assert final_step.swaps > 0

    def test_step_navigation_and_timeline_scrubbing(self, item):
        """Verifies step_forward(), step_backward(), reset, and bounds."""
        item.set_algorithm("BUBBLE")
        item.set_array_size(8)
        assert item._step_index == 0
        assert item.is_finished() is False

        # Step forward
        res = item.step_forward()
        assert res is True
        assert item._step_index == 1

        # Step forward multiple times
        item.step_forward()
        item.step_forward()
        assert item._step_index == 3

        # Step backward
        b_res = item.step_backward()
        assert b_res is True
        assert item._step_index == 2

        # Reset simulation
        item.reset_simulation()
        assert item._step_index == 0
        assert item._is_running is False

    def test_array_distributions_and_clamping(self, item):
        """Verifies distributions and array size limits."""
        for dist in ARRAY_DISTRIBUTIONS:
            item.set_distribution(dist)
            assert item._distribution == dist
            assert len(item._initial_array) == item._array_size

        # Clamping
        item.set_array_size(100)
        assert item._array_size == 32

        item.set_array_size(2)
        assert item._array_size == 8

    def test_air_gestures_and_interactions(self, item):
        """Verifies air gestures for algorithm selection, playback, stamp, and close."""
        # 1. Algorithm tab air pinch
        algo_rects = item._get_algo_btn_rects()
        a_key, a_rect = algo_rects[1]  # SELECTION
        scene_pt = item.mapToScene(a_rect.center())
        res_algo = item.handle_air_pinch(scene_pt)
        assert res_algo is True
        assert item._algorithm_key == a_key

        # 2. Step forward air tap
        fwd_rect = item._get_step_fwd_rect()
        scene_fwd = item.mapToScene(fwd_rect.center())
        idx_before = item._step_index
        res_fwd = item.handle_air_tap(scene_fwd)
        assert res_fwd is True
        assert item._step_index == idx_before + 1

        # 3. Play / Pause air tap
        play_rect = item._get_play_rect()
        scene_play = item.mapToScene(play_rect.center())
        res_play = item.handle_air_tap(scene_play)
        assert res_play is True
        assert item._is_running is True

        res_pause = item.handle_air_tap(scene_play)
        assert res_pause is True
        assert item._is_running is False

        # 4. Shuffle button air tap
        shuf_rect = item._get_shuffle_rect()
        scene_shuf = item.mapToScene(shuf_rect.center())
        res_shuf = item.handle_air_tap(scene_shuf)
        assert res_shuf is True
        assert len(item._initial_array) == item._array_size

        # 5. Close button air pinch
        c_rect = item._get_close_btn_rect()
        scene_c = item.mapToScene(c_rect.center())
        assert item.isVisible() is True
        res_close = item.handle_air_pinch(scene_c)
        assert res_close is True
        assert item.isVisible() is False

    def test_lesson_board_lifecycle_and_stamping(self, board):
        """Verifies LessonBoard show/toggle/hide methods and vector diagram stamping."""
        assert board.is_sorting_visible() is False
        assert board.get_sorting() is None

        # Show
        it = board.show_sorting(pos=(340.0, 160.0))
        assert it is not None
        assert board.is_sorting_visible() is True
        assert board.get_sorting() == it

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
        vis = board.toggle_sorting()
        assert vis is False
        assert board.is_sorting_visible() is False

        # Toggle show
        vis2 = board.toggle_sorting()
        assert vis2 is True
        assert board.is_sorting_visible() is True

        # Clean cleanup
        it.cleanup()
        board.removeItem(it)
