"""
AirTeach AI - Interactive Computer Science Sorting Algorithm & Big-O Complexity Visualizer
===========================================================================================
Provides an authentic computer science algorithms visualizer:
  - 6 Fundamental Sorting Algorithms:
      1. Bubble Sort: O(n²) comparisons & swaps, O(1) space, Stable.
      2. Selection Sort: O(n²) comparisons, O(n) swaps, O(1) space, Unstable.
      3. Insertion Sort: O(n²) worst, O(n) best, O(1) space, Stable, Adaptive.
      4. Merge Sort: O(n log n) time, O(n) auxiliary buffer space, Stable, Divide & Conquer.
      5. Quick Sort: O(n log n) avg, O(n²) worst, O(log n) stack, Unstable, Lomuto partition.
      6. Heap Sort: O(n log n) guaranteed, O(1) in-place space, Unstable, Binary Max-Heap.
  - Interactive Features:
      - Real-time animated bar chart with glowing indices:
          • Yellow bars: Currently being compared (i, j).
          • Red / Cyan bars: Swapped / overwritten.
          • Green bars: Confirmed sorted in final positions.
      - Full timeline scrubbability: Step Forward (▶|), Step Backward (|◀), Run/Pause (▶/⏸), Reset (↺).
      - Array generator distributions: Random, Nearly Sorted, Reversed, Few Unique.
      - Live Telemetry HUD: Real comparison count, swap/write count, Big-O Theoretical Table, Progress %.
  - Touchless air gestures (air pinch / air tap) and permanent vector stamping to LessonBoard.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.ui.stem_tools import InformaticsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class SortStep:
    array: List[int]
    comparing: Tuple[int, int] = (-1, -1)
    swapping: Tuple[int, int] = (-1, -1)
    sorted_indices: Set[int] = field(default_factory=set)
    description: str = ""
    comparisons: int = 0
    swaps: int = 0


ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "BUBBLE": {
        "name": "Bubble Sort (Nổi bọt)",
        "best": "O(n)",
        "avg": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
        "stable": "Có (Stable)",
        "desc": "So sánh từng cặp liền kề và hoán đổi nếu sai thứ tự. Phần tử lớn nhất nổi dần về cuối.",
    },
    "SELECTION": {
        "name": "Selection Sort (Chọn trực tiếp)",
        "best": "O(n²)",
        "avg": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
        "stable": "Không (Unstable)",
        "desc": "Tìm phần tử nhỏ nhất trong đoạn chưa sắp xếp rồi đổi chỗ với vị trí đầu tiên.",
    },
    "INSERTION": {
        "name": "Insertion Sort (Chèn trực tiếp)",
        "best": "O(n)",
        "avg": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
        "stable": "Có (Stable)",
        "desc": "Xây dựng mảng đã sắp xếp dần bằng cách chèn từng phần tử vào đúng vị trí tiền tố.",
    },
    "MERGE": {
        "name": "Merge Sort (Trộn chia để trị)",
        "best": "O(n log n)",
        "avg": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
        "stable": "Có (Stable)",
        "desc": "Chia đôi mảng đệ quy thành các mảng con rồi trộn lại theo thứ tự tăng dần.",
    },
    "QUICK": {
        "name": "Quick Sort (Phân đoạn nhanh)",
        "best": "O(n log n)",
        "avg": "O(n log n)",
        "worst": "O(n²)",
        "space": "O(log n)",
        "stable": "Không (Unstable)",
        "desc": "Chọn chốt (pivot), phân hoạch mảng thành hai nửa nhỏ hơn và lớn hơn chốt.",
    },
    "HEAP": {
        "name": "Heap Sort (Vun đống cực đại)",
        "best": "O(n log n)",
        "avg": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(1)",
        "stable": "Không (Unstable)",
        "desc": "Dựng cây nhị phân Max-Heap, lần lượt đưa gốc lớn nhất về cuối mảng.",
    },
}

ARRAY_DISTRIBUTIONS = ["RANDOM", "NEARLY_SORTED", "REVERSED", "FEW_UNIQUE"]


class InteractiveSortingItem(QGraphicsObject):
    """
    Interactive Computer Science Sorting Algorithm & Big-O Complexity Visualizer Item.
    Provides step-by-step sorting traces, animated bar charts, comparisons telemetry,
    and formal algorithm analysis tables.
    """

    step_executed = Signal(int, int)  # (current_step, total_steps)
    algorithm_finished = Signal()
    closed = Signal()

    def __init__(self, pos: Tuple[float, float] = (300.0, 140.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])

        self._width = 630.0
        self._height = 435.0

        # Model Config
        self._algorithm_key: str = "BUBBLE"
        self._array_size: int = 16
        self._distribution: str = "RANDOM"
        self._initial_array: List[int] = []

        # Step Execution State
        self._steps: List[SortStep] = []
        self._step_index: int = 0
        self._is_running: bool = False

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(120)  # ~8 steps per second
        self._timer.timeout.connect(self._on_timer_tick)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(35.0)

        self.generate_new_array()

    # ── Geometry & Bounds ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Array Generation & Algorithms Traces ──────────────────────────────────

    def generate_new_array(self) -> None:
        """Generates raw array values based on selected size and distribution."""
        n = self._array_size
        if self._distribution == "RANDOM":
            vals = list(range(10, 10 + n * 5, 5))
            random.shuffle(vals)
        elif self._distribution == "NEARLY_SORTED":
            vals = list(range(10, 10 + n * 5, 5))
            # Swap 2 random adjacent pairs
            for _ in range(max(1, n // 6)):
                idx = random.randint(0, n - 2)
                vals[idx], vals[idx + 1] = vals[idx + 1], vals[idx]
        elif self._distribution == "REVERSED":
            vals = list(range(10 + (n - 1) * 5, 5, -5))
        elif self._distribution == "FEW_UNIQUE":
            unique_pool = [15, 35, 55, 75]
            vals = [random.choice(unique_pool) for _ in range(n)]
        else:
            vals = list(range(10, 10 + n * 5, 5))
            random.shuffle(vals)

        self._initial_array = vals
        self.compile_steps()

    def set_algorithm(self, algo_key: str) -> None:
        if algo_key in ALGORITHMS and algo_key != self._algorithm_key:
            self._algorithm_key = algo_key
            self.compile_steps()

    def set_distribution(self, dist_name: str) -> None:
        if dist_name in ARRAY_DISTRIBUTIONS and dist_name != self._distribution:
            self._distribution = dist_name
            self.generate_new_array()

    def set_array_size(self, size: int) -> None:
        clamped = max(8, min(32, int(size)))
        if clamped != self._array_size:
            self._array_size = clamped
            self.generate_new_array()

    # ── Full Trace Compilers (Zero Mock Data) ─────────────────────────────────

    def compile_steps(self) -> None:
        """Pre-computes authentic execution steps for the selected sorting algorithm."""
        self.pause()
        arr = list(self._initial_array)
        steps: List[SortStep] = []

        # Initial frame
        steps.append(SortStep(
            array=list(arr),
            comparing=(-1, -1),
            swapping=(-1, -1),
            sorted_indices=set(),
            description="Mảng ban đầu chưa sắp xếp.",
            comparisons=0,
            swaps=0,
        ))

        key = self._algorithm_key
        if key == "BUBBLE":
            self._trace_bubble_sort(arr, steps)
        elif key == "SELECTION":
            self._trace_selection_sort(arr, steps)
        elif key == "INSERTION":
            self._trace_insertion_sort(arr, steps)
        elif key == "MERGE":
            self._trace_merge_sort(arr, steps)
        elif key == "QUICK":
            self._trace_quick_sort(arr, steps)
        elif key == "HEAP":
            self._trace_heap_sort(arr, steps)

        # Final complete frame
        all_sorted = set(range(len(arr)))
        steps.append(SortStep(
            array=list(arr),
            comparing=(-1, -1),
            swapping=(-1, -1),
            sorted_indices=all_sorted,
            description="Mảng đã được sắp xếp hoàn toàn!",
            comparisons=steps[-1].comparisons if steps else 0,
            swaps=steps[-1].swaps if steps else 0,
        ))

        self._steps = steps
        self._step_index = 0
        self.update()

    def _trace_bubble_sort(self, a: List[int], steps: List[SortStep]) -> None:
        n = len(a)
        comp_count = 0
        swap_count = 0
        sorted_set: Set[int] = set()

        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                comp_count += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(j, j + 1),
                    swapping=(-1, -1),
                    sorted_indices=set(sorted_set),
                    description=f"So sánh a[{j}]={a[j]} và a[{j+1}]={a[j+1]}",
                    comparisons=comp_count,
                    swaps=swap_count,
                ))

                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    swap_count += 1
                    swapped = True
                    steps.append(SortStep(
                        array=list(a),
                        comparing=(-1, -1),
                        swapping=(j, j + 1),
                        sorted_indices=set(sorted_set),
                        description=f"Hoán đổi a[{j}] và a[{j+1}]",
                        comparisons=comp_count,
                        swaps=swap_count,
                    ))

            sorted_set.add(n - i - 1)
            if not swapped:
                for k in range(n - i):
                    sorted_set.add(k)
                break

    def _trace_selection_sort(self, a: List[int], steps: List[SortStep]) -> None:
        n = len(a)
        comp_count = 0
        swap_count = 0
        sorted_set: Set[int] = set()

        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                comp_count += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(min_idx, j),
                    swapping=(-1, -1),
                    sorted_indices=set(sorted_set),
                    description=f"Tìm min: so sánh a[{min_idx}]={a[min_idx]} với a[{j}]={a[j]}",
                    comparisons=comp_count,
                    swaps=swap_count,
                ))
                if a[j] < a[min_idx]:
                    min_idx = j

            if min_idx != i:
                a[i], a[min_idx] = a[min_idx], a[i]
                swap_count += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(-1, -1),
                    swapping=(i, min_idx),
                    sorted_indices=set(sorted_set),
                    description=f"Đưa min a[{min_idx}]={a[i]} về vị trí i={i}",
                    comparisons=comp_count,
                    swaps=swap_count,
                ))

            sorted_set.add(i)

        sorted_set.add(n - 1)

    def _trace_insertion_sort(self, a: List[int], steps: List[SortStep]) -> None:
        n = len(a)
        comp_count = 0
        swap_count = 0
        sorted_set: Set[int] = {0}

        for i in range(1, n):
            key = a[i]
            j = i - 1
            while j >= 0:
                comp_count += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(j, j + 1),
                    swapping=(-1, -1),
                    sorted_indices=set(sorted_set),
                    description=f"So sánh chèn: a[{j}]={a[j]} với key={key}",
                    comparisons=comp_count,
                    swaps=swap_count,
                ))
                if a[j] > key:
                    a[j + 1] = a[j]
                    swap_count += 1
                    steps.append(SortStep(
                        array=list(a),
                        comparing=(-1, -1),
                        swapping=(j, j + 1),
                        sorted_indices=set(sorted_set),
                        description=f"Dịch chuyển a[{j}] sang phải",
                        comparisons=comp_count,
                        swaps=swap_count,
                    ))
                    j -= 1
                else:
                    break
            a[j + 1] = key
            sorted_set.add(i)

    def _trace_merge_sort(self, a: List[int], steps: List[SortStep]) -> None:
        comp_count = [0]
        write_count = [0]
        sorted_set: Set[int] = set()

        def merge(left: int, mid: int, right: int):
            left_part = a[left:mid + 1]
            right_part = a[mid + 1:right + 1]
            i = 0
            j = 0
            k = left

            while i < len(left_part) and j < len(right_part):
                comp_count[0] += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(left + i, mid + 1 + j),
                    swapping=(-1, -1),
                    sorted_indices=set(sorted_set),
                    description=f"Trộn: so sánh {left_part[i]} và {right_part[j]}",
                    comparisons=comp_count[0],
                    swaps=write_count[0],
                ))

                if left_part[i] <= right_part[j]:
                    a[k] = left_part[i]
                    i += 1
                else:
                    a[k] = right_part[j]
                    j += 1
                write_count[0] += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(-1, -1),
                    swapping=(k, k),
                    sorted_indices=set(sorted_set),
                    description=f"Ghi a[{k}]={a[k]} vào mảng chính",
                    comparisons=comp_count[0],
                    swaps=write_count[0],
                ))
                k += 1

            while i < len(left_part):
                a[k] = left_part[i]
                write_count[0] += 1
                i += 1
                k += 1

            while j < len(right_part):
                a[k] = right_part[j]
                write_count[0] += 1
                j += 1
                k += 1

            if left == 0 and right == len(a) - 1:
                for idx in range(len(a)):
                    sorted_set.add(idx)

        def merge_sort_rec(left: int, right: int):
            if left < right:
                mid = (left + right) // 2
                merge_sort_rec(left, mid)
                merge_sort_rec(mid + 1, right)
                merge(left, mid, right)

        merge_sort_rec(0, len(a) - 1)

    def _trace_quick_sort(self, a: List[int], steps: List[SortStep]) -> None:
        comp_count = [0]
        swap_count = [0]
        sorted_set: Set[int] = set()

        def partition(low: int, high: int) -> int:
            pivot = a[high]
            i = low - 1
            for j in range(low, high):
                comp_count[0] += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(j, high),
                    swapping=(-1, -1),
                    sorted_indices=set(sorted_set),
                    description=f"So sánh a[{j}]={a[j]} với chốt pivot={pivot}",
                    comparisons=comp_count[0],
                    swaps=swap_count[0],
                ))
                if a[j] <= pivot:
                    i += 1
                    if i != j:
                        a[i], a[j] = a[j], a[i]
                        swap_count[0] += 1
                        steps.append(SortStep(
                            array=list(a),
                            comparing=(-1, -1),
                            swapping=(i, j),
                            sorted_indices=set(sorted_set),
                            description=f"Đổi chỗ a[{i}] và a[{j}]",
                            comparisons=comp_count[0],
                            swaps=swap_count[0],
                        ))
            a[i + 1], a[high] = a[high], a[i + 1]
            swap_count[0] += 1
            sorted_set.add(i + 1)
            steps.append(SortStep(
                array=list(a),
                comparing=(-1, -1),
                swapping=(i + 1, high),
                sorted_indices=set(sorted_set),
                description=f"Đưa pivot={pivot} về vị trí chính xác tại {i+1}",
                comparisons=comp_count[0],
                swaps=swap_count[0],
            ))
            return i + 1

        def quick_sort_rec(low: int, high: int):
            if low < high:
                pi = partition(low, high)
                quick_sort_rec(low, pi - 1)
                quick_sort_rec(pi + 1, high)
            elif low == high:
                sorted_set.add(low)

        quick_sort_rec(0, len(a) - 1)

    def _trace_heap_sort(self, a: List[int], steps: List[SortStep]) -> None:
        n = len(a)
        comp_count = [0]
        swap_count = [0]
        sorted_set: Set[int] = set()

        def heapify(length: int, root: int):
            largest = root
            left = 2 * root + 1
            right = 2 * root + 2

            if left < length:
                comp_count[0] += 1
                if a[left] > a[largest]:
                    largest = left

            if right < length:
                comp_count[0] += 1
                if a[right] > a[largest]:
                    largest = right

            if largest != root:
                a[root], a[largest] = a[largest], a[root]
                swap_count[0] += 1
                steps.append(SortStep(
                    array=list(a),
                    comparing=(-1, -1),
                    swapping=(root, largest),
                    sorted_indices=set(sorted_set),
                    description=f"Vun đống: hoán đổi a[{root}] và con lớn hơn a[{largest}]",
                    comparisons=comp_count[0],
                    swaps=swap_count[0],
                ))
                heapify(length, largest)

        # Build maxheap
        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)

        # Extract elements
        for i in range(n - 1, 0, -1):
            a[i], a[0] = a[0], a[i]
            swap_count[0] += 1
            sorted_set.add(i)
            steps.append(SortStep(
                array=list(a),
                comparing=(-1, -1),
                swapping=(0, i),
                sorted_indices=set(sorted_set),
                description=f"Đưa gốc Max={a[i]} về vị trí cuối đoạn a[{i}]",
                comparisons=comp_count[0],
                swaps=swap_count[0],
            ))
            heapify(i, 0)

        sorted_set.add(0)

    # ── Playback Controls ─────────────────────────────────────────────────────

    def step_forward(self) -> bool:
        if self._step_index < len(self._steps) - 1:
            self._step_index += 1
            self.step_executed.emit(self._step_index, len(self._steps))
            self.update()
            if self._step_index == len(self._steps) - 1:
                self.pause()
                self.algorithm_finished.emit()
            return True
        else:
            self.pause()
            return False

    def step_backward(self) -> bool:
        if self._step_index > 0:
            self.pause()
            self._step_index -= 1
            self.step_executed.emit(self._step_index, len(self._steps))
            self.update()
            return True
        return False

    def play(self) -> None:
        if not self._is_running:
            if self._step_index >= len(self._steps) - 1:
                self._step_index = 0
            self._is_running = True
            self._timer.start()
            self.update()

    def pause(self) -> None:
        if self._is_running:
            self._is_running = False
            self._timer.stop()
            self.update()

    def toggle_play_pause(self) -> None:
        if self._is_running:
            self.pause()
        else:
            self.play()

    def reset_simulation(self) -> None:
        self.pause()
        self._step_index = 0
        self.update()

    def _on_timer_tick(self) -> None:
        if not self.step_forward():
            self.pause()

    # ── State Queries ─────────────────────────────────────────────────────────

    def get_current_step(self) -> SortStep:
        if not self._steps:
            return SortStep(array=[])
        return self._steps[min(self._step_index, len(self._steps) - 1)]

    def is_finished(self) -> bool:
        return self._step_index >= len(self._steps) - 1

    def cleanup(self) -> None:
        if hasattr(self, "_timer") and self._timer.isActive():
            self._timer.stop()

    def close(self) -> None:
        self.cleanup()
        self.setVisible(False)
        self.closed.emit()

    # ── UI Rectangles ─────────────────────────────────────────────────────────

    def _get_close_btn_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 10.0, 22.0, 22.0)

    def _get_stamp_btn_rect(self) -> QRectF:
        return QRectF(self._width - 150.0, 10.0, 110.0, 24.0)

    def _get_algo_btn_rects(self) -> List[Tuple[str, QRectF]]:
        keys = list(ALGORITHMS.keys())
        rects = []
        x = 14.0
        y = 42.0
        w = 96.0
        h = 24.0
        for i, k in enumerate(keys):
            rects.append((k, QRectF(x + i * (w + 4.0), y, w, h)))
        return rects

    def _get_step_back_rect(self) -> QRectF:
        return QRectF(386.0, 386.0, 52.0, 32.0)

    def _get_play_rect(self) -> QRectF:
        return QRectF(444.0, 386.0, 58.0, 32.0)

    def _get_step_fwd_rect(self) -> QRectF:
        return QRectF(508.0, 386.0, 52.0, 32.0)

    def _get_reset_rect(self) -> QRectF:
        return QRectF(566.0, 386.0, 50.0, 32.0)

    def _get_shuffle_rect(self) -> QRectF:
        return QRectF(14.0, 386.0, 80.0, 32.0)

    def _get_dist_btn_rects(self) -> List[Tuple[str, QRectF]]:
        names = ["RANDOM", "NEARLY_SORTED", "REVERSED", "FEW_UNIQUE"]
        labels = ["Ngẫu nhiên", "Gần xong", "Đảo ngược", "Ít giá trị"]
        rects = []
        x = 100.0
        y = 388.0
        w = 66.0
        h = 28.0
        for i, (name, lbl) in enumerate(zip(names, labels)):
            rects.append((name, QRectF(x + i * (w + 4.0), y, w, h)))
        return rects

    # ── Painting & Rendering ──────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor("#090e1a"))
        grad.setColorAt(1.0, QColor("#050812"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Algorithm Tabs
        self._paint_algo_tabs(painter)

        # 4. Bar Chart Visualization Pane (Left)
        self._paint_bar_chart(painter)

        # 5. Diagnostic HUD & Big-O Complexity Table (Right)
        self._paint_diagnostic_hud(painter)

        # 6. Bottom Playback & Generator Controls
        self._paint_bottom_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 26, "📊 THUẬT TOÁN SẮP XẾP & ĐỘ PHỨC TẠP BIG-O (SORTING ALGORITHMS)")

        # Stamp Button
        s_rect = self._get_stamp_btn_rect()
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawRoundedRect(s_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu bảng")

        # Close Button
        c_rect = self._get_close_btn_rect()
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(c_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_algo_tabs(self, painter: QPainter) -> None:
        for key, rect in self._get_algo_btn_rects():
            a_data = ALGORITHMS[key]
            is_active = (key == self._algorithm_key)
            bg = QColor("#0369a1") if is_active else QColor("#1e293b")
            border = QColor("#38bdf8") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            short_title = a_data["name"].split()[0]
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, short_title)

    def _paint_bar_chart(self, painter: QPainter) -> None:
        chart_rect = QRectF(14.0, 74.0, 360.0, 302.0)
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(chart_rect, 10.0, 10.0)

        step = self.get_current_step()
        arr = step.array
        if not arr:
            return

        max_val = max(max(arr), 1)
        n = len(arr)
        total_w = chart_rect.width() - 20.0
        bar_w = max(4.0, total_w / n - 2.5)
        avail_h = chart_rect.height() - 44.0

        for i, val in enumerate(arr):
            bx = chart_rect.left() + 10.0 + i * (bar_w + 2.5)
            h = (val / max_val) * avail_h
            by = chart_rect.bottom() - 22.0 - h
            bar_rect = QRectF(bx, by, bar_w, h)

            # Color coding
            if i in step.sorted_indices:
                col = QColor("#22c55e")  # Sorted (Green)
            elif i in step.swapping:
                col = QColor("#ef4444")  # Swap / Overwrite (Red)
            elif i in step.comparing:
                col = QColor("#f59e0b")  # Comparing (Yellow)
            else:
                col = QColor("#0284c7")  # Default (Blue)

            painter.setBrush(QBrush(col))
            painter.setPen(QPen(col.lighter(130), 1.0))
            painter.drawRoundedRect(bar_rect, 2.5, 2.5)

            # Value label on top of bar if enough space
            if n <= 16:
                painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
                painter.setPen(QColor("#f8fafc"))
                painter.drawText(
                    QRectF(bx - 4.0, by - 14.0, bar_w + 8.0, 12.0),
                    Qt.AlignmentFlag.AlignCenter,
                    str(val),
                )

        # Status text below chart
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#cbd5e1"))
        desc = step.description if step.description else "Đang khởi tạo mảng..."
        painter.drawText(
            QRectF(chart_rect.left() + 10.0, chart_rect.bottom() - 18.0, chart_rect.width() - 20.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            desc,
        )

    def _paint_diagnostic_hud(self, painter: QPainter) -> None:
        hud_rect = QRectF(386.0, 74.0, 230.0, 302.0)
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(hud_rect.adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignTop, "📈 ĐỘ PHỨC TẠP & THỐNG KÊ")

        a_data = ALGORITHMS[self._algorithm_key]

        # Algorithm description
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            QRectF(hud_rect.left() + 12.0, hud_rect.top() + 32.0, 206.0, 42.0),
            Qt.TextFlag.TextWordWrap,
            a_data["desc"],
        )

        # Big-O Table Box
        table_y = hud_rect.top() + 78.0
        metrics = [
            ("Thời gian Tốt nhất (Best):", a_data["best"]),
            ("Thời gian Trung bình (Avg):", a_data["avg"]),
            ("Thời gian Xấu nhất (Worst):", a_data["worst"]),
            ("Bộ nhớ phụ (Space):", a_data["space"]),
            ("Tính Ổn định (Stability):", a_data["stable"]),
        ]

        curr_y = table_y
        for label, val in metrics:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(hud_rect.left() + 12.0, curr_y, label)

            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(
                QRectF(hud_rect.left() + 120.0, curr_y - 11.0, 95.0, 14.0),
                Qt.AlignmentFlag.AlignRight,
                val,
            )
            curr_y += 18.0

        # Divider
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawLine(hud_rect.left() + 10.0, curr_y + 4.0, hud_rect.right() - 10.0, curr_y + 4.0)
        curr_y += 16.0

        # Live Real-time Telemetry
        step = self.get_current_step()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(hud_rect.left() + 12.0, curr_y, "ĐẾM THAO TÁC THỰC TẾ:")
        curr_y += 18.0

        t_lines = [
            ("Số lần so sánh (C):", f"{step.comparisons}"),
            ("Số lần hoán đổi/ghi (S):", f"{step.swaps}"),
            ("Tiến độ thực hiện:", f"{self._step_index}/{max(1, len(self._steps) - 1)}"),
        ]
        for lbl, v in t_lines:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(hud_rect.left() + 12.0, curr_y, lbl)

            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f59e0b") if "so sánh" in lbl else QColor("#ef4444") if "hoán đổi" in lbl else QColor("#22c55e"))
            painter.drawText(
                QRectF(hud_rect.left() + 120.0, curr_y - 11.0, 95.0, 14.0),
                Qt.AlignmentFlag.AlignRight,
                v,
            )
            curr_y += 18.0

        # Progress Bar
        pb_rect = QRectF(hud_rect.left() + 12.0, hud_rect.bottom() - 20.0, 206.0, 8.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pb_rect, 4.0, 4.0)

        progress = self._step_index / max(1, len(self._steps) - 1)
        if progress > 0.0:
            fill_rect = QRectF(pb_rect.left(), pb_rect.top(), pb_rect.width() * progress, pb_rect.height())
            painter.setBrush(QBrush(QColor("#22c55e")))
            painter.drawRoundedRect(fill_rect, 4.0, 4.0)

    def _paint_bottom_controls(self, painter: QPainter) -> None:
        # Shuffle button
        shuf_r = self._get_shuffle_rect()
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawRoundedRect(shuf_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(shuf_r, Qt.AlignmentFlag.AlignCenter, "🔀 Xáo mảng")

        # Distribution buttons
        for name, rect in self._get_dist_btn_rects():
            is_active = (name == self._distribution)
            bg = QColor("#4338ca") if is_active else QColor("#1e293b")
            border = QColor("#818cf8") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            labels_dict = {"RANDOM": "Ngẫu nhiên", "NEARLY_SORTED": "Gần xong", "REVERSED": "Đảo ngược", "FEW_UNIQUE": "Ít giá trị"}
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, labels_dict[name])

        # Step Back
        b_rect = self._get_step_back_rect()
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawRoundedRect(b_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, "|◀ Lùi")

        # Play / Pause
        p_rect = self._get_play_rect()
        is_p = self._is_running
        painter.setBrush(QBrush(QColor("#16a34a") if not is_p else QColor("#d97706")))
        painter.setPen(QPen(QColor("#4ade80") if not is_p else QColor("#fcd34d"), 1.2))
        painter.drawRoundedRect(p_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(p_rect, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if is_p else "▶ Chạy")

        # Step Forward
        f_rect = self._get_step_fwd_rect()
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawRoundedRect(f_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(f_rect, Qt.AlignmentFlag.AlignCenter, "Tiến ▶|")

        # Reset
        r_rect = self._get_reset_rect()
        painter.setBrush(QBrush(QColor("#334155")))
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawRoundedRect(r_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_rect, Qt.AlignmentFlag.AlignCenter, "↺ Đầu")

    # ── Mouse Interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()

        # Close
        if self._get_close_btn_rect().contains(loc):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_btn_rect().contains(loc):
            self.stamp_to_board()
            event.accept()
            return

        # Algorithm selection
        for key, rect in self._get_algo_btn_rects():
            if rect.contains(loc):
                self.set_algorithm(key)
                event.accept()
                return

        # Shuffle
        if self._get_shuffle_rect().contains(loc):
            self.generate_new_array()
            event.accept()
            return

        # Distributions
        for name, rect in self._get_dist_btn_rects():
            if rect.contains(loc):
                self.set_distribution(name)
                event.accept()
                return

        # Playback controls
        if self._get_step_back_rect().contains(loc):
            self.step_backward()
            event.accept()
            return

        if self._get_play_rect().contains(loc):
            self.toggle_play_pause()
            event.accept()
            return

        if self._get_step_fwd_rect().contains(loc):
            self.step_forward()
            event.accept()
            return

        if self._get_reset_rect().contains(loc):
            self.reset_simulation()
            event.accept()
            return

        super().mousePressEvent(event)

    # ── Touchless Air Gesture Handlers ────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_btn_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_btn_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Algorithm selection
        for key, rect in self._get_algo_btn_rects():
            if rect.contains(loc):
                self.set_algorithm(key)
                return True

        # Shuffle
        if self._get_shuffle_rect().contains(loc):
            self.generate_new_array()
            return True

        # Distributions
        for name, rect in self._get_dist_btn_rects():
            if rect.contains(loc):
                self.set_distribution(name)
                return True

        # Playback controls
        if self._get_step_back_rect().contains(loc):
            self.step_backward()
            return True

        if self._get_play_rect().contains(loc):
            self.toggle_play_pause()
            return True

        if self._get_step_fwd_rect().contains(loc):
            self.step_forward()
            return True

        if self._get_reset_rect().contains(loc):
            self.reset_simulation()
            return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── LessonBoard Permanent Vector Stamping ─────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of current sorting state onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        step = self.get_current_step()
        diagram_items = InformaticsTools.create_sorting_diagram(
            center=(cx, cy + 30.0),
            width=540.0,
            height=320.0,
            algorithm=self._algorithm_key,
            array=list(step.array),
            comparisons=step.comparisons,
            swaps=step.swaps,
        )
        for it in diagram_items:
            board.addItem(it)

        if hasattr(board, "commit_state_action"):
            board.commit_state_action("Stamp Sorting Algorithm Diagram")
        logger.info(f"Sorting diagram stamped to LessonBoard at ({cx:.1f}, {cy:.1f}).")
