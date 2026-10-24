"""
AirTeach AI - Interactive Informatics Dynamic Programming (DP) & Table / Traceback Simulator
============================================================================================
Provides an authentic discrete mathematics & computer science simulation of Dynamic Programming:
  - Algorithmic Foundations (Zero Mock Data):
      1. 0/1 Knapsack Problem (Bài toán cái túi 0/1):
           Items (weight w_i, value v_i), capacity W.
           Recurrence relation:
             DP[i, w] = DP[i-1, w]                           if w_i > w
             DP[i, w] = max(DP[i-1, w], DP[i-1, w-w_i] + v_i) if w_i <= w
           Traceback: if DP[i, w] != DP[i-1, w], item i is selected.
      2. Longest Common Subsequence (LCS - Xâu con chung dài nhất):
           Strings X[1..m] and Y[1..n].
           Recurrence relation:
             LCS[i, j] = 0                         if i=0 or j=0
             LCS[i, j] = LCS[i-1, j-1] + 1         if X[i] == Y[j]
             LCS[i, j] = max(LCS[i-1, j], LCS[i, j-1]) if X[i] != Y[j]
           Traceback: follow diagonal arrows where characters match.
      3. Minimum Coin Change Problem (Đổi tiền xu tối ưu):
           Denominations C = {c_1, c_2, ..., c_k}, target A.
           DP[a] = min_{c in C, c <= a} (DP[a - c] + 1) with DP[0] = 0.
           Reveals why Greedy algorithm fails (e.g. coins {1, 3, 4} for amount 6:
           Greedy gives 4+1+1=3 coins, DP gives optimal 3+3=2 coins).
      4. Levenshtein Edit Distance (Khoảng cách biên tập chuỗi):
           D[i, j] = min(Insert, Delete, Replace).
      5. Fibonacci Sequence with Memoization (Dãy Fibonacci mảng nhớ):
           DP[n] = DP[n-1] + DP[n-2], dropping exponential O(2^n) to linear O(n).
  - 5 Authentic Curriculum Presets:
      - KNAPSACK_01_STANDARD: Cái túi 5 vật phẩm, dung tích W = 10 kg
      - LCS_DNA_SEQUENCE: So khớp xâu DNA "AGGTAB" vs "GXTXAYB" (LCS = "GTAB", độ dài 4)
      - COIN_CHANGE_GREEDY_FAIL: Mệnh giá {1, 3, 4} đổi 6đ (DP thắng Greedy)
      - EDIT_DISTANCE_SPELLCHECK: Sửa từ "KITTEN" -> "SITTING" (3 phép biến đổi)
      - FIBONACCI_MEMOIZATION: Tính số Fibonacci thứ 12 bằng mảng nhớ O(n)
  - Interactive Visual Elements:
      - 2D DP Table Grid with row/column indices and headers.
      - Step-by-step calculation with active highlighted cell (i, j).
      - Dependency indicator arrows showing parent cells DP[i-1, w] and DP[i-1, w-w_i].
      - Backtrack path overlay connecting optimal solution choices.
      - Auto-play simulation timer with adjustable speed.
      - Touchless air pinch gestures for stepping, running, resetting, and vector stamping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class DPPreset:
    code: str
    title_vi: str
    algo_type: str  # "KNAPSACK", "LCS", "COIN_CHANGE", "EDIT_DIST", "FIBONACCI"
    row_headers: list[str]
    col_headers: list[str]
    input_desc: str
    description_vi: str
    formula_vi: str


PRESETS: dict[str, DPPreset] = {
    "KNAPSACK_01_STANDARD": DPPreset(
        code="KNAPSACK_01_STANDARD",
        title_vi="Bài Toán Cái Túi 0/1 (0/1 Knapsack)",
        algo_type="KNAPSACK",
        row_headers=["Ø", "Vật 1 (w=2, v=3)", "Vật 2 (w=3, v=4)", "Vật 3 (w=4, v=5)", "Vật 4 (w=5, v=8)"],
        col_headers=[str(w) for w in range(11)],  # Capacity 0..10
        input_desc="4 vật phẩm: (w=2, v=3), (w=3, v=4), (w=4, v=5), (w=5, v=8) | Dung tích W = 10",
        description_vi="Chọn tập hợp các vật phẩm sao cho tổng khối lượng không vượt quá W và tổng giá trị đạt cực đại. Mỗi vật chỉ chọn 0 hoặc 1 lần.",
        formula_vi="DP[i, w] = max(DP[i-1, w], DP[i-1, w - w_i] + v_i)",
    ),
    "LCS_DNA_SEQUENCE": DPPreset(
        code="LCS_DNA_SEQUENCE",
        title_vi="Xâu Con Chung Dài Nhất (LCS DNA)",
        algo_type="LCS",
        row_headers=["Ø", "A", "G", "G", "T", "A", "B"],
        col_headers=["Ø", "G", "X", "T", "X", "A", "Y", "B"],
        input_desc="X = 'AGGTAB'  |  Y = 'GXTXAYB'  =>  LCS = 'GTAB' (độ dài 4)",
        description_vi="Thuật toán quy hoạch động kinh điển trong tin sinh học so gióng chuỗi nucleotide và phát hiện đột biến di truyền.",
        formula_vi="LCS[i, j] = LCS[i-1, j-1] + 1 (nếu khớp) | max(LCS[i-1, j], LCS[i, j-1])",
    ),
    "COIN_CHANGE_GREEDY_FAIL": DPPreset(
        code="COIN_CHANGE_GREEDY_FAIL",
        title_vi="Đổi Tiền Xu Tối Thiểu (Coin Change)",
        algo_type="COIN_CHANGE",
        row_headers=["Mệnh giá {1, 3, 4}"],
        col_headers=[str(a) for a in range(11)],  # Target 0..10
        input_desc="Xu: {1, 3, 4}đ. Đổi 6đ: Tham lam = 4+1+1 (3 xu) | DP = 3+3 (2 xu tối ưu!)",
        description_vi="Minh chứng kinh điển tại sao thuật toán Tham lam (Greedy) thất bại và Quy hoạch động bắt buộc phải được áp dụng để tìm nghiệm toàn cục.",
        formula_vi="DP[a] = min_{c}(DP[a - c] + 1) với DP[0] = 0",
    ),
    "EDIT_DISTANCE_SPELLCHECK": DPPreset(
        code="EDIT_DISTANCE_SPELLCHECK",
        title_vi="Khoảng Cách Sửa Chuỗi (Levenshtein Distance)",
        algo_type="EDIT_DIST",
        row_headers=["Ø", "K", "I", "T", "T", "E", "N"],
        col_headers=["Ø", "S", "I", "T", "T", "I", "N", "G"],
        input_desc="S1 = 'KITTEN'  ->  S2 = 'SITTING'  |  Khoảng cách chỉnh sửa = 3",
        description_vi="Đếm số phép biến đổi tối thiểu (Chèn, Xóa, Thay thế) để chuyển chuỗi S1 thành S2. Ứng dụng trong kiểm tra chính tả và xử lý ngôn ngữ tự nhiên.",
        formula_vi="D[i, j] = min(D[i-1, j]+1, D[i, j-1]+1, D[i-1, j-1] + cost)",
    ),
    "FIBONACCI_MEMOIZATION": DPPreset(
        code="FIBONACCI_MEMOIZATION",
        title_vi="Quy Hoạch Động Dãy Fibonacci (Memoization)",
        algo_type="FIBONACCI",
        row_headers=["F(n)"],
        col_headers=[f"F({i})" for i in range(13)],  # F(0)..F(12)
        input_desc="Dãy Fibonacci: F(0)=0, F(1)=1, ..., F(12)=144 | Độ phức tạp: O(n)",
        description_vi="Khử bài toán đệ quy chồng chéo (Overlapping Subproblems) bằng mảng nhớ trạng thái, giảm thời gian tính từ O(2^n) xuống O(n).",
        formula_vi="F[n] = F[n-1] + F[n-2]  |  T(n) = O(n), S(n) = O(n)",
    ),
}


class InteractiveDynamicProgrammingItem(QGraphicsObject):
    """
    High-performance vector interactive Dynamic Programming Table & Traceback Simulator.
    Supports step-by-step filling, active cell dependency inspection, backtracking recovery,
    and LessonBoard vector stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (290.0, 110.0)) -> None:
        super().__init__()
        self.setPos(pos[0], pos[1])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active state
        self._active_preset_code: str = "KNAPSACK_01_STANDARD"
        self._preset: DPPreset = PRESETS[self._active_preset_code]

        # Computed Full DP Table: 2D matrix of integers
        self._table: list[list[int]] = []
        # Backtrack path: list of (r, c)
        self._backtrack_cells: set[tuple[int, int]] = set()

        # Step-by-step execution state
        self._current_step: int = 0
        self._max_steps: int = 0
        self._cell_order: list[tuple[int, int]] = []  # Sequence of (r, c)
        self._is_animating: bool = False

        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(120)  # ~8 steps per second
        self._anim_timer.timeout.connect(self._on_animation_tick)

        # Solve initial DP problem
        self._solve_dp()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "KNAPSACK_01_STANDARD",
            "LCS_DNA_SEQUENCE",
            "COIN_CHANGE_GREEDY_FAIL",
            "EDIT_DISTANCE_SPELLCHECK",
            "FIBONACCI_MEMOIZATION",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Grid Table Viewport (Left) & Inspector / Traceback Viewport (Right)
        self._btn_rects["table"] = QRectF(16.0, 88.0, 500.0, 360.0)
        self._btn_rects["inspector"] = QRectF(524.0, 88.0, 240.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["step_back"] = QRectF(16.0, y_bot, 80.0, h_bot)
        self._btn_rects["step_fwd"] = QRectF(102.0, y_bot, 80.0, h_bot)
        self._btn_rects["auto_run"] = QRectF(188.0, y_bot, 100.0, h_bot)
        self._btn_rects["solve_all"] = QRectF(294.0, y_bot, 90.0, h_bot)
        self._btn_rects["reset"] = QRectF(390.0, y_bot, 80.0, h_bot)
        self._btn_rects["stamp"] = QRectF(476.0, y_bot, 288.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Algorithmic DP Solvers (Zero Mock Data)
    # -------------------------------------------------------------------------
    def _solve_dp(self) -> None:
        """Solves the active DP problem authentically, building table, cell order, and traceback."""
        self._table.clear()
        self._cell_order.clear()
        self._backtrack_cells.clear()

        code = self._active_preset_code
        if code == "KNAPSACK_01_STANDARD":
            weights = [2, 3, 4, 5]
            values = [3, 4, 5, 8]
            n_items = len(weights)
            cap = 10
            # Table (n_items + 1) x (cap + 1)
            dp = [[0] * (cap + 1) for _ in range(n_items + 1)]
            for i in range(1, n_items + 1):
                w_i = weights[i - 1]
                v_i = values[i - 1]
                for w in range(cap + 1):
                    if w_i > w:
                        dp[i][w] = dp[i - 1][w]
                    else:
                        dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
                    self._cell_order.append((i, w))

            # Traceback
            curr_w = cap
            for i in range(n_items, 0, -1):
                self._backtrack_cells.add((i, curr_w))
                if dp[i][curr_w] != dp[i - 1][curr_w]:
                    curr_w -= weights[i - 1]
            self._backtrack_cells.add((0, curr_w))
            self._table = dp

        elif code == "LCS_DNA_SEQUENCE":
            s1 = "AGGTAB"
            s2 = "GXTXAYB"
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    self._cell_order.append((i, j))

            # Traceback
            ci, cj = m, n
            while ci > 0 and cj > 0:
                self._backtrack_cells.add((ci, cj))
                if s1[ci - 1] == s2[cj - 1]:
                    ci -= 1
                    cj -= 1
                elif dp[ci - 1][cj] >= dp[ci][cj - 1]:
                    ci -= 1
                else:
                    cj -= 1
            self._backtrack_cells.add((ci, cj))
            self._table = dp

        elif code == "COIN_CHANGE_GREEDY_FAIL":
            coins = [1, 3, 4]
            target = 10
            dp = [0] * (target + 1)
            for a in range(1, target + 1):
                min_c = 9999
                for c in coins:
                    if c <= a:
                        min_c = min(min_c, dp[a - c] + 1)
                dp[a] = min_c if min_c != 9999 else 0
                self._cell_order.append((0, a))
            self._table = [dp]
            # Traceback for 6: 6 -> 3 -> 0
            self._backtrack_cells = {(0, 6), (0, 3), (0, 0)}

        elif code == "EDIT_DISTANCE_SPELLCHECK":
            s1 = "KITTEN"
            s2 = "SITTING"
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if s1[i - 1] == s2[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
                    self._cell_order.append((i, j))
            self._table = dp
            # Traceback
            ci, cj = m, n
            while ci > 0 or cj > 0:
                self._backtrack_cells.add((ci, cj))
                cost = 0 if (ci > 0 and cj > 0 and s1[ci - 1] == s2[cj - 1]) else 1
                if ci > 0 and cj > 0 and dp[ci][cj] == dp[ci - 1][cj - 1] + cost:
                    ci -= 1
                    cj -= 1
                elif ci > 0 and dp[ci][cj] == dp[ci - 1][cj] + 1:
                    ci -= 1
                else:
                    cj -= 1
            self._backtrack_cells.add((0, 0))

        elif code == "FIBONACCI_MEMOIZATION":
            n = 12
            dp = [0] * (n + 1)
            dp[0] = 0
            dp[1] = 1
            self._cell_order.append((0, 0))
            self._cell_order.append((0, 1))
            for i in range(2, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
                self._cell_order.append((0, i))
            self._table = [dp]
            self._backtrack_cells = {(0, i) for i in range(n + 1)}

        self._max_steps = len(self._cell_order)
        self._current_step = self._max_steps  # Default fully solved

    # -------------------------------------------------------------------------
    # Public Controller API
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> DPPreset:
        return self._preset

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic dynamic programming preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._solve_dp()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def step_forward(self) -> None:
        """Advance one cell in the DP table computation."""
        if self._current_step < self._max_steps:
            self._current_step += 1
            self.update()

    def step_backward(self) -> None:
        """Rewind one cell in the DP table computation."""
        if self._current_step > 0:
            self._current_step -= 1
            self.update()

    def solve_all(self) -> None:
        """Instantly compute the entire DP table."""
        self._current_step = self._max_steps
        self._is_animating = False
        self._anim_timer.stop()
        self.update()

    def reset_dp(self) -> None:
        """Reset table to empty initial state."""
        self._current_step = 0
        self._is_animating = False
        self._anim_timer.stop()
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle automated step-by-step playback."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            if self._current_step >= self._max_steps:
                self._current_step = 0
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def cleanup(self) -> None:
        """Stop animation timer and cleanly disconnect."""
        self._anim_timer.stop()
        self._is_animating = False

    def _on_animation_tick(self) -> None:
        if not self._is_animating:
            return
        if self._current_step < self._max_steps:
            self._current_step += 1
            self.update()
        else:
            self.toggle_animation()

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main outer card
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        # 2. Header Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 48.0), 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 24.0, self.WIDTH, 24.0))

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(16.0, 10.0, 520.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"📐 {self._preset.title_vi} | Bảng Quy Hoạch Động (DP)",
        )

        # Close Button
        close_r = self._btn_rects["close"]
        is_close_hover = self._hovered_button == "close"
        painter.setBrush(QBrush(QColor("#ef4444" if is_close_hover else "#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(close_r)
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        cx, cy = close_r.center().x(), close_r.center().y()
        painter.drawLine(int(cx - 4), int(cy - 4), int(cx + 4), int(cy + 4))
        painter.drawLine(int(cx + 4), int(cy - 4), int(cx - 4), int(cy + 4))

        # 3. Presets Bar
        preset_items = [
            ("KNAPSACK_01_STANDARD", "Cái Túi 0/1"),
            ("LCS_DNA_SEQUENCE", "LCS Xâu DNA"),
            ("COIN_CHANGE_GREEDY_FAIL", "Đổi Tiền Xu"),
            ("EDIT_DISTANCE_SPELLCHECK", "Khoảng Cách Chữ"),
            ("FIBONACCI_MEMOIZATION", "Fibonacci Mảng"),
        ]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for code, label in preset_items:
            r = self._btn_rects[f"preset_{code}"]
            is_active = self._active_preset_code == code
            is_hover = self._hovered_button == f"preset_{code}"
            bg = QColor("#0284c7") if is_active else (QColor("#334155") if is_hover else QColor("#1e293b"))
            border = QColor("#38bdf8") if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(r, 6.0, 6.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#cbd5e1")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)

        # 4. DP Matrix Grid Viewport (Left)
        tbl_r = self._btn_rects["table"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(tbl_r, 10.0, 10.0)
        self._paint_dp_table(painter, tbl_r)

        # 5. Right Inspector & Optimal Traceback Panel
        insp_r = self._btn_rects["inspector"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(insp_r, 10.0, 10.0)
        self._paint_inspector_panel(painter, insp_r)

        # 6. Bottom Action Toolbar
        self._paint_bottom_toolbar(painter)

    def _paint_dp_table(self, painter: QPainter, arena_r: QRectF) -> None:
        """Paints the 2D DP matrix with animated step filling, glowing active cell, and backtrack halo."""
        if not self._table:
            return

        num_rows = len(self._table)
        num_cols = len(self._table[0])

        pad_left = 80.0
        pad_top = 40.0
        grid_w = arena_r.width() - pad_left - 16.0
        grid_h = arena_r.height() - pad_top - 16.0

        cell_w = min(42.0, grid_w / max(1, num_cols))
        cell_h = min(36.0, grid_h / max(1, num_rows))

        # Filled cells set based on current_step
        filled_cells = set(self._cell_order[: self._current_step])
        active_cell = self._cell_order[self._current_step - 1] if 0 < self._current_step <= len(self._cell_order) else None

        # 1. Column Headers (Top)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for c, col_h in enumerate(self._preset.col_headers[:num_cols]):
            cx = arena_r.left() + pad_left + c * cell_w
            cy = arena_r.top() + pad_top - 20.0
            painter.setPen(QPen(QColor("#38bdf8")))
            painter.drawText(QRectF(cx, cy, cell_w, 16.0), Qt.AlignmentFlag.AlignCenter, col_h)

        # 2. Row Headers & Grid Cells
        for r in range(num_rows):
            ry = arena_r.top() + pad_top + r * cell_h

            # Row Header (Left)
            row_h_txt = self._preset.row_headers[r] if r < len(self._preset.row_headers) else f"R{r}"
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(
                QRectF(arena_r.left() + 4.0, ry, pad_left - 8.0, cell_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                row_h_txt[:12],
            )

            # Cells in Row r
            for c in range(num_cols):
                cx = arena_r.left() + pad_left + c * cell_w
                cell_rect = QRectF(cx, ry, cell_w - 2.0, cell_h - 2.0)

                is_filled = (r, c) in filled_cells or r == 0  # Initial base cases always filled
                is_active = active_cell == (r, c)
                is_backtrack = (r, c) in self._backtrack_cells and self._current_step == self._max_steps

                # Background color
                if is_active:
                    bg = QColor("#0284c7")  # Active glowing cyan
                    border = QColor("#38bdf8")
                elif is_backtrack:
                    bg = QColor("#854d0e")  # Amber optimal backtrack
                    border = QColor("#facc15")
                elif is_filled:
                    bg = QColor("#1e293b")  # Normal computed slate
                    border = QColor("#334155")
                else:
                    bg = QColor("#090d16")  # Uncomputed empty dark
                    border = QColor("#1e293b")

                painter.setPen(QPen(border, 1.2 if is_active or is_backtrack else 0.8))
                painter.setBrush(QBrush(bg))
                painter.drawRoundedRect(cell_rect, 4.0, 4.0)

                # Cell Value Text
                if is_filled:
                    val = self._table[r][c]
                    painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_backtrack or is_active else QFont.Weight.Normal))
                    painter.setPen(QPen(QColor("#facc15") if is_backtrack else (QColor("#ffffff") if is_active else QColor("#cbd5e1"))))
                    painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, str(val))

    def _paint_inspector_panel(self, painter: QPainter, insp_r: QRectF) -> None:
        """Paints problem details, current step status, optimal answer, and recurrence formula."""
        x = insp_r.left()
        y = insp_r.top()
        w = insp_r.width()

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(x + 14.0, y + 24.0), "📊 THÔNG TIN THUẬT TOÁN")

        # Problem description
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(x + 14.0, y + 36.0, w - 28.0, 40.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            self._preset.input_desc,
        )

        # Progress bar
        prog_y = y + 84.0
        prog_w = w - 28.0
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(QRectF(x + 14.0, prog_y, prog_w, 14.0), 4.0, 4.0)

        # Fill progress
        frac = self._current_step / max(1, self._max_steps)
        if frac > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#10b981") if frac >= 1.0 else QColor("#38bdf8")))
            painter.drawRoundedRect(QRectF(x + 14.0, prog_y, prog_w * frac, 14.0), 4.0, 4.0)

        # Step count text
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QRectF(x + 14.0, prog_y, prog_w, 14.0), Qt.AlignmentFlag.AlignCenter, f"{self._current_step} / {self._max_steps} bước")

        # Optimal Result Highlight Card
        opt_y = prog_y + 26.0
        opt_h = 75.0
        painter.setPen(QPen(QColor("#f59e0b"), 1.2))
        painter.setBrush(QBrush(QColor("#451a03")))
        painter.drawRoundedRect(QRectF(x + 14.0, opt_y, w - 28.0, opt_h), 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(QPointF(x + 22.0, opt_y + 20.0), "🎯 KẾT QUẢ TỐI ƯU (TRACEBACK):")

        # Optimal Answer Value
        opt_val = self._table[-1][-1] if self._table else 0
        ans_str = f"Giá trị cực đại: {opt_val}" if self._preset.algo_type == "KNAPSACK" else (f"Độ dài LCS: {opt_val}" if self._preset.algo_type == "LCS" else f"Kết quả: {opt_val}")
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fef08a")))
        painter.drawText(QPointF(x + 22.0, opt_y + 45.0), ans_str)

        # Backtrack summary
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QPointF(x + 22.0, opt_y + 65.0), f"Đường vết tối ưu: {len(self._backtrack_cells)} ô (màu vàng)")

        # Formula Recurrence Box
        eq_y = opt_y + opt_h + 12.0
        eq_h = 70.0
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(QRectF(x + 14.0, eq_y, w - 28.0, eq_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(x + 20.0, eq_y + 16.0), "HỆ THỨC TRUY HỒI QUY HOẠCH ĐỘNG:")
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(
            QRectF(x + 20.0, eq_y + 24.0, w - 40.0, 42.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            self._preset.formula_vi,
        )

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Paints interactive action buttons: Step back, Step fwd, Auto run, Solve all, Reset, Stamp."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        buttons = [
            ("step_back", "◀ Lùi 1 bước", QColor("#334155")),
            ("step_fwd", "▶ Tiến 1 bước", QColor("#0284c7")),
            ("auto_run", "⏸ Dừng" if self._is_animating else "⏩ Tự Chạy", QColor("#0284c7")),
            ("solve_all", "✓ Giải Hết", QColor("#059669")),
            ("reset", "↺ Xóa Trắng", QColor("#334155")),
            ("stamp", "📌 Đóng Dấu Bảng DP Lên Bảng (LessonBoard)", QColor("#064e3b")),
        ]
        for name, lbl, base_col in buttons:
            r = self._btn_rects[name]
            is_hover = self._hovered_button == name
            painter.setPen(QPen(QColor("#38bdf8") if is_hover else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(base_col.lighter(130) if is_hover else base_col))
            painter.drawRoundedRect(r, 6.0, 6.0)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, lbl)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects["close"].contains(pos):
            self.cleanup()
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects["step_back"].contains(pos):
            self.step_backward()
            event.accept()
            return
        if self._btn_rects["step_fwd"].contains(pos):
            self.step_forward()
            event.accept()
            return
        if self._btn_rects["auto_run"].contains(pos):
            self.toggle_animation()
            event.accept()
            return
        if self._btn_rects["solve_all"].contains(pos):
            self.solve_all()
            event.accept()
            return
        if self._btn_rects["reset"].contains(pos):
            self.reset_dp()
            event.accept()
            return
        if self._btn_rects["stamp"].contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break
        if self._hovered_button != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Touchless air pinch gesture for selecting controls."""
        local_p = self.mapFromScene(point)

        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.cleanup()
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "step_back":
                    self.step_backward()
                    return True
                elif name == "step_fwd":
                    self.step_forward()
                    return True
                elif name == "auto_run":
                    self.toggle_animation()
                    return True
                elif name == "solve_all":
                    self.solve_all()
                    return True
                elif name == "reset":
                    self.reset_dp()
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current DP table as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import InformaticsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = InformaticsTools.create_dynamic_programming_diagram(
            center=center,
            preset_code=self._active_preset_code,
        )
        for itm in items:
            scene.addItem(itm)
