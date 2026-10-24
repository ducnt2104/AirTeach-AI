"""
AirTeach AI - Interactive Informatics Binary Tree & Huffman Encoding Simulator
=============================================================================
Provides an authentic interactive computer science simulation for high school & university algorithms:
  - Theoretical & Algorithmic Foundations (Zero Mock Data):
      1. Binary Search Tree (Cây nhị phân tìm kiếm - BST):
           - Invariant: Key(Left) < Key(Node) <= Key(Right)
           - Insertion & Search in O(h) time, where h is tree height (O(log n) balanced, O(n) degenerate)
           - Tree Traversals:
               * In-Order (L -> Node -> R): Strictly ascending sorted keys!
               * Pre-Order (Node -> L -> R): Prefix structure serialization
               * Post-Order (L -> R -> Node): Bottom-up evaluation / memory free
               * Level-Order (BFS queue): Breadth-first breadth layers
           - Metrics: Height h, Balance Factor BF = h_L - h_R, AVL balance check (|BF| <= 1).
      2. Huffman Coding & Information Theory (Mã hóa Huffman & Lý thuyết thông tin):
           - Greedy optimal prefix tree construction via priority queue (min-heap)
           - Prefix-free property: No codeword is a prefix of another
           - Shannon Entropy: H(X) = -sum(p_i * log2(p_i)) bits/symbol
           - Average Codeword Length: L_avg = sum(p_i * length_i)
           - Redundancy R = L_avg - H(X) >= 0; Huffman Efficiency eta = H(X) / L_avg * 100%
           - Compression Ratio vs 8-bit ASCII: (1 - L_avg / 8) * 100%
      3. Arithmetic Expression Tree (Cây biểu thức số học):
           - Operators as internal nodes, operands as leaves
           - In-order traversal reconstructs infix algebraic expression
           - Post-order traversal reconstructs Reverse Polish Notation (RPN / Postfix)
  - 5 Authentic Curriculum Presets:
      - BST_BALANCED_SEARCH: Cây BST cân bằng [50, 30, 70, 20, 40, 60, 80]
      - BST_DEGENERATE_UNBALANCED: Cây suy biến [10, 20, 30, 40, 50] (Trường hợp xấu nhất O(n))
      - HUFFMAN_TEXT_COMPRESSION: Nén văn bản "AIRTEACH AI ROBOTICS" bằng cây Huffman
      - HUFFMAN_DNA_GENOME: Cây mã hóa 4 nucleotide (A, T, G, C) theo tần số sinh học
      - ARITHMETIC_EXPRESSION: Cây biểu thức số học ((A + B) * (C - D) / E)
  - Classroom Controls & Gestures:
      - Traversal step-by-step animation (In-order, Pre-order, Post-order, Level-order)
      - Search key input with visual path highlighting and comparison counter
      - Touchless air pinch and drag gesture support
      - Vector diagram stamping onto LessonBoard
"""
from __future__ import annotations

import heapq
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models: Tree Nodes & Presets
# -----------------------------------------------------------------------------
@dataclass
class TreeNode:
    """Authentic binary tree node with 2D rendering layout coordinates."""
    key: int
    val: str = ""
    left: Optional[TreeNode] = None
    right: Optional[TreeNode] = None
    x: float = 0.0
    y: float = 0.0
    freq: int = 1
    code: str = ""
    is_operator: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def height(self) -> int:
        hl = self.left.height() if self.left else 0
        hr = self.right.height() if self.right else 0
        return 1 + max(hl, hr)

    def size(self) -> int:
        sl = self.left.size() if self.left else 0
        sr = self.right.size() if self.right else 0
        return 1 + sl + sr

    def balance_factor(self) -> int:
        hl = self.left.height() if self.left else 0
        hr = self.right.height() if self.right else 0
        return hl - hr


@dataclass(order=True)
class HuffmanHeapItem:
    """Min-heap priority item for Huffman greedy tree assembly."""
    freq: int
    count_id: int
    node: TreeNode = field(compare=False)


@dataclass
class TreePreset:
    code: str
    title_vi: str
    tree_type: str  # "BST", "HUFFMAN", "EXPR"
    description_vi: str
    data_source: Any


PRESETS: dict[str, TreePreset] = {
    "BST_BALANCED_SEARCH": TreePreset(
        code="BST_BALANCED_SEARCH",
        title_vi="Cây BST Cân Bằng (Balanced BST Search)",
        tree_type="BST",
        description_vi="Cây nhị phân tìm kiếm chuẩn với cấu trúc đối xứng. Tìm kiếm, chèn và xóa có độ phức tạp thời gian trung bình O(log n).",
        data_source=[50, 30, 70, 20, 40, 60, 80],
    ),
    "BST_DEGENERATE_UNBALANCED": TreePreset(
        code="BST_DEGENERATE_UNBALANCED",
        title_vi="Cây Suy Biến Lệch (Degenerate Worst-Case)",
        tree_type="BST",
        description_vi="Cây nhị phân bị suy biến thành danh sách liên kết đơn khi chèn dãy khóa đã sắp xếp. Chiều cao h = n, độ phức tạp O(n).",
        data_source=[10, 20, 30, 40, 50, 60],
    ),
    "HUFFMAN_TEXT_COMPRESSION": TreePreset(
        code="HUFFMAN_TEXT_COMPRESSION",
        title_vi="Nén Văn Bản Huffman (Optimal Prefix Code)",
        tree_type="HUFFMAN",
        description_vi="Mã hóa tiền tố tối ưu cho thông điệp 'AIRTEACH AI ROBOTICS'. Ký tự xuất hiện nhiều có mã bit ngắn nhất.",
        data_source="AIRTEACH AI ROBOTICS",
    ),
    "HUFFMAN_DNA_GENOME": TreePreset(
        code="HUFFMAN_DNA_GENOME",
        title_vi="Mã Hóa Bộ Gen DNA (Nucleotide A-T-G-C)",
        tree_type="HUFFMAN",
        description_vi="Mã hóa 4 bazơ nucleotide với tần suất sinh học thực: A (35%), T (30%), G (20%), C (15%).",
        data_source={"A": 35, "T": 30, "G": 20, "C": 15},
    ),
    "ARITHMETIC_EXPRESSION": TreePreset(
        code="ARITHMETIC_EXPRESSION",
        title_vi="Cây Biểu Thức Số Học ((A + B) * (C - D) / E)",
        tree_type="EXPR",
        description_vi="Toán tử là nút trong, toán hạng là nút lá. Duyệt trung tố tái tạo biểu thức đại số, hậu tố tạo mã Ba Lan nghịch đảo.",
        data_source="EXPR",
    ),
}


# -----------------------------------------------------------------------------
# Interactive Binary Tree Graphics Object
# -----------------------------------------------------------------------------
class InteractiveBinaryTreeItem(QGraphicsObject):
    """
    High-performance vector interactive Binary Tree & Huffman Simulator.
    Supports touchless air gestures, traversal animations, and LessonBoard stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    traversal_completed = Signal(str, list)
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
        self._active_preset_code: str = "BST_BALANCED_SEARCH"
        self._root: Optional[TreeNode] = None
        self._all_nodes: list[TreeNode] = []

        # Traversal & Search State
        self._active_traversal_mode: str = "IN_ORDER"
        self._traversal_sequence: list[TreeNode] = []
        self._traversal_index: int = -1
        self._is_animating: bool = False
        self._search_key: int = 40
        self._search_path: list[TreeNode] = []
        self._search_found: bool = False
        self._search_comparisons: int = 0

        # Animation Timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(400)  # 400ms per traversal step
        self._anim_timer.timeout.connect(self._on_traversal_step)

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

        # Build initial preset tree
        self.load_preset(self._active_preset_code)

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "BST_BALANCED_SEARCH",
            "BST_DEGENERATE_UNBALANCED",
            "HUFFMAN_TEXT_COMPRESSION",
            "HUFFMAN_DNA_GENOME",
            "ARITHMETIC_EXPRESSION",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Main tree drawing canvas viewport
        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 470.0, 360.0)

        # Traversal buttons toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["btn_inorder"] = QRectF(16.0, y_bot, 96.0, h_bot)
        self._btn_rects["btn_preorder"] = QRectF(118.0, y_bot, 96.0, h_bot)
        self._btn_rects["btn_postorder"] = QRectF(220.0, y_bot, 96.0, h_bot)
        self._btn_rects["btn_levelorder"] = QRectF(322.0, y_bot, 96.0, h_bot)
        self._btn_rects["btn_search"] = QRectF(424.0, y_bot, 110.0, h_bot)
        self._btn_rects["stamp"] = QRectF(542.0, y_bot, 222.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Core Algorithms: Tree Construction & Layout (Zero Mock)
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Loads and builds the authentic binary tree / Huffman tree for a preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        preset = PRESETS[preset_code]
        self._anim_timer.stop()
        self._is_animating = False
        self._traversal_sequence = []
        self._traversal_index = -1
        self._search_path = []

        if preset.tree_type == "BST":
            self._root = None
            for key in preset.data_source:
                self._insert_bst(key)
        elif preset.tree_type == "HUFFMAN":
            self._build_huffman_tree(preset.data_source)
        elif preset.tree_type == "EXPR":
            self._build_expression_tree()

        # Compute geometric 2D layout coordinates for rendering
        self._compute_layout()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def _insert_bst(self, key: int) -> None:
        """Authentic BST key insertion maintaining invariant."""
        new_node = TreeNode(key=key, val=str(key))
        if self._root is None:
            self._root = new_node
            return

        curr = self._root
        while True:
            if key < curr.key:
                if curr.left is None:
                    curr.left = new_node
                    break
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = new_node
                    break
                curr = curr.right

    def _build_huffman_tree(self, source: Any) -> None:
        """Construct authentic Huffman optimal prefix code tree via min-heap."""
        freq_map: dict[str, int] = {}
        if isinstance(source, str):
            for ch in source:
                freq_map[ch] = freq_map.get(ch, 0) + 1
        elif isinstance(source, dict):
            freq_map = dict(source)

        heap: list[HuffmanHeapItem] = []
        count_id = 0
        for ch, freq in freq_map.items():
            disp_ch = "␣" if ch == " " else ch
            node = TreeNode(key=freq, val=disp_ch, freq=freq)
            heapq.heappush(heap, HuffmanHeapItem(freq=freq, count_id=count_id, node=node))
            count_id += 1

        if not heap:
            self._root = None
            return

        if len(heap) == 1:
            item = heapq.heappop(heap)
            self._root = TreeNode(key=item.freq, val="", freq=item.freq, left=item.node)
        else:
            while len(heap) > 1:
                item1 = heapq.heappop(heap)
                item2 = heapq.heappop(heap)
                combined_freq = item1.freq + item2.freq
                parent = TreeNode(
                    key=combined_freq,
                    val=f"{combined_freq}",
                    freq=combined_freq,
                    left=item1.node,
                    right=item2.node,
                )
                heapq.heappush(heap, HuffmanHeapItem(freq=combined_freq, count_id=count_id, node=parent))
                count_id += 1
            self._root = heapq.heappop(heap).node

        # Assign prefix binary codes: '0' for left, '1' for right
        self._assign_huffman_codes(self._root, "")

    def _assign_huffman_codes(self, node: Optional[TreeNode], code: str) -> None:
        if node is None:
            return
        node.code = code
        if node.left:
            self._assign_huffman_codes(node.left, code + "0")
        if node.right:
            self._assign_huffman_codes(node.right, code + "1")

    def _build_expression_tree(self) -> None:
        """Builds authentic arithmetic expression tree ((A + B) * (C - D) / E)."""
        # Leaf operands
        nA = TreeNode(key=1, val="A")
        nB = TreeNode(key=2, val="B")
        nC = TreeNode(key=3, val="C")
        nD = TreeNode(key=4, val="D")
        nE = TreeNode(key=5, val="E")

        # Subtrees
        nPlus = TreeNode(key=10, val="+", left=nA, right=nB, is_operator=True)
        nMinus = TreeNode(key=20, val="−", left=nC, right=nD, is_operator=True)
        nMul = TreeNode(key=30, val="×", left=nPlus, right=nMinus, is_operator=True)
        nDiv = TreeNode(key=40, val="÷", left=nMul, right=nE, is_operator=True)

        self._root = nDiv

    def _compute_layout(self) -> None:
        """Computes 2D rendering (x, y) coordinates for all tree nodes."""
        canvas_r = self._btn_rects["canvas"]
        self._all_nodes = []
        if self._root is None:
            return

        tree_h = self._root.height()
        top_y = canvas_r.top() + 38.0
        bot_y = canvas_r.bottom() - 38.0
        dy = (bot_y - top_y) / max(1, tree_h - 1) if tree_h > 1 else 0.0

        # Assign x coordinates using in-order traversal index for clean non-overlapping layout
        inorder_nodes: list[TreeNode] = []
        self._collect_inorder(self._root, inorder_nodes)
        n_total = len(inorder_nodes)

        left_x = canvas_r.left() + 32.0
        right_x = canvas_r.right() - 32.0
        dx = (right_x - left_x) / max(1, n_total - 1) if n_total > 1 else 0.0

        for i, node in enumerate(inorder_nodes):
            node.x = left_x + i * dx

        # Assign y coordinates based on depth
        def _set_depth(node: Optional[TreeNode], depth: int) -> None:
            if node is None:
                return
            node.y = top_y + depth * dy
            self._all_nodes.append(node)
            _set_depth(node.left, depth + 1)
            _set_depth(node.right, depth + 1)

        _set_depth(self._root, 0)

    # -------------------------------------------------------------------------
    # Traversal & Search Algorithms
    # -------------------------------------------------------------------------
    def _collect_inorder(self, node: Optional[TreeNode], res: list[TreeNode]) -> None:
        if node:
            self._collect_inorder(node.left, res)
            res.append(node)
            self._collect_inorder(node.right, res)

    def _collect_preorder(self, node: Optional[TreeNode], res: list[TreeNode]) -> None:
        if node:
            res.append(node)
            self._collect_preorder(node.left, res)
            self._collect_preorder(node.right, res)

    def _collect_postorder(self, node: Optional[TreeNode], res: list[TreeNode]) -> None:
        if node:
            self._collect_postorder(node.left, res)
            self._collect_postorder(node.right, res)
            res.append(node)

    def _collect_levelorder(self, root: Optional[TreeNode], res: list[TreeNode]) -> None:
        if not root:
            return
        q = deque([root])
        while q:
            node = q.popleft()
            res.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

    def start_traversal(self, mode: str = "IN_ORDER") -> None:
        """Starts step-by-step traversal animation."""
        self._active_traversal_mode = mode
        self._search_path = []
        self._traversal_sequence = []
        if self._root:
            if mode == "IN_ORDER":
                self._collect_inorder(self._root, self._traversal_sequence)
            elif mode == "PRE_ORDER":
                self._collect_preorder(self._root, self._traversal_sequence)
            elif mode == "POST_ORDER":
                self._collect_postorder(self._root, self._traversal_sequence)
            elif mode == "LEVEL_ORDER":
                self._collect_levelorder(self._root, self._traversal_sequence)

        self._traversal_index = 0
        self._is_animating = True
        self._anim_timer.start()
        self.update()

    def _on_traversal_step(self) -> None:
        """Advances active visited node in traversal sequence."""
        if self._traversal_index < len(self._traversal_sequence) - 1:
            self._traversal_index += 1
            self.update()
        else:
            self._anim_timer.stop()
            self._is_animating = False
            keys = [n.val for n in self._traversal_sequence]
            self.traversal_completed.emit(self._active_traversal_mode, keys)
            self.update()

    def execute_bst_search(self, target_key: int) -> tuple[bool, int, list[TreeNode]]:
        """Authentic BST search tracking search path and comparison count."""
        self._search_key = target_key
        self._search_path = []
        self._search_comparisons = 0
        self._search_found = False

        curr = self._root
        while curr:
            self._search_path.append(curr)
            self._search_comparisons += 1
            if curr.key == target_key:
                self._search_found = True
                break
            elif target_key < curr.key:
                curr = curr.left
            else:
                curr = curr.right

        self.update()
        return self._search_found, self._search_comparisons, self._search_path

    # -------------------------------------------------------------------------
    # Physical & Information Theory Metrics (Zero Mock)
    # -------------------------------------------------------------------------
    @property
    def tree_height(self) -> int:
        return self._root.height() if self._root else 0

    @property
    def node_count(self) -> int:
        return self._root.size() if self._root else 0

    @property
    def is_avl_balanced(self) -> bool:
        """Check if binary tree satisfies AVL height balance (|BF| <= 1 for all nodes)."""
        def _check(node: Optional[TreeNode]) -> bool:
            if not node:
                return True
            if abs(node.balance_factor()) > 1:
                return False
            return _check(node.left) and _check(node.right)
        return _check(self._root)

    def calculate_huffman_metrics(self) -> dict[str, float]:
        """Calculates Shannon entropy, avg codeword length, and compression ratio."""
        leaves = [n for n in self._all_nodes if n.is_leaf and n.code]
        if not leaves:
            return {"entropy": 0.0, "avg_len": 0.0, "efficiency": 0.0, "compression_ratio": 0.0}

        total_freq = sum(leaf.freq for leaf in leaves) or 1
        entropy = 0.0
        avg_len = 0.0

        for leaf in leaves:
            p = leaf.freq / total_freq
            if p > 0.0:
                entropy -= p * math.log2(p)
                avg_len += p * len(leaf.code)

        efficiency = (entropy / avg_len * 100.0) if avg_len > 0 else 0.0
        compression = (1.0 - avg_len / 8.0) * 100.0  # vs 8-bit ASCII

        return {
            "entropy": entropy,
            "avg_len": avg_len,
            "efficiency": efficiency,
            "compression_ratio": compression,
        }

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        bg_rect = self.boundingRect()
        card_path = QPainterPath()
        card_path.addRoundedRect(bg_rect, 14.0, 14.0)
        painter.fillPath(card_path, QColor(15, 23, 42, 245))  # Deep Slate #0f172a
        painter.setPen(QPen(QColor(16, 185, 129, 180), 1.5))  # Emerald green accent
        painter.drawPath(card_path)

        # 2. Header
        self._paint_header(painter)

        # 3. Presets Bar
        self._paint_presets_bar(painter)

        # 4. Main Tree Viewport Canvas
        self._paint_tree_canvas(painter)

        # 5. Right Analytical Metrics Card
        self._paint_analytics_card(painter)

        # 6. Bottom Controls Toolbar
        self._paint_action_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(16.0, 12.0, 560.0, 24.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "🌳 CẤU TRÚC DỮ LIỆU CÂY NHỊ PHÂN & MÃ HÓA HUFFMAN",
        )

        preset = PRESETS.get(self._active_preset_code)
        sub_text = preset.title_vi if preset else ""
        sub_font = QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(16.0, 34.0, 560.0, 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"Mô hình: {sub_text}  •  Duyệt cây, Tìm kiếm nhị phân & Nén dữ liệu",
        )

        # Close button
        close_rect = self._btn_rects["close"]
        is_hover = (self._hovered_button == "close")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(239, 68, 68, 200 if is_hover else 120))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

    def _paint_presets_bar(self, painter: QPainter) -> None:
        preset_items = [
            ("BST_BALANCED_SEARCH", "BST Cân Bằng"),
            ("BST_DEGENERATE_UNBALANCED", "BST Suy Biến"),
            ("HUFFMAN_TEXT_COMPRESSION", "Huffman Chữ"),
            ("HUFFMAN_DNA_GENOME", "Huffman DNA"),
            ("ARITHMETIC_EXPRESSION", "Biểu Thức Số Học"),
        ]
        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font)

        for code, label in preset_items:
            rect = self._btn_rects[f"preset_{code}"]
            is_active = (self._active_preset_code == code)
            is_hover = (self._hovered_button == f"preset_{code}")

            if is_active:
                painter.setBrush(QColor(16, 185, 129, 220))  # Emerald-500
                painter.setPen(QPen(QColor(255, 255, 255), 1.2))
            elif is_hover:
                painter.setBrush(QColor(51, 65, 85, 220))
                painter.setPen(QPen(QColor(52, 211, 153), 1.0))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _paint_tree_canvas(self, painter: QPainter) -> None:
        canvas_r = self._btn_rects["canvas"]

        # Background frame
        painter.setBrush(QColor(10, 15, 26, 230))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(canvas_r, 10.0, 10.0)

        if not self._root:
            return

        is_huffman = (PRESETS[self._active_preset_code].tree_type == "HUFFMAN")

        # 1. Draw Edges connecting parent to children
        def _draw_edges(node: TreeNode) -> None:
            if node.left:
                self._draw_branch(painter, node, node.left, "0" if is_huffman else "")
                _draw_edges(node.left)
            if node.right:
                self._draw_branch(painter, node, node.right, "1" if is_huffman else "")
                _draw_edges(node.right)

        _draw_edges(self._root)

        # 2. Draw Nodes
        visited_nodes = self._traversal_sequence[: self._traversal_index + 1] if self._traversal_index >= 0 else []
        active_node = self._traversal_sequence[self._traversal_index] if (0 <= self._traversal_index < len(self._traversal_sequence)) else None

        for node in self._all_nodes:
            is_active = (node is active_node)
            is_visited = (node in visited_nodes)
            is_in_search = (node in self._search_path)
            self._draw_node(painter, node, is_active, is_visited, is_in_search)

    def _draw_branch(self, painter: QPainter, parent: TreeNode, child: TreeNode, label: str) -> None:
        is_search_edge = (parent in self._search_path and child in self._search_path)
        if is_search_edge:
            pen = QPen(QColor(245, 158, 11), 2.5, Qt.PenStyle.SolidLine)  # Amber search path
        else:
            pen = QPen(QColor(71, 85, 105), 1.6, Qt.PenStyle.SolidLine)

        painter.setPen(pen)
        painter.drawLine(QPointF(parent.x, parent.y), QPointF(child.x, child.y))

        # Branch label ('0' or '1' in Huffman mode)
        if label:
            mid_x = (parent.x + child.x) / 2.0
            mid_y = (parent.y + child.y) / 2.0
            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor(52, 211, 153) if label == "0" else QColor(56, 189, 248))
            painter.drawText(QRectF(mid_x - 8.0, mid_y - 10.0, 16.0, 16.0), int(Qt.AlignmentFlag.AlignCenter), label)

    def _draw_node(
        self,
        painter: QPainter,
        node: TreeNode,
        is_active: bool,
        is_visited: bool,
        is_in_search: bool,
    ) -> None:
        r = 17.0
        # Color coding
        if is_active:
            fill_c = QColor(16, 185, 129)  # Emerald green pulse
            stroke_c = QColor(255, 255, 255)
            # Glowing halo
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(16, 185, 129, 60))
            painter.drawEllipse(QPointF(node.x, node.y), r + 8.0, r + 8.0)
        elif is_in_search:
            fill_c = QColor(245, 158, 11)  # Amber search path
            stroke_c = QColor(255, 255, 255)
        elif is_visited:
            fill_c = QColor(14, 165, 233)  # Sky blue visited
            stroke_c = QColor(186, 230, 253)
        elif node.is_operator:
            fill_c = QColor(168, 85, 247)  # Purple operator
            stroke_c = QColor(216, 180, 254)
        else:
            fill_c = QColor(30, 41, 59)    # Slate node
            stroke_c = QColor(100, 116, 139)

        painter.setPen(QPen(stroke_c, 1.8))
        painter.setBrush(fill_c)
        painter.drawEllipse(QPointF(node.x, node.y), r, r)

        # Label inside node
        disp_txt = node.val or str(node.key)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRectF(node.x - r, node.y - r, 2 * r, 2 * r),
            int(Qt.AlignmentFlag.AlignCenter),
            disp_txt,
        )

        # Optional bottom subscript: Huffman codeword
        if node.code and node.is_leaf:
            painter.setFont(QFont("Consolas", 7, QFont.Weight.DemiBold))
            painter.setPen(QColor(52, 211, 153))
            painter.drawText(
                QRectF(node.x - 25.0, node.y + r + 2.0, 50.0, 14.0),
                int(Qt.AlignmentFlag.AlignCenter),
                node.code,
            )

    def _paint_analytics_card(self, painter: QPainter) -> None:
        card_r = QRectF(496.0, 88.0, 268.0, 360.0)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(card_r, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(QRectF(card_r.left() + 12.0, card_r.top() + 10.0, 240.0, 20.0), "📊 THỐNG KÊ & LÝ THUYẾT")

        # Tree metrics
        preset = PRESETS.get(self._active_preset_code)
        tree_type = preset.tree_type if preset else "BST"

        metrics_lines = [
            ("Tổng số nút (Nodes):", f"{self.node_count}"),
            ("Chiều cao cây (Height h):", f"{self.tree_height}"),
            ("Cân bằng AVL (|BF| ≤ 1):", "Đạt chuẩn ✔" if self.is_avl_balanced else "Không cân bằng ✘"),
        ]

        curr_y = card_r.top() + 36.0
        painter.setFont(QFont("Segoe UI", 8))
        for lbl, val in metrics_lines:
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(QPointF(card_r.left() + 12.0, curr_y), lbl)
            painter.setPen(QColor(248, 250, 252))
            painter.drawText(QRectF(card_r.right() - 110.0, curr_y - 12.0, 100.0, 16.0), int(Qt.AlignmentFlag.AlignRight), val)
            curr_y += 18.0

        # Traversal sequence display
        curr_y += 6.0
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(card_r.left() + 12.0, curr_y, card_r.right() - 12.0, curr_y)
        curr_y += 12.0

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QPointF(card_r.left() + 12.0, curr_y), f"Thứ tự duyệt: {self._active_traversal_mode}")
        curr_y += 16.0

        visited_txt = " ➔ ".join(n.val for n in self._traversal_sequence[: self._traversal_index + 1]) if self._traversal_index >= 0 else "(Chưa duyệt)"
        if len(visited_txt) > 42:
            visited_txt = visited_txt[:39] + "..."
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(QRectF(card_r.left() + 12.0, curr_y - 4.0, 244.0, 32.0), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.TextWordWrap), visited_txt)
        curr_y += 34.0

        # Special metrics according to tree type
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(card_r.left() + 12.0, curr_y, card_r.right() - 12.0, curr_y)
        curr_y += 14.0

        if tree_type == "HUFFMAN":
            hm = self.calculate_huffman_metrics()
            huff_lines = [
                ("Entropy Shannon H(X):", f"{hm['entropy']:.3f} bit"),
                ("Chiều dài trung bình L̄:", f"{hm['avg_len']:.3f} bit"),
                ("Hiệu suất nén η:", f"{hm['efficiency']:.1f}%"),
                ("Tỷ lệ nén vs ASCII-8:", f"{hm['compression_ratio']:.1f}%"),
            ]
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(52, 211, 153))
            painter.drawText(QPointF(card_r.left() + 12.0, curr_y), "MÃ HÓA HUFFMAN TỐI ƯU")
            curr_y += 18.0

            painter.setFont(QFont("Segoe UI", 8))
            for lbl, val in huff_lines:
                painter.setPen(QColor(148, 163, 184))
                painter.drawText(QPointF(card_r.left() + 12.0, curr_y), lbl)
                painter.setPen(QColor(52, 211, 153))
                painter.drawText(QRectF(card_r.right() - 110.0, curr_y - 12.0, 100.0, 16.0), int(Qt.AlignmentFlag.AlignRight), val)
                curr_y += 18.0
        else:
            # BST Search status
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(245, 158, 11))
            painter.drawText(QPointF(card_r.left() + 12.0, curr_y), "TÌM KIẾM NHỊ PHÂN (BST)")
            curr_y += 18.0

            status_str = f"Khóa {self._search_key}: {'Tìm thấy ✔' if self._search_found else 'Không thấy'}" if self._search_path else "Chưa tìm kiếm"
            bst_lines = [
                ("Trạng thái tìm kiếm:", status_str),
                ("Số phép so sánh:", f"{self._search_comparisons}"),
                ("Độ phức tạp trường hợp:", "O(log n) tốt nhất" if self.is_avl_balanced else "O(n) suy biến"),
            ]
            painter.setFont(QFont("Segoe UI", 8))
            for lbl, val in bst_lines:
                painter.setPen(QColor(148, 163, 184))
                painter.drawText(QPointF(card_r.left() + 12.0, curr_y), lbl)
                painter.setPen(QColor(245, 158, 11))
                painter.drawText(QRectF(card_r.right() - 130.0, curr_y - 12.0, 120.0, 16.0), int(Qt.AlignmentFlag.AlignRight), val)
                curr_y += 18.0

    def _paint_action_toolbar(self, painter: QPainter) -> None:
        btns = [
            ("btn_inorder", "Trung tố (LNR)"),
            ("btn_preorder", "Tiền tố (NLR)"),
            ("btn_postorder", "Hậu tố (LRN)"),
            ("btn_levelorder", "Theo tầng (BFS)"),
            ("btn_search", "🔍 Tìm khóa 40"),
            ("stamp", "📌 Đóng dấu Vector lên Bảng"),
        ]
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)

        for name, label in btns:
            rect = self._btn_rects[name]
            is_hover = (self._hovered_button == name)
            if name == "stamp":
                painter.setBrush(QColor(217, 119, 6, 230 if is_hover else 180))  # Amber-600
                painter.setPen(QPen(QColor(251, 191, 36), 1.0))
            elif name == "btn_search":
                painter.setBrush(QColor(245, 158, 11, 220 if is_hover else 160))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))
            else:
                painter.setBrush(QColor(16, 185, 129, 220 if is_hover else 160))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))

            painter.drawRoundedRect(rect, 6.0, 6.0)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

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
        if self._btn_rects["btn_inorder"].contains(pos):
            self.start_traversal("IN_ORDER")
            event.accept()
            return
        if self._btn_rects["btn_preorder"].contains(pos):
            self.start_traversal("PRE_ORDER")
            event.accept()
            return
        if self._btn_rects["btn_postorder"].contains(pos):
            self.start_traversal("POST_ORDER")
            event.accept()
            return
        if self._btn_rects["btn_levelorder"].contains(pos):
            self.start_traversal("LEVEL_ORDER")
            event.accept()
            return
        if self._btn_rects["btn_search"].contains(pos):
            self.execute_bst_search(40)
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
                elif name == "btn_inorder":
                    self.start_traversal("IN_ORDER")
                    return True
                elif name == "btn_preorder":
                    self.start_traversal("PRE_ORDER")
                    return True
                elif name == "btn_postorder":
                    self.start_traversal("POST_ORDER")
                    return True
                elif name == "btn_levelorder":
                    self.start_traversal("LEVEL_ORDER")
                    return True
                elif name == "btn_search":
                    self.execute_bst_search(40)
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def cleanup(self) -> None:
        """Stop running animation timer."""
        self._anim_timer.stop()
        self._is_animating = False

    def stamp_to_board(self) -> None:
        """Stamps current binary tree diagram as permanent vector elements onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import InformaticsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = InformaticsTools.create_binary_tree_diagram(
            center=center,
            preset_code=self._active_preset_code,
            traversal_mode=self._active_traversal_mode,
        )
        for itm in items:
            scene.addItem(itm)
