"""
AirTeach AI – Interactive Computer Science & Information Theory: Huffman Coding & Shannon Entropy
=================================================================================================
Provides an authentic computer science and information theory simulation of Huffman's optimal
prefix-free coding algorithm (David Huffman 1952) and Shannon Source Coding Theorem (Claude Shannon 1948):
  - Shannon Information Entropy:
      H(X) = - ∑ [p_i * log₂(p_i)]  (bits / symbol)
      Defines the theoretical lower bound on average code length.
  - Huffman Minimum-Redundancy Tree Construction:
      1. Initialize min-priority queue (min-heap) of leaf nodes ordered by frequency.
      2. Greedily extract two minimum-frequency nodes u and v.
      3. Create parent node w with f(w) = f(u) + f(v), left child u (bit 0), right child v (bit 1).
      4. Repeat until a single root node remains.
  - Average Code Length & Coding Efficiency:
      L̄ = ∑ [p_i * l_i]  (bits / symbol)
      Efficiency: η = H(X) / L̄ <= 1.0
      Redundancy: R = L̄ - H(X) >= 0
  - Compression Ratio vs Fixed 8-bit ASCII:
      Compression Ratio = 8.0 / L̄
      Space Saved = (1 - L̄ / 8.0) * 100%
  - 5 Curriculum Presets:
      1. CLASSIC_ABRACADABRA: "ABRACADABRA" (N=11, highly skewed frequency 'A' gets 1 bit code).
      2. VIETNAMESE_PROVERB: "HOC THAY KHONG TAY HOC BAN" (Vietnamese natural language frequency distribution).
      3. DNA_GENOMICS_SEQUENCE: "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGT" (Genomic 4-base alphabet A,C,G,T).
      4. UNIFORM_DISTRIBUTION_WORST_CASE: "ABCDEFGH" (Equal 1/8 probabilities, perfectly balanced tree L̄ = H(X) = 3 bits).
      5. EXTREME_SKEW_FIBONACCI: "AAAAABBBCCD" (Fibonacci-like frequency skew yielding maximal caterpillar tree depth).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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


@dataclass
class HuffmanNode:
    char: Optional[str]
    freq: int
    left: Optional[HuffmanNode] = None
    right: Optional[HuffmanNode] = None
    node_id: int = 0
    # Visual coordinates for tree layout
    x: float = 0.0
    y: float = 0.0

    def __lt__(self, other: HuffmanNode) -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        # Tie breaker: leaf nodes before internal nodes, then by char or id
        return (self.char or "~") < (other.char or "~")


@dataclass(frozen=True)
class HuffmanPreset:
    code: str
    title_vi: str
    badge_label: str
    text_data: str
    description: str


PRESETS: dict[str, HuffmanPreset] = {
    "CLASSIC_ABRACADABRA": HuffmanPreset(
        code="CLASSIC_ABRACADABRA",
        title_vi="Thuật toán Mã hóa Huffman: Chuỗi Cổ điển ABRACADABRA",
        badge_label="ABRACADABRA (N=11)",
        text_data="ABRACADABRA",
        description="Từ 'ABRACADABRA' có 5 chữ 'A' (chiếm 45.5%). Ký tự 'A' được gán mã nhị phân ngắn nhất (1 bit '0'), giúp nén tối ưu chuỗi dữ liệu.",
    ),
    "VIETNAMESE_PROVERB": HuffmanPreset(
        code="VIETNAMESE_PROVERB",
        title_vi="Phân bố Tần suất Ký tự Tiếng Việt: Tục ngữ 'Học thầy...'",
        badge_label="TIẾNG VIỆT TỰ NHIÊN",
        text_data="HOC THAY KHONG TAY HOC BAN",
        description="Văn bản tự nhiên tiếng Việt có tần suất xuất hiện không đều giữa các phụ âm và nguyên âm. Mã tiền tố Huffman nén tiết kiệm trên 60% so với mã 8-bit ASCII.",
    ),
    "DNA_GENOMICS_SEQUENCE": HuffmanPreset(
        code="DNA_GENOMICS_SEQUENCE",
        title_vi="Nén Chuỗi Sinh học Phân tử DNA (4 Bazơ A, C, G, T)",
        badge_label="TRÌNH TỰ GEN DNA",
        text_data="AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGT",
        description="Chuỗi gen DNA gồm 4 bazơ nitơ với phân bố không đồng nhất trong các vùng giàu T hoặc G. Cây Huffman tối ưu hóa số bit biểu diễn từng codon.",
    ),
    "UNIFORM_DISTRIBUTION_WORST_CASE": HuffmanPreset(
        code="UNIFORM_DISTRIBUTION_WORST_CASE",
        title_vi="Phân bố Đều: Cây Nhị phân Cân bằng Tuyệt đối L̄ = H(X) = 3",
        badge_label="PHÂN BỐ ĐỀU (8 KÝ TỰ)",
        text_data="ABCDEFGH",
        description="Khi 8 ký tự có tần số bằng nhau (1/8), Entropy Shannon đạt cực đại H(X) = log₂(8) = 3.0 bits. Cây Huffman hoàn toàn cân bằng với mọi mã dài đúng 3 bits.",
    ),
    "EXTREME_SKEW_FIBONACCI": HuffmanPreset(
        code="EXTREME_SKEW_FIBONACCI",
        title_vi="Độ lệch Tần số Cực hạn: Cây Huffman Sâu Dạng Con Sâu",
        badge_label="LỆCH FIBONACCI",
        text_data="AAAAAAAAAAAAABBBBBBBCCCCDD",
        description="Khi tần số ký tự tăng vọt theo cấp số, cây Huffman sinh ra bị lệch tối đa về một phía (caterpillar tree), chiều sâu cây đạt n - 1.",
    ),
}


class InteractiveHuffmanCodingItem(QGraphicsObject):
    """
    Interactive Huffman Coding, Prefix-Free Trees & Shannon Entropy Simulator.
    Provides step-by-step min-heap tree construction, live frequency analysis,
    entropy calculations, and compression performance metrics.
    """

    preset_changed = Signal(str)
    tree_updated = Signal()
    stamp_requested = Signal(dict)

    CARD_W = 940.0
    CARD_H = 630.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation State
        self._active_preset_code: str = "CLASSIC_ABRACADABRA"
        self._preset: HuffmanPreset = PRESETS[self._active_preset_code]
        self._text_data: str = self._preset.text_data

        # Huffman Tree Structure
        self._frequencies: dict[str, int] = {}
        self._probabilities: dict[str, float] = {}
        self._huffman_codes: dict[str, str] = {}
        self._root_node: Optional[HuffmanNode] = None

        # Build steps animation
        self._build_steps: list[tuple[str, list[HuffmanNode]]] = []
        self._current_step_idx: int = 0
        self._is_animating: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Step Timer
        self._step_timer = QTimer()
        self._step_timer.setInterval(800)  # 800ms per merge step
        self._step_timer.timeout.connect(self._on_step_tick)

        # Initialize tree from default preset
        self._rebuild_huffman_tree()

    # -------------------------------------------------------------------------
    # Algorithm & Mathematical Physics
    # -------------------------------------------------------------------------
    @property
    def text_data(self) -> str:
        return self._text_data

    @property
    def active_preset(self) -> HuffmanPreset:
        return self._preset

    @property
    def frequencies(self) -> dict[str, int]:
        return self._frequencies

    @property
    def huffman_codes(self) -> dict[str, str]:
        return self._huffman_codes

    def compute_shannon_entropy(self) -> float:
        """
        Calculates Shannon Information Entropy:
        H(X) = - ∑ [p_i * log₂(p_i)] in bits per symbol.
        """
        entropy = 0.0
        for p in self._probabilities.values():
            if p > 1e-12:
                entropy -= p * math.log2(p)
        return entropy

    def compute_average_code_length(self) -> float:
        """
        Calculates expected code length:
        L̄ = ∑ [p_i * length(code_i)] in bits per symbol.
        """
        avg_len = 0.0
        for char, p in self._probabilities.items():
            code = self._huffman_codes.get(char, "")
            avg_len += p * len(code)
        return avg_len

    def compute_compression_metrics(self) -> tuple[float, float, int, int]:
        """
        Returns (efficiency_eta, space_saved_pct, total_original_bits, total_compressed_bits).
        """
        entropy = self.compute_shannon_entropy()
        avg_len = self.compute_average_code_length()
        total_chars = len(self._text_data)

        # ASCII standard 8 bits per character
        orig_bits = total_chars * 8
        compressed_bits = sum(self._frequencies[c] * len(self._huffman_codes[c]) for c in self._frequencies)

        efficiency = (entropy / avg_len) if avg_len > 1e-6 else 1.0
        saved_pct = ((orig_bits - compressed_bits) / float(orig_bits) * 100.0) if orig_bits > 0 else 0.0

        return min(1.0, efficiency), saved_pct, orig_bits, compressed_bits

    def _rebuild_huffman_tree(self) -> None:
        """Constructs authentic Huffman min-heap tree and generates prefix codes."""
        # Count frequencies
        self._frequencies.clear()
        for char in self._text_data:
            self._frequencies[char] = self._frequencies.get(char, 0) + 1

        total_chars = float(len(self._text_data)) if self._text_data else 1.0
        self._probabilities = {c: count / total_chars for c, count in self._frequencies.items()}

        # Create leaf nodes
        node_counter = 0
        nodes: list[HuffmanNode] = []
        for char, freq in sorted(self._frequencies.items()):
            node_counter += 1
            node = HuffmanNode(char=char, freq=freq, node_id=node_counter)
            nodes.append(node)

        # If only 1 unique symbol
        if len(nodes) == 1:
            single = nodes[0]
            root = HuffmanNode(char=None, freq=single.freq, left=single, node_id=999)
            self._root_node = root
            self._huffman_codes = {single.char: "0"}
            self._layout_tree(self._root_node, 230.0, 95.0, 180.0, 1)
            self.tree_updated.emit()
            return

        # Greedily merge in min-heap priority queue
        heap = list(nodes)
        heapq.heapify(heap)

        self._build_steps.clear()
        step_id = 0

        while len(heap) > 1:
            u = heapq.heappop(heap)
            v = heapq.heappop(heap)

            node_counter += 1
            parent = HuffmanNode(char=None, freq=u.freq + v.freq, left=u, right=v, node_id=node_counter)
            heapq.heappush(heap, parent)

            step_id += 1
            self._build_steps.append((f"Ghép '{u.char or 'Node'}' ({u.freq}) & '{v.char or 'Node'}' ({v.freq}) -> Tổng {parent.freq}", list(heap)))

        self._root_node = heap[0] if heap else None

        # Traverse tree to generate prefix-free codes
        self._huffman_codes.clear()
        self._generate_codes(self._root_node, "")

        # Layout tree node positions
        if self._root_node:
            self._layout_tree(self._root_node, 240.0, 90.0, 190.0, 1)

        self._current_step_idx = len(self._build_steps)
        self.tree_updated.emit()
        self.update()

    def _generate_codes(self, node: Optional[HuffmanNode], current_code: str) -> None:
        if not node:
            return
        if node.char is not None:
            self._huffman_codes[node.char] = current_code if current_code else "0"
            return
        if node.left:
            self._generate_codes(node.left, current_code + "0")
        if node.right:
            self._generate_codes(node.right, current_code + "1")

    def _layout_tree(self, node: Optional[HuffmanNode], x: float, y: float, dx: float, depth: int) -> None:
        """Calculates visual 2D coordinates for rendering the binary tree hierarchy."""
        if not node:
            return
        node.x = x
        node.y = y

        next_dx = max(26.0, dx * 0.52)
        next_y = y + 54.0

        if node.left:
            self._layout_tree(node.left, x - dx * 0.5, next_y, next_dx, depth + 1)
        if node.right:
            self._layout_tree(node.right, x + dx * 0.5, next_y, next_dx, depth + 1)

    # -------------------------------------------------------------------------
    # Preset & Control Methods
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads designated Huffman curriculum preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._text_data = self._preset.text_data
        self._rebuild_huffman_tree()
        self.preset_changed.emit(preset_code)

    def set_custom_text(self, text: str) -> None:
        """Updates input text and recalculates optimal prefix codes."""
        if not text.strip():
            return
        self._text_data = text.strip()
        self._rebuild_huffman_tree()

    def step_next_build(self) -> None:
        """Steps forward in tree merging animation."""
        if self._current_step_idx < len(self._build_steps):
            self._current_step_idx += 1
            self.update()

    def reset_build_steps(self) -> None:
        """Resets tree build animation to initial step."""
        self._current_step_idx = 0
        self.update()

    def toggle_animation(self) -> None:
        """Toggles automated step-by-step playback."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._step_timer.start()
        else:
            self._step_timer.stop()
        self.update()

    def _on_step_tick(self) -> None:
        if self._current_step_idx < len(self._build_steps):
            self._current_step_idx += 1
            self.update()
        else:
            # Loop animation gently
            self._current_step_idx = 0
            self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card & Acrylic Frame
        card_rect = self.boundingRect()
        bg_brush = QBrush(QColor(10, 15, 29, 245))
        border_pen = QPen(QColor(56, 189, 248, 200) if self.isSelected() else QColor(30, 41, 59, 220), 1.8)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header & Badges
        self._paint_header(painter)

        # 3. Left Panel: Huffman Binary Tree Visualization
        self._paint_tree_panel(painter)

        # 4. Right Panel: Optimal Prefix Code Table & Metrics
        self._paint_table_panel(painter)

        # 5. Bottom Digital Readout Bar
        self._paint_readout_bar(painter)

        # 6. Interactive In-Air Controls Bar
        self._paint_controls_bar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QPen(QColor("#f8fafc")))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 28, "🌲 Cây Mã hóa Huffman & Lý thuyết Thông tin Shannon")

        # Badge
        badge_text = self._preset.badge_label
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        badge_rect = QRectF(520.0, 12.0, 175.0, 22.0)
        painter.setBrush(QBrush(QColor(16, 185, 129, 50)))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # Subtitle / Description
        painter.setPen(QPen(QColor("#94a3b8")))
        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.drawText(20, 46, self._preset.description)

    def _paint_tree_panel(self, painter: QPainter) -> None:
        """Renders the Huffman binary tree with labeled edges '0' and '1'."""
        tree_rect = QRectF(20.0, 56.0, 460.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(tree_rect, 10.0, 10.0)

        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(34, 78, "CÂY TIỀN TỐ TỐI ƯU HUFFMAN (PREFIX-FREE BINARY TREE)")

        if not self._root_node:
            return

        # Draw tree edges
        self._paint_tree_edges(painter, self._root_node)

        # Draw tree nodes
        self._paint_tree_nodes(painter, self._root_node)

    def _paint_tree_edges(self, painter: QPainter, node: HuffmanNode) -> None:
        """Recursively renders binary tree branches with '0' (left) and '1' (right) bits."""
        if node.left:
            p1 = QPointF(node.x, node.y)
            p2 = QPointF(node.left.x, node.left.y)
            painter.setPen(QPen(QColor("#38bdf8"), 2.0))
            painter.drawLine(p1, p2)

            # Label '0'
            mid = (p1 + p2) * 0.5
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#38bdf8")))
            painter.drawText(int(mid.x() - 14), int(mid.y() + 2), "0")

            self._paint_tree_edges(painter, node.left)

        if node.right:
            p1 = QPointF(node.x, node.y)
            p2 = QPointF(node.right.x, node.right.y)
            painter.setPen(QPen(QColor("#f59e0b"), 2.0))
            painter.drawLine(p1, p2)

            # Label '1'
            mid = (p1 + p2) * 0.5
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(int(mid.x() + 6), int(mid.y() + 2), "1")

            self._paint_tree_edges(painter, node.right)

    def _paint_tree_nodes(self, painter: QPainter, node: HuffmanNode) -> None:
        """Recursively renders node badges (leaves in green, internal nodes in slate blue)."""
        is_leaf = node.char is not None
        node_rad = 17.0 if is_leaf else 13.0

        node_rect = QRectF(node.x - node_rad, node.y - node_rad, node_rad * 2.0, node_rad * 2.0)

        if is_leaf:
            # Leaf node badge
            painter.setBrush(QBrush(QColor(16, 185, 129, 230)))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawRoundedRect(node_rect, 6.0, 6.0)

            # Draw Character & Frequency
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#020617")))
            display_char = f"'{node.char}'" if node.char != " " else "'␣'"
            painter.drawText(node_rect, Qt.AlignmentFlag.AlignCenter, f"{display_char}\n{node.freq}")
        else:
            # Internal node
            painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
            painter.setPen(QPen(QColor("#38bdf8"), 1.5))
            painter.drawEllipse(node_rect)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(node_rect, Qt.AlignmentFlag.AlignCenter, str(node.freq))

        if node.left:
            self._paint_tree_nodes(painter, node.left)
        if node.right:
            self._paint_tree_nodes(painter, node.right)

    def _paint_table_panel(self, painter: QPainter) -> None:
        """Renders the generated code table, frequency probabilities, and Shannon terms."""
        tbl_rect = QRectF(494.0, 56.0, 426.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(tbl_rect, 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(508, 78, "BẢNG MÃ TIỀN TỐ & XÁC SUẤT SHANNON")

        # Table Column Headers
        y_row = 104.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(510, int(y_row), "KÝ TỰ")
        painter.drawText(554, int(y_row), "TẦN SỐ (f)")
        painter.drawText(620, int(y_row), "XÁC SUẤT (p)")
        painter.drawText(695, int(y_row), "MÃ HUFFMAN")
        painter.drawText(780, int(y_row), "ĐỘ DÀI (l)")
        painter.drawText(840, int(y_row), "-p·log₂(p)")

        # Divider
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(QPointF(508.0, y_row + 6.0), QPointF(906.0, y_row + 6.0))

        # Rows
        y_row += 22.0
        sorted_chars = sorted(self._frequencies.keys(), key=lambda c: (-self._frequencies[c], c))

        painter.setFont(QFont("Segoe UI", 8))
        row_count = 0
        for char in sorted_chars:
            if row_count >= 10:
                painter.setPen(QPen(QColor("#64748b")))
                painter.drawText(510, int(y_row), f"... và {len(sorted_chars) - 10} ký tự khác")
                break

            freq = self._frequencies[char]
            prob = self._probabilities[char]
            code = self._huffman_codes.get(char, "")
            l_len = len(code)
            shannon_term = -prob * math.log2(prob) if prob > 1e-9 else 0.0

            display_char = f"'{char}'" if char != " " else "'␣'"

            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(512, int(y_row), display_char)
            painter.drawText(566, int(y_row), str(freq))
            painter.drawText(626, int(y_row), f"{prob * 100:.1f}%")

            # Code in bold cyan/green
            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#10b981")))
            painter.drawText(695, int(y_row), code)

            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(792, int(y_row), f"{l_len} bit")

            painter.setPen(QPen(QColor("#38bdf8")))
            painter.drawText(846, int(y_row), f"{shannon_term:.3f}")

            y_row += 22.0
            row_count += 1

        # Encoded string sample
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(QPointF(508.0, 360.0), QPointF(906.0, 360.0))

        encoded_str = "".join(self._huffman_codes.get(c, "") for c in self._text_data)
        trunc_str = encoded_str[:42] + ("..." if len(encoded_str) > 42 else "")
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(510, 380, f"CHUỖI ĐÃ MÃ HÓA ({len(encoded_str)} bits):")
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QPen(QColor("#22c55e")))
        painter.drawText(510, 400, trunc_str)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#64748b")))
        painter.drawText(510, 422, f"Chuỗi gốc ({len(self._text_data)} ký tự): \"{self._text_data[:24]}...\"")

    def _paint_readout_bar(self, painter: QPainter) -> None:
        """Renders information theory formulas, entropy, average length, and storage savings."""
        bar_rect = QRectF(20.0, 456.0, 900.0, 68.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bar_rect, 8.0, 8.0)

        entropy = self.compute_shannon_entropy()
        avg_len = self.compute_average_code_length()
        eta, saved_pct, orig_b, comp_b = self.compute_compression_metrics()

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(36, 478, f"ENTROPY SHANNON H(X) = -∑ pᵢ·log₂(pᵢ) = {entropy:.3f} bits/ký tự")
        painter.drawText(36, 498, f"ĐỘ DÀI MÃ TRUNG BÌNH L̄ = ∑ pᵢ·lᵢ = {avg_len:.3f} bits/ký tự")
        painter.drawText(36, 516, f"ĐỘ DƯ THỪA (REDUNDANCY): R = L̄ - H(X) = {max(0.0, avg_len - entropy):.3f} bits")

        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(480, 478, f"HIỆU SUẤT MÃ HÓA: η = H(X) / L̄ = {eta * 100.0:.1f}% (Cận Shannon: H(X) ≤ L̄ < H(X)+1)")
        painter.drawText(480, 498, f"KÍCH THƯỚC: Gốc 8-bit ASCII = {orig_b} bits  →  Huffman = {comp_b} bits")
        painter.drawText(480, 516, f"DUNG LƯỢNG TIẾT KIỆM ĐƯỢC: {saved_pct:.1f}% (Tỷ lệ nén: {8.0 / avg_len if avg_len > 0 else 1.0:.2f}x)")

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders in-air buttons for preset cycling, animation steps, and stamping."""
        bar_y = 536.0
        self._buttons.clear()

        btn_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(btn_font)

        # 1. Preset Selector Button
        preset_rect = QRectF(20.0, bar_y, 160.0, 32.0)
        self._draw_btn(painter, preset_rect, "btn_preset", f"Presets: {self._preset.badge_label}")

        # 2. Step Forward Button
        step_rect = QRectF(188.0, bar_y, 110.0, 32.0)
        self._draw_btn(painter, step_rect, "btn_step", "⏭ Từng bước (Step)")

        # 3. Play / Pause Button
        play_label = "⏸ Tạm dừng" if self._is_animating else "▶ Tự động ghép"
        play_rect = QRectF(306.0, bar_y, 115.0, 32.0)
        self._draw_btn(painter, play_rect, "btn_play", play_label)

        # 4. Reset Tree Button
        reset_rect = QRectF(429.0, bar_y, 100.0, 32.0)
        self._draw_btn(painter, reset_rect, "btn_reset", "↺ Đặt lại cây")

        # 5. Quick Text Switchers
        self._draw_btn(painter, QRectF(537.0, bar_y, 75.0, 32.0), "btn_txt_dna", "DNA Gen")
        self._draw_btn(painter, QRectF(620.0, bar_y, 82.0, 32.0), "btn_txt_viet", "Tiếng Việt")
        self._draw_btn(painter, QRectF(710.0, bar_y, 80.0, 32.0), "btn_txt_abra", "Abracadabra")

        # 6. Stamp to Canvas Button
        stamp_rect = QRectF(798.0, bar_y, 85.0, 32.0)
        self._draw_btn(painter, stamp_rect, "btn_stamp", "📌 Đóng dấu")

        # 7. Close Button (Top right)
        close_rect = QRectF(self.CARD_W - 38.0, 10.0, 26.0, 26.0)
        self._draw_btn(painter, close_rect, "btn_close", "✕")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, label: str) -> None:
        self._buttons[key] = rect
        is_hovered = self._hovered_btn == key

        btn_bg = QColor(30, 41, 59, 220) if not is_hovered else QColor(16, 185, 129, 220)
        btn_pen = QPen(QColor("#10b981") if is_hovered else QColor(51, 65, 85), 1.2)

        painter.setBrush(QBrush(btn_bg))
        painter.setPen(btn_pen)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Mouse Events
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_hover = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_hover != self._hovered_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered_btn = None
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
            self._is_dragging = True
            self._last_mouse_pos = event.scenePos()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging:
            delta = event.scenePos() - self._last_mouse_pos
            self._last_mouse_pos = event.scenePos()
            self.setPos(self.pos() + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging = False
        super().mouseReleaseEvent(event)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_step":
            self.step_next_build()
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_reset":
            self.reset_build_steps()
        elif key == "btn_txt_dna":
            self.load_preset("DNA_GENOMICS_SEQUENCE")
        elif key == "btn_txt_viet":
            self.load_preset("VIETNAMESE_PROVERB")
        elif key == "btn_txt_abra":
            self.load_preset("CLASSIC_ABRACADABRA")
        elif key == "btn_stamp":
            self.stamp_diagram()
        elif key == "btn_close":
            self.setVisible(False)

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current state onto canvas."""
        entropy = self.compute_shannon_entropy()
        avg_len = self.compute_average_code_length()
        eta, saved_pct, orig_b, comp_b = self.compute_compression_metrics()

        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 640.0,
            "height": 340.0,
            "preset_code": self._active_preset_code,
            "text_sample": self._text_data[:20],
            "entropy": entropy,
            "avg_code_len": avg_len,
            "efficiency_pct": eta * 100.0,
            "saved_storage_pct": saved_pct,
            "codes": dict(self._huffman_codes),
        }
        self.stamp_requested.emit(stamp_data)
