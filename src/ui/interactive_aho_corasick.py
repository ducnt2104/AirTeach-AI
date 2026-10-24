"""
AirTeach AI – Interactive Computer Science & Automata Theory: Aho-Corasick Multi-Pattern Automaton
==================================================================================================
Provides an authentic computer science simulation of the optimal Aho-Corasick string matching algorithm
(Alfred Aho & Margaret Corasick 1975, Bell Labs, CACM):
  - Optimal Linear Time Complexity:
      Preprocessing Trie & Failure Transitions: O(m) where m = ∑ |p_i|
      Text Searching: O(n + z) where n = |Text|, z = number of occurrences
      Total Complexity: O(n + m + z)  (Versus Naive O(n * m))
  - Three Mathematical State Transition Functions:
      1. Goto Function g(s, c): Direct Trie edge from state s on input character c.
         For root s = 0: if no pattern starts with c, g(0, c) = 0 (self-loop).
      2. Failure Function f(s): Longest proper suffix of the string leading to state s
         that is also a prefix of some pattern in the dictionary.
         Calculated breadth-first (BFS) level by level.
      3. Output Function out(s): Set of all patterns recognized when entering state s,
         including dictionary suffix propagation: out(s) = out(s) ∪ out(f(s)).
  - Real-World Production Implementations:
      Powering Unix `fgrep`, Snort / Suricata Network Intrusion Detection Systems (NIDS),
      and bioinformatics DNA restriction enzyme motif search engines.
  - 5 Curriculum Presets:
      1. CLASSIC_BELL_LABS_HE_SHE_HIS_HERS: {"he", "she", "his", "hers"} on "ushers".
      2. DNA_BIOINFORMATICS_RESTRICTION_SITES: {"GAATTC", "GGATCC", "AAGCTT", "CTGCAG"} (EcoRI, BamHI, HindIII, PstI).
      3. NATURAL_LANGUAGE_KEYWORD_FILTER: {"toan", "ly", "hoa", "sinh", "tin"} (Vietnamese STEM subjects).
      4. OVERLAPPING_PREFIX_SUFFIX_STRESS: {"aa", "aaa", "aaaa", "aab"} on "aaaaaab".
      5. CYBERSECURITY_SNORT_IDS_SIGNATURES: {"root", "admin", "login", "eval", "exec"} on HTTP payload.
"""

from __future__ import annotations

import collections
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
class ACNode:
    state_id: int
    char: Optional[str] = None
    depth: int = 0
    children: dict[str, int] = field(default_factory=dict)  # char -> child_state_id
    fail_state: int = 0
    output_patterns: list[str] = field(default_factory=list)
    # 2D visual layout coordinates
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class ACPreset:
    code: str
    title_vi: str
    badge_label: str
    patterns: tuple[str, ...]
    sample_text: str
    description: str


PRESETS: dict[str, ACPreset] = {
    "CLASSIC_BELL_LABS_HE_SHE_HIS_HERS": ACPreset(
        code="CLASSIC_BELL_LABS_HE_SHE_HIS_HERS",
        title_vi="Ví dụ Cổ điển Bell Labs (Aho-Corasick 1975): 'he', 'she', 'his', 'hers'",
        badge_label="BELL LABS 1975",
        patterns=("he", "she", "his", "hers"),
        sample_text="ushers",
        description="Quét chuỗi 'ushers': Khi đọc 'ush', từ trạng thái 'she' (5) chuyển thất bại f(5)=2 ('he'). Thuật toán đồng thời phát hiện cả 'she' (vị trí 1-3), 'he' (vị trí 2-3) và 'hers' (vị trí 2-5).",
    ),
    "DNA_BIOINFORMATICS_RESTRICTION_SITES": ACPreset(
        code="DNA_BIOINFORMATICS_RESTRICTION_SITES",
        title_vi="Tin Sinh học: Quét Đồng thời Vị trí Cắt Enzyme Giới hạn DNA",
        badge_label="TIN SINH HỌC DNA",
        patterns=("GAATTC", "GGATCC", "AAGCTT", "CTGCAG"),
        sample_text="ACGGAATTCGCTAGGATCCAAGCTTCCTGCAGTA",
        description="Quét bộ gen DNA với 4 vị trí nhận diện enzyme cắt giới hạn kinh điển: EcoRI (GAATTC), BamHI (GGATCC), HindIII (AAGCTT), PstI (CTGCAG) trong duy nhất một lượt duyệt.",
    ),
    "NATURAL_LANGUAGE_KEYWORD_FILTER": ACPreset(
        code="NATURAL_LANGUAGE_KEYWORD_FILTER",
        title_vi="Lọc Từ khóa Tự nhiên Đa mẫu: Môn học STEM",
        badge_label="TỪ KHÓA STEM",
        patterns=("toan", "ly", "hoa", "sinh", "tin"),
        sample_text="hoc sinh chuyen tin va toan thi khoa hoc hoa ly",
        description="Tìm kiếm đồng thời toàn bộ danh mục từ khóa môn học trong văn bản tiếng Việt. Tối ưu hóa bộ lọc nội dung văn bản cho công cụ tìm kiếm và kiểm duyệt.",
    ),
    "OVERLAPPING_PREFIX_SUFFIX_STRESS": ACPreset(
        code="OVERLAPPING_PREFIX_SUFFIX_STRESS",
        title_vi="Thử thách Chồng lấn Tiền tố / Hậu tố Cực hạn",
        badge_label="CHỒNG LẤN CỰC HẠN",
        patterns=("aa", "aaa", "aaaa", "aab"),
        sample_text="aaaaaab",
        description="Tập mẫu có tính trùng lặp cao kiểm tra liên kết thất bại (failure chain) và liên kết từ điển (dictionary suffix links), kích hoạt nhiều kết quả khớp ở mỗi ký tự.",
    ),
    "CYBERSECURITY_SNORT_IDS_SIGNATURES": ACPreset(
        code="CYBERSECURITY_SNORT_IDS_SIGNATURES",
        title_vi="An ninh Mạng: Nhận dạng Chữ ký Tấn công (Snort / Suricata NIDS)",
        badge_label="CHỮ KÝ SNORT NIDS",
        patterns=("root", "admin", "login", "eval", "exec"),
        sample_text="POST /admin/login.php?exec=eval(root_priv)",
        description="Mô phỏng cơ chế kiểm tra gói tin sâu (Deep Packet Inspection) của hệ thống phát hiện xâm nhập Snort, dò quét hàng ngàn chuỗi mã độc hại song song theo thời gian thực.",
    ),
}


class InteractiveAhoCorasickItem(QGraphicsObject):
    """
    Interactive Aho-Corasick Automaton & Multi-Pattern String Matching Simulator.
    Simulates Trie tree construction, BFS failure link resolution, suffix output propagation,
    and live real-time text stream scanning.
    """

    preset_changed = Signal(str)
    match_found = Signal(str, int)       # pattern, text_index
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
        self._active_preset_code: str = "CLASSIC_BELL_LABS_HE_SHE_HIS_HERS"
        self._preset: ACPreset = PRESETS[self._active_preset_code]
        self._patterns: list[str] = list(self._preset.patterns)
        self._sample_text: str = self._preset.sample_text

        # Automaton State
        self._nodes: dict[int, ACNode] = {}
        self._current_state_id: int = 0
        self._text_scan_idx: int = 0
        self._matches_history: list[tuple[str, int, int]] = []  # (pattern, start_idx, end_idx)

        # Visual layout
        self._show_failure_links: bool = True

        # Playback control
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Build Automaton
        self._build_automaton()

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(700)  # 700 ms per text character step
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Aho-Corasick Core Construction & Search Algorithms
    # -------------------------------------------------------------------------
    @property
    def patterns(self) -> list[str]:
        return self._patterns

    @property
    def sample_text(self) -> str:
        return self._sample_text

    @property
    def active_preset(self) -> ACPreset:
        return self._preset

    @property
    def nodes_count(self) -> int:
        return len(self._nodes)

    def _build_automaton(self) -> None:
        """Constructs Trie, computes BFS failure transitions and dictionary outputs."""
        self._nodes.clear()
        self._nodes[0] = ACNode(state_id=0, char=None, depth=0)
        state_counter = 0

        # Phase 1: Build Goto Trie
        for pattern in self._patterns:
            curr = 0
            for depth, ch in enumerate(pattern, start=1):
                if ch not in self._nodes[curr].children:
                    state_counter += 1
                    self._nodes[state_counter] = ACNode(state_id=state_counter, char=ch, depth=depth)
                    self._nodes[curr].children[ch] = state_counter
                curr = self._nodes[curr].children[ch]
            self._nodes[curr].output_patterns.append(pattern)

        # Phase 2: Compute Failure Function f(s) using BFS Queue
        queue = collections.deque()

        # Level 1 children fail to root (state 0)
        for ch, child_id in self._nodes[0].children.items():
            self._nodes[child_id].fail_state = 0
            queue.append(child_id)

        while queue:
            parent_id = queue.popleft()
            for ch, child_id in self._nodes[parent_id].children.items():
                queue.append(child_id)

                fail_cand = self._nodes[parent_id].fail_state
                while fail_cand != 0 and ch not in self._nodes[fail_cand].children:
                    fail_cand = self._nodes[fail_cand].fail_state

                if ch in self._nodes[fail_cand].children and self._nodes[fail_cand].children[ch] != child_id:
                    self._nodes[child_id].fail_state = self._nodes[fail_cand].children[ch]
                else:
                    self._nodes[child_id].fail_state = 0

                # Phase 3: Suffix Link Output Propagation
                fail_target = self._nodes[child_id].fail_state
                if fail_target != 0:
                    for pat in self._nodes[fail_target].output_patterns:
                        if pat not in self._nodes[child_id].output_patterns:
                            self._nodes[child_id].output_patterns.append(pat)

        # Layout visual coordinates
        self._layout_automaton_nodes()
        self.reset_scan()

    def _layout_automaton_nodes(self) -> None:
        """Positions automaton states in clean hierarchical layers for canvas rendering."""
        # Group by depth
        levels: dict[int, list[int]] = collections.defaultdict(list)
        for s_id, node in self._nodes.items():
            levels[node.depth].append(s_id)

        max_depth = max(levels.keys()) if levels else 0
        x_start = 60.0
        x_spacing = 380.0 / max(1, max_depth)

        for depth, state_ids in levels.items():
            state_x = x_start + depth * x_spacing
            n_states = len(state_ids)
            y_spacing = 300.0 / float(n_states + 1)
            for idx, s_id in enumerate(state_ids):
                self._nodes[s_id].x = state_x
                self._nodes[s_id].y = 80.0 + (idx + 1) * y_spacing

    def reset_scan(self) -> None:
        """Resets text stream scanner back to initial position."""
        self._current_state_id = 0
        self._text_scan_idx = 0
        self._matches_history.clear()
        self.update()

    def step_text_scan(self) -> None:
        """Advances text scanner by exactly one character."""
        if self._text_scan_idx >= len(self._sample_text):
            # Loop text stream
            self.reset_scan()
            return

        ch = self._sample_text[self._text_scan_idx]
        curr = self._current_state_id

        # Follow failure transitions until character matches or reaches root
        while curr != 0 and ch not in self._nodes[curr].children:
            curr = self._nodes[curr].fail_state

        if ch in self._nodes[curr].children:
            curr = self._nodes[curr].children[ch]
        else:
            curr = 0

        self._current_state_id = curr

        # Check matched patterns at this state
        if self._nodes[curr].output_patterns:
            for pat in self._nodes[curr].output_patterns:
                start_i = self._text_scan_idx - len(pat) + 1
                end_i = self._text_scan_idx
                self._matches_history.append((pat, start_i, end_i))
                self.match_found.emit(pat, start_i)

        self._text_scan_idx += 1
        self.update()

    def toggle_failure_links(self) -> None:
        """Toggles visual rendering of red failure transition links."""
        self._show_failure_links = not self._show_failure_links
        self.update()

    def toggle_animation(self) -> None:
        """Toggles automated text scanning playback."""
        self._is_running = not self._is_running
        self.update()

    def _on_sim_tick(self) -> None:
        if not self._is_running:
            return
        self.step_text_scan()

    # -------------------------------------------------------------------------
    # Presets & Control Methods
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads designated Aho-Corasick curriculum preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._patterns = list(self._preset.patterns)
        self._sample_text = self._preset.sample_text
        self._build_automaton()
        self.preset_changed.emit(preset_code)

    def _layout_buttons(self) -> None:
        """Initializes button bounding rectangles."""
        bar_y = 536.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 32.0)
        self._buttons["btn_step"] = QRectF(188.0, bar_y, 100.0, 32.0)
        self._buttons["btn_play"] = QRectF(296.0, bar_y, 85.0, 32.0)
        self._buttons["btn_reset"] = QRectF(389.0, bar_y, 80.0, 32.0)
        self._buttons["btn_toggle_fail"] = QRectF(477.0, bar_y, 140.0, 32.0)
        self._buttons["btn_p_dna"] = QRectF(625.0, bar_y, 80.0, 32.0)
        self._buttons["btn_p_snort"] = QRectF(712.0, bar_y, 80.0, 32.0)
        self._buttons["btn_stamp"] = QRectF(800.0, bar_y, 120.0, 32.0)

    def set_custom_patterns_and_text(self, patterns: list[str], text: str) -> None:
        """Sets custom patterns and text stream."""
        if not patterns or not text:
            return
        self._patterns = [p.strip() for p in patterns if p.strip()]
        self._sample_text = text.strip()
        self._active_preset_code = "CUSTOM"
        self._preset = ACPreset(
            code="CUSTOM",
            title_vi="Tùy biến người dùng (Custom Patterns)",
            badge_label="CUSTOM",
            patterns=tuple(self._patterns),
            sample_text=self._sample_text,
            description="Mô hình otomat Aho-Corasick với tập mẫu do giáo viên tự định nghĩa.",
        )
        self._build_automaton()

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

        # 3. Left Panel: Aho-Corasick Trie & Automaton State Machine
        self._paint_automaton_panel(painter)

        # 4. Right Panel: Live Text Stream Scanner & Matches Table
        self._paint_scanner_panel(painter)

        # 5. Bottom Digital Readout Bar
        self._paint_readout_bar(painter)

        # 6. Interactive In-Air Controls Bar
        self._paint_controls_bar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QPen(QColor("#f8fafc")))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 28, "🔤 Máy Tự động Khớp Đa Mẫu Aho-Corasick (Trie & Failure Links)")

        # Badge
        badge_text = self._preset.badge_label
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        badge_rect = QRectF(590.0, 12.0, 195.0, 22.0)
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

    def _paint_automaton_panel(self, painter: QPainter) -> None:
        """Renders the state machine graph with goto edges and failure transitions."""
        auto_rect = QRectF(20.0, 56.0, 460.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(auto_rect, 10.0, 10.0)

        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(34, 78, "CÂY TRIE & LIÊN KẾT THẤT BẠI (FAILURE TRANSITIONS)")

        # 1. Draw Failure Links (Curved dashed red lines)
        if self._show_failure_links:
            painter.setPen(QPen(QColor(239, 68, 68, 160), 1.2, Qt.PenStyle.DashLine))
            for s_id, node in self._nodes.items():
                if node.depth >= 2 and node.fail_state != 0:
                    target_node = self._nodes[node.fail_state]
                    curve_path = QPainterPath()
                    curve_path.moveTo(node.x, node.y)
                    # Quadratic control point bending downward
                    ctrl_x = (node.x + target_node.x) * 0.5
                    ctrl_y = max(node.y, target_node.y) + 30.0
                    curve_path.quadTo(ctrl_x, ctrl_y, target_node.x, target_node.y)
                    painter.drawPath(curve_path)

        # 2. Draw Goto Transitions (Solid cyan arrows)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        for s_id, node in self._nodes.items():
            for ch, child_id in node.children.items():
                child_node = self._nodes[child_id]
                p1 = QPointF(node.x, node.y)
                p2 = QPointF(child_node.x, child_node.y)

                painter.setPen(QPen(QColor("#38bdf8"), 1.8))
                painter.drawLine(p1, p2)

                # Character label on edge
                mid_pt = (p1 + p2) * 0.5
                painter.setPen(QPen(QColor("#38bdf8")))
                painter.drawText(int(mid_pt.x() - 4), int(mid_pt.y() - 4), ch)

        # 3. Draw States / Nodes
        for s_id, node in self._nodes.items():
            is_active = (s_id == self._current_state_id)
            is_match = len(node.output_patterns) > 0
            rad = 14.0

            n_rect = QRectF(node.x - rad, node.y - rad, 2 * rad, 2 * rad)

            if is_active:
                # Glowing current state
                painter.setBrush(QBrush(QColor(245, 158, 11, 230)))
                painter.setPen(QPen(QColor("#ffffff"), 2.2))
            elif is_match:
                # Terminal pattern match state (Green double ring)
                painter.setBrush(QBrush(QColor(16, 185, 129, 230)))
                painter.setPen(QPen(QColor("#ffffff"), 1.5))
            else:
                # Normal state
                painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
                painter.setPen(QPen(QColor("#38bdf8"), 1.2))

            painter.drawEllipse(n_rect)

            # Double ring for accepting match states
            if is_match and not is_active:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#10b981"), 1.0))
                painter.drawEllipse(QRectF(node.x - rad - 3.0, node.y - rad - 3.0, 2 * (rad + 3.0), 2 * (rad + 3.0)))

            # State ID Number
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#020617") if (is_active or is_match) else QColor("#f8fafc")))
            painter.drawText(n_rect, Qt.AlignmentFlag.AlignCenter, str(s_id))

    def _paint_scanner_panel(self, painter: QPainter) -> None:
        """Renders live sliding text stream scanner buffer and match events table."""
        scan_rect = QRectF(494.0, 56.0, 426.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(scan_rect, 10.0, 10.0)

        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(508, 78, "DÒNG DỮ LIỆU ĐỌC VÀ BẢNG KHỚP TỪ KHÓA")

        # 1. Text Stream Sliding Window Box
        stream_box = QRectF(508.0, 95.0, 398.0, 75.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(stream_box, 6.0, 6.0)

        # Render characters in stream box
        text_len = len(self._sample_text)
        char_w = 14.0
        start_draw_x = stream_box.left() + 15.0

        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        for i, ch in enumerate(self._sample_text):
            if i >= 24:
                painter.setPen(QPen(QColor("#64748b")))
                painter.drawText(int(start_draw_x + 24 * char_w), int(stream_box.top() + 32), "...")
                break

            cx_char = start_draw_x + i * char_w
            is_current = (i == self._text_scan_idx)
            is_past = (i < self._text_scan_idx)

            if is_current:
                # Active scanner cursor box
                cursor_rect = QRectF(cx_char - 2.0, stream_box.top() + 12.0, char_w, 28.0)
                painter.setBrush(QBrush(QColor(245, 158, 11, 200)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(cursor_rect, 3.0, 3.0)

                painter.setPen(QPen(QColor("#020617")))
            elif is_past:
                painter.setPen(QPen(QColor("#38bdf8")))
            else:
                painter.setPen(QPen(QColor("#64748b")))

            painter.drawText(int(cx_char), int(stream_box.top() + 32), ch)

        # Scanner cursor indicator arrow
        if self._text_scan_idx < 24:
            cur_x = start_draw_x + self._text_scan_idx * char_w + 4.0
            cur_y = stream_box.top() + 48.0
            painter.setPen(QPen(QColor("#f59e0b"), 2.0))
            painter.drawLine(QPointF(cur_x, cur_y), QPointF(cur_x, cur_y + 12.0))
            painter.drawText(int(cur_x - 12), int(cur_y + 22), f"i={self._text_scan_idx}")

        # 2. Match Occurrences Table
        tbl_box = QRectF(508.0, 185.0, 398.0, 245.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 200)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(tbl_box, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(518, 204, "TỪ KHÓA ĐÃ KHỚP (MATCH)")
        painter.drawText(690, 204, "VỊ TRÍ (START-END)")
        painter.drawText(820, 204, "TRẠNG THÁI")

        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(QPointF(516.0, 210.0), QPointF(896.0, 210.0))

        # Recent matches list
        y_row = 230.0
        painter.setFont(QFont("Segoe UI", 8))
        recent_matches = self._matches_history[-8:] if self._matches_history else []

        if not recent_matches:
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
            painter.setPen(QPen(QColor("#64748b")))
            painter.drawText(520, 240, "Đang quét dòng ký tự... Chưa phát hiện mẫu trùng khớp.")
        else:
            for pat, s_i, e_i in recent_matches:
                painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
                painter.setPen(QPen(QColor("#10b981")))
                painter.drawText(520, int(y_row), f"'{pat}'")

                painter.setFont(QFont("Segoe UI", 8))
                painter.setPen(QPen(QColor("#e2e8f0")))
                painter.drawText(700, int(y_row), f"[{s_i} ... {e_i}]")

                painter.setPen(QPen(QColor("#38bdf8")))
                painter.drawText(825, int(y_row), "XÁC NHẬN ✓")

                y_row += 22.0

    def _paint_readout_bar(self, painter: QPainter) -> None:
        """Draws digital metrics, complexity, and active state info."""
        bar_rect = QRectF(20.0, 456.0, 900.0, 68.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bar_rect, 8.0, 8.0)

        n_nodes = len(self._nodes)
        n_pats = len(self._patterns)
        total_m = sum(len(p) for p in self._patterns)
        total_n = len(self._sample_text)
        curr_node = self._nodes.get(self._current_state_id)
        fail_id = curr_node.fail_state if curr_node else 0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(36, 478, f"TRẠNG THÁI HIỆN TẠI: State #{self._current_state_id} (Độ sâu Trie: {curr_node.depth if curr_node else 0})")
        painter.drawText(36, 498, f"LIÊN KẾT THẤT BẠI: f({self._current_state_id}) = {fail_id} (Longest proper suffix-prefix)")
        painter.drawText(36, 516, f"ĐỘ PHỨC TẠP TỐI ƯU: O(n + m + z) = O({total_n} + {total_m} + {len(self._matches_history)}) thao tác tuyến tính")

        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(480, 478, f"SỐ LƯỢNG MẪU: k = {n_pats} mẫu | Tổng độ dài m = {total_m} ký tự")
        painter.drawText(480, 498, f"KẾT QUẢ KHỚP: Đã tìm thấy {len(self._matches_history)} vị trí xuất hiện trong chuỗi")
        painter.drawText(480, 516, f"HIỆU NĂNG VƯỢT TRỘI: Tiết kiệm ~{max(1.0, (total_n * total_m) / max(1, total_n + total_m)):.1f}x so với thuật toán ngây thơ (Naive search)")

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        bar_y = 536.0
        self._buttons.clear()

        btn_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(btn_font)

        # 1. Preset Selector Button
        preset_rect = QRectF(20.0, bar_y, 160.0, 32.0)
        self._draw_btn(painter, preset_rect, "btn_preset", f"Presets: {self._preset.badge_label}")

        # 2. Step Scan Button
        step_rect = QRectF(188.0, bar_y, 100.0, 32.0)
        self._draw_btn(painter, step_rect, "btn_step", "⏭ Quét 1 ký tự")

        # 3. Play / Pause Button
        play_label = "⏸ Tạm dừng" if self._is_running else "▶ Tiếp tục"
        play_rect = QRectF(296.0, bar_y, 85.0, 32.0)
        self._draw_btn(painter, play_rect, "btn_play", play_label)

        # 4. Reset Index Button
        reset_rect = QRectF(389.0, bar_y, 80.0, 32.0)
        self._draw_btn(painter, reset_rect, "btn_reset", "↺ Đặt lại")

        # 5. Toggle Failure Links Button
        fail_label = "Liên kết thất bại: BẬT" if self._show_failure_links else "Liên kết: ẨN"
        fail_rect = QRectF(477.0, bar_y, 140.0, 32.0)
        self._draw_btn(painter, fail_rect, "btn_toggle_fail", fail_label)

        # 6. Quick Presets Switchers
        self._draw_btn(painter, QRectF(625.0, bar_y, 80.0, 32.0), "btn_p_dna", "DNA Gen")
        self._draw_btn(painter, QRectF(712.0, bar_y, 80.0, 32.0), "btn_p_snort", "Snort NIDS")

        # 7. Stamp to Canvas Button
        stamp_rect = QRectF(800.0, bar_y, 85.0, 32.0)
        self._draw_btn(painter, stamp_rect, "btn_stamp", "📌 Đóng dấu")

        # 8. Close Button (Top right)
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
            self.step_text_scan()
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_reset":
            self.reset_scan()
        elif key == "btn_toggle_fail":
            self.toggle_failure_links()
        elif key == "btn_p_dna":
            self.load_preset("DNA_BIOINFORMATICS_RESTRICTION_SITES")
        elif key == "btn_p_snort":
            self.load_preset("CYBERSECURITY_SNORT_IDS_SIGNATURES")
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
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 640.0,
            "height": 340.0,
            "preset_code": self._active_preset_code,
            "patterns": list(self._patterns),
            "sample_text": self._sample_text[:24],
            "n_states": len(self._nodes),
            "matches_count": len(self._matches_history),
        }
        self.stamp_requested.emit(stamp_data)
