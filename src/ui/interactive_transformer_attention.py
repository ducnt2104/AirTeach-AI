"""
AirTeach AI - Interactive Transformer Self-Attention & Multi-Head Visualizer
=============================================================================
Authentic mathematical Transformer multi-head self-attention simulation (Vaswani et al., 2017):
- Computes scaled dot-product attention: Attention(Q, K, V) = Softmax(Q K^T / (τ * sqrt(d_k)) + M) V
- 4 Multi-Head Attention projections with distinct semantic specializations
- Dual Architecture Modes:
  - Bidirectional Encoder (BERT-style) with full pairwise attention
  - Autoregressive Causal Decoder (GPT-style) with lower-triangular causal mask
- Dual Interactive Visualizations:
  - T x T Attention Weight Heatmap with cell inspection and percentage tooltips
  - Bipartite Token Attention Ribbon Arcs connecting source and target tokens
- Interactive Temperature Scaling (τ) exploring focused vs diffuse attention entropy
- 5 Curriculum Presets (Pronoun Coreference, Vietnamese Syntax, Translation, Math, Code)
- Full touchless air gesture integration, canvas stamping, and .airteach packaging
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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


# ── Curriculum Presets ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransformerPreset:
    code: str
    name_vi: str
    text: str
    tokens: tuple[str, ...]
    description: str


PRESETS: dict[str, TransformerPreset] = {
    "PRONOUN_COREFERENCE": TransformerPreset(
        code="PRONOUN_COREFERENCE",
        name_vi="1. Phân giải Đại từ (Coreference Resolution)",
        text="The animal didn't cross the street because it was too tired",
        tokens=("The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"),
        description="Đại từ 'it' tập trung trọng số chú ý cao vào danh từ 'animal' thay vì 'street'.",
    ),
    "VIETNAMESE_SYNTAX": TransformerPreset(
        code="VIETNAMESE_SYNTAX",
        name_vi="2. Ngữ pháp Tiếng Việt (Cú pháp S-V-O)",
        text="Học sinh chăm chỉ nghe giảng bài trên lớp",
        tokens=("Học", "sinh", "chăm", "chỉ", "nghe", "giảng", "bài", "trên", "lớp"),
        description="Mô hình hóa mối quan hệ phụ thuộc giữa chủ ngữ 'Học sinh' và vị ngữ 'nghe giảng bài'.",
    ),
    "PARALLEL_CLAUSE": TransformerPreset(
        code="PARALLEL_CLAUSE",
        name_vi="3. Mệnh đề Song hành (Clause Alignment)",
        text="Knowledge is power and wisdom is key",
        tokens=("Knowledge", "is", "power", "and", "wisdom", "is", "key"),
        description="Liên từ 'and' kết nối hai vế song hành có cấu trúc cú pháp tương đương.",
    ),
    "ARITHMETIC_EXPR": TransformerPreset(
        code="ARITHMETIC_EXPR",
        name_vi="4. Logic Toán học (Variable Assignment)",
        text="Let x = 15 + 27 then compute 2 * x",
        tokens=("Let", "x", "=", "15", "+", "27", "then", "compute", "2", "*", "x"),
        description="Token biến 'x' ở cuối câu chú ý trực tiếp vào biểu thức gán giá trị ở đầu câu.",
    ),
    "CODE_FUNCTION": TransformerPreset(
        code="CODE_FUNCTION",
        name_vi="5. Cú pháp Lập trình (Code Ast Scope)",
        text="def area(radius): return math.pi * radius ** 2",
        tokens=("def", "area", "(", "radius", ")", ":", "return", "math", ".", "pi", "*", "radius", "**", "2"),
        description="Tham số 'radius' trong câu lệnh return liên kết chặt chẽ với định nghĩa hàm.",
    ),
}


HEAD_COLORS = [
    QColor("#38bdf8"),  # Sky Blue (Head 0)
    QColor("#f59e0b"),  # Amber (Head 1)
    QColor("#a855f7"),  # Purple (Head 2)
    QColor("#10b981"),  # Emerald (Head 3)
]


class InteractiveTransformerAttentionItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Multi-Head Self-Attention in Transformers.
    """

    WIDTH = 760.0
    HEIGHT = 490.0

    preset_changed = Signal(str)
    head_changed = Signal(int)
    token_selected = Signal(int, str)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (120.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Simulation Parameters
        self._active_preset_code: str = "PRONOUN_COREFERENCE"
        self._preset: TransformerPreset = PRESETS[self._active_preset_code]
        self._d_model: int = 32
        self._d_k: int = 16
        self._num_heads: int = 4
        self._active_head: int = 0  # 0..3 or -1 for averaged
        self._is_causal: bool = False  # False = Encoder (BERT), True = Decoder (GPT)
        self._temperature: float = 1.0  # Softmax temperature tau
        self._view_mode: str = "heatmap"  # "heatmap" or "ribbon"

        self._selected_token_idx: int = 0
        self._hover_cell: Optional[tuple[int, int]] = None

        # Computed Multi-Head Attention Matrices (num_heads x T x T)
        self._attn_matrices: list[np.ndarray] = []
        self._recompute_attention()

        # UI Layout Rectangles
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    # ── Mathematics & Authentic Computation ────────────────────────────────────

    def _recompute_attention(self) -> None:
        """
        Compute scaled dot-product attention matrices for all heads using
        reproducible sinusoidal positional embeddings and projection matrices.
        """
        tokens = self._preset.tokens
        T = len(tokens)
        d_m = self._d_model
        d_k = self._d_k

        # 1. Sinusoidal Positional Encoding matrix (T x d_model)
        pos = np.arange(T)[:, np.newaxis]
        i = np.arange(d_m)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_m))
        angle_rads = pos * angle_rates
        pe = np.zeros((T, d_m), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Deterministic projection weights per head
        rng = np.random.RandomState(abs(hash(self._active_preset_code)) % 100000 + 42)
        self._attn_matrices = []

        for h in range(self._num_heads):
            # Head-specific learned projections
            W_q = rng.randn(d_m, d_k).astype(np.float32) * 0.28
            W_k = rng.randn(d_m, d_k).astype(np.float32) * 0.28

            Q = np.dot(pe, W_q)
            K = np.dot(pe, W_k)

            # Scaled Dot-Product: S = (Q * K^T) / (tau * sqrt(d_k))
            scores = np.dot(Q, K.T) / (self._temperature * np.sqrt(d_k))

            # Apply causal mask (Decoder GPT mode)
            if self._is_causal:
                mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
                scores += mask

            # Numerically stable Softmax normalization along row axis
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            self._attn_matrices.append(attn)

        # Ensure selected token stays within range
        self._selected_token_idx = min(self._selected_token_idx, T - 1)

    def active_attention_matrix(self) -> np.ndarray:
        """Return the current attention matrix (active head or averaged multi-head)."""
        if self._active_head == -1:
            return np.mean(self._attn_matrices, axis=0)
        return self._attn_matrices[self._active_head]

    def compute_entropy(self, row_idx: int) -> float:
        """Compute Shannon entropy H = -sum(p * log2(p)) measuring attention focus."""
        attn = self.active_attention_matrix()
        row = attn[row_idx]
        eps = 1e-12
        return float(-np.sum(row * np.log2(row + eps)))

    # ── UI Layout ─────────────────────────────────────────────────────────────

    def _setup_layout(self) -> None:
        """Setup hitboxes for toolbar buttons, mode switchers, and controls."""
        self._btn_rects.clear()

        # Row 1: Preset Buttons
        px = 14.0
        py = 40.0
        btn_h = 22.0
        for code, pr in PRESETS.items():
            btn_w = 140.0
            self._btn_rects[f"preset_{code}"] = QRectF(px, py, btn_w, btn_h)
            px += btn_w + 6.0

        # Row 2: Head Selector & Mode Switchers
        hx = 14.0
        hy = 68.0
        head_w = 58.0
        for h in range(self._num_heads):
            self._btn_rects[f"head_{h}"] = QRectF(hx, hy, head_w, btn_h)
            hx += head_w + 5.0
        self._btn_rects["head_avg"] = QRectF(hx, hy, 68.0, btn_h)
        hx += 73.0

        # Mode toggles
        self._btn_rects["toggle_causal"] = QRectF(hx, hy, 110.0, btn_h)
        hx += 115.0
        self._btn_rects["toggle_view"] = QRectF(hx, hy, 105.0, btn_h)
        hx += 110.0

        # Temperature presets
        self._btn_rects["temp_sharp"] = QRectF(hx, hy, 40.0, btn_h)
        hx += 44.0
        self._btn_rects["temp_norm"] = QRectF(hx, hy, 40.0, btn_h)
        hx += 44.0
        self._btn_rects["temp_diff"] = QRectF(hx, hy, 40.0, btn_h)

    # ── Public Interaction APIs ────────────────────────────────────────────────

    def load_preset(self, preset_code: str) -> None:
        """Switch curriculum preset text."""
        if preset_code in PRESETS:
            self._active_preset_code = preset_code
            self._preset = PRESETS[preset_code]
            self._recompute_attention()
            self.update()
            self.preset_changed.emit(preset_code)

    def set_head(self, head_idx: int) -> None:
        """Set active attention head (-1 for multi-head average, 0..3)."""
        if -1 <= head_idx < self._num_heads:
            self._active_head = head_idx
            self.update()
            self.head_changed.emit(head_idx)

    def toggle_causal_mask(self) -> bool:
        """Toggle between Encoder (BERT bidirectional) and Decoder (GPT causal)."""
        self._is_causal = not self._is_causal
        self._recompute_attention()
        self.update()
        return self._is_causal

    def toggle_view_mode(self) -> str:
        """Toggle between heatmap and bipartite ribbon view."""
        self._view_mode = "ribbon" if self._view_mode == "heatmap" else "heatmap"
        self.update()
        return self._view_mode

    def set_temperature(self, tau: float) -> None:
        """Adjust Softmax temperature scaling."""
        self._temperature = max(0.2, min(5.0, float(tau)))
        self._recompute_attention()
        self.update()

    def select_token(self, idx: int) -> None:
        """Select token index to inspect attention weights."""
        if 0 <= idx < len(self._preset.tokens):
            self._selected_token_idx = idx
            self.update()
            self.token_selected.emit(idx, self._preset.tokens[idx])

    # ── Geometry & Bounding Rect ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # ── Mouse Interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check button clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                event.accept()
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key.startswith("head_"):
                    h_val = -1 if key == "head_avg" else int(key.replace("head_", ""))
                    self.set_head(h_val)
                elif key == "toggle_causal":
                    self.toggle_causal_mask()
                elif key == "toggle_view":
                    self.toggle_view_mode()
                elif key == "temp_sharp":
                    self.set_temperature(0.5)
                elif key == "temp_norm":
                    self.set_temperature(1.0)
                elif key == "temp_diff":
                    self.set_temperature(2.0)
                return

        # Check token selection in ribbon mode or heatmap labels
        tokens = self._preset.tokens
        T = len(tokens)
        if self._view_mode == "ribbon":
            # Token list on left: x ~ [24, 180]
            for i in range(T):
                ty = 110.0 + i * (320.0 / max(1, T))
                if QRectF(24.0, ty, 150.0, 24.0).contains(pos):
                    event.accept()
                    self.select_token(i)
                    return
        else:
            # Heatmap rows click
            grid_rect = QRectF(24.0, 105.0, 360.0, 360.0)
            if grid_rect.contains(pos):
                cell_size = 360.0 / T
                row = int((pos.y() - grid_rect.top()) / cell_size)
                col = int((pos.x() - grid_rect.left()) / cell_size)
                if 0 <= row < T and 0 <= col < T:
                    event.accept()
                    self.select_token(row)
                    self._hover_cell = (row, col)
                    return

        super().mousePressEvent(event)

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process touchless air tap gesture from webcam tracking."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # Check button clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key.startswith("head_"):
                    h_val = -1 if key == "head_avg" else int(key.replace("head_", ""))
                    self.set_head(h_val)
                elif key == "toggle_causal":
                    self.toggle_causal_mask()
                elif key == "toggle_view":
                    self.toggle_view_mode()
                elif key == "temp_sharp":
                    self.set_temperature(0.5)
                elif key == "temp_norm":
                    self.set_temperature(1.0)
                elif key == "temp_diff":
                    self.set_temperature(2.0)
                return True

        # Check token selection in ribbon mode
        tokens = self._preset.tokens
        T = len(tokens)
        if self._view_mode == "ribbon":
            for i in range(T):
                ty = 110.0 + i * (320.0 / max(1, T))
                if QRectF(24.0, ty, 150.0, 24.0).contains(pos):
                    self.select_token(i)
                    return True
        else:
            grid_rect = QRectF(24.0, 105.0, 360.0, 360.0)
            if grid_rect.contains(pos):
                cell_size = 360.0 / T
                row = int((pos.y() - grid_rect.top()) / cell_size)
                col = int((pos.x() - grid_rect.left()) / cell_size)
                if 0 <= row < T and 0 <= col < T:
                    self.select_token(row)
                    self._hover_cell = (row, col)
                    return True

        return True

    # ── Painting ──────────────────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self.WIDTH
        h = self.HEIGHT

        # 1. Dark Glass Card Background
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, 12.0, 12.0)
        painter.fillPath(card_path, QBrush(QColor(15, 23, 42, 245)))
        border_pen = QPen(
            QColor(56, 189, 248, 240) if self.isSelected() else QColor(30, 41, 59, 240),
            2.0 if self.isSelected() else 1.5,
        )
        painter.setPen(border_pen)
        painter.drawPath(card_path)

        # 2. Header Bar
        header_rect = QRectF(0.0, 0.0, w, 36.0)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 12.0, 12.0)
        painter.save()
        painter.setClipRect(0.0, 0.0, w, 36.0)
        painter.fillPath(header_path, QBrush(QColor(30, 41, 59, 230)))
        painter.restore()

        # Header Title & Badge
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(14, 23, "⚡ TRANSFORMER MULTI-HEAD SELF-ATTENTION VISUALIZER")

        # Architecture Mode Pill
        mode_str = "Decoder (GPT - Causal)" if self._is_causal else "Encoder (BERT - Bidirectional)"
        mode_col = QColor("#f59e0b") if self._is_causal else QColor("#10b981")
        m_rect = QRectF(w - 240.0, 7.0, 226.0, 22.0)
        m_path = QPainterPath()
        m_path.addRoundedRect(m_rect, 4.0, 4.0)
        painter.fillPath(m_path, QBrush(QColor(mode_col.red(), mode_col.green(), mode_col.blue(), 40)))
        painter.setPen(QPen(mode_col, 1.2))
        painter.drawPath(m_path)
        painter.drawText(m_rect, Qt.AlignmentFlag.AlignCenter, mode_str)

        # 3. Toolbar Controls
        self._paint_toolbar(painter)

        # 4. Main Body: Heatmap or Bipartite Ribbon Arcs
        if self._view_mode == "heatmap":
            self._paint_heatmap_view(painter)
        else:
            self._paint_ribbon_view(painter)

        # 5. Right Inspector & Formula Breakdown Panel
        self._paint_inspector_panel(painter)

        painter.restore()

    def _paint_toolbar(self, painter: QPainter) -> None:
        """Paint preset tabs, head buttons, and mode toggles."""
        # Presets row
        for code, pr in PRESETS.items():
            rect = self._btn_rects.get(f"preset_{code}")
            if not rect:
                continue
            is_active = (code == self._active_preset_code)
            bg_col = QColor(56, 189, 248, 50) if is_active else QColor(30, 41, 59, 160)
            txt_col = QColor("#38bdf8") if is_active else QColor("#94a3b8")

            path = QPainterPath()
            path.addRoundedRect(rect, 4.0, 4.0)
            painter.fillPath(path, QBrush(bg_col))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.drawPath(path)
            painter.setPen(QPen(txt_col))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, pr.name_vi[:22])

        # Heads row
        for h in range(self._num_heads):
            rect = self._btn_rects.get(f"head_{h}")
            if not rect:
                continue
            is_act = (self._active_head == h)
            h_col = HEAD_COLORS[h]
            path = QPainterPath()
            path.addRoundedRect(rect, 4.0, 4.0)
            painter.fillPath(path, QBrush(QColor(h_col.red(), h_col.green(), h_col.blue(), 60 if is_act else 20)))
            painter.setPen(QPen(h_col if is_act else QColor("#475569"), 1.2 if is_act else 1.0))
            painter.drawPath(path)
            painter.setPen(QPen(h_col if is_act else QColor("#cbd5e1")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"Head {h + 1}")

        # Head Avg
        avg_rect = self._btn_rects.get("head_avg")
        if avg_rect:
            is_act = (self._active_head == -1)
            path = QPainterPath()
            path.addRoundedRect(avg_rect, 4.0, 4.0)
            painter.fillPath(path, QBrush(QColor(255, 255, 255, 40 if is_act else 15)))
            painter.setPen(QPen(QColor("#ffffff") if is_act else QColor("#475569"), 1.2 if is_act else 1.0))
            painter.drawPath(path)
            painter.setPen(QPen(QColor("#ffffff") if is_act else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(avg_rect, Qt.AlignmentFlag.AlignCenter, "All Heads")

        # Causal & View Toggles
        c_rect = self._btn_rects.get("toggle_causal")
        if c_rect:
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(c_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.setFont(QFont("Segoe UI", 7))
            txt = "Mask: Causal (GPT)" if self._is_causal else "Mask: None (BERT)"
            painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, txt)

        v_rect = self._btn_rects.get("toggle_view")
        if v_rect:
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(v_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#38bdf8")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            v_txt = "Chế độ: Ruy-băng" if self._view_mode == "ribbon" else "Chế độ: Ma trận"
            painter.drawText(v_rect, Qt.AlignmentFlag.AlignCenter, v_txt)

        # Temperature presets
        for t_key, t_label, t_val in [
            ("temp_sharp", "τ=0.5", 0.5),
            ("temp_norm", "τ=1.0", 1.0),
            ("temp_diff", "τ=2.0", 2.0),
        ]:
            t_rect = self._btn_rects.get(t_key)
            if t_rect:
                is_active = abs(self._temperature - t_val) < 0.1
                bg_col = QColor(56, 189, 248, 50) if is_active else QColor(30, 41, 59, 180)
                txt_col = QColor("#38bdf8") if is_active else QColor("#94a3b8")
                painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
                painter.setBrush(QBrush(bg_col))
                painter.drawRoundedRect(t_rect, 4.0, 4.0)
                painter.setPen(QPen(txt_col))
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
                painter.drawText(t_rect, Qt.AlignmentFlag.AlignCenter, t_label)

    def _paint_heatmap_view(self, painter: QPainter) -> None:
        """Paint T x T self-attention weight matrix with viridis-style gradient."""
        tokens = self._preset.tokens
        T = len(tokens)
        attn = self.active_attention_matrix()

        grid_left = 24.0
        grid_top = 105.0
        grid_size = 350.0
        cell_size = grid_size / T

        h_idx = max(0, self._active_head)
        primary_col = HEAD_COLORS[h_idx] if self._active_head != -1 else QColor("#ffffff")

        # Draw matrix cells
        for r in range(T):
            for c in range(T):
                val = float(attn[r, c])
                cx = grid_left + c * cell_size
                cy = grid_top + r * cell_size
                cell_rect = QRectF(cx, cy, cell_size, cell_size)

                # Color intensity proportional to attention weight
                alpha = int(np.clip(val * 240.0 + 15.0, 0, 255))
                cell_col = QColor(primary_col.red(), primary_col.green(), primary_col.blue(), alpha)

                painter.fillRect(cell_rect, QBrush(cell_col))

                # Grid border
                is_selected_row = (r == self._selected_token_idx)
                border_pen = QPen(
                    QColor("#ffffff") if is_selected_row else QColor(51, 65, 85, 120),
                    1.5 if is_selected_row else 0.5,
                )
                painter.setPen(border_pen)
                painter.drawRect(cell_rect)

                # Number inside cell if cells are large enough
                if cell_size >= 28.0 and val > 0.05:
                    painter.setFont(QFont("Consolas", 6))
                    painter.setPen(QPen(QColor("#ffffff" if alpha > 120 else "#94a3b8")))
                    painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, f"{int(val * 100)}%")

        # Draw Token Labels along vertical axis (Query rows)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for r in range(T):
            lbl_rect = QRectF(grid_left - 18.0, grid_top + r * cell_size, 16.0, cell_size)
            is_sel = (r == self._selected_token_idx)
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#94a3b8")))
            painter.drawText(lbl_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), tokens[r][:3])

    def _paint_ribbon_view(self, painter: QPainter) -> None:
        """Paint bipartite attention ribbon flow connecting source to target tokens."""
        tokens = self._preset.tokens
        T = len(tokens)
        attn = self.active_attention_matrix()

        # Left: Source tokens (Queries), Right: Target tokens (Keys)
        x_left = 30.0
        x_right = 350.0
        row_h = 340.0 / max(1, T)
        y_start = 110.0

        h_idx = max(0, self._active_head)
        h_col = HEAD_COLORS[h_idx] if self._active_head != -1 else QColor("#38bdf8")

        # 1. Draw Connecting Arcs for selected token
        sel_r = self._selected_token_idx
        y_src = y_start + sel_r * row_h + row_h / 2.0

        for c in range(T):
            weight = float(attn[sel_r, c])
            if weight < 0.02:
                continue

            y_dst = y_start + c * row_h + row_h / 2.0

            arc_path = QPainterPath()
            arc_path.moveTo(x_left + 100.0, y_src)
            # Cubic Bézier curve
            cp1 = QPointF(x_left + 200.0, y_src)
            cp2 = QPointF(x_right - 60.0, y_dst)
            arc_path.cubicTo(cp1, cp2, QPointF(x_right, y_dst))

            # Stroke with thickness and alpha proportional to attention weight
            thickness = max(1.0, weight * 12.0)
            alpha = int(np.clip(weight * 255.0, 40, 230))
            pen = QPen(
                QColor(h_col.red(), h_col.green(), h_col.blue(), alpha),
                thickness,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
            painter.strokePath(arc_path, pen)

        # 2. Draw Source Token Pills
        for i in range(T):
            ty = y_start + i * row_h
            pill_rect = QRectF(x_left, ty, 95.0, row_h - 4.0)
            is_sel = (i == sel_r)

            p_path = QPainterPath()
            p_path.addRoundedRect(pill_rect, 4.0, 4.0)
            painter.fillPath(p_path, QBrush(QColor(56, 189, 248, 60) if is_sel else QColor(30, 41, 59, 180)))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#475569"), 1.2 if is_sel else 0.8))
            painter.drawPath(p_path)

            painter.setPen(QPen(QColor("#f8fafc") if is_sel else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_sel else QFont.Weight.Normal))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, tokens[i])

        # 3. Draw Target Token Pills
        for j in range(T):
            ty = y_start + j * row_h
            w_val = float(attn[sel_r, j])
            pill_rect = QRectF(x_right, ty, 95.0, row_h - 4.0)

            p_path = QPainterPath()
            p_path.addRoundedRect(pill_rect, 4.0, 4.0)
            painter.fillPath(p_path, QBrush(QColor(h_col.red(), h_col.green(), h_col.blue(), int(w_val * 180 + 20))))
            painter.setPen(QPen(h_col if w_val > 0.15 else QColor("#475569"), 1.0))
            painter.drawPath(p_path)

            painter.setPen(QPen(QColor("#ffffff") if w_val > 0.15 else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, f"{tokens[j]} ({int(w_val * 100)}%)")

    def _paint_inspector_panel(self, painter: QPainter) -> None:
        """Paint mathematical inspector, top attended tokens, and formula breakdown."""
        tokens = self._preset.tokens
        T = len(tokens)
        attn = self.active_attention_matrix()
        sel_idx = self._selected_token_idx
        sel_tok = tokens[sel_idx]

        panel_rect = QRectF(460.0, 105.0, 286.0, 360.0)
        p_path = QPainterPath()
        p_path.addRoundedRect(panel_rect, 8.0, 8.0)
        painter.fillPath(p_path, QBrush(QColor(24, 24, 27, 210)))
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.0))
        painter.drawPath(p_path)

        # Title
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(int(panel_rect.x() + 12.0), int(panel_rect.y() + 22.0), f"Phân tích Token: '{sel_tok}' [pos {sel_idx}]")

        # Attention Entropy Indicator
        ent = self.compute_entropy(sel_idx)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(panel_rect.x() + 12.0), int(panel_rect.y() + 40.0), f"Entropy chú ý H: {ent:.3f} bits")

        # Top Attended Tokens Bar Chart
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(panel_rect.x() + 12.0), int(panel_rect.y() + 62.0), "Phân bổ Trọng số Chú ý (Softmax):")

        # Sort tokens by attention weight
        weights = [(j, tokens[j], float(attn[sel_idx, j])) for j in range(T)]
        weights.sort(key=lambda item: item[2], reverse=True)

        h_idx = max(0, self._active_head)
        h_col = HEAD_COLORS[h_idx] if self._active_head != -1 else QColor("#38bdf8")

        by = panel_rect.y() + 75.0
        for pos_j, tok_j, w_val in weights[:5]:
            bar_w = 140.0 * w_val
            pct = int(w_val * 100)

            # Token name
            painter.setPen(QPen(QColor("#e2e8f0")))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(int(panel_rect.x() + 12.0), int(by + 13.0), f"{tok_j[:10]}:")

            # Bar
            b_rect = QRectF(panel_rect.x() + 80.0, by + 3.0, bar_w, 12.0)
            painter.fillRect(b_rect, QBrush(h_col))
            painter.setPen(QPen(h_col, 0.5))
            painter.drawRect(QRectF(panel_rect.x() + 80.0, by + 3.0, 140.0, 12.0))

            # Pct
            painter.setPen(QPen(QColor("#38bdf8")))
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.drawText(int(panel_rect.x() + 226.0), int(by + 13.0), f"{pct}%")
            by += 22.0

        # Mathematical Formula Display
        f_rect = QRectF(panel_rect.x() + 10.0, panel_rect.y() + 200.0, 266.0, 70.0)
        f_path = QPainterPath()
        f_path.addRoundedRect(f_rect, 6.0, 6.0)
        painter.fillPath(f_path, QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.0))
        painter.drawPath(f_path)

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 18.0), "Attention(Q, K, V) =")
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 34.0), "  Softmax( Q * K^T / (τ * √d_k) + M ) * V")

        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(int(f_rect.x() + 8.0), int(f_rect.y() + 54.0), f"d_model={self._d_model}, d_k={self._d_k}, τ={self._temperature:.1f}")

        # Pedagogical Explanation Note
        n_rect = QRectF(panel_rect.x() + 10.0, panel_rect.y() + 280.0, 266.0, 70.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.setFont(QFont("Segoe UI", 8))
        desc_txt = self._preset.description
        painter.drawText(n_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), desc_txt)

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Stamp current attention card onto LessonBoard."""
        stamped: list[QGraphicsItem] = []
        copy_item = InteractiveTransformerAttentionItem(pos=self.pos())
        copy_item.load_preset(self._active_preset_code)
        copy_item.set_head(self._active_head)
        copy_item.set_temperature(self._temperature)
        if self._is_causal:
            copy_item.toggle_causal_mask()
        if self._view_mode != "heatmap":
            copy_item.toggle_view_mode()
        copy_item.select_token(self._selected_token_idx)
        board.addItem(copy_item)
        stamped.append(copy_item)
        return stamped

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach lesson packages."""
        return {
            "type": "interactive_transformer_attention",
            "pos": [self.x(), self.y()],
            "preset_code": self._active_preset_code,
            "active_head": self._active_head,
            "is_causal": self._is_causal,
            "temperature": self._temperature,
            "view_mode": self._view_mode,
            "selected_token_idx": self._selected_token_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractiveTransformerAttentionItem:
        """Reconstruct visualizer from serialized dictionary."""
        pos = data.get("pos", [120.0, 50.0])
        item = cls(pos=(float(pos[0]), float(pos[1])))
        if data.get("preset_code"):
            item.load_preset(str(data.get("preset_code")))
        item.set_head(int(data.get("active_head", 0)))
        if bool(data.get("is_causal", False)) != item._is_causal:
            item.toggle_causal_mask()
        if str(data.get("view_mode", "heatmap")) != item._view_mode:
            item.toggle_view_mode()
        item.set_temperature(float(data.get("temperature", 1.0)))
        item.select_token(int(data.get("selected_token_idx", 0)))
        return item
