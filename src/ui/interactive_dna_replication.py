"""
AirTeach AI - Interactive Biology DNA Double Helix Replication & Base Pairing Simulator
========================================================================================
Provides an interactive biological simulation demonstrating:
  - Watson-Crick DNA complementary base pairing (Nguyên tắc bổ sung):
      A pairs with T (2 hydrogen bonds: A = T)
      G pairs with C (3 hydrogen bonds: G ≡ C)
  - Semi-conservative replication mechanism (Nguyên tắc bán bảo tồn).
  - DNA replication fork (Chạc ba tái bản) with Helicase unwinding parent strands (5' -> 3' and 3' -> 5').
  - Dynamic daughter strand synthesis with DNA Polymerase enzyme badge.
  - Interactive unzipping progress slider [0% to 100%].
  - Point mutation tester (Đột biến gen: thay thế cặp nucleotit).
  - Stamping permanent vector DNA diagrams onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
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

# Standard CPK / Biological Color Palette for Nitrogenous Bases
BASE_COLORS: dict[str, tuple[QColor, QColor, int]] = {
    # symbol: (fill_color, text_color, hydrogen_bonds)
    "A": (QColor("#3b82f6"), QColor("#ffffff"), 2),  # Adenine (Blue)
    "T": (QColor("#ef4444"), QColor("#ffffff"), 2),  # Thymine (Red)
    "G": (QColor("#f59e0b"), QColor("#ffffff"), 3),  # Guanine (Amber/Gold)
    "C": (QColor("#10b981"), QColor("#ffffff"), 3),  # Cytosine (Green)
}

COMPLEMENT_MAP: dict[str, str] = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
}

DEFAULT_PARENT_SEQUENCE = ["A", "T", "G", "C", "C", "G", "T", "A", "T", "A", "C", "G"]


class InteractiveDnaReplicationItem(QGraphicsObject):
    """
    Interactive DNA double helix replication simulator featuring a dynamic
    replication fork, complementary base pairing, and semi-conservative synthesis.
    """

    unzip_changed = Signal(float)  # (progress: 0.0 to 1.0)
    mutation_changed = Signal(bool) # (has_mutation)

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 660.0,
        height: float = 430.0,
        parent_sequence: Optional[list[str]] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Sequences: parent strand 1 (top) and parent strand 2 (bottom, complementary)
        self._seq1 = list(parent_sequence) if parent_sequence else list(DEFAULT_PARENT_SEQUENCE)
        self._seq2 = [COMPLEMENT_MAP.get(b, "A") for b in self._seq1]
        self._has_mutation = False
        self._mutation_idx = -1

        # Unzipping replication progress: 0.0 (fully zipped) to 1.0 (fully replicated)
        self._unzip_progress = 0.40
        self._is_auto_replicating = False

        # Replication animation timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(35)  # ~28 FPS
        self._anim_timer.timeout.connect(self._on_replicate_tick)

        # Interactive dragging state
        self._active_drag: Optional[str] = None  # "FORK", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_play = QRectF(self._width - 245.0, 10.0, 90.0, 26.0)
        self._rect_btn_mutate = QRectF(20.0, 10.0, 140.0, 26.0)
        self._rect_btn_reset = QRectF(168.0, 10.0, 100.0, 26.0)

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin for labels and glow."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Biological Mechanics & Calculations ─────────────────────────────────

    def sequence_length(self) -> int:
        """Returns the number of base pairs in the DNA sequence."""
        return len(self._seq1)

    def unzip_progress(self) -> float:
        """Returns current unzipping progress in [0.0, 1.0]."""
        return self._unzip_progress

    def is_auto_replicating(self) -> bool:
        """Returns True if the auto-replication animation is active."""
        return self._is_auto_replicating

    def has_mutation(self) -> bool:
        """Returns True if a point mutation is currently simulated."""
        return self._has_mutation

    def count_hydrogen_bonds(self) -> int:
        """
        Calculates total hydrogen bonds across the parent double helix:
          A=T has 2 bonds, G≡C has 3 bonds.
        """
        total = 0
        for b1, b2 in zip(self._seq1, self._seq2):
            if (b1, b2) in [("A", "T"), ("T", "A")]:
                total += 2
            elif (b1, b2) in [("G", "C"), ("C", "G")]:
                total += 3
            else:
                total += 1  # Mismatched mutation bond
        return total

    def base_pair_at(self, idx: int) -> tuple[str, str]:
        """Returns (base_parent1, base_parent2) at sequence index."""
        if 0 <= idx < len(self._seq1):
            return self._seq1[idx], self._seq2[idx]
        return "", ""

    # ─── Mutators & Controls ─────────────────────────────────────────────────

    def set_unzip_progress(self, progress: float) -> None:
        """Sets replication fork progress [0.0, 1.0]."""
        self._unzip_progress = max(0.0, min(1.0, float(progress)))
        self.unzip_changed.emit(self._unzip_progress)
        self.update()

    def toggle_auto_replication(self) -> bool:
        """Toggles automatic replication fork progression."""
        self._is_auto_replicating = not self._is_auto_replicating
        if self._is_auto_replicating:
            if self._unzip_progress >= 0.99:
                self._unzip_progress = 0.0
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_auto_replicating

    def _on_replicate_tick(self) -> None:
        """Timer callback advancing replication progress."""
        if self._is_auto_replicating:
            new_p = self._unzip_progress + 0.015
            if new_p >= 1.0:
                new_p = 1.0
                self._is_auto_replicating = False
                self._anim_timer.stop()
            self.set_unzip_progress(new_p)

    def trigger_point_mutation(self) -> None:
        """
        Simulates a point mutation (Đột biến điểm) by mutating a pair
        (e.g. replacing A-T with G-C, or inducing a mismatch).
        """
        self._has_mutation = True
        self._mutation_idx = 5  # Mutate 6th base pair
        # Introduce transversion A-T -> C-G
        self._seq1[self._mutation_idx] = "C"
        self._seq2[self._mutation_idx] = "G"
        self.mutation_changed.emit(True)
        self.update()

    def reset_sequence(self) -> None:
        """Restores wild-type DNA parent sequence and resets unzipping."""
        self._seq1 = list(DEFAULT_PARENT_SEQUENCE)
        self._seq2 = [COMPLEMENT_MAP.get(b, "A") for b in self._seq1]
        self._has_mutation = False
        self._mutation_idx = -1
        self._unzip_progress = 0.35
        self.mutation_changed.emit(False)
        self.update()

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive DNA replication simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height

        # 1. Container Frosted Glass Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Render Replication Fork & Strands
        self._draw_replication_fork(painter)

        # 3. Biology Theory HUD Card (Bottom Right)
        self._draw_biology_hud(painter)

        # 4. Interactive Control Buttons
        self._draw_buttons(painter)

        painter.restore()

    def _draw_replication_fork(self, painter: QPainter) -> None:
        """
        Renders parent strands, synthesized daughter strands, base pair rungs,
        and DNA Polymerase enzyme badge.
        """
        n_bp = len(self._seq1)
        x_start = 65.0
        x_end = self._width - 45.0
        step_x = (x_end - x_start) / float(n_bp)
        y_center = self._height * 0.50

        # Unzipping index threshold based on progress
        unzip_idx = self._unzip_progress * (n_bp - 1)

        # Backbone curve paths
        path_p1 = QPainterPath()  # Parent 1 (Top, 5' -> 3')
        path_p2 = QPainterPath()  # Parent 2 (Bottom, 3' -> 5')
        path_d1 = QPainterPath()  # Daughter 1 (complementary to parent 1)
        path_d2 = QPainterPath()  # Daughter 2 (complementary to parent 2)

        # Fork divergence amplitude
        max_div = 60.0

        for i in range(n_bp):
            bx = x_start + i * step_x

            # Separation offset: 0 if i > unzip_idx, increases as i < unzip_idx
            if i <= unzip_idx:
                sep = max_div * ((unzip_idx - i) / max(1.0, unzip_idx))
            else:
                sep = 0.0

            # Sine modulation for double helix twist
            twist = 20.0 * math.sin(i * 0.8)
            y_p1 = y_center - 28.0 - sep + twist * (0.0 if sep > 10.0 else 1.0)
            y_p2 = y_center + 28.0 + sep - twist * (0.0 if sep > 10.0 else 1.0)

            if i == 0:
                path_p1.moveTo(bx, y_p1)
                path_p2.moveTo(bx, y_p2)
            else:
                path_p1.lineTo(bx, y_p1)
                path_p2.lineTo(bx, y_p2)

            # Draw Base Pair Rung
            b1 = self._seq1[i]
            b2 = self._seq2[i]
            is_unzipped = (sep > 12.0)

            if not is_unzipped:
                # Intact double helix base pair with hydrogen bonds
                self._draw_intact_base_pair(painter, bx, y_p1, y_p2, b1, b2, i == self._mutation_idx)
            else:
                # Separated fork rungs with newly synthesized daughter bases
                self._draw_separated_base_pair(painter, bx, y_p1, y_p2, b1, b2, sep)

        # Draw Sugar-Phosphate Backbones
        # Parent 1: Blue backbone
        pen_p1 = QPen(QColor("#2563eb"), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_p1)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_p1)

        # Parent 2: Red backbone
        pen_p2 = QPen(QColor("#dc2626"), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_p2)
        painter.drawPath(path_p2)

        # Terminal labels: 5' and 3'
        f_term = QFont("Segoe UI", 9)
        f_term.setBold(True)
        painter.setFont(f_term)
        painter.setPen(QPen(QColor("#60a5fa")))
        painter.drawText(QPointF(20.0, y_center - 28.0 - max_div), "5' (Mẹ)")
        painter.drawText(QPointF(x_end + 10.0, y_center - 24.0), "3'")

        painter.setPen(QPen(QColor("#f87171")))
        painter.drawText(QPointF(20.0, y_center + 36.0 + max_div), "3' (Mẹ)")
        painter.drawText(QPointF(x_end + 10.0, y_center + 32.0), "5'")

        # Helicase / DNA Polymerase Enzyme Marker at fork junction
        fork_x = x_start + unzip_idx * step_x
        self._draw_enzyme_badge(painter, fork_x, y_center)

    def _draw_intact_base_pair(
        self,
        painter: QPainter,
        x: float,
        y1: float,
        y2: float,
        b1: str,
        b2: str,
        is_mutated: bool,
    ) -> None:
        """Draws paired bases connected by hydrogen bonds in the intact double helix."""
        c1, tc1, h_bonds = BASE_COLORS.get(b1, (QColor("#64748b"), QColor("#ffffff"), 2))
        c2, tc2, _ = BASE_COLORS.get(b2, (QColor("#64748b"), QColor("#ffffff"), 2))

        # Base 1 block
        rect_b1 = QRectF(x - 10.0, y1 + 4.0, 20.0, 18.0)
        painter.setPen(QPen(c1.darker(130), 1.2))
        painter.setBrush(QBrush(c1))
        painter.drawRoundedRect(rect_b1, 4.0, 4.0)

        # Base 2 block
        rect_b2 = QRectF(x - 10.0, y2 - 22.0, 20.0, 18.0)
        painter.setPen(QPen(c2.darker(130), 1.2))
        painter.setBrush(QBrush(c2))
        painter.drawRoundedRect(rect_b2, 4.0, 4.0)

        # Symbols
        f_sym = QFont("Segoe UI", 9)
        f_sym.setBold(True)
        painter.setFont(f_sym)
        painter.setPen(QPen(tc1))
        painter.drawText(rect_b1, Qt.AlignmentFlag.AlignCenter, b1)
        painter.setPen(QPen(tc2))
        painter.drawText(rect_b2, Qt.AlignmentFlag.AlignCenter, b2)

        # Hydrogen bond dashed rungs
        pen_hb = QPen(QColor("#cbd5e1") if not is_mutated else QColor("#f43f5e"), 1.4, Qt.PenStyle.DashLine)
        painter.setPen(pen_hb)
        gap_y1 = y1 + 22.0
        gap_y2 = y2 - 22.0

        if h_bonds == 2:
            # 2 bonds
            painter.drawLine(QPointF(x - 3.5, gap_y1), QPointF(x - 3.5, gap_y2))
            painter.drawLine(QPointF(x + 3.5, gap_y1), QPointF(x + 3.5, gap_y2))
        else:
            # 3 bonds (G-C)
            painter.drawLine(QPointF(x - 5.0, gap_y1), QPointF(x - 5.0, gap_y2))
            painter.drawLine(QPointF(x, gap_y1), QPointF(x, gap_y2))
            painter.drawLine(QPointF(x + 5.0, gap_y1), QPointF(x + 5.0, gap_y2))

        # Mutation alert indicator
        if is_mutated:
            painter.setPen(QPen(QColor("#ef4444"), 1.2))
            painter.setBrush(QBrush(QColor(239, 68, 68, 50)))
            painter.drawEllipse(QPointF(x, (y1 + y2) * 0.5), 14.0, 14.0)

    def _draw_separated_base_pair(
        self,
        painter: QPainter,
        x: float,
        y1: float,
        y2: float,
        b1: str,
        b2: str,
        sep: float,
    ) -> None:
        """Draws separated parent bases and newly synthesized daughter bases."""
        c1, tc1, _ = BASE_COLORS.get(b1, (QColor("#64748b"), QColor("#ffffff"), 2))
        c2, tc2, _ = BASE_COLORS.get(b2, (QColor("#64748b"), QColor("#ffffff"), 2))

        # Parent 1 Base
        rect_b1 = QRectF(x - 9.0, y1 + 3.0, 18.0, 16.0)
        painter.setPen(QPen(c1.darker(130), 1.2))
        painter.setBrush(QBrush(c1))
        painter.drawRoundedRect(rect_b1, 3.0, 3.0)

        # Parent 2 Base
        rect_b2 = QRectF(x - 9.0, y2 - 19.0, 18.0, 16.0)
        painter.setPen(QPen(c2.darker(130), 1.2))
        painter.setBrush(QBrush(c2))
        painter.drawRoundedRect(rect_b2, 3.0, 3.0)

        f_sym = QFont("Segoe UI", 8)
        f_sym.setBold(True)
        painter.setFont(f_sym)
        painter.setPen(QPen(tc1))
        painter.drawText(rect_b1, Qt.AlignmentFlag.AlignCenter, b1)
        painter.setPen(QPen(tc2))
        painter.drawText(rect_b2, Qt.AlignmentFlag.AlignCenter, b2)

        # Synthesized Daughter 1 Base (complementary to b1)
        d_b1 = COMPLEMENT_MAP.get(b1, "T")
        cd1, tcd1, _ = BASE_COLORS.get(d_b1, (QColor("#64748b"), QColor("#ffffff"), 2))
        rect_d1 = QRectF(x - 9.0, y1 + 24.0, 18.0, 16.0)
        painter.setPen(QPen(QColor("#38bdf8"), 1.4, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(cd1))
        painter.drawRoundedRect(rect_d1, 3.0, 3.0)
        painter.setPen(QPen(tcd1))
        painter.drawText(rect_d1, Qt.AlignmentFlag.AlignCenter, d_b1)

        # Synthesized Daughter 2 Base (complementary to b2)
        d_b2 = COMPLEMENT_MAP.get(b2, "A")
        cd2, tcd2, _ = BASE_COLORS.get(d_b2, (QColor("#64748b"), QColor("#ffffff"), 2))
        rect_d2 = QRectF(x - 9.0, y2 - 40.0, 18.0, 16.0)
        painter.setPen(QPen(QColor("#f43f5e"), 1.4, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(cd2))
        painter.drawRoundedRect(rect_d2, 3.0, 3.0)
        painter.setPen(QPen(tcd2))
        painter.drawText(rect_d2, Qt.AlignmentFlag.AlignCenter, d_b2)

    def _draw_enzyme_badge(self, painter: QPainter, x: float, y: float) -> None:
        """Renders the DNA Helicase & Polymerase enzyme badge with glowing aura."""
        # Helicase (Fork unzipper)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(168, 85, 247, 80)))
        painter.drawEllipse(QPointF(x, y), 22.0, 22.0)

        painter.setPen(QPen(QColor("#ffffff"), 1.8))
        painter.setBrush(QBrush(QColor("#9333ea")))
        painter.drawEllipse(QPointF(x, y), 16.0, 16.0)

        f_enz = QFont("Segoe UI", 8)
        f_enz.setBold(True)
        painter.setFont(f_enz)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QRectF(x - 16.0, y - 16.0, 32.0, 32.0), Qt.AlignmentFlag.AlignCenter, "Hel")

        # Label tag above
        painter.setPen(QPen(QColor("#c084fc")))
        painter.drawText(QPointF(x - 28.0, y - 24.0), "Helicase")

    def _draw_biology_hud(self, painter: QPainter) -> None:
        """Renders the Biological Theory & Base Pairing Rules HUD Card."""
        hud_w = 260.0
        hud_h = 100.0
        hud_x = self._width - hud_w - 18.0
        hud_y = self._height - hud_h - 18.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 8.0, hud_w - 20.0, 18.0), "Nguyên Tắc Tự Nhân Đôi DNA")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        painter.drawText(QRectF(hud_x + 10.0, hud_y + 28.0, hud_w - 20.0, 16.0), "Bổ sung: A = T (2 lk H) | G ≡ C (3 lk H)")
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 46.0, hud_w - 20.0, 16.0), "Bán bảo tồn: 1 mạch cũ mẹ + 1 mạch mới con")

        h_total = self.count_hydrogen_bonds()
        pct_unzip = int(self._unzip_progress * 100)
        line3 = f"Tiến độ tháo xoắn: {pct_unzip}% | Tổng liên kết H: {h_total}"
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 64.0, hud_w - 20.0, 16.0), line3)

        if self._has_mutation:
            painter.setPen(QPen(QColor("#f43f5e")))
            painter.drawText(QRectF(hud_x + 10.0, hud_y + 80.0, hud_w - 20.0, 16.0), "⚠️ Phát hiện đột biến điểm (Thay thế cặp nu)!")

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders interactive buttons: auto-replication, mutation test, reset, stamp, close."""
        f_btn = QFont("Segoe UI", 8)
        f_btn.setBold(True)
        painter.setFont(f_btn)

        # Close button [✕]
        painter.setPen(QPen(QColor("#ef4444"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
        painter.drawRoundedRect(self._rect_btn_close, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # Stamp button [📌 Đóng dấu]
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor(56, 189, 248, 40)))
        painter.drawRoundedRect(self._rect_btn_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0f2fe")))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # Play / Pause auto-replication [▶ Nhân đôi] / [⏸ Tạm dừng]
        play_txt = "⏸ Tạm dừng" if self._is_auto_replicating else "▶ Nhân đôi"
        play_col = QColor("#f59e0b") if self._is_auto_replicating else QColor("#10b981")
        painter.setPen(QPen(play_col, 1.2))
        painter.setBrush(QBrush(QColor(play_col.red(), play_col.green(), play_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_play, 6.0, 6.0)
        painter.setPen(QPen(play_col))
        painter.drawText(self._rect_btn_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # Mutation button [🧬 Thử đột biến]
        painter.setPen(QPen(QColor("#ec4899"), 1.2))
        painter.setBrush(QBrush(QColor(236, 72, 153, 40)))
        painter.drawRoundedRect(self._rect_btn_mutate, 6.0, 6.0)
        painter.setPen(QPen(QColor("#fbcfe8")))
        painter.drawText(self._rect_btn_mutate, Qt.AlignmentFlag.AlignCenter, "🧬 Thử đột biến điểm")

        # Reset button [↺ Khôi phục]
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
        painter.drawRoundedRect(self._rect_btn_reset, 6.0, 6.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "↺ Khôi phục")

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles unzipping slider drag or button clicks."""
        pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_play.contains(pos):
            self.toggle_auto_replication()
            event.accept()
            return

        if self._rect_btn_mutate.contains(pos):
            self.trigger_point_mutation()
            event.accept()
            return

        if self._rect_btn_reset.contains(pos):
            self.reset_sequence()
            event.accept()
            return

        if self._rect_btn_stamp.contains(pos):
            scene = self.scene()
            if scene is not None:
                board = getattr(scene, "board", None)
                if board is None and hasattr(scene, "views") and scene.views():
                    view = scene.views()[0]
                    board = view if hasattr(view, "addItem") else getattr(view, "board", None)
                if board is not None:
                    self.stamp_to_canvas(board)
            event.accept()
            return

        # Fork dragging horizontally
        self._active_drag = "FORK"
        self._update_fork_from_x(pos.x())
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates unzipping progress when dragging across horizontal axis."""
        if self._active_drag == "FORK":
            self._update_fork_from_x(event.pos().x())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases drag state."""
        self._active_drag = None
        super().mouseReleaseEvent(event)

    def _update_fork_from_x(self, px: float) -> None:
        """Maps x coordinate to unzip progress [0.0, 1.0]."""
        x_start = 65.0
        x_end = self._width - 45.0
        progress = (px - x_start) / max(1.0, x_end - x_start)
        self.set_unzip_progress(progress)

    # ─── Touchless Air Gesture Integration ───────────────────────────────────

    def set_air_hover_pos(self, scene_pos: Optional[QPointF]) -> None:
        """Sets touchless air pointer hover position."""
        if scene_pos is not None:
            self._air_hover_pos = self.mapFromScene(scene_pos)
        else:
            self._air_hover_pos = None
        self.update()

    def handle_air_drag(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch drag of replication fork."""
        pos = self.mapFromScene(scene_pos)
        self._update_fork_from_x(pos.x())
        return True

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Touchless air tap targeting control buttons."""
        pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(pos):
            self.close()
            return True

        if self._rect_btn_play.contains(pos):
            self.toggle_auto_replication()
            return True

        if self._rect_btn_mutate.contains(pos):
            self.trigger_point_mutation()
            return True

        if self._rect_btn_reset.contains(pos):
            self.reset_sequence()
            return True

        if self._rect_btn_stamp.contains(pos):
            target_board = board
            if target_board is None and self.scene() is not None:
                scene = self.scene()
                target_board = getattr(scene, "board", None)
            if target_board is not None:
                self.stamp_to_canvas(target_board)
                return True

        return False

    def close(self) -> None:
        """Stops animation timer and removes this interactive item from its scene."""
        self._anim_timer.stop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps vector DNA double helix and replication labels
        onto LessonBoard via BiologyTools.create_dna_double_helix.
        """
        from src.ui.stem_tools import BiologyTools

        origin = self.pos()
        y_center = origin.y() + self._height * 0.50
        x_start = origin.x() + 50.0
        x_end = origin.x() + self._width - 50.0

        stamped = BiologyTools.create_dna_double_helix(
            start=(x_start, y_center),
            end=(x_end, y_center),
            turns=2.0,
            amplitude=30.0,
            rungs=len(self._seq1),
            color1=QColor("#2563eb"),
            color2=QColor("#dc2626"),
            show_labels=True,
        )

        for itm in stamped:
            board.addItem(itm)

        return stamped
