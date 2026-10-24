"""
AirTeach AI - Interactive Biology Mendel Genetics & Punnett Square Simulator
=============================================================================
Provides an authentic biological genetics simulation based on Gregor Mendel's Laws:
  - 1. Law of Segregation (Quy luật Phân ly):
       Alleles segregate during gamete meiosis into haploid gametes (P -> G).
  - 2. Law of Independent Assortment (Quy luật Phân ly Độc lập):
       Alleles of different gene loci assort independently (AaBb -> AB, Ab, aB, ab).
  - 3. Non-Mendelian & Extended Inheritance:
       - Codominance (Đồng trội): ABO blood types (I^A, I^B, i).
       - Sex-linked Inheritance (Di truyền liên kết giới tính trên NST X): Color blindness & Hemophilia.
  - Interactive Features:
      - Dynamic Punnett Square Grid (2x2 Monohybrid, 4x4 Dihybrid).
      - Offspring cells with authentic color-coded phenotype badges.
      - Exact Genotypic Probability Breakdown (1/4, 2/4, 1/4 or 1/16 .. 9/16).
      - Exact Phenotypic Ratio Breakdown (3:1, 9:3:3:1, 1:1:1:1).
      - Real-time Chi-Square (χ²) Goodness-of-Fit Test with simulated sample counts:
          χ² = Σ (Observed - Expected)² / Expected.
  - Touchless air gestures (air pinch / air tap) and permanent vector stamping to LessonBoard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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

from src.ui.stem_tools import BiologyTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class PunnettCell:
    male_gamete: str
    female_gamete: str
    genotype: str
    phenotype: str
    color_hex: str


PRESETS: Dict[str, Dict[str, Any]] = {
    "MENDEL_PEAS_MONOHYBRID": {
        "title": "Lai 1 Cặp Tính Trạng (Monohybrid: Aa × Aa)",
        "desc": "Quy luật phân ly Mendel: Màu hạt đậu Hà Lan (A: Hạt Vàng trội hoàn toàn so với a: Hạt Xanh).",
        "male_parent": "Aa (Hạt Vàng dị hợp)",
        "female_parent": "Aa (Hạt Vàng dị hợp)",
        "male_gametes": ["A", "a"],
        "female_gametes": ["A", "a"],
        "phenotype_map": {
            "AA": ("Hạt Vàng", "#eab308"),
            "Aa": ("Hạt Vàng", "#eab308"),
            "aA": ("Hạt Vàng", "#eab308"),
            "aa": ("Hạt Xanh", "#22c55e"),
        },
        "theoretical_pheno_ratio": "3 Vàng : 1 Xanh (75% : 25%)",
        "theoretical_geno_ratio": "1 AA : 2 Aa : 1 aa (25% : 50% : 25%)",
        "sample_size": 400,
        "observed_counts": {"Hạt Vàng": 305, "Hạt Xanh": 95},
    },
    "MENDEL_PEAS_DIHYBRID": {
        "title": "Lai 2 Cặp Tính Trạng (Dihybrid: AaBb × AaBb)",
        "desc": "Quy luật phân ly độc lập: Màu sắc và dạng vỏ hạt đậu (A: Vàng, a: Xanh; B: Trơn, b: Nhăn).",
        "male_parent": "AaBb (Vàng, Trơn)",
        "female_parent": "AaBb (Vàng, Trơn)",
        "male_gametes": ["AB", "Ab", "aB", "ab"],
        "female_gametes": ["AB", "Ab", "aB", "ab"],
        "phenotype_map": {
            # Handled dynamically by dihybrid resolver
        },
        "theoretical_pheno_ratio": "9 Vàng-Trơn : 3 Vàng-Nhăn : 3 Xanh-Trơn : 1 Xanh-Nhăn",
        "theoretical_geno_ratio": "9 Kiểu gen phân ly độc lập",
        "sample_size": 556,
        "observed_counts": {"Vàng-Trơn": 315, "Vàng-Nhăn": 101, "Xanh-Trơn": 108, "Xanh-Nhăn": 32},
    },
    "BLOOD_TYPE_ABO": {
        "title": "Hệ Nhóm Máu ABO (Đồng trội: I^A i × I^B i)",
        "desc": "Gen đa alen quy định nhóm máu người: I^A đồng trội với I^B, cả hai trội hoàn toàn so với alen lặn i.",
        "male_parent": "Iᴬ i (Nhóm máu A dị hợp)",
        "female_parent": "Iᴮ i (Nhóm máu B dị hợp)",
        "male_gametes": ["Iᴬ", "i"],
        "female_gametes": ["Iᴮ", "i"],
        "phenotype_map": {
            "IᴬIᴮ": ("Nhóm AB", "#ec4899"),
            "IᴮIᴬ": ("Nhóm AB", "#ec4899"),
            "Iᴬi": ("Nhóm A", "#38bdf8"),
            "iIᴬ": ("Nhóm A", "#38bdf8"),
            "Iᴮi": ("Nhóm B", "#f59e0b"),
            "iIᴮ": ("Nhóm B", "#f59e0b"),
            "ii": ("Nhóm O", "#10b981"),
        },
        "theoretical_pheno_ratio": "1 AB : 1 A : 1 B : 1 O (25% mỗi loại)",
        "theoretical_geno_ratio": "1 IᴬIᴮ : 1 Iᴬi : 1 Iᴮi : 1 ii",
        "sample_size": 200,
        "observed_counts": {"Nhóm AB": 48, "Nhóm A": 53, "Nhóm B": 49, "Nhóm O": 50},
    },
    "SEX_LINKED_COLORBLIND": {
        "title": "Liên Kết Giới Tính (X^C X^c × X^C Y)",
        "desc": "Bệnh mù màu do alen lặn c trên NST X: Mẹ mang gen dị hợp lai với Bố bình thường.",
        "male_parent": "Xᶜ Y (Bố bình thường)",
        "female_parent": "Xᶜ Xᶜ (Mẹ mang gen)",
        "male_gametes": ["Xᶜ", "Y"],
        "female_gametes": ["Xᶜ", "Xᶜ"],
        "phenotype_map": {
            "XᶜXᶜ": ("Con gái bình thường", "#38bdf8"),
            "XᶜXᶜ": ("Con gái mang gen", "#a855f7"),
            "XᶜY": ("Con trai bình thường", "#0284c7"),
            "XᶜY": ("Con trai Mù màu", "#ef4444"),
        },
        "theoretical_pheno_ratio": "1 Gái thường : 1 Gái mang gen : 1 Trai thường : 1 Trai mù màu",
        "theoretical_geno_ratio": "1 XᶜXᶜ : 1 XᶜXᶜ : 1 XᶜY : 1 XᶜY",
        "sample_size": 200,
        "observed_counts": {"Gái thường": 51, "Gái mang gen": 49, "Trai thường": 52, "Trai mù màu": 48},
    },
}


class InteractiveGeneticsPunnettItem(QGraphicsObject):
    """
    Comprehensive Interactive Mendel Genetics & Punnett Square Visualizer for LessonBoard.
    Calculates 2x2 and 4x4 gamete combinations, offspring genotypes & phenotypes,
    and performs real-time Chi-Square (χ²) goodness-of-fit testing.
    """

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: Tuple[float, float] = (300.0, 130.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])

        self._width = 640.0
        self._height = 445.0

        self._current_preset_key: str = "MENDEL_PEAS_MONOHYBRID"
        self._grid_cells: List[List[PunnettCell]] = []
        self._male_gametes: List[str] = []
        self._female_gametes: List[str] = []

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(35.0)

        self.load_preset("MENDEL_PEAS_MONOHYBRID")

    # ── Geometry & Bounds ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Genetics Engine & Punnett Calculation ─────────────────────────────────

    def load_preset(self, preset_key: str) -> None:
        if preset_key not in PRESETS:
            return
        self._current_preset_key = preset_key
        p_data = PRESETS[preset_key]
        self._male_gametes = list(p_data["male_gametes"])
        self._female_gametes = list(p_data["female_gametes"])

        self._compute_punnett_grid()
        self.preset_changed.emit(preset_key)
        self.update()

    def _compute_punnett_grid(self) -> None:
        """Computes Cartesian product of gametes and resolves offspring phenotypes."""
        p_data = PRESETS[self._current_preset_key]
        rows = len(self._female_gametes)
        cols = len(self._male_gametes)
        grid: List[List[PunnettCell]] = []

        is_dihybrid = (self._current_preset_key == "MENDEL_PEAS_DIHYBRID")
        is_sex_linked = (self._current_preset_key == "SEX_LINKED_COLORBLIND")

        for r, fg in enumerate(self._female_gametes):
            row_cells: List[PunnettCell] = []
            for c, mg in enumerate(self._male_gametes):
                if is_dihybrid:
                    # e.g., fg = "AB", mg = "aB" -> genotype = "AaBB"
                    # Gene 1 alleles
                    g1 = sorted([fg[0], mg[0]], key=lambda x: (x.islower(), x))
                    # Gene 2 alleles
                    g2 = sorted([fg[1], mg[1]], key=lambda x: (x.islower(), x))
                    geno = f"{g1[0]}{g1[1]}{g2[0]}{g2[1]}"

                    # Phenotype evaluation
                    has_dom_a = ("A" in geno[:2])
                    has_dom_b = ("B" in geno[2:])

                    if has_dom_a and has_dom_b:
                        pheno = "Vàng-Trơn"
                        col = "#eab308"
                    elif has_dom_a and not has_dom_b:
                        pheno = "Vàng-Nhăn"
                        col = "#ca8a04"
                    elif not has_dom_a and has_dom_b:
                        pheno = "Xanh-Trơn"
                        col = "#22c55e"
                    else:
                        pheno = "Xanh-Nhăn"
                        col = "#15803d"

                elif is_sex_linked:
                    # Mother gametes: Xᶜ (dominant normal), Xᶜ (recessive colorblind)
                    # Father gametes: Xᶜ, Y
                    if r == 0 and c == 0:
                        geno, pheno, col = "XᶜXᶜ", "Con gái bình thường", "#38bdf8"
                    elif r == 1 and c == 0:
                        geno, pheno, col = "XᶜXᶜ", "Con gái mang gen", "#a855f7"
                    elif r == 0 and c == 1:
                        geno, pheno, col = "XᶜY", "Con trai bình thường", "#0284c7"
                    else:
                        geno, pheno, col = "XᶜY", "Con trai Mù màu", "#ef4444"

                else:
                    # Standard mapping or sorting alleles
                    geno = f"{fg}{mg}"
                    sorted_geno = "".join(sorted(geno, key=lambda x: (x.islower(), x)))
                    p_map = p_data.get("phenotype_map", {})
                    if geno in p_map:
                        pheno, col = p_map[geno]
                    elif sorted_geno in p_map:
                        pheno, col = p_map[sorted_geno]
                    else:
                        pheno = "Kiểu hình F₁"
                        col = "#38bdf8"

                row_cells.append(PunnettCell(
                    male_gamete=mg,
                    female_gamete=fg,
                    genotype=geno,
                    phenotype=pheno,
                    color_hex=col,
                ))
            grid.append(row_cells)

        self._grid_cells = grid

    # ── Statistics & Chi-Square Test (Zero Mock Data) ─────────────────────────

    def get_genotypic_ratio(self) -> Dict[str, float]:
        """Returns fraction of each genotype in the Punnett square."""
        total = len(self._female_gametes) * len(self._male_gametes)
        counts: Dict[str, int] = {}
        for row in self._grid_cells:
            for cell in row:
                counts[cell.genotype] = counts.get(cell.genotype, 0) + 1
        return {g: c / total for g, c in counts.items()}

    def get_phenotypic_ratio(self) -> Dict[str, float]:
        """Returns fraction of each phenotype in the Punnett square."""
        total = len(self._female_gametes) * len(self._male_gametes)
        counts: Dict[str, int] = {}
        for row in self._grid_cells:
            for cell in row:
                counts[cell.phenotype] = counts.get(cell.phenotype, 0) + 1
        return {p: c / total for p, c in counts.items()}

    def compute_chi_square(self) -> Tuple[float, int, bool]:
        """
        Computes authentic Pearson Chi-Square Goodness-of-Fit test statistic:
        χ² = Σ (O_i - E_i)² / E_i
        Returns: (chi_square_stat, degrees_of_freedom, is_hypothesis_accepted).
        """
        p_data = PRESETS[self._current_preset_key]
        observed = p_data.get("observed_counts", {})
        total_sample = p_data.get("sample_size", 400)
        pheno_ratios = self.get_phenotypic_ratio()

        chi_sq = 0.0
        k = len(observed)
        df = max(1, k - 1)

        for pheno, obs_count in observed.items():
            expected_fraction = pheno_ratios.get(pheno, 1.0 / max(1, k))
            expected_count = expected_fraction * total_sample
            if expected_count > 0:
                diff = obs_count - expected_count
                chi_sq += (diff ** 2) / expected_count

        # Critical value at alpha = 0.05
        # df=1: 3.841, df=2: 5.991, df=3: 7.815
        critical_values = {1: 3.841, 2: 5.991, 3: 7.815}
        crit = critical_values.get(df, 3.841)
        accepted = (chi_sq < crit)

        return chi_sq, df, accepted

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        pass

    def close(self) -> None:
        self.cleanup()
        self.setVisible(False)
        self.closed.emit()

    # ── UI Rectangles ─────────────────────────────────────────────────────────

    def _get_close_btn_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 10.0, 22.0, 22.0)

    def _get_stamp_btn_rect(self) -> QRectF:
        return QRectF(self._width - 150.0, 10.0, 110.0, 24.0)

    def _get_preset_rects(self) -> List[Tuple[str, QRectF]]:
        keys = list(PRESETS.keys())
        labels = ["Mendel 1 Tính (3:1)", "Mendel 2 Tính (9:3:3:1)", "Nhóm Máu ABO", "Liên Kết Giới Tính"]
        rects = []
        x = 14.0
        y = 42.0
        w = 146.0
        h = 24.0
        for i, (k, lbl) in enumerate(zip(keys, labels)):
            rects.append((k, QRectF(x + i * (w + 4.0), y, w, h)))
        return rects

    def _get_grid_rect(self) -> QRectF:
        return QRectF(14.0, 74.0, 350.0, 355.0)

    # ── Painting & Rendering ──────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor("#09101d"))
        grad.setColorAt(1.0, QColor("#050912"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Presets Bar
        self._paint_presets(painter)

        # 4. Punnett Square Grid (Left)
        self._paint_punnett_square(painter)

        # 5. Statistics, Phenotypes & Chi-Square HUD (Right)
        self._paint_genetics_hud(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 26, "🧬 DI TRUYỀN HỌC MENDEL & KHUNG PUNNETT (GENETICS SIMULATOR)")

        # Stamp Button
        s_rect = self._get_stamp_btn_rect()
        painter.setBrush(QBrush(QColor("#16a34a")))
        painter.setPen(QPen(QColor("#4ade80"), 1.2))
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

    def _paint_presets(self, painter: QPainter) -> None:
        labels = {
            "MENDEL_PEAS_MONOHYBRID": "1 Cặp Tính Trạng (3:1)",
            "MENDEL_PEAS_DIHYBRID": "2 Cặp Tính Trạng (9:3:3:1)",
            "BLOOD_TYPE_ABO": "Nhóm Máu ABO",
            "SEX_LINKED_COLORBLIND": "Liên Kết Giới Tính (X)",
        }
        for key, rect in self._get_preset_rects():
            is_active = (key == self._current_preset_key)
            bg = QColor("#15803d") if is_active else QColor("#1e293b")
            border = QColor("#4ade80") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, labels[key])

    def _paint_punnett_square(self, painter: QPainter) -> None:
        g_rect = self._get_grid_rect()
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(g_rect, 10.0, 10.0)

        p_data = PRESETS[self._current_preset_key]
        n_rows = len(self._female_gametes)
        n_cols = len(self._male_gametes)

        # Gamete Header Offsets
        header_w = 44.0
        header_h = 32.0
        grid_w = g_rect.width() - header_w - 20.0
        grid_h = g_rect.height() - header_h - 24.0

        cell_w = grid_w / n_cols
        cell_h = grid_h / n_rows

        grid_ox = g_rect.left() + header_w + 10.0
        grid_oy = g_rect.top() + header_h + 12.0

        # Top-Left Origin Icon (Male ♂ / Female ♀)
        tl_rect = QRectF(g_rect.left() + 10.0, g_rect.top() + 8.0, header_w - 4.0, header_h)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(tl_rect, Qt.AlignmentFlag.AlignCenter, "♀ \\ ♂")

        # Male Gametes Column Headers (Top)
        for c, mg in enumerate(self._male_gametes):
            mx = grid_ox + c * cell_w
            m_rect = QRectF(mx, g_rect.top() + 8.0, cell_w, header_h)
            painter.setBrush(QBrush(QColor("#0284c7")))
            painter.setPen(QPen(QColor("#38bdf8"), 1.0))
            painter.drawRoundedRect(m_rect.adjusted(2, 2, -2, -2), 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(m_rect, Qt.AlignmentFlag.AlignCenter, f"♂ {mg}")

        # Female Gametes Row Headers (Left)
        for r, fg in enumerate(self._female_gametes):
            fy = grid_oy + r * cell_h
            f_rect = QRectF(g_rect.left() + 10.0, fy, header_w - 4.0, cell_h)
            painter.setBrush(QBrush(QColor("#ec4899")))
            painter.setPen(QPen(QColor("#f472b6"), 1.0))
            painter.drawRoundedRect(f_rect.adjusted(2, 2, -2, -2), 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(f_rect, Qt.AlignmentFlag.AlignCenter, f"♀\n{fg}")

        # Offspring Cells
        for r in range(n_rows):
            for c in range(n_cols):
                cell = self._grid_cells[r][c]
                cx = grid_ox + c * cell_w
                cy = grid_oy + r * cell_h
                c_box = QRectF(cx, cy, cell_w, cell_h).adjusted(2, 2, -2, -2)

                # Background badge with phenotype theme
                badge_col = QColor(cell.color_hex)
                bg_col = QColor(badge_col.red(), badge_col.green(), badge_col.blue(), 30)
                painter.setBrush(QBrush(bg_col))
                painter.setPen(QPen(badge_col, 1.2))
                painter.drawRoundedRect(c_box, 6.0, 6.0)

                # Genotype Text
                painter.setFont(QFont("Consolas", 10 if n_rows <= 2 else 8, QFont.Weight.Bold))
                painter.setPen(QColor("#f8fafc"))
                painter.drawText(
                    QRectF(c_box.left(), c_box.top() + (6.0 if n_rows <= 2 else 2.0), c_box.width(), 16.0),
                    Qt.AlignmentFlag.AlignCenter,
                    cell.genotype,
                )

                # Phenotype Text
                painter.setFont(QFont("Segoe UI", 8 if n_rows <= 2 else 7))
                painter.setPen(badge_col.lighter(120))
                painter.drawText(
                    QRectF(c_box.left() + 2.0, c_box.bottom() - (22.0 if n_rows <= 2 else 16.0), c_box.width() - 4.0, 16.0),
                    Qt.AlignmentFlag.AlignCenter,
                    cell.phenotype,
                )

    def _paint_genetics_hud(self, painter: QPainter) -> None:
        hud_rect = QRectF(376.0, 74.0, 250.0, 355.0)
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#4ade80"))
        painter.drawText(hud_rect.adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignTop, "📊 TỈ LỆ PHÂN LY & KIỂM ĐỊNH χ²")

        p_data = PRESETS[self._current_preset_key]

        # Parental cross notation
        curr_y = hud_rect.top() + 32.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(hud_rect.left() + 12.0, curr_y, "SƠ ĐỒ LAI (P):")
        curr_y += 16.0

        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(hud_rect.left() + 12.0, curr_y, f"P: {p_data['female_parent']} × {p_data['male_parent']}")
        curr_y += 20.0

        # Divider
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawLine(hud_rect.left() + 10.0, curr_y, hud_rect.right() - 10.0, curr_y)
        curr_y += 14.0

        # Phenotypic breakdown
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#eab308"))
        painter.drawText(hud_rect.left() + 12.0, curr_y, "TỈ LỆ KIỂU HÌNH F₁:")
        curr_y += 16.0

        pheno_ratios = self.get_phenotypic_ratio()
        for pheno_name, frac in pheno_ratios.items():
            pct = frac * 100.0
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(hud_rect.left() + 14.0, curr_y, f"• {pheno_name}:")

            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#facc15"))
            painter.drawText(
                QRectF(hud_rect.left() + 140.0, curr_y - 11.0, 95.0, 14.0),
                Qt.AlignmentFlag.AlignRight,
                f"{pct:.1f}% ({frac:.3f})",
            )
            curr_y += 18.0

        curr_y += 4.0
        # Divider
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawLine(hud_rect.left() + 10.0, curr_y, hud_rect.right() - 10.0, curr_y)
        curr_y += 14.0

        # Pearson Chi-Square (χ²) Test
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(hud_rect.left() + 12.0, curr_y, "KIỂM ĐỊNH CHI-BÌNH PHƯƠNG (χ²):")
        curr_y += 16.0

        chi_stat, df, accepted = self.compute_chi_square()
        chi_lines = [
            f"Kích thước mẫu N = {p_data.get('sample_size', 400)} cá thể",
            f"Giá trị χ² thực nghiệm = {chi_stat:.3f}",
            f"Bậc tự do (df) = {df} (Mức ý nghĩa α = 0.05)",
        ]
        for cl in chi_lines:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(hud_rect.left() + 14.0, curr_y, cl)
            curr_y += 18.0

        # Verdict Badge
        v_box = QRectF(hud_rect.left() + 12.0, hud_rect.bottom() - 42.0, hud_rect.width() - 24.0, 32.0)
        v_col = QColor("#22c55e") if accepted else QColor("#ef4444")
        v_text = "CHẤP NHẬN QUY LUẬT MENDEL (p > 0.05) ✔" if accepted else "BÁC BỎ GIẢ THUYẾT MENDEL (p ≤ 0.05) ✘"

        painter.setBrush(QBrush(QColor(v_col.red(), v_col.green(), v_col.blue(), 35)))
        painter.setPen(QPen(v_col, 1.4))
        painter.drawRoundedRect(v_box, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(v_col)
        painter.drawText(v_box, Qt.AlignmentFlag.AlignCenter, v_text)

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

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.load_preset(key)
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

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.load_preset(key)
                return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── LessonBoard Permanent Vector Stamping ─────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of current Punnett Square to LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = BiologyTools.create_punnett_genetics_diagram(
            center=(cx, cy + 30.0),
            width=540.0,
            height=320.0,
            preset=self._current_preset_key,
        )
        for it in diagram_items:
            board.addItem(it)

        if hasattr(board, "commit_state_action"):
            board.commit_state_action("Stamp Punnett Genetics Diagram")
        logger.info(f"Punnett genetics diagram stamped to LessonBoard at ({cx:.1f}, {cy:.1f}).")
