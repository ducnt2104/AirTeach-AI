"""
AirTeach AI – Interactive Biology & Genetics: Mendelian Inheritance, Punnett Square & Gene Linkage Simulator
=============================================================================================================
Provides an authentic biological genetics simulation of classical and modern heredity:
  - Gregor Mendel's Laws of Inheritance:
      1. Law of Segregation (Quy luật Phân ly):
         Monohybrid cross Aa × Aa yields gametes (1/2 A, 1/2 a).
         Zygote Genotype Ratio: 1 AA : 2 Aa : 1 aa (25% homozygous dominant, 50% heterozygous, 25% homozygous recessive).
         Phenotype Ratio: 3 Dominant : 1 Recessive (Complete dominance).
      2. Law of Independent Assortment (Quy luật Phân ly độc lập):
         Dihybrid cross AaBb × AaBb yields 4 gamete types each (AB, Ab, aB, ab).
         16-cell Punnett Square with 9 genotypes yielding classical phenotypic ratio:
         9 A_B_ (Dominant-Dominant) : 3 A_bb (Dominant-Recessive) : 3 aaB_ (Recessive-Dominant) : 1 aabb (Double Recessive).
      3. Incomplete Dominance (Trội không hoàn toàn):
         Heterozygote exhibits intermediate blended phenotype (e.g., Red RR × White WW -> 100% Pink RW).
         Phenotype ratio matches genotype ratio: 1 Red : 2 Pink : 1 White.
      4. Codominance & Human ABO Blood System (Đồng trội & Nhóm máu ABO):
         Alleles I^A, I^B, I^O (A and B are codominant, O is recessive).
         Phenotypes: Blood A (I^A I^A, I^A I^O), Blood B (I^B I^B, I^B I^O), Blood AB (I^A I^B), Blood O (I^O I^O).
      5. Thomas Hunt Morgan's Gene Linkage & Crossing Over (Liên kết gen & Hoán vị gen):
         Recombination frequency f = (Recombinant offspring / Total offspring) * 100%.
         Mapping distance measured in centiMorgans (cM).
  - 5 Curriculum Presets:
      1. MONOHYBRID_PEA_PLANT: Mendel's classic pea shape (Smooth S vs Wrinkled s), 3:1 ratio.
      2. DIHYBRID_INDEPENDENT_9_3_3_1: Pea shape (Y/y) & Color (R/r), 16-cell Punnett square (9:3:3:1).
      3. INCOMPLETE_DOMINANCE_FLOWERS: Snapdragon petal color (RR Red × WW White -> RW Pink), 1:2:1 ratio.
      4. CODOMINANCE_ABO_BLOOD: Human ABO blood type inheritance with codominance (I^A, I^B, I^O).
      5. MORGAN_SEX_LINKED_HEMOPHILIA: X-linked recessive genetic disorder (X^H / X^h) across generations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GeneticsPreset:
    code: str
    title_vi: str
    badge_label: str
    parent_male: str
    parent_female: str
    is_dihybrid: bool
    description: str


PRESETS: dict[str, GeneticsPreset] = {
    "MONOHYBRID_PEA_PLANT": GeneticsPreset(
        code="MONOHYBRID_PEA_PLANT",
        title_vi="Quy luật Phân ly Mendel (Lai một cặp tính trạng Đậu Hà Lan)",
        badge_label="PHÂN LY 3:1",
        parent_male="Aa",
        parent_female="Aa",
        is_dihybrid=False,
        description="Mendel lai hai cây đậu F₁ dị hợp hạt vàng (Aa × Aa). Tỷ lệ kiểu gen 1 AA : 2 Aa : 1 aa; Tỷ lệ kiểu hình 3 Hạt vàng : 1 Hạt xanh.",
    ),
    "DIHYBRID_INDEPENDENT_9_3_3_1": GeneticsPreset(
        code="DIHYBRID_INDEPENDENT_9_3_3_1",
        title_vi="Quy luật Phân ly Độc lập (Lai 2 cặp tính trạng 9:3:3:1)",
        badge_label="ĐỘC LẬP 9:3:3:1",
        parent_male="AaBb",
        parent_female="AaBb",
        is_dihybrid=True,
        description="Lai hai cặp tính trạng độc lập AaBb × AaBb tạo lưới Punnett 16 ô với tỷ lệ kiểu hình kinh điển 9 Vàng-Trơn : 3 Vàng-Nhăn : 3 Xanh-Trơn : 1 Xanh-Nhăn.",
    ),
    "INCOMPLETE_DOMINANCE_FLOWERS": GeneticsPreset(
        code="INCOMPLETE_DOMINANCE_FLOWERS",
        title_vi="Trội Không Hoàn toàn (Màu hoa Mõm chó R_W)",
        badge_label="TRỘI KHÔNG HOÀN TOÀN",
        parent_male="RW",
        parent_female="RW",
        is_dihybrid=False,
        description="Trội lặn không hoàn toàn giữa alen Đỏ (R) và Trắng (W). Cá thể dị hợp RW biểu hiện màu Hồng trung gian, tỷ lệ kiểu hình trùng kiểu gen 1 Đỏ : 2 Hồng : 1 Trắng.",
    ),
    "CODOMINANCE_ABO_BLOOD": GeneticsPreset(
        code="CODOMINANCE_ABO_BLOOD",
        title_vi="Đồng trội & Đa alen (Hệ Nhóm máu ABO ở Người)",
        badge_label="NHÓM MÁU ABO",
        parent_male="AB",                   # I^A I^B
        parent_female="AO",                 # I^A I^O
        is_dihybrid=False,
        description="Hệ nhóm máu ABO do 3 alen I^A, I^B, I^O quy định. I^A và I^B đồng trội hoàn toàn so với I^O lặn, giải thích di truyền nhóm máu từ cha mẹ sang con cái.",
    ),
    "MORGAN_SEX_LINKED_HEMOPHILIA": GeneticsPreset(
        code="MORGAN_SEX_LINKED_HEMOPHILIA",
        title_vi="Di truyền Liên kết Giới tính (Bệnh Máu khó đông Hemophilia)",
        badge_label="LIÊN KẾT GIỚI TÍNH",
        parent_male="XhY",                  # Diseased Male
        parent_female="XHXh",               # Carrier Female
        is_dihybrid=False,
        description="Gen bệnh máu khó đông nằm trên NST giới tính X không có alen tương ứng trên Y. Tỷ lệ phân ly bệnh biểu hiện lệch giữa con trai và con gái.",
    ),
}


class InteractivePunnettGeneticsItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Biological Genetics:
    Mendelian Inheritance, Monohybrid/Dihybrid Punnett Squares, and Codominance.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    preset_changed = Signal(str)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
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
        )

        # Simulation Preset & Genetic Configuration
        self._active_preset_code: str = "MONOHYBRID_PEA_PLANT"
        self._preset: GeneticsPreset = PRESETS[self._active_preset_code]

        self._parent_male: str = self._preset.parent_male
        self._parent_female: str = self._preset.parent_female

        # Selected Punnett cell for interactive inspection (-1 = none)
        self._selected_cell_idx: int = -1

        # Animation state
        self._pulse_alpha: float = 0.0
        self._timer = QTimer()
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

    # -------------------------------------------------------------------------
    # Genetics Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    def get_male_gametes(self) -> list[str]:
        """Extracts gametes produced by male parent."""
        p = self._parent_male
        if self._preset.is_dihybrid:
            # e.g., AaBb -> AB, Ab, aB, ab
            a1, a2 = p[0], p[1]
            b1, b2 = p[2], p[3]
            return [f"{a1}{b1}", f"{a1}{b2}", f"{a2}{b1}", f"{a2}{b2}"]
        else:
            # Monohybrid e.g. Aa -> A, a
            if len(p) == 2:
                return [p[0], p[1]]
            elif p.startswith("X"):
                # e.g. XhY -> Xh, Y
                return ["Xʰ", "Y"]
            return ["A", "a"]

    def get_female_gametes(self) -> list[str]:
        """Extracts gametes produced by female parent."""
        p = self._parent_female
        if self._preset.is_dihybrid:
            a1, a2 = p[0], p[1]
            b1, b2 = p[2], p[3]
            return [f"{a1}{b1}", f"{a1}{b2}", f"{a2}{b1}", f"{a2}{b2}"]
        else:
            if len(p) == 2:
                return [p[0], p[1]]
            elif p.startswith("X"):
                # e.g. XHXh -> XH, Xh
                return ["Xᴴ", "Xʰ"]
            return ["A", "a"]

    def combine_gametes(self, g_male: str, g_female: str) -> str:
        """Combines two gametes into a zygote genotype in standardized canonical order."""
        if self._preset.is_dihybrid:
            # g_male has a,b; g_female has a,b
            a_pair = sorted([g_male[0], g_female[0]])
            b_pair = sorted([g_male[1], g_female[1]])
            return f"{a_pair[0]}{a_pair[1]}{b_pair[0]}{b_pair[1]}"
        else:
            if "X" in g_male or "X" in g_female:
                # Sex linked
                return f"{g_female}{g_male}"
            return "".join(sorted([g_male, g_female]))

    def calculate_punnett_grid(self) -> list[dict[str, str]]:
        """Calculates 4-cell or 16-cell Punnett square matrix."""
        m_gametes = self.get_male_gametes()
        f_gametes = self.get_female_gametes()
        grid = []
        for i_f, g_f in enumerate(f_gametes):
            for i_m, g_m in enumerate(m_gametes):
                zyg = self.combine_gametes(g_m, g_f)
                grid.append({
                    "male_gamete": g_m,
                    "female_gamete": g_f,
                    "zygote": zyg,
                    "row": i_f,
                    "col": i_m,
                })
        return grid

    def calculate_phenotype_distribution(self) -> dict[str, float]:
        """Calculates exact statistical phenotype percentages."""
        grid = self.calculate_punnett_grid()
        total = float(len(grid))
        counts: dict[str, int] = {}

        for cell in grid:
            zyg = cell["zygote"]
            pt = self._resolve_phenotype(zyg)
            counts[pt] = counts.get(pt, 0) + 1

        return {k: (v / total) * 100.0 for k, v in counts.items()}

    def _resolve_phenotype(self, zygote: str) -> str:
        """Determines morphological phenotype based on active genetic model."""
        code = self._active_preset_code
        if code == "MONOHYBRID_PEA_PLANT":
            return "Vàng (Trội A)" if "A" in zygote else "Xanh (Lặn a)"
        elif code == "DIHYBRID_INDEPENDENT_9_3_3_1":
            has_a = "A" in zygote[0:2]
            has_b = "B" in zygote[2:4]
            if has_a and has_b:
                return "Vàng - Trơn (9/16)"
            elif has_a and not has_b:
                return "Vàng - Nhăn (3/16)"
            elif not has_a and has_b:
                return "Xanh - Trơn (3/16)"
            else:
                return "Xanh - Nhăn (1/16)"
        elif code == "INCOMPLETE_DOMINANCE_FLOWERS":
            if zygote == "RR":
                return "Hoa Đỏ (RR)"
            elif zygote == "WW":
                return "Hoa Trắng (WW)"
            else:
                return "Hoa Hồng (RW)"
        elif code == "CODOMINANCE_ABO_BLOOD":
            if "A" in zygote and "B" in zygote:
                return "Nhóm máu AB (Đồng trội)"
            elif "A" in zygote:
                return "Nhóm máu A"
            elif "B" in zygote:
                return "Nhóm máu B"
            else:
                return "Nhóm máu O (Lặn)"
        elif code == "MORGAN_SEX_LINKED_HEMOPHILIA":
            if "Y" in zygote:
                return "Nam Bệnh (XʰY)" if "Xʰ" in zygote else "Nam Bình thường (XᴴY)"
            else:
                if zygote == "XʰXʰ":
                    return "Nữ Bệnh (XʰXʰ)"
                elif "Xʰ" in zygote:
                    return "Nữ Mang gen bệnh (XᴴXʰ)"
                else:
                    return "Nữ Bình thường (XᴴXᴴ)"
        return zygote

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> GeneticsPreset:
        return self._preset

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum genetics preset and sets up parents."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._parent_male = self._preset.parent_male
        self._parent_female = self._preset.parent_female
        self._selected_cell_idx = -1
        self.preset_changed.emit(preset_code)
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()
        scene = self.scene()
        if scene:
            from src.ui.stem_tools import BiologyTools
            cx = self.x() + self.WIDTH / 2.0
            cy = self.y() + self.HEIGHT + 40.0
            cross_str = f"{self.active_preset.parent_male} × {self.active_preset.parent_female}"
            ratio_str = getattr(self.active_preset, "badge_label", "3:1")
            items = BiologyTools.create_punnett_genetics_diagram(
                center=(cx, cy),
                preset_code=self.active_preset.code,
                parent_cross=cross_str,
                phenotype_ratio=ratio_str,
            )
            for itm in items:
                scene.addItem(itm)

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._pulse_alpha = (math.sin(self._pulse_alpha + 0.1) + 1.0) / 2.0
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#080c14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Punnett Square & Parent Gametes)
        left_w = 430.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_punnett_viewport(painter, view_rect)

        # 4. Right Statistical Ratio & Genetic Analysis Panel
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Genetics Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "SINH HỌC DI TRUYỀN: QUY LUẬT MENDEL & KHUNG PUNNETT")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Phân ly độc lập • Khung Punnett • Trội không hoàn toàn • Đồng trội ABO • Liên kết giới tính")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_punnett_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # Draw Cross Banner (P: Male × Female)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        p_txt = f"PHÉP LAI P: ♂ {self._parent_male}  ×  ♀ {self._parent_female}"
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), p_txt)

        # Punnett Square Grid Dimensions
        is_dihybrid = self._preset.is_dihybrid
        n_dim = 4 if is_dihybrid else 2

        grid_ox = rect.left() + 70.0
        grid_oy = rect.top() + 65.0
        grid_sz = 220.0
        cell_sz = grid_sz / float(n_dim)

        m_gametes = self.get_male_gametes()
        f_gametes = self.get_female_gametes()

        # 1. Male Gametes Headers (Top row)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(grid_ox - 45.0, grid_oy - 12.0), "G_♂ ➔")

        for col, g_m in enumerate(m_gametes):
            cx = grid_ox + col * cell_sz
            hdr_r = QRectF(cx, grid_oy - 24.0, cell_sz, 20.0)
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8"), 1.0))
            painter.drawRoundedRect(hdr_r, 3.0, 3.0)
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(hdr_r, Qt.AlignmentFlag.AlignCenter, g_m)

        # 2. Female Gametes Headers (Left column)
        painter.setPen(QColor("#ec4899"))
        painter.drawText(QPointF(grid_ox - 45.0, grid_oy + 14.0), "G_♀")

        for row, g_f in enumerate(f_gametes):
            cy = grid_oy + row * cell_sz
            hdr_r = QRectF(grid_ox - 32.0, cy, 26.0, cell_sz)
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#ec4899"), 1.0))
            painter.drawRoundedRect(hdr_r, 3.0, 3.0)
            painter.setPen(QColor("#ec4899"))
            painter.drawText(hdr_r, Qt.AlignmentFlag.AlignCenter, g_f)

        # 3. Zygote Cells Grid
        grid_data = self.calculate_punnett_grid()
        for idx, cell in enumerate(grid_data):
            r = cell["row"]
            c = cell["col"]
            cell_x = grid_ox + c * cell_sz
            cell_y = grid_oy + r * cell_sz
            c_rect = QRectF(cell_x, cell_y, cell_sz, cell_sz)

            is_sel = (idx == self._selected_cell_idx)
            # Background color by phenotype
            zyg = cell["zygote"]
            pt = self._resolve_phenotype(zyg)

            # Cell fill
            if "Trơn" in pt or "Vàng (" in pt or "RR" in pt or "AB (" in pt:
                bg_col = QColor(251, 191, 36, 45)  # Gold
                border_col = QColor("#fbbf24")
            elif "Nhăn" in pt or "Xanh (" in pt or "WW" in pt or "O (" in pt:
                bg_col = QColor(56, 189, 248, 40)   # Cyan
                border_col = QColor("#38bdf8")
            elif "Hồng" in pt or "Bệnh" in pt:
                bg_col = QColor(236, 72, 153, 45)   # Pink
                border_col = QColor("#ec4899")
            else:
                bg_col = QColor(34, 197, 94, 40)    # Green
                border_col = QColor("#22c55e")

            if is_sel:
                bg_col = QColor(14, 165, 233, 100)
                border_col = QColor("#38bdf8")

            painter.setBrush(QBrush(bg_col))
            painter.setPen(QPen(border_col, 1.5 if is_sel else 0.8))
            painter.drawRoundedRect(c_rect, 3.0, 3.0)

            # Genotype Text
            f_size = 9 if not is_dihybrid else 7
            painter.setFont(QFont("Segoe UI", f_size, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, zyg)

        painter.restore()

    # -------------------------------------------------------------------------
    # Right Statistical Ratio & Genetic Analysis Panel
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "TỶ LỆ PHÂN LY DI TRUYỀN (F₁/F₂):")

        # Summary Badge Card
        y += 18.0
        card_r = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(card_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        n_cells = 16 if self._preset.is_dihybrid else 4
        painter.drawText(QRectF(ox + 8.0, y, 70.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"{n_cells} Tổ hợp")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(
            QRectF(ox + 80.0, y + 2.0, w - 85.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter,
            f"Số kiểu gen: {'9' if self._preset.is_dihybrid else '3'} loại\n{self._preset.badge_label}",
        )

        # Phenotype Distribution Table
        y += 42.0
        dist = self.calculate_phenotype_distribution()

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "PHÂN BỐ KIỂU HÌNH THỰC TẾ:")

        y += 16.0
        colors = [QColor("#fbbf24"), QColor("#38bdf8"), QColor("#22c55e"), QColor("#ec4899"), QColor("#cbd5e1")]

        for i, (p_name, pct) in enumerate(dist.items()):
            col = colors[i % len(colors)]
            # Draw progress bar
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), f"• {p_name}: {pct:.1f}%")

            bar_r = QRectF(ox, y + 4.0, w, 4.0)
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_r, 2.0, 2.0)

            fill_r = QRectF(ox, y + 4.0, w * (pct / 100.0), 4.0)
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(fill_r, 2.0, 2.0)

            y += 24.0

        # Action Buttons
        y = rect.bottom() - 32.0
        stamp_btn = QRectF(ox, y, w, 22.0)
        self._btn_rects["action_stamp"] = stamp_btn

        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(stamp_btn, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_btn, Qt.AlignmentFlag.AlignCenter, "📌 Ghim Bảng Punnett vào Bài Giảng")

    # -------------------------------------------------------------------------
    # Bottom Genetics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ DI TRUYỀN HỌC MENDEL & MORGAN:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Khung Punnett: Công cụ đại số trực quan biểu diễn sự kết hợp ngẫu nhiên giữa các loại giao tử đực và cái.\n"
            f"• Định luật phân ly độc lập chứng minh các cặp nhân tố di truyền phân ly độc lập trong quá trình phát sinh giao tử.\n"
            f"• Quy luật: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Stamp buttons
        for s_key in ["stamp", "action_stamp"]:
            if self._btn_rects.get(s_key) and self._btn_rects[s_key].contains(local_pos):
                self.stamp_to_board()
                return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
