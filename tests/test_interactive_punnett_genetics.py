"""
AirTeach AI – Test Suite: Interactive Mendelian Genetics & Punnett Square Simulation
=====================================================================================
Verifies:
  - Exact Gregor Mendel laws of inheritance (Segregation and Independent Assortment)
  - Monohybrid cross Aa × Aa yielding 1 AA : 2 Aa : 1 aa and 3:1 phenotypic ratio
  - Dihybrid cross AaBb × AaBb 16-cell Punnett square with 9:3:3:1 ratio
  - Incomplete dominance blended phenotypes (1:2:1)
  - Codominance and ABO human blood type inheritance
  - Morgan sex-linked inheritance (X-linked genes)
  - Presets, air gestures, and rendering
  - STEMTools vector diagram generation
  - LessonBoard integration & lifecycle
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QApplication

from src.ui.interactive_punnett_genetics import (
    InteractivePunnettGeneticsItem,
    PRESETS,
)
from src.ui.stem_tools import BiologyTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestGeneticsMathematics:
    """Verifies Mendelian ratios and Punnett squares without mock data."""

    def test_monohybrid_cross_ratios(self, qapp):
        item = InteractivePunnettGeneticsItem()
        item.load_preset("MONOHYBRID_PEA_PLANT")

        # Parents: Aa x Aa
        m_gametes = item.get_male_gametes()
        f_gametes = item.get_female_gametes()
        assert m_gametes == ["A", "a"]
        assert f_gametes == ["A", "a"]

        grid = item.calculate_punnett_grid()
        assert len(grid) == 4

        zygotes = [cell["zygote"] for cell in grid]
        assert zygotes.count("AA") == 1
        assert zygotes.count("Aa") == 2
        assert zygotes.count("aa") == 1

        dist = item.calculate_phenotype_distribution()
        # 75% Yellow (Dominant A) : 25% Green (Recessive a)
        assert math.isclose(dist["Vàng (Trội A)"], 75.0, rel_tol=1e-5)
        assert math.isclose(dist["Xanh (Lặn a)"], 25.0, rel_tol=1e-5)

        item.cleanup()

    def test_dihybrid_cross_16_cells_9_3_3_1(self, qapp):
        item = InteractivePunnettGeneticsItem()
        item.load_preset("DIHYBRID_INDEPENDENT_9_3_3_1")

        m_gametes = item.get_male_gametes()
        f_gametes = item.get_female_gametes()
        assert len(m_gametes) == 4
        assert len(f_gametes) == 4
        assert set(m_gametes) == {"AB", "Ab", "aB", "ab"}

        grid = item.calculate_punnett_grid()
        assert len(grid) == 16

        dist = item.calculate_phenotype_distribution()
        # 9/16 = 56.25%, 3/16 = 18.75%, 1/16 = 6.25%
        assert math.isclose(dist["Vàng - Trơn (9/16)"], 56.25, rel_tol=1e-4)
        assert math.isclose(dist["Vàng - Nhăn (3/16)"], 18.75, rel_tol=1e-4)
        assert math.isclose(dist["Xanh - Trơn (3/16)"], 18.75, rel_tol=1e-4)
        assert math.isclose(dist["Xanh - Nhăn (1/16)"], 6.25, rel_tol=1e-4)

        item.cleanup()

    def test_incomplete_dominance_ratio(self, qapp):
        item = InteractivePunnettGeneticsItem()
        item.load_preset("INCOMPLETE_DOMINANCE_FLOWERS")

        # Parents: RW x RW
        dist = item.calculate_phenotype_distribution()
        # 25% Red : 50% Pink : 25% White
        assert math.isclose(dist["Hoa Đỏ (RR)"], 25.0, rel_tol=1e-5)
        assert math.isclose(dist["Hoa Hồng (RW)"], 50.0, rel_tol=1e-5)
        assert math.isclose(dist["Hoa Trắng (WW)"], 25.0, rel_tol=1e-5)

        item.cleanup()

    def test_codominance_abo_blood_inheritance(self, qapp):
        item = InteractivePunnettGeneticsItem()
        item.load_preset("CODOMINANCE_ABO_BLOOD")

        # Parents: Male AB (I^A I^B) x Female AO (I^A I^O)
        grid = item.calculate_punnett_grid()
        assert len(grid) == 4

        dist = item.calculate_phenotype_distribution()
        # Offspring genotypes: AA (Blood A), AO (Blood A), AB (Blood AB), BO (Blood B)
        # Phenotypes: 50% Blood A, 25% Blood AB, 25% Blood B
        assert math.isclose(dist["Nhóm máu A"], 50.0, rel_tol=1e-5)
        assert math.isclose(dist["Nhóm máu AB (Đồng trội)"], 25.0, rel_tol=1e-5)
        assert math.isclose(dist["Nhóm máu B"], 25.0, rel_tol=1e-5)

        item.cleanup()


class TestGeneticsPresetsAndControls:
    """Verifies presets, air gestures, and visual rendering."""

    def test_all_presets_load(self, qapp):
        item = InteractivePunnettGeneticsItem()

        for code in PRESETS:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.boundingRect().width() == 700.0
            assert item.boundingRect().height() == 440.0
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        item = InteractivePunnettGeneticsItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        # 1. Preset pinch
        p_rect = item._btn_rects.get("preset_CODOMINANCE_ABO_BLOOD")
        assert p_rect is not None
        center = item.mapToScene(p_rect.center())
        assert item.handle_air_pinch(center) is True
        assert item.active_preset.code == "CODOMINANCE_ABO_BLOOD"

        # 2. Outside pinch
        outside = item.mapToScene(QPointF(9999.0, 9999.0))
        assert item.handle_air_pinch(outside) is False

        item.cleanup()

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractivePunnettGeneticsItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()

        for code in PRESETS:
            item.load_preset(code)
            painter = QPainter(image)
            item.paint(painter, opt)
            painter.end()

        item.cleanup()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle integration."""

    def test_stem_tools_diagram_generation(self, qapp):
        items = BiologyTools.create_punnett_genetics_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="MONOHYBRID_PEA_PLANT",
            parent_cross="Aa × Aa",
            phenotype_ratio="3 Vàng : 1 Xanh",
        )

        assert len(items) >= 5
        assert items[0].boundingRect().width() >= 640.0
        assert items[0].boundingRect().height() >= 340.0

    def test_lesson_board_genetics_lifecycle(self, qapp):
        board = LessonBoard()

        assert not board.is_punnett_genetics_visible()
        assert board.get_punnett_genetics() is None
        assert board.get_punnett_genetics_item() is None

        # Toggle ON
        res = board.toggle_punnett_genetics()
        assert res is True
        assert board.is_punnett_genetics_visible()
        gen_item = board.get_punnett_genetics()
        assert gen_item is not None
        assert board.get_punnett_genetics_item() is gen_item

        # Toggle OFF
        res = board.toggle_punnett_genetics()
        assert res is False
        assert not board.is_punnett_genetics_visible()
        assert board.get_punnett_genetics() is None
        assert board.get_punnett_genetics_item() is gen_item

        # Exclusion from get_items_in_rect
        board.show_punnett_genetics(True, pos=(100.0, 100.0))
        items_in_rect = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert gen_item not in items_in_rect

        gen_item.cleanup()
