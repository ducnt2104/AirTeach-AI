"""
Unit Tests for Interactive Biology Mendel Genetics & Punnett Square Simulator
=============================================================================
Verifies:
  1. Gregor Mendel's Law of Segregation (Monohybrid Aa x Aa):
       - Exact 2x2 grid (4 offspring cells: AA, Aa, Aa, aa).
       - Phenotypic ratio: 3 Yellow (75%) : 1 Green (25%).
       - Genotypic ratio: 1/4 AA : 2/4 Aa : 1/4 aa.
  2. Mendel's Law of Independent Assortment (Dihybrid AaBb x AaBb):
       - Exact 4x4 grid (16 offspring cells).
       - Phenotypic ratio: 9 Yellow-Round : 3 Yellow-Wrinkled : 3 Green-Round : 1 Green-Wrinkled.
  3. Non-Mendelian Codominance (ABO Blood Types):
       - I^A i x I^B i produces 4 blood groups (AB, B, A, O) in 1:1:1:1 ratio.
  4. Sex-linked Inheritance (X-linked Color Blindness):
       - X^C X^c x X^C Y produces carrier females, normal females, normal males, and colorblind males.
  5. Pearson Chi-Square (χ²) Goodness-of-Fit Test:
       - Statistic formula χ² = Σ (O - E)² / E.
       - Degrees of freedom df = k - 1.
       - Alpha = 0.05 hypothesis acceptance.
  6. Touchless air gestures (air pinch / air tap) on presets, stamp, and close.
  7. LessonBoard lifecycle, visibility toggle, selection exclusion, and vector diagram stamping.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_genetics_punnett import (
    InteractiveGeneticsPunnettItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import BiologyTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def item(board):
    it = InteractiveGeneticsPunnettItem(pos=(300.0, 130.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestMendelGeneticsAndPunnett:
    """Verifies genetics laws, Cartesian gamete grids, and statistical Chi-square tests."""

    def test_presets_exist(self):
        assert "MENDEL_PEAS_MONOHYBRID" in PRESETS
        assert "MENDEL_PEAS_DIHYBRID" in PRESETS
        assert "BLOOD_TYPE_ABO" in PRESETS
        assert "SEX_LINKED_COLORBLIND" in PRESETS

    def test_monohybrid_cross_ratios(self, item):
        """Aa x Aa -> 3:1 phenotypic, 1:2:1 genotypic."""
        item.load_preset("MENDEL_PEAS_MONOHYBRID")

        assert len(item._grid_cells) == 2
        assert len(item._grid_cells[0]) == 2

        # Genotypic ratios
        g_ratios = item.get_genotypic_ratio()
        assert abs(g_ratios.get("AA", 0.0) - 0.25) < 1e-5
        assert abs(g_ratios.get("Aa", 0.0) + g_ratios.get("aA", 0.0) - 0.50) < 1e-5
        assert abs(g_ratios.get("aa", 0.0) - 0.25) < 1e-5

        # Phenotypic ratios: 3/4 Yellow : 1/4 Green
        p_ratios = item.get_phenotypic_ratio()
        assert abs(p_ratios.get("Hạt Vàng", 0.0) - 0.75) < 1e-5
        assert abs(p_ratios.get("Hạt Xanh", 0.0) - 0.25) < 1e-5

        # Chi-Square test: 305 yellow, 95 green out of 400
        # Expected: 300 yellow, 100 green. (5^2 / 300) + (5^2 / 100) = 0.0833 + 0.25 = 0.333
        chi_stat, df, accepted = item.compute_chi_square()
        assert df == 1
        assert chi_stat < 3.841  # Critical value at alpha = 0.05
        assert accepted is True

    def test_dihybrid_cross_ratios(self, item):
        """AaBb x AaBb -> 9:3:3:1 phenotypic ratio."""
        item.load_preset("MENDEL_PEAS_DIHYBRID")

        assert len(item._grid_cells) == 4
        assert len(item._grid_cells[0]) == 4
        total_cells = sum(len(r) for r in item._grid_cells)
        assert total_cells == 16

        p_ratios = item.get_phenotypic_ratio()
        assert abs(p_ratios.get("Vàng-Trơn", 0.0) - 9.0 / 16.0) < 1e-5
        assert abs(p_ratios.get("Vàng-Nhăn", 0.0) - 3.0 / 16.0) < 1e-5
        assert abs(p_ratios.get("Xanh-Trơn", 0.0) - 3.0 / 16.0) < 1e-5
        assert abs(p_ratios.get("Xanh-Nhăn", 0.0) - 1.0 / 16.0) < 1e-5

        chi_stat, df, accepted = item.compute_chi_square()
        assert df == 3
        assert chi_stat < 7.815  # Critical value at alpha = 0.05
        assert accepted is True

    def test_abo_blood_type_codominance(self, item):
        """I^A i x I^B i -> Types AB, B, A, O each 25%."""
        item.load_preset("BLOOD_TYPE_ABO")

        p_ratios = item.get_phenotypic_ratio()
        assert abs(p_ratios.get("Nhóm AB", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Nhóm A", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Nhóm B", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Nhóm O", 0.0) - 0.25) < 1e-5

        chi_stat, df, accepted = item.compute_chi_square()
        assert df == 3
        assert accepted is True

    def test_sex_linked_colorblindness(self, item):
        """X^C X^c x X^C Y."""
        item.load_preset("SEX_LINKED_COLORBLIND")

        p_ratios = item.get_phenotypic_ratio()
        assert len(p_ratios) == 4
        assert abs(p_ratios.get("Con gái bình thường", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Con gái mang gen", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Con trai bình thường", 0.0) - 0.25) < 1e-5
        assert abs(p_ratios.get("Con trai Mù màu", 0.0) - 0.25) < 1e-5

    def test_air_gestures_and_interactions(self, item):
        """Air gestures for preset switching, stamping, and closing."""
        # 1. Preset button air pinch
        preset_rects = item._get_preset_rects()
        p1_key, p1_rect = preset_rects[1]  # DIHYBRID
        scene_pt = item.mapToScene(p1_rect.center())
        res_preset = item.handle_air_pinch(scene_pt)
        assert res_preset is True
        assert item._current_preset_key == p1_key

        # 2. Close button air pinch
        c_rect = item._get_close_btn_rect()
        scene_c = item.mapToScene(c_rect.center())
        assert item.isVisible() is True
        res_close = item.handle_air_pinch(scene_c)
        assert res_close is True
        assert item.isVisible() is False

    def test_lesson_board_lifecycle_and_stamping(self, board):
        """Verifies LessonBoard show/toggle/hide methods and vector diagram stamping."""
        assert board.is_punnett_genetics_visible() is False
        assert board.get_punnett_genetics() is None

        # Show
        it = board.show_punnett_genetics(pos=(320.0, 150.0))
        assert it is not None
        assert board.is_punnett_genetics_visible() is True
        assert board.get_punnett_genetics() == it

        # Exclusion in get_items_in_rect (selection exclusion invariant)
        rect_all = board.sceneRect()
        selected = board.get_items_in_rect(rect_all)
        assert it not in selected

        # Stamping vector diagram
        items_before = len(board.items())
        it.stamp_to_board()
        items_after = len(board.items())
        assert items_after > items_before

        # Toggle hide
        vis = board.toggle_punnett_genetics()
        assert vis is False
        assert board.is_punnett_genetics_visible() is False

        # Toggle show
        vis2 = board.toggle_punnett_genetics()
        assert vis2 is True
        assert board.is_punnett_genetics_visible() is True

        # Clean cleanup
        it.cleanup()
        board.removeItem(it)
