"""
AirTeach AI — Tests for Interactive Protein Folding & Ramachandran Dihedral Angle Explorer (Milestone 108)
Verifies stereochemical dihedral angle basins, Pauling-Corey hydrogen bonding networks,
Gibbs free energy denaturation thermodynamics, touchless air pinch, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_protein_folding import (
    InteractiveProteinFoldingItem,
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


def test_protein_folding_initialization(qapp):
    item = InteractiveProteinFoldingItem(pos=(140.0, 60.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 60.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 660.0, 400.0)
        assert item.active_preset.code == "ALPHA_HELIX_KERATIN_MYOGLOBIN"
        assert item.phi_deg == -57.0
        assert item.psi_deg == -47.0
        assert item.temp_c == 37.0
        assert item.ph == 7.4
        assert item.folding_free_energy_kj < 0.0
        assert item.fraction_folded > 0.85
        assert item.hydrogen_bond_count >= 7
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_protein_folding_all_presets(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        # 1. Alpha Helix
        item.load_preset("ALPHA_HELIX_KERATIN_MYOGLOBIN")
        assert item.active_preset.structure_type == "alpha_helix"
        reg, fav = item.check_ramachandran_region(item.phi_deg, item.psi_deg)
        assert fav is True
        assert "α-Helix" in reg
        assert item.fraction_folded > 0.85

        # 2. Beta Sheet
        item.load_preset("BETA_SHEET_SILK_FIBROIN")
        assert item.active_preset.structure_type == "beta_sheet"
        reg, fav = item.check_ramachandran_region(item.phi_deg, item.psi_deg)
        assert fav is True
        assert "β" in reg
        assert item.fraction_folded > 0.85

        # 3. Collagen Triple Helix
        item.load_preset("COLLAGEN_TRIPLE_HELIX")
        assert item.active_preset.structure_type == "collagen"
        reg, fav = item.check_ramachandran_region(item.phi_deg, item.psi_deg)
        assert fav is True
        assert "Collagen" in reg

        # 4. Thermal Denaturation
        item.load_preset("THERMAL_DENATURATION_UNFOLDED")
        assert item.active_preset.structure_type == "random_coil"
        assert item.temp_c >= 80.0
        assert item.folding_free_energy_kj > 0.0
        assert item.fraction_folded < 0.25

        # 5. Prion Amyloid
        item.load_preset("PRION_MISFOLDING_AMYLOID")
        assert item.active_preset.structure_type == "amyloid_fibril"
        assert item.ph == 5.0
    finally:
        item.cleanup()


def test_ramachandran_regions(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        # Core Alpha
        reg, fav = item.check_ramachandran_region(-60.0, -45.0)
        assert fav is True
        assert "α-Helix" in reg

        # Core Beta
        reg, fav = item.check_ramachandran_region(-140.0, 130.0)
        assert fav is True
        assert "β" in reg

        # Collagen
        reg, fav = item.check_ramachandran_region(-70.0, 150.0)
        assert fav is True
        assert "Collagen" in reg

        # Left-handed Alpha
        reg, fav = item.check_ramachandran_region(60.0, 45.0)
        assert fav is True
        assert "trái" in reg or "Glycine" in reg

        # Disallowed Steric Clash
        reg, fav = item.check_ramachandran_region(0.0, 0.0)
        assert fav is False
        assert "Vùng cấm" in reg
    finally:
        item.cleanup()


def test_denaturation_thermodynamics_and_ph(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        item.load_preset("ALPHA_HELIX_KERATIN_MYOGLOBIN")
        # Physiological state
        item.set_temperature_c(37.0)
        item.set_ph(7.4)
        dg_phys = item.folding_free_energy_kj
        fn_phys = item.fraction_folded
        assert dg_phys < 0.0
        assert fn_phys > 0.90

        # High heat thermal denaturation
        item.set_temperature_c(95.0)
        dg_hot = item.folding_free_energy_kj
        fn_hot = item.fraction_folded
        assert dg_hot > dg_phys
        assert fn_hot < 0.20

        # Acidic denaturation at pH 1.5
        item.set_temperature_c(37.0)
        item.set_ph(1.5)
        dg_acid = item.folding_free_energy_kj
        fn_acid = item.fraction_folded
        assert dg_acid > dg_phys
        assert fn_acid < fn_phys
    finally:
        item.cleanup()


def test_parameter_setters_and_clamping(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        # Angles clamping [-180, 180]
        item.set_dihedral_angles(250.0, -300.0)
        assert item.phi_deg == 180.0
        assert item.psi_deg == -180.0

        item.set_dihedral_angles(-50.0, 120.0)
        assert item.phi_deg == -50.0
        assert item.psi_deg == 120.0

        # Temperature clamping [0, 100]
        item.set_temperature_c(150.0)
        assert item.temp_c == 100.0
        item.set_temperature_c(-20.0)
        assert item.temp_c == 0.0

        # pH clamping [1.0, 14.0]
        item.set_ph(18.0)
        assert item.ph == 14.0
        item.set_ph(-5.0)
        assert item.ph == 1.0
    finally:
        item.cleanup()


def test_animation_tick_and_toggle(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        rot_init = item._rot_y
        for _ in range(5):
            item._on_animation_tick()
        assert item._rot_y != rot_init

        # Toggle off
        assert item.toggle_animation() is False
        assert item.is_animating is False

        # Toggle on
        assert item.toggle_animation() is True
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveProteinFoldingItem(pos=(100.0, 50.0))
    try:
        # Outside
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Preset button click via air pinch
        btn_beta = item._btn_rects["preset_BETA_SHEET_SILK_FIBROIN"]
        scene_pt_beta = item.mapToScene(btn_beta.center())
        assert item.handle_air_pinch(scene_pt_beta) is True
        assert item.active_preset.code == "BETA_SHEET_SILK_FIBROIN"

        # Direct Ramachandran canvas click
        canvas_r = item._btn_rects["ramachandran_canvas"]
        scene_pt_canvas = item.mapToScene(canvas_r.center())
        assert item.handle_air_pinch(scene_pt_canvas) is True
        assert abs(item.phi_deg - 0.0) < 5.0
        assert abs(item.psi_deg - 0.0) < 5.0

        # Temp steppers
        t_before = item.temp_c
        btn_tu = item._btn_rects["temp_up"]
        scene_pt_tu = item.mapToScene(btn_tu.center())
        assert item.handle_air_pinch(scene_pt_tu) is True
        assert item.temp_c == t_before + 5.0

        btn_td = item._btn_rects["temp_down"]
        scene_pt_td = item.mapToScene(btn_td.center())
        assert item.handle_air_pinch(scene_pt_td) is True
        assert item.temp_c == t_before

        # pH steppers
        ph_before = item.ph
        btn_pu = item._btn_rects["ph_up"]
        scene_pt_pu = item.mapToScene(btn_pu.center())
        assert item.handle_air_pinch(scene_pt_pu) is True
        assert item.ph == ph_before + 0.5

        # Play/pause
        btn_play = item._btn_rects["toggle_anim"]
        scene_pt_play = item.mapToScene(btn_play.center())
        anim_before = item.is_animating
        assert item.handle_air_pinch(scene_pt_play) is True
        assert item.is_animating != anim_before
    finally:
        item.cleanup()


def test_stem_tools_diagram_generation_and_stamp(qapp):
    items = BiologyTools.create_protein_folding_diagram(
        center=(400.0, 300.0),
        preset_code="ALPHA_HELIX_KERATIN_MYOGLOBIN",
        phi_deg=-57.0,
        psi_deg=-47.0,
        temp_c=37.0,
        ph=7.4,
        fraction_folded=0.95,
    )
    assert isinstance(items, list)
    assert len(items) >= 15

    # Check text contents
    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "CẤU TRÚC BẬC HAI" in combined or "PROTEIN" in combined or "RAMACHANDRAN" in combined
    assert "Thông số" in combined or "Pauling" in combined or "hydro" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveProteinFoldingItem(pos=(100.0, 50.0))
    board.addItem(item)
    try:
        stamp_called = []
        item.stamp_requested.connect(lambda: stamp_called.append(True))
        item.stamp_to_board()
        assert len(stamp_called) == 1
        assert len(board.items()) > 1
    finally:
        item.cleanup()
        board.clear()


def test_lesson_board_lifecycle(qapp):
    board = LessonBoard()
    try:
        assert board.is_protein_folding_visible() is False
        assert board.get_protein_folding() is None

        # Show
        shown_item = board.show_protein_folding(True)
        assert shown_item is not None
        assert board.is_protein_folding_visible() is True
        assert board.get_protein_folding() is shown_item
        assert board.get_protein_folding_item() is shown_item

        # Toggle off
        vis = board.toggle_protein_folding()
        assert vis is False
        assert board.is_protein_folding_visible() is False
        assert board.get_protein_folding() is None

        # Toggle on
        vis = board.toggle_protein_folding()
        assert vis is True
        assert board.is_protein_folding_visible() is True

        # Hide
        board.show_protein_folding(False)
        assert board.is_protein_folding_visible() is False
    finally:
        if board.get_protein_folding_item():
            board.get_protein_folding_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveProteinFoldingItem()
    try:
        img = QImage(660, 400, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
