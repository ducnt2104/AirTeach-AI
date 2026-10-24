"""
AirTeach AI — Tests for Interactive Cellular Respiration & ETC Simulator (Milestone 107)
Verifies bioenergetics formulas, proton motive force, ATP synthase rotation,
metabolic inhibitor presets, touchless air pinch, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_cellular_respiration import (
    InteractiveCellularRespirationItem,
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


def test_cellular_respiration_initialization(qapp):
    item = InteractiveCellularRespirationItem(pos=(150.0, 80.0))
    try:
        assert item.pos().x() == 150.0
        assert item.pos().y() == 80.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 640.0, 380.0)
        assert item.active_preset.code == "STANDARD_AEROBIC_RESPIRATION"
        assert item.oxygen_level == 1.0
        assert item.proton_motive_force_mv >= 180.0
        assert item.atp_synthesis_rate_per_sec > 10.0
        assert len(item._h_plus_particles) == 24
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_cellular_respiration_all_presets(qapp):
    item = InteractiveCellularRespirationItem()
    try:
        # 1. Standard Aerobic
        item.load_preset("STANDARD_AEROBIC_RESPIRATION")
        assert item.active_preset.atp_yield == 32
        assert item.proton_motive_force_mv >= 180.0
        assert item.atp_synthesis_rate_per_sec > 12.0

        # 2. Cyanide Poisoning
        item.load_preset("CYANIDE_POISONING_COMPLEX_IV")
        assert item.active_preset.complex_iv_active is False
        assert item.active_preset.atp_yield == 2
        assert item.proton_motive_force_mv == 35.0
        assert item.atp_synthesis_rate_per_sec == 0.5

        # 3. DNP Uncoupling
        item.load_preset("DNP_UNCOUPLING_THERMOGENESIS")
        assert item.active_preset.uncoupled_dnp is True
        assert item.proton_motive_force_mv == 25.0
        assert item.atp_synthesis_rate_per_sec == 0.5

        # 4. Rotenone Inhibition
        item.load_preset("ROTENONE_COMPLEX_I_INHIBITION")
        assert item.active_preset.complex_i_active is False
        assert item.active_preset.atp_yield == 16
        assert item.proton_motive_force_mv < 200.0
        assert item.atp_synthesis_rate_per_sec > 5.0

        # 5. Hypoxia / Fermentation
        item.load_preset("HYPOXIA_ANAEROBIC_FERMENTATION")
        assert item.oxygen_level == 0.0
        assert item.active_preset.atp_yield == 2
        assert item.proton_motive_force_mv == 35.0
        assert item.atp_synthesis_rate_per_sec == 0.5
    finally:
        item.cleanup()


def test_oxygen_level_clamping_and_signals(qapp):
    item = InteractiveCellularRespirationItem()
    try:
        received = []
        item.oxygen_changed.connect(lambda val: received.append(val))

        item.set_oxygen_level(0.75)
        assert item.oxygen_level == 0.75
        assert len(received) == 1
        assert received[-1] == 0.75

        # Upper clamp
        item.set_oxygen_level(1.5)
        assert item.oxygen_level == 1.0

        # Lower clamp
        item.set_oxygen_level(-0.4)
        assert item.oxygen_level == 0.0
    finally:
        item.cleanup()


def test_animation_tick_and_particle_transport(qapp):
    item = InteractiveCellularRespirationItem()
    try:
        initial_rot = item._atp_rotor_angle
        # Advance 5 simulation ticks
        for _ in range(5):
            item._on_animation_tick()
        assert item._atp_rotor_angle != initial_rot

        # Toggle off
        assert item.toggle_animation() is False
        assert item.is_animating is False

        # Toggle on
        assert item.toggle_animation() is True
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveCellularRespirationItem(pos=(100.0, 50.0))
    try:
        # Outside bounding rect
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Switch preset to CYANIDE_POISONING_COMPLEX_IV via air pinch
        btn_cn = item._btn_rects["preset_CYANIDE_POISONING_COMPLEX_IV"]
        scene_pt_cn = item.mapToScene(btn_cn.center())
        assert item.handle_air_pinch(scene_pt_cn) is True
        assert item.active_preset.code == "CYANIDE_POISONING_COMPLEX_IV"

        # Oxygen stepper up
        cur_o2 = item.oxygen_level
        btn_up = item._btn_rects["o2_up"]
        scene_pt_up = item.mapToScene(btn_up.center())
        assert item.handle_air_pinch(scene_pt_up) is True
        assert item.oxygen_level >= cur_o2

        # Oxygen stepper down
        cur_o2 = item.oxygen_level
        btn_down = item._btn_rects["o2_down"]
        scene_pt_down = item.mapToScene(btn_down.center())
        assert item.handle_air_pinch(scene_pt_down) is True
        assert item.oxygen_level <= cur_o2

        # Toggle animation button
        btn_play = item._btn_rects["toggle_anim"]
        scene_pt_play = item.mapToScene(btn_play.center())
        was_anim = item.is_animating
        assert item.handle_air_pinch(scene_pt_play) is True
        assert item.is_animating != was_anim

        # Slider direct scrub
        btn_sl = item._btn_rects["o2_slider_bounds"]
        scene_pt_sl = item.mapToScene(QPointF(btn_sl.left() + btn_sl.width() * 0.4, btn_sl.center().y()))
        assert item.handle_air_pinch(scene_pt_sl) is True
        assert abs(item.oxygen_level - 0.4) < 0.05
    finally:
        item.cleanup()


def test_biology_tools_create_cellular_respiration_diagram(qapp):
    items = BiologyTools.create_cellular_respiration_diagram(
        center=(400.0, 300.0),
        preset_code="STANDARD_AEROBIC_RESPIRATION",
        oxygen_level=1.0,
        pmf_mv=200.0,
        atp_rate=16.0,
    )
    assert isinstance(items, list)
    assert len(items) >= 12

    # Check for meaningful biological labels
    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "Hô hấp tế bào" in combined or "Respiration" in combined or "ETC" in combined
    assert "Phức hệ" in combined or "Complex" in combined or "ATP" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveCellularRespirationItem(pos=(120.0, 80.0))
    board.addItem(item)
    try:
        stamp_called = []
        item.stamp_requested.connect(lambda: stamp_called.append(True))
        item.stamp_to_board()
        assert len(stamp_called) == 1
        # Scene should contain stamped items in addition to item
        assert len(board.items()) > 1
    finally:
        item.cleanup()
        board.clear()


def test_lesson_board_lifecycle(qapp):
    board = LessonBoard()
    try:
        assert board.is_cellular_respiration_visible() is False
        assert board.get_cellular_respiration() is None

        # Show
        shown_item = board.show_cellular_respiration(True)
        assert shown_item is not None
        assert board.is_cellular_respiration_visible() is True
        assert board.get_cellular_respiration() is shown_item
        assert board.get_cellular_respiration_item() is shown_item

        # Toggle off
        vis = board.toggle_cellular_respiration()
        assert vis is False
        assert board.is_cellular_respiration_visible() is False
        assert board.get_cellular_respiration() is None

        # Toggle on
        vis = board.toggle_cellular_respiration()
        assert vis is True
        assert board.is_cellular_respiration_visible() is True

        # Hide
        board.show_cellular_respiration(False)
        assert board.is_cellular_respiration_visible() is False
    finally:
        if board.get_cellular_respiration_item():
            board.get_cellular_respiration_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveCellularRespirationItem()
    try:
        img = QImage(640, 380, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
