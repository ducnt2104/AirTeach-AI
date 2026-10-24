"""
AirTeach AI — Tests for Interactive Semiconductor P-N Diode & Shockley Equation Simulator (Milestone 112)
Verifies solid-state physics, Shockley ideal diode equation, thermal voltage, space-charge depletion width,
energy band diagrams, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_semiconductor_diode import (
    InteractiveSemiconductorDiodeItem,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_diode_initialization(qapp):
    item = InteractiveSemiconductorDiodeItem(pos=(140.0, 50.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 680.0, 410.0)
        assert item.active_preset.code == "SILICON_PN_RECTIFIER"
        assert item.bias_voltage == 0.65
        assert item.temperature_k == 300.0
        assert abs(item.thermal_voltage_vt - 0.02586) < 0.001
        assert item.current_amperes > 0.0
        assert 0.05 <= item.depletion_width_norm <= 0.95
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_shockley_equation_and_thermal_voltage(qapp):
    item = InteractiveSemiconductorDiodeItem()
    try:
        item.load_preset("SILICON_PN_RECTIFIER")
        # Zero bias: I = 0
        item.set_bias_voltage(0.0)
        assert abs(item.current_amperes) < 1e-11

        # Forward bias: I increases exponentially
        item.set_bias_voltage(0.70)
        i_fwd = item.current_amperes
        assert i_fwd > 0.001  # > 1 mA

        # Reverse bias: small negative saturation current
        item.set_bias_voltage(-2.0)
        i_rev = item.current_amperes
        assert i_rev < 0.0
        assert abs(i_rev) < 1e-6

        # Zener breakdown
        item.load_preset("ZENER_BREAKDOWN_REGULATOR")
        item.set_bias_voltage(-6.0)  # Beyond breakdown (-5.1V)
        assert item.current_amperes < -0.01  # Heavy negative conduction
    finally:
        item.cleanup()


def test_all_five_presets(qapp):
    item = InteractiveSemiconductorDiodeItem()
    try:
        for p_code in PRESETS:
            item.load_preset(p_code)
            assert item.active_preset.code == p_code
            assert item.active_preset.bandgap_ev > 0.5
            assert item.active_preset.v_bi > 0.2
            assert item.depletion_width_norm > 0.0
    finally:
        item.cleanup()


def test_voltage_and_temperature_setters_and_clamping(qapp):
    item = InteractiveSemiconductorDiodeItem()
    try:
        # Bias voltage clamping [-10.0, +1.5]
        item.set_bias_voltage(3.5)
        assert item.bias_voltage == 1.5

        item.set_bias_voltage(-15.0)
        assert item.bias_voltage == -10.0

        item.set_bias_voltage(0.55)
        assert item.bias_voltage == 0.55

        # Temperature clamping [200, 450]
        item.set_temperature_k(550.0)
        assert item.temperature_k == 450.0

        item.set_temperature_k(120.0)
        assert item.temperature_k == 200.0

        item.set_temperature_k(320.0)
        assert item.temperature_k == 320.0
    finally:
        item.cleanup()


def test_animation_toggle_and_tick(qapp):
    item = InteractiveSemiconductorDiodeItem()
    try:
        phase_init = item._anim_phase
        item._on_anim_tick()
        assert item._anim_phase != phase_init

        # Toggle off
        assert item.toggle_animation() is False
        assert item.is_animating is False

        # Toggle on
        assert item.toggle_animation() is True
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_air_pinch_interactions(qapp):
    item = InteractiveSemiconductorDiodeItem(pos=(100.0, 50.0))
    try:
        # Outside
        assert item.handle_air_pinch(QPointF(10.0, 10.0)) is False

        # Preset button click
        btn_ge = item._btn_rects["preset_GERMANIUM_DIODE"]
        scene_pt_ge = item.mapToScene(btn_ge.center())
        assert item.handle_air_pinch(scene_pt_ge) is True
        assert item.active_preset.code == "GERMANIUM_DIODE"

        # Voltage Up / Down
        v_before = item.bias_voltage
        btn_vu = item._btn_rects["v_up"]
        scene_pt_vu = item.mapToScene(btn_vu.center())
        assert item.handle_air_pinch(scene_pt_vu) is True
        assert abs(item.bias_voltage - (v_before + 0.05)) < 1e-4

        btn_vd = item._btn_rects["v_down"]
        scene_pt_vd = item.mapToScene(btn_vd.center())
        assert item.handle_air_pinch(scene_pt_vd) is True
        assert abs(item.bias_voltage - v_before) < 1e-4

        # Temp Up / Down
        t_before = item.temperature_k
        btn_tu = item._btn_rects["temp_up"]
        scene_pt_tu = item.mapToScene(btn_tu.center())
        assert item.handle_air_pinch(scene_pt_tu) is True
        assert item.temperature_k == t_before + 10.0

        # Toggle animation
        btn_play = item._btn_rects["toggle_anim"]
        scene_pt_play = item.mapToScene(btn_play.center())
        assert item.handle_air_pinch(scene_pt_play) is True
        assert item.is_animating is False
        assert item.handle_air_pinch(scene_pt_play) is True
        assert item.is_animating is True

        # I-V curve direct scrub
        iv_r = item._iv_curve_rect
        scene_pt_iv = item.mapToScene(iv_r.center())
        assert item.handle_air_pinch(scene_pt_iv) is True
    finally:
        item.cleanup()


def test_stem_tools_diagram_generation_and_stamp(qapp):
    items = PhysicsTools.create_semiconductor_diode_diagram(
        center=(400.0, 300.0),
        preset_code="SILICON_PN_RECTIFIER",
        bias_v=0.65,
        current_a=0.025,
        temp_k=300.0,
    )
    assert isinstance(items, list)
    assert len(items) >= 15

    all_texts = []
    for itm in items:
        if hasattr(itm, "toPlainText"):
            all_texts.append(itm.toPlainText())
    combined = " ".join(all_texts)
    assert "BÁN DẪN" in combined or "P-N" in combined
    assert "SHOCKLEY" in combined or "Điện áp" in combined or "Dòng điện" in combined


def test_stamping_to_scene(qapp):
    board = LessonBoard()
    item = InteractiveSemiconductorDiodeItem(pos=(100.0, 50.0))
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
        assert board.is_semiconductor_diode_visible() is False
        assert board.get_semiconductor_diode() is None

        # Show
        shown_item = board.show_semiconductor_diode(True)
        assert shown_item is not None
        assert board.is_semiconductor_diode_visible() is True
        assert board.get_semiconductor_diode() is shown_item
        assert board.get_semiconductor_diode_item() is shown_item

        # Toggle off
        vis = board.toggle_semiconductor_diode()
        assert vis is False
        assert board.is_semiconductor_diode_visible() is False
        assert board.get_semiconductor_diode() is None

        # Toggle on
        vis = board.toggle_semiconductor_diode()
        assert vis is True
        assert board.is_semiconductor_diode_visible() is True

        # Hide
        board.show_semiconductor_diode(False)
        assert board.is_semiconductor_diode_visible() is False
    finally:
        if board.get_semiconductor_diode_item():
            board.get_semiconductor_diode_item().cleanup()
        board.clear()


def test_paint_headless(qapp):
    item = InteractiveSemiconductorDiodeItem()
    try:
        img = QImage(680, 410, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()
        assert not img.isNull()
    finally:
        item.cleanup()
