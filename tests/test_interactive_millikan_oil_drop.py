"""
AirTeach AI – Tests for Interactive Millikan Oil Drop & Elementary Charge Simulator (Milestone 116)
Verifies elementary charge quantization (q = n*e), Stokes viscous drag, Cunningham slip correction,
electric force balance, X-ray ionization, air pinch gestures, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_millikan_oil_drop import (
    InteractiveMillikanOilDropItem,
    PRESETS,
    E_CHARGE,
    G_ACCEL,
    RHO_OIL,
    RHO_AIR,
    ETA_AIR,
    PLATE_DISTANCE_M,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_millikan_initialization(qapp):
    item = InteractiveMillikanOilDropItem(pos=(140.0, 50.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 700.0, 440.0)
        assert item.active_preset.code == "BALANCED_SUSPENSION_SINGLE_ELECTRON"
        assert item.electrons_count == -1
        assert abs(item.droplet_radius_um - 0.85) < 0.01
        assert item.droplet_mass_kg > 0.0
        assert item.net_gravity_force_n > 0.0
        assert item.electric_force_n > 0.0
        # Balance check: velocity near 0 at balanced voltage
        assert abs(item.terminal_velocity_mps) < 1e-4
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_charge_quantization_and_elementary_charge(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        for n in [-1, -2, -3, -4, -5]:
            item.set_electrons_count(n)
            assert item.electrons_count == n
            expected_q = float(n) * E_CHARGE
            assert abs(item.droplet_charge_c - expected_q) < 1e-25
            # Quantization ratio must be exact integer
            ratio = item.droplet_charge_c / -E_CHARGE
            assert abs(ratio - abs(n)) < 1e-6
    finally:
        item.cleanup()


def test_stokes_drag_and_cunningham_correction(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        # With Cunningham correction enabled on 1.45 μm droplet
        item._cunningham_enabled = True
        f_corr = item.cunningham_correction_factor
        assert f_corr > 1.0  # Slip factor > 1
        assert 1.02 < f_corr < 1.15
        assert item.effective_air_viscosity < ETA_AIR

        # With Cunningham correction disabled
        item._cunningham_enabled = False
        assert item.cunningham_correction_factor == 1.0
        assert item.effective_air_viscosity == ETA_AIR
    finally:
        item.cleanup()


def test_balanced_voltage_and_equilibrium(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        item.set_electrons_count(-1)
        v_bal = item.balanced_voltage_v
        assert v_bal > 50.0 and v_bal < 500.0

        # Exact balance: v = 0
        item.set_voltage(v_bal)
        assert abs(item.terminal_velocity_mps) < 1e-5

        # Higher voltage: droplet accelerates upward (v > 0)
        item.set_voltage(v_bal + 50.0)
        assert item.terminal_velocity_mps > 0.0

        # Lower voltage: droplet falls downward (v < 0)
        item.set_voltage(v_bal - 50.0)
        assert item.terminal_velocity_mps < 0.0

        # Zero voltage: pure Stokes free fall
        item.set_voltage(0.0)
        assert item.terminal_velocity_mps < -1e-5
        # v1 = F_net_g / (6 * pi * eta * r)
        stokes_coeff = 6.0 * math.pi * item.effective_air_viscosity * item._radius_m
        expected_v1 = -item.net_gravity_force_n / stokes_coeff
        assert abs(item.terminal_velocity_mps - expected_v1) < 1e-8
    finally:
        item.cleanup()


def test_xray_ionization_and_atomizer_spray(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        init_n = item.electrons_count
        item.fire_xray_pulse()
        assert item._is_xray_active is True
        assert item._xray_timer_ticks > 0
        # Charge jumped
        assert item.electrons_count in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]

        # Atomizer spray
        item.spray_new_droplet()
        assert 1.0 <= item.droplet_radius_um <= 2.5
        assert -5 <= item.electrons_count <= -1
        assert item._droplet_y_norm == 0.20
    finally:
        item.cleanup()


def test_all_curriculum_presets(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        preset_keys = [
            "BALANCED_SUSPENSION_SINGLE_ELECTRON",
            "MULTI_ELECTRON_CHARGES",
            "TERMINAL_VELOCITY_STOKES_FALL",
            "XRAY_IONIZATION_CHARGE_JUMP",
            "STOKES_CUNNINGHAM_CORRECTION",
        ]
        for key in preset_keys:
            item.load_preset(key)
            assert item.active_preset.code == key
            assert len(item.active_preset.title_vi) > 0
            assert len(item.active_preset.description) > 0
            assert item.electrons_count == item.active_preset.electrons
    finally:
        item.cleanup()


def test_parameter_controls_and_steppers(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        # Voltage clamping
        item.set_voltage(300.5)
        assert item.voltage_v == 300.5
        item.set_voltage(-50.0)
        assert item.voltage_v == 0.0
        item.set_voltage(1500.0)
        assert item.voltage_v == 1000.0

        # Radius clamping
        item.set_droplet_radius_um(2.5)
        assert abs(item.droplet_radius_um - 2.5) < 0.01
        item.set_droplet_radius_um(0.1)
        assert abs(item.droplet_radius_um - 0.5) < 0.01
        item.set_droplet_radius_um(10.0)
        assert abs(item.droplet_radius_um - 4.0) < 0.01

        # Voltage toggle
        v_on = item._is_voltage_on
        item.toggle_voltage()
        assert item._is_voltage_on != v_on
        item.toggle_voltage()
        assert item._is_voltage_on == v_on
    finally:
        item.cleanup()


def test_animation_and_position_update(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        # Set to falling condition
        item.set_voltage(0.0)
        init_y = item._droplet_y_norm

        for _ in range(10):
            item._on_tick()

        # Droplet must move downward (+ norm y towards 1.0)
        assert item._droplet_y_norm >= init_y
        assert 0.08 <= item._droplet_y_norm <= 0.92

        # Animation toggle
        assert item.is_animating is True
        item.toggle_animation()
        assert item.is_animating is False
        item.toggle_animation()
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_touchless_air_gestures(qapp):
    item = InteractiveMillikanOilDropItem(pos=(0.0, 0.0))
    try:
        # Pre-render to build button rects cache
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        item.paint(p, None, None)
        p.end()

        # 1. Preset tabs
        r_preset = item._btn_rects.get("preset_MULTI_ELECTRON_CHARGES")
        assert r_preset is not None
        assert item.handle_air_pinch(r_preset.center()) is True
        assert item.active_preset.code == "MULTI_ELECTRON_CHARGES"

        # 2. Voltage steppers
        r_vp10 = item._btn_rects.get("v_p10")
        assert r_vp10 is not None
        cur_v = item.voltage_v
        assert item.handle_air_pinch(r_vp10.center()) is True
        assert abs(item.voltage_v - (cur_v + 10.0)) < 1e-4

        # 3. Auto balance button
        r_bal = item._btn_rects.get("auto_balance")
        assert r_bal is not None
        assert item.handle_air_pinch(r_bal.center()) is True
        assert abs(item.voltage_v - item.balanced_voltage_v) < 0.1

        # 4. Out of bounds returns False
        assert item.handle_air_pinch(QPointF(-100.0, -100.0)) is False
    finally:
        item.cleanup()


def test_paint_rendering_all_presets(qapp):
    item = InteractiveMillikanOilDropItem()
    try:
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        for preset_key in PRESETS:
            item.load_preset(preset_key)
            item.fire_xray_pulse()
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()
            assert not img.isNull()
    finally:
        item.cleanup()


def test_stem_tools_vector_diagram(qapp):
    items = PhysicsTools.create_millikan_oil_drop_diagram(
        center=(400.0, 300.0),
        width=640.0,
        height=340.0,
        preset_code="BALANCED_SUSPENSION_SINGLE_ELECTRON",
        voltage_v=285.0,
        electrons_n=-1,
        radius_um=1.45,
        velocity_mms=0.0,
    )
    assert len(items) >= 10
    assert items[0].rect().width() == 640.0
    assert items[0].rect().height() == 340.0


def test_lesson_board_integration(qapp):
    board = LessonBoard()
    try:
        assert board.is_millikan_oil_drop_visible() is False
        assert board.get_millikan_oil_drop() is None

        # Toggle ON
        vis1 = board.toggle_millikan_oil_drop()
        assert vis1 is True
        assert board.is_millikan_oil_drop_visible() is True
        item = board.get_millikan_oil_drop()
        assert item is not None
        assert isinstance(item, InteractiveMillikanOilDropItem)

        # Excluded from stroke selection
        excluded = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert item not in excluded

        # Toggle OFF
        vis2 = board.toggle_millikan_oil_drop()
        assert vis2 is False
        assert board.is_millikan_oil_drop_visible() is False
        assert board.get_millikan_oil_drop() is None
        assert board.get_millikan_oil_drop_item() is not None
    finally:
        item = board.get_millikan_oil_drop_item()
        if item is not None and hasattr(item, "cleanup"):
            item.cleanup()
