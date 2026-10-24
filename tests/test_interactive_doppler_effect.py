"""
AirTeach AI – Tests for Interactive Doppler Effect, Sonic Boom & Mach Cone Simulator (Milestone 114)
Verifies acoustic Doppler shift (f' = f_0·v/(v∓v_s)), wavelength compression/dilation,
Mach cone angle (sin μ = 1/M), supersonic shockwaves, cosmological redshift, air gestures, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_doppler_effect import (
    InteractiveDopplerEffectItem,
    PRESETS,
    SOUND_SPEED_AIR_20C,
    SPEED_OF_LIGHT,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_doppler_initialization(qapp):
    item = InteractiveDopplerEffectItem(pos=(140.0, 50.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 700.0, 440.0)
        assert item.active_preset.code == "SUBSONIC_MOVING_SOURCE"
        assert item.mach_number == 0.50
        assert item.source_frequency_hz == 440.0
        assert item.sound_speed_mps == 343.0
        assert abs(item.source_velocity_mps - 171.5) < 0.1
        # Ahead: f' = 440 / (1 - 0.5) = 880 Hz (double frequency)
        assert abs(item.observed_frequency_ahead_hz - 880.0) < 0.1
        # Behind: f' = 440 / (1 + 0.5) = 293.33 Hz
        assert abs(item.observed_frequency_behind_hz - (440.0 / 1.5)) < 0.1
        # Wavelength ratios
        assert abs(item.wavelength_ahead_ratio - 0.5) < 1e-4
        assert abs(item.wavelength_behind_ratio - 1.5) < 1e-4
        # Subsonic has no Mach cone
        assert item.mach_angle_deg is None
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_doppler_frequency_and_wavelength_formulas(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        # Stationary source: M = 0.0 ==> f' = f_0, λ = λ_0
        item.set_mach_number(0.0)
        assert item.observed_frequency_ahead_hz == item.source_frequency_hz
        assert item.observed_frequency_behind_hz == item.source_frequency_hz
        assert item.wavelength_ahead_ratio == 1.0
        assert item.wavelength_behind_ratio == 1.0

        # Subsonic M = 0.75 ==> f'_ahead = f_0 / 0.25 = 4 * f_0
        item.set_mach_number(0.75)
        assert abs(item.observed_frequency_ahead_hz - 4.0 * item.source_frequency_hz) < 0.1
        assert abs(item.wavelength_ahead_ratio - 0.25) < 1e-4
        assert abs(item.wavelength_behind_ratio - 1.75) < 1e-4

        # Sonic barrier: M = 1.0 ==> f'_ahead = inf, λ_ahead = 0
        item.set_mach_number(1.0)
        assert item.observed_frequency_ahead_hz == float("inf")
        assert item.wavelength_ahead_ratio == 0.0
        assert item.wavelength_behind_ratio == 2.0
    finally:
        item.cleanup()


def test_mach_cone_angle_and_supersonic_shockwaves(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        # Subsonic (M < 1): no Mach cone
        item.set_mach_number(0.8)
        assert item.mach_angle_deg is None

        # Sonic barrier (M = 1.0): sin(μ) = 1/1 ==> μ = 90° (normal plane)
        item.set_mach_number(1.0)
        assert item.mach_angle_deg is not None
        assert abs(item.mach_angle_deg - 90.0) < 0.01

        # Supersonic M = 1.5: sin(μ) = 1/1.5 ==> μ ≈ 41.81°
        item.set_mach_number(1.5)
        assert item.mach_angle_deg is not None
        expected_mu = math.degrees(math.asin(1.0 / 1.5))
        assert abs(item.mach_angle_deg - expected_mu) < 0.01
        assert 41.7 < item.mach_angle_deg < 41.9

        # Supersonic M = 2.0: sin(μ) = 1/2.0 ==> μ = 30.0°
        item.set_mach_number(2.0)
        assert item.mach_angle_deg is not None
        assert abs(item.mach_angle_deg - 30.0) < 0.01
    finally:
        item.cleanup()


def test_all_curriculum_presets(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        preset_keys = [
            "SUBSONIC_MOVING_SOURCE",
            "SONIC_BARRIER_MACH_ONE",
            "SUPERSONIC_MACH_CONE",
            "DOPPLER_RADAR_ECHOCARDIOGRAPHY",
            "RELATIVISTIC_COSMOLOGICAL_REDSHIFT",
        ]
        for key in preset_keys:
            item.load_preset(key)
            assert item.active_preset.code == key
            assert len(item.active_preset.title_vi) > 0
            assert len(item.active_preset.description) > 0
            assert item.mach_number == item.active_preset.mach_number
    finally:
        item.cleanup()


def test_radar_and_cosmological_redshift_formulas(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        # Radar beat frequency
        item.load_preset("DOPPLER_RADAR_ECHOCARDIOGRAPHY")
        assert item.beat_frequency_radar_hz > 0.0
        # Check order of magnitude (~kHz for 24 GHz radar at vehicle speed)
        assert item.beat_frequency_radar_hz > 1000.0

        # Cosmological redshift
        item.load_preset("RELATIVISTIC_COSMOLOGICAL_REDSHIFT")
        assert item.cosmological_redshift_z > 0.0
        # For M = 0.85 (beta ~ 0.51), z = sqrt(1.51/0.49) - 1 ≈ 0.75
        assert 0.4 < item.cosmological_redshift_z < 2.0
    finally:
        item.cleanup()


def test_parameter_controls_and_steppers(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        # Mach clamping
        item.set_mach_number(1.8)
        assert item.mach_number == 1.8
        item.set_mach_number(-1.0)
        assert item.mach_number == 0.0
        item.set_mach_number(5.0)
        assert item.mach_number == 2.5

        # Frequency setter & clamping
        item.set_source_frequency_hz(500.0)
        assert item.source_frequency_hz == 500.0
        item.set_source_frequency_hz(50.0)
        assert item.source_frequency_hz == 100.0
        item.set_source_frequency_hz(5000.0)
        assert item.source_frequency_hz == 2000.0

        # Sound speed setter & clamping
        item.set_sound_speed_mps(1482.0)
        assert item.sound_speed_mps == 1482.0
        item.set_sound_speed_mps(20.0)
        assert item.sound_speed_mps == 100.0
        item.set_sound_speed_mps(5000.0)
        assert item.sound_speed_mps == 3000.0

        # Toggle direction
        init_dir = item._direction
        item.toggle_direction()
        assert item._direction == -init_dir
        item.toggle_direction()
        assert item._direction == init_dir
    finally:
        item.cleanup()


def test_animation_and_wavefront_generation(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        assert len(item._wavefronts) == 0

        # Run several ticks
        for _ in range(40):
            item._on_tick()

        # Wavefronts should be created
        assert len(item._wavefronts) > 0

        # Reset clears wavefronts
        item.reset_wavefronts()
        assert len(item._wavefronts) == 0
        assert item._sim_time == 0.0

        # Animation toggle
        assert item.is_animating is True
        item.toggle_animation()
        assert item.is_animating is False
        item.toggle_animation()
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_touchless_air_gestures(qapp):
    item = InteractiveDopplerEffectItem(pos=(0.0, 0.0))
    try:
        # Pre-render to build button rects cache
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        item.paint(p, None, None)
        p.end()

        # 1. Preset tabs
        r_sup = item._btn_rects.get("preset_SUPERSONIC_MACH_CONE")
        assert r_sup is not None
        assert item.handle_air_pinch(r_sup.center()) is True
        assert item.active_preset.code == "SUPERSONIC_MACH_CONE"

        # 2. Mach steppers
        r_mp01 = item._btn_rects.get("m_p01")
        assert r_mp01 is not None
        cur_m = item.mach_number
        assert item.handle_air_pinch(r_mp01.center()) is True
        assert abs(item.mach_number - (cur_m + 0.1)) < 1e-4

        # 3. Direction toggle
        r_dir = item._btn_rects.get("toggle_dir")
        assert r_dir is not None
        old_dir = item._direction
        assert item.handle_air_pinch(r_dir.center()) is True
        assert item._direction == -old_dir

        # 4. Out of bounds returns False
        assert item.handle_air_pinch(QPointF(-100.0, -100.0)) is False
    finally:
        item.cleanup()


def test_paint_rendering_all_presets(qapp):
    item = InteractiveDopplerEffectItem()
    try:
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        for preset_key in PRESETS:
            item.load_preset(preset_key)
            # Advance ticks to generate wavefronts
            for _ in range(15):
                item._on_tick()

            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()
            assert not img.isNull()
    finally:
        item.cleanup()


def test_stem_tools_vector_diagram(qapp):
    presets = [
        ("SUBSONIC_MOVING_SOURCE", 0.5, 880.0, 293.3, None),
        ("SONIC_BARRIER_MACH_ONE", 1.0, float("inf"), 220.0, 90.0),
        ("SUPERSONIC_MACH_CONE", 1.5, float("inf"), 176.0, 41.8),
    ]

    for p_code, mach, f_ah, f_be, mu in presets:
        items = PhysicsTools.create_doppler_effect_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code=p_code,
            mach_number=mach,
            freq_ahead_hz=f_ah,
            freq_behind_hz=f_be,
            mach_angle_deg=mu,
        )
        assert len(items) >= 5
        assert items[0].rect().width() == 640.0
        assert items[0].rect().height() == 340.0


def test_lesson_board_integration(qapp):
    board = LessonBoard()
    try:
        assert board.is_doppler_effect_visible() is False
        assert board.get_doppler_effect() is None

        # Toggle ON
        vis1 = board.toggle_doppler_effect()
        assert vis1 is True
        assert board.is_doppler_effect_visible() is True
        item = board.get_doppler_effect()
        assert item is not None
        assert isinstance(item, InteractiveDopplerEffectItem)

        # Excluded from stroke selection
        excluded = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert item not in excluded

        # Toggle OFF
        vis2 = board.toggle_doppler_effect()
        assert vis2 is False
        assert board.is_doppler_effect_visible() is False
        assert board.get_doppler_effect() is None
        assert board.get_doppler_effect_item() is not None
    finally:
        item = board.get_doppler_effect_item()
        if item is not None and hasattr(item, "cleanup"):
            item.cleanup()
