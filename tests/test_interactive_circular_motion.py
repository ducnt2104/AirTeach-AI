"""
AirTeach AI - Unit Tests for Interactive Physics Circular Motion & Centripetal Acceleration Simulator
=====================================================================================================
Comprehensive automated unit tests verifying:
  1. Authentic circular kinematics: angular velocity ω = v/R, period T = 2π/ω, frequency f = 1/T.
  2. Centripetal acceleration a_c = v²/R and centripetal force F_c = m·v²/R.
  3. Banked curve road physics: optimal angle θ_ideal = arctan(v²/(gR)) and friction limit v_max = √(μ_s·g·R).
  4. Vertical loop minimum critical apex velocity v_top_min = √(gR) and bottom normal force N_bottom.
  5. 5 Curriculum Presets (Turntable, Banked Curve, Conical Pendulum, Vertical Loop, Satellite Orbit).
  6. Air pinch gesture interactions, parameter stepper buttons, and orbital animation tick.
  7. Vector diagram stamping via PhysicsTools.create_circular_motion_diagram.
  8. LessonBoard lifecycle and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_circular_motion import (
    GRAVITY_G,
    InteractiveCircularMotionItem,
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


class TestCircularMotionKinematics:
    """Verifies authentic physics formulas for uniform circular motion and centripetal dynamics."""

    def test_basic_kinematics_and_centripetal_quantities(self, qapp):
        item = InteractiveCircularMotionItem()
        item.set_radius(4.0)  # R = 4.0 m
        item.set_speed(8.0)   # v = 8.0 m/s
        item.set_mass(2.0)    # m = 2.0 kg

        # ω = v / R = 8.0 / 4.0 = 2.0 rad/s
        assert abs(item.angular_velocity_rads - 2.0) < 1e-4

        # T = 2π / ω = π ≈ 3.14159 s
        assert abs(item.period_seconds - math.pi) < 1e-4

        # f = 1 / T = 1 / π ≈ 0.3183 Hz
        assert abs(item.frequency_hz - (1.0 / math.pi)) < 1e-4

        # a_c = v² / R = 64.0 / 4.0 = 16.0 m/s²
        assert abs(item.centripetal_acceleration_ms2 - 16.0) < 1e-4

        # F_c = m · a_c = 2.0 * 16.0 = 32.0 N
        assert abs(item.centripetal_force_newtons - 32.0) < 1e-4

        item.cleanup()

    def test_banked_curve_optimal_angle(self, qapp):
        item = InteractiveCircularMotionItem()
        item.load_preset("BANKED_CURVE_HIGHWAY")
        # R = 50.0 m, v = 20.0 m/s
        # tan(θ) = v² / (g·R) = 400.0 / (9.80665 * 50.0) = 400 / 490.3325 ≈ 0.81577
        # θ_ideal = arctan(0.81577) in degrees ≈ 39.2°
        theta_ideal = item.ideal_banking_angle_deg
        expected_theta = math.degrees(math.atan((20.0 ** 2) / (GRAVITY_G * 50.0)))
        assert abs(theta_ideal - expected_theta) < 1e-3
        assert 35.0 < theta_ideal < 45.0

        item.cleanup()

    def test_horizontal_turntable_friction_threshold(self, qapp):
        item = InteractiveCircularMotionItem()
        item.load_preset("HORIZONTAL_TURNTABLE")
        # R = 2.0 m, μ_s = 0.55
        # v_max = √(μ_s · g · R) = √(0.55 * 9.80665 * 2.0) = √(10.7873) ≈ 3.284 m/s
        v_max_safe = item.max_safe_friction_speed_ms
        expected_v_max = math.sqrt(0.55 * GRAVITY_G * 2.0)
        assert abs(v_max_safe - expected_v_max) < 1e-3

        item.cleanup()

    def test_vertical_loop_critical_apex_speed(self, qapp):
        item = InteractiveCircularMotionItem()
        item.load_preset("VERTICAL_LOOP_ROLLERCOASTER")
        # R = 8.0 m
        # v_top_min = √(g·R) = √(9.80665 * 8.0) ≈ 8.857 m/s
        v_top_crit = item.critical_vertical_top_speed_ms
        expected_crit = math.sqrt(GRAVITY_G * 8.0)
        assert abs(v_top_crit - expected_crit) < 1e-3

        # Normal force at bottom: N = m(v²/R + g)
        item.set_speed(10.0)
        item.set_mass(350.0)
        # a_c = 100 / 8 = 12.5 m/s²
        # N = 350 * (12.5 + 9.80665) = 350 * 22.30665 = 7807.3 N
        n_bot = item.vertical_bottom_normal_force_newtons
        expected_n = 350.0 * (12.5 + GRAVITY_G)
        assert abs(n_bot - expected_n) < 1e-3

        item.cleanup()


class TestCircularMotionInteraction:
    """Verifies presets, air pinch gestures, parameter stepping, and vector stamping."""

    def test_all_five_presets_load(self, qapp):
        item = InteractiveCircularMotionItem()
        for code in ["HORIZONTAL_TURNTABLE", "BANKED_CURVE_HIGHWAY", "CONICAL_PENDULUM", "VERTICAL_LOOP_ROLLERCOASTER", "GEOCENTRIC_SATELLITE_ORBIT"]:
            ok = item.load_preset(code)
            assert ok is True
            assert item.active_preset_code == code
            assert item.preset is not None
            assert item.radius_m > 0.0
            assert item.speed_ms > 0.0
            assert item.mass_kg > 0.0
        item.cleanup()

    def test_air_pinch_preset_switch(self, qapp):
        item = InteractiveCircularMotionItem()
        btn_r = item._btn_rects["preset_VERTICAL_LOOP_ROLLERCOASTER"]
        pinch_pt = item.mapToScene(btn_r.center())

        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "VERTICAL_LOOP_ROLLERCOASTER"

        item.cleanup()

    def test_air_pinch_parameter_steppers(self, qapp):
        item = InteractiveCircularMotionItem()
        r_init = item.radius_m
        v_init = item.speed_ms
        m_init = item.mass_kg

        # Increase radius
        inc_r_pt = item.mapToScene(item._btn_rects["inc_radius"].center())
        assert item.handle_air_pinch(inc_r_pt) is True
        assert item.radius_m > r_init

        # Decrease speed
        dec_v_pt = item.mapToScene(item._btn_rects["dec_speed"].center())
        assert item.handle_air_pinch(dec_v_pt) is True
        assert item.speed_ms < v_init

        # Increase mass
        inc_m_pt = item.mapToScene(item._btn_rects["inc_mass"].center())
        assert item.handle_air_pinch(inc_m_pt) is True
        assert item.mass_kg > m_init

        item.cleanup()

    def test_animation_tick_and_toggle(self, qapp):
        item = InteractiveCircularMotionItem()
        ang_start = item._current_angle_rad

        item._on_animation_tick()
        assert item._current_angle_rad != ang_start
        assert item._elapsed_time_s > 0.0

        # Toggle pause
        is_anim = item.toggle_animation()
        assert is_anim is False
        assert item.is_animating is False

        ang_paused = item._current_angle_rad
        item._on_animation_tick()
        assert item._current_angle_rad == ang_paused  # No advance when paused

        item.cleanup()

    def test_stamp_creates_vector_diagram(self, qapp):
        board = LessonBoard()
        item = InteractiveCircularMotionItem(pos=(200.0, 100.0))
        board.addItem(item)

        count_before = len(board.items())
        item.stamp_to_board()
        count_after = len(board.items())
        assert count_after > count_before

        item.cleanup()

    def test_physics_tools_vector_diagram(self, qapp):
        items = PhysicsTools.create_circular_motion_diagram(
            center=(400.0, 300.0),
            preset_code="HORIZONTAL_TURNTABLE",
            radius_m=2.5,
            speed_ms=5.0,
            mass_kg=1.0,
        )
        assert len(items) >= 10
        texts = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("CHUYỂN ĐỘNG TRÒN ĐỀU" in t.upper() for t in texts)
        assert any("LỰC HƯỚNG TÂM" in t.upper() or "F_c" in t for t in texts)


class TestCircularMotionBoardIntegration:
    """Verifies LessonBoard lifecycle and exclusion from lasso selection."""

    def test_board_lifecycle(self, qapp):
        board = LessonBoard()
        assert board.is_circular_motion_visible() is False
        assert board.get_circular_motion() is None

        # Show
        item = board.show_circular_motion(True)
        assert item is not None
        assert board.is_circular_motion_visible() is True
        assert board.get_circular_motion() is item
        assert board.get_circular_motion_item() is item

        # Toggle off
        vis = board.toggle_circular_motion()
        assert vis is False
        assert board.is_circular_motion_visible() is False

        # Toggle on
        vis = board.toggle_circular_motion()
        assert vis is True
        assert board.is_circular_motion_visible() is True

        item.cleanup()

    def test_lasso_exclusion(self, qapp):
        board = LessonBoard()
        item = board.show_circular_motion(True, pos=(250.0, 150.0))
        rect = QRectF(200.0, 100.0, 800.0, 600.0)

        # Simulator widget should be excluded from selection lasso
        selected_items = board.get_items_in_rect(rect)
        assert item not in selected_items

        item.cleanup()
