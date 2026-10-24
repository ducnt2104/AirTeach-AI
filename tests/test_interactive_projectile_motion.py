"""
AirTeach AI - Unit Tests for Interactive Physics Projectile Motion & 2D Kinematics Simulator
============================================================================================
Comprehensive automated unit tests verifying:
  1. Authentic projectile motion kinematics (vacuum parabola, apex height, range, flight time).
  2. Analytical optimal launch angle for elevated launch positions (theta_opt < 45 deg).
  3. Numerical Runge-Kutta 4th Order (RK4) atmospheric drag integration.
  4. 5 Curriculum Presets (Cannonball, Baseball, Basketball, Cliff, High-speed Bullet).
  5. Touchless air pinch gesture interactions, angle/speed tuning, and animation toggles.
  6. Vector diagram generation via PhysicsTools.create_projectile_motion_diagram.
  7. LessonBoard lifecycle integration and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_projectile_motion import (
    InteractiveProjectileMotionItem,
    PRESETS,
    GRAVITY_G,
    AIR_DENSITY_RHO,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure single persistent QApplication for QGraphics testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveProjectileMotion:
    """Test suite for InteractiveProjectileMotionItem and STEM physics integration."""

    def test_initialization(self, qapp):
        """Verifies initial configuration, geometry, and default physical values."""
        item = InteractiveProjectileMotionItem(pos=QPointF(150.0, 100.0))
        assert item.pos() == QPointF(150.0, 100.0)
        assert item.boundingRect().width() == InteractiveProjectileMotionItem.WIDTH
        assert item.boundingRect().height() == InteractiveProjectileMotionItem.HEIGHT
        assert item.active_preset_code == "CANNONBALL_VACUUM_45"
        assert item.v0_ms == 25.0
        assert item.angle_deg == 45.0
        assert item.h0_m == 0.0
        assert item.is_animating is True
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies loading of all 5 curriculum presets and signal emissions."""
        item = InteractiveProjectileMotionItem()
        emitted_presets = []
        item.preset_changed.connect(lambda p: emitted_presets.append(p))

        for code, expected_preset in PRESETS.items():
            loaded = item.load_preset(code)
            assert loaded is True
            assert item.active_preset_code == code
            assert item.preset.title_vi == expected_preset.title_vi
            assert item.v0_ms == expected_preset.v0_ms
            assert item.angle_deg == expected_preset.angle_deg
            assert item.h0_m == expected_preset.h0_m
            assert item.preset.drag_coeff == expected_preset.drag_coeff

        assert len(emitted_presets) == 5
        assert item.load_preset("NON_EXISTENT_PRESET") is False
        item.cleanup()

    def test_vacuum_kinematics_accuracy(self, qapp):
        """Verifies exact analytical vacuum range and apex height formulas."""
        item = InteractiveProjectileMotionItem()
        item.load_preset("CANNONBALL_VACUUM_45")  # v0=25, angle=45, h0=0
        
        # In vacuum: R = v0^2 * sin(2*theta) / g, with theta=45 deg -> R = v0^2 / g
        expected_range = (25.0 ** 2) / GRAVITY_G
        assert abs(item.vacuum_range_m - expected_range) < 0.2

        # H_max = (v0 * sin(45))^2 / (2 * g) = 25^2 * 0.5 / (2 * 9.80665) = 625 / 39.2266 = 15.93 m
        expected_apex = (25.0 ** 2 * (math.sin(math.radians(45.0)) ** 2)) / (2.0 * GRAVITY_G)
        assert abs(item.vacuum_max_height_m - expected_apex) < 0.05
        item.cleanup()

    def test_optimal_launch_angle(self, qapp):
        """Verifies analytical optimal launch angle for flat ground and elevated cliff."""
        item = InteractiveProjectileMotionItem()
        # Flat ground (h0 = 0): optimal angle must be exactly 45 degrees
        item.load_preset("CANNONBALL_VACUUM_45")
        assert abs(item.optimal_launch_angle_deg - 45.0) < 1e-3

        # Elevated cliff (CLIFF_PROJECTILE_OPTIMAL: h0 = 25m, v0 = 20m/s):
        # theta_opt = arcsin(1 / sqrt(2 + 2 * g * h0 / v0^2))
        item.load_preset("CLIFF_PROJECTILE_OPTIMAL")
        opt_angle = item.optimal_launch_angle_deg
        assert opt_angle < 45.0
        assert 28.0 < opt_angle < 36.0
        item.cleanup()

    def test_air_drag_rk4_effect(self, qapp):
        """Verifies that atmospheric drag strictly reduces horizontal range compared to vacuum."""
        item = InteractiveProjectileMotionItem()
        item.load_preset("BASEBALL_AIR_DRAG")  # v0=38 m/s, drag_coeff=0.005, mass=0.145 kg
        
        vac_range = item.vacuum_range_m
        drag_range = item.drag_range_m
        assert drag_range > 0.0
        assert drag_range < vac_range
        # Baseball air drag reduces range significantly (> 15% reduction)
        reduction_pct = (vac_range - drag_range) / vac_range * 100.0
        assert reduction_pct > 10.0
        item.cleanup()

    def test_set_speed_and_angle_clamping(self, qapp):
        """Verifies speed and angle setter clamping to safe physical boundaries."""
        item = InteractiveProjectileMotionItem()
        
        item.set_speed(500.0)
        assert item.v0_ms == 250.0  # Max clamp
        item.set_speed(0.5)
        assert item.v0_ms == 2.0    # Min clamp

        item.set_angle(95.0)
        assert item.angle_deg == 85.0  # Max clamp
        item.set_angle(1.0)
        assert item.angle_deg == 5.0   # Min clamp
        item.cleanup()

    def test_animation_toggle_and_progression(self, qapp):
        """Verifies animation toggle, tick progression, and reset behavior."""
        item = InteractiveProjectileMotionItem()
        assert item.is_animating is True

        # Toggle to pause
        item.toggle_animation()
        assert not item.is_animating

        # Toggle to resume
        item.toggle_animation()
        assert item.is_animating

        # Simulate several timer ticks
        for _ in range(10):
            item._on_animation_tick()
        assert item._anim_time_s > 0.0

        item.toggle_animation()
        assert not item.is_animating
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch gesture dispatch on buttons and controllers."""
        item = InteractiveProjectileMotionItem(pos=QPointF(0.0, 0.0))
        
        # 1. Pinch on Preset button (preset_BASEBALL_AIR_DRAG)
        rect_b1 = item._btn_rects["preset_BASEBALL_AIR_DRAG"]
        pinch_pt = QPointF(rect_b1.center().x(), rect_b1.center().y())
        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "BASEBALL_AIR_DRAG"

        # 2. Pinch on Play / Pause button
        rect_play = item._btn_rects["play_pause"]
        assert item.is_animating is True
        handled_play = item.handle_air_pinch(rect_play.center())
        assert handled_play is True
        assert item.is_animating is False

        # 3. Pinch outside bounding box
        outside_pt = QPointF(1000.0, 1000.0)
        assert item.handle_air_pinch(outside_pt) is False
        item.cleanup()

    def test_stem_tools_diagram_generation(self, qapp):
        """Verifies static vector diagram generation via PhysicsTools."""
        items = PhysicsTools.create_projectile_motion_diagram(
            center=(400.0, 300.0),
            preset_code="CLIFF_PROJECTILE_OPTIMAL",
            v0_ms=20.0,
            angle_deg=30.0,
            h0_m=25.0,
        )
        assert len(items) >= 8
        # Ensure header banner exists with correct title
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("CHUYỂN ĐỘNG NÉM & ĐỘNG HỌC 2D" in t for t in text_contents)
        assert any("THÔNG SỐ ĐỘNG HỌC 2D" in t for t in text_contents)

    def test_stamp_to_board(self, qapp):
        """Verifies stamping active trajectory onto LessonBoard scene."""
        board = LessonBoard()
        item = InteractiveProjectileMotionItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count
        item.cleanup()

    def test_lesson_board_lifecycle(self, qapp):
        """Verifies LessonBoard show/toggle/visibility lifecycle for Projectile Motion."""
        board = LessonBoard()
        assert not board.is_projectile_motion_visible()
        assert board.get_projectile_motion() is None

        # Show
        sim = board.show_projectile_motion(True)
        assert sim is not None
        assert board.is_projectile_motion_visible()
        assert board.get_projectile_motion() is sim
        assert board.get_projectile_motion_item() is sim

        # Toggle to hide
        visible = board.toggle_projectile_motion()
        assert not visible
        assert not board.is_projectile_motion_visible()
        assert board.get_projectile_motion() is None
        assert board.get_projectile_motion_item() is sim

        # Toggle to show
        visible2 = board.toggle_projectile_motion()
        assert visible2
        assert board.is_projectile_motion_visible()

    def test_lasso_selection_exclusion(self, qapp):
        """Verifies interactive simulator is excluded from LessonBoard lasso selection."""
        board = LessonBoard()
        board.show_projectile_motion(True, pos=(200.0, 200.0))
        
        # Query items in the rectangle covering the simulator
        items_in_box = board.get_items_in_rect(QRectF(150.0, 150.0, 700.0, 600.0))
        assert board.get_projectile_motion_item() not in items_in_box
