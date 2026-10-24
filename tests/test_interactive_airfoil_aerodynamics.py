"""
Unit test suite for InteractiveAirfoilAerodynamicsItem (Milestone 105).
Covers NACA 4-digit airfoil geometry generation, thin airfoil theory, stall hysteresis,
lift and drag force polar curves, streamline particles, touchless air gestures, stamping, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_airfoil_aerodynamics import (
    InteractiveAirfoilAerodynamicsItem,
    PRESETS,
    AirfoilPreset,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


class TestInteractiveAirfoilAerodynamics:

    def test_initialization(self, qapp):
        """Verifies default properties, dimensions, and flags of InteractiveAirfoilAerodynamicsItem."""
        item = InteractiveAirfoilAerodynamicsItem(pos=(180.0, 100.0))
        assert item.pos() == QPointF(180.0, 100.0)
        assert item.WIDTH == 640.0
        assert item.HEIGHT == 380.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 640.0, 380.0)
        assert item.active_preset.code == "NACA_2412_GENERAL_AVIATION"
        assert item.angle_of_attack_deg == 4.0
        assert item.wind_speed_ms == 35.0
        assert item.is_animating is True
        assert item.is_stalled is False
        item.cleanup()

    def test_all_five_presets(self, qapp):
        """Verifies that all 5 airfoil presets load cleanly and update aerodynamic constants."""
        item = InteractiveAirfoilAerodynamicsItem()
        preset_codes = [
            "NACA_2412_GENERAL_AVIATION",
            "NACA_0012_SYMMETRIC",
            "NACA_4415_HIGH_LIFT",
            "SUPERCRITICAL_TRANSONIC",
            "SYMMETRIC_THIN_AEROFOIL",
        ]
        for code in preset_codes:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert 0.0 <= item.active_preset.max_camber_m <= 0.05
            assert 0.04 <= item.active_preset.thickness_t <= 0.20
            assert -5.0 <= item.active_preset.alpha_0_deg <= 0.5
            assert 10.0 <= item.active_preset.alpha_stall_deg <= 20.0
        item.cleanup()

    def test_naca_coordinates_geometry(self, qapp):
        """Verifies analytical NACA 4-digit airfoil upper and lower coordinate generation."""
        item = InteractiveAirfoilAerodynamicsItem()

        # 1. Test Symmetric NACA 0012: upper and lower thickness must be symmetric
        item.load_preset("NACA_0012_SYMMETRIC")
        upper_pts, lower_pts = item.generate_naca_coordinates(30)
        assert len(upper_pts) == 31
        assert len(lower_pts) == 31

        # Leading edge and trailing edge close at y ~ 0
        assert abs(upper_pts[0][1]) < 0.01
        assert abs(lower_pts[0][1]) < 0.01

        # Symmetric check: y_upper == -y_lower for x at mid-chord
        mid_idx = 15
        assert abs(upper_pts[mid_idx][1] + lower_pts[mid_idx][1]) < 1e-3

        # 2. Test Cambered NACA 2412: upper surface has positive camber
        item.load_preset("NACA_2412_GENERAL_AVIATION")
        u2412, l2412 = item.generate_naca_coordinates(30)
        # Cambered airfoil has higher upper surface than lower surface depth
        assert u2412[mid_idx][1] > abs(l2412[mid_idx][1])
        item.cleanup()

    def test_thin_airfoil_lift_and_stall(self, qapp):
        """Verifies lift coefficient C_L behavior: zero lift at alpha_0, linear slope, and stall drop-off."""
        item = InteractiveAirfoilAerodynamicsItem()
        item.load_preset("NACA_0012_SYMMETRIC")  # alpha_0 = 0.0 deg, stall = 14.0 deg

        # Zero lift at zero AoA
        item.set_angle_of_attack(0.0)
        assert abs(item.lift_coefficient) < 0.01
        assert item.is_stalled is False

        # Linear slope region: C_L ≈ 2*pi*alpha_rad ≈ 0.11 * alpha_deg
        item.set_angle_of_attack(5.0)
        expected_cl_5 = 2.0 * math.pi * math.radians(5.0)  # ~0.548
        assert abs(item.lift_coefficient - expected_cl_5) < 0.05
        assert item.is_stalled is False

        # Stall region: alpha > 14.0 deg
        item.set_angle_of_attack(18.0)
        assert item.is_stalled is True
        # Past stall, C_L must be lower than purely extrapolated linear slope (2*pi*18deg ~ 1.97)
        assert item.lift_coefficient < 2.0 * math.pi * math.radians(18.0)
        item.cleanup()

    def test_drag_polar_and_induced_drag(self, qapp):
        """Verifies total drag C_D = C_D0 + C_L^2 / (pi*e*AR) + C_D_sep."""
        item = InteractiveAirfoilAerodynamicsItem()
        item.load_preset("NACA_2412_GENERAL_AVIATION")

        # Zero lift AoA produces minimum drag C_D0
        item.set_angle_of_attack(item.active_preset.alpha_0_deg)
        cd_min = item.drag_coefficient
        assert abs(cd_min - item.active_preset.cd0) < 0.005

        # Higher AoA increases induced drag
        item.set_angle_of_attack(8.0)
        cd_8 = item.drag_coefficient
        assert cd_8 > cd_min

        # Stall surges separation drag
        item.set_angle_of_attack(22.0)
        cd_stall = item.drag_coefficient
        assert cd_stall > cd_8 * 2.0
        item.cleanup()

    def test_aerodynamic_forces_and_pressure(self, qapp):
        """Verifies dynamic pressure q = 0.5*rho*v^2, Lift L = C_L*q*S, and Drag D = C_D*q*S."""
        item = InteractiveAirfoilAerodynamicsItem()
        item.set_wind_speed(40.0)
        item.set_angle_of_attack(6.0)

        # q = 0.5 * 1.225 * 40^2 = 980 Pa
        expected_q = 0.5 * 1.225 * (40.0 ** 2)
        assert abs(item.dynamic_pressure_pa - expected_q) < 1e-3

        expected_lift = item.lift_coefficient * expected_q * item._wing_area_s
        expected_drag = item.drag_coefficient * expected_q * item._wing_area_s

        assert abs(item.lift_force_n - expected_lift) < 1e-2
        assert abs(item.drag_force_n - expected_drag) < 1e-2
        assert item.lift_to_drag_ratio == item.lift_coefficient / item.drag_coefficient
        item.cleanup()

    def test_parameter_setters_and_clamping(self, qapp):
        """Verifies clamping of Angle of Attack [-10, 25] and Wind Speed [10, 80]."""
        item = InteractiveAirfoilAerodynamicsItem()

        # AoA clamping
        item.set_angle_of_attack(-30.0)
        assert item.angle_of_attack_deg == -10.0
        item.set_angle_of_attack(50.0)
        assert item.angle_of_attack_deg == 25.0

        # Wind speed clamping
        item.set_wind_speed(2.0)
        assert item.wind_speed_ms == 10.0
        item.set_wind_speed(150.0)
        assert item.wind_speed_ms == 80.0
        item.cleanup()

    def test_animation_toggle_and_particles(self, qapp):
        """Verifies smoke streak particle integration and animation toggle."""
        item = InteractiveAirfoilAerodynamicsItem()
        assert item.is_animating is True

        p0_x = item._particles[0]["x"]
        item._on_animation_tick()
        assert item._particles[0]["x"] > p0_x

        # Toggle off
        is_now_off = item.toggle_animation()
        assert is_now_off is False
        assert item.is_animating is False

        # Toggle on
        is_now_on = item.toggle_animation()
        assert is_now_on is True
        assert item.is_animating is True
        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        """Verifies touchless air pinch interaction on buttons, steppers, and wind tunnel."""
        item = InteractiveAirfoilAerodynamicsItem(pos=(100.0, 100.0))

        # Air pinch outside
        assert item.handle_air_pinch(QPointF(50.0, 50.0)) is False

        # Air pinch on NACA 0012 preset button
        r_naca0012 = item._btn_rects.get("preset_NACA_0012_SYMMETRIC")
        assert r_naca0012 is not None
        item.handle_air_pinch(item.mapToScene(r_naca0012.center()))
        assert item.active_preset.code == "NACA_0012_SYMMETRIC"

        # Air pinch on AoA up stepper
        r_au = item._btn_rects.get("aoa_up")
        cur_aoa = item.angle_of_attack_deg
        item.handle_air_pinch(item.mapToScene(r_au.center()))
        assert item.angle_of_attack_deg == cur_aoa + 1.0

        # Air pinch on Wind speed down stepper
        r_vd = item._btn_rects.get("v_down")
        cur_v = item.wind_speed_ms
        item.handle_air_pinch(item.mapToScene(r_vd.center()))
        assert item.wind_speed_ms == cur_v - 5.0

        # Air pinch inside Wind Tunnel Chamber to adjust AoA
        chamber_pt = QPointF(190.0, 140.0)  # Above center -> positive AoA
        item.handle_air_pinch(item.mapToScene(chamber_pt))
        assert item.angle_of_attack_deg > 0.0
        item.cleanup()

    def test_stem_tools_diagram_generation_and_stamp(self, qapp):
        """Verifies PhysicsTools.create_airfoil_diagram and stamping to LessonBoard."""
        board = LessonBoard()
        item = InteractiveAirfoilAerodynamicsItem(pos=QPointF(100.0, 100.0))
        board.addItem(item)
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Test direct diagram creation via PhysicsTools
        items = PhysicsTools.create_airfoil_diagram(
            center=(400.0, 300.0),
            preset_code="NACA_2412_GENERAL_AVIATION",
            aoa_deg=5.0,
            wind_speed_ms=40.0,
            cl=0.72,
            cd=0.03,
            is_stalled=False,
        )
        assert len(items) >= 12
        text_contents = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("KHÍ ĐỘNG HỌC" in t for t in text_contents)
        assert any("Bernoulli" in t for t in text_contents)
        item.cleanup()

    def test_lesson_board_lifecycle_and_lasso_exclusion(self, qapp):
        """Verifies LessonBoard show/toggle/getter lifecycle and lasso exclusion."""
        board = LessonBoard()

        # Initially not visible
        assert board.is_airfoil_aerodynamics_visible() is False
        assert board.get_airfoil_aerodynamics() is None

        # Show
        item = board.show_airfoil_aerodynamics(True, pos=(200.0, 150.0))
        assert item is not None
        assert board.is_airfoil_aerodynamics_visible() is True
        assert board.get_airfoil_aerodynamics() is item
        assert board.get_airfoil_aerodynamics_item() is item
        assert item.pos() == QPointF(200.0, 150.0)

        # Toggle hide
        new_state = board.toggle_airfoil_aerodynamics()
        assert new_state is False
        assert board.is_airfoil_aerodynamics_visible() is False
        assert board.get_airfoil_aerodynamics() is None

        # Toggle show
        new_state_2 = board.toggle_airfoil_aerodynamics()
        assert new_state_2 is True
        assert board.is_airfoil_aerodynamics_visible() is True

        # Verify exclusion in lasso selection
        rect = QRectF(100.0, 100.0, 500.0, 500.0)
        items_in_rect = board.get_items_in_rect(rect)
        assert item not in items_in_rect
        item.cleanup()

    def test_main_window_toggle_slot(self, qapp):
        """Verifies MainWindow slot _on_toggle_airfoil_aerodynamics properly delegates to LessonBoard."""
        from src.ui.main_window import MainWindow
        win = MainWindow()
        assert win._scene.is_airfoil_aerodynamics_visible() is False

        # Toggle via slot
        res = win._on_toggle_airfoil_aerodynamics()
        assert res is True
        assert win._scene.is_airfoil_aerodynamics_visible() is True

        # Toggle off
        res_off = win._on_toggle_airfoil_aerodynamics()
        assert res_off is False
        assert win._scene.is_airfoil_aerodynamics_visible() is False
        win.close()
