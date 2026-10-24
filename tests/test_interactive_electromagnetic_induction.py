"""
AirTeach AI - Unit Tests for Interactive Physics Electromagnetic Induction Simulator
=====================================================================================
Comprehensive automated unit tests verifying:
  1. Magnetic flux calculations Φ = N·B·S·cos(α) and Faraday's Law E = -dΦ/dt.
  2. AC Generator peak electromotive force E_0 = ω·N·B·S and π/2 phase lag vs flux.
  3. Bar magnet through solenoid induction: polarity reversal upon entering vs exiting.
  4. Sliding conductor on rails EMF E = B·l·v and Lenz's Law opposing magnetic forces.
  5. 5 Curriculum Presets (Magnet/Solenoid, AC Generator, Transformer, Eddy Current, Sliding Rail).
  6. Air pinch gesture interactions, parameter stepper buttons, and oscilloscope buffer updating.
  7. Vector diagram stamping via PhysicsTools.create_electromagnetic_induction_diagram.
  8. LessonBoard lifecycle and lasso selection exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_electromagnetic_induction import (
    InteractiveElectromagneticInductionItem,
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


class TestInductionElectromagnetics:
    """Verifies authentic physical electromagnetic formulas and Faraday/Lenz equations."""

    def test_ac_generator_peak_emf_and_flux(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        item.load_preset("ROTATING_AC_GENERATOR")

        # Preset values: N = 200, B = 0.25 T, S = 50 cm² = 0.005 m², ω = 31.42 rad/s
        n = item.turns_n
        b = item.b_field
        s = item._coil_area_m2
        omega = item.speed_param

        # Peak flux: Φ_0 = N·B·S = 200 * 0.25 * 0.005 = 0.25 Wb
        phi_0 = n * b * s
        assert abs(phi_0 - 0.25) < 1e-4

        # Peak EMF: E_0 = ω·Φ_0 = 31.42 * 0.25 = 7.855 V
        e_0 = omega * phi_0
        assert abs(e_0 - (31.42 * 0.25)) < 1e-3

        # At α = 0: Φ = Φ_0, E = 0
        item._coil_angle_rad = 0.0
        assert abs(item.current_magnetic_flux_webers - phi_0) < 1e-4
        assert abs(item.current_induced_emf_volts - 0.0) < 1e-4

        # At α = π/2: Φ = 0, E = E_0 (Quarter period phase shift)
        item._coil_angle_rad = math.pi / 2.0
        assert abs(item.current_magnetic_flux_webers - 0.0) < 1e-4
        assert abs(item.current_induced_emf_volts - e_0) < 1e-3

        item.cleanup()

    def test_bar_magnet_solenoid_polarity_reversal(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        item.load_preset("BAR_MAGNET_SOLENOID")

        # Approach from above (z = -60 mm, v_z > 0 approaching center z=0)
        item._magnet_z_pos = -60.0
        item._magnet_velocity = 1.0
        emf_approaching = item.current_induced_emf_volts

        # Recede away below (z = +60 mm, v_z > 0 moving away from center z=0)
        item._magnet_z_pos = 60.0
        item._magnet_velocity = 1.0
        emf_receding = item.current_induced_emf_volts

        # Direction of EMF should reverse due to Lenz's law (opposite signs)
        assert emf_approaching != 0.0
        assert emf_receding != 0.0
        assert (emf_approaching * emf_receding) < 0.0  # Opposite signs

        item.cleanup()

    def test_sliding_conductor_emf(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        item.load_preset("SLIDING_CONDUCTOR_RAIL")
        # E = B · l · v
        # B = 0.5 T, l = 0.15 m, v = 2.0 m/s
        item._elapsed_time_s = 0.0  # cos(0) = 1.0 -> v = 2.0 m/s
        emf = item.current_induced_emf_volts
        expected_emf = 0.50 * 0.15 * 2.0  # 0.15 V
        assert abs(emf - expected_emf) < 1e-3

        item.cleanup()


class TestInductionInteraction:
    """Verifies presets, air pinch gestures, parameter stepping, and vector stamping."""

    def test_all_five_presets_load(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        for code in ["BAR_MAGNET_SOLENOID", "ROTATING_AC_GENERATOR", "MUTUAL_INDUCTION_TRANSFORMER", "EDDY_CURRENT_PENDULUM", "SLIDING_CONDUCTOR_RAIL"]:
            ok = item.load_preset(code)
            assert ok is True
            assert item.active_preset_code == code
            assert item.preset is not None
            assert item.turns_n > 0
            assert item.b_field > 0.0
            assert item.speed_param > 0.0
        item.cleanup()

    def test_air_pinch_preset_switch(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        btn_r = item._btn_rects["preset_ROTATING_AC_GENERATOR"]
        pinch_pt = item.mapToScene(btn_r.center())

        handled = item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert item.active_preset_code == "ROTATING_AC_GENERATOR"

        item.cleanup()

    def test_air_pinch_steppers(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        v_init = item.speed_param
        n_init = item.turns_n
        b_init = item.b_field

        # Increase speed
        inc_v_pt = item.mapToScene(item._btn_rects["inc_speed"].center())
        assert item.handle_air_pinch(inc_v_pt) is True
        assert item.speed_param > v_init

        # Increase turns
        inc_n_pt = item.mapToScene(item._btn_rects["inc_turns"].center())
        assert item.handle_air_pinch(inc_n_pt) is True
        assert item.turns_n > n_init

        # Decrease field
        dec_b_pt = item.mapToScene(item._btn_rects["dec_field"].center())
        assert item.handle_air_pinch(dec_b_pt) is True
        assert item.b_field < b_init

        item.cleanup()

    def test_animation_tick_updates_oscilloscope(self, qapp):
        item = InteractiveElectromagneticInductionItem()
        item.load_preset("ROTATING_AC_GENERATOR")

        # Initial scope buffer
        last_emf_before = item._scope_emf[-1]
        item._on_animation_tick()
        assert item._elapsed_time_s > 0.0
        assert item._coil_angle_rad > 0.0

        # Scope buffer should have updated
        assert len(item._scope_emf) == 80
        assert len(item._scope_flux) == 80

        # Toggle pause
        is_anim = item.toggle_animation()
        assert is_anim is False
        assert item.is_animating is False

        t_paused = item._elapsed_time_s
        item._on_animation_tick()
        assert item._elapsed_time_s == t_paused  # No advance when paused

        item.cleanup()

    def test_stamp_creates_vector_diagram(self, qapp):
        board = LessonBoard()
        item = InteractiveElectromagneticInductionItem(pos=(200.0, 100.0))
        board.addItem(item)

        count_before = len(board.items())
        item.stamp_to_board()
        count_after = len(board.items())
        assert count_after > count_before

        item.cleanup()

    def test_physics_tools_vector_diagram(self, qapp):
        items = PhysicsTools.create_electromagnetic_induction_diagram(
            center=(400.0, 300.0),
            preset_code="BAR_MAGNET_SOLENOID",
            turns_n=600,
            b_field=0.20,
            speed_param=1.5,
        )
        assert len(items) >= 10
        texts = [itm.toPlainText() for itm in items if hasattr(itm, "toPlainText")]
        assert any("CẢM ỨNG ĐIỆN TỪ" in t.upper() for t in texts)
        assert any("FARADAY" in t.upper() or "TỪ THÔNG" in t.upper() for t in texts)


class TestInductionBoardIntegration:
    """Verifies LessonBoard lifecycle and lasso selection exclusion."""

    def test_board_lifecycle(self, qapp):
        board = LessonBoard()
        assert board.is_electromagnetic_induction_visible() is False
        assert board.get_electromagnetic_induction() is None

        # Show
        item = board.show_electromagnetic_induction(True)
        assert item is not None
        assert board.is_electromagnetic_induction_visible() is True
        assert board.get_electromagnetic_induction() is item
        assert board.get_electromagnetic_induction_item() is item

        # Toggle off
        vis = board.toggle_electromagnetic_induction()
        assert vis is False
        assert board.is_electromagnetic_induction_visible() is False

        # Toggle on
        vis = board.toggle_electromagnetic_induction()
        assert vis is True
        assert board.is_electromagnetic_induction_visible() is True

        item.cleanup()

    def test_lasso_exclusion(self, qapp):
        board = LessonBoard()
        item = board.show_electromagnetic_induction(True, pos=(250.0, 150.0))
        rect = QRectF(200.0, 100.0, 800.0, 600.0)

        # Simulator widget should be excluded from lasso selection
        selected_items = board.get_items_in_rect(rect)
        assert item not in selected_items

        item.cleanup()
