"""
AirTeach AI - Interactive Physics Magnetism & Lorentz Force Unit Tests
======================================================================
Verifies:
  - Lorentz Force Law F⃗_L = q(v⃗ x B⃗) and strict perpendicularity (F⃗_L ⊥ v⃗).
  - Kinetic energy and speed conservation (|v| = const, work W = 0).
  - Cyclotron / Larmor radius R = mv / (|q|B) and period T = 2πm / (|q|B).
  - Particle types: Electron (e⁻), Proton (p⁺), Alpha particle (He²⁺).
  - Magnetic field direction reversal (⊗ Into vs ⊙ Out).
  - Continuous trajectory integration via Boris velocity rotation.
  - Touchless air gestures (pinch drag, air tap buttons).
  - Vector diagram stamping onto LessonBoard.
  - LessonBoard integration and region exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_lorentz_force import (
    InteractiveLorentzForceItem,
    PARTICLE_SPECS,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveLorentzForce:
    """Test suite for InteractiveLorentzForceItem magnetism simulation."""

    def test_lorentz_init_and_properties(self, qapp):
        """Verifies initial configuration of Lorentz deflection simulation."""
        item = InteractiveLorentzForceItem(
            pos=(200.0, 150.0),
            particle_type="ELECTRON",
            b_field=1.0,
            velocity=160.0,
            b_direction="INTO",
        )

        assert item.pos() == QPointF(200.0, 150.0)
        assert item._particle_type == "ELECTRON"
        assert item._b_field == 1.0
        assert item._speed == 160.0
        assert item._b_direction == "INTO"
        assert item._is_animating is True

        # Cyclotron radius for Electron (m=1.0, q=-1.0, B=1.0, v=160.0)
        # R = (m * v) / (|q| * 0.016 * 100) = 160 / 1.6 = 100.0
        expected_r = 100.0
        assert abs(item.cyclotron_radius() - expected_r) < 1e-3

        # Period T = 2 * pi * m / (|q| * B) = 2 * pi * 1 / 1 = 6.283
        expected_t = 2.0 * math.pi
        assert abs(item.cyclotron_period() - expected_t) < 1e-3
        item.close()

    def test_lorentz_force_perpendicularity(self, qapp):
        """
        Verifies fundamental invariant of magnetic Lorentz force:
        F⃗_L is strictly perpendicular to velocity v⃗ (F⃗_L · v⃗ = 0).
        """
        item = InteractiveLorentzForceItem()

        fx, fy = item.lorentz_force_vector()
        vx = item._vel.x()
        vy = item._vel.y()

        dot_product = fx * vx + fy * vy
        assert abs(dot_product) < 1e-6
        item.close()

    def test_kinetic_energy_conservation(self, qapp):
        """
        Verifies speed remains strictly constant over simulation ticks,
        proving that the magnetic Lorentz force does zero mechanical work (W = 0).
        """
        item = InteractiveLorentzForceItem(velocity=180.0)
        initial_speed = math.hypot(item._vel.x(), item._vel.y())
        assert abs(initial_speed - 180.0) < 1e-4

        # Run 20 simulation integration ticks
        for _ in range(20):
            item._on_sim_tick()
            curr_speed = math.hypot(item._vel.x(), item._vel.y())
            assert abs(curr_speed - initial_speed) < 1e-4

        item.close()

    def test_particle_types_deflection_and_cyclotron_radius(self, qapp):
        """
        Verifies different particle types have distinct Cyclotron radii and
        opposite deflection signs (Proton bends opposite to Electron).
        """
        item = InteractiveLorentzForceItem(b_direction="INTO")

        # 1. Electron (q = -1, m = 1)
        item.set_particle_type("ELECTRON")
        r_electron = item.cyclotron_radius()
        fx_e, fy_e = item.lorentz_force_vector()

        # 2. Proton (q = +1, m = 6)
        item.set_particle_type("PROTON")
        r_proton = item.cyclotron_radius()
        fx_p, fy_p = item.lorentz_force_vector()

        # Radius of heavier proton must be larger (R_p = 6 * R_e)
        assert r_proton > r_electron
        assert abs(r_proton / r_electron - 6.0) < 1e-3

        # Force direction must be opposite due to opposite charge sign
        assert (fy_e * fy_p) < 0.0

        # 3. Alpha Particle (q = +2, m = 12)
        item.set_particle_type("ALPHA")
        r_alpha = item.cyclotron_radius()
        # R_alpha = 12 / (2 * ...) = 6 * R_e
        assert abs(r_alpha - r_proton) < 1e-3
        item.close()

    def test_b_field_direction_reversal(self, qapp):
        """
        Verifies reversing magnetic field direction (INTO -> OUT)
        inverts the Lorentz force direction.
        """
        item = InteractiveLorentzForceItem(b_direction="INTO")
        _, fy_into = item.lorentz_force_vector()

        dir_now = item.toggle_b_direction()
        assert dir_now == "OUT"
        _, fy_out = item.lorentz_force_vector()

        # Force must flip sign
        assert (fy_into * fy_out) < 0.0
        assert abs(fy_into + fy_out) < 1e-4
        item.close()

    def test_play_pause_and_reset(self, qapp):
        """Verifies play/pause state toggle and particle launch reset."""
        item = InteractiveLorentzForceItem()
        assert item._is_animating is True

        state = item.toggle_animation()
        assert state is False
        assert item._is_animating is False

        state = item.toggle_animation()
        assert state is True
        assert item._is_animating is True

        # Run some steps
        for _ in range(10):
            item._on_sim_tick()

        assert item._pos != item._launch_pos
        item.reset_particle()
        assert item._pos == item._launch_pos
        item.close()

    def test_touchless_air_gestures(self, qapp):
        """Verifies touchless air tap button dispatch and air drag."""
        board = LessonBoard()
        item = InteractiveLorentzForceItem(pos=(100.0, 100.0))
        board.addItem(item)

        # 1. Air tap Play toggle
        btn_play_scene = item.mapToScene(item._rect_btn_play.center())
        was_animating = item._is_animating
        consumed = item.handle_air_tap(btn_play_scene, board=board)
        assert consumed is True
        assert item._is_animating != was_animating

        # 2. Air tap B-direction toggle
        btn_bdir_scene = item.mapToScene(item._rect_btn_bdir.center())
        dir_before = item._b_direction
        consumed = item.handle_air_tap(btn_bdir_scene, board=board)
        assert consumed is True
        assert item._b_direction != dir_before

        # 3. Air tap Particle button (PROTON)
        btn_proton_scene = item.mapToScene(item._rect_btn_particles[1].center())
        consumed = item.handle_air_tap(btn_proton_scene, board=board)
        assert consumed is True
        assert item._particle_type == "PROTON"

        # 4. Air hover pos
        item.set_air_hover_pos(QPointF(200.0, 200.0))
        assert item._air_hover_pos is not None
        item.set_air_hover_pos(None)
        assert item._air_hover_pos is None

        # 5. Air tap Close button
        btn_close_scene = item.mapToScene(item._rect_btn_close.center())
        consumed = item.handle_air_tap(btn_close_scene, board=board)
        assert consumed is True
        assert item.scene() is None

    def test_stamp_to_canvas_and_lessonboard(self, qapp):
        """Verifies vector diagram stamping onto LessonBoard and region exclusion."""
        board = LessonBoard()
        item = InteractiveLorentzForceItem(pos=(200.0, 150.0))
        board.addItem(item)

        stamped = item.stamp_to_canvas(board)
        assert len(stamped) >= 15  # Chamber box, B-field symbols, particle, vectors, text HUD

        for itm in stamped:
            assert itm.scene() is board

        # Test LessonBoard lifecycle
        board.show_lorentz_force(True, pos=(300.0, 250.0), particle_type="ALPHA", b_direction="OUT")
        assert board.is_lorentz_force_visible() is True
        lf_item = board.get_lorentz_force_item()
        assert lf_item is not None
        assert lf_item.pos() == QPointF(300.0, 250.0)
        assert lf_item._particle_type == "ALPHA"
        assert lf_item._b_direction == "OUT"

        # Region exclusion test
        sel_rect = QRectF(250.0, 200.0, 400.0, 350.0)
        items_inside = board.get_items_in_rect(sel_rect)
        assert lf_item not in items_inside

        # Toggle visibility
        is_vis = board.toggle_lorentz_force()
        assert is_vis is False
        assert board.is_lorentz_force_visible() is False

        is_vis = board.toggle_lorentz_force()
        assert is_vis is True
        assert board.is_lorentz_force_visible() is True

        if hasattr(lf_item, "close"):
            lf_item.close()
        item.close()
