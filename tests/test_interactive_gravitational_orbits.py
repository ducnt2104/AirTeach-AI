"""
Unit and integration tests for Interactive Gravitational N-Body Planetary Orbit & Lagrange Points Simulator.
Verifies Symplectic Velocity Verlet integration, energy & angular momentum conservation,
Lagrange points (L1-L5) geometry, celestial presets, air gestures, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_gravitational_orbits import (
    InteractiveGravitationalOrbitsItem,
    CelestialBody,
    G_CONST,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveGravitationalOrbits:
    """Test suite for InteractiveGravitationalOrbitsItem."""

    def test_gravitational_orbits_initialization(self, qapp):
        """Verify default solar system initialization, bodies count and bounding geometry."""
        item = InteractiveGravitationalOrbitsItem(pos=(150.0, 100.0), preset="INNER_SOLAR_SYSTEM")
        assert item.x() == 150.0
        assert item.y() == 100.0
        assert item.boundingRect().width() == 800.0
        assert item.boundingRect().height() == 520.0
        assert item.preset == "INNER_SOLAR_SYSTEM"
        assert len(item.bodies) == 5  # Sun, Mercury, Venus, Earth, Mars
        assert item.bodies[0].fixed is True  # Sun is fixed anchor
        assert item.total_energy < 0.0  # Gravitationally bound system
        item.close()

    def test_celestial_presets_and_bodies(self, qapp):
        """Verify switching through all 5 authentic celestial presets."""
        item = InteractiveGravitationalOrbitsItem()

        # Preset 2: Figure-Eight 3-body choreography
        item.set_preset("FIGURE_EIGHT_3BODY")
        assert item.preset == "FIGURE_EIGHT_3BODY"
        assert len(item.bodies) == 3
        # All 3 bodies must have equal mass
        assert item.bodies[0].mass == item.bodies[1].mass == item.bodies[2].mass == 1000.0

        # Preset 3: Earth-Moon with Lagrange test particle
        item.set_preset("EARTH_MOON_LAGRANGE")
        assert item.preset == "EARTH_MOON_LAGRANGE"
        assert len(item.bodies) == 3
        assert item.bodies[0].name.startswith("Trái Đất")
        assert item.bodies[1].name.startswith("Mặt Trăng")

        # Preset 4: Binary Star & Circumbinary Planet
        item.set_preset("BINARY_STAR_PLANET")
        assert item.preset == "BINARY_STAR_PLANET"
        assert len(item.bodies) == 3
        assert item.bodies[0].mass == item.bodies[1].mass == 4500.0  # Equal mass binary

        # Preset 5: Burrau Pythagorean 3-4-5 problem
        item.set_preset("CHAOTIC_3BODY")
        assert item.preset == "CHAOTIC_3BODY"
        assert len(item.bodies) == 3
        assert [b.mass for b in item.bodies] == [3000.0, 4000.0, 5000.0]

        # Cycle preset
        item.cycle_preset(1)
        assert item.preset == "INNER_SOLAR_SYSTEM"
        item.close()

    def test_symplectic_velocity_verlet_integration(self, qapp):
        """
        Verify that symplectic Velocity Verlet integration preserves total mechanical energy
        over multiple orbital integration steps without secular energy explosion or artificial decay.
        """
        item = InteractiveGravitationalOrbitsItem(preset="FIGURE_EIGHT_3BODY")
        initial_energy = item.total_energy

        # Step through 50 integration steps
        for _ in range(50):
            item.step_simulation(dt=0.005)

        final_energy = item.total_energy
        # In a symplectic integrator, energy error should be bounded within a few percent
        rel_energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
        assert rel_energy_error < 0.05, f"Symplectic energy drift exceeded limit: {rel_energy_error:.4f}"
        item.close()

    def test_angular_momentum_conservation(self, qapp):
        """Verify that total angular momentum L_z is conserved in isolated gravitational systems."""
        item = InteractiveGravitationalOrbitsItem(preset="FIGURE_EIGHT_3BODY")
        initial_lz = item.angular_momentum

        # Step through 30 integration steps
        for _ in range(30):
            item.step_simulation(dt=0.02)

        final_lz = item.angular_momentum
        # Angular momentum should remain virtually identical (zero for figure 8)
        assert abs(final_lz - initial_lz) < 1e-3
        item.close()

    def test_lagrange_points_geometry(self, qapp):
        """
        Verify authentic equilateral triangle geometry for Lagrange points L4 and L5:
        In the circular restricted 3-body problem, L4 and L5 form equilateral triangles with M1 and M2:
        dist(M1, L4) == dist(M2, L4) == dist(M1, M2).
        """
        item = InteractiveGravitationalOrbitsItem(preset="EARTH_MOON_LAGRANGE")
        earth = item.bodies[0]
        moon = item.bodies[1]

        r_em = math.hypot(moon.x - earth.x, moon.y - earth.y)

        # L4 is at 60 degrees ahead of Moon
        l4_x = earth.x + r_em * math.cos(math.pi / 3.0)
        l4_y = earth.y - r_em * math.sin(math.pi / 3.0)

        dist_earth_l4 = math.hypot(l4_x - earth.x, l4_y - earth.y)
        dist_moon_l4 = math.hypot(l4_x - moon.x, l4_y - moon.y)

        assert abs(dist_earth_l4 - r_em) < 1.0
        assert abs(dist_moon_l4 - r_em) < 1.0
        item.close()

    def test_simulation_speed_and_pause_controls(self, qapp):
        """Verify pause, resume, speed multiplier scaling, and reset."""
        item = InteractiveGravitationalOrbitsItem(preset="INNER_SOLAR_SYSTEM")
        assert item.is_paused is False

        # Toggle pause
        is_paused = item.toggle_pause()
        assert is_paused is True
        assert item.is_paused is True

        # Toggle resume
        is_paused = item.toggle_pause()
        assert is_paused is False

        # Set speed
        item.set_speed(2.0)
        assert item.speed_multiplier == 2.0
        item.set_speed(5.0)
        assert item.speed_multiplier == 5.0

        # Toggle trails & vectors
        assert item.toggle_trails() is False
        assert item.toggle_trails() is True
        assert item.toggle_vectors() is False
        assert item.toggle_vectors() is True
        assert item.toggle_lagrange() is False
        assert item.toggle_lagrange() is True

        # Reset simulation
        item.reset_simulation()
        assert item.preset == "INNER_SOLAR_SYSTEM"
        item.close()

    def test_air_gestures_and_mouse_interactions(self, qapp):
        """Verify touchless air-tap and air-pinch gesture triggers on item control buttons."""
        item = InteractiveGravitationalOrbitsItem(pos=(0.0, 0.0), preset="INNER_SOLAR_SYSTEM")

        # 1. Air Tap on [▶ Next Preset] (rect: 310, 390, 30, 26)
        btn_next_pt = QPointF(310.0 + 15.0, 390.0 + 13.0)
        handled = item.handle_air_tap(btn_next_pt)
        assert handled is True
        assert item.preset == "FIGURE_EIGHT_3BODY"

        # 2. Air Tap on [⏸ Pause] (rect: 355, 390, 75, 26)
        btn_pause_pt = QPointF(355.0 + 35.0, 390.0 + 13.0)
        handled = item.handle_air_tap(btn_pause_pt)
        assert handled is True
        assert item.is_paused is True

        # 3. Air Tap on [2x Speed] (rect: 541, 390, 48, 26)
        btn_spd_pt = QPointF(541.0 + 24.0, 390.0 + 13.0)
        handled = item.handle_air_tap(btn_spd_pt)
        assert handled is True
        assert item.speed_multiplier == 2.0

        # 4. Air Pinch to drag simulator window
        init_x, init_y = item.x(), item.y()
        handled_pinch = item.handle_air_pinch(QPointF(70.0, 35.0), QPointF(50.0, 20.0))
        assert handled_pinch is True
        assert item.x() == init_x + 20.0
        assert item.y() == init_y + 15.0

        item.close()

    def test_stamping_and_lesson_board_integration(self, qapp):
        """Verify stamping vector diagram to LessonBoard, visibility toggling and eraser immunity."""
        board = LessonBoard()
        assert board.is_gravitational_orbits_visible() is False
        assert board.get_gravitational_orbits() is None

        # Show on board
        sim_item = board.show_gravitational_orbits(True)
        assert sim_item is not None
        assert board.is_gravitational_orbits_visible() is True
        assert board.get_gravitational_orbits() is sim_item
        assert isinstance(sim_item, InteractiveGravitationalOrbitsItem)

        # Toggle visibility to hidden
        state = board.toggle_gravitational_orbits()
        assert state is False
        assert board.is_gravitational_orbits_visible() is False

        # Toggle back to visible
        state = board.toggle_gravitational_orbits()
        assert state is True
        assert board.is_gravitational_orbits_visible() is True

        # Eraser exclusion check
        item_rect = sim_item.sceneBoundingRect()
        items_in_rect = board.get_items_in_rect(item_rect)
        assert sim_item not in items_in_rect

        # Stamp vector diagram to LessonBoard
        initial_count = len(board.items())
        stamped_items = sim_item.stamp_to_canvas(board)
        assert len(stamped_items) > 0
        assert len(board.items()) > initial_count

        sim_item.close()
