"""
AirTeach AI – Test Suite: Interactive Astrophysics & Orbital Mechanics (Hohmann Transfer)
==========================================================================================
Verifies:
  - Exact Vis-Viva equation: v² = μ(2/r - 1/a)
  - Circular orbital velocities v_circ = sqrt(μ / r)
  - Transfer orbit semi-major axis a = (r₁ + r₂) / 2 and eccentricity e = |r₂ - r₁| / (r₁ + r₂)
  - Impulsive delta-V burns Δv₁ and Δv₂
  - One-way transfer duration t = π * sqrt(a³ / μ)
  - Escape velocity v_esc = sqrt(2 * μ / r)
  - All 5 curriculum presets
  - Air gestures, steppers, and rendering
  - STEMTools vector diagram generation
  - LessonBoard integration & lifecycle
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QApplication

from src.ui.interactive_hohmann_transfer import (
    InteractiveHohmannTransferItem,
    PRESETS,
    MU_EARTH,
    MU_SUN,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestAstrodynamicsMathematics:
    """Verifies Keplerian orbits, Vis-Viva equation, and Hohmann transfer budgets without mock data."""

    def test_leo_to_geo_exact_budgets(self, qapp):
        item = InteractiveHohmannTransferItem()
        item.load_preset("LEO_TO_GEO_TRANSFER")

        # LEO: r1 = 6678 km = 6.678e6 m
        # GEO: r2 = 42164 km = 4.2164e7 m
        # v_c1 = sqrt(MU_EARTH / r1) ≈ 7726 m/s (7.726 km/s)
        assert 7720.0 < item.v_circ_1 < 7730.0

        # v_c2 = sqrt(MU_EARTH / r2) ≈ 3075 m/s (3.075 km/s)
        assert 3070.0 < item.v_circ_2 < 3080.0

        # a_trans = (r1 + r2) / 2 = 24421 km
        assert math.isclose(item.a_trans_km, 24421.0, rel_tol=1e-4)

        # e_trans = (r2 - r1) / (r1 + r2) ≈ 0.7265
        assert 0.72 < item.eccentricity_trans < 0.73

        # First burn Delta v1 ≈ 2.425 km/s (2425 m/s)
        assert 2420.0 < item.delta_v_1 < 2435.0

        # Second burn Delta v2 ≈ 1.465 km/s (1465 m/s)
        assert 1460.0 < item.delta_v_2 < 1475.0

        # Total Delta v ≈ 3.89 km/s
        assert 3880.0 < item.delta_v_total < 3900.0

        # Transfer time t = pi * sqrt(a^3 / mu) ≈ 18950 s ≈ 5.26 hours
        assert 5.20 < item.transfer_time_hours < 5.35

        item.cleanup()

    def test_earth_to_mars_heliocentric_transfer(self, qapp):
        item = InteractiveHohmannTransferItem()
        item.load_preset("EARTH_TO_MARS_INTERPLANETARY")

        assert item.mu == MU_SUN
        # Earth r1 = 1.496e8 km, Mars r2 = 2.279e8 km
        # a_trans ≈ 1.8875e8 km
        assert 1.88e8 < item.a_trans_km < 1.90e8

        # Heliocentric transfer time ≈ 259 days (~8.5 months)
        assert 250.0 < item.transfer_time_days < 270.0

        item.cleanup()

    def test_escape_velocity_relation(self, qapp):
        item = InteractiveHohmannTransferItem()
        item.load_preset("LEO_TO_GEO_TRANSFER")

        # v_esc = sqrt(2) * v_circ
        expected_esc = math.sqrt(2.0) * item.v_circ_1
        assert math.isclose(item.v_escape_1, expected_esc, rel_tol=1e-5)
        # In LEO, v_esc ≈ 10.92 km/s
        assert 10.8 < (item.v_escape_1 / 1000.0) < 11.1

        item.cleanup()

    def test_vis_viva_conservation_along_ellipse(self, qapp):
        item = InteractiveHohmannTransferItem()
        item.load_preset("LEO_TO_GEO_TRANSFER")

        # Specific orbital energy epsilon = -mu / (2*a)
        epsilon_expected = -item.mu / (2.0 * item.a_trans_m)

        # Check at theta = 0 (periapsis r = r1)
        r_peri = item.current_radius_m(0.0)
        assert math.isclose(r_peri, item.r1_m, rel_tol=1e-5)
        v_peri = item.current_velocity_m(r_peri)
        eps_peri = (v_peri ** 2) / 2.0 - item.mu / r_peri
        assert math.isclose(eps_peri, epsilon_expected, rel_tol=1e-4)

        # Check at theta = pi (apoapsis r = r2)
        r_apo = item.current_radius_m(math.pi)
        assert math.isclose(r_apo, item.r2_m, rel_tol=1e-5)
        v_apo = item.current_velocity_m(r_apo)
        eps_apo = (v_apo ** 2) / 2.0 - item.mu / r_apo
        assert math.isclose(eps_apo, epsilon_expected, rel_tol=1e-4)

        # Velocity at periapsis must be strictly greater than at apoapsis (Kepler's 2nd Law)
        assert v_peri > v_apo

        item.cleanup()


class TestOrbitalPresetsAndControls:
    """Verifies presets, flight phase, simulation speed, and touchless air gestures."""

    def test_all_presets_load(self, qapp):
        item = InteractiveHohmannTransferItem()

        for code in PRESETS:
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.boundingRect().width() == 700.0
            assert item.boundingRect().height() == 440.0
        item.cleanup()

    def test_flight_phase_and_speed_controls(self, qapp):
        item = InteractiveHohmannTransferItem()

        # Phases clamped to [0, 2]
        item.set_flight_phase(0)
        assert item.flight_phase == 0
        item.set_flight_phase(2)
        assert item.flight_phase == 2
        item.set_flight_phase(5)
        assert item.flight_phase == 2

        # Speed cycling: 1 -> 2 -> 5 -> 1
        assert item._sim_speed == 1.0
        item.cycle_sim_speed()
        assert item._sim_speed == 2.0
        item.cycle_sim_speed()
        assert item._sim_speed == 5.0
        item.cycle_sim_speed()
        assert item._sim_speed == 1.0

        # Animation toggle
        assert item.is_animating
        item.toggle_animation()
        assert not item.is_animating
        item.toggle_animation()
        assert item.is_animating

        item.cleanup()

    def test_air_pinch_interactions(self, qapp):
        item = InteractiveHohmannTransferItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        # 1. Preset pinch
        p_rect = item._btn_rects.get("preset_EARTH_TO_MOON_TRANSLUNAR")
        assert p_rect is not None
        center = item.mapToScene(p_rect.center())
        assert item.handle_air_pinch(center) is True
        assert item.active_preset.code == "EARTH_TO_MOON_TRANSLUNAR"

        # 2. Toggle animation pinch
        anim_rect = item._btn_rects.get("toggle_anim")
        assert anim_rect is not None
        center_anim = item.mapToScene(anim_rect.center())
        was_animating = item.is_animating
        assert item.handle_air_pinch(center_anim) is True
        assert item.is_animating != was_animating

        # 3. Outside pinch
        outside = item.mapToScene(QPointF(9999.0, 9999.0))
        assert item.handle_air_pinch(outside) is False

        item.cleanup()

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveHohmannTransferItem()
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()

        for code in PRESETS:
            item.load_preset(code)
            painter = QPainter(image)
            item.paint(painter, opt)
            painter.end()

        item.cleanup()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle integration."""

    def test_stem_tools_diagram_generation(self, qapp):
        items = PhysicsTools.create_hohmann_transfer_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="LEO_TO_GEO_TRANSFER",
            r1_km=6678.0,
            r2_km=42164.0,
            delta_v1_kms=2.425,
            delta_v2_kms=1.465,
            transfer_hours=5.26,
        )

        assert len(items) >= 5
        assert items[0].boundingRect().width() >= 640.0
        assert items[0].boundingRect().height() >= 340.0

    def test_lesson_board_hohmann_lifecycle(self, qapp):
        board = LessonBoard()

        assert not board.is_hohmann_transfer_visible()
        assert board.get_hohmann_transfer() is None
        assert board.get_hohmann_transfer_item() is None

        # Toggle ON
        res = board.toggle_hohmann_transfer()
        assert res is True
        assert board.is_hohmann_transfer_visible()
        hohmann_item = board.get_hohmann_transfer()
        assert hohmann_item is not None
        assert board.get_hohmann_transfer_item() is hohmann_item

        # Toggle OFF
        res = board.toggle_hohmann_transfer()
        assert res is False
        assert not board.is_hohmann_transfer_visible()
        assert board.get_hohmann_transfer() is None
        assert board.get_hohmann_transfer_item() is hohmann_item

        # Exclusion from get_items_in_rect
        board.show_hohmann_transfer(True, pos=(100.0, 100.0))
        items_in_rect = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert hohmann_item not in items_in_rect

        hohmann_item.cleanup()
