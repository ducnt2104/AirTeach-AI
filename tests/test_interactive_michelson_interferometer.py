"""
Tests for Interactive Michelson Interferometer & LIGO Gravitational Wave Detector Simulator.
Verifies authentic physics, fringe counting, coherence decay, sodium doublet beats,
wedge spacing, LIGO quadrupole strain, gestures, rendering, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_michelson_interferometer import (
    PRESETS,
    SOURCES,
    InteractiveMichelsonInterferometerItem,
    InterferometerMode,
    LightSource,
    MichelsonPreset,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestMichelsonPhysics:
    """Authentic mathematical and physical optics verification."""

    def test_optical_path_difference_and_fringe_counting(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("HE_NE_CIRCULAR")

        # Laser He-Ne λ = 632.8 nm = 0.6328 μm
        assert item.wavelength_nm == 632.8
        assert item.delta_d_um == 10.0

        # Optical path difference Δ = 2 * Δd = 20.0 μm
        assert item.compute_optical_path_difference_um() == 20.0

        # Fringe count ΔN = 2 * Δd / λ = 20.0 / 0.6328 ≈ 31.6055
        n_fringe = item.compute_fringe_shift_count()
        assert math.isclose(n_fringe, 20.0 / 0.6328, rel_tol=1e-4)

        # Move mirror by half-wavelength (0.3164 μm) => exactly 1 fringe shift
        item.step_mirror_position(0.3164)
        n_fringe_after = item.compute_fringe_shift_count()
        assert math.isclose(n_fringe_after - n_fringe, 1.0, rel_tol=1e-3)

    def test_sodium_doublet_beats_phenomenon(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("SODIUM_DOUBLET_BEATS")

        source = item._source
        assert source.wavelength_nm == 589.0
        assert source.secondary_wavelength_nm == 589.59

        # Sodium doublet beat formula: V(d) = |cos(2π * Δλ * d / λ̄²)|
        # Destructive node where fringes vanish at d ≈ 147.1 μm
        item._delta_d_um = 147.12
        v_min = item.compute_visibility()
        assert v_min < 0.05  # Almost zero contrast (destructive beat node)

        # Constructive beat peak at d ≈ 294.24 μm
        item._delta_d_um = 294.24
        v_max = item.compute_visibility()
        assert v_max > 0.85  # Restored high contrast

    def test_white_light_coherence_decay(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("WHITE_LIGHT_ZERO_PATH")

        # Broadband white light source (Δλ = 300 nm)
        source = item._source
        assert source.spectral_width_nm == 300.0
        assert math.isclose(source.coherence_length_um, (0.55 ** 2) / 0.3, rel_tol=1e-2)

        # At zero OPD (Δd = 0): Maximum visibility
        item.reset_zero_path()
        assert item.compute_visibility() == 1.0

        # Displacing mirror by 5 μm (>> L_c ≈ 1 μm) causes complete coherence loss
        item._delta_d_um = 5.0
        v_lost = item.compute_visibility()
        assert v_lost < 1e-4

    def test_wedge_fringe_spacing(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("WEDGE_LINEAR_FRINGES")

        # Fringe spacing: i = λ / (2 * α)
        spacing_mm = item.compute_wedge_fringe_spacing_mm()
        alpha_rad = math.radians(item.tilt_angle_deg)
        expected_mm = (item.wavelength_nm * 1e-6) / (2.0 * alpha_rad)
        assert math.isclose(spacing_mm, expected_mm, rel_tol=1e-4)

    def test_ligo_gravitational_wave_chirp_strain(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("LIGO_GRAVITATIONAL_CHIRP")

        # Check binary chirp strain at various points
        h_t, delta_L, p_det = item.compute_ligo_strain(0.5)
        assert abs(h_t) <= 1.5e-21
        # Physical arm deformation: ΔL = L * h ≈ 4000 * 10^-21 ≈ 4 * 10^-18 m
        assert math.isclose(delta_L, 4000.0 * h_t, rel_tol=1e-5)
        # Power between 0 and 100%
        assert 0.0 <= p_det <= 100.0


class TestMichelsonPresetsAndControls:
    """Verifies presets, modes, steppers, and air gestures."""

    def test_all_presets_load(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            assert item.active_preset.code == code
            assert item.mode in [InterferometerMode.CIRCULAR, InterferometerMode.WEDGE, InterferometerMode.LIGO]

    def test_parameter_steppers_and_modes(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("HE_NE_CIRCULAR")

        init_d = item.delta_d_um
        item.step_mirror_position(5.0)
        assert math.isclose(item.delta_d_um, init_d + 5.0)

        init_lam = item.wavelength_nm
        item.step_wavelength(20.0)
        assert math.isclose(item.wavelength_nm, init_lam + 20.0)

        # Mode switching
        item.set_mode(InterferometerMode.WEDGE)
        assert item.mode == InterferometerMode.WEDGE
        item.set_mode(InterferometerMode.LIGO)
        assert item.mode == InterferometerMode.LIGO

        # Animation toggle
        assert item._is_running is True
        item.toggle_animation()
        assert item._is_running is False
        item.toggle_animation()
        assert item._is_running is True

    def test_air_pinch_button_interactions(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        item.load_preset("HE_NE_CIRCULAR")

        # Paint once to populate buttons dict
        img = QImage(960, 650, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()

        assert "btn_d_plus_1" in item._buttons
        plus_rect = item._buttons["btn_d_plus_1"]
        target_scene_pt = item.mapToScene(plus_rect.center())

        initial_d = item.delta_d_um
        handled = item.handle_air_pinch(target_scene_pt)
        assert handled is True
        assert math.isclose(item.delta_d_um, initial_d + 1.0)

        # Test stamp button
        stamp_handled = item.handle_air_pinch(item.mapToScene(item._buttons["btn_stamp"].center()))
        assert stamp_handled is True

    def test_paint_rendering_all_modes(self, qapp):
        item = InteractiveMichelsonInterferometerItem()
        img = QImage(960, 650, QImage.Format.Format_ARGB32)

        for code in PRESETS.keys():
            item.load_preset(code)
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle."""

    def test_physics_tools_michelson_diagram(self, qapp):
        items = PhysicsTools.create_michelson_interferometer_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="HE_NE_CIRCULAR",
            mode="circular",
            source_name="Laser He-Ne",
            wavelength_nm=632.8,
            delta_d_um=10.0,
            visibility_pct=100.0,
            fringe_count=31.6,
        )
        assert len(items) > 10
        assert any("GIAO THOA KẾ MICHELSON" in getattr(item, "toPlainText", lambda: "")() for item in items)

    def test_lesson_board_michelson_lifecycle(self, qapp):
        board = LessonBoard()

        # Initially not visible
        assert board.is_michelson_interferometer_visible() is False
        assert board.get_michelson_interferometer() is None

        # Show
        item = board.show_michelson_interferometer(True)
        assert item is not None
        assert board.is_michelson_interferometer_visible() is True
        assert board.get_michelson_interferometer() is item

        # Toggle hide
        new_state = board.toggle_michelson_interferometer()
        assert new_state is False
        assert board.is_michelson_interferometer_visible() is False

        # Toggle show
        new_state = board.toggle_michelson_interferometer()
        assert new_state is True
        assert board.is_michelson_interferometer_visible() is True

        # Ensure item is excluded from selection box in get_items_in_rect
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        selected_items = board.get_items_in_rect(selection_rect)
        assert item not in selected_items
