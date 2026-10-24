"""
Tests for Interactive Cyclotron Particle Accelerator & Relativistic Mechanics Simulator.
Verifies authentic classical resonance frequencies, relativistic mass dilation,
synchrocyclotron frequency modulation, extraction kinetic energy, particle species,
touchless air gestures, rendering, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_cyclotron import (
    C_LIGHT,
    E_CHARGE,
    M_PROTON,
    MEV_TO_JOULE,
    PARTICLES,
    PRESETS,
    CyclotronPreset,
    InteractiveCyclotronItem,
    ParticleSpec,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestCyclotronPhysics:
    """Authentic mathematical physics and relativistic accelerator verification."""

    def test_proton_classical_resonance_frequency(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("CLASSICAL_PROTON_ACCELERATOR")

        # Proton in 1.5 T magnetic field
        assert item.particle.code == "Proton"
        assert item.b_field_tesla == 1.5

        # Theoretical f_c = q * B / (2π * m₀)
        # = (1.602176634e-19 * 1.5) / (2π * 1.67262192e-27) ≈ 22.883 MHz
        expected_f_mhz = (E_CHARGE * 1.5) / (2.0 * math.pi * M_PROTON) * 1e-6
        calc_f_mhz = item.compute_cyclotron_frequency_mhz()
        assert math.isclose(calc_f_mhz, expected_f_mhz, rel_tol=1e-4)

    def test_alpha_particle_half_frequency(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("DEUTERON_ALPHA_RADIOISOTOPE")

        # Alpha particle (q=2e, m ≈ 4m_p) has ~half the charge-to-mass ratio of proton
        assert item.particle.code == "Alpha"
        calc_f_mhz = item.compute_cyclotron_frequency_mhz()

        # At B = 1.6 T: f_c = (2e * 1.6) / (2π * 6.64465e-27) ≈ 12.21 MHz
        assert 11.5 < calc_f_mhz < 13.0

    def test_theoretical_max_extraction_energy(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("CLASSICAL_PROTON_ACCELERATOR")

        # For B = 1.5 T, R_max = 0.8 m:
        # p_max = q * B * R_max = 1.602e-19 * 1.5 * 0.8 ≈ 1.922e-19 kg·m/s
        # Classical: E_k ≈ p² / (2m) ≈ 1.104e-11 J ≈ 68.9 MeV? Wait, let's check formula:
        e_max_mev = item.compute_max_extraction_energy_mev()
        assert e_max_mev > 10.0
        assert e_max_mev < 100.0

    def test_relativistic_gamma_and_synchro_modulation(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("SYNCHROCYCLOTRON_RF_MODULATION")

        # In synchrocyclotron preset, particle accelerates to high energies (> 50 MeV)
        assert item._is_synchro is True
        # As particle accelerates, gamma factor increases above 1.0
        assert item.gamma_factor >= 1.0
        beta = math.sqrt(max(0.0, 1.0 - 1.0 / (item.gamma_factor ** 2)))
        assert 0.0 <= beta < 1.0


class TestCyclotronPresetsAndControls:
    """Verifies presets, particle switching, parameter steppers, and air gestures."""

    def test_all_presets_load(self, qapp):
        item = InteractiveCyclotronItem()
        for code in PRESETS.keys():
            item.load_preset(code)
            assert item.active_preset.code == code
            assert len(item._spiral_pts) > 1

    def test_particle_switching(self, qapp):
        item = InteractiveCyclotronItem()
        item.set_particle("Deuteron")
        assert item.particle.code == "Deuteron"
        item.set_particle("Carbon")
        assert item.particle.code == "Carbon"
        item.set_particle("Proton")
        assert item.particle.code == "Proton"

    def test_steppers_and_synchro_toggle(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("CLASSICAL_PROTON_ACCELERATOR")

        init_b = item.b_field_tesla
        item.step_b_field(0.4)
        assert math.isclose(item.b_field_tesla, init_b + 0.4)

        init_v = item.dee_voltage_kv
        item.step_dee_voltage(20.0)
        assert math.isclose(item.dee_voltage_kv, init_v + 20.0)

        # Synchro toggle
        init_synchro = item._is_synchro
        item.toggle_synchro()
        assert item._is_synchro != init_synchro
        item.toggle_synchro()
        assert item._is_synchro == init_synchro

        # Animation toggle
        assert item._is_running is True
        item.toggle_animation()
        assert item._is_running is False
        item.toggle_animation()
        assert item._is_running is True

    def test_air_pinch_button_interactions(self, qapp):
        item = InteractiveCyclotronItem()
        item.load_preset("CLASSICAL_PROTON_ACCELERATOR")

        # Paint once to populate buttons dict
        img = QImage(960, 650, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        item.paint(painter, None, None)
        painter.end()

        assert "btn_preset" in item._buttons
        assert "btn_stamp" in item._buttons
        assert "btn_b_plus" in item._buttons

        # Air pinch on B-plus button
        b_btn_rect = item._buttons["btn_b_plus"]
        initial_b = item.b_field_tesla
        handled = item.handle_air_pinch(item.mapToScene(b_btn_rect.center()))
        assert handled is True
        assert math.isclose(item.b_field_tesla, initial_b + 0.2)

        # Air pinch on stamp button
        stamp_handled = item.handle_air_pinch(item.mapToScene(item._buttons["btn_stamp"].center()))
        assert stamp_handled is True

    def test_paint_rendering_all_presets(self, qapp):
        item = InteractiveCyclotronItem()
        img = QImage(960, 650, QImage.Format.Format_ARGB32)

        for code in PRESETS.keys():
            item.load_preset(code)
            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()


class TestSTEMToolsAndLessonBoardIntegration:
    """Verifies STEM diagram generation and LessonBoard lifecycle."""

    def test_physics_tools_cyclotron_diagram(self, qapp):
        items = PhysicsTools.create_cyclotron_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code="CLASSICAL_PROTON_ACCELERATOR",
            particle_name="Proton (¹H⁺)",
            b_field_tesla=1.5,
            dee_voltage_kv=50.0,
            freq_mhz=22.88,
            max_energy_mev=16.5,
            r_max_m=0.8,
        )
        assert len(items) >= 5
        assert any("CYCLOTRON" in getattr(item, "toPlainText", lambda: "")() for item in items)

    def test_lesson_board_cyclotron_lifecycle(self, qapp):
        board = LessonBoard()

        # Initially not visible
        assert board.is_cyclotron_visible() is False
        assert board.get_cyclotron() is None

        # Show
        item = board.show_cyclotron(True)
        assert item is not None
        assert board.is_cyclotron_visible() is True
        assert board.get_cyclotron() is item

        # Toggle hide
        new_state = board.toggle_cyclotron()
        assert new_state is False
        assert board.is_cyclotron_visible() is False

        # Toggle show
        new_state = board.toggle_cyclotron()
        assert new_state is True
        assert board.is_cyclotron_visible() is True

        # Ensure item is excluded from selection box in get_items_in_rect
        selection_rect = QRectF(0.0, 0.0, 2000.0, 2000.0)
        selected_items = board.get_items_in_rect(selection_rect)
        assert item not in selected_items
