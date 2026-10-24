# tests/test_interactive_stress_strain.py
"""
Unit and integration tests for InteractiveStressStrainItem,
PhysicsTools.create_stress_strain_diagram, and LessonBoard integration.
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF

from src.ui.interactive_stress_strain import (
    InteractiveStressStrainItem,
    MaterialPreset,
    MATERIAL_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveStressStrain:
    def test_material_presets_physical_validity(self, qapp):
        """Verify all material presets conform to physical mechanics constraints."""
        assert len(MATERIAL_PRESETS) == 5
        assert "STRUCTURAL_STEEL_A36_DUCTILE" in MATERIAL_PRESETS
        assert "AEROSPACE_ALUMINUM_7075_T6" in MATERIAL_PRESETS
        assert "TITANIUM_ALLOY_TI_6AL_4V_GRADE5" in MATERIAL_PRESETS
        assert "GRAY_CAST_IRON_CLASS30_BRITTLE" in MATERIAL_PRESETS
        assert "COPPER_C11000_ELECTROLYTIC_TOUGH_PITCH" in MATERIAL_PRESETS

        for code, preset in MATERIAL_PRESETS.items():
            assert preset.youngs_modulus_GPa > 0.0
            assert preset.yield_strength_MPa > 0.0
            assert preset.uts_strength_MPa >= preset.yield_strength_MPa
            assert preset.fracture_strain_pct > 0.0
            assert 0.0 < preset.poissons_ratio < 0.5
            assert len(preset.title_vi) > 0
            assert preset.accent_color.isValid()

            # Yield strain must be less than UTS strain
            ey = preset.yield_strain_pct
            assert ey < preset.uts_strain_pct
            assert preset.uts_strain_pct <= preset.fracture_strain_pct

    def test_hookes_law_elastic_modulus(self, qapp):
        """Verify linear Hookean elasticity sigma = E * epsilon below yield point."""
        item = InteractiveStressStrainItem()
        item.set_material_preset("STRUCTURAL_STEEL_A36_DUCTILE")
        preset = item._preset

        # At half the yield strain:
        test_strain_pct = preset.yield_strain_pct * 0.5
        item.set_strain_pct(test_strain_pct)

        expected_stress = (test_strain_pct / 100.0) * (preset.youngs_modulus_GPa * 1000.0)
        assert math.isclose(item.current_stress_MPa, expected_stress, rel_tol=1e-3)
        assert not item.is_fractured
        assert "đàn hồi" in item.deformation_regime_name.lower()

        # Verify Modulus of Resilience Ur = sigma_y^2 / (2E)
        expected_ur = ((preset.yield_strength_MPa ** 2) / (2.0 * preset.youngs_modulus_GPa * 1000.0)) * 1000.0
        assert math.isclose(item.modulus_of_resilience_kJ_m3, expected_ur, rel_tol=1e-3)

    def test_plastic_hardening_and_necking(self, qapp):
        """Verify plastic strain hardening and necking softening up to fracture."""
        item = InteractiveStressStrainItem()
        item.set_material_preset("STRUCTURAL_STEEL_A36_DUCTILE")
        preset = item._preset

        # At UTS strain: stress should equal UTS strength
        item.set_strain_pct(preset.uts_strain_pct)
        assert math.isclose(item.current_stress_MPa, preset.uts_strength_MPa, rel_tol=1e-2)
        assert "Biến cứng" in item.deformation_regime_name or "Thắt" in item.deformation_regime_name

        # At Necking region between UTS and fracture:
        neck_strain = (preset.uts_strain_pct + preset.fracture_strain_pct) * 0.5
        item.set_strain_pct(neck_strain)
        assert item.current_stress_MPa < preset.uts_strength_MPa
        assert item.current_stress_MPa > preset.fracture_strength_MPa
        assert "Thắt" in item.deformation_regime_name

        # At Fracture strain:
        item.set_strain_pct(preset.fracture_strain_pct + 1.0)
        assert item.is_fractured
        assert item.current_stress_MPa == 0.0
        assert "gãy" in item.deformation_regime_name.lower()

    def test_brittle_material_mechanics(self, qapp):
        """Verify brittle Gray Cast Iron has minimal plastic necking before rupture."""
        item = InteractiveStressStrainItem()
        item.set_material_preset("GRAY_CAST_IRON_CLASS30_BRITTLE")
        preset = item._preset

        assert preset.fracture_strain_pct <= 1.5
        assert preset.uts_strain_pct == preset.fracture_strain_pct

        # Right before fracture
        item.set_strain_pct(preset.fracture_strain_pct * 0.95)
        assert not item.is_fractured

        # Beyond fracture
        item.set_strain_pct(preset.fracture_strain_pct + 0.1)
        assert item.is_fractured
        assert item.current_stress_MPa == 0.0

    def test_specimen_geometry_and_load_cell(self, qapp):
        """Verify tensile force F = sigma * A0 and elongation delta_L = epsilon * L0."""
        item = InteractiveStressStrainItem()
        item.set_material_preset("AEROSPACE_ALUMINUM_7075_T6")
        item.set_strain_pct(2.0)

        # Cross section area A0 = pi * r^2
        r = item.gauge_diameter_mm / 2.0
        a0 = math.pi * (r ** 2)
        expected_force_kN = (item.current_stress_MPa * a0) / 1000.0
        assert math.isclose(item.tensile_force_kN, expected_force_kN, rel_tol=1e-3)

        expected_dl = (2.0 / 100.0) * item.gauge_length_mm
        assert math.isclose(item.elongation_mm, expected_dl, rel_tol=1e-3)

    def test_interactive_controls_and_buttons(self, qapp):
        """Verify buttons: pull strain +1.0%, release -1.0%, cycle material, reset."""
        item = InteractiveStressStrainItem()
        item.set_material_preset("STRUCTURAL_STEEL_A36_DUCTILE")
        init_strain = item._current_strain_pct

        # Pull button
        item.step_strain(1.0)
        assert math.isclose(item._current_strain_pct, init_strain + 1.0, rel_tol=1e-3)

        # Release button
        item.step_strain(-0.5)
        assert math.isclose(item._current_strain_pct, init_strain + 0.5, rel_tol=1e-3)

        # Cycle preset
        orig_code = item._active_preset_code
        item.cycle_preset()
        assert item._active_preset_code != orig_code

        # Reset
        item.reset_test()
        assert item._current_strain_pct == 0.0
        assert item.current_stress_MPa == 0.0
        assert not item.is_fractured

    def test_stamping_and_vector_diagram(self, qapp):
        """Verify stamp request produces complete vector diagram items."""
        item = InteractiveStressStrainItem()
        stamped_data = {}

        def on_stamp(data):
            stamped_data.update(data)

        item.stamp_requested.connect(on_stamp)
        item.request_stamp()

        assert "preset_code" in stamped_data
        assert "material_name" in stamped_data
        assert "youngs_modulus_GPa" in stamped_data
        assert "yield_strength_MPa" in stamped_data

        # Generate diagram via PhysicsTools
        items = PhysicsTools.create_stress_strain_diagram(
            center=(400.0, 300.0),
            preset_code=stamped_data["preset_code"],
            material_name=stamped_data["material_name"],
            youngs_modulus_GPa=stamped_data["youngs_modulus_GPa"],
            yield_strength_MPa=stamped_data["yield_strength_MPa"],
            uts_strength_MPa=stamped_data["uts_strength_MPa"],
            current_strain_pct=stamped_data["current_strain_pct"],
            current_stress_MPa=stamped_data["current_stress_MPa"],
            tensile_force_kN=stamped_data["tensile_force_kN"],
            elongation_mm=stamped_data["elongation_mm"],
            is_fractured=stamped_data["is_fractured"],
            regime=stamped_data["regime"],
        )

        assert len(items) >= 20
        # Check title text presence
        has_title = any(
            getattr(itm, "toPlainText", lambda: "")().startswith("⚙ ĐƯỜNG CONG ỨNG SUẤT")
            for itm in items
        )
        assert has_title

    def test_lesson_board_integration(self, qapp):
        """Verify LessonBoard visibility toggling and stamping lifecycle."""
        board = LessonBoard()

        assert not board.is_stress_strain_visible()
        item = board.show_stress_strain(True, pos=(150.0, 80.0))
        assert item is not None
        assert board.is_stress_strain_visible()
        assert board.get_stress_strain() is item

        # Toggle visibility
        is_vis = board.toggle_stress_strain()
        assert not is_vis
        assert not board.is_stress_strain_visible()

        # Re-show
        is_vis = board.toggle_stress_strain()
        assert is_vis
        assert board.is_stress_strain_visible()

        # Stamp onto board
        initial_item_count = len(board.items())
        item.request_stamp()
        assert len(board.items()) > initial_item_count
