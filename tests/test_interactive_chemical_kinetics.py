"""
Unit tests for Interactive Chemical Kinetics, Arrhenius Activation Energy & Dynamic Equilibrium Simulator.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_chemical_kinetics import (
    InteractiveChemicalKineticsItem,
    KINETICS_PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import ChemistryTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def kinetics_sim(qapp):
    item = InteractiveChemicalKineticsItem(
        pos=(200.0, 150.0),
        width=780.0,
        height=500.0,
        preset="HABER_BOSCH",
        temperature_k=350.0,
    )
    yield item
    item.close()


class TestInteractiveChemicalKinetics:
    """Test suite for InteractiveChemicalKineticsItem."""

    def test_kinetics_initialization(self, kinetics_sim: InteractiveChemicalKineticsItem):
        assert kinetics_sim.preset == "HABER_BOSCH"
        assert kinetics_sim.temperature_k == pytest.approx(350.0, abs=1e-3)
        assert kinetics_sim.has_catalyst is False
        assert kinetics_sim.conc_a == pytest.approx(0.90, abs=1e-3)
        assert kinetics_sim.conc_c == pytest.approx(0.10, abs=1e-3)
        assert kinetics_sim.forward_rate_vf > 0.0
        assert kinetics_sim.reverse_rate_vr >= 0.0
        assert kinetics_sim.is_running is True

        # Initially [A] is large, [C] is small => Qc < Kc => Shifts forward
        assert "Thuận" in kinetics_sim.le_chatelier_shift

    def test_reaction_presets(self, kinetics_sim: InteractiveChemicalKineticsItem):
        expected_presets = ["HABER_BOSCH", "HYDROGEN_IODIDE", "NITROGEN_DIOXIDE", "GENERIC"]

        for pr in expected_presets:
            kinetics_sim.set_preset(pr)
            assert kinetics_sim.preset == pr
            assert kinetics_sim.conc_a == pytest.approx(0.90, abs=1e-3)
            spec = KINETICS_PRESETS[pr]
            assert len(spec[0]) > 0  # Name
            assert len(spec[1]) > 0  # Equation
            assert spec[3] > 0.0     # Ea > 0

    def test_arrhenius_temperature_dependence(self, kinetics_sim: InteractiveChemicalKineticsItem):
        kinetics_sim.set_temperature(300.0)
        kf_cold = kinetics_sim.rate_constant_kf
        kr_cold = kinetics_sim.rate_constant_kr

        kinetics_sim.set_temperature(450.0)
        kf_hot = kinetics_sim.rate_constant_kf
        kr_hot = kinetics_sim.rate_constant_kr

        # According to Arrhenius law, rate constants increase with temperature
        assert kf_hot > kf_cold
        assert kr_hot > kr_cold

    def test_catalyst_acceleration(self, kinetics_sim: InteractiveChemicalKineticsItem):
        kinetics_sim.reset_simulation()
        assert kinetics_sim.has_catalyst is False
        vf_uncat = kinetics_sim.forward_rate_vf
        vr_uncat = kinetics_sim.reverse_rate_vr
        kc_uncat = kinetics_sim.equilibrium_constant_kc

        # Toggle catalyst ON
        kinetics_sim.toggle_catalyst()
        assert kinetics_sim.has_catalyst is True
        vf_cat = kinetics_sim.forward_rate_vf
        vr_cat = kinetics_sim.reverse_rate_vr
        kc_cat = kinetics_sim.equilibrium_constant_kc

        # Catalyst increases reaction rates by lowering Ea
        assert vf_cat > vf_uncat
        assert vr_cat > vr_uncat
        # Catalyst lowers Ea_f and Ea_r equally, so Kc remains unchanged
        assert kc_cat == pytest.approx(kc_uncat, rel=1e-2)

    def test_le_chatelier_perturbations(self, kinetics_sim: InteractiveChemicalKineticsItem):
        kinetics_sim.reset_simulation()

        # 1. Add reactants [A] => increases forward rate vf
        init_vf = kinetics_sim.forward_rate_vf
        kinetics_sim.add_reactants(0.40)
        assert kinetics_sim.conc_a > 1.20
        assert kinetics_sim.forward_rate_vf > init_vf
        assert "Thuận" in kinetics_sim.le_chatelier_shift

        # 2. Add products [C] => increases reverse rate vr
        init_vr = kinetics_sim.reverse_rate_vr
        kinetics_sim.add_products(0.60)
        assert kinetics_sim.conc_c > 0.60
        assert kinetics_sim.reverse_rate_vr > init_vr

    def test_kinetics_ode_step(self, kinetics_sim: InteractiveChemicalKineticsItem):
        kinetics_sim.reset_simulation()
        init_a = kinetics_sim.conc_a
        init_c = kinetics_sim.conc_c

        # Step forward with dt = 0.5s
        for _ in range(10):
            kinetics_sim._kinetics_step(0.05)

        # Reactants should be consumed, products should be formed
        assert kinetics_sim.conc_a < init_a
        assert kinetics_sim.conc_c > init_c

    def test_air_gestures_and_mouse_interactions(self, kinetics_sim: InteractiveChemicalKineticsItem):
        # Air Tap on Hydrogen Iodide preset
        pt_hi = kinetics_sim.mapToScene(kinetics_sim._rect_btn_hi.center())
        assert kinetics_sim.handle_air_tap(pt_hi) is True
        assert kinetics_sim.preset == "HYDROGEN_IODIDE"

        # Air Tap on Catalyst toggle
        pt_cat = kinetics_sim.mapToScene(kinetics_sim._rect_btn_catalyst.center())
        init_cat = kinetics_sim.has_catalyst
        assert kinetics_sim.handle_air_tap(pt_cat) is True
        assert kinetics_sim.has_catalyst != init_cat

        # Air Tap on Add Reactants
        pt_add = kinetics_sim.mapToScene(kinetics_sim._rect_btn_add_react.center())
        c_before = kinetics_sim.conc_a
        assert kinetics_sim.handle_air_tap(pt_add) is True
        assert kinetics_sim.conc_a > c_before

        # Air Tap on Reset
        pt_reset = kinetics_sim.mapToScene(kinetics_sim._rect_btn_reset.center())
        assert kinetics_sim.handle_air_tap(pt_reset) is True
        assert kinetics_sim.conc_a == pytest.approx(0.90, abs=1e-3)

        # Air Pinch on Profile box (temperature modulation)
        prof_center = QPointF(
            kinetics_sim._prof_x + kinetics_sim._prof_w * 0.5,
            kinetics_sim._prof_y + kinetics_sim._prof_h * 0.3
        )
        scene_pt = kinetics_sim.mapToScene(prof_center)
        consumed = kinetics_sim.handle_air_pinch(scene_pt)
        assert consumed is True
        assert kinetics_sim.temperature_k > 380.0

    def test_stamping_and_lesson_board_integration(self, qapp):
        board = LessonBoard()

        # Check visibility toggle
        assert board.is_chemical_kinetics_visible() is False
        item = board.show_chemical_kinetics(True)
        assert item is not None
        assert board.is_chemical_kinetics_visible() is True
        assert board.get_chemical_kinetics() is item

        # Toggle visibility
        is_now_vis = board.toggle_chemical_kinetics()
        assert is_now_vis is False
        assert board.is_chemical_kinetics_visible() is False

        # Toggle back on
        board.toggle_chemical_kinetics()
        assert board.is_chemical_kinetics_visible() is True

        # Eraser exclusion check
        sim_item = board.get_chemical_kinetics_item()
        sim_rect = sim_item.sceneBoundingRect()
        items_in_rect = board.get_items_in_rect(sim_rect)
        assert sim_item not in items_in_rect

        # Stamping vector diagram
        stamped_items = sim_item.stamp_to_canvas(board)
        assert len(stamped_items) > 0

        # Clean up
        sim_item.close()
