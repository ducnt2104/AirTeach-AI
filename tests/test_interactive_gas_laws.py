"""
Unit tests for Interactive Ideal Gas Laws & Maxwell-Boltzmann Kinetic Theory Simulator.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication

from src.ui.interactive_gas_laws import (
    InteractiveGasLawsItem,
    GAS_SPECS,
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
def gas_sim(qapp):
    item = InteractiveGasLawsItem(
        pos=(200.0, 150.0),
        width=740.0,
        height=480.0,
        gas_type="HELIUM",
        temperature_k=300.0,
        piston_fraction=0.70,
        process_mode="FREE",
    )
    yield item
    item.close()


class TestInteractiveGasLaws:
    """Test suite for InteractiveGasLawsItem."""

    def test_gas_laws_init_and_properties(self, gas_sim: InteractiveGasLawsItem):
        assert gas_sim.gas_type == "HELIUM"
        assert gas_sim.temperature_k == pytest.approx(300.0, rel=1e-2)
        assert gas_sim._piston_fraction == pytest.approx(0.70, rel=1e-2)
        assert gas_sim.process_mode == "FREE"
        assert gas_sim.is_running is True

        # Volume in Liters: 22.4 * 0.70 = 15.68 L
        assert gas_sim.volume_liters == pytest.approx(15.68, rel=1e-2)

        # Theoretical pressure P = nRT / V
        p_theo = gas_sim.theoretical_pressure_kpa
        assert p_theo > 0.0
        assert gas_sim.pressure_kpa == pytest.approx(p_theo, rel=1e-1)

        # RMS speed v_rms should be positive and finite
        assert gas_sim.v_rms > 0.0

    def test_kinetic_particle_step_and_pressure_updates(self, gas_sim: InteractiveGasLawsItem):
        # Step simulation several times
        for _ in range(40):
            gas_sim.step_simulation(0.028)

        # All particles must remain strictly within chamber boundaries
        left = gas_sim._chamber_x
        right = gas_sim._piston_x
        top = gas_sim._chamber_y
        bot = gas_sim._chamber_y + gas_sim._chamber_h

        for p in gas_sim._particles:
            assert p.x >= left - 1.0
            assert p.x <= right + 1.0
            assert p.y >= top - 1.0
            assert p.y <= bot + 1.0

        assert gas_sim.pressure_kpa > 0.0

    def test_isothermal_process_boyle_mariotte(self, gas_sim: InteractiveGasLawsItem):
        gas_sim.set_process_mode("ISOTHERMAL")
        assert gas_sim.process_mode == "ISOTHERMAL"

        t_initial = gas_sim.temperature_k
        v1 = gas_sim.volume_liters
        p1 = gas_sim.theoretical_pressure_kpa

        # Compress volume (e.g. from 0.70 to 0.40)
        gas_sim.set_piston_fraction(0.40)
        v2 = gas_sim.volume_liters
        p2 = gas_sim.theoretical_pressure_kpa

        # In isothermal, temperature must remain strictly constant
        assert gas_sim.temperature_k == pytest.approx(t_initial, rel=1e-4)

        # Boyle-Mariotte Law: P1 * V1 = P2 * V2
        assert (p1 * v1) == pytest.approx(p2 * v2, rel=1e-3)
        assert p2 > p1

    def test_isobaric_process_charles(self, gas_sim: InteractiveGasLawsItem):
        gas_sim.set_process_mode("ISOBARIC")
        assert gas_sim.process_mode == "ISOBARIC"

        p_initial = gas_sim.theoretical_pressure_kpa
        v1 = gas_sim.volume_liters
        t1 = gas_sim.temperature_k

        # Heat gas by 100K
        gas_sim.set_temperature(t1 + 100.0)
        v2 = gas_sim.volume_liters
        t2 = gas_sim.temperature_k

        # Charles's Law: V1 / T1 = V2 / T2
        assert (v1 / t1) == pytest.approx(v2 / t2, rel=2e-2)
        # Pressure should stay approximately constant
        assert gas_sim.theoretical_pressure_kpa == pytest.approx(p_initial, rel=2e-2)

    def test_isochoric_process_gay_lussac(self, gas_sim: InteractiveGasLawsItem):
        gas_sim.set_process_mode("ISOCHORIC")
        assert gas_sim.process_mode == "ISOCHORIC"

        v_initial = gas_sim.volume_liters
        p1 = gas_sim.theoretical_pressure_kpa
        t1 = gas_sim.temperature_k

        # In isochoric, piston cannot move
        gas_sim.set_piston_fraction(0.35)
        assert gas_sim.volume_liters == pytest.approx(v_initial, rel=1e-4)

        # Heating gas increases pressure proportionally: P1 / T1 = P2 / T2
        gas_sim.set_temperature(t1 + 150.0)
        p2 = gas_sim.theoretical_pressure_kpa
        t2 = gas_sim.temperature_k

        assert (p1 / t1) == pytest.approx(p2 / t2, rel=1e-3)
        assert p2 > p1

    def test_gas_species_switching_and_speeds(self, gas_sim: InteractiveGasLawsItem):
        gas_sim.set_temperature(300.0)

        # Helium (m = 4)
        gas_sim.set_gas_type("HELIUM")
        v_he = gas_sim.v_rms

        # Nitrogen (m = 28)
        gas_sim.set_gas_type("NITROGEN")
        v_n2 = gas_sim.v_rms

        # Argon (m = 40)
        gas_sim.set_gas_type("ARGON")
        v_ar = gas_sim.v_rms

        # Kinetic theory: lighter molecules have higher rms speed
        assert v_he > v_n2 > v_ar
        # Theoretical ratio: v_he / v_ar = sqrt(40 / 4) = sqrt(10) ≈ 3.162
        assert (v_he / v_ar) == pytest.approx(math.sqrt(40.0 / 4.0), rel=1e-2)

    def test_touchless_air_gestures(self, gas_sim: InteractiveGasLawsItem):
        # 1. Air pinch on piston handle to compress
        piston_handle_pos = gas_sim.mapToScene(QPointF(gas_sim._piston_x + 5.0, gas_sim._chamber_y + 50.0))

        # Pinch press
        assert gas_sim.handle_air_pinch(piston_handle_pos, is_pressed=True) is True
        assert gas_sim._is_dragging_piston is True

        # Pinch drag to new position
        drag_pos = gas_sim.mapToScene(QPointF(gas_sim._chamber_x + 150.0, gas_sim._chamber_y + 50.0))
        assert gas_sim.handle_air_pinch(drag_pos, is_pressed=True) is True

        # Pinch release
        assert gas_sim.handle_air_pinch(drag_pos, is_pressed=False) is True
        assert gas_sim._is_dragging_piston is False

        # 2. Air tap on buttons
        heat_btn_pos = gas_sim.mapToScene(gas_sim._rect_btn_heat.center())
        prev_t = gas_sim.temperature_k
        assert gas_sim.handle_air_tap(heat_btn_pos) is True
        assert gas_sim.temperature_k > prev_t

        cool_btn_pos = gas_sim.mapToScene(gas_sim._rect_btn_cool.center())
        assert gas_sim.handle_air_tap(cool_btn_pos) is True

        gas_n2_pos = gas_sim.mapToScene(gas_sim._rect_btn_gas_n2.center())
        assert gas_sim.handle_air_tap(gas_n2_pos) is True
        assert gas_sim.gas_type == "NITROGEN"

        reset_pos = gas_sim.mapToScene(gas_sim._rect_btn_reset.center())
        assert gas_sim.handle_air_tap(reset_pos) is True
        assert gas_sim.gas_type == "HELIUM"
        assert gas_sim.temperature_k == pytest.approx(300.0, rel=1e-2)

    def test_stem_tools_create_gas_chamber_and_stamping(self, qapp):
        # Verify vector drawing generation
        items = ChemistryTools.create_gas_chamber_diagram(
            center=(400.0, 300.0),
            width=380.0,
            height=230.0,
            piston_fraction=0.70,
            temperature=300.0,
            pressure_kpa=101.3,
            gas_type="HELIUM",
            particle_count=20,
        )
        assert len(items) >= 20  # Cylinder, piston, rod, handle, particles, trails, gauge, needle, thermo, banner

        board = LessonBoard()
        gas_sim = InteractiveGasLawsItem(pos=(200.0, 150.0))
        board.addItem(gas_sim)

        # Stamping
        stamp_called = False

        def on_stamp():
            nonlocal stamp_called
            stamp_called = True

        gas_sim.stamp_requested.connect(on_stamp)
        stamped = gas_sim.stamp_to_canvas(board)
        assert stamp_called is True
        assert len(stamped) >= 20

        # LessonBoard lifecycle
        board.show_gas_laws(True, pos=(300.0, 200.0))
        assert board.is_gas_laws_visible() is True
        gl_item = board.get_gas_laws()
        assert gl_item is not None
        assert gl_item.pos() == QPointF(300.0, 200.0)

        # Toggle hide & show
        board.toggle_gas_laws()
        assert board.get_gas_laws() is None
        assert board.is_gas_laws_visible() is False

        board.toggle_gas_laws()
        assert board.get_gas_laws() is not None

        # Exclusion from eraser
        items_in_rect = board.get_items_in_rect(board.sceneRect())
        assert board.get_gas_laws() not in items_in_rect
