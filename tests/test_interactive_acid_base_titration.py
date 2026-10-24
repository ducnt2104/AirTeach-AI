"""
Unit tests for Interactive Chemistry Acid-Base Titration & pH Buffer Curve Simulator.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_acid_base_titration import InteractiveAcidBaseTitrationItem
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import ChemistryTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def titration_sim(qapp):
    item = InteractiveAcidBaseTitrationItem(
        pos=(200.0, 150.0),
        width=760.0,
        height=490.0,
        acid_type="WEAK_ACID",
        indicator="PHENOLPHTHALEIN",
        v_titrant=0.0,
        c_acid=0.10,
        c_base=0.10,
        v_acid_init=25.0,
    )
    yield item
    item.close()


class TestInteractiveAcidBaseTitration:
    """Test suite for InteractiveAcidBaseTitrationItem."""

    def test_titration_init_and_properties(self, titration_sim: InteractiveAcidBaseTitrationItem):
        assert titration_sim.acid_type == "WEAK_ACID"
        assert titration_sim.indicator == "PHENOLPHTHALEIN"
        assert titration_sim.v_titrant_ml == pytest.approx(0.0, abs=1e-3)
        assert titration_sim.v_eq_ml == pytest.approx(25.0, abs=1e-3)
        assert titration_sim.is_dispensing is False

        # Initial weak acid pH ≈ 2.88
        assert titration_sim.current_ph == pytest.approx(2.88, abs=0.05)

    def test_weak_acid_henderson_hasselbalch_buffer_and_equivalence(self, titration_sim: InteractiveAcidBaseTitrationItem):
        # 1. Initial pH (0 mL)
        assert titration_sim.calculate_ph_at_volume(0.0) == pytest.approx(2.88, abs=0.05)

        # 2. Buffer Region & Half-Equivalence Point (V = 12.5 mL = 0.5 * Veq)
        # At half-equivalence: [A-] = [HA] => pH = pKa = 4.76
        ph_half = titration_sim.calculate_ph_at_volume(12.5)
        assert ph_half == pytest.approx(4.76, abs=0.02)

        # 3. Buffer capacity: Small addition around 12.5 mL changes pH very little
        ph_10 = titration_sim.calculate_ph_at_volume(10.0)
        ph_15 = titration_sim.calculate_ph_at_volume(15.0)
        assert 4.0 < ph_10 < ph_half < ph_15 < 5.5

        # 4. Equivalence Point (V = 25.0 mL)
        # Salt hydrolysis of acetate creates basic pH = 8.73
        ph_eq = titration_sim.calculate_ph_at_volume(25.0)
        assert ph_eq == pytest.approx(8.73, abs=0.08)

        # 5. Post-Equivalence Point (V = 40.0 mL)
        # Driven by excess strong base NaOH: pH > 12.0
        ph_post = titration_sim.calculate_ph_at_volume(40.0)
        assert ph_post > 12.0

    def test_strong_acid_equivalence_point_neutral(self, titration_sim: InteractiveAcidBaseTitrationItem):
        titration_sim.set_acid_type("STRONG_ACID")
        assert titration_sim.acid_type == "STRONG_ACID"

        # 1. Initial strong acid pH: [H+] = 0.10M => pH = 1.00
        assert titration_sim.calculate_ph_at_volume(0.0) == pytest.approx(1.00, abs=0.05)

        # 2. Strong acid pre-equivalence
        assert titration_sim.calculate_ph_at_volume(10.0) < 2.0
        assert titration_sim.calculate_ph_at_volume(20.0) < 3.0

        # 3. Equivalence point (V = 25.0 mL): Neutral salt NaCl => pH = 7.00
        assert titration_sim.calculate_ph_at_volume(25.0) == pytest.approx(7.00, abs=0.05)

        # 4. Post-equivalence
        assert titration_sim.calculate_ph_at_volume(35.0) > 12.0

    def test_first_derivative_peak_detection(self, titration_sim: InteractiveAcidBaseTitrationItem):
        # The slope dpH/dV should peak sharply around V = 25.0 mL
        deriv_buffer = titration_sim.calculate_ph_derivative(12.5)
        deriv_eq = titration_sim.calculate_ph_derivative(25.0)

        assert deriv_buffer < 0.2
        assert deriv_eq > 5.0
        assert deriv_eq > 10.0 * deriv_buffer

    def test_indicator_color_transitions(self, titration_sim: InteractiveAcidBaseTitrationItem):
        # 1. Phenolphthalein: colorless at acidic/neutral pH, pink/magenta at pH > 8.2
        titration_sim.set_indicator("PHENOLPHTHALEIN")
        titration_sim.set_titrant_volume(5.0)  # Acidic (pH ~ 4.2)
        col_acid = titration_sim.get_solution_flask_color()
        assert col_acid.alpha() < 100  # Transparent/pale

        titration_sim.set_titrant_volume(30.0) # Basic (pH > 11)
        col_base = titration_sim.get_solution_flask_color()
        assert col_base.red() > 200 and col_base.alpha() > 150 # Vivid pink/red

        # 2. Methyl Orange: Red at pH < 3.1, Yellow at pH > 4.4
        titration_sim.set_indicator("METHYL_ORANGE")
        titration_sim.set_titrant_volume(0.0)  # pH ~ 2.88
        col_mo_red = titration_sim.get_solution_flask_color()
        assert col_mo_red.red() > 200

        titration_sim.set_titrant_volume(20.0) # pH > 5.0
        col_mo_yellow = titration_sim.get_solution_flask_color()
        assert col_mo_yellow.green() > 150

        # 3. Universal Indicator: color changes across spectrum
        titration_sim.set_indicator("UNIVERSAL")
        col_univ = titration_sim.get_solution_flask_color()
        assert isinstance(col_univ, QColor)

    def test_touchless_air_gestures(self, titration_sim: InteractiveAcidBaseTitrationItem):
        # 1. Air pinch on burette column to scrub titrant volume
        bur_scene_pos = titration_sim.mapToScene(QPointF(80.0, 135.0))
        assert titration_sim.handle_air_pinch(bur_scene_pos, is_pressed=True) is True
        assert titration_sim.v_titrant_ml > 0.0

        # Release pinch
        assert titration_sim.handle_air_pinch(bur_scene_pos, is_pressed=False) is False

        # 2. Air tap on buttons
        # Drop small (+0.5 mL)
        btn_drop_s = titration_sim.mapToScene(titration_sim._rect_btn_drop_small.center())
        prev_v = titration_sim.v_titrant_ml
        assert titration_sim.handle_air_tap(btn_drop_s) is True
        assert titration_sim.v_titrant_ml == pytest.approx(prev_v + 0.5, abs=1e-3)

        # Drop large (+2.0 mL)
        btn_drop_l = titration_sim.mapToScene(titration_sim._rect_btn_drop_large.center())
        prev_v2 = titration_sim.v_titrant_ml
        assert titration_sim.handle_air_tap(btn_drop_l) is True
        assert titration_sim.v_titrant_ml == pytest.approx(prev_v2 + 2.0, abs=1e-3)

        # Auto-flow toggle
        btn_flow = titration_sim.mapToScene(titration_sim._rect_btn_auto_flow.center())
        assert titration_sim.handle_air_tap(btn_flow) is True
        assert titration_sim.is_dispensing is True
        assert titration_sim.handle_air_tap(btn_flow) is True
        assert titration_sim.is_dispensing is False

        # Indicator buttons
        btn_methyl = titration_sim.mapToScene(titration_sim._rect_btn_ind_methyl.center())
        assert titration_sim.handle_air_tap(btn_methyl) is True
        assert titration_sim.indicator == "METHYL_ORANGE"

        # Reset button
        btn_reset = titration_sim.mapToScene(titration_sim._rect_btn_reset.center())
        assert titration_sim.handle_air_tap(btn_reset) is True
        assert titration_sim.v_titrant_ml == pytest.approx(0.0, abs=1e-3)

    def test_stem_tools_create_titration_curve_and_stamping(self, qapp):
        # Verify vector drawing generation
        items = ChemistryTools.create_titration_curve_diagram(
            center=(400.0, 300.0),
            width=420.0,
            height=260.0,
            acid_type="WEAK_ACID",
            v_titrant=25.0,
            v_eq=25.0,
            c_acid=0.10,
            c_base=0.10,
        )
        assert len(items) >= 12  # Banner, burette, fluid, valve, flask, axes, labels, guides, curve, E-point, half-eq

        board = LessonBoard()
        titration_sim = InteractiveAcidBaseTitrationItem(pos=(200.0, 150.0))
        board.addItem(titration_sim)

        # Stamping
        stamp_called = False

        def on_stamp():
            nonlocal stamp_called
            stamp_called = True

        titration_sim.stamp_requested.connect(on_stamp)
        stamped = titration_sim.stamp_to_canvas(board)
        assert stamp_called is True
        assert len(stamped) >= 12

        # LessonBoard lifecycle
        board.show_titration(True, pos=(300.0, 200.0))
        assert board.is_titration_visible() is True
        tit_item = board.get_titration()
        assert tit_item is not None
        assert tit_item.pos() == QPointF(300.0, 200.0)

        # Toggle hide & show
        board.toggle_titration()
        assert board.get_titration() is None
        assert board.is_titration_visible() is False

        board.toggle_titration()
        assert board.get_titration() is not None

        # Exclusion from eraser
        items_in_rect = board.get_items_in_rect(board.sceneRect())
        assert board.get_titration() not in items_in_rect
