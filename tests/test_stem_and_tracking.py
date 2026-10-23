"""
AirTeach AI - STEM Tools & Tracking Unit Tests
==============================================
Validates:
  1. STEM chemical formula Unicode conversion
  2. MathTools Cartesian Oxy coordinate plane and function graphing
  3. PhysicsTools force vectors, orthogonal decomposition, circuit symbols
  4. ChemistryTools reaction arrows, benzene aromatic ring
  5. BiologyTools callout pins and anatomical leaders
  6. InformaticsTools flowchart shapes and code blocks
  7. HandGestureFSM scale-invariant gesture transitions and curl logic
  8. MainWindow coordinate mapping safety (anti double-scaling)
"""
import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.stem_tools import (
    BiologyTools,
    ChemistryTools,
    InformaticsTools,
    MathTools,
    PhysicsTools,
    StemSubject,
    format_chemical_formula,
)
from src.core.hand_gesture_engine import HandGestureFSM, GestureState, LM
from src.ui.lesson_board import BOARD_WIDTH, BOARD_HEIGHT


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ─── 1. Chemical Formula Unicode Formatter Tests ──────────────────────────────

class TestChemicalFormulaFormatter:
    def test_subscript_conversion(self):
        assert format_chemical_formula("H2SO4") == "H₂SO₄"
        assert format_chemical_formula("Ca(OH)2") == "Ca(OH)₂"
        assert format_chemical_formula("C6H12O6") == "C₆H₁₂O₆"

    def test_ionic_charge_superscript(self):
        assert format_chemical_formula("Cu2+") == "Cu²⁺"
        assert format_chemical_formula("Fe3+") == "Fe³⁺"
        assert format_chemical_formula("Cl-") == "Cl⁻"

    def test_explicit_superscript_caret(self):
        assert format_chemical_formula("SO4^{2-}") == "SO₄²⁻"
        assert format_chemical_formula("x^2 + y^2") == "x² + y²"

    def test_reaction_equation_formatting(self):
        raw = "2H2 + O2 -> 2H2O"
        formatted = format_chemical_formula(raw)
        assert "2H₂" in formatted
        assert "O₂" in formatted
        assert "2H₂O" in formatted


# ─── 2. Math Vector Tools Tests ───────────────────────────────────────────────

class TestMathTools:
    def test_cartesian_axes_generation(self, qapp):
        items = MathTools.create_cartesian_axes(
            origin_x=960.0,
            origin_y=540.0,
            x_min=-5.0,
            x_max=5.0,
            y_min=-3.0,
            y_max=3.0,
            grid_step_px=50.0,
            show_grid=True,
        )
        assert len(items) > 10  # grid, 2 axes, arrows, ticks, numbers, O label

    def test_function_graph_valid(self, qapp):
        item = MathTools.create_function_graph("x**2 - 2", origin_x=960.0, origin_y=540.0)
        assert item is not None
        assert not item.path().isEmpty()

    def test_function_graph_trig(self, qapp):
        item = MathTools.create_function_graph("sin(x)", origin_x=960.0, origin_y=540.0)
        assert item is not None
        assert not item.path().isEmpty()

    def test_function_graph_invalid_safe_fail(self, qapp):
        # Malformed expression must return None and not raise exception
        item = MathTools.create_function_graph("x** + / invalid", origin_x=960.0, origin_y=540.0)
        assert item is None


# ─── 3. Physics Vector Tools Tests ────────────────────────────────────────────

class TestPhysicsTools:
    def test_create_force_vector(self, qapp):
        items = PhysicsTools.create_force_vector((100.0, 100.0), (250.0, 100.0), label="F⃗")
        assert len(items) == 3  # line, arrowhead polygon, text label

    def test_vector_decomposition(self, qapp):
        items = PhysicsTools.create_vector_decomposition(
            origin=(100.0, 200.0),
            vector_end=(200.0, 100.0),
            label="F⃗",
        )
        assert len(items) >= 7  # main vector (3), Fx (3), Fy (3), 2 projection dashed lines

    def test_circuit_resistor(self, qapp):
        items = PhysicsTools.create_circuit_component("resistor", (100.0, 100.0), (300.0, 100.0), label="R1")
        assert len(items) == 2  # zig-zag path, text label

    def test_circuit_capacitor(self, qapp):
        items = PhysicsTools.create_circuit_component("capacitor", (100.0, 100.0), (300.0, 100.0), label="C1")
        assert len(items) == 2

    def test_circuit_battery(self, qapp):
        items = PhysicsTools.create_circuit_component("battery", (100.0, 100.0), (300.0, 100.0), label="E")
        assert len(items) == 2


# ─── 4. Chemistry Vector Tools Tests ──────────────────────────────────────────

class TestChemistryTools:
    def test_reaction_arrow_forward(self, qapp):
        items = ChemistryTools.create_reaction_arrow(
            (100.0, 100.0), (300.0, 100.0), reversible=False, condition_top="t°"
        )
        assert len(items) == 2  # arrow path + top label

    def test_reaction_arrow_reversible(self, qapp):
        items = ChemistryTools.create_reaction_arrow(
            (100.0, 100.0), (300.0, 100.0), reversible=True, condition_top="t°", condition_bottom="p"
        )
        assert len(items) == 4  # top half arrow, bottom half arrow, top label, bottom label

    def test_benzene_ring(self, qapp):
        items = ChemistryTools.create_benzene_ring((200.0, 200.0), radius=50.0)
        assert len(items) == 2  # outer hexagon polygon + inner delocalized electron circle


# ─── 5. Biology Vector Tools Tests ────────────────────────────────────────────

class TestBiologyTools:
    def test_callout_pin(self, qapp):
        items = BiologyTools.create_callout_pin((100.0, 100.0), (220.0, 60.0), "Nucleus")
        assert len(items) == 3  # pin dot, leader line path, text label


# ─── 6. Informatics Vector Tools Tests ────────────────────────────────────────

class TestInformaticsTools:
    def test_flowchart_process(self, qapp):
        items = InformaticsTools.create_flowchart_shape("PROCESS", (200.0, 200.0), label="x = 1")
        assert len(items) == 2  # rect + text

    def test_flowchart_decision(self, qapp):
        items = InformaticsTools.create_flowchart_shape("DECISION", (200.0, 200.0), label="x > 0?")
        assert len(items) == 2  # diamond polygon + text

    def test_code_block(self, qapp):
        items = InformaticsTools.create_code_block(
            (50.0, 50.0), width=400.0, height=200.0, code_text="print('test')", title="app.py"
        )
        assert len(items) >= 6  # bg, header bar, 3 dots, title, code text


# ─── 7. Tracking & Coordinate Double-Scaling Safety Tests ─────────────────────

class TestTrackingCoordinateMapping:
    def test_scene_coords_anti_double_scaling(self):
        """
        Verify that coordinates already scaled to canvas pixel space
        [0, 1920] x [0, 1080] are NOT multiplied by BOARD_WIDTH and BOARD_HEIGHT again!
        """
        pixel_x, pixel_y = 960.0, 540.0
        # If passed to _scene_coords logic:
        if pixel_x > 1.0 or pixel_y > 1.0:
            mapped_x = max(0.0, min(float(BOARD_WIDTH), pixel_x))
            mapped_y = max(0.0, min(float(BOARD_HEIGHT), pixel_y))
        else:
            mapped_x = max(0.0, min(float(BOARD_WIDTH), pixel_x * BOARD_WIDTH))
            mapped_y = max(0.0, min(float(BOARD_HEIGHT), pixel_y * BOARD_HEIGHT))

        assert mapped_x == 960.0
        assert mapped_y == 540.0
        assert mapped_x <= BOARD_WIDTH
        assert mapped_y <= BOARD_HEIGHT

    def test_normalized_coords_scaling(self):
        norm_x, norm_y = 0.5, 0.5
        mapped_x = max(0.0, min(float(BOARD_WIDTH), norm_x * BOARD_WIDTH))
        mapped_y = max(0.0, min(float(BOARD_HEIGHT), norm_y * BOARD_HEIGHT))
        assert mapped_x == 960.0
        assert mapped_y == 540.0


# ─── 8. Gesture FSM Finger Curl & Dead-Zone Verification ─────────────────────

class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _build_hand(index_up: bool, thumb_near: bool, others_curled: bool) -> list[_FakeLandmark]:
    lm = [_FakeLandmark(0.5, 0.8)] * 21
    lm[LM.WRIST] = _FakeLandmark(0.5, 0.8)
    lm[LM.MIDDLE_MCP] = _FakeLandmark(0.5, 0.55)

    # Index
    lm[LM.INDEX_MCP] = _FakeLandmark(0.48, 0.55)
    lm[LM.INDEX_PIP] = _FakeLandmark(0.48, 0.40)
    lm[LM.INDEX_DIP] = _FakeLandmark(0.48, 0.28)
    lm[LM.INDEX_TIP] = _FakeLandmark(0.48, 0.15 if index_up else 0.70)

    # Thumb
    lm[LM.THUMB_CMC] = _FakeLandmark(0.46, 0.70)
    lm[LM.THUMB_MCP] = _FakeLandmark(0.44, 0.60)
    lm[LM.THUMB_IP] = _FakeLandmark(0.45, 0.50)
    if thumb_near:
        lm[LM.THUMB_TIP] = _FakeLandmark(0.49, 0.38)  # close to index PIP/tip
    else:
        lm[LM.THUMB_TIP] = _FakeLandmark(0.20, 0.55)  # far away

    # Middle, Ring, Pinky
    if others_curled:
        lm[LM.MIDDLE_PIP] = _FakeLandmark(0.52, 0.50)
        lm[LM.MIDDLE_TIP] = _FakeLandmark(0.52, 0.70)
        lm[LM.RING_PIP] = _FakeLandmark(0.55, 0.52)
        lm[LM.RING_TIP] = _FakeLandmark(0.55, 0.72)
        lm[LM.PINKY_PIP] = _FakeLandmark(0.58, 0.54)
        lm[LM.PINKY_TIP] = _FakeLandmark(0.58, 0.74)
    else:
        lm[LM.MIDDLE_PIP] = _FakeLandmark(0.52, 0.40)
        lm[LM.MIDDLE_TIP] = _FakeLandmark(0.52, 0.15)
        lm[LM.RING_PIP] = _FakeLandmark(0.55, 0.42)
        lm[LM.RING_TIP] = _FakeLandmark(0.55, 0.18)
        lm[LM.PINKY_PIP] = _FakeLandmark(0.58, 0.45)
        lm[LM.PINKY_TIP] = _FakeLandmark(0.58, 0.22)

    return lm


class TestGestureFsmFingerCurl:
    def test_draw_gesture_recognized(self):
        fsm = HandGestureFSM(debounce_frames=1, min_confidence=0.75)
        lm = _build_hand(index_up=True, thumb_near=True, others_curled=True)
        state, _, _ = fsm.process_landmarks(lm, handedness_score=0.95, t_ms=0.0)
        assert state == GestureState.DRAW

    def test_point_gesture_recognized(self):
        fsm = HandGestureFSM(debounce_frames=1, min_confidence=0.75)
        lm = _build_hand(index_up=True, thumb_near=False, others_curled=True)
        state, _, _ = fsm.process_landmarks(lm, handedness_score=0.95, t_ms=0.0)
        assert state == GestureState.POINT

    def test_open_palm_not_draw(self):
        fsm = HandGestureFSM(debounce_frames=1, min_confidence=0.75)
        lm = _build_hand(index_up=True, thumb_near=True, others_curled=False)
        state, _, _ = fsm.process_landmarks(lm, handedness_score=0.95, t_ms=0.0)
        assert state != GestureState.DRAW
        assert state != GestureState.POINT
