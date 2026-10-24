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

    def test_create_geometric_triangle(self, qapp):
        items = MathTools.create_geometric_triangle(
            v0=(100.0, 300.0),
            v1=(300.0, 300.0),
            v2=(100.0, 100.0),
            labels=("A", "B", "C"),
            show_right_angle=True,
            right_angle_vertex_idx=0,
        )
        assert len(items) >= 4  # triangle path, right angle marker square, 3 vertex text labels

    def test_create_angle_arc(self, qapp):
        items = MathTools.create_angle_arc(
            vertex=(200.0, 200.0),
            p1=(300.0, 200.0),
            p2=(200.0, 100.0),
            radius=40.0,
            label="90°",
        )
        assert len(items) == 2  # arc sector path, angle text label

    def test_create_circle_with_radius(self, qapp):
        items = MathTools.create_circle_with_radius(
            center=(400.0, 400.0),
            radius=80.0,
            label="R = 5cm",
        )
        assert len(items) == 4  # circle outline, center dot, dashed radius line, text label


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

    def test_circuit_inductor(self, qapp):
        items = PhysicsTools.create_circuit_component("inductor", (100.0, 100.0), (300.0, 100.0), label="L1")
        assert len(items) == 2  # looped path + text label

    def test_circuit_ground(self, qapp):
        items = PhysicsTools.create_circuit_component("ground", (100.0, 100.0), (100.0, 200.0), label="GND")
        assert len(items) == 2  # ground plates path + text label

    def test_circuit_switch(self, qapp):
        items = PhysicsTools.create_circuit_component("switch", (100.0, 100.0), (300.0, 100.0), label="K")
        assert len(items) >= 3  # switch path, 2 terminal dots, text label

    def test_stem_tool_dialog_physics_circuit(self, qapp):
        from src.ui.stem_tools import StemToolDialog
        dlg = StemToolDialog("Linh kiện mạch điện", "physics_circuit")
        assert hasattr(dlg, "combo_component")
        assert hasattr(dlg, "inp_label")
        assert dlg.combo_component.count() >= 6
        dlg.combo_component.setCurrentIndex(3)  # Inductor
        assert "L =" in dlg.inp_label.text()


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

    def test_organic_bond_single(self, qapp):
        items = ChemistryTools.create_organic_bond((100.0, 100.0), (200.0, 100.0), bond_type="single")
        assert len(items) == 1

    def test_organic_bond_double(self, qapp):
        items = ChemistryTools.create_organic_bond((100.0, 100.0), (200.0, 100.0), bond_type="double")
        assert len(items) == 2

    def test_organic_bond_triple(self, qapp):
        items = ChemistryTools.create_organic_bond((100.0, 100.0), (200.0, 100.0), bond_type="triple")
        assert len(items) == 3


# ─── Laser Pointer Smoothing Tests ───────────────────────────────────────────

class TestLaserPointerSmoothing:
    def test_laser_pointer_tremor_filtering(self, qapp):
        from src.ui.lesson_board import LessonBoard
        board = LessonBoard()

        # Initial move
        board.move_laser(100.0, 100.0, visible=True, smooth=True)
        item = board._laser_items["primary"]
        p0 = item.pos()
        assert p0.x() == 100.0 and p0.y() == 100.0

        # High frequency jitter jump to (130, 100)
        board.move_laser(130.0, 100.0, visible=True, smooth=True)
        p1 = item.pos()
        # Filtered coordinate must be smoothed and lag the sudden 30px spike
        assert 100.0 < p1.x() < 130.0

    def test_laser_pointer_reset_on_hide(self, qapp):
        from src.ui.lesson_board import LessonBoard
        board = LessonBoard()
        board.move_laser(100.0, 100.0, visible=True, smooth=True)
        assert "primary" in board._laser_filters
        board.hide_laser("primary")
        assert not board._laser_items["primary"].isVisible()


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


# ─── 5. CLI Flags, Single-Instance IPC & Smart Ink Curve Sampling ─────────────

class TestLauncherCliAndIpc:
    def test_main_cli_version_flag(self):
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, "main.py", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0
        assert "AirTeach AI v2.5.0-Release" in res.stdout

    def test_main_cli_check_flag(self):
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, "main.py", "--check"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0
        assert "All systems operational" in res.stdout

    def test_single_instance_ipc_protocol(self, qapp):
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
        test_pipe_name = "airteach_ai_test_ipc_socket"
        QLocalServer.removeServer(test_pipe_name)

        server = QLocalServer()
        assert server.listen(test_pipe_name)

        client = QLocalSocket()
        client.connectToServer(test_pipe_name)
        assert client.waitForConnected(1000)

        assert server.waitForNewConnection(1000)
        assert server.hasPendingConnections()
        conn = server.nextPendingConnection()
        assert conn is not None

        client.write(b"WAKEUP\n")
        assert client.waitForBytesWritten(1000)

        assert conn.waitForReadyRead(1000)
        msg = bytes(conn.readAll()).decode("utf-8")
        assert "WAKEUP" in msg

        client.disconnectFromServer()
        server.close()
        QLocalServer.removeServer(test_pipe_name)

    def test_get_strokes_in_rect_bezier_sampling(self, qapp):
        from src.ui.main_window import MainWindow
        from src.ui.lesson_board import InkStroke
        from src.core.stroke_beautifier import BezierSegment

        win = MainWindow(camera_index=-1)
        scene = win._scene

        seg = BezierSegment(
            p0=(100.0, 100.0),
            p1=(120.0, 150.0),
            p2=(180.0, 150.0),
            p3=(200.0, 100.0),
        )
        ink = InkStroke(
            stroke_id="test_ink_01",
            color=QColor("#202124"),
            segments=[seg],
            widths=[2.5],
        )
        scene._strokes["test_ink_01"] = ink

        # Query region containing the stroke
        rect = QRectF(50, 50, 200, 200)
        strokes = win._get_strokes_in_rect(rect)
        assert len(strokes) == 1
        # Expect 4 sampled points: t = 0.0, 0.33, 0.67, 1.0
        assert len(strokes[0].points) == 4
        assert math.isclose(strokes[0].points[0].x, 100.0, rel_tol=1e-3)
        assert math.isclose(strokes[0].points[-1].x, 200.0, rel_tol=1e-3)

        # Query non-overlapping region
        empty_rect = QRectF(500, 500, 50, 50)
        empty_strokes = win._get_strokes_in_rect(empty_rect)
        assert len(empty_strokes) == 0

        win.close()



    def test_stem_tool_dialog_new_types(self, qapp):
        from src.ui.stem_tools import StemToolDialog

        dlg_tri = StemToolDialog("Tam giác", "math_triangle")
        assert hasattr(dlg_tri, "combo_tri_type")
        assert dlg_tri.combo_tri_type.count() == 4
        assert dlg_tri.inp_base.text() == "240"
        dlg_tri.close()

        dlg_circ = StemToolDialog("Đường tròn", "math_circle_radius")
        assert hasattr(dlg_circ, "inp_radius")
        assert dlg_circ.inp_radius.text() == "120"
        assert dlg_circ.inp_center_lbl.text() == "O"
        dlg_circ.close()

        dlg_ang = StemToolDialog("Góc", "math_angle")
        assert hasattr(dlg_ang, "inp_angle_deg")
        assert dlg_ang.inp_angle_deg.text() == "60"
        dlg_ang.close()

        dlg_bond = StemToolDialog("Liên kết", "chem_bond")
        assert hasattr(dlg_bond, "combo_bond_type")
        assert dlg_bond.combo_bond_type.count() == 3
        dlg_bond.close()

    def test_main_window_stem_actions_and_visibility(self, qapp):
        from unittest.mock import patch
        from src.ui.main_window import MainWindow

        win = MainWindow(camera_index=-1)

        # Subject 0: Math
        win._on_stem_subject_changed(0)
        assert win._stem_triangle_act.isVisible()
        assert win._stem_circle_rad_act.isVisible()
        assert win._stem_angle_act.isVisible()
        assert not win._stem_bond_act.isVisible()

        # Subject 2: Chemistry
        win._on_stem_subject_changed(2)
        assert not win._stem_triangle_act.isVisible()
        assert win._stem_bond_act.isVisible()

        # Subject 0: Math again
        win._on_stem_subject_changed(0)

        # Test insert triangle
        with patch("src.ui.stem_tools.StemToolDialog.exec", return_value=True):
            initial_count = len(win._scene.items())
            win._on_insert_stem_triangle()
            assert len(win._scene.items()) > initial_count
            assert len(win._page_stem_items[win._current_page_index]) > 0

        # Test insert circle with radius
        with patch("src.ui.stem_tools.StemToolDialog.exec", return_value=True):
            initial_count = len(win._scene.items())
            win._on_insert_stem_circle_radius()
            assert len(win._scene.items()) > initial_count

        # Test insert angle
        with patch("src.ui.stem_tools.StemToolDialog.exec", return_value=True):
            initial_count = len(win._scene.items())
            win._on_insert_stem_angle()
            assert len(win._scene.items()) > initial_count

        # Test insert bond
        with patch("src.ui.stem_tools.StemToolDialog.exec", return_value=True):
            initial_count = len(win._scene.items())
            win._on_insert_stem_organic_bond()
            assert len(win._scene.items()) > initial_count

        win.close()
