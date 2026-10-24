"""
AirTeach AI - Interactive STEM Logic Gate & Circuit Simulator Tests
===================================================================
Verifies:
1. InteractiveLogicGateItem boolean evaluation across all gate types (AND, OR, NOT, NAND, NOR, XOR).
2. Clickable/air-tappable input pin toggling (0/1), output signal emission, and truth table HUD.
3. Magnetic connection pin ports and snap detection.
4. InformaticsTools Manhattan orthogonal wire router and flowchart cardinal pin snapping.
5. MainWindow integration and touchless air tap toggling on the whiteboard canvas.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.core.hand_gesture_engine import GestureState
from src.core.multi_hand_coordinator import HandLandmarkData, HandRole
from src.ui.interactive_logic_gate import InteractiveLogicGateItem
from src.ui.main_window import MainWindow
from src.ui.stem_tools import InformaticsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── 1. Boolean Evaluation Tests ───────────────────────────────────────────────


def test_interactive_gate_boolean_evaluations(qapp):
    """Verify authentic boolean simulation across all logic gate types."""
    # 1. AND Gate
    and_gate = InteractiveLogicGateItem("AND", (500, 300))
    and_gate.set_input_state("A", 0)
    and_gate.set_input_state("B", 0)
    assert and_gate.evaluate() == 0

    and_gate.set_input_state("A", 1)
    and_gate.set_input_state("B", 0)
    assert and_gate.evaluate() == 0

    and_gate.set_input_state("A", 1)
    and_gate.set_input_state("B", 1)
    assert and_gate.evaluate() == 1

    # 2. OR Gate
    or_gate = InteractiveLogicGateItem("OR", (500, 300))
    or_gate.set_input_state("A", 0)
    or_gate.set_input_state("B", 0)
    assert or_gate.evaluate() == 0

    or_gate.set_input_state("A", 1)
    or_gate.set_input_state("B", 0)
    assert or_gate.evaluate() == 1

    # 3. NOT Gate (Inverter)
    not_gate = InteractiveLogicGateItem("NOT", (500, 300))
    not_gate.set_input_state("A", 0)
    assert not_gate.evaluate() == 1

    not_gate.set_input_state("A", 1)
    assert not_gate.evaluate() == 0

    # 4. NAND Gate
    nand_gate = InteractiveLogicGateItem("NAND", (500, 300))
    nand_gate.set_input_state("A", 1)
    nand_gate.set_input_state("B", 1)
    assert nand_gate.evaluate() == 0
    nand_gate.set_input_state("B", 0)
    assert nand_gate.evaluate() == 1

    # 5. NOR Gate
    nor_gate = InteractiveLogicGateItem("NOR", (500, 300))
    nor_gate.set_input_state("A", 0)
    nor_gate.set_input_state("B", 0)
    assert nor_gate.evaluate() == 1
    nor_gate.set_input_state("A", 1)
    assert nor_gate.evaluate() == 0

    # 6. XOR Gate
    xor_gate = InteractiveLogicGateItem("XOR", (500, 300))
    xor_gate.set_input_state("A", 0)
    xor_gate.set_input_state("B", 0)
    assert xor_gate.evaluate() == 0
    xor_gate.set_input_state("A", 1)
    xor_gate.set_input_state("B", 0)
    assert xor_gate.evaluate() == 1
    xor_gate.set_input_state("A", 1)
    xor_gate.set_input_state("B", 1)
    assert xor_gate.evaluate() == 0


# ── 2. Pin Toggling & Signal Tests ────────────────────────────────────────────


def test_interactive_gate_pin_toggle_and_signals(qapp):
    """Verify input pin toggling emits state_changed and updates output."""
    gate = InteractiveLogicGateItem("AND", (400, 400))
    emitted = []
    gate.state_changed.connect(lambda gt, out: emitted.append((gt, out)))

    # Toggle A: 0 -> 1
    new_a = gate.toggle_input("A")
    assert new_a == 1
    assert gate._input_states["A"] == 1
    assert emitted[-1] == ("AND", 0)

    # Toggle B: 0 -> 1
    new_b = gate.toggle_input("B")
    assert new_b == 1
    assert gate._input_states["B"] == 1
    assert emitted[-1] == ("AND", 1)

    # Truth table HUD toggle
    tt_signals = []
    gate.truth_table_toggled.connect(lambda visible: tt_signals.append(visible))
    assert gate._show_truth_table is False

    is_vis = gate.toggle_truth_table()
    assert is_vis is True
    assert gate._show_truth_table is True
    assert tt_signals == [True]

    is_vis2 = gate.toggle_truth_table()
    assert is_vis2 is False
    assert tt_signals == [True, False]


# ── 3. Connection Pin Ports & Magnetic Snapping ───────────────────────────────


def test_interactive_gate_magnetic_pin_snapping(qapp):
    """Verify connection pin coordinates and magnetic snap detection."""
    scene = QGraphicsScene(0, 0, 1920, 1080)
    gate = InteractiveLogicGateItem("AND", (600, 400))
    scene.addItem(gate)

    in_a_pos = gate.get_input_scene_pos("A")
    in_b_pos = gate.get_input_scene_pos("B")
    out_y_pos = gate.get_output_scene_pos()

    assert in_a_pos.x() < 600.0
    assert out_y_pos.x() > 600.0

    # Near Pin A within 10px -> Snapped
    snap_a = gate.snap_pin_at(QPointF(in_a_pos.x() + 5.0, in_a_pos.y() - 3.0), tolerance=20.0)
    assert snap_a is not None
    pin_name, snapped_pos = snap_a
    assert pin_name == "A"
    assert snapped_pos == in_a_pos

    # Near Output Pin Y within 10px -> Snapped
    snap_y = gate.snap_pin_at(QPointF(out_y_pos.x() - 4.0, out_y_pos.y() + 2.0), tolerance=20.0)
    assert snap_y is not None
    assert snap_y[0] == "Y"
    assert snap_y[1] == out_y_pos

    # Far away point -> Not snapped
    far_snap = gate.snap_pin_at(QPointF(100.0, 100.0), tolerance=20.0)
    assert far_snap is None


# ── 4. Touchless Air Click Interaction ────────────────────────────────────────


def test_interactive_gate_air_click(qapp):
    """Verify air_click_at allows touchless toggling of pins and truth table."""
    scene = QGraphicsScene(0, 0, 1920, 1080)
    gate = InteractiveLogicGateItem("OR", (700, 500))
    scene.addItem(gate)

    assert gate._input_states["A"] == 0

    in_a_pos = gate.get_input_scene_pos("A")
    # Air tap directly over pin A
    toggled = gate.air_click_at(in_a_pos.x() - 10.0, in_a_pos.y())
    assert toggled is True
    assert gate._input_states["A"] == 1

    # Air tap over Truth table button
    tt_pos = gate.mapToScene(QPointF(0.0, -gate._height / 2.0 - 15.0))
    toggled_tt = gate.air_click_at(tt_pos.x(), tt_pos.y())
    assert toggled_tt is True
    assert gate._show_truth_table is True

    # Air tap in empty space
    assert gate.air_click_at(10.0, 10.0) is False


# ── 5. InformaticsTools Manhattan Wire & Flowchart Pin Snapping ───────────────


def test_informatics_tools_flowchart_pins_and_manhattan_wire(qapp):
    """Verify cardinal pins computation and orthogonal wire generation."""
    # 1. Flowchart pins
    pins = InformaticsTools.get_flowchart_pins(center=(500.0, 300.0), width=160.0, height=80.0)
    assert "top" in pins and "bottom" in pins and "left" in pins and "right" in pins
    assert pins["top"] == (500.0, 260.0)
    assert pins["bottom"] == (500.0, 340.0)
    assert pins["left"] == (420.0, 300.0)
    assert pins["right"] == (580.0, 300.0)

    # 2. Nearest pin snapping
    snapped = InformaticsTools.snap_to_nearest_pin((585.0, 305.0), pins, tolerance=25.0)
    assert snapped is not None
    pin_id, pin_coord = snapped
    assert pin_id == "right"
    assert pin_coord == (580.0, 300.0)

    # 3. Manhattan orthogonal wire
    wire_items = InformaticsTools.create_manhattan_wire(
        start=(100.0, 100.0),
        end=(300.0, 250.0),
        color=QColor("#10b981"),
        has_junction_dot=True,
    )
    assert len(wire_items) == 2  # Wire path item + junction dot item

    # 4. Interactive gate factory method
    gate_item = InformaticsTools.create_interactive_logic_gate("XOR", (800.0, 600.0))
    assert isinstance(gate_item, InteractiveLogicGateItem)
    assert gate_item._gate_type == "XOR"


# ── 6. MainWindow Integration Tests ───────────────────────────────────────────


def test_main_window_interactive_logic_gate_integration(qapp):
    """Verify MainWindow integrates interactive logic gate and handles air tap events."""
    win = MainWindow(camera_index=-1)
    win.show()

    # Create and add interactive gate directly to active canvas
    gate = InformaticsTools.create_interactive_logic_gate("AND", (600.0, 400.0))
    win._add_stem_item_to_current_page(gate)
    assert gate in win._scene.items()

    # Simulate air tap event over pin A
    in_a_pos = gate.get_input_scene_pos("A")
    h_tap = HandLandmarkData(
        track_id="hand_0",
        role=HandRole.PRIMARY,
        gesture_state=GestureState.POINT,
        canvas_x=in_a_pos.x() - 10.0,
        canvas_y=in_a_pos.y(),
        index_tip_norm=(0.5, 0.5),
        is_air_tap=True,
    )

    initial_a = gate._input_states["A"]
    win._on_multi_gesture_event([h_tap])
    assert gate._input_states["A"] != initial_a

    win.close()
