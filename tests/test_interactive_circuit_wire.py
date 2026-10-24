"""
Unit Tests for Interactive Circuit Wire Orthogonal Routing & Signal Propagation
================================================================================
Covers:
1. Forward & reverse Manhattan 90-degree orthogonal polyline generation.
2. Real-time digital boolean state propagation across interconnected gates.
3. Multi-gate cascade circuits (AND gate -> NOT gate -> OR gate).
4. Dynamic route recalculation when gates are translated on the canvas.
5. Visual rendering with high/low state neon halos and solder junction dots.
6. Board-level wire management and touchless air tap interaction.
"""
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_circuit_wire import InteractiveCircuitWireItem
from src.ui.interactive_logic_gate import InteractiveLogicGateItem
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_manhattan_forward_routing(qapp):
    """Test 90-degree orthogonal polyline routing for standard forward gate layout."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(400.0, 150.0))

    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="A")

    points = wire._points
    assert len(points) == 4, f"Forward wire should have 4 points (2 bends), got {len(points)}"

    # Point 0 is at source output
    s_pos = gate_and.get_output_scene_pos()
    assert abs(points[0].x() - s_pos.x()) < 0.1
    assert abs(points[0].y() - s_pos.y()) < 0.1

    # Point 3 is at target input A
    t_pos = gate_or.get_input_scene_pos("A")
    assert abs(points[3].x() - t_pos.x()) < 0.1
    assert abs(points[3].y() - t_pos.y()) < 0.1

    # Orthogonal check: consecutive segments must be horizontal or vertical
    for i in range(len(points) - 1):
        dx = abs(points[i + 1].x() - points[i].x())
        dy = abs(points[i + 1].y() - points[i].y())
        assert dx < 1e-4 or dy < 1e-4, f"Segment {i} -> {i+1} is not orthogonal: dx={dx}, dy={dy}"


def test_manhattan_reverse_loop_routing(qapp):
    """Test 90-degree orthogonal polyline routing when target gate is behind source gate."""
    gate_and = InteractiveLogicGateItem("AND", center=(500.0, 200.0))
    gate_or = InteractiveLogicGateItem("OR", center=(200.0, 300.0))

    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="B")

    points = wire._points
    # Reverse loop should have 6 points (4 bends)
    assert len(points) == 6, f"Reverse loop wire should have 6 points (4 bends), got {len(points)}"

    # All segments must be strictly orthogonal
    for i in range(len(points) - 1):
        dx = abs(points[i + 1].x() - points[i].x())
        dy = abs(points[i + 1].y() - points[i].y())
        assert dx < 1e-4 or dy < 1e-4, f"Segment {i} -> {i+1} is not orthogonal: dx={dx}, dy={dy}"


def test_live_boolean_signal_propagation(qapp):
    """Test real-time digital logic propagation from AND output to OR input."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(350.0, 100.0))

    # AND gate initially (A=0, B=0) -> output=0
    assert gate_and.output_value() == 0

    propagated_values = []
    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="A")
    wire.signal_propagated.connect(lambda v: propagated_values.append(v))

    # Initial target input A should be 0
    assert wire.signal_value() == 0
    assert gate_or._input_states["A"] == 0
    assert gate_or.output_value() == 0

    # Set AND gate inputs: A=1, B=1 -> output=1
    gate_and.set_input_state("A", 1)
    gate_and.set_input_state("B", 1)

    assert gate_and.output_value() == 1
    # Wire should propagate 1 to OR gate input A
    assert wire.signal_value() == 1
    assert gate_or._input_states["A"] == 1
    assert gate_or.output_value() == 1  # OR(1, 0) == 1
    assert 1 in propagated_values

    # Toggle AND gate A back to 0 -> output drops to 0
    gate_and.toggle_input("A")
    assert gate_and.output_value() == 0
    assert wire.signal_value() == 0
    assert gate_or._input_states["A"] == 0
    assert gate_or.output_value() == 0  # OR(0, 0) == 0


def test_multi_gate_cascade_circuit(qapp):
    """Test multi-gate cascade: AND -> NOT (NAND equivalent)."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 150.0))
    gate_not = InteractiveLogicGateItem("NOT", center=(300.0, 150.0))

    wire = InteractiveCircuitWireItem(gate_and, gate_not, target_pin="A")

    # (0 AND 0) = 0 -> NOT(0) = 1
    assert wire.signal_value() == 0
    assert gate_not.output_value() == 1

    # (1 AND 1) = 1 -> NOT(1) = 0
    gate_and.set_input_state("A", 1)
    gate_and.set_input_state("B", 1)
    assert wire.signal_value() == 1
    assert gate_not.output_value() == 0


def test_dynamic_re_routing_on_gate_movement(qapp):
    """Test that moving either gate automatically updates wire geometry."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(400.0, 100.0))

    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="A")
    orig_p0 = QPointF(wire._points[0])

    # Move gate_and to new position
    gate_and.setPos(150.0, 220.0)
    # The wire should update its start point to the new output position
    new_s_pos = gate_and.get_output_scene_pos()
    assert abs(wire._points[0].x() - new_s_pos.x()) < 0.1
    assert abs(wire._points[0].y() - new_s_pos.y()) < 0.1
    assert wire._points[0] != orig_p0

    # Move gate_or
    gate_or.setPos(550.0, 300.0)
    new_t_pos = gate_or.get_input_scene_pos("A")
    assert abs(wire._points[-1].x() - new_t_pos.x()) < 0.1
    assert abs(wire._points[-1].y() - new_t_pos.y()) < 0.1


def test_circuit_wire_rendering(qapp):
    """Verify wire boundingRect and QPainter rendering in high and low states."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(300.0, 120.0))
    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="A")

    # Bounding rect
    br = wire.boundingRect()
    assert not br.isEmpty()
    assert br.width() > 100.0
    assert br.height() > 10.0

    # Render Low state on QImage
    img = QImage(600, 400, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#0f172a"))
    painter = QPainter(img)
    wire.paint(painter, None, None)

    # Change to High state and render
    gate_and.set_input_state("A", 1)
    gate_and.set_input_state("B", 1)
    assert wire.signal_value() == 1
    wire.paint(painter, None, None)
    painter.end()


def test_board_circuit_wire_management(qapp):
    """Test LessonBoard integration for creating, listing, and removing wires."""
    board = LessonBoard()
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(400.0, 100.0))
    gate_not = InteractiveLogicGateItem("NOT", center=(700.0, 100.0))
    board.addItem(gate_and)
    board.addItem(gate_or)
    board.addItem(gate_not)

    assert len(board.get_circuit_wires()) == 0

    wire1 = board.connect_logic_gates(gate_and, gate_or, "A")
    wire2 = board.connect_logic_gates(gate_or, gate_not, "A")

    assert len(board.get_circuit_wires()) == 2
    assert wire1 in board.get_circuit_wires()
    assert wire2 in board.get_circuit_wires()

    # Disconnect wire 1
    removed = board.remove_circuit_wire(wire1)
    assert removed is True
    assert len(board.get_circuit_wires()) == 1
    assert wire1 not in board.get_circuit_wires()

    # Clear all wires
    board.clear_circuit_wires()
    assert len(board.get_circuit_wires()) == 0


def test_wire_touchless_air_click(qapp):
    """Test touchless air tapping on the midpoint badge of the wire."""
    gate_and = InteractiveLogicGateItem("AND", center=(100.0, 100.0))
    gate_or = InteractiveLogicGateItem("OR", center=(300.0, 100.0))
    wire = InteractiveCircuitWireItem(gate_and, gate_or, target_pin="A")

    mb = wire._midpoint_badge
    # Air tap directly on midpoint badge
    assert wire.air_click_at(mb.x(), mb.y()) is True
    # Air tap within 10px
    assert wire.air_click_at(mb.x() + 5.0, mb.y() - 5.0) is True
    # Air tap far away
    assert wire.air_click_at(mb.x() + 100.0, mb.y() + 100.0) is False
