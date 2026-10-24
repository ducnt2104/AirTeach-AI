# tests/test_interactive_logic_circuits.py
"""
Unit and integration tests for InteractiveLogicCircuitsItem,
InformaticsTools.create_logic_circuits_diagram, and LessonBoard integration.
"""

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF

from src.ui.interactive_logic_circuits import (
    InteractiveLogicCircuitsItem,
    CircuitPreset,
    CIRCUIT_PRESETS,
)
from src.ui.stem_tools import InformaticsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveLogicCircuits:
    def test_circuit_presets_validity(self, qapp):
        """Verify all digital logic presets conform to Boolean logic standards."""
        assert len(CIRCUIT_PRESETS) == 5
        assert "XOR_PARITY_GATE" in CIRCUIT_PRESETS
        assert "HALF_ADDER_CIRCUIT" in CIRCUIT_PRESETS
        assert "FULL_ADDER_CIRCUIT" in CIRCUIT_PRESETS
        assert "NAND_UNIVERSAL_GATE" in CIRCUIT_PRESETS
        assert "SR_LATCH_BISTABLE" in CIRCUIT_PRESETS

        for code, preset in CIRCUIT_PRESETS.items():
            assert preset.inputs_count in (1, 2, 3)
            assert len(preset.input_names) == preset.inputs_count
            assert len(preset.output_names) >= 1
            assert len(preset.truth_table) == (2 ** preset.inputs_count)
            assert len(preset.boolean_formula) > 0

    def test_half_adder_boolean_logic(self, qapp):
        """Verify Binary Half Adder: Sum = A ^ B, Cout = A & B."""
        item = InteractiveLogicCircuitsItem()
        item.set_circuit_preset("HALF_ADDER_CIRCUIT")

        # Test all 4 combinations
        table = [
            ((0, 0), (0, 0)),
            ((0, 1), (1, 0)),
            ((1, 0), (1, 0)),
            ((1, 1), (0, 1)),
        ]
        for (a, b), (exp_s, exp_c) in table:
            item.set_input(0, a)
            item.set_input(1, b)
            outputs = item.current_outputs
            assert outputs == (exp_s, exp_c)

    def test_full_adder_boolean_logic(self, qapp):
        """Verify 1-Bit Full Adder with Carry: Sum = A^B^Cin, Cout = (A&B)|(Cin&(A^B))."""
        item = InteractiveLogicCircuitsItem()
        item.set_circuit_preset("FULL_ADDER_CIRCUIT")

        # Test all 8 combinations
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    item.set_input(0, a)
                    item.set_input(1, b)
                    item.set_input(2, cin)

                    expected_sum = a ^ b ^ cin
                    expected_cout = (a & b) | (cin & (a ^ b))

                    outputs = item.current_outputs
                    assert outputs == (expected_sum, expected_cout)

    def test_nand_universal_gate(self, qapp):
        """Verify NAND gate: Y = NOT(A AND B)."""
        item = InteractiveLogicCircuitsItem()
        item.set_circuit_preset("NAND_UNIVERSAL_GATE")

        item.set_input(0, 0)
        item.set_input(1, 0)
        assert item.current_outputs == (1,)

        item.set_input(0, 0)
        item.set_input(1, 1)
        assert item.current_outputs == (1,)

        item.set_input(0, 1)
        item.set_input(1, 0)
        assert item.current_outputs == (1,)

        item.set_input(0, 1)
        item.set_input(1, 1)
        assert item.current_outputs == (0,)

    def test_sr_latch_sequential_logic(self, qapp):
        """Verify SR Latch: Set, Reset, and Invalid states."""
        item = InteractiveLogicCircuitsItem()
        item.set_circuit_preset("SR_LATCH_BISTABLE")

        # Set: S=1, R=0 => Q=1, Q_bar=0
        item.set_input(0, 1)
        item.set_input(1, 0)
        assert item.current_outputs == (1, 0)

        # Reset: S=0, R=1 => Q=0, Q_bar=1
        item.set_input(0, 0)
        item.set_input(1, 1)
        assert item.current_outputs == (0, 1)

        # Invalid: S=1, R=1 => Q=0, Q_bar=0
        item.set_input(0, 1)
        item.set_input(1, 1)
        assert item.current_outputs == (0, 0)

    def test_interactive_controls_and_buttons(self, qapp):
        """Verify input toggle buttons, clear reset, and cycling presets."""
        item = InteractiveLogicCircuitsItem()
        item.set_circuit_preset("HALF_ADDER_CIRCUIT")

        item.set_input(0, 0)
        item.toggle_input(0)
        assert item.current_inputs[0] == 1

        item.toggle_input(0)
        assert item.current_inputs[0] == 0

        # Reset all
        item.set_input(0, 1)
        item.set_input(1, 1)
        item.reset_inputs()
        assert item.current_inputs == (0, 0)

        # Cycle preset
        orig_code = item._active_preset_code
        item.cycle_preset()
        assert item._active_preset_code != orig_code

    def test_stamping_and_vector_diagram(self, qapp):
        """Verify stamp request produces complete vector diagram items."""
        item = InteractiveLogicCircuitsItem()
        stamped_data = {}

        def on_stamp(data):
            stamped_data.update(data)

        item.stamp_requested.connect(on_stamp)
        item.request_stamp()

        assert "preset_code" in stamped_data
        assert "boolean_formula" in stamped_data
        assert "inputs" in stamped_data
        assert "outputs" in stamped_data

        items = InformaticsTools.create_logic_circuits_diagram(
            center=(400.0, 300.0),
            preset_code=stamped_data["preset_code"],
            title_vi=stamped_data["title_vi"],
            circuit_name=stamped_data["circuit_name"],
            boolean_formula=stamped_data["boolean_formula"],
            inputs=stamped_data["inputs"],
            outputs=stamped_data["outputs"],
            input_names=stamped_data["input_names"],
            output_names=stamped_data["output_names"],
            description=stamped_data["description"],
        )

        assert len(items) >= 15
        has_title = any(
            getattr(itm, "toPlainText", lambda: "")().startswith("🔌 ĐIỆN TỬ SỐ")
            for itm in items
        )
        assert has_title

    def test_lesson_board_integration(self, qapp):
        """Verify LessonBoard visibility toggling and stamping lifecycle."""
        board = LessonBoard()

        assert not board.is_logic_circuits_visible()
        item = board.show_logic_circuits(True, pos=(150.0, 80.0))
        assert item is not None
        assert board.is_logic_circuits_visible()
        assert board.get_logic_circuits() is item

        # Toggle visibility
        is_vis = board.toggle_logic_circuits()
        assert not is_vis
        assert not board.is_logic_circuits_visible()

        # Re-show
        is_vis = board.toggle_logic_circuits()
        assert is_vis
        assert board.is_logic_circuits_visible()

        # Stamp onto board
        initial_count = len(board.items())
        item.request_stamp()
        assert len(board.items()) > initial_count
