# src/ui/interactive_logic_circuits.py
"""
Interactive Modern Electronics & Digital Computing: Logic Gates (AND, OR, NOT, NAND, NOR, XOR, XNOR),
Truth Tables, Boolean Algebra, Half/Full Adder & Karnaugh Map (K-Map) Simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsObject,
    QGraphicsItem,
)


@dataclass(frozen=True)
class CircuitPreset:
    """Represents a digital logic scenario."""
    code: str
    title_vi: str
    circuit_name: str
    category: str                 # "Basic Gate", "Universal Gate", "Arithmetic", "Sequential"
    boolean_formula: str          # e.g. "Y = A ⊕ B"
    inputs_count: int             # 1, 2, or 3
    input_names: list[str]        # ["A", "B"] or ["A", "B", "Cin"]
    output_names: list[str]       # ["Y"] or ["Sum", "Cout"]
    truth_table: list[tuple[tuple[int, ...], tuple[int, ...]]] # [((A, B), (Y,)), ...]
    description: str


CIRCUIT_PRESETS: dict[str, CircuitPreset] = {
    "XOR_PARITY_GATE": CircuitPreset(
        code="XOR_PARITY_GATE",
        title_vi="Cổng XOR (Exclusive OR - Phép Cộng Nhị Phân Không Nhớ)",
        circuit_name="2-Input XOR Gate (74HC86)",
        category="Basic Gate",
        boolean_formula="Y = A ⊕ B = A·B̄ + Ā·B",
        inputs_count=2,
        input_names=["A", "B"],
        output_names=["Y (XOR)"],
        truth_table=[
            ((0, 0), (0,)),
            ((0, 1), (1,)),
            ((1, 0), (1,)),
            ((1, 1), (0,)),
        ],
        description="Đầu ra bằng 1 khi hai đầu vào khác nhau (số bit 1 lẻ). Là khối lõi của mạch cộng nhị phân và kiểm tra chẵn lẻ.",
    ),
    "HALF_ADDER_CIRCUIT": CircuitPreset(
        code="HALF_ADDER_CIRCUIT",
        title_vi="Mạch Nửa Cộng (Half Adder - XOR + AND)",
        circuit_name="Binary Half Adder",
        category="Arithmetic",
        boolean_formula="Sum = A ⊕ B  |  C_out = A · B",
        inputs_count=2,
        input_names=["A", "B"],
        output_names=["Sum", "C_out"],
        truth_table=[
            ((0, 0), (0, 0)),
            ((0, 1), (1, 0)),
            ((1, 0), (1, 0)),
            ((1, 1), (0, 1)),
        ],
        description="Mạch cộng 2 bit nhị phân cơ bản: bit Tổng S tạo bởi cổng XOR và bit Nhớ C_out tạo bởi cổng AND.",
    ),
    "FULL_ADDER_CIRCUIT": CircuitPreset(
        code="FULL_ADDER_CIRCUIT",
        title_vi="Mạch Cộng Đầy Đủ (Full Adder - 3 Bit)",
        circuit_name="1-Bit Full Adder (74HC283 Core)",
        category="Arithmetic",
        boolean_formula="Sum = A ⊕ B ⊕ C_in  |  C_out = (A·B) + (C_in·(A⊕B))",
        inputs_count=3,
        input_names=["A", "B", "C_in"],
        output_names=["Sum", "C_out"],
        truth_table=[
            ((0, 0, 0), (0, 0)),
            ((0, 0, 1), (1, 0)),
            ((0, 1, 0), (1, 0)),
            ((0, 1, 1), (0, 1)),
            ((1, 0, 0), (1, 0)),
            ((1, 0, 1), (0, 1)),
            ((1, 1, 0), (0, 1)),
            ((1, 1, 1), (1, 1)),
        ],
        description="Mạch cộng nhị phân hoàn chỉnh xử lý cả số nhớ C_in từ tầng trước, cho phép ghép nối thành ALU 32-bit / 64-bit.",
    ),
    "NAND_UNIVERSAL_GATE": CircuitPreset(
        code="NAND_UNIVERSAL_GATE",
        title_vi="Cổng Vạn Năng NAND (Universal Gate - 74HC00)",
        circuit_name="2-Input NAND Gate",
        category="Universal Gate",
        boolean_formula="Y = (A · B)̄ = Ā + B̄ (De Morgan)",
        inputs_count=2,
        input_names=["A", "B"],
        output_names=["Y (NAND)"],
        truth_table=[
            ((0, 0), (1,)),
            ((0, 1), (1,)),
            ((1, 0), (1,)),
            ((1, 1), (0,)),
        ],
        description="Cổng logic vạn năng: Mọi hàm Boolean và mọi mạch máy tính (CPU/RAM) đều có thể xây dựng hoàn toàn chỉ bằng cổng NAND.",
    ),
    "SR_LATCH_BISTABLE": CircuitPreset(
        code="SR_LATCH_BISTABLE",
        title_vi="Chốt SR (SR Latch - Bộ Nhớ 1-Bit Phần Tử Nhớ)",
        circuit_name="NOR-Gate Cross-Coupled SR Latch",
        category="Sequential",
        boolean_formula="Q_next = S + R̄·Q  (S·R = 0 bắt buộc)",
        inputs_count=2,
        input_names=["Set (S)", "Reset (R)"],
        output_names=["Q", "Q̄"],
        truth_table=[
            ((0, 0), (0, 1)), # Giữ trạng thái (Hold)
            ((0, 1), (0, 1)), # Reset về 0
            ((1, 0), (1, 0)), # Set lên 1
            ((1, 1), (0, 0)), # Cấm (Invalid)
        ],
        description="Phần tử nhớ bất biến bistable cơ bản nhất cấu thành thanh ghi (Register) và SRAM trong vi xử lý hiện đại.",
    ),
}


class InteractiveLogicCircuitsItem(QGraphicsObject):
    """
    Pedagogical Interactive Logic Gates, Truth Tables, Boolean Algebra,
    Adders & Digital Computing Circuit Simulator.
    """

    circuit_computed = Signal(dict)
    stamp_requested = Signal(dict)

    CARD_W = 680.0
    CARD_H = 380.0

    def __init__(self, parent: Optional[QGraphicsItem] = None, pos: tuple[float, float] = (80.0, 40.0)):
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(150.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)

        self._active_preset_code: str = "HALF_ADDER_CIRCUIT"
        self._preset: CircuitPreset = CIRCUIT_PRESETS[self._active_preset_code]

        # Live Input bit states: [A, B, Cin]
        self._input_states: list[int] = [1, 0, 0]

        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._layout_buttons()

    # ------------------------------------------------------------------
    # Mathematical & Boolean Core Calculations
    # ------------------------------------------------------------------

    @property
    def current_inputs(self) -> tuple[int, ...]:
        return tuple(self._input_states[:self._preset.inputs_count])

    @property
    def current_outputs(self) -> tuple[int, ...]:
        """Calculates exact digital circuit outputs based on current inputs."""
        inputs = self.current_inputs
        code = self._active_preset_code

        if code == "XOR_PARITY_GATE":
            a, b = inputs[0], inputs[1]
            return (a ^ b,)

        elif code == "HALF_ADDER_CIRCUIT":
            a, b = inputs[0], inputs[1]
            s = a ^ b
            cout = a & b
            return (s, cout)

        elif code == "FULL_ADDER_CIRCUIT":
            a, b, cin = inputs[0], inputs[1], inputs[2]
            s = a ^ b ^ cin
            cout = (a & b) | (cin & (a ^ b))
            return (s, cout)

        elif code == "NAND_UNIVERSAL_GATE":
            a, b = inputs[0], inputs[1]
            y = 1 if not (a and b) else 0
            return (y,)

        elif code == "SR_LATCH_BISTABLE":
            s, r = inputs[0], inputs[1]
            if s == 1 and r == 0:
                return (1, 0)
            elif s == 0 and r == 1:
                return (0, 1)
            elif s == 1 and r == 1:
                return (0, 0) # Invalid state
            else:
                return (1, 0) # Default hold state

        return (0,)

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches digital logic preset."""
        if preset_code in CIRCUIT_PRESETS:
            self._active_preset_code = preset_code
            self._preset = CIRCUIT_PRESETS[preset_code]
            self._emit_computed_values()
            self._layout_buttons()
            self.update()

    def set_circuit_preset(self, preset_code: str) -> None:
        """Alias for set_preset."""
        self.set_preset(preset_code)

    def toggle_input(self, index: int) -> None:
        """Toggles input bit at index between 0 and 1."""
        if 0 <= index < len(self._input_states):
            self._input_states[index] = 1 - self._input_states[index]
            self._emit_computed_values()
            self.update()

    def set_input(self, index: int, val: int) -> None:
        """Sets input bit at index."""
        if 0 <= index < len(self._input_states):
            self._input_states[index] = 1 if val else 0
            self._emit_computed_values()
            self.update()

    def cycle_preset(self) -> None:
        """Cycles to next circuit preset."""
        self._handle_button_click("preset_next")

    def reset_inputs(self) -> None:
        """Resets all inputs to zero."""
        self._input_states = [0, 0, 0]
        self._emit_computed_values()
        self.update()

    def _emit_computed_values(self) -> None:
        data = {
            "preset_code": self._active_preset_code,
            "title_vi": self._preset.title_vi,
            "circuit_name": self._preset.circuit_name,
            "boolean_formula": self._preset.boolean_formula,
            "inputs": self.current_inputs,
            "outputs": self.current_outputs,
        }
        self.circuit_computed.emit(data)

    # ------------------------------------------------------------------
    # UI Layout & Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        ctrl_x = 400.0
        row1_y = 265.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["toggle_a"] = QRectF(ctrl_x + 104.0, row1_y, 75.0, btn_h)
        self._buttons["toggle_b"] = QRectF(ctrl_x + 184.0, row1_y, 75.0, btn_h)

        row2_y = 298.0
        self._buttons["toggle_c"] = QRectF(ctrl_x, row2_y, 110.0, btn_h)
        self._buttons["reset_all"] = QRectF(ctrl_x + 118.0, row2_y, 141.0, btn_h)

        row3_y = 336.0
        self._buttons["stamp"] = QRectF(ctrl_x, row3_y, 259.0, 28.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(15, 23, 42))
        grad.setColorAt(1.0, QColor(10, 15, 30))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(51, 65, 85), 1.5))
        painter.drawRoundedRect(self.boundingRect(), 10.0, 10.0)

        # 2. Header
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(16.0, 24.0, "🔌 ĐIỆN TỬ SỐ & CỔNG LOGIC (DIGITAL LOGIC & BOOLEAN ALGEBRA)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        sub_text = f"{self._preset.title_vi} | Công thức: {self._preset.boolean_formula}"
        painter.drawText(16.0, 42.0, sub_text)

        # 3. Left Panel: Interactive Schematic Diagram
        schem_rect = QRectF(16.0, 52.0, 365.0, 312.0)
        painter.setBrush(QBrush(QColor(10, 15, 26)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(schem_rect, 6.0, 6.0)

        self._paint_schematic(painter, schem_rect)

        # 4. Right Panel: Dynamic Truth Table
        self._paint_truth_table(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_schematic(self, painter: QPainter, rect: QRectF) -> None:
        p = self._preset
        inputs = self.current_inputs
        outputs = self.current_outputs

        # Paint Interactive Input Switches on Left
        sw_x = rect.x() + 25.0
        start_y = rect.y() + 60.0
        y_gap = 60.0 if p.inputs_count <= 2 else 45.0

        for i in range(p.inputs_count):
            iy = start_y + i * y_gap
            val = inputs[i]
            val_col = QColor(34, 197, 94) if val == 1 else QColor(100, 116, 139)

            # Switch box
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.setPen(QPen(val_col, 1.5))
            sw_box = QRectF(sw_x - 14.0, iy - 14.0, 28.0, 28.0)
            painter.drawRoundedRect(sw_box, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(val_col)
            painter.drawText(sw_box, Qt.AlignmentFlag.AlignCenter, str(val))

            # Input label
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor(226, 232, 240))
            painter.drawText(sw_x - 20.0, iy - 18.0, p.input_names[i])

            # Wire from switch to center gate
            painter.setPen(QPen(val_col, 2.0))
            painter.drawLine(sw_x + 14.0, iy, rect.x() + 150.0, iy)

        # Center Logic Gate Box / Symbol
        gate_rect = QRectF(rect.x() + 150.0, rect.y() + 50.0, 100.0, 160.0)
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawRoundedRect(gate_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(gate_rect, Qt.AlignmentFlag.AlignCenter, p.category + "\n" + p.circuit_name.split()[0])

        # Output Wires & LEDs
        out_x = rect.x() + 310.0
        out_start_y = rect.y() + 85.0
        out_gap = 70.0 if len(outputs) > 1 else 0.0

        for j in range(len(outputs)):
            oy = out_start_y + j * out_gap
            out_val = outputs[j]
            out_col = QColor(34, 197, 94) if out_val == 1 else QColor(100, 116, 139)

            # Wire from gate to LED
            painter.setPen(QPen(out_col, 2.0))
            painter.drawLine(rect.x() + 250.0, oy, out_x - 14.0, oy)

            # LED Indicator
            led_grad = QRadialGradient(out_x, oy, 14.0)
            if out_val == 1:
                led_grad.setColorAt(0.0, QColor(255, 255, 255))
                led_grad.setColorAt(0.5, QColor(34, 197, 94))
                led_grad.setColorAt(1.0, QColor(22, 101, 52))
            else:
                led_grad.setColorAt(0.0, QColor(71, 85, 105))
                led_grad.setColorAt(1.0, QColor(30, 41, 59))
            painter.setBrush(QBrush(led_grad))
            painter.setPen(QPen(QColor(255, 255, 255), 1.2 if out_val == 1 else 0.5))
            painter.drawEllipse(QPointF(out_x, oy), 12.0, 12.0)

            # Output Bit Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(out_col)
            painter.drawText(out_x - 5.0, oy + 4.0, str(out_val))

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor(248, 250, 252))
            painter.drawText(out_x - 20.0, oy + 26.0, p.output_names[j])

        # Bottom Educational Explanation inside Schematic
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(QRectF(rect.x() + 10.0, rect.y() + 240.0, rect.width() - 20.0, 65.0),
                         Qt.TextFlag.TextWordWrap, p.description)

    def _paint_truth_table(self, painter: QPainter) -> None:
        tt_x = 400.0
        tt_y = 56.0
        tt_w = 260.0
        tt_h = 195.0

        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(QRectF(tt_x, tt_y, tt_w, tt_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(tt_x + 10.0, tt_y + 18.0, "📋 BẢNG CHÂN TRỊ (TRUTH TABLE)")

        p = self._preset
        rows = p.truth_table
        cur_in = self.current_inputs

        # Header row
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(148, 163, 184))
        in_header = " | ".join(p.input_names)
        out_header = " | ".join(p.output_names)
        painter.drawText(tt_x + 12.0, tt_y + 36.0, f"{in_header}  ⟹  {out_header}")

        # Horizontal separator
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(tt_x + 10.0, tt_y + 42.0, tt_x + tt_w - 10.0, tt_y + 42.0)

        # Draw Table Rows
        row_y = tt_y + 58.0
        row_gap = 16.5 if len(rows) <= 4 else 15.0

        for r_in, r_out in rows:
            is_active_row = (r_in == cur_in)
            if is_active_row:
                # Highlight active state row
                painter.setBrush(QBrush(QColor(251, 191, 36, 40)))
                painter.setPen(QPen(QColor(251, 191, 36, 150), 1.0))
                painter.drawRoundedRect(QRectF(tt_x + 8.0, row_y - 12.0, tt_w - 16.0, 16.0), 3.0, 3.0)

            in_str = "    ".join(str(b) for b in r_in)
            out_str = "    ".join(str(b) for b in r_out)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active_row else QFont.Weight.Normal))
            painter.setPen(QColor(251, 191, 36) if is_active_row else QColor(226, 232, 240))
            painter.drawText(tt_x + 20.0, row_y, in_str)
            painter.drawText(tt_x + 140.0, row_y, f"⟹   {out_str}")

            row_y += row_gap

    def _paint_buttons(self, painter: QPainter) -> None:
        btn_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "toggle_a": f"Bit A: {self._input_states[0]}",
            "toggle_b": f"Bit B: {self._input_states[1]}",
            "toggle_c": f"Bit Cin: {self._input_states[2]}",
            "reset_all": "🔄 Về 0 (Clear)",
            "stamp": "📌 Đóng Dấu Bài Học (Stamp Diagram)",
        }

        for key, rect in self._buttons.items():
            is_hovered = (self._hovered_btn == key)
            if key == "stamp":
                bg_col = QColor(14, 165, 233) if not is_hovered else QColor(56, 189, 248)
                txt_col = QColor(255, 255, 255)
            elif key.startswith("toggle_"):
                bg_col = QColor(30, 41, 59) if not is_hovered else QColor(51, 65, 85)
                txt_col = QColor(34, 197, 94)
            else:
                bg_col = QColor(30, 41, 59) if not is_hovered else QColor(51, 65, 85)
                txt_col = QColor(226, 232, 240)

            painter.setBrush(QBrush(bg_col))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.setPen(txt_col)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, btn_labels.get(key, key))

    # ------------------------------------------------------------------
    # Mouse Interaction
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_h = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_h != self._hovered_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _handle_button_click(self, btn_key: str) -> None:
        preset_keys = list(CIRCUIT_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "toggle_a":
            self.toggle_input(0)
        elif btn_key == "toggle_b":
            self.toggle_input(1)
        elif btn_key == "toggle_c":
            self.toggle_input(2)
        elif btn_key == "reset_all":
            self.reset_inputs()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "title_vi": self._preset.title_vi,
            "circuit_name": self._preset.circuit_name,
            "boolean_formula": self._preset.boolean_formula,
            "inputs": self.current_inputs,
            "outputs": self.current_outputs,
            "input_names": self._preset.input_names,
            "output_names": self._preset.output_names,
            "description": self._preset.description,
        }
        self.stamp_requested.emit(data)
