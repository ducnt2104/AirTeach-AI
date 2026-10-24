"""
AirTeach AI - Interactive STEM Logic Gate & Circuit Simulator
=============================================================
Provides real-time interactive digital logic gates (AND, OR, NOT, NAND, NOR, XOR)
with clickable/air-tappable input toggles (0/1), live boolean evaluation,
dynamic glowing signal wires, snap connection ports, and interactive Truth Table HUD.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
)


class InteractiveLogicGateItem(QGraphicsObject):
    """
    Interactive STEM Logic Gate with clickable/air-tappable binary input toggles,
    real-time output simulation, truth table visualization, and magnetic snap pins.
    """

    state_changed = Signal(str, int)  # (gate_type, output_value)
    truth_table_toggled = Signal(bool)
    position_changed = Signal()

    def __init__(
        self,
        gate_type: str = "AND",
        center: tuple[float, float] = (960.0, 540.0),
        width: float = 130.0,
        height: float = 75.0,
        input_labels: tuple[str, ...] = ("A", "B"),
        output_label: str = "Y",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(center[0], center[1])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._gate_type = gate_type.upper().strip()
        self._width = float(width)
        self._height = float(height)
        self._output_label = output_label
        self._show_truth_table: bool = False

        # Input pins initialization
        if "NOT" in self._gate_type or "INVERTER" in self._gate_type:
            self._inputs = ["A"]
        else:
            self._inputs = list(input_labels) if input_labels else ["A", "B"]

        self._input_states: dict[str, int] = {pin: 0 for pin in self._inputs}
        self._pin_len: float = 24.0
        self._toggle_radius: float = 12.0
        self._bubble_radius: float = 4.5

        # Colors & Pens
        self._color_body = QColor("#3b82f6")  # Blue accent
        self._color_high = QColor("#10b981")  # Emerald Green (1 / High)
        self._color_low = QColor("#64748b")   # Slate Gray (0 / Low)
        self._bg_brush = QBrush(QColor("#1e293b"))  # Dark slate core

        self.setZValue(15.0)

    # ── Simulation & Boolean Logic ─────────────────────────────────────────────

    def evaluate(self) -> int:
        """Compute the boolean output for current input states."""
        vals = [self._input_states[pin] for pin in self._inputs]
        gt = self._gate_type

        if "NOT" in gt or "INVERTER" in gt:
            return 1 if (vals[0] if vals else 0) == 0 else 0
        elif "AND" in gt and "NAND" not in gt:
            return 1 if all(v == 1 for v in vals) else 0
        elif "NAND" in gt:
            return 0 if all(v == 1 for v in vals) else 1
        elif "OR" in gt and "NOR" not in gt and "XOR" not in gt:
            return 1 if any(v == 1 for v in vals) else 0
        elif "NOR" in gt:
            return 0 if any(v == 1 for v in vals) else 1
        elif "XOR" in gt:
            return 1 if sum(vals) % 2 == 1 else 0
        else:
            return 1 if all(v == 1 for v in vals) else 0

    def output_value(self) -> int:
        return self.evaluate()

    def set_input_state(self, pin_name: str, state: int) -> None:
        """Set a specific input pin state (0 or 1)."""
        if pin_name in self._input_states:
            self._input_states[pin_name] = 1 if state else 0
            self.prepareGeometryChange()
            self.update()
            self.state_changed.emit(self._gate_type, self.evaluate())

    def toggle_input(self, pin_name: str) -> int:
        """Toggle an input pin between 0 and 1."""
        if pin_name in self._input_states:
            new_val = 1 if self._input_states[pin_name] == 0 else 0
            self.set_input_state(pin_name, new_val)
            return new_val
        return 0

    def toggle_truth_table(self) -> bool:
        """Toggle Truth Table HUD overlay visibility."""
        self.prepareGeometryChange()
        self._show_truth_table = not self._show_truth_table
        self.update()
        self.truth_table_toggled.emit(self._show_truth_table)
        return self._show_truth_table

    # ── Geometry & Pin Coordinates ─────────────────────────────────────────────

    def _has_bubble(self) -> bool:
        return any(k in self._gate_type for k in ("NOT", "NAND", "NOR"))

    def _get_input_offsets(self) -> list[tuple[str, float, float]]:
        """Return (pin_name, local_x, local_y) for each input pin toggle center."""
        w_body = self._width - self._pin_len * 2.0
        x_pin_start = -w_body / 2.0 - self._pin_len
        n = len(self._inputs)
        offsets = []
        if n == 1:
            offsets.append((self._inputs[0], x_pin_start - self._toggle_radius, 0.0))
        else:
            spacing = self._height * 0.55 / max(1, n - 1)
            start_y = -self._height * 0.275
            for i, name in enumerate(self._inputs):
                y = start_y + i * spacing
                offsets.append((name, x_pin_start - self._toggle_radius, y))
        return offsets

    def _get_output_offset(self) -> tuple[float, float]:
        """Return (local_x, local_y) for output pin badge center."""
        w_body = self._width - self._pin_len * 2.0
        x_body_end = w_body / 2.0 + (self._bubble_radius * 2.0 if self._has_bubble() else 0.0)
        return (x_body_end + self._pin_len + self._toggle_radius, 0.0)

    def get_input_scene_pos(self, pin_name: str) -> QPointF:
        """Return exact scene coordinates for an input connection port."""
        for name, lx, ly in self._get_input_offsets():
            if name == pin_name:
                return self.mapToScene(QPointF(lx + self._toggle_radius, ly))
        return self.scenePos()

    def get_output_scene_pos(self) -> QPointF:
        """Return exact scene coordinates for the output connection port."""
        lx, ly = self._get_output_offset()
        return self.mapToScene(QPointF(lx - self._toggle_radius, ly))

    def snap_pin_at(self, scene_pos: QPointF, tolerance: float = 28.0) -> Optional[tuple[str, QPointF]]:
        """
        Check if scene_pos is close to any pin.
        Returns (pin_id, snapped_scene_pos) if within tolerance, else None.
        """
        # Check inputs
        for name, lx, ly in self._get_input_offsets():
            pt = self.mapToScene(QPointF(lx + self._toggle_radius, ly))
            if math.hypot(scene_pos.x() - pt.x(), scene_pos.y() - pt.y()) <= tolerance:
                return (name, pt)

        # Check output
        out_pt = self.get_output_scene_pos()
        if math.hypot(scene_pos.x() - out_pt.x(), scene_pos.y() - out_pt.y()) <= tolerance:
            return (self._output_label, out_pt)

        return None

    # ── Bounding Rect & Rendering ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        left = -self._width / 2.0 - self._pin_len - self._toggle_radius * 2.5 - 10.0
        right = self._width / 2.0 + self._pin_len + self._toggle_radius * 2.5 + 10.0
        top = -self._height / 2.0 - 30.0
        bottom = self._height / 2.0 + 30.0

        if self._show_truth_table:
            right += 140.0
            bottom += 40.0

        return QRectF(left, top, right - left, bottom - top)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        gt = self._gate_type
        w_body = self._width - self._pin_len * 2.0 - (self._bubble_radius * 2.0 if self._has_bubble() else 0.0)
        h = self._height
        x_left = -w_body / 2.0
        x_right = x_left + w_body
        cy = 0.0

        # 1. Draw input wires and clickable toggles
        for name, lx, ly in self._get_input_offsets():
            val = self._input_states[name]
            wire_color = self._color_high if val == 1 else self._color_low
            painter.setPen(QPen(wire_color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(lx + self._toggle_radius, ly), QPointF(x_left, ly))

            # Toggle Badge
            painter.setBrush(QBrush(QColor("#065f46" if val == 1 else "#1e293b")))
            painter.setPen(QPen(wire_color, 1.8))
            badge_rect = QRectF(lx - self._toggle_radius, ly - self._toggle_radius, self._toggle_radius * 2, self._toggle_radius * 2)
            painter.drawEllipse(badge_rect)

            # Badge Text (0 or 1)
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff" if val == 1 else "#94a3b8")))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(val))

            # Pin Label (A, B)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.drawText(QRectF(lx - 28, ly - 9, 16, 18), Qt.AlignmentFlag.AlignRight, name)

        # 2. Draw body IEEE shape
        body_path = QPainterPath()
        is_not = "NOT" in gt or "INVERTER" in gt
        is_or = ("OR" in gt or "NOR" in gt or "XOR" in gt) and not is_not
        is_xor = "XOR" in gt

        if is_not:
            body_path.moveTo(x_left, cy - h / 2.0)
            body_path.lineTo(x_right, cy)
            body_path.lineTo(x_left, cy + h / 2.0)
            body_path.closeSubpath()
        elif is_or:
            body_path.moveTo(x_left, cy - h / 2.0)
            body_path.quadTo(x_left + w_body * 0.3, cy, x_left, cy + h / 2.0)
            body_path.quadTo(x_left + w_body * 0.7, cy + h * 0.45, x_right, cy)
            body_path.quadTo(x_left + w_body * 0.7, cy - h * 0.45, x_left, cy - h / 2.0)
        else:
            # AND / NAND
            body_path.moveTo(x_left, cy - h / 2.0)
            body_path.lineTo(x_left + w_body * 0.45, cy - h / 2.0)
            body_path.cubicTo(
                x_right + w_body * 0.15, cy - h / 2.0,
                x_right + w_body * 0.15, cy + h / 2.0,
                x_left + w_body * 0.45, cy + h / 2.0,
            )
            body_path.lineTo(x_left, cy + h / 2.0)
            body_path.closeSubpath()

        # Fill and stroke body
        painter.setBrush(self._bg_brush)
        painter.setPen(QPen(self._color_body, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(body_path)

        # Draw XOR extra input arc
        if is_xor:
            xor_arc = QPainterPath()
            xor_offset = 8.0
            xor_arc.moveTo(x_left - xor_offset, cy - h / 2.0)
            xor_arc.quadTo(x_left - xor_offset + w_body * 0.3, cy, x_left - xor_offset, cy + h / 2.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self._color_body, 2.5, Qt.PenStyle.SolidLine))
            painter.drawPath(xor_arc)

        # Inverter bubble
        if self._has_bubble():
            bx = x_right + self._bubble_radius
            bubble_rect = QRectF(bx - self._bubble_radius, cy - self._bubble_radius, self._bubble_radius * 2, self._bubble_radius * 2)
            painter.setBrush(self._bg_brush)
            painter.setPen(QPen(self._color_body, 2.0))
            painter.drawEllipse(bubble_rect)
            x_pin_out_start = bx + self._bubble_radius
        else:
            x_pin_out_start = x_right

        # Gate type label centered inside body
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#93c5fd")))
        painter.drawText(QRectF(x_left, cy - 10, w_body, 20), Qt.AlignmentFlag.AlignCenter, gt)

        # 3. Output wire & Badge
        out_val = self.evaluate()
        out_color = self._color_high if out_val == 1 else self._color_low
        out_lx, out_ly = self._get_output_offset()

        # Wire
        painter.setPen(QPen(out_color, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x_pin_out_start, cy), QPointF(out_lx - self._toggle_radius, out_ly))

        # Output Badge
        if out_val == 1:
            # Halo glow
            halo_pen = QPen(QColor(16, 185, 129, 60), 6.0)
            painter.setPen(halo_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(out_lx - self._toggle_radius - 2, out_ly - self._toggle_radius - 2, (self._toggle_radius + 2) * 2, (self._toggle_radius + 2) * 2))

        out_badge_rect = QRectF(out_lx - self._toggle_radius, out_ly - self._toggle_radius, self._toggle_radius * 2, self._toggle_radius * 2)
        painter.setBrush(QBrush(QColor("#065f46" if out_val == 1 else "#1e293b")))
        painter.setPen(QPen(out_color, 2.0))
        painter.drawEllipse(out_badge_rect)

        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff" if out_val == 1 else "#94a3b8")))
        painter.drawText(out_badge_rect, Qt.AlignmentFlag.AlignCenter, str(out_val))

        # Output Label (Y)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(out_lx + self._toggle_radius + 6, out_ly - 9, 20, 18), Qt.AlignmentFlag.AlignLeft, self._output_label)

        # 4. Truth Table Button [📊] above gate
        tt_btn_rect = QRectF(-20.0, -h / 2.0 - 24.0, 40.0, 18.0)
        painter.setBrush(QBrush(QColor("#334155")))
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.drawRoundedRect(tt_btn_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(tt_btn_rect, Qt.AlignmentFlag.AlignCenter, "📊 BCL")

        # 5. Truth Table Overlay if expanded
        if self._show_truth_table:
            self._paint_truth_table(painter, out_lx + 35.0, -h / 2.0)

    def _paint_truth_table(self, painter: QPainter, tx: float, ty: float) -> None:
        """Render compact truth table with active state row highlighted."""
        is_single = len(self._inputs) == 1
        num_rows = 2 if is_single else 4
        row_h = 18.0
        header_h = 22.0
        tbl_w = 110.0
        tbl_h = header_h + num_rows * row_h + 8.0

        # Acrylic background
        tbl_rect = QRectF(tx, ty, tbl_w, tbl_h)
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawRoundedRect(tbl_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#93c5fd")))
        painter.drawText(QRectF(tx, ty + 2, tbl_w, 14), Qt.AlignmentFlag.AlignCenter, f"{self._gate_type} Truth Table")

        # Header
        col_hdr = "A   Y" if is_single else "A  B   Y"
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(tx, ty + 16, tbl_w, 14), Qt.AlignmentFlag.AlignCenter, col_hdr)

        # Rows
        combinations = [(0,), (1,)] if is_single else [(0, 0), (0, 1), (1, 0), (1, 1)]
        current_vals = tuple(self._input_states[pin] for pin in self._inputs)

        for i, comb in enumerate(combinations):
            ry = ty + header_h + 4.0 + i * row_h
            r_rect = QRectF(tx + 4, ry, tbl_w - 8, row_h)

            # Evaluate this row's expected output
            if is_single:
                row_out = 1 if (comb[0] == 0 and ("NOT" in self._gate_type or "INVERTER" in self._gate_type)) else (0 if "NOT" in self._gate_type else comb[0])
                row_txt = f"{comb[0]}   {row_out}"
            else:
                a, b = comb
                if "NAND" in self._gate_type:
                    row_out = 0 if (a == 1 and b == 1) else 1
                elif "AND" in self._gate_type:
                    row_out = 1 if (a == 1 and b == 1) else 0
                elif "NOR" in self._gate_type:
                    row_out = 0 if (a == 1 or b == 1) else 1
                elif "XOR" in self._gate_type:
                    row_out = 1 if (a != b) else 0
                else:  # OR
                    row_out = 1 if (a == 1 or b == 1) else 0
                row_txt = f"{a}  {b}   {row_out}"

            # Highlight active combination
            if comb == current_vals:
                painter.setBrush(QBrush(QColor(16, 185, 129, 65)))
                painter.setPen(QPen(QColor("#10b981"), 1.0))
                painter.drawRoundedRect(r_rect, 3.0, 3.0)
                painter.setPen(QPen(QColor("#34d399")))
                row_txt = "▶ " + row_txt
            else:
                painter.setPen(QPen(QColor("#94a3b8")))
                row_txt = "  " + row_txt

            painter.setFont(QFont("Consolas", 8))
            painter.drawText(r_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, row_txt)

    # ── Mouse & Air Gesture Clicks ─────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Truth table button
        h = self._height
        tt_btn_rect = QRectF(-20.0, -h / 2.0 - 24.0, 40.0, 18.0)
        if tt_btn_rect.contains(pos):
            self.toggle_truth_table()
            event.accept()
            return

        # Check Input Toggle badges
        for name, lx, ly in self._get_input_offsets():
            dist = math.hypot(pos.x() - lx, pos.y() - ly)
            if dist <= self._toggle_radius * 1.5:
                self.toggle_input(name)
                event.accept()
                return

        super().mousePressEvent(event)

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Allows touchless air tapping on input badges or truth table button.
        Returns True if a component was toggled.
        """
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))

        # Check Truth table button
        h = self._height
        tt_btn_rect = QRectF(-20.0, -h / 2.0 - 24.0, 40.0, 18.0)
        if tt_btn_rect.contains(local_pos):
            self.toggle_truth_table()
            return True

        # Check Input Toggle badges
        for name, lx, ly in self._get_input_offsets():
            dist = math.hypot(local_pos.x() - lx, local_pos.y() - ly)
            if dist <= self._toggle_radius * 1.8:
                self.toggle_input(name)
                return True

        return False

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed.emit()
        return super().itemChange(change, value)

