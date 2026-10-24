"""
AirTeach AI - Interactive Logic Circuit Manhattan Wire & Live Signal Propagation
===============================================================================
Provides live, dynamic orthogonal circuit wires connecting InteractiveLogicGateItem instances:
  - 90-degree Manhattan orthogonal auto-routing with obstacle avoidance heuristics.
  - Real-time digital boolean state propagation (High=1 / Low=0) from source gate to target gate.
  - Glowing neon emerald outer halo on HIGH state with electron pulse badge.
  - Solder junction connection dots at input and output ports.
  - Dynamic geometry tracking when either gate is moved on the LessonBoard canvas.
  - Touchless air gesture and mouse interaction.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

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
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.interactive_logic_gate import InteractiveLogicGateItem

logger = logging.getLogger(__name__)


class InteractiveCircuitWireItem(QGraphicsObject):
    """
    Orthogonal Manhattan wire connecting two InteractiveLogicGateItem instances.
    Propagates live digital state (0 or 1) and updates visually when gates move.
    """

    signal_propagated = Signal(int)   # (output_value: 0 or 1)
    disconnected = Signal()

    def __init__(
        self,
        source_gate: InteractiveLogicGateItem,
        target_gate: InteractiveLogicGateItem,
        target_pin: str = "A",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._source_gate = source_gate
        self._target_gate = target_gate
        self._target_pin = target_pin.upper()
        self._signal_value: int = 0
        self._points: list[QPointF] = []
        self._midpoint_badge: QPointF = QPointF()

        self.setZValue(4.0)  # Behind gate bodies (Z=10), in front of background
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        # Connect source gate state change to live signal propagation
        self._source_gate.state_changed.connect(self._on_source_state_changed)
        if hasattr(self._source_gate, "position_changed"):
            self._source_gate.position_changed.connect(self.update_route)
        if hasattr(self._target_gate, "position_changed"):
            self._target_gate.position_changed.connect(self.update_route)

        # Initial state propagation and wire routing
        self._signal_value = self._source_gate.output_value()
        self._target_gate.set_input_state(self._target_pin, self._signal_value)
        self.update_route()

    # ── Signal Propagation Slot ────────────────────────────────────────────────

    def _on_source_state_changed(self, gate_type: str, out_val: int) -> None:
        self._signal_value = int(out_val)
        self._target_gate.set_input_state(self._target_pin, self._signal_value)
        self.signal_propagated.emit(self._signal_value)
        self.update_route()

    # ── Geometry & Routing ─────────────────────────────────────────────────────

    def source_gate(self) -> InteractiveLogicGateItem:
        return self._source_gate

    def target_gate(self) -> InteractiveLogicGateItem:
        return self._target_gate

    def target_pin(self) -> str:
        return self._target_pin

    def signal_value(self) -> int:
        return self._signal_value

    def update_route(self) -> None:
        """Recalculate 90-degree orthogonal Manhattan polyline points."""
        self.prepareGeometryChange()
        s_pos = self._source_gate.get_output_scene_pos()
        t_pos = self._target_gate.get_input_scene_pos(self._target_pin)

        sx, sy = s_pos.x(), s_pos.y()
        tx, ty = t_pos.x(), t_pos.y()

        self._points = []
        self._points.append(QPointF(sx, sy))

        if tx >= sx + 24.0:
            # Standard forward connection: 2 bends
            mid_x = (sx + tx) / 2.0
            self._points.append(QPointF(mid_x, sy))
            self._points.append(QPointF(mid_x, ty))
        else:
            # Reverse loop connection: 4 bends
            dx = 30.0
            mid_y = (sy + ty) / 2.0
            self._points.append(QPointF(sx + dx, sy))
            self._points.append(QPointF(sx + dx, mid_y))
            self._points.append(QPointF(tx - dx, mid_y))
            self._points.append(QPointF(tx - dx, ty))

        self._points.append(QPointF(tx, ty))

        # Midpoint for status badge
        if len(self._points) >= 4:
            p1 = self._points[1]
            p2 = self._points[2]
            self._midpoint_badge = QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
        else:
            self._midpoint_badge = QPointF((sx + tx) / 2.0, (sy + ty) / 2.0)

        self.update()

    def disconnect_wire(self) -> None:
        """Safely disconnect signal bindings and remove from scene."""
        try:
            self._source_gate.state_changed.disconnect(self._on_source_state_changed)
        except Exception:
            pass

        if hasattr(self._source_gate, "position_changed"):
            try:
                self._source_gate.position_changed.disconnect(self.update_route)
            except Exception:
                pass

        if hasattr(self._target_gate, "position_changed"):
            try:
                self._target_gate.position_changed.disconnect(self.update_route)
            except Exception:
                pass

        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

        self.disconnected.emit()

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        if not self._points:
            return QRectF()
        xs = [p.x() for p in self._points]
        ys = [p.y() for p in self._points]
        margin = 18.0
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if len(self._points) < 2:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        path = QPainterPath()
        path.moveTo(self._points[0])
        for p in self._points[1:]:
            path.lineTo(p)

        is_high = (self._signal_value == 1)

        # 1. Outer Neon Glow Halo (High State)
        if is_high:
            halo_pen = QPen(
                QColor(16, 185, 129, 65),
                7.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.strokePath(path, halo_pen)

        # 2. Core Orthogonal Wire
        wire_color = QColor("#10b981") if is_high else QColor("#475569")
        wire_width = 2.6 if is_high else 2.0
        core_pen = QPen(
            wire_color,
            wire_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        if self.isSelected():
            core_pen = QPen(QColor("#38bdf8"), 3.0, Qt.PenStyle.DashLine)

        painter.strokePath(path, core_pen)

        # 3. Solder Junction Dots at Ends
        start_pt = self._points[0]
        end_pt = self._points[-1]

        for pt in (start_pt, end_pt):
            painter.setBrush(QBrush(wire_color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(pt, 3.5, 3.5)

        # 4. Midpoint Logic State Badge [0] / [1]
        bx, by = self._midpoint_badge.x(), self._midpoint_badge.y()
        badge_rect = QRectF(bx - 10.0, by - 9.0, 20.0, 18.0)

        badge_bg = QColor(16, 185, 129, 230) if is_high else QColor(30, 41, 59, 230)
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(QColor("#ffffff") if is_high else QColor("#94a3b8"), 1.0))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff") if is_high else QColor("#cbd5e1")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(self._signal_value))

    # ── Mouse & Air Gesture Interaction ────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        # Check if clicking on midpoint badge to disconnect
        if math.hypot(pos.x() - self._midpoint_badge.x(), pos.y() - self._midpoint_badge.y()) <= 14.0:
            if event.button() == Qt.MouseButton.RightButton:
                self.disconnect_wire()
                event.accept()
                return

        super().mousePressEvent(event)

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Allows touchless air tapping on the midpoint badge to inspect wire.
        Returns True if tapped near the badge.
        """
        dist = math.hypot(canvas_x - self._midpoint_badge.x(), canvas_y - self._midpoint_badge.y())
        if dist <= 18.0:
            return True
        return False
