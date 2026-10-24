"""
AirTeach AI - Interactive Computer Science Finite State Machine (FSM) & Automata (DFA/NFA) Simulator
====================================================================================================
Provides an authentic computer science and formal language theory simulation:
  - Formal 5-tuple M = <Q, Σ, δ, q₀, F>:
      - Q: Finite set of internal states (e.g., {q₀, q₁, q₂}).
      - Σ: Input tape alphabet (e.g., {0, 1} or {a, b}).
      - δ: State transition function δ: Q × Σ → Q.
      - q₀ ∈ Q: Initial start state (marked with entry arrow ➔).
      - F ⊆ Q: Set of final / accepting states (marked with concentric double circle ⊚).
  - Authentic Computing Presets:
      1. EVEN_ZEROS: Recognizes binary strings with an even number of '0's (L = {w ∈ {0,1}* | |w|₀ is even}).
      2. MOD_3: Binary modulo 3 counter (L = {w ∈ {0,1}* | value(w) mod 3 == 0}).
      3. ENDS_WITH_ABB: Substring pattern matching lexer (L = {w ∈ {a,b}* | w ends with 'abb'}).
      4. ODD_PARITY: Telecommunications parity bit error detector (L = {w ∈ {0,1}* | |w|₁ is odd}).
  - Interactive Step-by-Step Simulation:
      - Visual Input Tape with animated reading head pointer (▲).
      - Step forward / backward through input stream.
      - Auto-run animation with glowing state transitions and traveling tokens.
      - Real-time verdict HUD: "ĐANG XỬ LÝ" / "CHẤP NHẬN (ACCEPTED)" / "TỪ CHỐI (REJECTED)".
  - Air gestures (air tap / air pinch) and permanent vector stamping to LessonBoard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.ui.stem_tools import InformaticsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass
class AutomatonState:
    id: str
    label: str
    x: float
    y: float
    is_start: bool = False
    is_accept: bool = False


@dataclass
class AutomatonTransition:
    source: str
    symbol: str
    target: str
    curvature: float = 0.0  # Perpendicular curve offset in pixels for bidirectional edges


PRESETS: Dict[str, Dict[str, Any]] = {
    "EVEN_ZEROS": {
        "title": "Số lượng bit '0' là Chẵn",
        "desc": "L = {w ∈ {0,1}* | |w|₀ chia hết cho 2}. Chấp nhận chuỗi nhị phân có số chữ số 0 là chẵn.",
        "alphabet": ["0", "1"],
        "default_input": "1001100",
        "states": [
            AutomatonState("q0", "q₀ (Chẵn)", 90.0, 150.0, is_start=True, is_accept=True),
            AutomatonState("q1", "q₁ (Lẻ)", 270.0, 150.0, is_start=False, is_accept=False),
        ],
        "transitions": [
            AutomatonTransition("q0", "0", "q1", curvature=-25.0),
            AutomatonTransition("q1", "0", "q0", curvature=-25.0),
            AutomatonTransition("q0", "1", "q0", curvature=0.0),  # Self-loop
            AutomatonTransition("q1", "1", "q1", curvature=0.0),  # Self-loop
        ],
    },
    "MOD_3": {
        "title": "Chia hết cho 3 (Binary mod 3)",
        "desc": "L = {w ∈ {0,1}* | giá trị thập phân mod 3 = 0}. Bộ đếm số dư khi đọc chuỗi nhị phân.",
        "alphabet": ["0", "1"],
        "default_input": "110",  # 6 in binary -> mod 3 == 0
        "states": [
            AutomatonState("q0", "q₀ (r=0)", 60.0, 150.0, is_start=True, is_accept=True),
            AutomatonState("q1", "q₁ (r=1)", 190.0, 85.0, is_start=False, is_accept=False),
            AutomatonState("q2", "q₂ (r=2)", 190.0, 215.0, is_start=False, is_accept=False),
        ],
        "transitions": [
            AutomatonTransition("q0", "0", "q0", curvature=0.0),   # 0*2+0 = 0
            AutomatonTransition("q0", "1", "q1", curvature=0.0),   # 0*2+1 = 1
            AutomatonTransition("q1", "0", "q2", curvature=0.0),   # 1*2+0 = 2
            AutomatonTransition("q1", "1", "q0", curvature=-18.0), # 1*2+1 = 3 = 0
            AutomatonTransition("q2", "0", "q1", curvature=18.0),  # 2*2+0 = 4 = 1
            AutomatonTransition("q2", "1", "q2", curvature=0.0),   # 2*2+1 = 5 = 2
        ],
    },
    "ENDS_WITH_ABB": {
        "title": "Kết thúc bằng chuỗi 'abb'",
        "desc": "L = {w ∈ {a,b}* | w kết thúc bằng 'abb'}. Bộ phân tích từ vựng (Lexical Pattern Matcher).",
        "alphabet": ["a", "b"],
        "default_input": "ababb",
        "states": [
            AutomatonState("q0", "q₀", 50.0, 150.0, is_start=True, is_accept=False),
            AutomatonState("q1", "q₁ (a)", 135.0, 150.0, is_start=False, is_accept=False),
            AutomatonState("q2", "q₂ (ab)", 220.0, 150.0, is_start=False, is_accept=False),
            AutomatonState("q3", "q₃ (abb)", 305.0, 150.0, is_start=False, is_accept=True),
        ],
        "transitions": [
            AutomatonTransition("q0", "a", "q1", curvature=0.0),
            AutomatonTransition("q0", "b", "q0", curvature=0.0),
            AutomatonTransition("q1", "a", "q1", curvature=0.0),
            AutomatonTransition("q1", "b", "q2", curvature=0.0),
            AutomatonTransition("q2", "a", "q1", curvature=-25.0),
            AutomatonTransition("q2", "b", "q3", curvature=0.0),
            AutomatonTransition("q3", "a", "q1", curvature=-45.0),
            AutomatonTransition("q3", "b", "q0", curvature=-60.0),
        ],
    },
    "ODD_PARITY": {
        "title": "Kiểm tra Parity Lẻ (Odd Parity)",
        "desc": "L = {w ∈ {0,1}* | số lượng bit 1 là số lẻ}. Ứng dụng trong phát hiện lỗi truyền dẫn mạng.",
        "alphabet": ["0", "1"],
        "default_input": "10110",  # 3 ones -> Accept
        "states": [
            AutomatonState("q0", "q₀ (Chẵn 1)", 90.0, 150.0, is_start=True, is_accept=False),
            AutomatonState("q1", "q₁ (Lẻ 1)", 270.0, 150.0, is_start=False, is_accept=True),
        ],
        "transitions": [
            AutomatonTransition("q0", "1", "q1", curvature=-25.0),
            AutomatonTransition("q1", "1", "q0", curvature=-25.0),
            AutomatonTransition("q0", "0", "q0", curvature=0.0),
            AutomatonTransition("q1", "0", "q1", curvature=0.0),
        ],
    },
}


class InteractiveAutomataItem(QGraphicsObject):
    """
    Interactive Finite State Machine & Automata (DFA/NFA) Simulator for LessonBoard.
    Visualizes state transition graph, input tape head, real-time step execution,
    and formal language acceptance verdicts.
    """

    state_changed = Signal()
    step_executed = Signal(str, str, str)  # (source, symbol, target)
    simulation_finished = Signal(bool)      # is_accepted
    closed = Signal()

    def __init__(self, pos: Tuple[float, float] = (300.0, 150.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])

        self._width = 620.0
        self._height = 430.0

        # Model Data
        self._current_preset_key: str = "EVEN_ZEROS"
        self._states: Dict[str, AutomatonState] = {}
        self._transitions: List[AutomatonTransition] = []
        self._alphabet: List[str] = ["0", "1"]
        self._start_state: str = "q0"

        # Simulation State
        self._input_string: str = "1001100"
        self._tape_index: int = 0  # Points to next symbol to read (0..len(input_string))
        self._current_state_id: str = "q0"
        self._history: List[Tuple[str, int]] = []  # [(state_id, tape_index), ...]
        self._is_running: bool = False
        self._last_traversed_edge: Optional[AutomatonTransition] = None

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(450)  # 450 ms per step in auto-run mode
        self._timer.timeout.connect(self._on_timer_step)

        # Pulse timer for active state glow
        self._pulse_timer = QTimer()
        self._pulse_timer.setInterval(40)  # 25 FPS
        self._pulse_timer.timeout.connect(self._on_pulse_tick)
        self._pulse_phase: float = 0.0
        self._pulse_timer.start()

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(35.0)

        self.load_preset("EVEN_ZEROS")

    # ── Geometry & Bounds ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Presets & Model Setup ──────────────────────────────────────────────────

    def load_preset(self, key: str) -> None:
        if key not in PRESETS:
            return
        self.pause()
        self._current_preset_key = key
        data = PRESETS[key]
        self._alphabet = list(data["alphabet"])
        self._input_string = str(data["default_input"])

        self._states.clear()
        for s in data["states"]:
            # Clone state object
            self._states[s.id] = AutomatonState(
                id=s.id,
                label=s.label,
                x=s.x,
                y=s.y,
                is_start=s.is_start,
                is_accept=s.is_accept,
            )
            if s.is_start:
                self._start_state = s.id

        self._transitions = [
            AutomatonTransition(t.source, t.symbol, t.target, t.curvature)
            for t in data["transitions"]
        ]

        self.reset_simulation()
        self.state_changed.emit()
        self.update()

    # ── Formal Automata Simulation Logic ──────────────────────────────────────

    def reset_simulation(self) -> None:
        """Resets tape head to start and state to q0."""
        self.pause()
        self._current_state_id = self._start_state
        self._tape_index = 0
        self._history = [(self._current_state_id, self._tape_index)]
        self._last_traversed_edge = None
        self.state_changed.emit()
        self.update()

    def step_forward(self) -> bool:
        """Reads one symbol under tape head, executes transition δ(q, a), and moves head right."""
        if self._tape_index >= len(self._input_string):
            self.pause()
            return False

        symbol = self._input_string[self._tape_index]
        target_state_id = self._find_target_state(self._current_state_id, symbol)

        if target_state_id is None:
            # Trap / Dead state (rejected)
            self.pause()
            logger.warning(f"No transition for ({self._current_state_id}, '{symbol}'). Machine halted.")
            return False

        # Find edge
        edge = None
        for t in self._transitions:
            if t.source == self._current_state_id and t.symbol == symbol and t.target == target_state_id:
                edge = t
                break
        self._last_traversed_edge = edge

        src_state = self._current_state_id
        self._current_state_id = target_state_id
        self._tape_index += 1
        self._history.append((self._current_state_id, self._tape_index))

        self.step_executed.emit(src_state, symbol, target_state_id)
        self.state_changed.emit()

        if self._tape_index == len(self._input_string):
            self.simulation_finished.emit(self.is_accepted())
            self.pause()

        self.update()
        return True

    def step_backward(self) -> bool:
        """Rewinds simulation by 1 step in execution history."""
        if len(self._history) <= 1:
            return False

        self.pause()
        self._history.pop()
        prev_state, prev_idx = self._history[-1]
        self._current_state_id = prev_state
        self._tape_index = prev_idx
        self._last_traversed_edge = None
        self.state_changed.emit()
        self.update()
        return True

    def play(self) -> None:
        if not self._is_running:
            if self._tape_index >= len(self._input_string):
                self.reset_simulation()
            self._is_running = True
            self._timer.start()
            self.update()

    def pause(self) -> None:
        if self._is_running:
            self._is_running = False
            self._timer.stop()
            self.update()

    def toggle_play_pause(self) -> None:
        if self._is_running:
            self.pause()
        else:
            self.play()

    def _on_timer_step(self) -> None:
        if not self.step_forward():
            self.pause()

    def _on_pulse_tick(self) -> None:
        self._pulse_phase = (self._pulse_phase + 0.1) % (2.0 * math.pi)
        self.update()

    def _find_target_state(self, current_state: str, symbol: str) -> Optional[str]:
        for t in self._transitions:
            if t.source == current_state and t.symbol == symbol:
                return t.target
        return None

    # ── Inspection & State Queries ────────────────────────────────────────────

    def is_accepted(self) -> bool:
        """Returns True if tape has been fully consumed and machine stopped in an accept state."""
        if self._tape_index < len(self._input_string):
            return False
        curr = self._states.get(self._current_state_id)
        return curr.is_accept if curr else False

    def is_finished(self) -> bool:
        return self._tape_index >= len(self._input_string)

    def get_verdict_text(self) -> Tuple[str, QColor]:
        """Returns (Status string, Theme color)."""
        if not self.is_finished():
            if self._is_running:
                return "ĐANG CHẠY (PROCESSING...)", QColor("#38bdf8")
            else:
                return f"ĐANG DỪNG (ĐÃ ĐỌC {self._tape_index}/{len(self._input_string)})", QColor("#94a3b8")
        else:
            if self.is_accepted():
                return "CHẤP NHẬN (ACCEPTED) ✔", QColor("#22c55e")
            else:
                return "TỪ CHỐI (REJECTED) ✘", QColor("#ef4444")

    def test_string(self, test_input: str) -> bool:
        """Batch evaluator for test assertions."""
        state = self._start_state
        for sym in test_input:
            nxt = self._find_target_state(state, sym)
            if nxt is None:
                return False
            state = nxt
        curr = self._states.get(state)
        return curr.is_accept if curr else False

    # ── Tape Editing ──────────────────────────────────────────────────────────

    def set_input_string(self, text: str) -> None:
        filtered = "".join([c for c in text if c in self._alphabet])
        self._input_string = filtered
        self.reset_simulation()

    def append_symbol(self, sym: str) -> None:
        if sym in self._alphabet and len(self._input_string) < 16:
            self._input_string += sym
            self.reset_simulation()

    def backspace_symbol(self) -> None:
        if self._input_string:
            self._input_string = self._input_string[:-1]
            self.reset_simulation()

    def clear_tape(self) -> None:
        self._input_string = ""
        self.reset_simulation()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if hasattr(self, "_timer") and self._timer.isActive():
            self._timer.stop()
        if hasattr(self, "_pulse_timer") and self._pulse_timer.isActive():
            self._pulse_timer.stop()

    def close(self) -> None:
        self.cleanup()
        self.setVisible(False)
        self.closed.emit()

    # ── UI Rectangles ─────────────────────────────────────────────────────────

    def _get_close_btn_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 10.0, 22.0, 22.0)

    def _get_stamp_btn_rect(self) -> QRectF:
        return QRectF(self._width - 150.0, 10.0, 110.0, 24.0)

    def _get_preset_rects(self) -> List[Tuple[str, QRectF]]:
        keys = list(PRESETS.keys())
        rects = []
        x = 14.0
        y = 42.0
        w = 140.0
        h = 24.0
        for i, k in enumerate(keys):
            rects.append((k, QRectF(x + i * (w + 6.0), y, w, h)))
        return rects

    def _get_tape_rect(self) -> QRectF:
        return QRectF(16.0, 290.0, 360.0, 48.0)

    def _get_step_btn_rect(self) -> QRectF:
        return QRectF(385.0, 296.0, 68.0, 36.0)

    def _get_play_btn_rect(self) -> QRectF:
        return QRectF(460.0, 296.0, 72.0, 36.0)

    def _get_reset_btn_rect(self) -> QRectF:
        return QRectF(538.0, 296.0, 68.0, 36.0)

    def _get_alphabet_btn_rects(self) -> List[Tuple[str, QRectF]]:
        rects = []
        x = 16.0
        y = 352.0
        w = 34.0
        h = 28.0
        for i, sym in enumerate(self._alphabet):
            rects.append((sym, QRectF(x + i * (w + 6.0), y, w, h)))
        # Backspace button
        bk_x = x + len(self._alphabet) * (w + 6.0)
        rects.append(("BK", QRectF(bk_x, y, 42.0, h)))
        # Clear button
        cl_x = bk_x + 48.0
        rects.append(("CLR", QRectF(cl_x, y, 42.0, h)))
        return rects

    # ── Rendering ─────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor("#0a0f1d"))
        grad.setColorAt(1.0, QColor("#060a14"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Presets Bar
        self._paint_presets(painter)

        # 4. State Diagram Graph (Nodes, Curved Directed Edges, Self-Loops)
        self._paint_automaton_graph(painter)

        # 5. Diagnostic HUD & Formal Theory Panel
        self._paint_status_hud(painter)

        # 6. Input Tape & Execution Tape Head
        self._paint_tape_and_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 26, "🤖 MÁY TRẠNG THÁI HỮU HẠN & AUTOMATA (FINITE STATE MACHINE / DFA)")

        # Stamp Button
        s_rect = self._get_stamp_btn_rect()
        painter.setBrush(QBrush(QColor("#4f46e5")))
        painter.setPen(QPen(QColor("#818cf8"), 1.2))
        painter.drawRoundedRect(s_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu bảng")

        # Close Button
        c_rect = self._get_close_btn_rect()
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(c_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_presets(self, painter: QPainter) -> None:
        for key, rect in self._get_preset_rects():
            p_data = PRESETS[key]
            is_active = (key == self._current_preset_key)
            bg = QColor("#4338ca") if is_active else QColor("#1e293b")
            border = QColor("#818cf8") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            # Truncate title if needed
            title = p_data["title"]
            if len(title) > 20:
                title = title[:18] + ".."
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, title)

    def _paint_automaton_graph(self, painter: QPainter) -> None:
        graph_rect = QRectF(16.0, 74.0, 360.0, 206.0)
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(graph_rect, 10.0, 10.0)

        # Scale and offset coordinates within graph_rect
        ox = graph_rect.left()
        oy = graph_rect.top()

        # 1. Draw Transitions
        for t in self._transitions:
            self._paint_transition_edge(painter, t, ox, oy)

        # 2. Draw States
        for s in self._states.values():
            self._paint_state_node(painter, s, ox, oy)

    def _paint_state_node(self, painter: QPainter, state: AutomatonState, ox: float, oy: float) -> None:
        cx = ox + state.x
        cy = oy + state.y
        radius = 24.0

        is_active = (state.id == self._current_state_id)

        # Glowing ring for active state
        if is_active:
            glow_r = radius + 5.0 + 3.0 * math.sin(self._pulse_phase)
            grad = QRadialGradient(cx, cy, glow_r)
            grad.setColorAt(0.0, QColor(56, 189, 248, 140))
            grad.setColorAt(0.8, QColor(56, 189, 248, 50))
            grad.setColorAt(1.0, QColor(56, 189, 248, 0))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # Start state arrow pointing to node
        if state.is_start:
            painter.setPen(QPen(QColor("#f59e0b"), 2.0))
            painter.drawLine(QPointF(cx - radius - 22.0, cy), QPointF(cx - radius - 2.0, cy))
            # Arrow head
            arrow = QPolygonF([
                QPointF(cx - radius - 2.0, cy),
                QPointF(cx - radius - 8.0, cy - 4.5),
                QPointF(cx - radius - 8.0, cy + 4.5),
            ])
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.drawPolygon(arrow)

        # Outer Node Body
        bg = QColor("#1e293b") if not is_active else QColor("#0369a1")
        border = QColor("#64748b") if not is_active else QColor("#38bdf8")
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 2.0 if is_active else 1.4))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Accept State (Double Circle ⊚)
        if state.is_accept:
            inner_r = radius - 4.5
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(border, 1.5))
            painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        # Node Label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(
            QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0),
            Qt.AlignmentFlag.AlignCenter,
            state.id,
        )

    def _paint_transition_edge(self, painter: QPainter, t: AutomatonTransition, ox: float, oy: float) -> None:
        src = self._states.get(t.source)
        tgt = self._states.get(t.target)
        if not src or not tgt:
            return

        is_traversed = (self._last_traversed_edge == t)
        edge_color = QColor("#38bdf8") if is_traversed else QColor("#64748b")
        pen_width = 2.2 if is_traversed else 1.4

        sx = ox + src.x
        sy = oy + src.y
        tx = ox + tgt.x
        ty = oy + tgt.y
        radius = 24.0

        if t.source == t.target:
            # Self-loop drawn above node
            loop_r = 16.0
            loop_cx = sx
            loop_cy = sy - radius - loop_r + 4.0

            path = QPainterPath()
            path.arcMoveTo(loop_cx - loop_r, loop_cy - loop_r, loop_r * 2.0, loop_r * 2.0, -40)
            path.arcTo(loop_cx - loop_r, loop_cy - loop_r, loop_r * 2.0, loop_r * 2.0, -40, 260)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(edge_color, pen_width))
            painter.drawPath(path)

            # Self-loop arrowhead
            arrow_pt = QPointF(sx + 8.0, sy - radius + 2.0)
            arrow = QPolygonF([
                arrow_pt,
                QPointF(arrow_pt.x() + 5.0, arrow_pt.y() - 5.0),
                QPointF(arrow_pt.x() - 1.0, arrow_pt.y() - 6.0),
            ])
            painter.setBrush(QBrush(edge_color))
            painter.drawPolygon(arrow)

            # Symbol Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f8fafc") if is_traversed else QColor("#cbd5e1"))
            painter.drawText(
                QRectF(loop_cx - 15.0, loop_cy - loop_r - 12.0, 30.0, 16.0),
                Qt.AlignmentFlag.AlignCenter,
                t.symbol,
            )
        else:
            # Directed Edge between distinct nodes
            dx = tx - sx
            dy = ty - sy
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                return

            ux = dx / dist
            uy = dy / dist

            # Perpendicular normal
            nx = -uy
            ny = ux

            start_pt = QPointF(sx + ux * radius, sy + uy * radius)
            end_pt = QPointF(tx - ux * radius, ty - uy * radius)

            ctrl_pt = QPointF(
                (sx + tx) / 2.0 + nx * t.curvature,
                (sy + ty) / 2.0 + ny * t.curvature,
            )

            path = QPainterPath()
            path.moveTo(start_pt)
            path.quadTo(ctrl_pt, end_pt)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(edge_color, pen_width))
            painter.drawPath(path)

            # Calculate tangent at end_pt for arrowhead
            t_tan_x = end_pt.x() - ctrl_pt.x()
            t_tan_y = end_pt.y() - ctrl_pt.y()
            tan_len = math.hypot(t_tan_x, t_tan_y)
            if tan_len > 1e-4:
                tux = t_tan_x / tan_len
                tuy = t_tan_y / tan_len
                tnx = -tuy
                tny = tux

                arrow_sz = 7.0
                arrow = QPolygonF([
                    end_pt,
                    QPointF(end_pt.x() - tux * arrow_sz + tnx * 4.0, end_pt.y() - tuy * arrow_sz + tny * 4.0),
                    QPointF(end_pt.x() - tux * arrow_sz - tnx * 4.0, end_pt.y() - tuy * arrow_sz - tny * 4.0),
                ])
                painter.setBrush(QBrush(edge_color))
                painter.drawPolygon(arrow)

            # Draw edge label at midpoint of curve
            mid_pt = QPointF(
                0.25 * start_pt.x() + 0.5 * ctrl_pt.x() + 0.25 * end_pt.x(),
                0.25 * start_pt.y() + 0.5 * ctrl_pt.y() + 0.25 * end_pt.y(),
            )
            # Offset slightly along normal
            lbl_pt = QPointF(mid_pt.x() + nx * 8.0, mid_pt.y() + ny * 8.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f8fafc") if is_traversed else QColor("#cbd5e1"))
            painter.drawText(
                QRectF(lbl_pt.x() - 12.0, lbl_pt.y() - 8.0, 24.0, 16.0),
                Qt.AlignmentFlag.AlignCenter,
                t.symbol,
            )

    def _paint_status_hud(self, painter: QPainter) -> None:
        hud_rect = QRectF(386.0, 74.0, 218.0, 206.0)
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(hud_rect.adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignTop, "📊 THÔNG SỐ VÀ TRẠNG THÁI")

        # Preset Description
        p_data = PRESETS[self._current_preset_key]
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            QRectF(hud_rect.left() + 12.0, hud_rect.top() + 32.0, 194.0, 36.0),
            Qt.TextFlag.TextWordWrap,
            p_data["desc"],
        )

        # Current State & Active Tape Head
        curr_state = self._states.get(self._current_state_id)
        state_lbl = curr_state.label if curr_state else self._current_state_id
        is_acc = curr_state.is_accept if curr_state else False

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#e2e8f0"))
        y_pos = hud_rect.top() + 74.0
        painter.drawText(hud_rect.left() + 12.0, y_pos, f"Trạng thái hiện tại: {state_lbl}")

        y_pos += 20.0
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(hud_rect.left() + 12.0, y_pos, f"Thuộc tập kết thúc (F): {'Có ⊚' if is_acc else 'Không'}")

        y_pos += 20.0
        rem = len(self._input_string) - self._tape_index
        painter.drawText(hud_rect.left() + 12.0, y_pos, f"Ký tự còn lại trên băng: {max(0, rem)}")

        # Verdict Box
        v_rect = QRectF(hud_rect.left() + 10.0, hud_rect.bottom() - 44.0, 198.0, 32.0)
        verdict_text, verdict_color = self.get_verdict_text()

        painter.setBrush(QBrush(QColor(verdict_color.red(), verdict_color.green(), verdict_color.blue(), 35)))
        painter.setPen(QPen(verdict_color, 1.4))
        painter.drawRoundedRect(v_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(verdict_color)
        painter.drawText(v_rect, Qt.AlignmentFlag.AlignCenter, verdict_text)

    def _paint_tape_and_controls(self, painter: QPainter) -> None:
        # 1. Tape Frame
        t_rect = self._get_tape_rect()
        painter.setBrush(QBrush(QColor("#0d1322")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(t_rect, 8.0, 8.0)

        # Draw Tape Cells
        cell_w = 26.0
        cell_h = 32.0
        start_x = t_rect.left() + 10.0
        start_y = t_rect.top() + 8.0

        n_cells = max(12, len(self._input_string))
        for i in range(n_cells):
            cx = start_x + i * (cell_w + 2.0)
            if cx + cell_w > t_rect.right() - 8.0:
                break
            c_rect = QRectF(cx, start_y, cell_w, cell_h)

            char_val = self._input_string[i] if i < len(self._input_string) else " "
            is_head = (i == self._tape_index)

            # Cell Background
            if is_head:
                c_bg = QColor("#0284c7")
                c_border = QColor("#38bdf8")
            elif i < self._tape_index:
                c_bg = QColor("#1e293b")
                c_border = QColor("#334155")
            else:
                c_bg = QColor("#111827")
                c_border = QColor("#1f2937")

            painter.setBrush(QBrush(c_bg))
            painter.setPen(QPen(c_border, 1.2 if is_head else 0.8))
            painter.drawRoundedRect(c_rect, 4.0, 4.0)

            # Character
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_head else QColor("#cbd5e1"))
            painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, char_val)

            # Head Pointer (▲)
            if is_head:
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.setPen(QColor("#38bdf8"))
                painter.drawText(
                    QRectF(cx, start_y + cell_h + 1.0, cell_w, 14.0),
                    Qt.AlignmentFlag.AlignCenter,
                    "▲",
                )

        # 2. Control Buttons: Step, Play/Pause, Reset
        step_r = self._get_step_btn_rect()
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawRoundedRect(step_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(step_r, Qt.AlignmentFlag.AlignCenter, "▶| Bước")

        play_r = self._get_play_btn_rect()
        is_p = self._is_running
        painter.setBrush(QBrush(QColor("#16a34a") if not is_p else QColor("#d97706")))
        painter.setPen(QPen(QColor("#4ade80") if not is_p else QColor("#fcd34d"), 1.2))
        painter.drawRoundedRect(play_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(play_r, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if is_p else "▶ Chạy")

        reset_r = self._get_reset_btn_rect()
        painter.setBrush(QBrush(QColor("#334155")))
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawRoundedRect(reset_r, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(reset_r, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

        # 3. Alphabet Quick Input Chips
        for sym, r in self._get_alphabet_btn_rects():
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(r, 4.0, 4.0)

            lbl = sym
            if sym == "BK":
                lbl = "⌫"
            elif sym == "CLR":
                lbl = "✕"

            painter.setFont(QFont("Consolas" if len(sym) == 1 else "Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, lbl)

    # ── Mouse Interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()

        # Close
        if self._get_close_btn_rect().contains(loc):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_btn_rect().contains(loc):
            self.stamp_to_board()
            event.accept()
            return

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.load_preset(key)
                event.accept()
                return

        # Step
        if self._get_step_btn_rect().contains(loc):
            self.step_forward()
            event.accept()
            return

        # Play/Pause
        if self._get_play_btn_rect().contains(loc):
            self.toggle_play_pause()
            event.accept()
            return

        # Reset
        if self._get_reset_btn_rect().contains(loc):
            self.reset_simulation()
            event.accept()
            return

        # Alphabet inputs
        for sym, rect in self._get_alphabet_btn_rects():
            if rect.contains(loc):
                if sym == "BK":
                    self.backspace_symbol()
                elif sym == "CLR":
                    self.clear_tape()
                else:
                    self.append_symbol(sym)
                event.accept()
                return

        super().mousePressEvent(event)

    # ── Touchless Air Gesture Handlers ────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_btn_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_btn_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.load_preset(key)
                return True

        # Step
        if self._get_step_btn_rect().contains(loc):
            self.step_forward()
            return True

        # Play / Pause
        if self._get_play_btn_rect().contains(loc):
            self.toggle_play_pause()
            return True

        # Reset
        if self._get_reset_btn_rect().contains(loc):
            self.reset_simulation()
            return True

        # Alphabet quick input
        for sym, rect in self._get_alphabet_btn_rects():
            if rect.adjusted(-4, -4, 4, 4).contains(loc):
                if sym == "BK":
                    self.backspace_symbol()
                elif sym == "CLR":
                    self.clear_tape()
                else:
                    self.append_symbol(sym)
                return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── LessonBoard Permanent Vector Stamping ─────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of current Automaton to LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = InformaticsTools.create_automata_diagram(
            center=(cx, cy + 30.0),
            width=540.0,
            height=320.0,
            preset=self._current_preset_key,
            input_string=self._input_string,
        )
        for it in diagram_items:
            board.addItem(it)

        if hasattr(board, "commit_state_action"):
            board.commit_state_action("Stamp Automata Diagram")
        logger.info(f"Automata diagram stamped to LessonBoard at ({cx:.1f}, {cy:.1f}).")
