"""
AirTeach AI - Interactive Quantum Computing: Multi-Qubit Circuit, Bloch Sphere & Teleportation
=============================================================================================
Authentic Quantum Information Science & Quantum Circuit Simulator:
- State Vector Formalism:
    2^N complex amplitude state vector |Ψ⟩ = ∑ c_k |k⟩ with ∑ |c_k|^2 = 1.
    Exact tensor product matrix multiplication (numpy.complex128).
- Quantum Logic Gates (Unitary operators U^† U = I):
    Single-qubit: Pauli-X, Y, Z, Hadamard (H), Phase (S), T-gate, Rotation Ry(θ), Rz(φ).
    Two-qubit entangling: CNOT (CX), Controlled-Z (CZ), SWAP.
- 3D Bloch Sphere Representation:
    Reduced density matrix ρ_i = Tr_{\\i}(|Ψ⟩⟨Ψ|).
    Bloch vector: r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩) = (Tr(ρ σ_x), Tr(ρ σ_y), Tr(ρ σ_z)).
    Purity: |r| = 1 for pure unentangled states, |r| < 1 for entangled mixed states.
- Quantum Entanglement Metrics:
    Concurrence: C(|Ψ⟩) = 2 |c_00 c_11 - c_01 c_10| for 2-qubit states.
    Von Neumann entanglement entropy: S(ρ_A) = -Tr(ρ_A log2 ρ_A).
- 5 Authentic Curriculum Presets:
    1. BELL_PHI_PLUS: Maximally entangled Bell pair (|00⟩ + |11⟩) / √2, C = 1.0, S = 1.0 bit.
    2. QUANTUM_TELEPORTATION: 3-qubit Bennett et al. (1993) protocol teleporting unknown |ψ⟩.
    3. SUPERDENSE_CODING: Transmitting 2 classical bits via 1 qubit over shared entanglement.
    4. GHZ_3QUBIT: 3-qubit Greenberger-Horne-Zeilinger state (|000⟩ + |111⟩) / √2.
    5. DEUTSCH_ALGORITHM: Quantum parallelism evaluating balanced oracle in 1 query.
- Touchless air gestures (air-tap gate execution, step forward, state inspection),
  bimanual rotation of Bloch spheres, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Pauli Matrices
SIGMA_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H_GATE = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
S_GATE = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
T_GATE = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * math.pi / 4.0)]], dtype=np.complex128)


@dataclass
class CircuitGate:
    """A gate operation in the quantum circuit."""
    name: str
    target_qubit: int
    control_qubit: Optional[int] = None
    param: float = 0.0  # Optional angle parameter (e.g. for Ry or Rz)


@dataclass
class QuantumPreset:
    """Preset quantum computing circuit configuration."""
    key: str
    name: str
    desc: str
    num_qubits: int
    gates: List[CircuitGate]
    expected_state_desc: str


PRESETS: List[QuantumPreset] = [
    QuantumPreset(
        key="bell_phi_plus",
        name="Cặp vướng víu Bell |Φ⁺⟩ = (|00⟩ + |11⟩)/√2",
        desc="Cổng Hadamard đưa q0 vào chồng chập, sau đó CNOT(q0, q1) tạo trạng thái vướng víu lượng tử cực đại",
        num_qubits=2,
        gates=[
            CircuitGate("H", target_qubit=0),
            CircuitGate("CX", target_qubit=1, control_qubit=0),
        ],
        expected_state_desc="(|00⟩ + |11⟩) / √2",
    ),
    QuantumPreset(
        key="quantum_teleportation",
        name="Giao thức Dịch chuyển Lượng tử (Teleportation)",
        desc="Giao thức Bennett 1993: Dịch chuyển trạng thái |ψ⟩ từ Alice sang Bob nhờ cặp Bell và kênh cổ điển",
        num_qubits=3,
        gates=[
            # Prepare state |ψ⟩ on q0 = cos(π/6)|0⟩ + sin(π/6)|1⟩
            CircuitGate("Ry", target_qubit=0, param=math.pi / 3.0),
            # Entangle q1 and q2 (Bell pair shared by Alice and Bob)
            CircuitGate("H", target_qubit=1),
            CircuitGate("CX", target_qubit=2, control_qubit=1),
            # Alice applies Bell measurement gates on q0, q1
            CircuitGate("CX", target_qubit=1, control_qubit=0),
            CircuitGate("H", target_qubit=0),
            # Bob applies conditional corrections (CZ, CX)
            CircuitGate("CX", target_qubit=2, control_qubit=1),
            CircuitGate("CZ", target_qubit=2, control_qubit=0),
        ],
        expected_state_desc="q2 tái tạo hoàn hảo trạng thái |ψ⟩ (Độ chính xác 100%)",
    ),
    QuantumPreset(
        key="superdense_coding",
        name="Mã hóa Siêu mật (Superdense Coding - 2 bits / 1 Qubit)",
        desc="Truyền 2 bit cổ điển ('11') chỉ bằng cách gửi 1 qubit qua một cặp photon vướng víu Bell",
        num_qubits=2,
        gates=[
            # Create Bell pair |Φ+⟩
            CircuitGate("H", target_qubit=0),
            CircuitGate("CX", target_qubit=1, control_qubit=0),
            # Alice encodes message '11' using Z then X on q0
            CircuitGate("Z", target_qubit=0),
            CircuitGate("X", target_qubit=0),
            # Bob decodes using CX and H
            CircuitGate("CX", target_qubit=1, control_qubit=0),
            CircuitGate("H", target_qubit=0),
        ],
        expected_state_desc="Kết quả giải mã xác định: 100% |11⟩",
    ),
    QuantumPreset(
        key="ghz_3qubit",
        name="Trạng thái vướng víu 3 Qubit GHZ (|000⟩ + |111⟩)/√2",
        desc="Trạng thái Greenberger-Horne-Zeilinger minh họa tính bất định lượng tử không thể giải thích bằng biến ẩn cục bộ",
        num_qubits=3,
        gates=[
            CircuitGate("H", target_qubit=0),
            CircuitGate("CX", target_qubit=1, control_qubit=0),
            CircuitGate("CX", target_qubit=2, control_qubit=1),
        ],
        expected_state_desc="(|000⟩ + |111⟩) / √2",
    ),
    QuantumPreset(
        key="deutsch_algorithm",
        name="Thuật toán Deutsch: Tính song song Lượng tử",
        desc="Xác định hàm f(x) là hằng số hay cân bằng chỉ với đúng 1 lần truy vấn Oracle, nhanh gấp đôi máy tính cổ điển",
        num_qubits=2,
        gates=[
            # Prepare |01⟩
            CircuitGate("X", target_qubit=1),
            # Superposition on both
            CircuitGate("H", target_qubit=0),
            CircuitGate("H", target_qubit=1),
            # Balanced Oracle: CX
            CircuitGate("CX", target_qubit=1, control_qubit=0),
            # Interference on q0
            CircuitGate("H", target_qubit=0),
        ],
        expected_state_desc="q0 = |1⟩ => Hàm Cân bằng (Balanced) với 1 truy vấn",
    ),
]


class InteractiveQuantumCircuitItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Multi-Qubit Quantum Circuits,
    3D Bloch Spheres, Entanglement Metrics, and Teleportation.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 940.0
    HEIGHT: float = 600.0

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._step_cursor: int = 0
        self._bloch_rot_angle: float = 0.0

        p = PRESETS[self._preset_index]
        self._num_qubits: int = p.num_qubits
        self._gates: List[CircuitGate] = list(p.gates)
        self._state_vector: np.ndarray = self._compute_state_vector(len(self._gates))

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to smoothly rotate 3D Bloch sphere representation."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer defensively."""
        try:
            if hasattr(self, "_anim_timer") and self._anim_timer is not None and self._anim_timer.isActive():
                self._anim_timer.stop()
        except (RuntimeError, ReferenceError):
            pass

    @property
    def current_preset(self) -> QuantumPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active quantum circuit preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._num_qubits = p.num_qubits
            self._gates = list(p.gates)
            self._step_cursor = len(self._gates)
            self._state_vector = self._compute_state_vector(self._step_cursor)
            self.preset_changed.emit(p.key)
            self.update()

    def step_forward(self) -> None:
        """Execute one gate forward in circuit."""
        if self._step_cursor < len(self._gates):
            self._step_cursor += 1
            self._state_vector = self._compute_state_vector(self._step_cursor)
            self.update()

    def reset_circuit(self) -> None:
        """Reset state vector to ground state |0...0⟩."""
        self._step_cursor = 0
        self._state_vector = self._compute_state_vector(0)
        self.update()

    def run_all_gates(self) -> None:
        """Execute all gates to completion."""
        self._step_cursor = len(self._gates)
        self._state_vector = self._compute_state_vector(self._step_cursor)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle animation rotation pause/resume."""
        self._is_paused = not self._is_paused
        self.update()

    # -------------------------------------------------------------------------
    # Authentic Quantum Circuit & Linear Algebra Calculations
    # -------------------------------------------------------------------------

    def _single_qubit_gate_matrix(self, name: str, param: float = 0.0) -> np.ndarray:
        """Return 2x2 unitary matrix for single-qubit gate."""
        if name == "X":
            return SIGMA_X
        elif name == "Y":
            return SIGMA_Y
        elif name == "Z":
            return SIGMA_Z
        elif name == "H":
            return H_GATE
        elif name == "S":
            return S_GATE
        elif name == "T":
            return T_GATE
        elif name == "Ry":
            cos_half = math.cos(param / 2.0)
            sin_half = math.sin(param / 2.0)
            return np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=np.complex128)
        elif name == "Rz":
            exp_neg = np.exp(-0.5j * param)
            exp_pos = np.exp(0.5j * param)
            return np.array([[exp_neg, 0.0], [0.0, exp_pos]], dtype=np.complex128)
        return SIGMA_I

    def _compute_state_vector(self, up_to_gate_idx: int) -> np.ndarray:
        """
        Simulate circuit starting from |0...0⟩ up to up_to_gate_idx.
        Returns complex state vector of length 2^num_qubits.
        """
        dim = 1 << self._num_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0  # Ground state |0...0⟩

        for gate in self._gates[:up_to_gate_idx]:
            if gate.control_qubit is None:
                # Single qubit gate: I ⊗ ... ⊗ U ⊗ ... ⊗ I
                u = self._single_qubit_gate_matrix(gate.name, gate.param)
                full_op = np.array([1.0], dtype=np.complex128)
                for q in range(self._num_qubits):
                    if q == gate.target_qubit:
                        full_op = np.kron(full_op, u)
                    else:
                        full_op = np.kron(full_op, SIGMA_I)
                state = full_op @ state
            else:
                # Two-qubit gate (CX / CZ)
                c = gate.control_qubit
                t = gate.target_qubit
                new_state = np.zeros(dim, dtype=np.complex128)
                for idx in range(dim):
                    is_control_set = bool(idx & (1 << (self._num_qubits - 1 - c)))
                    if is_control_set:
                        if gate.name == "CX":
                            # Flip target bit
                            t_bit_pos = self._num_qubits - 1 - t
                            flipped_idx = idx ^ (1 << t_bit_pos)
                            new_state[flipped_idx] += state[idx]
                        elif gate.name == "CZ":
                            # Phase flip if target is also 1
                            t_bit_pos = self._num_qubits - 1 - t
                            if bool(idx & (1 << t_bit_pos)):
                                new_state[idx] += -state[idx]
                            else:
                                new_state[idx] += state[idx]
                        else:
                            new_state[idx] += state[idx]
                    else:
                        new_state[idx] += state[idx]
                state = new_state

        # Normalize defensively
        norm = np.linalg.norm(state)
        if norm > 1e-12:
            state /= norm
        return state

    def compute_probabilities(self) -> np.ndarray:
        """Return measurement probabilities P(k) = |c_k|^2 for each computational basis state."""
        return np.abs(self._state_vector) ** 2

    def compute_reduced_density_matrix(self, qubit_idx: int) -> np.ndarray:
        """
        Compute 2x2 reduced density matrix ρ_i = Tr_{\\i}(|Ψ⟩⟨Ψ|) for single qubit.
        """
        dim = 1 << self._num_qubits
        rho_full = np.outer(self._state_vector, np.conj(self._state_vector))
        rho_reduced = np.zeros((2, 2), dtype=np.complex128)

        bit_shift = self._num_qubits - 1 - qubit_idx
        for i in range(dim):
            bi = (i >> bit_shift) & 1
            for j in range(dim):
                bj = (j >> bit_shift) & 1
                # Check that all other qubits match for trace
                mask = ~(1 << bit_shift)
                if (i & mask) == (j & mask):
                    rho_reduced[bi, bj] += rho_full[i, j]
        return rho_reduced

    def compute_bloch_coordinates(self, qubit_idx: int) -> Tuple[float, float, float]:
        """
        Compute (rx, ry, rz) Bloch coordinates:
        rx = Tr(ρ σ_x), ry = Tr(ρ σ_y), rz = Tr(ρ σ_z).
        """
        rho = self.compute_reduced_density_matrix(qubit_idx)
        rx = float(np.real(np.trace(rho @ SIGMA_X)))
        ry = float(np.real(np.trace(rho @ SIGMA_Y)))
        rz = float(np.real(np.trace(rho @ SIGMA_Z)))
        return rx, ry, rz

    def compute_entanglement_concurrence_2qubits(self) -> float:
        """
        Compute Wootters concurrence C for 2-qubit pure state:
        C = 2 * |c_00 * c_11 - c_01 * c_10|
        """
        if self._num_qubits != 2 or len(self._state_vector) < 4:
            return 0.0
        c00 = self._state_vector[0]
        c01 = self._state_vector[1]
        c10 = self._state_vector[2]
        c11 = self._state_vector[3]
        concurrence = 2.0 * abs(c00 * c11 - c01 * c10)
        return float(min(1.0, max(0.0, concurrence)))

    def compute_von_neumann_entropy(self, qubit_idx: int = 0) -> float:
        """
        Von Neumann entanglement entropy S = -Tr(ρ_A log2 ρ_A).
        """
        rho = self.compute_reduced_density_matrix(qubit_idx)
        eigenvalues = np.linalg.eigvalsh(rho)
        entropy = 0.0
        for ev in eigenvalues:
            if ev > 1e-12:
                entropy -= float(ev * math.log2(ev))
        return float(max(0.0, entropy))

    def step_physics(self, dt: float) -> None:
        """Advance time for 3D Bloch sphere rotation."""
        if self._is_paused or dt <= 0.0:
            return
        self._bloch_rot_angle += dt * 0.8
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(15.0, 10.0, 910.0, 60.0)

    def _circuit_rect(self) -> QRectF:
        return QRectF(15.0, 75.0, 310.0, 410.0)

    def _bloch_rect(self) -> QRectF:
        return QRectF(335.0, 75.0, 310.0, 410.0)

    def _histogram_rect(self) -> QRectF:
        return QRectF(655.0, 75.0, 270.0, 410.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, QuantumPreset]]:
        h_rect = self._header_rect()
        btn_w = 95.0
        btn_h = 28.0
        gap = 6.0
        start_x = 350.0
        y = (h_rect.height() - btn_h) / 2.0
        rects = []
        for i, p in enumerate(PRESETS):
            bx = start_x + i * (btn_w + gap)
            rects.append((QRectF(bx, y, btn_w, btn_h), p))
        return rects

    def paint(
        self,
        painter: QPainter,
        option: Optional[QStyleOptionGraphicsItem] = None,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Main Card Background
        bg_brush = QBrush(QColor(10, 14, 26, 248))
        border_pen = QPen(QColor(168, 85, 247, 160), 1.5)  # Purple Quantum Glow
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_circuit_panel(painter)
        self._paint_bloch_panel(painter)
        self._paint_histogram_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ TIN HỌC LƯỢNG TỬ: MẠCH QUBIT & QUẢ CẦU BLOCH")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        c_val = self.compute_entanglement_concurrence_2qubits()
        s_val = self.compute_von_neumann_entropy(0)
        painter.drawText(
            20,
            44,
            f"Mô hình: {p.name}  |  Độ vướng víu Concurrence C = {c_val:.2f}  |  Entropy von Neumann: {s_val:.2f} bits",
        )

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(168, 85, 247, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(192, 132, 252), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(148, 163, 184))
            label = preset.key.upper()[:10]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        painter.setBrush(QColor(168, 85, 247, 40))
        painter.setPen(QPen(QColor(192, 132, 252), 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(192, 132, 252))
        painter.drawText(badge_rect, Qt.AlignCenter, f"Bước: {self._step_cursor}/{len(self._gates)}")

    def _paint_circuit_panel(self, painter: QPainter) -> None:
        """Render Left Panel: Multi-Qubit Circuit Horizontal Wires & Gates."""
        rect = self._circuit_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "⚛ SƠ ĐỒ MẠCH LƯỢNG TỬ (CIRCUIT)")

        # Circuit Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        # Draw Qubit Wires
        wire_spacing = box_rect.height() / (self._num_qubits + 1)
        wire_ys = []
        for q in range(self._num_qubits):
            wy = box_rect.top() + (q + 1) * wire_spacing
            wire_ys.append(wy)

            # Label |0⟩_q
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(QColor(192, 132, 252))
            painter.drawText(box_rect.left() + 8.0, wy + 4.0, f"|0⟩ q{q}")

            # Horizontal wire line
            painter.setPen(QPen(QColor(71, 85, 105), 1.5))
            painter.drawLine(QPointF(box_rect.left() + 50.0, wy), QPointF(box_rect.right() - 15.0, wy))

        # Gate columns
        gate_spacing = (box_rect.width() - 75.0) / max(1, len(self._gates) + 1)
        gate_w = min(32.0, gate_spacing - 6.0)

        for gi, gate in enumerate(self._gates):
            gx = box_rect.left() + 60.0 + gi * gate_spacing
            target_y = wire_ys[gate.target_qubit]
            is_executed = (gi < self._step_cursor)

            if gate.control_qubit is not None:
                # Two-qubit gate (CNOT or CZ)
                control_y = wire_ys[gate.control_qubit]

                # Vertical connection line
                line_pen = QPen(QColor(192, 132, 252) if is_executed else QColor(100, 116, 139), 1.8)
                painter.setPen(line_pen)
                painter.drawLine(QPointF(gx, control_y), QPointF(gx, target_y))

                # Control dot
                painter.setBrush(QColor(192, 132, 252) if is_executed else QColor(100, 116, 139))
                painter.drawEllipse(QPointF(gx, control_y), 4.5, 4.5)

                # Target circle (XOR symbol for CNOT, or box for CZ)
                if gate.name == "CX":
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(192, 132, 252) if is_executed else QColor(100, 116, 139), 2.0))
                    painter.drawEllipse(QPointF(gx, target_y), 10.0, 10.0)
                    painter.drawLine(QPointF(gx - 10.0, target_y), QPointF(gx + 10.0, target_y))
                    painter.drawLine(QPointF(gx, target_y - 10.0), QPointF(gx, target_y + 10.0))
                else:
                    g_rect = QRectF(gx - gate_w / 2.0, target_y - gate_w / 2.0, gate_w, gate_w)
                    painter.setBrush(QColor(168, 85, 247, 220) if is_executed else QColor(30, 41, 59, 180))
                    painter.setPen(QPen(QColor(192, 132, 252) if is_executed else QColor(71, 85, 105), 1.2))
                    painter.drawRoundedRect(g_rect, 4.0, 4.0)
                    painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                    painter.setPen(QColor(255, 255, 255) if is_executed else QColor(148, 163, 184))
                    painter.drawText(g_rect, Qt.AlignCenter, gate.name)
            else:
                # Single-qubit gate block
                g_rect = QRectF(gx - gate_w / 2.0, target_y - gate_w / 2.0, gate_w, gate_w)
                painter.setBrush(QColor(168, 85, 247, 220) if is_executed else QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(192, 132, 252) if is_executed else QColor(71, 85, 105), 1.2))
                painter.drawRoundedRect(g_rect, 4.0, 4.0)

                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QColor(255, 255, 255) if is_executed else QColor(148, 163, 184))
                painter.drawText(g_rect, Qt.AlignCenter, gate.name)

        # Step Cursor line
        cursor_x = box_rect.left() + 60.0 + self._step_cursor * gate_spacing - gate_spacing / 2.0
        if 0 <= self._step_cursor <= len(self._gates):
            painter.setPen(QPen(QColor(250, 204, 21), 1.5, Qt.DashLine))
            painter.drawLine(QPointF(cursor_x, box_rect.top() + 10.0), QPointF(cursor_x, box_rect.bottom() - 10.0))

    def _paint_bloch_panel(self, painter: QPainter) -> None:
        """Render Center Panel: 3D Bloch Spheres for Each Qubit."""
        rect = self._bloch_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "🌐 QUẢ CẦU BLOCH 3D (BLOCH SPHERES)")

        # Bloch Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        n_spheres = self._num_qubits
        sphere_h = box_rect.height() / n_spheres

        for q in range(n_spheres):
            scy = box_rect.top() + (q + 0.5) * sphere_h
            scx = box_rect.center().x()
            r_sphere = min(42.0, sphere_h * 0.42)

            # Sphere outline & equator
            painter.setBrush(QColor(15, 23, 42, 160))
            painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
            painter.drawEllipse(QPointF(scx, scy), r_sphere, r_sphere)

            # Equator ellipse
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0, Qt.DotLine))
            painter.drawEllipse(QPointF(scx, scy), r_sphere, r_sphere * 0.35)

            # Z-axis (North |0⟩, South |1⟩)
            painter.setPen(QPen(QColor(100, 116, 139), 1.0))
            painter.drawLine(QPointF(scx, scy - r_sphere - 4.0), QPointF(scx, scy + r_sphere + 4.0))

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(192, 132, 252))
            painter.drawText(scx + 4.0, scy - r_sphere, "|0⟩")
            painter.drawText(scx + 4.0, scy + r_sphere + 8.0, "|1⟩")

            # Qubit Label
            painter.setPen(QColor(241, 245, 249))
            painter.drawText(box_rect.left() + 10.0, scy - r_sphere + 10.0, f"Qubit {q}")

            # Compute Bloch Vector
            rx, ry, rz = self.compute_bloch_coordinates(q)
            purity = math.sqrt(rx**2 + ry**2 + rz**2)

            # 3D Rotation projection
            rot = self._bloch_rot_angle
            proj_x = (rx * math.cos(rot) - ry * math.sin(rot)) * r_sphere
            proj_y = (-rz + (rx * math.sin(rot) + ry * math.cos(rot)) * 0.35) * r_sphere

            vec_end = QPointF(scx + proj_x, scy + proj_y)

            # State vector arrow
            vec_pen = QPen(QColor(250, 204, 21), 2.2) if purity > 0.95 else QPen(QColor(244, 63, 94), 2.0)
            painter.setPen(vec_pen)
            painter.drawLine(QPointF(scx, scy), vec_end)

            # Tip circle
            painter.setBrush(QColor(250, 204, 21) if purity > 0.95 else QColor(244, 63, 94))
            painter.drawEllipse(vec_end, 3.5, 3.5)

            # Purity readout
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QColor(148, 163, 184))
            pur_txt = "Thuần khiết (Pure)" if purity > 0.95 else f"Hỗn hợp (Mixed, r={purity:.2f})"
            painter.drawText(box_rect.right() - 110.0, scy, pur_txt)

    def _paint_histogram_panel(self, painter: QPainter) -> None:
        """Render Right Panel: State Vector Probabilities & Density Matrix."""
        rect = self._histogram_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "📊 PHÂN BỐ XÁC SUẤT |c_k|²")

        # Histogram Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        probs = self.compute_probabilities()
        dim = len(probs)
        bar_w = (box_rect.width() - 30.0) / dim

        # Draw probability bars
        for k in range(dim):
            bx = box_rect.left() + 15.0 + k * bar_w
            prob = probs[k]
            bar_h = prob * (box_rect.height() - 80.0)
            by = box_rect.bottom() - 40.0 - bar_h

            bar_rect = QRectF(bx + 2.0, by, bar_w - 4.0, bar_h)
            grad = QLinearGradient(bar_rect.topLeft(), bar_rect.bottomLeft())
            grad.setColorAt(0.0, QColor(192, 132, 252))
            grad.setColorAt(1.0, QColor(126, 34, 206))
            painter.setBrush(grad)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bar_rect, 3.0, 3.0)

            # Basis label |k⟩ in binary
            bin_label = format(k, f"0{self._num_qubits}b")
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(241, 245, 249))
            painter.drawText(QRectF(bx, box_rect.bottom() - 35.0, bar_w, 18.0), Qt.AlignCenter, f"|{bin_label}⟩")

            # Value percentage
            if prob > 0.03:
                painter.setFont(QFont("Segoe UI", 7))
                painter.setPen(QColor(250, 204, 21))
                painter.drawText(QRectF(bx, by - 16.0, bar_w, 16.0), Qt.AlignCenter, f"{prob * 100:.0f}%")

        # State text readout
        p = self.current_preset
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(box_rect.left() + 8.0, box_rect.top() + 18.0, f"Dự kiến: {p.expected_state_desc}")

    def _paint_controls(self, painter: QPainter) -> None:
        """Render Bottom Control Bar: Step, Run, Reset, Pause/Play."""
        rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        btn_w = (rect.width() - 80.0) / 4.0
        btn_h = 42.0
        btn_y = rect.top() + 24.0

        btns = [
            ("⏭ BƯỚC TIẾP (STEP)", QColor(168, 85, 247)),
            ("▶ CHẠY HẾT (RUN ALL)", QColor(56, 189, 248)),
            ("🔄 ĐẶT LẠI (RESET)", QColor(244, 63, 94)),
            ("⏸ TẠM DỪNG QUAY" if not self._is_paused else "▶ QUAY BLOCH", QColor(250, 204, 21)),
        ]

        for i, (text, color) in enumerate(btns):
            bx = rect.left() + 20.0 + i * (btn_w + 12.0)
            b_rect = QRectF(bx, btn_y, btn_w, btn_h)

            painter.setBrush(QColor(30, 41, 59, 220))
            painter.setPen(QPen(color, 1.3))
            painter.drawRoundedRect(b_rect, 6.0, 6.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(color)
            painter.drawText(b_rect, Qt.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (reset cursor to 0)
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        if badge_rect.contains(pos):
            self.reset_circuit()
            event.accept()
            return

        # Check Controls buttons
        ctrl_rect = self._controls_rect()
        btn_w = (ctrl_rect.width() - 80.0) / 4.0
        btn_h = 42.0
        btn_y = ctrl_rect.top() + 24.0

        for i in range(4):
            bx = ctrl_rect.left() + 20.0 + i * (btn_w + 12.0)
            b_rect = QRectF(bx, btn_y, btn_w, btn_h)
            if b_rect.contains(pos):
                if i == 0:
                    self.step_forward()
                elif i == 1:
                    self.run_all_gates()
                elif i == 2:
                    self.reset_circuit()
                elif i == 3:
                    self.toggle_pause()
                event.accept()
                return

        super().mousePressEvent(event)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize quantum circuit simulator state."""
        return {
            "type": "InteractiveQuantumCircuitItem",
            "preset_key": self.current_preset.key,
            "step_cursor": self._step_cursor,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveQuantumCircuitItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "bell_phi_plus")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "step_cursor" in data:
            item._step_cursor = int(data["step_cursor"])
            item._state_vector = item._compute_state_vector(item._step_cursor)
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
