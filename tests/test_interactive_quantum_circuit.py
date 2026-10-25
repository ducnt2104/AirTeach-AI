"""
AirTeach AI - Interactive Quantum Computing Circuit Tests
=========================================================
Authentic verification of multi-qubit state vectors, Bell states,
quantum teleportation fidelity, superdense coding, GHZ state,
Bloch coordinates, entanglement metrics, controls, and serialization.
"""
import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

from src.ui.interactive_quantum_circuit import (
    InteractiveQuantumCircuitItem,
    PRESETS,
    QuantumPreset,
    H_GATE,
    SIGMA_X,
    SIGMA_Z,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def qc_item(qapp):
    item = InteractiveQuantumCircuitItem()
    yield item
    item.stop()


def test_bell_state_phi_plus(qc_item):
    """
    Test authentic Bell state creation: |Phi+> = (|00> + |11>) / sqrt(2).
    Probabilities: P(00) = 0.5, P(11) = 0.5, P(01) = 0, P(10) = 0.
    Concurrence C = 1.0, Von Neumann Entropy S = 1.0 bit.
    """
    qc_item.set_preset(0)  # Bell state |Phi+>
    qc_item.run_all_gates()

    probs = qc_item.compute_probabilities()
    assert abs(probs[0] - 0.5) < 1e-4, f"P(00) should be 0.5, got {probs[0]}"
    assert abs(probs[3] - 0.5) < 1e-4, f"P(11) should be 0.5, got {probs[3]}"
    assert abs(probs[1]) < 1e-4, f"P(01) should be 0, got {probs[1]}"
    assert abs(probs[2]) < 1e-4, f"P(10) should be 0, got {probs[2]}"

    concurrence = qc_item.compute_entanglement_concurrence_2qubits()
    assert abs(concurrence - 1.0) < 1e-4, f"Concurrence should be 1.0, got {concurrence}"

    entropy = qc_item.compute_von_neumann_entropy(0)
    assert abs(entropy - 1.0) < 1e-4, f"Von Neumann entropy should be 1.0, got {entropy}"


def test_quantum_teleportation(qc_item):
    """
    Test Bennett 1993 3-qubit quantum teleportation protocol.
    Prepares |psi> on q0 and teleports it to q2 through Bell pair and feedforward.
    """
    qc_item.set_preset(1)  # Teleportation
    qc_item.run_all_gates()

    # Reduced density matrix on Bob's qubit (q2) should match |psi> = cos(pi/6)|0> + sin(pi/6)|1>
    rho2 = qc_item.compute_reduced_density_matrix(2)
    expected_p0 = math.cos(math.pi / 6.0) ** 2  # ~= 0.75
    expected_p1 = math.sin(math.pi / 6.0) ** 2  # ~= 0.25

    assert abs(np.real(rho2[0, 0]) - expected_p0) < 0.05, f"Bob's q2 P(0) should be ~0.75, got {rho2[0, 0]}"
    assert abs(np.real(rho2[1, 1]) - expected_p1) < 0.05, f"Bob's q2 P(1) should be ~0.25, got {rho2[1, 1]}"


def test_superdense_coding(qc_item):
    """
    Test superdense coding: Alice transmits '11' over shared Bell pair.
    Bob decodes with CX and H, obtaining state |11> with 100% probability.
    """
    qc_item.set_preset(2)  # Superdense coding
    qc_item.run_all_gates()

    probs = qc_item.compute_probabilities()
    # State |11> is index 3
    assert abs(probs[3] - 1.0) < 1e-4, f"P(11) should be 1.0, got {probs[3]}"
    assert abs(probs[0]) < 1e-4
    assert abs(probs[1]) < 1e-4
    assert abs(probs[2]) < 1e-4


def test_ghz_state(qc_item):
    """
    Test 3-qubit GHZ state: |GHZ> = (|000> + |111>) / sqrt(2).
    P(000) = 0.5 (index 0), P(111) = 0.5 (index 7).
    """
    qc_item.set_preset(3)  # GHZ
    qc_item.run_all_gates()

    probs = qc_item.compute_probabilities()
    assert abs(probs[0] - 0.5) < 1e-4, f"P(000) should be 0.5, got {probs[0]}"
    assert abs(probs[7] - 0.5) < 1e-4, f"P(111) should be 0.5, got {probs[7]}"
    for i in range(1, 7):
        assert abs(probs[i]) < 1e-4, f"Intermediate state {i} should be 0, got {probs[i]}"


def test_deutsch_algorithm(qc_item):
    """
    Test Deutsch algorithm determining balanced function.
    q0 measurement must deterministically yield |1> with probability 1.0.
    """
    qc_item.set_preset(4)  # Deutsch algorithm
    qc_item.run_all_gates()

    rho0 = qc_item.compute_reduced_density_matrix(0)
    # P(1) on q0 is rho0[1, 1]
    assert abs(np.real(rho0[1, 1]) - 1.0) < 1e-4, f"Deutsch q0 should be |1>, got P(1)={rho0[1, 1]}"


def test_bloch_coordinates_pure_vs_mixed(qc_item):
    """
    Test Bloch coordinates:
    For ground state |0...0>, q0 has Bloch vector (0, 0, 1), purity = 1.0.
    For maximally entangled Bell state, single-qubit reduced state is maximally mixed:
    Bloch vector (0, 0, 0), purity = 0.0.
    """
    qc_item.reset_circuit()
    rx, ry, rz = qc_item.compute_bloch_coordinates(0)
    assert abs(rx) < 1e-4 and abs(ry) < 1e-4 and abs(rz - 1.0) < 1e-4

    # Run Bell state
    qc_item.set_preset(0)
    qc_item.run_all_gates()
    rx_m, ry_m, rz_m = qc_item.compute_bloch_coordinates(0)
    purity = math.sqrt(rx_m**2 + ry_m**2 + rz_m**2)
    assert purity < 1e-3, f"Entangled qubit Bloch vector should be (0,0,0), got purity {purity}"


def test_step_forward_and_reset(qc_item):
    """Test stepping gate by gate increments cursor and recalculates state."""
    qc_item.set_preset(0)  # Bell state: 2 gates (H, CX)
    qc_item.reset_circuit()
    assert qc_item._step_cursor == 0

    qc_item.step_forward()
    assert qc_item._step_cursor == 1
    # After H(q0), state is (|00> + |10>) / sqrt(2)
    probs_step1 = qc_item.compute_probabilities()
    assert abs(probs_step1[0] - 0.5) < 1e-4
    assert abs(probs_step1[2] - 0.5) < 1e-4  # |10> is binary 10_2 = 2

    qc_item.step_forward()
    assert qc_item._step_cursor == 2
    probs_step2 = qc_item.compute_probabilities()
    assert abs(probs_step2[3] - 0.5) < 1e-4  # |11> is binary 11_2 = 3


def test_preset_switching(qc_item):
    """Test switching across all 5 presets."""
    emitted = []
    qc_item.preset_changed.connect(lambda k: emitted.append(k))

    for idx, p in enumerate(PRESETS):
        qc_item.set_preset(idx)
        assert qc_item.current_preset.key == p.key
        assert qc_item._num_qubits == p.num_qubits

    assert len(emitted) == len(PRESETS)


def test_mouse_interactions(qc_item):
    """Test clicking preset pills, step, run all, and reset buttons."""
    scene = QGraphicsScene()
    scene.addItem(qc_item)
    try:
        # Click Preset 1 using exact geometry
        rects = qc_item._preset_button_rects()
        assert len(rects) >= 2
        p1_center = rects[1][0].center()

        event_p1 = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_p1.setPos(p1_center)
        event_p1.setButton(Qt.MouseButton.LeftButton)
        qc_item.mousePressEvent(event_p1)
        assert qc_item.current_preset.key == "quantum_teleportation"

        # Click Reset button (Col 2)
        ctrl_rect = qc_item._controls_rect()
        btn_w = (ctrl_rect.width() - 80.0) / 4.0
        btn_h = 42.0
        btn_y = ctrl_rect.top() + 24.0

        reset_rect = QRectF(ctrl_rect.left() + 20.0 + 2 * (btn_w + 12.0), btn_y, btn_w, btn_h)
        event_reset = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_reset.setPos(reset_rect.center())
        event_reset.setButton(Qt.MouseButton.LeftButton)
        qc_item.mousePressEvent(event_reset)
        assert qc_item._step_cursor == 0

        # Click Step button (Col 0)
        step_rect = QRectF(ctrl_rect.left() + 20.0, btn_y, btn_w, btn_h)
        event_step = QGraphicsSceneMouseEvent(QGraphicsSceneMouseEvent.Type.GraphicsSceneMousePress)
        event_step.setPos(step_rect.center())
        event_step.setButton(Qt.MouseButton.LeftButton)
        qc_item.mousePressEvent(event_step)
        assert qc_item._step_cursor == 1
    finally:
        scene.removeItem(qc_item)


def test_serialization_roundtrip(qc_item):
    """Test serialization to dict and reconstruction from dict."""
    qc_item.set_preset(2)
    qc_item.run_all_gates()
    qc_item.setPos(180.0, 220.0)

    data = qc_item.to_dict()
    assert data["type"] == "InteractiveQuantumCircuitItem"
    assert data["preset_key"] == "superdense_coding"
    assert data["pos"] == [180.0, 220.0]

    reconstructed = InteractiveQuantumCircuitItem.from_dict(data)
    try:
        assert reconstructed.current_preset.key == "superdense_coding"
        assert reconstructed.pos().x() == 180.0
        assert reconstructed.pos().y() == 220.0
    finally:
        reconstructed.stop()


def test_paint_render(qc_item):
    """Verify that paint renders into QPainter/QImage without crashing."""
    img = QImage(940, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    try:
        qc_item.paint(painter, None, None)
    finally:
        painter.end()
    assert img.pixelColor(20, 20).isValid()


def test_stop_cleanup(qc_item):
    """Verify stopping the item halts the animation timer."""
    assert qc_item._anim_timer.isActive()
    qc_item.stop()
    assert not qc_item._anim_timer.isActive()
