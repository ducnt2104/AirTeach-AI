"""
Unit tests for InteractiveQuantumTunnelingItem (Quantum Mechanics & Barrier Tunneling).
"""
import pytest
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_quantum_tunneling import (
    InteractiveQuantumTunnelingItem,
    PRESETS,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_initial_state(qapp):
    """Verify default initial state of quantum simulator."""
    item = InteractiveQuantumTunnelingItem()
    assert item._preset_code == "RECTANGULAR_BARRIER"
    assert item._energy == pytest.approx(4.20)
    assert item._v0 == pytest.approx(6.0)
    assert item._barrier_width == pytest.approx(1.2)
    assert item._mode == "stationary"
    assert item._show_probability_density is True
    assert item._show_wavefunction_re is True


def test_presets_loading(qapp):
    """Verify all 5 curriculum presets load with authentic physics parameters."""
    item = InteractiveQuantumTunnelingItem()

    for code, preset in PRESETS.items():
        item.load_preset(code)
        assert item._preset_code == code
        assert item._preset.scenario_type == preset.scenario_type
        assert item._energy == pytest.approx(preset.energy)
        assert item._v0 == pytest.approx(preset.v0)
        assert item._barrier_width == pytest.approx(preset.barrier_width)
        assert item._quantum_n == preset.quantum_n


def test_tunneling_probability_conservation(qapp):
    """Verify exact flux conservation T + R = 1.0 across tunneling and above-barrier regimes."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("RECTANGULAR_BARRIER")

    # 1. Tunneling regime: E < V0
    item.set_v0(8.0)
    item.set_energy(3.0)
    T, R = item.compute_tunneling_probability()
    assert 0.0 < T < 1.0
    assert 0.0 < R < 1.0
    assert T + R == pytest.approx(1.0, abs=1e-5)

    # 2. Above-barrier regime: E > V0 (quantum reflection)
    item.set_energy(10.0)
    T_above, R_above = item.compute_tunneling_probability()
    assert 0.0 < T_above <= 1.0
    assert T_above + R_above == pytest.approx(1.0, abs=1e-5)
    # Transmission should be higher than below barrier
    assert T_above > T

    # 3. Resonance limit: E = V0
    item.set_energy(8.0)
    T_res, R_res = item.compute_tunneling_probability()
    assert 0.0 < T_res < 1.0
    assert T_res + R_res == pytest.approx(1.0, abs=1e-5)


def test_infinite_well_boundary_conditions(qapp):
    """Verify infinite well wavefunctions vanish outside [ -L/2, L/2 ]."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("INFINITE_WELL")
    L = item._barrier_width  # 4.0 -> inside is [-2, 2]

    x_test = np.array([-3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0])
    v_arr, psi_arr, prob_arr = item.evaluate_wavefunction(x_test)

    # Outside well (|x| >= 2.0), psi must be strictly 0.0
    assert psi_arr[0] == pytest.approx(0.0)
    assert psi_arr[1] == pytest.approx(0.0)
    assert psi_arr[5] == pytest.approx(0.0)
    assert psi_arr[6] == pytest.approx(0.0)

    # Inside well, probability density is positive
    assert prob_arr[2] > 0.0
    assert prob_arr[3] >= 0.0
    assert prob_arr[4] > 0.0


def test_finite_well_evanescent_decay(qapp):
    """Verify finite well exhibits exponential decay in the classically forbidden region."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("FINITE_WELL")
    item.set_quantum_n(1)  # ground state (even parity)

    x_test = np.array([0.0, 1.5, 2.0, 2.8, 3.8])
    # Well boundary is a = 3.6 / 2 = 1.8
    _, psi_arr, prob_arr = item.evaluate_wavefunction(x_test)

    # Outside the well (x=2.0, 2.8, 3.8), wavefunction decays monotonically
    assert abs(psi_arr[2]) > abs(psi_arr[3]) > abs(psi_arr[4])
    assert prob_arr[2] > prob_arr[3] > prob_arr[4]


def test_alpha_decay_gamow_model(qapp):
    """Verify Alpha decay Gamow tunneling calculation."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("ALPHA_DECAY")
    T, R = item.compute_tunneling_probability()
    assert 0.0 < T < 1.0
    assert T + R == pytest.approx(1.0, abs=1e-5)

    # Increasing particle energy dramatically increases tunneling probability (Geiger-Nuttall law)
    item.set_energy(item._energy + 2.0)
    T_higher, _ = item.compute_tunneling_probability()
    assert T_higher > T


def test_harmonic_oscillator_states(qapp):
    """Verify Harmonic oscillator wavefunctions and turning points."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("HARMONIC_OSCILLATOR")

    for n in (0, 1, 2, 3):
        item.set_quantum_n(n)
        assert item._quantum_n == n
        x_pts = np.linspace(-3.0, 3.0, 50)
        v_arr, psi_arr, prob_arr = item.evaluate_wavefunction(x_pts)
        assert np.all(v_arr >= 0.0)
        assert np.max(prob_arr) > 0.0


def test_dynamic_wavepacket_evolution(qapp):
    """Verify wavepacket time evolution advances cleanly without NaN or Inf."""
    item = InteractiveQuantumTunnelingItem()
    item.load_preset("RECTANGULAR_BARRIER")
    item.toggle_mode()
    assert item._mode == "dynamic"
    assert item._is_playing is True

    x_pts = np.linspace(-4.5, 4.5, 100)
    psi_1, prob_1 = item.evaluate_wavepacket(x_pts)
    assert np.all(np.isfinite(psi_1))
    assert np.all(np.isfinite(prob_1))

    # Advance tick
    item._on_sim_tick()
    assert item._sim_time > 0.0
    psi_2, prob_2 = item.evaluate_wavepacket(x_pts)
    assert np.all(np.isfinite(psi_2))
    assert not np.array_equal(psi_1, psi_2)

    # Pause and reset
    item.pause()
    assert item._is_playing is False
    item.reset_sim()
    assert item._sim_time == 0.0


def test_energy_and_v0_clamping(qapp):
    """Verify energy and V0 setters adhere to physical clamping boundaries."""
    item = InteractiveQuantumTunnelingItem()
    item.set_energy(-5.0)
    assert item._energy == pytest.approx(0.1)

    item.set_energy(200.0)
    assert item._energy == pytest.approx(12.0)

    item.set_v0(-10.0)
    assert item._v0 == pytest.approx(0.5)

    item.set_v0(999.0)
    assert item._v0 == pytest.approx(15.0)

    item.set_quantum_n(-2)
    assert item._quantum_n == 0
    item.set_quantum_n(10)
    assert item._quantum_n == 5


def test_serialization_roundtrip(qapp):
    """Verify to_dict and from_dict preserve complete simulation state."""
    item = InteractiveQuantumTunnelingItem(pos=(150.0, 80.0))
    item.load_preset("FINITE_WELL")
    item.set_energy(3.8)
    item.set_v0(9.2)
    item.set_quantum_n(2)
    item.toggle_mode()
    item.toggle_wavefunction()

    data = item.to_dict()
    assert data["type"] == "interactive_quantum_tunneling"
    assert data["preset_code"] == "FINITE_WELL"
    assert data["energy"] == pytest.approx(3.8)
    assert data["v0"] == pytest.approx(9.2)
    assert data["quantum_n"] == 2
    assert data["mode"] == "dynamic"
    assert data["show_wavefunction_re"] is False

    restored = InteractiveQuantumTunnelingItem.from_dict(data)
    assert restored._preset_code == "FINITE_WELL"
    assert restored._energy == pytest.approx(3.8)
    assert restored._v0 == pytest.approx(9.2)
    assert restored._quantum_n == 2
    assert restored._mode == "dynamic"
    assert restored._show_wavefunction_re is False


def test_handle_air_tap_preset_and_buttons(qapp):
    """Verify touchless air-tap interaction on presets, buttons, and sliders."""
    item = InteractiveQuantumTunnelingItem()

    # Tap on FINITE_WELL preset
    p_rect = item._btn_rects["preset_FINITE_WELL"]
    tap_pt = item.mapToScene(p_rect.center())
    handled = item.handle_air_tap(tap_pt)
    assert handled is True
    assert item._preset_code == "FINITE_WELL"

    # Tap on toggle_mode
    m_rect = item._btn_rects["toggle_mode"]
    tap_m = item.mapToScene(m_rect.center())
    prev_mode = item._mode
    assert item.handle_air_tap(tap_m) is True
    assert item._mode != prev_mode

    # Tap on n_plus
    prev_n = item._quantum_n
    n_plus_rect = item._btn_rects["n_plus"]
    tap_n = item.mapToScene(n_plus_rect.center())
    assert item.handle_air_tap(tap_n) is True
    assert item._quantum_n == prev_n + 1

    # Tap on energy slider
    e_rect = item._slider_rect_energy
    tap_e = item.mapToScene(e_rect.center())
    assert item.handle_air_tap(tap_e) is True
    assert 5.0 <= item._energy <= 7.0

    # Tap outside returns False
    out_pt = item.mapToScene(QPointF(-100.0, -100.0))
    assert item.handle_air_tap(out_pt) is False


def test_close_signal_emission(qapp):
    """Verify closed signal is emitted upon tapping close button."""
    item = InteractiveQuantumTunnelingItem()
    emitted = False

    def on_closed():
        nonlocal emitted
        emitted = True

    item.closed.connect(on_closed)
    c_rect = item._btn_rects["btn_close"]
    tap_c = item.mapToScene(c_rect.center())
    assert item.handle_air_tap(tap_c) is True
    assert emitted is True


def test_qpainter_rendering(qapp):
    """Verify QPainter renders without crashes or exceptions."""
    item = InteractiveQuantumTunnelingItem()
    img = QImage(800, 600, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()


def test_canvas_stamping(qapp):
    """Verify canvas stamping clones the quantum simulation onto scene."""
    scene = QGraphicsScene()
    item = InteractiveQuantumTunnelingItem(pos=(140.0, 70.0))
    item.load_preset("ALPHA_DECAY")
    item.set_energy(5.5)

    stamped = item.stamp_to_canvas(scene)
    assert len(stamped) == 1
    clone = stamped[0]
    assert isinstance(clone, InteractiveQuantumTunnelingItem)
    assert clone._preset_code == "ALPHA_DECAY"
    assert clone._energy == pytest.approx(5.5)


def test_lesson_board_quantum_integration(qapp):
    """Verify LessonBoard lifecycle methods for Quantum Tunneling simulator."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_quantum_tunneling_visible() is False
    assert board.get_quantum_tunneling_item() is None

    # Show item
    item = board.show_quantum_tunneling(True, pos=(200.0, 100.0))
    assert item is not None
    assert board.is_quantum_tunneling_visible() is True
    assert board.get_quantum_tunneling_item() is item
    assert item.pos() == QPointF(200.0, 100.0)

    # Toggle off
    is_vis = board.toggle_quantum_tunneling()
    assert is_vis is False
    assert board.is_quantum_tunneling_visible() is False

    # Toggle on
    is_vis = board.toggle_quantum_tunneling()
    assert is_vis is True
    assert board.is_quantum_tunneling_visible() is True


def test_main_window_quantum_toggle(qapp):
    """Verify MainWindow menu action and slot for Quantum Tunneling visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_quantum_tunneling")
        assert win._scene.is_quantum_tunneling_visible() is False

        # Toggle on
        is_vis = win._on_toggle_quantum_tunneling()
        assert is_vis is True
        assert win._scene.is_quantum_tunneling_visible() is True

        # Toggle off
        is_vis = win._on_toggle_quantum_tunneling()
        assert is_vis is False
        assert win._scene.is_quantum_tunneling_visible() is False
    finally:
        win.close()

