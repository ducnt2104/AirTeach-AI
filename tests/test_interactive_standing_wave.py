"""
Unit Test Suite for Interactive Standing Wave Resonance Simulator
=================================================================
Tests:
  - Wave geometry initialization and bounding box.
  - Standing wave mathematical resonance conditions (Fixed-Fixed vs Fixed-Free).
  - Wavelength (lambda = 2L/k vs 4L/(2k+1)) and wave velocity (v = lambda * f).
  - Nodes (Nút sóng, u = 0) and Antinodes (Bụng sóng, |u| = 2A).
  - Spatial-temporal wave displacement u(x, t) and envelope.
  - Harmonic mode switching (k = 1..6) and amplitude adjustment.
  - Play / Pause animation toggle and time stepping.
  - Mouse dragging and touchless air gesture interaction (Air Drag & Air Tap).
  - Stamping permanent vector diagrams to LessonBoard and MainWindow integration.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_standing_wave import InteractiveStandingWaveItem
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication is running for Qt GUI objects."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_standing_wave_initialization_and_geometry(qapp):
    """Test default parameters, bounding box, and geometry coordinates."""
    item = InteractiveStandingWaveItem(
        pos=(250.0, 180.0),
        width=600.0,
        height=380.0,
        string_length=480.0,
        harmonic_k=3,
        amplitude=30.0,
        frequency_hz=2.0,
    )
    assert item.pos() == QPointF(250.0, 180.0)
    assert item.boundingRect().width() >= 600.0
    assert item.boundingRect().height() >= 380.0
    assert item.harmonic_k() == 3
    assert item.amplitude() == pytest.approx(30.0, abs=1e-3)
    assert item.frequency() == pytest.approx(2.0, abs=1e-3)
    assert not item.is_fixed_free()
    item.close()


def test_standing_wave_fixed_fixed_resonance(qapp):
    """Verify Fixed-Fixed boundary condition: L = k * (lambda / 2)."""
    L = 600.0
    k = 3
    f = 1.5
    item = InteractiveStandingWaveItem(string_length=L, harmonic_k=k, frequency_hz=f)

    # lambda = 2L / k = 1200 / 3 = 400.0
    expected_lam = (2.0 * L) / k
    assert item.wavelength() == pytest.approx(expected_lam, abs=1e-3)

    # v = lambda * f = 400 * 1.5 = 600 px/s
    assert item.wave_speed() == pytest.approx(expected_lam * f, abs=1e-3)

    # Nodes: k + 1 = 4 nodes
    # Antinodes: k = 3 antinodes
    num_nodes, num_antis = item.count_nodes_and_antinodes()
    assert num_nodes == 4
    assert num_antis == 3

    nodes = item.node_positions()
    assert len(nodes) == 4
    # First node is at string start, last node at string end
    assert nodes[0] == pytest.approx(item._x_string_start, abs=1e-3)
    assert nodes[-1] == pytest.approx(item._x_string_end, abs=1e-3)

    # Equal spacing between nodes: L / 3
    for i in range(len(nodes) - 1):
        assert (nodes[i + 1] - nodes[i]) == pytest.approx(L / 3.0, abs=1e-3)

    # Antinodes at midpoints between consecutive nodes
    antis = item.antinode_positions()
    assert len(antis) == 3
    for i in range(len(antis)):
        mid = (nodes[i] + nodes[i + 1]) * 0.5
        assert antis[i] == pytest.approx(mid, abs=1e-3)

    item.close()


def test_standing_wave_fixed_free_resonance(qapp):
    """Verify Fixed-Free boundary condition: L = (2k + 1) * (lambda / 4)."""
    L = 500.0
    k = 2
    item = InteractiveStandingWaveItem(string_length=L, harmonic_k=k)
    item.toggle_boundary_mode()
    assert item.is_fixed_free() is True

    # lambda = 4L / (2k + 1) = 2000 / 5 = 400.0
    expected_lam = (4.0 * L) / (2 * k + 1)
    assert item.wavelength() == pytest.approx(expected_lam, abs=1e-3)

    num_nodes, num_antis = item.count_nodes_and_antinodes()
    assert num_nodes == 3
    assert num_antis == 3

    # Toggle back
    item.toggle_boundary_mode()
    assert item.is_fixed_free() is False
    item.close()


def test_wave_displacement_and_envelope(qapp):
    """Verify u(x, t) = 0 at all nodes for all time t, and peak envelope at antinodes."""
    item = InteractiveStandingWaveItem(string_length=480.0, harmonic_k=2, amplitude=40.0)
    nodes = item.node_positions()
    antis = item.antinode_positions()

    # At all nodes, displacement is identically zero regardless of time
    for t_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for nx in nodes:
            disp = item.displacement_at(nx, t_test)
            assert disp == pytest.approx(0.0, abs=1e-5)

    # At all antinodes, envelope is exactly equal to amplitude A
    for bx in antis:
        env = item.envelope_at(bx)
        assert env == pytest.approx(40.0, abs=1e-3)

    # Displacement never exceeds amplitude
    for x_test in range(int(item._x_string_start), int(item._x_string_end), 20):
        disp = item.displacement_at(float(x_test), 0.33)
        assert abs(disp) <= 40.0 + 1e-4

    item.close()


def test_harmonic_and_amplitude_mutators(qapp):
    """Test setting harmonic k (1..6) and amplitude A."""
    item = InteractiveStandingWaveItem()

    # Harmonic changes
    item.set_harmonic_k(5)
    assert item.harmonic_k() == 5
    num_nodes, num_antis = item.count_nodes_and_antinodes()
    assert num_nodes == 6
    assert num_antis == 5

    # Clamping
    item.set_harmonic_k(10)
    assert item.harmonic_k() == 6
    item.set_harmonic_k(-2)
    assert item.harmonic_k() == 1

    # Amplitude changes
    item.set_amplitude(55.0)
    assert item.amplitude() == pytest.approx(55.0, abs=1e-3)

    # Frequency changes
    item.set_frequency(3.2)
    assert item.frequency() == pytest.approx(3.2, abs=1e-3)

    item.close()


def test_play_pause_and_animation_timer(qapp):
    """Test play/pause toggle and time progression."""
    item = InteractiveStandingWaveItem()
    assert item.is_playing() is True

    # Pause
    playing = item.toggle_play()
    assert playing is False
    assert item.is_playing() is False

    t_saved = item._time
    item._on_animation_tick()
    # While paused, tick does not advance time
    assert item._time == t_saved

    # Resume
    playing = item.toggle_play()
    assert playing is True
    assert item.is_playing() is True

    item._on_animation_tick()
    assert item._time > t_saved

    item.close()


def test_mouse_and_touchless_air_gestures(qapp):
    """Test touchless air drag and air tap interactions."""
    scene = QGraphicsScene()
    item = InteractiveStandingWaveItem(pos=(100.0, 100.0), amplitude=30.0)
    scene.addItem(item)

    # Hover
    item.set_air_hover_pos(QPointF(150.0, 150.0))
    assert item._air_hover_pos is not None
    item.set_air_hover_pos(None)
    assert item._air_hover_pos is None

    # Air tap on boundary mode toggle button
    btn_b_scene = item.mapToScene(item._rect_btn_boundary.center())
    assert item.is_fixed_free() is False
    tapped = item.handle_air_tap(btn_b_scene)
    assert tapped is True
    assert item.is_fixed_free() is True

    # Air tap on harmonic k=4 button
    rect_k4 = item._rect_btn_harmonics[3]
    tapped_k = item.handle_air_tap(item.mapToScene(rect_k4.center()))
    assert tapped_k is True
    assert item.harmonic_k() == 4

    # Air tap on play/pause
    btn_play_scene = item.mapToScene(item._rect_btn_play.center())
    tapped_play = item.handle_air_tap(btn_play_scene)
    assert tapped_play is True
    assert item.is_playing() is False

    # Air drag near amplitude handle
    antis = item.antinode_positions()
    bx = antis[0]
    by = item._y_baseline - item._amplitude
    handle_scene = item.mapToScene(QPointF(bx, by))

    dragged = item.handle_air_drag(handle_scene)
    assert dragged is True

    # Drag upward to increase amplitude
    drag_up_scene = QPointF(handle_scene.x(), handle_scene.y() - 20.0)
    item.handle_air_drag(drag_up_scene)
    assert item.amplitude() > 30.0

    item.close()


def test_stamp_to_canvas_and_main_window_integration(qapp):
    """Test stamping vector diagram to LessonBoard and MainWindow action toggles."""
    win = MainWindow()
    board = win._scene
    assert not board.is_standing_wave_visible()

    # Show item
    board.show_standing_wave(True)
    assert board.is_standing_wave_visible()
    item = board.get_standing_wave_item()
    assert item is not None
    assert item.isVisible()

    # Stamp to canvas
    stamped_items = item.stamp_to_canvas(board)
    assert len(stamped_items) >= 4  # baseline, envelope upper, envelope lower, wave profile, text card
    for itm in stamped_items:
        assert itm.scene() == board

    # Hide and Toggle
    board.show_standing_wave(False)
    assert not board.is_standing_wave_visible()

    is_vis = board.toggle_standing_wave()
    assert is_vis is True
    assert board.is_standing_wave_visible()

    # MainWindow slot
    win._on_toggle_standing_wave()
    assert not board.is_standing_wave_visible()

    item.close()
    win.close()
