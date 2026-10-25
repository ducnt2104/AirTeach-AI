"""
Unit Tests for Interactive Acoustic Wave Physics: Doppler Spectrogram, Ultrasonic Echolocation & Sonar Beamforming
=====================================================================================================================
Validates authentic acoustics, Doppler shift calculations, Mach cone geometry,
phased array beamforming array factors, TOF echolocation, touchless air gestures,
canvas stamping, and .airteach serialization.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_acoustic_sonar import (
    PRESETS,
    SPEED_OF_SOUND_AIR,
    InteractiveAcousticSonarItem,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def acoustic_item(qapp):
    item = InteractiveAcousticSonarItem(pos=(100.0, 100.0))
    # Keep reference to scene on the item so it is not garbage collected
    scene = QGraphicsScene()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_initialization_and_physical_constants(acoustic_item):
    """Test initial dimensions, authentic sound speed, and scene placement."""
    assert acoustic_item._width == 940.0
    assert acoustic_item._height == 650.0
    assert acoustic_item._c == SPEED_OF_SOUND_AIR
    assert math.isclose(acoustic_item._c, 343.0, rel_tol=1e-3)
    assert acoustic_item.pos() == QPointF(100.0, 100.0)
    assert acoustic_item._is_playing is True


def test_all_five_presets_exist():
    """Verify the 5 curriculum presets exist with valid parameters."""
    expected_keys = [
        "AMBULANCE_DOPPLER",
        "SUPERSONIC_MACH_CONE",
        "BAT_ECHOLOCATION",
        "SONAR_BEAMFORMING",
        "DOPPLER_SPECTROGRAM",
    ]
    for key in expected_keys:
        assert key in PRESETS
        p = PRESETS[key]
        assert p.code == key
        assert len(p.name_vi) > 0
        assert p.source_freq_hz > 0
        assert p.mode in ("doppler", "mach", "echolocation", "beamforming", "spectrogram")


def test_preset_switching_and_signals(acoustic_item):
    """Verify switching presets updates state and emits signals."""
    preset_signal_emitted = []
    vel_signal_emitted = []

    acoustic_item.preset_changed.connect(lambda code: preset_signal_emitted.append(code))
    acoustic_item.velocity_changed.connect(lambda v: vel_signal_emitted.append(v))

    acoustic_item.set_preset("SUPERSONIC_MACH_CONE")
    assert acoustic_item._current_preset == "SUPERSONIC_MACH_CONE"
    assert acoustic_item._mode == "mach"
    assert acoustic_item.mach_number > 1.0
    assert len(preset_signal_emitted) == 1
    assert preset_signal_emitted[0] == "SUPERSONIC_MACH_CONE"
    assert len(vel_signal_emitted) == 1


def test_doppler_frequency_shift_approach_and_recede(acoustic_item):
    """Verify classical Doppler effect: approach f_o > f_s, recede f_o < f_s."""
    acoustic_item.set_preset("AMBULANCE_DOPPLER")
    acoustic_item._source_freq = 700.0
    acoustic_item._source_velocity = 34.3  # Moving left to right at +34.3 m/s

    # Case 1: Source at x = -10m, observer at x = 0m (Approaching)
    acoustic_item._source_x_m = -10.0
    acoustic_item._source_y_m = 0.0
    acoustic_item._observer_x_m = 0.0
    acoustic_item._observer_y_m = 1.0
    f_approach = acoustic_item.calculate_observed_frequency()
    assert f_approach > acoustic_item._source_freq, f"Expected approach f_o > 700, got {f_approach}"

    # Case 2: Source at x = +10m, observer at x = 0m (Receding)
    acoustic_item._source_x_m = 10.0
    f_recede = acoustic_item.calculate_observed_frequency()
    assert f_recede < acoustic_item._source_freq, f"Expected recede f_o < 700, got {f_recede}"


def test_mach_number_and_mach_cone_angle(acoustic_item):
    """Verify Mach number M = v/c and Mach angle μ = arcsin(1/M)."""
    # Subsonic: M < 1.0 => Mach angle is None
    acoustic_item.set_velocity(171.5)  # 0.5 Mach
    assert math.isclose(acoustic_item.mach_number, 0.5, rel_tol=1e-3)
    assert acoustic_item.mach_angle_deg is None

    # Supersonic: M = 2.0 => μ = arcsin(0.5) = 30°
    acoustic_item.set_velocity(686.0)  # 2.0 Mach
    assert math.isclose(acoustic_item.mach_number, 2.0, rel_tol=1e-3)
    assert acoustic_item.mach_angle_deg is not None
    assert math.isclose(acoustic_item.mach_angle_deg, 30.0, abs_tol=0.1)


def test_echolocation_time_of_flight(acoustic_item):
    """Verify round trip TOF calculation: Δt = 2*d / c."""
    acoustic_item.set_preset("BAT_ECHOLOCATION")
    acoustic_item._obstacle_distance_m = 6.86  # meters
    # Round-trip distance = 13.72 m
    # Time = 13.72 / 343 = 0.040 s = 40.0 ms
    tof_ms = acoustic_item.calculate_echolocation_tof_ms()
    assert math.isclose(tof_ms, 40.0, rel_tol=1e-3)


def test_phased_array_beamforming_directivity(acoustic_item):
    """Verify phased array Array Factor AF(θ): maximum 1.0 at steered angle θ_0."""
    acoustic_item.set_preset("SONAR_BEAMFORMING")
    acoustic_item.set_beam_angle(30.0)

    # Main lobe at θ = 30°
    af_peak = acoustic_item.calculate_array_factor(math.radians(30.0))
    assert math.isclose(af_peak, 1.0, rel_tol=1e-3)

    # Off-axis at θ = -30° should be significantly lower
    af_off = acoustic_item.calculate_array_factor(math.radians(-30.0))
    assert af_off < 0.5


def test_step_simulation_and_spectrogram_update(acoustic_item):
    """Verify simulation steps advance time and generate wavefronts."""
    acoustic_item.set_preset("AMBULANCE_DOPPLER")
    init_time = acoustic_item._sim_time_sec

    # Step simulation several times
    for _ in range(5):
        acoustic_item.step_simulation(0.05)

    assert acoustic_item._sim_time_sec > init_time
    assert len(acoustic_item._wavefronts) > 0
    assert len(acoustic_item._spectrogram_history) > 0


def test_play_pause_and_reset(acoustic_item):
    """Verify play/pause stops advancing and reset clears buffers."""
    acoustic_item.toggle_play_pause()
    assert acoustic_item._is_playing is False

    t_before = acoustic_item._sim_time_sec
    acoustic_item.step_simulation(0.1)
    assert acoustic_item._sim_time_sec == t_before

    acoustic_item.toggle_play_pause()
    assert acoustic_item._is_playing is True

    acoustic_item._reset_simulation()
    assert acoustic_item._sim_time_sec == 0.0
    assert len(acoustic_item._wavefronts) == 0


def test_sliders_clamping(acoustic_item):
    """Verify sliders clamp to their allowed physical limits."""
    # Velocity clamping: [-700, 700]
    acoustic_item.set_velocity(1000.0)
    assert acoustic_item._source_velocity == 700.0
    acoustic_item.set_velocity(-1000.0)
    assert acoustic_item._source_velocity == -700.0

    # Frequency clamping: [100, 50000]
    acoustic_item.set_frequency(10.0)
    assert acoustic_item._source_freq == 100.0
    acoustic_item.set_frequency(60000.0)
    assert acoustic_item._source_freq == 50000.0

    # Beam angle clamping: [-60, 60]
    acoustic_item.set_beam_angle(90.0)
    assert acoustic_item._beam_angle_deg == 60.0
    acoustic_item.set_beam_angle(-90.0)
    assert acoustic_item._beam_angle_deg == -60.0


def test_touchless_air_tap_gesture(acoustic_item):
    """Verify air tap triggers presets, playback, and close actions."""
    # Air tap on Preset 2: SUPERSONIC_MACH_CONE
    # Button rect: start_x + 1 * (176 + 8) = 20 + 184 = 204, y = 58..86
    p_tap = acoustic_item.mapToScene(QPointF(220.0, 70.0))
    handled = acoustic_item.handle_air_tap(p_tap)
    assert handled is True
    assert acoustic_item._current_preset == "SUPERSONIC_MACH_CONE"

    # Air tap on Play/Pause button
    canvas_bottom = 96 + 470
    p_play_tap = acoustic_item.mapToScene(QPointF(40.0, canvas_bottom + 20.0))
    was_playing = acoustic_item._is_playing
    acoustic_item.handle_air_tap(p_play_tap)
    assert acoustic_item._is_playing != was_playing


def test_serialization_and_canvas_stamping(acoustic_item):
    """Verify .airteach serialization and canvas cloning."""
    acoustic_item.set_preset("BAT_ECHOLOCATION")
    acoustic_item.set_frequency(42000.0)
    acoustic_item._obstacle_distance_m = 8.5

    data = acoustic_item.to_dict()
    assert data["type"] == "InteractiveAcousticSonarItem"
    assert data["preset"] == "BAT_ECHOLOCATION"
    assert data["source_freq"] == 42000.0
    assert data["obstacle_distance_m"] == 8.5

    # Reconstruct from dict
    restored = InteractiveAcousticSonarItem.from_dict(data)
    assert restored._current_preset == "BAT_ECHOLOCATION"
    assert restored._source_freq == 42000.0
    assert restored._obstacle_distance_m == 8.5

    # Stamp to canvas
    stamped = acoustic_item.stamp_to_canvas()
    assert stamped is not None
    assert isinstance(stamped, InteractiveAcousticSonarItem)
    assert stamped._current_preset == "BAT_ECHOLOCATION"
    assert stamped.pos() == QPointF(acoustic_item.x() + 30.0, acoustic_item.y() + 30.0)


def test_paint_rendering_all_modes(acoustic_item):
    """Verify QPainter renders without error across all 5 simulation presets."""
    img = QImage(940, 650, QImage.Format_ARGB32)
    painter = QPainter(img)

    preset_keys = list(PRESETS.keys())
    for key in preset_keys:
        acoustic_item.set_preset(key)
        acoustic_item.step_simulation(0.04)
        acoustic_item.paint(painter, None, None)

    painter.end()
    assert not img.isNull()


def test_slider_move_handler(acoustic_item):
    """Verify _handle_slider_move maps track positions accurately to value ranges."""
    # Test setting velocity via slider move helper
    acoustic_item._handle_slider_move(
        mouse_x=100.0,
        track_left=0.0,
        track_width=200.0,
        min_val=-500.0,
        max_val=500.0,
        setter_func=acoustic_item.set_velocity,
    )
    # fraction = 100/200 = 0.5 => -500 + 0.5 * 1000 = 0.0
    assert math.isclose(acoustic_item._source_velocity, 0.0, abs_tol=1e-3)


def test_close_action_and_signals(acoustic_item):
    """Verify close button air tap triggers closed signal."""
    closed_called = []
    acoustic_item.closed.connect(lambda: closed_called.append(True))

    close_pos = acoustic_item.mapToScene(QPointF(acoustic_item._width - 24, 24))
    res = acoustic_item.handle_air_tap(close_pos)
    assert res is True
    assert len(closed_called) == 1


def test_lesson_board_acoustic_sonar_integration(qapp):
    """Verify LessonBoard lifecycle methods for Acoustic Sonar visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_acoustic_sonar_visible() is False
    assert board.get_acoustic_sonar_item() is None

    # Show item
    item = board.show_acoustic_sonar(True, pos=(120.0, 90.0))
    assert item is not None
    assert board.is_acoustic_sonar_visible() is True
    assert board.get_acoustic_sonar_item() is item
    assert item.pos() == QPointF(120.0, 90.0)

    # Toggle off
    is_vis = board.toggle_acoustic_sonar()
    assert is_vis is False
    assert board.is_acoustic_sonar_visible() is False

    # Toggle on
    is_vis = board.toggle_acoustic_sonar()
    assert is_vis is True
    assert board.is_acoustic_sonar_visible() is True


def test_main_window_acoustic_sonar_toggle(qapp):
    """Verify MainWindow menu action and slot for Acoustic Sonar visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_acoustic_sonar")
        assert win._scene.is_acoustic_sonar_visible() is False

        # Toggle on
        is_vis = win._on_toggle_acoustic_sonar()
        assert is_vis is True
        assert win._scene.is_acoustic_sonar_visible() is True

        # Toggle off
        is_vis = win._on_toggle_acoustic_sonar()
        assert is_vis is False
        assert win._scene.is_acoustic_sonar_visible() is False
    finally:
        win.close()

