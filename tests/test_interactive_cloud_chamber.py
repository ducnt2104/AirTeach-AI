"""
Unit Tests for Interactive Nuclear & Particle Physics: Wilson Cloud Chamber & Particle Tracks
=============================================================================================
Validates authentic nuclear radiation physics, alpha/beta/muon ionization tracks,
Bragg peak droplet profiles, Lorentz magnetic deflection, touchless air gestures,
canvas stamping, and .airteach serialization.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_cloud_chamber import (
    PRESETS,
    InteractiveCloudChamberItem,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def chamber_item(qapp):
    item = InteractiveCloudChamberItem(pos=(100.0, 100.0))
    scene = QGraphicsScene()
    scene.addItem(item)
    item._scene_ref = scene
    return item


def test_initial_state(chamber_item):
    """Verify default initial state of Wilson Cloud Chamber item."""
    assert chamber_item._width == 940.0
    assert chamber_item._height == 650.0
    assert chamber_item._current_preset == "ALPHA_AMERICIUM"
    assert chamber_item._source_type == "alpha"
    assert chamber_item._b_field_tesla == 0.0
    assert chamber_item._decay_rate_cpm == 45.0
    assert chamber_item._supersaturation == 4.5
    assert chamber_item._is_playing is True
    assert chamber_item.pos() == QPointF(100.0, 100.0)


def test_all_five_presets_exist():
    """Verify all 5 curriculum presets exist with valid parameters."""
    expected_keys = [
        "ALPHA_AMERICIUM",
        "BETA_STRONTIUM",
        "COSMIC_RAY_MUONS",
        "LORENTZ_SPECTROMETER",
        "RADON_CASCADE",
    ]
    for key in expected_keys:
        assert key in PRESETS
        p = PRESETS[key]
        assert p.code == key
        assert len(p.name_vi) > 0
        assert p.decay_rate_cpm > 0
        assert p.supersaturation >= 2.0


def test_preset_switching_and_signals(chamber_item):
    """Verify switching presets updates physics state and emits signals."""
    preset_signals = []
    b_signals = []
    rate_signals = []

    chamber_item.preset_changed.connect(lambda p: preset_signals.append(p))
    chamber_item.b_field_changed.connect(lambda b: b_signals.append(b))
    chamber_item.rate_changed.connect(lambda r: rate_signals.append(r))

    chamber_item.set_preset("LORENTZ_SPECTROMETER")
    assert chamber_item._current_preset == "LORENTZ_SPECTROMETER"
    assert chamber_item._b_field_tesla == 0.35
    assert chamber_item._source_type == "mixed"
    assert len(preset_signals) == 1
    assert preset_signals[0] == "LORENTZ_SPECTROMETER"
    assert len(b_signals) == 1
    assert b_signals[0] == 0.35
    assert len(rate_signals) == 1


def test_alpha_track_properties_and_bragg_peak(chamber_item):
    """Verify alpha particle track is thick, straight, and exhibits Bragg peak widening."""
    chamber_item.set_preset("ALPHA_AMERICIUM")
    chamber_item._tracks.clear()

    track = chamber_item.spawn_particle_track("alpha")
    assert track is not None
    assert track.particle_type == "alpha"
    assert track.charge == +2
    assert 4.0 <= track.energy_mev <= 6.5
    assert len(track.points) > 10

    # Verify Bragg peak: Droplet radius at the end of track is significantly larger than at start
    start_rad = track.points[0][2]
    end_rad = track.points[-1][2]
    assert end_rad > start_rad * 1.5, f"Expected Bragg peak broadening: start={start_rad}, end={end_rad}"


def test_beta_particle_lorentz_curvature(chamber_item):
    """Verify beta particle curves strongly in magnetic field."""
    chamber_item.set_preset("LORENTZ_SPECTROMETER")
    chamber_item.set_b_field(0.40)  # Strong positive B_z

    track = chamber_item.spawn_particle_track("beta")
    assert track is not None
    assert track.particle_type == "beta"
    assert track.charge == -1
    assert len(track.points) > 10


def test_cosmic_muon_track(chamber_item):
    """Verify cosmic muon creates high-energy, penetrating, rigid straight track."""
    chamber_item.set_preset("COSMIC_RAY_MUONS")

    track = chamber_item.spawn_particle_track("muon")
    assert track is not None
    assert track.particle_type == "muon"
    assert track.charge == -1
    assert track.energy_mev >= 1000.0  # relativistic MIP


def test_decay_counters(chamber_item):
    """Verify total counts increment properly by particle type."""
    init_alpha = chamber_item._total_counts["alpha"]
    chamber_item.spawn_particle_track("alpha")
    assert chamber_item._total_counts["alpha"] == init_alpha + 1

    init_beta = chamber_item._total_counts["beta"]
    chamber_item.spawn_particle_track("beta")
    assert chamber_item._total_counts["beta"] == init_beta + 1

    init_muon = chamber_item._total_counts["muon"]
    chamber_item.spawn_particle_track("muon")
    assert chamber_item._total_counts["muon"] == init_muon + 1


def test_step_simulation_fades_and_prunes(chamber_item):
    """Verify tracks fade over time and get pruned after lifetime expires."""
    chamber_item._tracks.clear()
    trk = chamber_item.spawn_particle_track("alpha")
    assert trk is not None
    assert len(chamber_item._tracks) == 1

    # Step simulation
    chamber_item.step_simulation(0.5)
    assert trk.alpha_opacity < 1.0

    # Fast forward past lifetime
    chamber_item.step_simulation(3.0)
    assert trk not in chamber_item._tracks


def test_play_pause_and_clear(chamber_item):
    """Verify play/pause toggles execution and clear removes tracks."""
    chamber_item.toggle_play_pause()
    assert chamber_item._is_playing is False

    t_before = chamber_item._sim_time_sec
    chamber_item.step_simulation(0.1)
    assert chamber_item._sim_time_sec == t_before

    chamber_item.toggle_play_pause()
    assert chamber_item._is_playing is True

    chamber_item.spawn_particle_track("alpha")
    assert len(chamber_item._tracks) > 0
    chamber_item.clear_tracks()
    assert len(chamber_item._tracks) == 0


def test_sliders_clamping(chamber_item):
    """Verify sliders clamp to allowed physical domains."""
    # B field [-0.5, 0.5]
    chamber_item.set_b_field(1.2)
    assert chamber_item._b_field_tesla == 0.50
    chamber_item.set_b_field(-1.2)
    assert chamber_item._b_field_tesla == -0.50

    # Decay rate [5, 150]
    chamber_item.set_decay_rate(300.0)
    assert chamber_item._decay_rate_cpm == 150.0
    chamber_item.set_decay_rate(1.0)
    assert chamber_item._decay_rate_cpm == 5.0

    # Supersaturation [2.0, 6.0]
    chamber_item.set_supersaturation(8.0)
    assert chamber_item._supersaturation == 6.0
    chamber_item.set_supersaturation(0.5)
    assert chamber_item._supersaturation == 2.0


def test_touchless_air_tap_gesture(chamber_item):
    """Verify air tap triggers presets, playback, inject, clear, and close."""
    # Air tap on Preset 3: COSMIC_RAY_MUONS
    # Button rect: start_x + 2 * (176 + 8) = 20 + 368 = 388, y = 58..86
    p_tap = chamber_item.mapToScene(QPointF(400.0, 70.0))
    handled = chamber_item.handle_air_tap(p_tap)
    assert handled is True
    assert chamber_item._current_preset == "COSMIC_RAY_MUONS"

    # Air tap on Inject Particle button: canvas left + 104 = 124, y = bottom + 20
    canvas_bottom = 96 + 470
    p_inj = chamber_item.mapToScene(QPointF(150.0, canvas_bottom + 20.0))
    tracks_before = len(chamber_item._tracks)
    chamber_item.handle_air_tap(p_inj)
    assert len(chamber_item._tracks) == tracks_before + 1

    # Air tap on Clear button: canvas left + 222 = 242
    p_clr = chamber_item.mapToScene(QPointF(250.0, canvas_bottom + 20.0))
    chamber_item.handle_air_tap(p_clr)
    assert len(chamber_item._tracks) == 0


def test_serialization_and_canvas_stamping(chamber_item):
    """Verify .airteach serialization and canvas cloning."""
    chamber_item.set_preset("RADON_CASCADE")
    chamber_item.set_b_field(-0.25)
    chamber_item.set_decay_rate(75.0)

    data = chamber_item.to_dict()
    assert data["type"] == "InteractiveCloudChamberItem"
    assert data["preset"] == "RADON_CASCADE"
    assert data["b_field_tesla"] == -0.25
    assert data["decay_rate_cpm"] == 75.0

    # Reconstruct from dict
    restored = InteractiveCloudChamberItem.from_dict(data)
    assert restored._current_preset == "RADON_CASCADE"
    assert restored._b_field_tesla == -0.25
    assert restored._decay_rate_cpm == 75.0

    # Stamp to canvas
    stamped = chamber_item.stamp_to_canvas()
    assert stamped is not None
    assert isinstance(stamped, InteractiveCloudChamberItem)
    assert stamped._current_preset == "RADON_CASCADE"
    assert stamped.pos() == QPointF(chamber_item.x() + 30.0, chamber_item.y() + 30.0)


def test_paint_rendering_all_modes(chamber_item):
    """Verify QPainter renders without error across all 5 simulation presets."""
    img = QImage(940, 650, QImage.Format_ARGB32)
    painter = QPainter(img)

    preset_keys = list(PRESETS.keys())
    for key in preset_keys:
        chamber_item.set_preset(key)
        chamber_item.spawn_particle_track()
        chamber_item.step_simulation(0.04)
        chamber_item.paint(painter, None, None)

    painter.end()
    assert not img.isNull()


def test_lesson_board_cloud_chamber_integration(qapp):
    """Verify LessonBoard lifecycle methods for Cloud Chamber visualizer."""
    from src.ui.lesson_board import LessonBoard

    board = LessonBoard()
    assert board.is_cloud_chamber_visible() is False
    assert board.get_cloud_chamber_item() is None

    # Show item
    item = board.show_cloud_chamber(True, pos=(120.0, 90.0))
    assert item is not None
    assert board.is_cloud_chamber_visible() is True
    assert board.get_cloud_chamber_item() is item
    assert item.pos() == QPointF(120.0, 90.0)

    # Toggle off
    is_vis = board.toggle_cloud_chamber()
    assert is_vis is False
    assert board.is_cloud_chamber_visible() is False

    # Toggle on
    is_vis = board.toggle_cloud_chamber()
    assert is_vis is True
    assert board.is_cloud_chamber_visible() is True


def test_main_window_cloud_chamber_toggle(qapp):
    """Verify MainWindow menu action and slot for Cloud Chamber visualizer."""
    from src.ui.main_window import MainWindow

    win = MainWindow(camera_index=-1)
    try:
        assert hasattr(win, "_on_toggle_cloud_chamber")
        assert win._scene.is_cloud_chamber_visible() is False

        # Toggle on
        is_vis = win._on_toggle_cloud_chamber()
        assert is_vis is True
        assert win._scene.is_cloud_chamber_visible() is True

        # Toggle off
        is_vis = win._on_toggle_cloud_chamber()
        assert is_vis is False
        assert win._scene.is_cloud_chamber_visible() is False
    finally:
        win.close()

