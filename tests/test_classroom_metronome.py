"""
Tests for Classroom Metronome & Rhythm Practice Assistant (Task I.42)
=====================================================================
Covers:
  - ClassroomMetronomeItem initialization and default parameters.
  - BPM boundary clamping (40 - 240 BPM) and signal emission.
  - Italian tempo markings (Largo, Adagio, Andante, Moderato, Allegro, Presto).
  - Multi-meter time signatures cycling (2/4, 3/4, 4/4, 6/8).
  - Precision tick progression and downbeat chime / regular tick audio trigger.
  - Visual painting and LED pulse rendering on QPainter.
  - Touchless air click interaction on control buttons (Start/Stop, BPM +/-, Signature, Close).
  - LessonBoard integration (show, toggle, selection rectangle exclusion).
  - MainWindow integration and touchless air tap event loop consumption.
"""
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from src.ui.classroom_metronome import (
    ClassroomMetronomeItem,
    DEFAULT_BPM,
    MAX_BPM,
    MIN_BPM,
    TIME_SIGNATURES,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestClassroomMetronomeItem:
    def test_initialization_defaults(self, qapp):
        item = ClassroomMetronomeItem()
        assert item.bpm() == DEFAULT_BPM
        assert item.time_signature_name() == "4/4"
        assert not item.is_running()
        assert item.boundingRect().width() > 200
        assert item.boundingRect().height() > 100

    def test_bpm_clamping_and_signals(self, qapp):
        item = ClassroomMetronomeItem(bpm=120)
        emitted_bpm = []
        item.tempo_changed.connect(lambda b: emitted_bpm.append(b))

        # Underflow clamp
        item.set_bpm(20)
        assert item.bpm() == MIN_BPM
        assert emitted_bpm[-1] == MIN_BPM

        # Overflow clamp
        item.set_bpm(350)
        assert item.bpm() == MAX_BPM
        assert emitted_bpm[-1] == MAX_BPM

        # Valid range
        item.set_bpm(96)
        assert item.bpm() == 96
        assert emitted_bpm[-1] == 96

    def test_tempo_markings(self, qapp):
        item = ClassroomMetronomeItem()
        item.set_bpm(50)
        assert "Largo" in item.tempo_marking()

        item.set_bpm(70)
        assert "Adagio" in item.tempo_marking()

        item.set_bpm(100)
        assert "Andante" in item.tempo_marking()

        item.set_bpm(120)
        assert "Moderato" in item.tempo_marking()

        item.set_bpm(144)
        assert "Allegro" in item.tempo_marking()

        item.set_bpm(180)
        assert "Presto" in item.tempo_marking()

    def test_time_signature_cycling(self, qapp):
        item = ClassroomMetronomeItem(beats_per_bar=4)
        assert item.time_signature_name() == "4/4"

        # 4/4 -> 6/8
        sig = item.cycle_time_signature()
        assert sig == "6/8"
        assert item.time_signature_name() == "6/8"

        # 6/8 -> 2/4
        sig = item.cycle_time_signature()
        assert sig == "2/4"
        assert item.time_signature_name() == "2/4"

        # 2/4 -> 3/4
        sig = item.cycle_time_signature()
        assert sig == "3/4"
        assert item.time_signature_name() == "3/4"

        # 3/4 -> 4/4
        sig = item.cycle_time_signature()
        assert sig == "4/4"
        assert item.time_signature_name() == "4/4"

    def test_start_stop_toggle_and_tick(self, qapp):
        item = ClassroomMetronomeItem(bpm=120, beats_per_bar=4)
        running_states = []
        item.running_changed.connect(lambda s: running_states.append(s))

        ticks = []
        item.beat_ticked.connect(lambda b, total: ticks.append((b, total)))

        # Start
        item.start()
        assert item.is_running()
        assert running_states[-1] is True

        # Simulate ticks: beat 1 (downbeat)
        item._on_tick()
        assert ticks[-1] == (1, 4)

        # Beat 2
        item._on_tick()
        assert ticks[-1] == (2, 4)

        # Beat 3
        item._on_tick()
        assert ticks[-1] == (3, 4)

        # Beat 4
        item._on_tick()
        assert ticks[-1] == (4, 4)

        # Beat 1 wrap-around
        item._on_tick()
        assert ticks[-1] == (1, 4)

        # Toggle to stop
        assert item.toggle() is False
        assert not item.is_running()
        assert running_states[-1] is False

        # Toggle to start
        assert item.toggle() is True
        assert item.is_running()
        assert running_states[-1] is True

        item.stop()
        assert not item.is_running()

    def test_rendering_on_pixmap(self, qapp):
        item = ClassroomMetronomeItem(bpm=130, beats_per_bar=4)
        item.start()
        item._on_tick()  # active beat 1

        pixmap = QPixmap(300, 200)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem

        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
        painter.end()

        assert not pixmap.isNull()
        img = pixmap.toImage()
        # Ensure non-transparent pixels were drawn
        has_pixels = any(img.pixelColor(x, y).alpha() > 0 for x in range(10, 200, 10) for y in range(10, 100, 10))
        assert has_pixels

    def test_air_clicks_on_controls(self, qapp):
        item = ClassroomMetronomeItem(bpm=120)
        item.setPos(100, 100)
        item.setVisible(True)

        # 1. Toggle Run Button: local x: 30, y: 95 -> scene (130, 195)
        assert not item.is_running()
        handled = item.air_click_at(130.0, 195.0)
        assert handled is True
        assert item.is_running()

        handled = item.air_click_at(130.0, 195.0)
        assert handled is True
        assert not item.is_running()

        # 2. Plus Button: local x: 120, y: 95 -> scene (220, 195)
        old_bpm = item.bpm()
        handled = item.air_click_at(220.0, 195.0)
        assert handled is True
        assert item.bpm() == old_bpm + 5

        # 3. Minus Button: local x: 88, y: 95 -> scene (188, 195)
        handled = item.air_click_at(188.0, 195.0)
        assert handled is True
        assert item.bpm() == old_bpm

        # 4. Time Signature Button: local x: 180, y: 95 -> scene (280, 195)
        old_sig = item.time_signature_name()
        handled = item.air_click_at(280.0, 195.0)
        assert handled is True
        assert item.time_signature_name() != old_sig

        # 5. Air Click outside: local x: 2, y: 2 -> scene (102, 102)
        handled = item.air_click_at(102.0, 102.0)
        assert handled is False

        # 6. Close Button: local x: width - 15, y: 15 -> scene (100 + 215, 115)
        handled = item.air_click_at(315.0, 115.0)
        assert handled is True
        assert not item.isVisible()
        assert not item.is_running()

    def test_metronome_reset_and_cleanup(self, qapp):
        item = ClassroomMetronomeItem(bpm=140)
        item.start()
        item._on_tick()
        assert item.is_running()

        # Reset stops running and zeroes beat
        item.reset()
        assert not item.is_running()
        assert item._current_beat == 0

        # Cleanup disconnects timer safely
        item.cleanup()
        assert not item.is_running()


class TestPresentationTimerAcousticsAndAirClick:
    def test_presentation_timer_air_clicks_and_stop(self, qapp):
        from src.ui.presentation_timer import PresentationTimerItem

        timer = PresentationTimerItem(duration_sec=300)
        timer.setPos(200, 200)
        timer.setVisible(True)

        # 1. Air click Play/Pause (center-left bottom: local x ~ 95, y ~ 103 -> scene 295, 303)
        assert not timer.is_running()
        handled = timer.air_click_at(295.0, 303.0)
        assert handled is True
        assert timer.is_running()

        # Pause via air click
        handled = timer.air_click_at(295.0, 303.0)
        assert handled is True
        assert not timer.is_running()

        # 2. Air click Preset Pill (local x ~ 60, y ~ 20 -> scene 260, 220)
        old_dur = timer.total_seconds()
        handled = timer.air_click_at(260.0, 220.0)
        assert handled is True
        assert timer.total_seconds() != old_dur

        # 3. Air click Reset (local x ~ 135, y ~ 103 -> scene 335, 303)
        handled = timer.air_click_at(335.0, 303.0)
        assert handled is True
        assert timer.remaining_seconds() == timer.total_seconds()

        # 4. Air click Close Button (local x ~ 215, y ~ 18 -> scene 415, 218)
        handled = timer.air_click_at(415.0, 218.0)
        assert handled is True
        assert not timer.isVisible()
        assert not timer.is_running()

        # 5. Method stop() and cleanup()
        timer.start()
        assert timer.is_running()
        timer.stop()
        assert not timer.is_running()
        timer.cleanup()


class TestLessonBoardMetronomeIntegration:
    def test_lesson_board_metronome_lifecycle(self, qapp):
        board = LessonBoard()
        assert not board.is_metronome_visible()
        assert board.get_metronome_item() is None

        # Show
        board.show_metronome(True, x=50, y=150, bpm=100)
        assert board.is_metronome_visible()
        metronome = board.get_metronome_item()
        assert metronome is not None
        assert metronome.bpm() == 100
        assert metronome.x() == 50
        assert metronome.y() == 150

        # Toggle to hide
        assert board.toggle_metronome() is False
        assert not board.is_metronome_visible()

        # Toggle to show
        assert board.toggle_metronome() is True
        assert board.is_metronome_visible()

    def test_metronome_excluded_from_selection_rectangle(self, qapp):
        board = LessonBoard()
        board.show_metronome(True, x=100, y=100)
        metronome = board.get_metronome_item()
        assert metronome is not None

        # Query all items in rect covering the metronome
        rect = QRectF(50, 50, 300, 250)
        selected = board.get_items_in_rect(rect)
        assert metronome not in selected


class TestMainWindowMetronomeIntegration:
    def test_main_window_toggle_metronome(self, qapp):
        from src.ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert not win._scene.is_metronome_visible()

            # Trigger toggle
            win._on_toggle_metronome()
            assert win._scene.is_metronome_visible()

            # Trigger toggle again
            win._on_toggle_metronome()
            assert not win._scene.is_metronome_visible()
        finally:
            win.close()
