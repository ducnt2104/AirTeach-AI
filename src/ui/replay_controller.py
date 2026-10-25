"""

AirTeach AI - Stroke Replay Controller

=======================================

Enables time-based playback of handwritten strokes and lesson steps.

Uses real timestamps_ms recorded during teaching to accurately reconstruct

the teacher's whiteboard notes, math derivations, and diagrams.



Features:

  - Variable playback speeds (1x, 2x, 4x)

  - Interactive time scrubber slider (QSlider)

  - Play, Pause, Resume, Reset controls

  - Real-time timecode display (MM:SS / MM:SS)

  - Smooth animation driven by QTimer

"""

from __future__ import annotations



import logging

import math

from dataclasses import dataclass

from typing import Optional, Callable



import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, Slot

from PySide6.QtGui import QFont, QIcon, QPainterPath

from PySide6.QtWidgets import (

    QComboBox,

    QFrame,

    QHBoxLayout,

    QLabel,

    QPushButton,

    QSlider,

    QSizePolicy,

    QWidget,

)



from src.ui.lesson_board_canvas import InkStroke, LessonBoardCanvas



logger = logging.getLogger(__name__)





# ─── Replay State ─────────────────────────────────────────────────────────────



@dataclass

class ReplayStrokePoint:

    """Individual point within a stroke with absolute and relative timestamp."""

    x: float

    y: float

    t_ms: float

    stroke_id: str


@dataclass
class ReplayTimelineEvent:
    """A timed presentation event (STEM simulation update, slide page switch, gesture, or note)."""
    time_ms: float
    event_type: str
    payload: dict
    event_id: str = ""





# ─── Replay Controller Widget ─────────────────────────────────────────────────



class ReplayController(QWidget):

    """

    Playback toolbar and timing engine for replaying strokes on LessonBoardCanvas.



    Signals:

        playback_state_changed(bool): Emitted with True when playing, False when paused.

        time_updated(float, float): Emitted with (current_time_ms, total_time_ms).

    """



    playback_state_changed = Signal(bool)

    time_updated           = Signal(float, float)
    chapter_reached        = Signal(int, str)
    event_triggered        = Signal(object)



    TIMER_INTERVAL_MS = 25  # ~40 FPS animation



    def __init__(

        self,

        canvas: Optional[LessonBoardCanvas] = None,

        parent: Optional[QWidget] = None,

    ) -> None:

        super().__init__(parent)

        self._canvas = canvas



        # Playback timing

        self._is_playing: bool = False

        self._speed_multiplier: float = 1.0

        self._current_time_ms: float = 0.0

        self._start_time_ms: float = 0.0

        self._total_duration_ms: float = 0.0



        # Cached stroke data for playback

        self._source_strokes: list[InkStroke] = []

        # Chapters and Timeline Telemetry
        self._chapters: list[tuple[float, str]] = []
        self._current_chapter_idx: int = -1
        self._timeline_events: list[ReplayTimelineEvent] = []
        self._event_handlers: dict[str, list[Callable[[ReplayTimelineEvent], None]]] = {}
        self._dispatched_event_ids: set[str] = set()

        self._timer = QTimer(self)

        self._timer.setInterval(self.TIMER_INTERVAL_MS)

        self._timer.timeout.connect(self._on_timer_tick)



        self._build_ui()



    def _build_ui(self) -> None:

        layout = QHBoxLayout(self)

        layout.setContentsMargins(12, 6, 12, 6)

        layout.setSpacing(10)



        # Replay indicator badge

        self._badge = QLabel("⏱️ PHÁT LẠI")

        self._badge.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        self._badge.setStyleSheet(

            "background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 4px;"

        )

        layout.addWidget(self._badge)



        # Play / Pause button

        self._btn_play = QPushButton("▶ Phát")

        self._btn_play.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        self._btn_play.setStyleSheet(

            "background: #1a73e8; color: white; border: none; border-radius: 4px; padding: 6px 14px;"

        )

        self._btn_play.clicked.connect(self.toggle_play_pause)

        layout.addWidget(self._btn_play)



        # Reset button

        self._btn_reset = QPushButton("⏮ Về đầu")

        self._btn_reset.setStyleSheet(

            "background: #f1f3f4; color: #3c4043; border: 1px solid #dadce0; border-radius: 4px; padding: 6px 10px;"

        )

        self._btn_reset.clicked.connect(self.reset_playback)

        layout.addWidget(self._btn_reset)



        # Time slider

        self._slider = QSlider(Qt.Orientation.Horizontal)

        self._slider.setRange(0, 1000)

        self._slider.setValue(0)

        self._slider.setStyleSheet("""

            QSlider::groove:horizontal {

                height: 6px;

                background: #dadce0;

                border-radius: 3px;

            }

            QSlider::sub-page:horizontal {

                background: #1a73e8;

                border-radius: 3px;

            }

            QSlider::handle:horizontal {

                background: #1a73e8;

                width: 14px;

                margin: -4px 0;

                border-radius: 7px;

            }

        """)

        self._slider.sliderMoved.connect(self._on_slider_scrubbed)

        layout.addWidget(self._slider, stretch=1)



        # Time display

        self._time_lbl = QLabel("00:00 / 00:00")

        self._time_lbl.setFont(QFont("Consolas", 10))

        self._time_lbl.setStyleSheet("color: #5f6368;")

        layout.addWidget(self._time_lbl)



        # Speed selector

        self._combo_chapters = QComboBox()
        self._combo_chapters.addItem("📖 Toàn bộ bài giảng")
        self._combo_chapters.setStyleSheet("padding: 4px 8px; border: 1px solid #dadce0; border-radius: 4px; max-width: 170px;")
        self._combo_chapters.currentIndexChanged.connect(self._on_chapter_selected)
        layout.addWidget(self._combo_chapters)

        speed_lbl = QLabel("Tốc độ:")

        speed_lbl.setFont(QFont("Segoe UI", 9))

        layout.addWidget(speed_lbl)



        self._combo_speed = QComboBox()

        self._combo_speed.addItems(["1.0x", "2.0x", "4.0x", "0.5x", "1.5x"])

        self._combo_speed.setCurrentText("1.0x")

        self._combo_speed.setStyleSheet("padding: 4px 8px; border: 1px solid #dadce0; border-radius: 4px;")

        self._combo_speed.currentIndexChanged.connect(self._on_speed_changed)

        layout.addWidget(self._combo_speed)



    # ─── Public API ───────────────────────────────────────────────────────────



    # ─── Chapter Navigation & Timeline API ────────────────────────────────────

    def add_chapter(self, time_ms: float, title: str) -> None:
        """Register a chapter marker at a specific timestamp in milliseconds."""
        self._chapters.append((float(time_ms), str(title)))
        self._chapters.sort(key=lambda c: c[0])
        self._rebuild_chapter_combo()

    def get_chapters(self) -> list[tuple[float, str]]:
        """Return list of registered chapters as (time_ms, title) tuples."""
        return list(self._chapters)

    def clear_chapters(self) -> None:
        """Clear all registered chapter markers."""
        self._chapters.clear()
        self._current_chapter_idx = -1
        self._rebuild_chapter_combo()

    def _rebuild_chapter_combo(self) -> None:
        """Update dropdown items for chapters."""
        self._combo_chapters.blockSignals(True)
        self._combo_chapters.clear()
        self._combo_chapters.addItem("📖 Toàn bộ bài giảng")
        for idx, (t, title) in enumerate(self._chapters):
            sec = int(t / 1000.0)
            time_str = f"{sec // 60:02d}:{sec % 60:02d}"
            self._combo_chapters.addItem(f"[{time_str}] {title}")
        self._combo_chapters.blockSignals(False)

    def _on_chapter_selected(self, index: int) -> None:
        """Handle user selecting a chapter from dropdown."""
        if index <= 0 or index > len(self._chapters):
            return
        ch_time, ch_title = self._chapters[index - 1]
        self.seek_to(ch_time)

    def jump_to_chapter(self, index: int) -> bool:
        """Jump to specific chapter by index (0-based)."""
        if 0 <= index < len(self._chapters):
            ch_time, ch_title = self._chapters[index]
            self.seek_to(ch_time)
            return True
        return False

    def jump_to_next_chapter(self) -> bool:
        """Jump to next sequential chapter after current playhead position."""
        for idx, (t, title) in enumerate(self._chapters):
            if t > self._current_time_ms + 100.0:
                self.seek_to(t)
                return True
        return False

    def jump_to_prev_chapter(self) -> bool:
        """Jump to previous chapter before current playhead position."""
        for idx in range(len(self._chapters) - 1, -1, -1):
            t, title = self._chapters[idx]
            if t < self._current_time_ms - 100.0:
                self.seek_to(t)
                return True
        self.seek_to(0.0)
        return True

    def add_timeline_event(self, event: ReplayTimelineEvent) -> None:
        """Register a timed telemetry or presentation event."""
        self._timeline_events.append(event)
        self._timeline_events.sort(key=lambda ev: ev.time_ms)

    def get_timeline_events(self) -> list[ReplayTimelineEvent]:
        """Return registered timeline events."""
        return list(self._timeline_events)

    def clear_timeline_events(self) -> None:
        """Clear all registered timeline events."""
        self._timeline_events.clear()
        self._dispatched_event_ids.clear()

    def register_event_handler(self, event_type: str, handler: Callable[[ReplayTimelineEvent], None]) -> None:
        """Register a handler callback for a specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def seek_to(self, target_time_ms: float) -> None:
        """Seek playback head to target timestamp in milliseconds."""
        self._current_time_ms = max(0.0, min(self._total_duration_ms, float(target_time_ms)))
        self._slider.blockSignals(True)
        self._slider.setValue(int(self._current_time_ms))
        self._slider.blockSignals(False)
        self._apply_timestamp_to_canvas(self._current_time_ms)
        self._update_time_label()
        self.time_updated.emit(self._current_time_ms, self._total_duration_ms)

    def get_progress(self) -> float:
        """Return playback progress as a normalized float [0.0, 1.0]."""
        if self._total_duration_ms <= 0:
            return 0.0
        return max(0.0, min(1.0, self._current_time_ms / self._total_duration_ms))

    def set_progress(self, progress: float) -> None:
        """Seek playback head using normalized progress [0.0, 1.0]."""
        self.seek_to(max(0.0, min(1.0, float(progress))) * self._total_duration_ms)

    def export_timeline_data(self) -> dict:
        """Export complete replay timeline metadata for lesson packaging."""
        return {
            "total_duration_ms": self._total_duration_ms,
            "start_time_ms": self._start_time_ms,
            "speed_multiplier": self._speed_multiplier,
            "chapters": [{"time_ms": t, "title": title} for t, title in self._chapters],
            "events": [
                {
                    "time_ms": ev.time_ms,
                    "event_type": ev.event_type,
                    "payload": ev.payload,
                    "event_id": ev.event_id,
                }
                for ev in self._timeline_events
            ],
        }

    def load_timeline_data(self, data: dict) -> None:
        """Load timeline metadata, chapters, and presentation events."""
        self.clear_chapters()
        self.clear_timeline_events()
        if not isinstance(data, dict):
            return
        for ch in data.get("chapters", []):
            if "time_ms" in ch and "title" in ch:
                self.add_chapter(float(ch["time_ms"]), str(ch["title"]))
        for ev in data.get("events", []):
            if "time_ms" in ev and "event_type" in ev:
                self.add_timeline_event(
                    ReplayTimelineEvent(
                        time_ms=float(ev["time_ms"]),
                        event_type=str(ev["event_type"]),
                        payload=dict(ev.get("payload", {})),
                        event_id=str(ev.get("event_id", "")),
                    )
                )
        if "total_duration_ms" in data:
            self._total_duration_ms = max(self._total_duration_ms, float(data["total_duration_ms"]))
            self._slider.setRange(0, int(self._total_duration_ms))
            self._update_time_label()

    def set_speed(self, speed: float) -> None:

        """Set playback speed multiplier (e.g. 0.5, 1.0, 1.5, 2.0, 4.0)."""

        target = f"{speed:.1f}x"

        idx = self._combo_speed.findText(target)

        if idx >= 0:

            self._combo_speed.setCurrentIndex(idx)

        else:

            self._speed_multiplier = max(0.1, min(10.0, float(speed)))

            logger.info("Replay playback custom speed set to %sx", self._speed_multiplier)



    def set_canvas(self, canvas: LessonBoardCanvas) -> None:

        """Attach to a lesson board canvas."""

        self._canvas = canvas



    def load_strokes(self, strokes: list[InkStroke]) -> None:

        """

        Load list of InkStroke objects with timestamps to prepare for replay.

        Computes the total duration based on recorded timestamps_ms.

        """

        self.stop()

        self._source_strokes = list(strokes)



        all_ts: list[float] = []

        for s in self._source_strokes:

            all_ts.extend(s.timestamps_ms)

            if s._path_item is not None and hasattr(s._path_item, "path"):

                if not hasattr(s, "_original_path"):

                    setattr(s, "_original_path", s._path_item.path())



        if all_ts:

            self._start_time_ms = min(all_ts)

            self._total_duration_ms = max(100.0, max(all_ts) - self._start_time_ms)

        else:

            self._start_time_ms = 0.0

            self._total_duration_ms = 1000.0



        self._current_time_ms = 0.0

        self._slider.setRange(0, int(self._total_duration_ms))

        self._slider.setValue(0)

        self._update_time_label()



    def toggle_play_pause(self) -> None:

        """Toggle between playing and paused state."""

        if self._is_playing:

            self.pause()

        else:

            self.play()



    def play(self) -> None:

        """Start playback from current timestamp."""

        if not self._source_strokes and self._canvas:

            # Grab current canvas strokes if none loaded

            self.load_strokes(list(self._canvas._strokes.values()))



        if not self._source_strokes:

            return



        if self._current_time_ms >= self._total_duration_ms:

            self._current_time_ms = 0.0



        self._is_playing = True

        self._btn_play.setText("⏸ Tạm dừng")

        self._btn_play.setStyleSheet(

            "background: #ea4335; color: white; border: none; border-radius: 4px; padding: 6px 14px;"

        )

        self._timer.start()

        self.playback_state_changed.emit(True)



    def pause(self) -> None:

        """Pause playback at current timestamp."""

        self._is_playing = False

        self._timer.stop()

        self._btn_play.setText("▶ Tiếp tục")

        self._btn_play.setStyleSheet(

            "background: #1a73e8; color: white; border: none; border-radius: 4px; padding: 6px 14px;"

        )

        self.playback_state_changed.emit(False)



    def stop(self) -> None:

        """Stop playback completely."""

        self.pause()

        self.restore_full_strokes()

        self._btn_play.setText("▶ Phát")

        self._btn_play.setStyleSheet(

            "background: #1a73e8; color: white; border: none; border-radius: 4px; padding: 6px 14px;"

        )



    def restore_full_strokes(self) -> None:

        """Restore all source strokes to their complete geometry and visibility."""

        for stroke in self._source_strokes:

            orig = getattr(stroke, "_original_path", None)

            if orig is not None and stroke._path_item is not None and stroke._path_item.scene() is not None:

                stroke._path_item.setPath(orig)

            for attr in ("_path_item", "_shape_item"):

                item = getattr(stroke, attr, None)

                if item is not None and item.scene() is not None:

                    item.setVisible(True)



    def reset_playback(self) -> None:

        """Jump to beginning (0 ms) and pause."""

        self.stop()

        self._current_time_ms = 0.0

        self._dispatched_event_ids.clear()
        self._current_chapter_idx = -1
        if hasattr(self, "_combo_chapters"):
            self._combo_chapters.blockSignals(True)
            self._combo_chapters.setCurrentIndex(0)
            self._combo_chapters.blockSignals(False)

        self._slider.setValue(0)

        self._apply_timestamp_to_canvas(0.0)

        self._update_time_label()



    # ─── Internal Timer & Animation ───────────────────────────────────────────



    def _on_timer_tick(self) -> None:

        """Advance time step by step and update canvas display."""

        dt = self.TIMER_INTERVAL_MS * self._speed_multiplier

        self._current_time_ms += dt



        if self._current_time_ms >= self._total_duration_ms:

            self._current_time_ms = self._total_duration_ms

            self._apply_timestamp_to_canvas(self._current_time_ms)

            self._slider.setValue(int(self._current_time_ms))

            self._update_time_label()

            self.stop()

            return



        self._slider.setValue(int(self._current_time_ms))

        self._apply_timestamp_to_canvas(self._current_time_ms)

        self._update_time_label()

        self.time_updated.emit(self._current_time_ms, self._total_duration_ms)



    def _on_slider_scrubbed(self, val: int) -> None:

        """Handle user manually dragging the time slider."""

        self._current_time_ms = float(val)

        self._apply_timestamp_to_canvas(self._current_time_ms)

        self._update_time_label()

        self.time_updated.emit(self._current_time_ms, self._total_duration_ms)



    def _on_speed_changed(self, index: int) -> None:

        """Change speed multiplier (0.5x, 1.0x, 1.5x, 2.0x, 4.0x)."""

        text = self._combo_speed.currentText().replace("x", "").strip()

        try:

            self._speed_multiplier = float(text)

        except ValueError:

            self._speed_multiplier = 1.0

        logger.info("Replay playback speed changed to %sx", self._speed_multiplier)



    def _apply_timestamp_to_canvas(self, target_ms: float) -> None:

        """

        Synchronize canvas strokes with the replay time.

        Strokes are progressively animated as their timestamps elapse.

        Supports simultaneous multi-hand strokes drawing in parallel.

        """

        if self._canvas is None:

            return



        # 1. Update Chapter tracking
        active_ch_idx = -1
        for idx, (ch_time, ch_title) in enumerate(self._chapters):
            if target_ms >= ch_time:
                active_ch_idx = idx
            else:
                break
        if active_ch_idx != self._current_chapter_idx:
            self._current_chapter_idx = active_ch_idx
            if 0 <= active_ch_idx < len(self._chapters):
                title = self._chapters[active_ch_idx][1]
                self.chapter_reached.emit(active_ch_idx, title)
                if hasattr(self, "_combo_chapters") and self._combo_chapters.count() > active_ch_idx + 1:
                    self._combo_chapters.blockSignals(True)
                    self._combo_chapters.setCurrentIndex(active_ch_idx + 1)
                    self._combo_chapters.blockSignals(False)

        # 2. Dispatch Timeline Presentation Events
        for ev in self._timeline_events:
            ev_key = ev.event_id or f"{ev.event_type}_{ev.time_ms}"
            if ev.time_ms <= target_ms:
                if ev_key not in self._dispatched_event_ids:
                    self._dispatched_event_ids.add(ev_key)
                    self.event_triggered.emit(ev)
                    handlers = self._event_handlers.get(ev.event_type, [])
                    for h in handlers:
                        try:
                            h(ev)
                        except Exception as exc:
                            logger.warning("Error in replay event handler for %s: %s", ev.event_type, exc)
            else:
                if ev_key in self._dispatched_event_ids:
                    self._dispatched_event_ids.remove(ev_key)

        abs_target = self._start_time_ms + target_ms



        for stroke in self._source_strokes:

            if not stroke.timestamps_ms:

                continue

            stroke_start = stroke.timestamps_ms[0]

            stroke_end = stroke.timestamps_ms[-1]



            # Before stroke starts

            if abs_target < stroke_start:

                for attr in ("_path_item", "_shape_item"):

                    item = getattr(stroke, attr, None)

                    if item is not None and item.scene() is not None:

                        item.setVisible(False)

                continue



            # Stroke is completed at or before abs_target

            if abs_target >= stroke_end:

                if stroke._path_item is not None and stroke._path_item.scene() is not None:

                    orig = getattr(stroke, "_original_path", None)

                    if orig is not None:

                        stroke._path_item.setPath(orig)

                    stroke._path_item.setVisible(True)

                if stroke._shape_item is not None and stroke._shape_item.scene() is not None:

                    stroke._shape_item.setVisible(True)

                continue



            # In-progress stroke: stroke_start <= abs_target < stroke_end

            if stroke._shape_item is not None and stroke._shape_item.scene() is not None:

                # Snapped shape is revealed when fully drawn

                stroke._shape_item.setVisible(False)



            if stroke._path_item is not None and stroke._path_item.scene() is not None:

                stroke._path_item.setVisible(True)

                dur = max(1.0, stroke_end - stroke_start)

                progress = min(1.0, max(0.0, (abs_target - stroke_start) / dur))

            

                if stroke.segments:

                    n_segs = len(stroke.segments)

                    val = progress * n_segs

                    k = int(math.floor(val))

                    u = val - k

            

                    partial_path = QPainterPath()

                    p_start = stroke.segments[0].p0

                    partial_path.moveTo(float(p_start[0]), float(p_start[1]))

            

                    # 1. Full prior segments

                    for seg_idx in range(min(k, n_segs)):

                        seg = stroke.segments[seg_idx]

                        partial_path.cubicTo(

                            float(seg.p1[0]), float(seg.p1[1]),

                            float(seg.p2[0]), float(seg.p2[1]),

                            float(seg.p3[0]), float(seg.p3[1]),

                        )

            

                    # 2. De Casteljau subdivision of active segment

                    if k < n_segs and u > 1e-4:

                        curr_seg = stroke.segments[k]

                        cp0 = curr_seg.p0

                        cp1 = curr_seg.p1

                        cp2 = curr_seg.p2

                        cp3 = curr_seg.p3

            

                        p01 = (1.0 - u) * cp0 + u * cp1

                        p12 = (1.0 - u) * cp1 + u * cp2

                        p23 = (1.0 - u) * cp2 + u * cp3

            

                        p012 = (1.0 - u) * p01 + u * p12

                        p123 = (1.0 - u) * p12 + u * p23

            

                        p0123 = (1.0 - u) * p012 + u * p123

            

                        partial_path.cubicTo(

                            float(p01[0]), float(p01[1]),

                            float(p012[0]), float(p012[1]),

                            float(p0123[0]), float(p0123[1]),

                        )

            

                    stroke._path_item.setPath(partial_path)

                elif hasattr(stroke, "points") and stroke.points:

                    pts = [p for p in stroke.points if getattr(p, "t_ms", 0.0) <= abs_target]

                    if len(pts) >= 2:

                        partial_path = QPainterPath()

                        partial_path.moveTo(float(pts[0].x), float(pts[0].y))

                        for pt in pts[1:]:

                            partial_path.lineTo(float(pt.x), float(pt.y))

                        stroke._path_item.setPath(partial_path)




    def _update_time_label(self) -> None:

        """Format current/total time into MM:SS."""

        cur_sec = int(self._current_time_ms / 1000.0)

        tot_sec = int(self._total_duration_ms / 1000.0)

        cur_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d}"

        tot_str = f"{tot_sec // 60:02d}:{tot_sec % 60:02d}"

        self._time_lbl.setText(f"{cur_str} / {tot_str}")

