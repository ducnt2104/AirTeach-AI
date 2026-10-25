"""
AirTeach AI - Classroom Metronome & Rhythm Practice Assistant
=============================================================
Provides an interactive on-board visual & acoustic rhythm pacer for LessonBoard:
  - Configurable tempo (40 - 240 BPM) with classic Italian speed markings.
  - Multi-meter time signatures (2/4, 3/4, 4/4, 6/8) for poetry, language recitation, and music.
  - Visual LED beat pulses with accented downbeat glow.
  - Authentic in-memory synthetic acoustic woodblock ticks and downbeat chime.
  - Movable on canvas with mouse and touchless air tap interaction.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
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
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.utils.audio_feedback import AudioFeedbackManager

logger = logging.getLogger(__name__)

TIME_SIGNATURES: list[tuple[int, str]] = [
    (2, "2/4"),
    (3, "3/4"),
    (4, "4/4"),
    (6, "6/8"),
]

MIN_BPM: int = 40
MAX_BPM: int = 240
DEFAULT_BPM: int = 120


class ClassroomMetronomeItem(QGraphicsObject):
    """
    Classroom Metronome overlay widget for LessonBoard.
    Displays tempo in BPM, rhythmic beat LEDs, time signature, and controls.
    """

    beat_ticked = Signal(int, int)  # (current_beat, total_beats)
    tempo_changed = Signal(int)     # (bpm)
    running_changed = Signal(bool)

    def __init__(
        self,
        bpm: int = DEFAULT_BPM,
        beats_per_bar: int = 4,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.width: float = 230.0
        self.height: float = 125.0

        self._bpm: int = max(MIN_BPM, min(MAX_BPM, int(bpm)))
        self._sig_idx: int = 2  # 4/4 default
        self._beats_per_bar: int = beats_per_bar
        self._current_beat: int = 0
        self._is_running: bool = False

        # Internal precision interval timer (parented for deterministic lifecycle)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # Flags: movable and selectable on scene
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

    # ── State Control & Properties ─────────────────────────────────────────────

    def bpm(self) -> int:
        return self._bpm

    def set_bpm(self, bpm: int) -> None:
        clamped = max(MIN_BPM, min(MAX_BPM, int(bpm)))
        if clamped != self._bpm:
            self._bpm = clamped
            if self._is_running:
                interval_ms = int(round(60000.0 / self._bpm))
                self._timer.setInterval(interval_ms)
            self.prepareGeometryChange()
            self.update()
            self.tempo_changed.emit(self._bpm)

    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        if not self._is_running:
            self._is_running = True
            self._current_beat = 0
            interval_ms = int(round(60000.0 / self._bpm))
            self._timer.start(interval_ms)
            self.running_changed.emit(True)
            self.update()

    def stop(self) -> None:
        if self._is_running:
            self._is_running = False
            self._timer.stop()
            self._current_beat = 0
            self.running_changed.emit(False)
            self.update()

    def reset(self) -> None:
        """Reset beat counter and stop playback."""
        self.stop()
        self._current_beat = 0
        self.update()

    def cleanup(self) -> None:
        """Safely stop tick timer and disconnect callbacks to prevent resource leaks."""
        self.stop()
        try:
            self._timer.timeout.disconnect()
        except Exception:
            pass

    def toggle(self) -> bool:
        if self._is_running:
            self.stop()
        else:
            self.start()
        return self._is_running

    def cycle_time_signature(self) -> str:
        """Cycle through 2/4 -> 3/4 -> 4/4 -> 6/8."""
        self._sig_idx = (self._sig_idx + 1) % len(TIME_SIGNATURES)
        self._beats_per_bar, name = TIME_SIGNATURES[self._sig_idx]
        self._current_beat = 0
        self.prepareGeometryChange()
        self.update()
        return name

    def time_signature_name(self) -> str:
        return TIME_SIGNATURES[self._sig_idx][1]

    def tempo_marking(self) -> str:
        """Return traditional musical tempo marking."""
        b = self._bpm
        if b < 60:
            return "Largo (Trang nghiêm)"
        elif b < 76:
            return "Adagio (Chậm rãi)"
        elif b < 109:
            return "Andante (Thong thả)"
        elif b < 133:
            return "Moderato (Vừa phải)"
        elif b < 169:
            return "Allegro (Nhanh, vui)"
        else:
            return "Presto (Rất nhanh)"

    # ── Timer Tick Callback ────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        self._current_beat = (self._current_beat % self._beats_per_bar) + 1

        # Acoustic feedback
        try:
            mgr = AudioFeedbackManager.instance()
            if self._current_beat == 1:
                mgr.play_metronome_bell()
            else:
                mgr.play_metronome_tick()
        except Exception:
            pass

        self.beat_ticked.emit(self._current_beat, self._beats_per_bar)
        self.update()

    # ── Bounding Rect & Painting ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(-6.0, -6.0, self.width + 12.0, self.height + 12.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Frosted acrylic background card
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0.0, 0.0, self.width, self.height, 14.0, 14.0)

        painter.fillPath(bg_path, QBrush(QColor(22, 24, 34, 235)))
        border_color = QColor(16, 185, 129, 180) if self._is_running else QColor(255, 255, 255, 35)
        painter.strokePath(bg_path, QPen(border_color, 1.5))

        # 2. Header: Title & Close Button
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#93c5fd")))
        painter.drawText(14, 20, "🎵 Máy Đếm Nhịp (Metronome)")

        # Close button [✕]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(self.width - 24), 20, "✕")

        # 3. Large Tempo Readout & Tempo Marking
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        bpm_txt = f"{self._bpm} BPM"
        painter.drawText(14, 50, bpm_txt)

        # Subtitle: Italian marking
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#a5b4fc")))
        painter.drawText(14, 65, self.tempo_marking())

        # 4. Beat LEDs
        n = self._beats_per_bar
        led_radius = 6.0
        led_spacing = 16.0
        total_w = (n - 1) * led_spacing
        start_x = self.width - total_w - 20.0
        led_y = 48.0

        for i in range(1, n + 1):
            lx = start_x + (i - 1) * led_spacing
            is_active = (self._is_running and self._current_beat == i)

            if is_active:
                if i == 1:
                    # Accent beat: bright rose/coral
                    led_color = QColor("#f43f5e")
                    halo_color = QColor(244, 63, 94, 70)
                else:
                    # Regular beat: bright emerald
                    led_color = QColor("#10b981")
                    halo_color = QColor(16, 185, 129, 70)

                # Outer halo
                painter.setBrush(QBrush(halo_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(lx, led_y), led_radius + 4.0, led_radius + 4.0)

                painter.setBrush(QBrush(led_color))
                painter.drawEllipse(QPointF(lx, led_y), led_radius, led_radius)
            else:
                painter.setBrush(QBrush(QColor("#334155")))
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.drawEllipse(QPointF(lx, led_y), led_radius - 1.0, led_radius - 1.0)

        # 5. Interactive Control Buttons row
        btn_y = 82.0
        btn_h = 28.0

        # [▶/⏸️] Toggle Run Button
        run_btn_w = 54.0
        run_rect = QRectF(14.0, btn_y, run_btn_w, btn_h)
        run_bg = QColor(16, 185, 129, 200) if self._is_running else QColor(79, 70, 229, 200)
        painter.setBrush(QBrush(run_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(run_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(run_rect, Qt.AlignmentFlag.AlignCenter, "⏸️" if self._is_running else "▶ Chạy")

        # [−] Tempo Down
        btn_minus_rect = QRectF(74.0, btn_y, 28.0, btn_h)
        painter.setBrush(QBrush(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(btn_minus_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(btn_minus_rect, Qt.AlignmentFlag.AlignCenter, "−")

        # [+] Tempo Up
        btn_plus_rect = QRectF(108.0, btn_y, 28.0, btn_h)
        painter.setBrush(QBrush(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(btn_plus_rect, 6.0, 6.0)
        painter.drawText(btn_plus_rect, Qt.AlignmentFlag.AlignCenter, "+")

        # [2/4 / 3/4 / 4/4 / 6/8] Time Signature Cycle Button
        sig_rect = QRectF(142.0, btn_y, 74.0, btn_h)
        painter.setBrush(QBrush(QColor(30, 41, 59, 220)))
        painter.setPen(QPen(QColor(148, 163, 184, 80), 1.0))
        painter.drawRoundedRect(sig_rect, 6.0, 6.0)

        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(sig_rect, Qt.AlignmentFlag.AlignCenter, self.time_signature_name())

    # ── Mouse & Air Gesture Clicks ─────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        btn_y = 82.0
        btn_h = 28.0

        # 1. Close Button [✕]
        if pos.x() >= self.width - 32.0 and pos.y() <= 28.0:
            self.stop()
            self.setVisible(False)
            event.accept()
            return

        # 2. Toggle Start/Stop [▶/⏸️]
        if 14.0 <= pos.x() <= 68.0 and btn_y <= pos.y() <= btn_y + btn_h:
            self.toggle()
            event.accept()
            return

        # 3. Tempo Minus [−]
        if 74.0 <= pos.x() <= 102.0 and btn_y <= pos.y() <= btn_y + btn_h:
            self.set_bpm(self._bpm - 5)
            event.accept()
            return

        # 4. Tempo Plus [+]
        if 108.0 <= pos.x() <= 136.0 and btn_y <= pos.y() <= btn_y + btn_h:
            self.set_bpm(self._bpm + 5)
            event.accept()
            return

        # 5. Cycle Time Signature [4/4]
        if 142.0 <= pos.x() <= 216.0 and btn_y <= pos.y() <= btn_y + btn_h:
            self.cycle_time_signature()
            event.accept()
            return

        super().mousePressEvent(event)

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Allows touchless air tapping on metronome controls.
        Returns True if a button was activated.
        """
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        px, py = local_pos.x(), local_pos.y()
        btn_y = 82.0
        btn_h = 28.0

        # 1. Close Button
        if px >= self.width - 32.0 and py <= 28.0:
            self.stop()
            self.setVisible(False)
            return True

        # 2. Toggle Start/Stop
        if 10.0 <= px <= 72.0 and btn_y - 4.0 <= py <= btn_y + btn_h + 4.0:
            self.toggle()
            return True

        # 3. Tempo Minus
        if 70.0 <= px <= 104.0 and btn_y - 4.0 <= py <= btn_y + btn_h + 4.0:
            self.set_bpm(self._bpm - 5)
            return True

        # 4. Tempo Plus
        if 106.0 <= px <= 140.0 and btn_y - 4.0 <= py <= btn_y + btn_h + 4.0:
            self.set_bpm(self._bpm + 5)
            return True

        # 5. Cycle Time Signature
        if 140.0 <= px <= 220.0 and btn_y - 4.0 <= py <= btn_y + btn_h + 4.0:
            self.cycle_time_signature()
            return True

        return False
