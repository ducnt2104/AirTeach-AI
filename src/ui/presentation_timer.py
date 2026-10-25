"""
AirTeach AI - Interactive Presentation Countdown Timer & Stopwatch
==================================================================
Provides an interactive on-board classroom timer overlay for LessonBoard:
  - Countdown timer with presets (1m, 3m, 5m, 10m, 15m) and Stopwatch mode.
  - Large high-contrast digital readout (MM:SS) and color-coded progress bar.
  - Interactive Start/Pause, Reset, and Preset cycle buttons.
  - Draggable anywhere on the canvas via touchless air pinch or mouse.
"""
from __future__ import annotations

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
    QStyleOptionGraphicsItem,
    QWidget,
)


class PresentationTimerItem(QGraphicsObject):
    """
    Classroom Countdown Timer & Stopwatch overlay widget for LessonBoard.
    Displays digital time, dynamic progress bar, and interactive control buttons.
    """

    timer_finished = Signal()
    time_updated = Signal(float)

    PRESETS = [
        (60, "1 phút"),
        (180, "3 phút"),
        (300, "5 phút"),
        (600, "10 phút"),
        (900, "15 phút"),
        (0, "Bấm giờ"),
    ]

    def __init__(
        self,
        duration_sec: int = 300,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.width: float = 230.0
        self.height: float = 120.0

        self._preset_idx: int = 2  # 5 minutes default
        self._total_seconds: float = float(duration_sec)
        self._remaining_seconds: float = float(duration_sec)
        self._is_running: bool = False
        self._is_stopwatch: bool = (duration_sec == 0)

        # Styling
        self._bg_color = QColor(28, 30, 33, 225)  # Dark acrylic
        self._border_color = QColor(66, 70, 75, 220)
        self._accent_color = QColor(26, 115, 232, 230)  # Primary Blue
        self._warn_color = QColor(251, 188, 4, 240)     # Amber (<30s)
        self._alert_color = QColor(234, 67, 53, 240)    # Red (<10s or 0)

        # Flags: movable and selectable
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(26.0)

        # Internal tick timer (100ms for smooth progress bar)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

    def boundingRect(self) -> QRectF:
        margin = 10.0
        return QRectF(-margin, -margin, self.width + 2 * margin, self.height + 2 * margin)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Acrylic Card Background
        card_rect = QRectF(0.0, 0.0, self.width, self.height)
        painter.setBrush(QBrush(self._bg_color))
        border_pen = QPen(self._border_color, 1.8)
        if self.isSelected():
            border_pen = QPen(QColor("#38bdf8"), 2.2, Qt.PenStyle.DashLine)
        elif self._is_running:
            border_pen = QPen(QColor("#1a73e8"), 2.0)
        painter.setPen(border_pen)
        painter.drawRoundedRect(card_rect, 12.0, 12.0)

        # 2. Top Header: Preset Pill Button & Close Button
        # Preset Pill (center-left)
        preset_name = self.PRESETS[self._preset_idx][1]
        pill_rect = QRectF(14.0, 10.0, 110.0, 22.0)
        painter.setBrush(QBrush(QColor(48, 52, 58, 200)))
        painter.setPen(QPen(QColor(80, 85, 92), 1.0))
        painter.drawRoundedRect(pill_rect, 11.0, 11.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#d1d5db")))
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, f"⏱️ {preset_name}")

        # Close "✕" Button (top-right)
        close_rect = QRectF(self.width - 28.0, 10.0, 18.0, 18.0)
        painter.setBrush(QBrush(QColor(234, 67, 53, 50)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(close_rect)
        painter.setPen(QPen(QColor("#fca5a5"), 1.2))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        # 3. Large Digital Time Readout (MM:SS)
        sec_val = max(0, int(round(self._remaining_seconds)))
        minutes = sec_val // 60
        seconds = sec_val % 60
        time_text = f"{minutes:02d}:{seconds:02d}"

        # Color evaluation
        if not self._is_stopwatch and self._remaining_seconds <= 10.0 and self._total_seconds > 10.0:
            time_color = self._alert_color
        elif not self._is_stopwatch and self._remaining_seconds <= 30.0 and self._total_seconds > 30.0:
            time_color = self._warn_color
        else:
            time_color = QColor("#ffffff")

        painter.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        painter.setPen(QPen(time_color))
        time_rect = QRectF(0.0, 32.0, self.width, 42.0)
        painter.drawText(time_rect, Qt.AlignmentFlag.AlignCenter, time_text)

        # 4. Progress Bar along Bottom (Countdown: 100% -> 0%, Stopwatch: pulses)
        bar_x = 14.0
        bar_y = 80.0
        bar_w = self.width - 28.0
        bar_h = 6.0

        # Background track
        painter.setBrush(QBrush(QColor(50, 54, 60, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3.0, 3.0)

        # Active filled bar
        if self._is_stopwatch:
            progress = min(1.0, (self._remaining_seconds % 60.0) / 60.0)
            fill_color = self._accent_color
        else:
            progress = max(0.0, min(1.0, self._remaining_seconds / max(1.0, self._total_seconds)))
            fill_color = time_color

        if progress > 0.0:
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * progress, bar_h), 3.0, 3.0)

        # 5. Interactive Bottom Controls: [ ▶ / ⏸ ] and [ ↺ ]
        ctrl_y = 92.0
        # Play / Pause
        btn_play_rect = QRectF(self.width / 2.0 - 35.0, ctrl_y, 30.0, 22.0)
        painter.setBrush(QBrush(QColor(48, 52, 58, 220)))
        painter.setPen(QPen(QColor(80, 85, 92), 1.0))
        painter.drawRoundedRect(btn_play_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8") if self._is_running else QColor("#4ade80")))
        painter.drawText(btn_play_rect, Qt.AlignmentFlag.AlignCenter, "⏸" if self._is_running else "▶")

        # Reset
        btn_reset_rect = QRectF(self.width / 2.0 + 5.0, ctrl_y, 30.0, 22.0)
        painter.setBrush(QBrush(QColor(48, 52, 58, 220)))
        painter.setPen(QPen(QColor(80, 85, 92), 1.0))
        painter.drawRoundedRect(btn_reset_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#9ca3af")))
        painter.drawText(btn_reset_rect, Qt.AlignmentFlag.AlignCenter, "↺")

    def mousePressEvent(self, event) -> None:
        pos = event.pos()

        # Check Close Button
        close_rect = QRectF(self.width - 28.0, 10.0, 18.0, 18.0)
        if close_rect.contains(pos):
            self.stop()
            self.setVisible(False)
            event.accept()
            return

        # Check Preset Pill Button (cycle presets)
        pill_rect = QRectF(14.0, 10.0, 110.0, 22.0)
        if pill_rect.contains(pos):
            self._cycle_preset()
            event.accept()
            return

        # Check Play / Pause Button
        btn_play_rect = QRectF(self.width / 2.0 - 35.0, 92.0, 30.0, 22.0)
        if btn_play_rect.contains(pos):
            self.toggle_run()
            event.accept()
            return

        # Check Reset Button
        btn_reset_rect = QRectF(self.width / 2.0 + 5.0, 92.0, 30.0, 22.0)
        if btn_reset_rect.contains(pos):
            self.reset()
            event.accept()
            return

        # Otherwise allow moving card
        super().mousePressEvent(event)

    def _on_tick(self) -> None:
        if not self._is_running:
            return

        if self._is_stopwatch:
            self._remaining_seconds += 0.1
            self.time_updated.emit(self._remaining_seconds)
            self.update()
        else:
            self._remaining_seconds -= 0.1
            if self._remaining_seconds <= 0.0:
                self._remaining_seconds = 0.0
                self._is_running = False
                self._timer.stop()
                self.timer_finished.emit()
                try:
                    from src.utils.audio_feedback import AudioFeedbackManager
                    AudioFeedbackManager.instance().play_timer_chime()
                except Exception:
                    pass
            self.time_updated.emit(self._remaining_seconds)
            self.update()

    def start(self) -> None:
        """Start countdown or stopwatch."""
        if not self._is_running:
            self._is_running = True
            self._timer.start()
            self.update()

    def pause(self) -> None:
        """Pause timer."""
        if self._is_running:
            self._is_running = False
            self._timer.stop()
            self.update()

    def toggle_run(self) -> bool:
        """Toggle between start and pause; returns is_running state."""
        if self._is_running:
            self.pause()
        else:
            self.start()
        return self._is_running

    def reset(self) -> None:
        """Reset time to initial duration and pause."""
        self.pause()
        if self._is_stopwatch:
            self._remaining_seconds = 0.0
        else:
            self._remaining_seconds = self._total_seconds
        self.update()

    def set_duration(self, seconds: int) -> None:
        """Set countdown duration in seconds (0 = Stopwatch mode)."""
        self.pause()
        if seconds <= 0:
            self._is_stopwatch = True
            self._total_seconds = 0.0
            self._remaining_seconds = 0.0
        else:
            self._is_stopwatch = False
            self._total_seconds = float(seconds)
            self._remaining_seconds = float(seconds)
        self.update()

    def _cycle_preset(self) -> None:
        """Cycle to next time preset."""
        self._preset_idx = (self._preset_idx + 1) % len(self.PRESETS)
        dur, _ = self.PRESETS[self._preset_idx]
        self.set_duration(dur)

    def is_running(self) -> bool:
        return self._is_running

    def remaining_seconds(self) -> float:
        return self._remaining_seconds

    def total_seconds(self) -> float:
        return self._total_seconds

    def is_stopwatch(self) -> bool:
        return self._is_stopwatch

    def stop(self) -> None:
        """Stop timer and reset to initial duration."""
        self.pause()
        self.reset()

    def air_click_at(self, canvas_x: float, canvas_y: float) -> bool:
        """
        Allows touchless air tapping on presentation timer controls.
        Returns True if an interactive button was activated.
        """
        local_pos = self.mapFromScene(QPointF(canvas_x, canvas_y))
        px, py = local_pos.x(), local_pos.y()

        # 1. Close Button (top-right)
        close_rect = QRectF(self.width - 32.0, 6.0, 26.0, 26.0)
        if close_rect.contains(px, py):
            self.stop()
            self.setVisible(False)
            return True

        # 2. Preset Pill Button (cycle presets)
        pill_rect = QRectF(10.0, 6.0, 118.0, 30.0)
        if pill_rect.contains(px, py):
            self._cycle_preset()
            return True

        # 3. Play / Pause Button
        btn_play_rect = QRectF(self.width / 2.0 - 40.0, 88.0, 40.0, 30.0)
        if btn_play_rect.contains(px, py):
            self.toggle_run()
            return True

        # 4. Reset Button
        btn_reset_rect = QRectF(self.width / 2.0, 88.0, 40.0, 30.0)
        if btn_reset_rect.contains(px, py):
            self.reset()
            return True

        return False

    def cleanup(self) -> None:
        """Safely stop tick timer and disconnect signals to prevent memory leaks."""
        self._timer.stop()
        try:
            self._timer.timeout.disconnect()
        except Exception:
            pass
