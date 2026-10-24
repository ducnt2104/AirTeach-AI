"""
AirTeach AI - Camera Widget
============================
QLabel subclass that displays the live webcam feed.

The actual camera loop runs inside HandTrackingWorker (QThread in
hand_gesture_engine.py). This widget only receives pre-processed BGR
frames via a Qt Signal and paints them to screen — UI thread is never
blocked by I/O.

Features:
  - Optional overlay of MediaPipe hand landmark skeleton
  - Optional CLAHE-enhanced preview toggle
  - Minimal 240×180 fixed preview; scales with widget resize
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

logger = logging.getLogger(__name__)


class CameraWidget(QLabel):
    """
    Displays the live webcam feed.

    Receives BGR numpy arrays via :py:meth:`on_frame` and converts them
    to QPixmap for display. No blocking I/O on the UI thread.

    Signals:
        frame_displayed: emitted each time a frame is painted.
    """

    frame_displayed = Signal()

    # Minimum display size
    _MIN_W = 240
    _MIN_H = 180

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(self._MIN_W, self._MIN_H)
        self.setStyleSheet(
            "background-color: #1a1a2e;"
            "border: 2px solid #3d5a80;"
            "border-radius: 8px;"
        )
        self.setText("📷 Camera offline")
        self.setStyleSheet(
            self.styleSheet()
            + "color: #8d99ae; font-size: 13px;"
        )
        self._show_landmarks = True
        self._current_frame: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def on_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Slot: receive a BGR uint8 frame from HandTrackingWorker and display it.
        Must be connected via Qt.ConnectionType.QueuedConnection so this runs
        on the UI thread regardless of which thread emits the signal.
        """
        self._current_frame = frame_bgr
        self._paint_frame(frame_bgr)

    def set_show_landmarks(self, show: bool) -> None:
        self._show_landmarks = show

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _paint_frame(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR numpy → QPixmap and paint to label, preserving aspect ratio."""
        try:
            import cv2  # noqa: PLC0415
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qi)
            target_w = max(self._MIN_W, self.width())
            target_h = max(self._MIN_H, self.height())
            scaled = pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.frame_displayed.emit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CameraWidget paint error: %s", exc)

    def sizeHint(self) -> QSize:
        return QSize(320, 240)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._current_frame is not None:
            self._paint_frame(self._current_frame)
