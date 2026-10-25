"""
AirTeach AI — Native Real-Time Camera Acquisition & Adaptive Backend Manager.

Provides multi-platform backend negotiation (DirectShow, MSMF, V4L2, AVFoundation),
fail-safe camera opening with fallback cascades, automatic disconnection detection,
resilient auto-reconnect controller, and camera hardware enumeration.
"""

from __future__ import annotations

import logging
import sys
import time
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStatus(str, Enum):
    """Operational status of the video capture device."""
    STANDBY = "standby"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class CameraBackendInfo:
    """Descriptor for a platform-specific OpenCV capture backend."""

    def __init__(self, backend_id: int, name: str, platform_pattern: str) -> None:
        self.backend_id = backend_id
        self.name = name
        self.platform_pattern = platform_pattern

    def __repr__(self) -> str:
        return f"<CameraBackendInfo {self.name} (id={self.backend_id})>"


def get_platform_backends(preferred_backend: Optional[int] = None) -> list[tuple[int, str]]:
    """
    Return an ordered list of (backend_id, backend_name) prioritized for current OS.

    On Windows: cv2.CAP_DSHOW (low latency DirectShow) -> cv2.CAP_MSMF (Media Foundation) -> cv2.CAP_ANY.
    On Linux: cv2.CAP_V4L2 -> cv2.CAP_ANY.
    On macOS: cv2.CAP_AVFOUNDATION -> cv2.CAP_ANY.
    """
    backends: list[tuple[int, str]] = []

    if preferred_backend is not None:
        pref_name = "CustomPreferred"
        if preferred_backend == cv2.CAP_DSHOW:
            pref_name = "DirectShow"
        elif preferred_backend == getattr(cv2, "CAP_MSMF", -1):
            pref_name = "Media Foundation"
        elif preferred_backend == getattr(cv2, "CAP_V4L2", -1):
            pref_name = "V4L2"
        elif preferred_backend == getattr(cv2, "CAP_AVFOUNDATION", -1):
            pref_name = "AVFoundation"
        elif preferred_backend == cv2.CAP_ANY:
            pref_name = "Default"
        backends.append((preferred_backend, pref_name))

    if sys.platform == "win32":
        candidates = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (getattr(cv2, "CAP_MSMF", None), "Media Foundation"),
            (cv2.CAP_ANY, "Default"),
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            (getattr(cv2, "CAP_V4L2", None), "V4L2"),
            (cv2.CAP_ANY, "Default"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            (getattr(cv2, "CAP_AVFOUNDATION", None), "AVFoundation"),
            (cv2.CAP_ANY, "Default"),
        ]
    else:
        candidates = [
            (cv2.CAP_ANY, "Default"),
        ]

    for b_id, name in candidates:
        if b_id is not None and not any(existing_id == b_id for existing_id, _ in backends):
            backends.append((b_id, name))

    return backends


def open_camera_capture(
    camera_index: int,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    preferred_backend: Optional[int] = None,
    test_read: bool = True,
) -> tuple[Optional[cv2.VideoCapture], str]:
    """
    Open webcam with adaptive backend fallback cascade.

    Args:
        camera_index: Index of the camera device (< 0 returns None immediately).
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Target framerate in frames per second.
        preferred_backend: Optional explicit OpenCV backend ID to prioritize.
        test_read: If True, tests reading 1 frame to verify the device is functional.

    Returns:
        (VideoCapture, backend_name) if opened successfully, otherwise (None, "None").
    """
    if camera_index < 0:
        logger.info("Camera index is negative (%d); skipping camera hardware initialization.", camera_index)
        return None, "Disabled"

    backend_list = get_platform_backends(preferred_backend)

    for backend_id, backend_name in backend_list:
        cap: Optional[cv2.VideoCapture] = None
        try:
            if backend_id == cv2.CAP_ANY:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, backend_id)

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            # Configure resolution and target frame rate
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))

            # Set buffer size to 1 to reduce input latency if supported by backend
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if test_read:
                ret, test_frame = cap.read()
                if not ret or test_frame is None or test_frame.size == 0:
                    logger.debug(
                        "Test frame read failed for camera %d with backend %s; releasing and trying next.",
                        camera_index,
                        backend_name,
                    )
                    cap.release()
                    continue

            logger.info(
                "Successfully opened camera %d with backend %s (%dx%d @ %dfps).",
                camera_index,
                backend_name,
                width,
                height,
                fps,
            )
            return cap, backend_name

        except Exception as exc:
            logger.warning(
                "Exception opening camera %d with backend %s: %s",
                camera_index,
                backend_name,
                exc,
            )
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    logger.error("Failed to open camera %d with all attempted backends: %s", camera_index, [b[1] for b in backend_list])
    return None, "None"


class CameraReconnectController:
    """
    Supervises camera stream health, detects sudden USB disconnects,
    manages backoff delays, and handles non-blocking re-connections.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        preferred_backend: Optional[int] = None,
        max_consecutive_failures: int = 30,
        reconnect_cooldown_s: float = 0.5,
        on_status_change: Optional[Callable[[CameraStatus, str], None]] = None,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.preferred_backend = preferred_backend
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.reconnect_cooldown_s = max(0.05, reconnect_cooldown_s)
        self.on_status_change = on_status_change

        self._status: CameraStatus = CameraStatus.STANDBY
        self._consecutive_failures: int = 0
        self._total_reconnects: int = 0
        self._last_reconnect_time: float = 0.0
        self._active_backend: str = "None"

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def set_status(self, new_status: CameraStatus, message: str = "") -> None:
        """Update operational status and notify listener if changed."""
        if self._status != new_status:
            self._status = new_status
            logger.info("Camera %d status changed to %s: %s", self.camera_index, new_status.value, message)
            if self.on_status_change:
                try:
                    self.on_status_change(new_status, message)
                except Exception as exc:
                    logger.warning("Error in on_status_change callback: %s", exc)

    def record_frame_success(self, backend_name: str = "") -> None:
        """Called whenever a frame is successfully acquired from the capture stream."""
        if backend_name:
            self._active_backend = backend_name
        self._consecutive_failures = 0
        if self._status != CameraStatus.CONNECTED:
            self.set_status(CameraStatus.CONNECTED, f"Stream active via {self._active_backend}")

    def record_frame_failure(self) -> bool:
        """
        Record a failed frame read attempt.

        Returns:
            True if failures exceeded max_consecutive_failures and reconnect is needed.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_consecutive_failures:
            if self._status != CameraStatus.RECONNECTING:
                self.set_status(
                    CameraStatus.RECONNECTING,
                    f"Frame read failed {self._consecutive_failures} consecutive times; requesting auto-reconnect",
                )
            return True
        return False

    def should_attempt_reconnect_now(self) -> bool:
        """Check if enough cooldown time has elapsed since the last reconnect attempt."""
        return (time.time() - self._last_reconnect_time) >= self.reconnect_cooldown_s

    def execute_reconnect(
        self,
        current_cap: Optional[cv2.VideoCapture] = None,
    ) -> tuple[Optional[cv2.VideoCapture], str]:
        """
        Safely dispose of the existing dead capture and attempt to re-open the camera.

        Returns:
            (new_cap, backend_name)
        """
        self._last_reconnect_time = time.time()
        self.set_status(CameraStatus.RECONNECTING, "Attempting hardware reconnection...")

        if current_cap is not None:
            try:
                current_cap.release()
            except Exception as exc:
                logger.debug("Exception releasing old capture during reconnect: %s", exc)

        new_cap, backend_name = open_camera_capture(
            camera_index=self.camera_index,
            width=self.width,
            height=self.height,
            fps=self.fps,
            preferred_backend=self.preferred_backend,
            test_read=True,
        )

        if new_cap is not None and new_cap.isOpened():
            self._total_reconnects += 1
            self._consecutive_failures = 0
            self._active_backend = backend_name
            self.set_status(CameraStatus.CONNECTED, f"Reconnected successfully via {backend_name}")
            return new_cap, backend_name
        else:
            self.set_status(CameraStatus.DISCONNECTED, "Hardware reconnection attempt failed")
            return None, "None"

    def get_telemetry(self) -> dict:
        """Return diagnostic metrics dictionary for monitoring and UI overlays."""
        return {
            "camera_index": self.camera_index,
            "status": self._status.value,
            "active_backend": self._active_backend,
            "consecutive_failures": self._consecutive_failures,
            "total_reconnects": self._total_reconnects,
            "target_resolution": f"{self.width}x{self.height}",
            "target_fps": self.fps,
        }


def enumerate_available_cameras(max_tested: int = 4) -> list[dict]:
    """
    Enumerate available physical camera hardware devices on the system.

    Args:
        max_tested: Maximum number of sequential device indices to probe (0..max_tested-1).

    Returns:
        List of dicts containing device index, name, backend, resolution, and fps.
    """
    discovered: list[dict] = []
    backends = get_platform_backends()

    for idx in range(max(1, max_tested)):
        for backend_id, backend_name in backends:
            cap: Optional[cv2.VideoCapture] = None
            try:
                if backend_id == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend_id)

                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        discovered.append({
                            "index": idx,
                            "name": f"Camera {idx} ({backend_name})",
                            "backend_name": backend_name,
                            "backend_id": backend_id,
                            "width": w,
                            "height": h,
                            "fps": round(fps, 1),
                        })
                        cap.release()
                        break  # Found working backend for this camera index
                if cap is not None:
                    cap.release()
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

    return discovered
