"""
Unit and integration tests for native camera acquisition, backend negotiation,
reconnection controllers, and background worker telemetry.
"""

import sys
import time
import pytest
import cv2
import numpy as np

from src.core.camera_capture import (
    CameraStatus,
    get_platform_backends,
    open_camera_capture,
    CameraReconnectController,
    enumerate_available_cameras,
)
from src.core.hand_gesture_engine import HandTrackingWorker


class TestCameraBackendNegotiation:
    """Verify platform-specific backend resolution and prioritization."""

    def test_platform_backends_windows_order(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        backends = get_platform_backends()
        assert len(backends) >= 2
        backend_names = [b[1] for b in backends]
        assert "DirectShow" in backend_names
        assert "Default" in backend_names
        # DirectShow should be the primary candidate on Windows
        assert backends[0][1] == "DirectShow"

    def test_platform_backends_linux_order(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        backends = get_platform_backends()
        assert len(backends) >= 1
        backend_names = [b[1] for b in backends]
        assert "Default" in backend_names
        if getattr(cv2, "CAP_V4L2", None) is not None:
            assert backends[0][1] == "V4L2"

    def test_platform_backends_with_preferred_override(self):
        preferred = cv2.CAP_ANY
        backends = get_platform_backends(preferred_backend=preferred)
        assert backends[0][0] == preferred
        assert backends[0][1] == "Default"

    def test_open_camera_capture_negative_index_is_noop(self):
        cap, backend = open_camera_capture(-1, 1280, 720, 30)
        assert cap is None
        assert backend == "Disabled"

        cap, backend = open_camera_capture(-99, 1920, 1080, 60)
        assert cap is None
        assert backend == "Disabled"


class TestCameraReconnectController:
    """Verify state transitions, failure counters, cooldowns, and telemetry."""

    def test_controller_initial_state_and_telemetry(self):
        ctrl = CameraReconnectController(
            camera_index=0,
            width=1280,
            height=720,
            fps=30,
            max_consecutive_failures=5,
            reconnect_cooldown_s=0.2,
        )
        assert ctrl.status == CameraStatus.STANDBY
        assert ctrl.consecutive_failures == 0
        assert ctrl.total_reconnects == 0
        assert ctrl.active_backend == "None"

        telemetry = ctrl.get_telemetry()
        assert telemetry["camera_index"] == 0
        assert telemetry["status"] == "standby"
        assert telemetry["target_resolution"] == "1280x720"
        assert telemetry["target_fps"] == 30

    def test_controller_success_and_failure_cycle(self):
        status_events: list[tuple[CameraStatus, str]] = []

        def on_status(status, msg):
            status_events.append((status, msg))

        ctrl = CameraReconnectController(
            camera_index=1,
            max_consecutive_failures=3,
            reconnect_cooldown_s=0.1,
            on_status_change=on_status,
        )

        # Successful frame
        ctrl.record_frame_success(backend_name="DirectShow")
        assert ctrl.status == CameraStatus.CONNECTED
        assert ctrl.consecutive_failures == 0
        assert ctrl.active_backend == "DirectShow"
        assert len(status_events) == 1
        assert status_events[0][0] == CameraStatus.CONNECTED

        # Frame failures below threshold
        assert ctrl.record_frame_failure() is False
        assert ctrl.consecutive_failures == 1
        assert ctrl.record_frame_failure() is False
        assert ctrl.consecutive_failures == 2
        assert ctrl.status == CameraStatus.CONNECTED

        # 3rd failure reaches threshold
        reconnect_needed = ctrl.record_frame_failure()
        assert reconnect_needed is True
        assert ctrl.consecutive_failures == 3
        assert ctrl.status == CameraStatus.RECONNECTING
        assert any(ev[0] == CameraStatus.RECONNECTING for ev in status_events)

    def test_controller_cooldown_timer(self):
        ctrl = CameraReconnectController(reconnect_cooldown_s=0.1)
        assert ctrl.should_attempt_reconnect_now() is True

        # Simulate execution of reconnect
        ctrl._last_reconnect_time = time.time()
        assert ctrl.should_attempt_reconnect_now() is False

        time.sleep(0.12)
        assert ctrl.should_attempt_reconnect_now() is True

    def test_controller_execute_reconnect_negative_index(self):
        ctrl = CameraReconnectController(camera_index=-1)
        new_cap, backend = ctrl.execute_reconnect(current_cap=None)
        assert new_cap is None
        assert backend == "None"
        assert ctrl.status == CameraStatus.DISCONNECTED


class TestHandTrackingWorkerCameraIntegration:
    """Verify HandTrackingWorker handles camera configuration and status updates."""

    def test_worker_negative_camera_index_clean_exit(self):
        status_updates: list[tuple[str, str]] = []

        def on_status(status_str, msg):
            status_updates.append((status_str, msg))

        worker = HandTrackingWorker(
            camera_index=-1,
            on_camera_status=on_status,
        )
        assert worker.current_backend_name == "None"

        # Calling start should gracefully recognize negative camera index
        worker.start()

        telemetry = worker.get_camera_status()
        assert telemetry["status"] == "standby"
        assert telemetry["camera_index"] == -1

    def test_worker_preferred_backend_configuration(self):
        worker = HandTrackingWorker(
            camera_index=-2,
            preferred_backend=cv2.CAP_DSHOW,
            max_consecutive_failures=15,
        )
        assert worker.preferred_backend == cv2.CAP_DSHOW
        assert worker.max_consecutive_failures == 15
        worker.start()
        status = worker.get_camera_status()
        assert status["status"] == "standby"


class TestCameraEnumeration:
    """Verify hardware enumeration utility executes cleanly."""

    def test_enumerate_available_cameras_runs_without_exceptions(self):
        # Scan at most 1 device to keep test sub-second
        devices = enumerate_available_cameras(max_tested=1)
        assert isinstance(devices, list)
        for dev in devices:
            assert "index" in dev
            assert "name" in dev
            assert "backend_name" in dev
            assert "width" in dev
            assert "height" in dev
            assert "fps" in dev


class TestQtCameraWorker:
    """Verify PySide6 signals and camera worker integration."""

    def test_qt_camera_worker_signals_and_telemetry(self, qapp):
        from src.ui.main_window import _QtCameraWorker

        worker = _QtCameraWorker(camera_index=-1)
        received_status = []

        worker.camera_status_changed.connect(lambda s, m: received_status.append((s, m)))

        # Simulate camera status change
        worker._on_camera_status_cb("connected", "Device online")
        assert len(received_status) == 1
        assert received_status[0] == ("connected", "Device online")

        telemetry = worker.get_camera_status()
        assert telemetry["camera_index"] == -1
        assert "status" in telemetry
