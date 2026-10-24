"""
Tests for AirTeach AI Crash Diagnostics, LessonBoard Recognized Export & Resiliency
===================================================================================
Verifies:
  - CrashDiagnostics logs structured JSON reports for uncaught exceptions
  - Automatic redaction of sensitive credentials (API keys, Bearer tokens, PINs)
  - TelemetryLogger fail-safe recovery on disk write failure
  - LessonBoard.get_recognized_items_for_export serialization
  - MultiHandCoordinator perspective coordinate accuracy and dead track memory eviction
"""
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsTextItem

from src.core.multi_hand_coordinator import (
    AirClickType,
    HandRole,
    MultiHandCoordinator,
    MultiHandMode,
)
from src.ui.lesson_board import LessonBoard
from src.utils.crash_diagnostics import (
    CrashDiagnostics,
    install_crash_handler,
    sanitize_text,
)
from src.utils.telemetry_logger import TelemetryLogger, TelemetryRecord
from tests.test_core_algorithms import _make_synthetic_landmarks


# ─────────────────────────────────────────────────────────────────────────────
# 1. Crash Diagnostics & Redaction Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sanitize_text_redacts_credentials():
    raw_text = (
        "Error in API request with key AIzaSyD9x8y7Z6w5v4u3t2s1r0q9p8o7n6m5l4k and "
        "auth Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz and secret pin='987654'"
    )
    clean = sanitize_text(raw_text)

    assert "AIzaSyD9x8y7Z6w5v4u3t2s1r0q9p8o7n6m5l4k" not in clean
    assert "[REDACTED_GOOGLE_API_KEY]" in clean
    assert "Bearer [REDACTED_TOKEN]" in clean
    assert "987654" not in clean
    assert "[REDACTED]" in clean


def test_crash_diagnostics_records_json(tmp_path):
    diag = CrashDiagnostics(reports_dir=tmp_path)

    try:
        raise ValueError("Critical simulated crash with secret pin='123456'")
    except ValueError as exc:
        exc_type, exc_val, exc_tb = sys.exc_info()
        report_path = diag.record_exception(exc_type, exc_val, exc_tb, thread_name="TestThread")

    assert report_path is not None
    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["exception_type"] == "ValueError"
    assert "123456" not in data["exception_message"]
    assert "[REDACTED]" in data["exception_message"]
    assert data["thread_name"] == "TestThread"
    assert "environment" in data
    assert "platform" in data["environment"]
    assert "python_version" in data["environment"]

    recent = diag.get_recent_crashes(limit=5)
    assert len(recent) == 1
    assert recent[0]["exception_type"] == "ValueError"


def test_crash_diagnostics_install_and_uninstall(tmp_path):
    orig_hook = sys.excepthook
    diag = install_crash_handler(reports_dir=tmp_path)
    assert sys.excepthook != orig_hook

    diag.uninstall()
    assert sys.excepthook == orig_hook


# ─────────────────────────────────────────────────────────────────────────────
# 2. Telemetry Logger Fail-Safe Retry Test
# ─────────────────────────────────────────────────────────────────────────────

def test_telemetry_logger_failsafe_recovery_on_oserror(tmp_path):
    log_file = tmp_path / "telemetry_test.csv"
    logger = TelemetryLogger(log_file)

    logger.record(fps=60.0, frame_latency_ms=16.6)
    logger.record(fps=59.5, frame_latency_ms=16.8)

    # Simulate OSError during disk write
    with patch.object(Path, "open", side_effect=OSError("Disk write error")):
        logger._flush_to_disk()

    # The 2 records should be restored to buffer rather than permanently lost
    assert len(logger._buffer) == 2

    # Now flush without error
    logger._flush_to_disk()
    assert len(logger._buffer) == 0
    assert log_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LessonBoard get_recognized_items_for_export Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_lesson_board_recognized_items_export(qapp):
    board = LessonBoard()

    # Add text overlay
    text_item = board.add_text_overlay("E = mc^2", 100.0, 150.0)
    text_item.setData(1, {
        "text": "E = mc^2",
        "latex": "E = mc^2",
        "conf": 0.98,
        "is_formula": True,
    })

    # Add dummy latex overlay
    dummy_rgba = np.full((30, 80, 4), 255, dtype=np.uint8)
    latex_item = board.add_latex_overlay(dummy_rgba, 200.0, 300.0)
    latex_item._latex_str = r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}"

    exported = board.get_recognized_items_for_export(page_index=1)
    assert len(exported) >= 2

    item0 = next((it for it in exported if it["text"] == "E = mc^2"), None)
    assert item0 is not None
    assert item0["page_index"] == 1
    assert item0["is_formula"] is True
    assert item0["confidence"] == 0.98
    assert len(item0["bounding_box"]) == 4

    item1 = next((it for it in exported if r"\int_0^\infty" in it["latex"]), None)
    assert item1 is not None
    assert item1["page_index"] == 1
    assert item1["is_formula"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. MultiHandCoordinator Perspective Accuracy & Dead Track Eviction Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_hand_coordinator_canvas_coordinates_accuracy():
    coordinator = MultiHandCoordinator(
        canvas_width=1920,
        canvas_height=1080,
        mode=MultiHandMode.DRAW_AND_TOOL,
    )

    # Center of frame: index tip at (0.50, 0.50)
    lm = _make_synthetic_landmarks(index_tip_y=0.50)
    for pt in lm:
        pt.x = 0.50
    lm[8].x = 0.50
    lm[8].y = 0.50

    raw_dets = [(lm, "Right", 0.92)]
    hands = coordinator.process_frame_hands(raw_dets, t_ms=0.0)

    assert len(hands) == 1
    h = hands[0]
    # In a 1920x1080 canvas, center should be near (960, 540), NOT pinned to (1920, 1080)
    assert 940.0 <= h.canvas_x <= 980.0, f"Expected center ~960, got {h.canvas_x}"
    assert 520.0 <= h.canvas_y <= 560.0, f"Expected center ~540, got {h.canvas_y}"


def test_multi_hand_coordinator_dead_track_eviction():
    coordinator = MultiHandCoordinator(
        canvas_width=1920,
        canvas_height=1080,
    )
    coordinator.tracker.max_missing_frames = 2

    lm = _make_synthetic_landmarks(index_tip_y=0.30)
    raw_dets = [(lm, "Right", 0.90)]

    # Frame 1: Hand appears
    coordinator.process_frame_hands(raw_dets, t_ms=0.0)
    assert len(coordinator._fsms) == 1
    tid = list(coordinator._fsms.keys())[0]

    # Frame 2 & 3: Hand missing, but within grace period
    coordinator.process_frame_hands([], t_ms=33.0)
    coordinator.process_frame_hands([], t_ms=66.0)
    assert tid in coordinator._fsms

    # Frame 4: Missing frames > max_missing_frames -> evicted
    coordinator.process_frame_hands([], t_ms=100.0)
    assert tid not in coordinator._fsms, "Dead track must be evicted from FSMs"
    assert tid not in coordinator.click_detector._history, "Dead track must be evicted from ClickDetector"
