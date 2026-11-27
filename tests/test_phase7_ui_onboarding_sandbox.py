#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 Verification Suite: Streamlined Modern UI (85-90%+ Canvas Share, Zero Overflow/Overlap
at 1366x768 & 1920x1080, Synchronized Dark/Light Mode), 4-Step Onboarding Tour & Live Gesture Sandbox.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QPushButton

import main as app_main
from cv_engine.user_guide_dialog import GestureSandboxWidget, OnboardingTourDialog, UserGuideDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _assert_no_overflow_or_overlap(container, label: str) -> None:
    cw = container.width()
    buttons = [b for b in container.findChildren(QPushButton) if b.isVisible()]
    rects = []
    for btn in buttons:
        g = btn.geometry()
        assert g.width() > 0 and g.height() > 0, f"{label}: zero-size button {btn.text()!r}"
        assert g.right() <= cw + 2, f"{label}: button {btn.text()!r} overflows container ({g.right()} > {cw})"
        rects.append((btn.text(), g))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            t1, r1 = rects[i]
            t2, r2 = rects[j]
            inter = r1.intersected(r2)
            assert inter.width() <= 0 or inter.height() <= 0, (
                f"{label}: buttons {t1!r} and {t2!r} overlap by {inter.width()}x{inter.height()}"
            )


def test_clean_ui_canvas_ratio_zero_overflow_and_theme_sync(qapp):
    orig_start = app_main.CameraHandWorker.start
    app_main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
    win = app_main.AirTeachMainWindow()
    try:
        win.stack.setCurrentIndex(1)
        win.show()

        for (w, h) in ((1366, 768), (1920, 1080)):
            win.resize(w, h)
            qapp.processEvents()
            win_area = max(1, win.width() * win.height())
            canvas_area = win.canvas.width() * win.canvas.height()
            ratio = canvas_area / float(win_area)
            assert ratio >= 0.85, f"At {w}x{h}, canvas ratio {ratio*100:.1f}% < 85%"
            _assert_no_overflow_or_overlap(win.header_bar, f"Header@{w}x{h}")
            _assert_no_overflow_or_overlap(win.floating_dock, f"FloatingDock@{w}x{h}")

        # Full synchronized Dark <-> Light Mode toggle
        assert win.dark_mode is True and win.canvas.dark_mode is True
        win._on_toggle_theme()
        qapp.processEvents()
        assert win.dark_mode is False and win.canvas.dark_mode is False
        assert win.theme_toggle_btn.text() == "🌙"
        assert "#F8FAFC" in win.header_bar.styleSheet()
        win._on_toggle_theme()
        qapp.processEvents()
        assert win.dark_mode is True and win.canvas.dark_mode is True
        assert win.theme_toggle_btn.text() == "☀"
    finally:
        app_main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
        win.shutdown_resources()
        win.close()


def test_onboarding_tour_and_live_gesture_sandbox(qapp):
    # 1. 4-step Onboarding Tour
    tour = OnboardingTourDialog()
    tour.show()
    qapp.processEvents()
    assert tour.stack.count() == 4
    assert "1 / 4" in tour.step_pill.text()
    tour._on_next()
    assert "2 / 4" in tour.step_pill.text()
    tour._on_next()
    tour._on_next()
    assert "4 / 4" in tour.step_pill.text()
    tour._on_prev()
    assert "3 / 4" in tour.step_pill.text()
    tour.close()

    # 2. 5-tab UserGuideDialog & Live Gesture Sandbox green/red feedback
    guide = UserGuideDialog()
    guide.show()
    qapp.processEvents()
    assert guide.tabs.count() == 5
    sb: GestureSandboxWidget = guide.sandbox

    # Unrecognized sideways palm -> Red feedback
    m_bad = sb.ingest_camera_packet({
        "pose_tag": "HOVER",
        "dual_hand": {"two_hand_mode": "SINGLE"},
        "vision_telemetry": {"palm_facing_ok": False},
    })
    assert m_bad == ""
    assert "CHƯA KHỚP" in sb.live_feedback_lbl.text()

    # Valid 1-hand INDEX + PINCH + OPEN_PALM + 2-hand DUAL_PINCH_ZOOM + DUAL_PALM_PAN -> Green feedback
    assert sb.ingest_camera_packet({"pose_tag": "INDEX", "dual_hand": {"two_hand_mode": "SINGLE"}}) == "INDEX"
    assert sb.ingest_camera_packet({"pose_tag": "PINCH", "dual_hand": {"two_hand_mode": "SINGLE"}}) == "PINCH"
    assert sb.ingest_camera_packet({"pose_tag": "OPEN_PALM", "dual_hand": {"two_hand_mode": "SINGLE"}}) == "OPEN_PALM"
    assert sb.ingest_camera_packet({"pose_tag": "PINCH", "dual_hand": {"two_hand_mode": "DUAL_PINCH_ZOOM"}}) == "DUAL_PINCH_ZOOM"
    assert sb.ingest_camera_packet({"pose_tag": "FIST", "dual_hand": {"two_hand_mode": "DUAL_PALM_PAN"}}) == "DUAL_PALM_PAN"
    assert len(sb.passed_gestures) == 5
    assert "5 / 8" in sb.progress_lbl.text()
    assert "KHỚP CHÍNH XÁC" in sb.live_feedback_lbl.text()

    sb.reset_progress()
    assert len(sb.passed_gestures) == 0
    guide.close()
