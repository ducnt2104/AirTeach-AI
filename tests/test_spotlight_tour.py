#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Suite for Interactive Spotlight Tour Overlay (`cv_engine/spotlight_tour.py`),
`UserGuideDialog` Spotlight Trigger (`cv_engine/user_guide_dialog.py`),
and `OPEN_AI_CHAT` Radial Pie Sub-Action (`cv_engine/radial_pie_menu.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
from PySide6.QtWidgets import QApplication

import main as app_main
from cv_engine.radial_pie_menu import (
    DEFAULT_PIE_DEADZONE_PX,
    PIE_MENU_SLICES,
    PIE_SUB_ACTIONS,
    resolve_pie_sub_item,
)
from cv_engine.spotlight_tour import (
    SpotlightStep,
    SpotlightTourOverlay,
    build_default_airteach_spotlight_steps,
)
from cv_engine.user_guide_dialog import UserGuideDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["test-spotlight-tour"])
    return app


def _make_studio_window(qapp: QApplication):
    orig_start = app_main.CameraHandWorker.start
    app_main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
    win = app_main.AirTeachMainWindow()
    win.stack.setCurrentIndex(1)
    win.resize(1366, 768)
    win.show()
    qapp.processEvents()
    return win, orig_start


def _teardown_studio_window(win, orig_start) -> None:
    app_main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
    win.shutdown_resources()
    win.close()


def test_spotlight_6_steps_navigation_cutouts_and_clamping_resolutions(qapp):
    win, orig_start = _make_studio_window(qapp)
    try:
        steps = build_default_airteach_spotlight_steps(win)
        assert len(steps) == 6
        assert [s.step_id for s in steps] == [
            "radial_menu",
            "pan_zoom",
            "object_canvas",
            "gestures",
            "documents",
            "ai_hybrid",
        ]

        overlay = SpotlightTourOverlay(win, steps)
        assert overlay.step_count == 6
        assert overlay.isVisible() is False

        finished_events: List[bool] = []
        sandbox_events: List[bool] = []
        overlay.tour_finished.connect(finished_events.append)
        overlay.open_sandbox_requested.connect(lambda: sandbox_events.append(True))

        # Test both 1366x768 and 1920x1080 including dynamic resize tracking via eventFilter
        for (w, h) in ((1366, 768), (1920, 1080)):
            win.resize(w, h)
            qapp.processEvents()
            overlay.start(0)
            qapp.processEvents()

            assert overlay.isVisible() is True
            assert overlay.width() == win.width()
            assert overlay.height() == win.height()

            for idx in range(overlay.step_count):
                assert overlay.current_step_index == idx
                assert f"Bước {idx + 1} / 6" in overlay.progress_pill.text()
                assert overlay.prev_btn.isEnabled() is (idx > 0)

                if idx == overlay.step_count - 1:
                    assert overlay.next_btn.text() == "Hoàn tất ✓"
                else:
                    assert overlay.next_btn.text() == "Tiếp theo"

                # Quick sandbox button ONLY on gestures step
                is_gestures = steps[idx].step_id == "gestures"
                assert overlay.sandbox_btn.isVisible() is is_gestures

                # Cutout rectangle must be valid & within parent bounds
                cutout: QRectF = overlay.active_spotlight_rect()
                assert cutout.width() >= 12.0 and cutout.height() >= 12.0
                assert cutout.left() >= 0.0 and cutout.top() >= 0.0
                assert cutout.right() <= win.width() + 1.0
                assert cutout.bottom() <= win.height() + 1.0

                # Callout card must NEVER clip off-screen at 1366x768 or 1920x1080
                cg = overlay.card.geometry()
                assert cg.width() >= 240 and cg.height() >= 140
                assert cg.left() >= 0, f"Step {idx}@{w}x{h}: card clips left ({cg})"
                assert cg.top() >= 0, f"Step {idx}@{w}x{h}: card clips top ({cg})"
                assert cg.right() < win.width(), f"Step {idx}@{w}x{h}: card clips right ({cg} vs {win.width()})"
                assert cg.bottom() < win.height(), f"Step {idx}@{w}x{h}: card clips bottom ({cg} vs {win.height()})"

                if idx < overlay.step_count - 1:
                    overlay.next_step()
                    qapp.processEvents()

            # Test prev_step backward navigation
            overlay.prev_step()
            qapp.processEvents()
            assert overlay.current_step_index == 4
            overlay.next_step()
            qapp.processEvents()
            assert overlay.current_step_index == 5

        # Dynamic resize while open updates overlay geometry & spotlight rect
        win.resize(1366, 768)
        qapp.processEvents()
        overlay.start(1)  # pan_zoom -> win.canvas.nav_hud
        qapp.processEvents()
        rect_small = overlay.active_spotlight_rect()
        win.resize(1920, 1080)
        qapp.processEvents()
        qapp.processEvents()
        assert overlay.size() == win.size()
        rect_large = overlay.active_spotlight_rect()
        assert rect_large.right() >= rect_small.right() + 190.0
        assert rect_large.bottom() >= rect_small.bottom() + 250.0
        cg_after = overlay.card.geometry()
        assert cg_after.left() >= 0 and cg_after.top() >= 0
        assert cg_after.right() < win.width() and cg_after.bottom() < win.height()

        # Gestures step fallback: mini_hud visible vs hidden -> toggle_cam_btn
        win.canvas.mini_hud.setVisible(True)
        qapp.processEvents()
        overlay.start(3)
        qapp.processEvents()
        rect_with_hud = overlay.active_spotlight_rect()
        assert rect_with_hud.height() > 50.0  # mini_hud is taller than header cam button

        win.canvas.mini_hud.setVisible(False)
        qapp.processEvents()
        rect_with_cam_btn = overlay.active_spotlight_rect()
        cam_tl = win.toggle_cam_btn.mapTo(win, QPoint(0, 0))
        assert abs(rect_with_cam_btn.center().x() - (cam_tl.x() + win.toggle_cam_btn.width() * 0.5)) < 15.0
        assert abs(rect_with_cam_btn.center().y() - (cam_tl.y() + win.toggle_cam_btn.height() * 0.5)) < 15.0

        # Clicking "🧪 Thử Cử Chỉ" emits open_sandbox_requested
        overlay.sandbox_btn.click()
        qapp.processEvents()
        assert len(sandbox_events) == 1
        assert overlay.isVisible() is False

        # Finish tour emits tour_finished(dont_show_again)
        overlay.start(5)
        qapp.processEvents()
        overlay.dont_show_chk.setChecked(False)
        overlay.next_step()  # On last step -> finish_tour()
        qapp.processEvents()
        assert overlay.isVisible() is False
        assert finished_events == [False]

        overlay.start(5)
        qapp.processEvents()
        overlay.dont_show_chk.setChecked(True)
        overlay.finish_tour()
        qapp.processEvents()
        assert finished_events == [False, True]
    finally:
        _teardown_studio_window(win, orig_start)


def test_spotlight_paint_event_oddeven_cutout_and_esc_skip(qapp):
    win, orig_start = _make_studio_window(qapp)
    try:
        win.resize(1366, 768)
        qapp.processEvents()
        steps = build_default_airteach_spotlight_steps(win)
        overlay = SpotlightTourOverlay(win, steps)

        skipped: List[bool] = []
        overlay.tour_finished.connect(skipped.append)

        overlay.start(0)  # radial_menu center cutout
        qapp.processEvents()

        # Render overlay onto a transparent offscreen QImage to verify paintEvent pixels
        img = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        overlay.render(img)

        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((img.height(), img.width(), 4))
        nonzero_pixels = int(np.count_nonzero(arr[:, :, 3]))
        assert nonzero_pixels > 50_000, f"Expected substantial painted overlay pixels, got {nonzero_pixels}"

        # Center of cutout (not overlapped by card) must remain transparent via OddEvenFill
        cutout = overlay.active_spotlight_rect()
        cx = int(round(cutout.center().x()))
        cy = int(round(cutout.center().y()))
        assert not overlay.card.geometry().contains(cx, cy), "Callout card should not cover cutout center"
        center_px = QColor(img.pixelColor(cx, cy))
        assert center_px.alpha() == 0, f"Cutout interior should be 100% clear, got alpha={center_px.alpha()}"

        # Top-left corner outside cutout & card must carry the semi-transparent dark backdrop
        bg_px = QColor(img.pixelColor(12, 12))
        assert 150 <= bg_px.alpha() <= 215, f"Expected semi-transparent backdrop alpha ~188, got {bg_px.alpha()}"

        # Cutout top edge must carry glowing accent stroke (non-zero bright cyan)
        edge_px = QColor(img.pixelColor(cx, int(round(cutout.top()))))
        assert edge_px.alpha() > 120 and edge_px.green() > 140, f"Expected glowing accent edge, got {edge_px}"

        # Pressing Esc triggers skip_tour() cleanly
        esc_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        overlay.keyPressEvent(esc_ev)
        qapp.processEvents()
        assert overlay.isVisible() is False
        assert skipped == [True]
    finally:
        _teardown_studio_window(win, orig_start)


def test_radial_pie_open_ai_chat_and_user_guide_spotlight_trigger(qapp):
    # 1. Verify all 8 main PIE_MENU_SLICES & DEFAULT_PIE_DEADZONE_PX remain 100% intact
    assert DEFAULT_PIE_DEADZONE_PX == 22.0
    assert [s.action_id for s in PIE_MENU_SLICES] == [
        "PEN",
        "ERASER",
        "SELECT",
        "PAN",
        "TEXT",
        "SHAPES",
        "UNDO",
        "CLEAR_OR_NEW",
    ]

    # 2. Verify OPEN_AI_CHAT in PIE_SUB_ACTIONS["CLEAR_OR_NEW"]
    clear_subs = PIE_SUB_ACTIONS["CLEAR_OR_NEW"]
    open_ai_specs = [s for s in clear_subs if s.action_id == "OPEN_AI_CHAT"]
    assert len(open_ai_specs) == 1
    spec = open_ai_specs[0]
    assert spec.short_label == "Mở AI Chat"
    assert spec.full_label_vi == "Mở Ngăn Trợ Lý AI Chat"
    assert spec.glyph == "🤖"
    assert spec.accent_hex == "#06B6D4"

    # Also verify resolve_pie_sub_item can hit OPEN_AI_CHAT on NW outer ring (slice 7)
    import math

    # Slice 7 (CLEAR_OR_NEW) center is 7 * 45 = 315 deg CW from North; 5th item (index 4) is at +44 deg from center -> 359 deg CW
    ang_cw = 315.0 + 44.0
    rad = math.radians(ang_cw - 90.0)
    hit = resolve_pie_sub_item(172.0 * math.cos(rad), 172.0 * math.sin(rad), 7)
    assert hit is not None and hit.action_id == "OPEN_AI_CHAT"

    # 3. Verify UserGuideDialog keeps 5 tabs & emits start_spotlight_requested on button click
    guide = UserGuideDialog()
    guide.show()
    qapp.processEvents()
    assert guide.tabs.count() == 5
    assert "Tour Spotlight" in guide.spotlight_btn.text()

    triggered: List[bool] = []
    guide.start_spotlight_requested.connect(lambda: triggered.append(True))
    guide.spotlight_btn.click()
    qapp.processEvents()
    assert triggered == [True]
    guide.close()


def test_spotlight_exact_1366x768_and_1920x1080_clamping(qapp):
    """Verify strict zero-clipping card clamping at exact 1366x768 and 1920x1080 across corner/edge targets."""
    from PySide6.QtWidgets import QPushButton, QWidget

    host = QWidget()
    tl_btn = QPushButton("Tài Liệu", host)
    tr_btn = QPushButton("Trợ Lý AI", host)
    br_hud = QWidget(host)
    bc_dock = QWidget(host)
    mid_hud = QWidget(host)

    def _layout_targets(w: int, h: int) -> None:
        host.setFixedSize(w, h)
        tl_btn.setGeometry(12, 6, 110, 30)
        tr_btn.setGeometry(w - 130, 6, 118, 30)
        mid_hud.setGeometry(w - 230, 52, 216, 130)
        br_hud.setGeometry(w - 390, h - 86, 376, 34)
        bc_dock.setGeometry((w - 860) // 2, h - 42, 860, 38)

    steps = [
        SpotlightStep("radial_menu", "Menu Tròn Chuột Phải", "Nội dung hướng dẫn menu tròn 8 hướng.", lambda: None, "#00F5D4", (0.31, 0.16, 0.38, 0.34), "Giữ chuột phải."),
        SpotlightStep("pan_zoom", "Di Chuyển & Zoom", "Nội dung hướng dẫn Pan & Zoom-to-Cursor.", lambda: br_hud, "#38BDF8"),
        SpotlightStep("object_canvas", "Object Độc Lập", "Nội dung hướng dẫn chọn, resize 8 góc, xoay.", lambda: bc_dock, "#A855F7"),
        SpotlightStep("gestures", "Cử Chỉ 1 & 2 Tay", "Nội dung hướng dẫn cử chỉ không chạm.", lambda: mid_hud, "#FEE440"),
        SpotlightStep("documents", "Mở Tài Liệu", "Nội dung hướng dẫn PDF, PPTX, DOCX, Ảnh.", lambda: tl_btn, "#10B981"),
        SpotlightStep("ai_hybrid", "AI Cloud & Local", "Nội dung hướng dẫn Gemini & GGUF Offline.", lambda: tr_btn, "#06B6D4"),
    ]

    host.show()
    overlay = SpotlightTourOverlay(host, steps)
    try:
        for (w, h) in ((1366, 768), (1920, 1080)):
            _layout_targets(w, h)
            qapp.processEvents()
            assert host.width() == w and host.height() == h
            overlay.start(0)
            qapp.processEvents()
            assert overlay.width() == w and overlay.height() == h

            for idx in range(overlay.step_count):
                cg = overlay.card.geometry()
                cut = overlay.active_spotlight_rect()
                assert cut.left() >= 0.0 and cut.top() >= 0.0
                assert cut.right() <= w and cut.bottom() <= h
                assert cg.left() >= 0 and cg.top() >= 0
                assert cg.right() < w and cg.bottom() < h
                if idx < overlay.step_count - 1:
                    overlay.next_step()
                    qapp.processEvents()
    finally:
        overlay.close()
        host.close()

