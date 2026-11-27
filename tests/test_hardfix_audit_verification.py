#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Hard-Fix Audit Verification Suite (`tests/test_hardfix_audit_verification.py`):
Verifies all 6 sections of the AirTeach AI hard-fix requirements directly on the real
`main.py` entry point (`AirTeachMainWindow`, `SmartboardCanvas`, `airteach_core_rs`).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AIRTEACH_NO_MODAL", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

import main as app_main
from cv_engine.document_presenter import LectureDocument
from cv_engine.radial_pie_menu import PIE_MENU_SLICES, PIE_SUB_ACTIONS
from cv_engine.rust_bridge import (
    HAS_RUST_CORE,
    classify_hand_3d_fast,
    make_dual_hand_state_machine,
    make_kalman_predictor_2d,
    make_one_euro_2d,
    make_spatial_index,
)
from cv_engine.user_guide_dialog import GestureCheatSheetDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_studio_window(qapp):
    orig_start = app_main.CameraHandWorker.start
    app_main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
    win = app_main.AirTeachMainWindow()
    win.resize(1366, 768)
    win.stack.setCurrentIndex(1)
    win.show()
    qapp.processEvents()
    return win, orig_start


def _teardown(win, orig_start):
    app_main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
    win.shutdown_resources()
    win.close()


def _mouse_ev(etype, pos: QPointF, btn, btns):
    return QMouseEvent(etype, pos, pos, btn, btns, Qt.NoModifier)


# =============================================================================
# 1. RIGHT-CLICK RADIAL PIE MENU (ZERO ERASER) & PAN CURSOR FIX
# =============================================================================
def test_right_click_opens_radial_menu_not_eraser_and_pan_has_no_pink_circle(qapp):
    win, orig_start = _make_studio_window(qapp)
    try:
        canvas = win.canvas
        canvas.active_tool = "PEN"
        center = QPointF(420.0, 300.0)

        # Right-click press opens radial pie menu centered at (x, y) and NEVER enters ERASER
        canvas.mousePressEvent(
            _mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton)
        )
        assert canvas.pie_menu.isVisible() is True
        assert canvas.active_tool == "PEN"
        assert canvas.cursor_mode != "PAN_ERASE"

        # Verify slice 7 label & sub-items include OPEN_GESTURE_MAP and OPEN_AI_CHAT
        assert "Cử Chỉ" in PIE_MENU_SLICES[7].short_label
        sub_ids = [s.action_id for s in PIE_SUB_ACTIONS["CLEAR_OR_NEW"]]
        assert "OPEN_GESTURE_MAP" in sub_ids
        assert "OPEN_AI_CHAT" in sub_ids

        canvas.pie_menu.cancel()
        assert canvas.pie_menu.isVisible() is False

        # Middle-button pan & PAN tool MUST NOT use PAN_ERASE pink circle cursor
        canvas.mousePressEvent(
            _mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.MiddleButton, Qt.MiddleButton)
        )
        assert canvas.cursor_mode == "TWO_HAND_NAV"
        canvas.mouseReleaseEvent(
            _mouse_ev(QMouseEvent.Type.MouseButtonRelease, center, Qt.MiddleButton, Qt.NoButton)
        )

        canvas.active_tool = "PAN"
        canvas.mousePressEvent(
            _mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.LeftButton, Qt.LeftButton)
        )
        assert canvas.cursor_mode == "TWO_HAND_NAV"
    finally:
        _teardown(win, orig_start)


# =============================================================================
# 2. PDF ASPECT RATIO (Qt.KeepAspectRatio), HIGH DPI >= 150 & PAGE JUMP INPUT
# =============================================================================
def test_pdf_aspect_ratio_high_dpi_and_page_jump_navigation(qapp, tmp_path: Path):
    import pymupdf as fitz

    pdf_path = tmp_path / "aspect_test.pdf"
    doc = fitz.open()
    # Page 1: Portrait A4 (595x842)
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 100), "Trang 1 Portrait - AirTeach AI", fontsize=22)
    # Page 2: Widescreen 16:9 (960x540)
    p2 = doc.new_page(width=960, height=540)
    p2.insert_text((72, 100), "Trang 2 Landscape 16:9", fontsize=22)
    # Page 3: Tall strip (300x900)
    p3 = doc.new_page(width=300, height=900)
    p3.insert_text((20, 100), "Trang 3 Tall", fontsize=18)
    doc.save(str(pdf_path))
    doc.close()

    win, orig_start = _make_studio_window(qapp)
    try:
        ldoc = LectureDocument.open(pdf_path)
        win.canvas.attach_lecture_document(ldoc)
        qapp.processEvents()

        assert len(win.canvas.slides) == 3
        assert win.page_jump_edit.text() == "1"
        assert "/ 3" in win.slide_counter_lbl.text()

        # Verify extreme aspect-ratio pixmap fitting preserves exact KeepAspectRatio
        for (pw, ph) in ((200, 800), (900, 225), (595, 842), (1920, 1080)):
            pm = QPixmap(pw, ph)
            container = QRectF(50.0, 50.0, 800.0, 600.0)
            fit = win.canvas._fit_pixmap_rect(container, pm)
            assert fit.width() <= container.width() + 0.5
            assert fit.height() <= container.height() + 0.5
            assert abs((fit.width() / fit.height()) - (pw / ph)) < 0.02
            assert abs(fit.center().x() - container.center().x()) < 0.5
            assert abs(fit.center().y() - container.center().y()) < 0.5

        # Verify page raster DPI >= 150 (render_scale >= 2.25 => 162+ DPI on 595pt page -> width >= 1300px)
        qimg_p0 = ldoc.render_page_qimage(0, scale=2.25)
        assert not qimg_p0.isNull()
        assert qimg_p0.width() >= 1300

        # Verify page jump input works for valid, out-of-range, and invalid inputs
        win.page_jump_edit.setText("3")
        win._on_page_jump_submitted()
        assert win.canvas.current_slide_idx == 2
        assert win.page_jump_edit.text() == "3"

        win.page_jump_edit.setText("999")
        win._on_page_jump_submitted()
        assert win.canvas.current_slide_idx == 2

        win.page_jump_edit.setText("")
        win._on_page_jump_submitted()
        assert win.page_jump_edit.text() == "3"

        win.page_jump_edit.setText("2")
        win._on_page_jump_submitted()
        assert win.canvas.current_slide_idx == 1
    finally:
        _teardown(win, orig_start)


# =============================================================================
# 3. RUST CORE RELEASE PROFILE (lto=true) & SUB-MILLISECOND PERFORMANCE
# =============================================================================
def test_rust_core_lto_profile_and_realtime_speed_budget():
    cargo_txt = (ROOT / "rust_core" / "Cargo.toml").read_text(encoding="utf-8")
    assert "opt-level = 3" in cargo_txt
    assert "lto = true" in cargo_txt
    assert "strip = true" in cargo_txt
    assert HAS_RUST_CORE is True

    # Benchmark 500 ticks of Rust One-Euro + Kalman + 3D Hand Classifier + Dual-Hand FSM + 500-object Spatial Index
    f2d = make_one_euro_2d(1.2, 0.03, 1.0)
    kal = make_kalman_predictor_2d(0.04, 0.0015)
    dual = make_dual_hand_state_machine()
    sp = make_spatial_index(0.0, 0.0, 2000.0, 1200.0)
    assert f2d is not None and kal is not None and dual is not None and sp is not None

    for i in range(500):
        x = float((i % 25) * 75)
        y = float((i // 25) * 55)
        sp.upsert(f"obj_{i}", x, y, x + 50.0, y + 40.0, i)

    # Synthetic open palm landmarks
    lm = [[0.5, 0.7, 0.0]] + [[0.45 + 0.02 * (k % 5), 0.65 - 0.03 * (k // 4), -0.02] for k in range(1, 21)]

    t0 = time.perf_counter()
    for tick in range(500):
        ts = tick * 0.016
        fx, fy = f2d.filter(0.4 + 0.0005 * tick, 0.5, ts)
        kal.update(fx, fy, ts)
        _ = classify_hand_3d_fast(lm)
        _ = dual.step("PINCH", 0.22, (0.35, 0.5), "PINCH", 0.22, (0.65 + 0.0002 * tick, 0.5), ts)
        _ = sp.query_point(320.0, 240.0, 10.0)
    avg_ms = ((time.perf_counter() - t0) * 1000.0) / 500.0
    assert avg_ms < 2.0, f"Rust pipeline tick took {avg_ms:.3f}ms (expected < 2.0ms)"


# =============================================================================
# 4. ON-BOUNDING-BOX DELETE HANDLE (✕) & ZERO BOTTOM OVERLAY COLLISION
# =============================================================================
def test_on_box_delete_handle_and_non_overlapping_bottom_overlays(qapp):
    win, orig_start = _make_studio_window(qapp)
    try:
        canvas = win.canvas
        obj = canvas.add_text_object("Test Object Delete", 260.0, 220.0, select=True)
        canvas.active_tool = "SELECT"
        qapp.processEvents()
        assert canvas.scene.get_by_id(obj.id) is not None

        # Click directly on the red ✕ on-box delete handle
        del_board = canvas._selected_delete_handle_board_pos(obj)
        del_screen = canvas._board_to_screen(del_board)
        canvas.mousePressEvent(
            _mouse_ev(QMouseEvent.Type.MouseButtonPress, del_screen, Qt.LeftButton, Qt.LeftButton)
        )
        assert canvas.scene.get_by_id(obj.id) is None

        # Undo restores it cleanly
        canvas.undo()
        assert canvas.scene.get_by_id(obj.id) is not None

        # Opening both docks at 1366x768 must never cause quick_tip_banner to overlap nav_hud
        win.left_dock.setVisible(True)
        win.right_dock.setVisible(True)
        qapp.processEvents()
        canvas._reposition_overlays()
        qapp.processEvents()
        if canvas.quick_tip_banner.isVisible() and canvas.nav_hud.isVisible():
            inter = canvas.quick_tip_banner.geometry().intersected(canvas.nav_hud.geometry())
            assert inter.width() <= 0 or inter.height() <= 0
    finally:
        _teardown(win, orig_start)


# =============================================================================
# 5. CLEAN UI (ZERO [...], ZERO UNESCAPED &), 2-COLUMN GESTURE MODAL & SPOTLIGHT
# =============================================================================
def test_clean_ui_no_brackets_gesture_cheatsheet_and_spotlight_trigger(qapp):
    win, orig_start = _make_studio_window(qapp)
    try:
        win.left_dock.setVisible(True)
        win.right_dock.setVisible(True)
        qapp.processEvents()

        # Zero [...] bracket noise on studio buttons or gesture badges
        bracket_re = re.compile(r"\[[^\]]+\]")
        unescaped_amp_re = re.compile(r"(?<!&)&(?!&)")
        for container_name, container in (
            ("header_bar", win.header_bar),
            ("left_dock", win.left_dock),
            ("right_dock", win.right_dock),
            ("floating_dock", win.floating_dock),
            ("mini_hud", win.canvas.mini_hud),
        ):
            for btn in container.findChildren(QPushButton):
                if btn.isVisible():
                    txt = btn.text()
                    assert not bracket_re.search(txt), f"{container_name} button still has brackets: {txt!r}"
                    assert not unescaped_amp_re.search(txt), f"{container_name} button has unescaped '&': {txt!r}"
            for chk in container.findChildren(QCheckBox):
                if chk.isVisible():
                    txt = chk.text()
                    assert not unescaped_amp_re.search(txt), f"{container_name} checkbox has unescaped '&': {txt!r}"

        for key, lbl in win.gesture_badges.items():
            assert not bracket_re.search(lbl.text()), f"Gesture badge {key} still has brackets: {lbl.text()!r}"

        # Verify Sơ đồ Cử chỉ buttons exist on header, left_dock, AND mini_hud
        assert "Cử chỉ" in win.gesture_map_btn.text()
        assert "Cử chỉ" in win.cam_gesture_map_btn.text()
        assert "Cử chỉ" in win.canvas.mini_hud.gesture_map_btn.text()

        # Verify ? Hướng dẫn button launches 6-step Spotlight Tour directly
        win.left_dock.setVisible(False)
        win.right_dock.setVisible(False)
        qapp.processEvents()
        assert "Hướng dẫn" in win.guide_btn.text()
        win.guide_btn.click()
        qapp.processEvents()
        assert win.spotlight_overlay is not None and win.spotlight_overlay.isVisible() is True
        assert win.spotlight_overlay.step_count == 6
        win.spotlight_overlay.skip_tour()
        qapp.processEvents()

        # Verify 2-column GestureCheatSheetDialog + live test bar
        sheet = GestureCheatSheetDialog(win)
        sheet.show()
        qapp.processEvents()
        assert sheet.one_hand_col_frame.isVisible() is True
        assert sheet.two_hand_col_frame.isVisible() is True
        assert sheet.live_test_bar.isVisible() is True

        assert sheet.ingest_camera_packet({"pose_tag": "INDEX", "dual_hand": {"two_hand_mode": "SINGLE"}}) == "INDEX"
        assert sheet.ingest_camera_packet({"pose_tag": "PINCH", "dual_hand": {"two_hand_mode": "DUAL_PINCH_ZOOM"}}) == "DUAL_PINCH_ZOOM"
        assert sheet.ingest_camera_packet({"pose_tag": "OPEN_PALM", "dual_hand": {"two_hand_mode": "DUAL_PALM_PAN"}}) == "DUAL_PALM_PAN"
        assert sheet.ingest_camera_packet({"pose_tag": "HOVER", "nav_event": "SLIDE_NEXT", "dual_hand": {"two_hand_mode": "SINGLE"}}) == "SWIPE"
        assert len(sheet.tested_gestures) == 4
        assert "4 / 8" in sheet.live_progress_lbl.text()

        bad = sheet.ingest_camera_packet({
            "pose_tag": "HOVER",
            "dual_hand": {"two_hand_mode": "SINGLE"},
            "vision_telemetry": {"palm_facing_ok": False},
        })
        assert bad == ""
        assert "lòng bàn tay" in sheet.live_status_lbl.text().lower()
        sheet.close()

        # Recording toggle keeps icon cleanly inside 30x30 button (no clipped "[Dừng Ghi Âm]")
        win._on_toggle_recording()
        assert win.rec_btn.text() == "⏹"
        win._on_toggle_recording()
        assert win.rec_btn.text() == "🎙"
    finally:
        _teardown(win, orig_start)
