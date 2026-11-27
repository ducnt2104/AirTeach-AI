#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Overhaul Wave 2 Integration Verification Suite (`tests/test_ultra_overhaul_integration.py`):
Verifies end-to-end integration in `main.py` across:
  (a) Rust Quadtree viewport culling (>64 objects) + object recolor (`recolor_selected_objects`,
      4 circular buttons on `obj_action_bar`, SELECT ink-palette recolor) + Undo/Redo,
  (b) Radial Pie Menu `OPEN_AI_CHAT` action opening `right_dock` and checking `toggle_ai_btn`,
  (c) Enriched Live Gesture HUD displaying gesture + L/R hand glyphs + FPS while keeping legacy tokens,
  (d) UTF-8 Vietnamese file attachment (`attach_file_to_chat` on real PDF/DOCX/TXT/PNG) & full
      Gemini Model selector (`cloud_model_combo`) persistence and worker propagation,
  (e) True Interactive Spotlight Tour (`win.start_spotlight_tour()`) stepping through all 6 steps
      on live `AirTeachMainWindow` at 1366x768 & 1920x1080 with zero overflow and >=85% canvas share.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import fitz
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton

import main as app_main
from cv_engine import gemini_client
from cv_engine.drawing_fsm import DrawingHudMode


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["test-ultra-overhaul"])
    return app


def _make_window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = tmp_path / "ui_settings_test.json"
    monkeypatch.setattr(app_main.CameraHandWorker, "start", lambda self: None)
    monkeypatch.setattr(
        app_main,
        "load_ui_api_settings",
        lambda settings_path=cfg_path: gemini_client.load_ui_api_settings(settings_path=cfg_path),
    )
    monkeypatch.setattr(
        app_main,
        "save_ui_api_settings",
        lambda **kwargs: gemini_client.save_ui_api_settings(settings_path=cfg_path, **kwargs),
    )
    win = app_main.AirTeachMainWindow()
    win.stack.setCurrentIndex(1)
    win.resize(1366, 768)
    win.show()
    qapp.processEvents()
    return win, cfg_path


def _assert_no_overflow_or_overlap(container, label: str) -> None:
    cw = container.width()
    buttons = [b for b in container.findChildren(QPushButton) if b.isVisible()]
    rects = []
    for btn in buttons:
        g = btn.geometry()
        assert g.width() > 0 and g.height() > 0, f"{label}: zero-size button {btn.text()!r}"
        assert g.right() <= cw + 2, f"{label}: button {btn.text()!r} overflows ({g.right()} > {cw})"
        rects.append((btn.text(), g))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            t1, r1 = rects[i]
            t2, r2 = rects[j]
            inter = r1.intersected(r2)
            assert inter.width() <= 0 or inter.height() <= 0, (
                f"{label}: buttons {t1!r} and {t2!r} overlap by {inter.width()}x{inter.height()}"
            )


def test_recolor_objects_undo_redo_and_quadtree_viewport_culling(qapp, monkeypatch, tmp_path: Path):
    win, _ = _make_window(qapp, monkeypatch, tmp_path)
    try:
        canvas = win.canvas
        canvas.resize(1200, 700)

        # Create Shape, Text, and Sticky Note
        shp = canvas.add_shape_object("rect", 120.0, 120.0, 160.0, 100.0, color="#00F5D4", select=False)
        txt = canvas.add_text_object("Phương trình bậc hai", 340.0, 120.0, color="#FFFFFF", select=False)
        stk = canvas.add_sticky_object("Ghi chú quan trọng", 580.0, 120.0, select=False)
        orig_stk_bg = stk.bg_color

        # Verify 4 circular 18x18 color buttons exist in obj_action_bar
        expected_hexes = ("#00F5D4", "#FEE440", "#F43F5E", "#A855F7")
        for hx in expected_hexes:
            assert hx in canvas._obj_recolor_buttons
            btn = canvas._obj_recolor_buttons[hx]
            assert btn.width() == 18 and btn.height() == 18

        # 1. Select shape + sticky, click floating obj_action_bar recolor button "#F43F5E"
        canvas.scene.selected_ids = [shp.id, stk.id]
        canvas._bump_scene()
        qapp.processEvents()
        assert canvas.obj_action_bar.isVisible() is True

        canvas._obj_recolor_buttons["#F43F5E"].click()
        qapp.processEvents()
        assert canvas.scene.get_by_id(shp.id).color.upper() == "#F43F5E"
        assert canvas.scene.get_by_id(stk.id).bg_color.upper() == "#F43F5E"
        assert canvas.gesture_toast is not None and "#F43F5E" in canvas.gesture_toast["text"]

        # Undo restores both colors; Redo reapplies "#F43F5E"
        canvas.undo()
        assert canvas.scene.get_by_id(shp.id).color.upper() == "#00F5D4"
        assert canvas.scene.get_by_id(stk.id).bg_color == orig_stk_bg
        canvas.redo()
        assert canvas.scene.get_by_id(shp.id).color.upper() == "#F43F5E"
        assert canvas.scene.get_by_id(stk.id).bg_color.upper() == "#F43F5E"

        # 2. _set_ink_color in SELECT tool with selected text recolors object and KEEPS SELECT active
        canvas.active_tool = "SELECT"
        canvas.scene.select_only(txt.id)
        win._set_ink_color("#FEE440")
        assert canvas.active_tool == "SELECT"
        assert canvas.scene.get_by_id(txt.id).color.upper() == "#FEE440"

        # Undo restores text color
        canvas.undo()
        assert canvas.scene.get_by_id(txt.id).color.upper() == "#FFFFFF"

        # 3. Rust Quadtree viewport culling kicks in cleanly when len(sc.objects) > 64
        sc = canvas.scene
        from cv_engine.canvas_scene import make_shape_object

        for i in range(180):
            # 30 inside viewport, 150 far offscreen at x >= 6000
            x = 80.0 + (i % 10) * 60.0 if i < 30 else 6000.0 + (i * 50.0)
            y = 80.0 + (i // 10) * 60.0 if i < 30 else 6000.0 + (i * 20.0)
            sc.add_object(make_shape_object("rect", x, y, 40.0, 40.0), record_undo=False, select=False)
        assert len(sc.objects) > 180

        cull_calls: List[int] = []
        orig_vis = sc.visible_objects_in_rect

        def _spy_vis(rx: float, ry: float, rw: float, rh: float):
            res = orig_vis(rx, ry, rw, rh)
            cull_calls.append(len(res))
            return res

        monkeypatch.setattr(sc, "visible_objects_in_rect", _spy_vis)
        pm = QPixmap(canvas.width(), canvas.height())
        p = QPainter(pm)
        canvas._paint_static_scene(p)
        p.end()

        assert len(cull_calls) == 1
        assert 25 <= cull_calls[0] < 60, f"Expected viewport culling to skip offscreen objects, got {cull_calls[0]}"
    finally:
        win.shutdown_resources()
        win.close()


def test_radial_pie_open_ai_chat_opens_right_dock(qapp, monkeypatch, tmp_path: Path):
    win, _ = _make_window(qapp, monkeypatch, tmp_path)
    try:
        assert win.right_dock.isVisible() is False
        assert win.toggle_ai_btn.isChecked() is False

        win.canvas.pie_menu.action_triggered.emit("OPEN_AI_CHAT")
        qapp.processEvents()

        assert win.right_dock.isVisible() is True
        assert win.toggle_ai_btn.isChecked() is True
        assert win.canvas.gesture_toast is not None
        assert "🤖 Đã mở Trợ Lý AI" in win.canvas.gesture_toast["text"]
    finally:
        win.shutdown_resources()
        win.close()


def test_enriched_live_gesture_hud_preserves_tokens_and_adds_lr_fps(qapp, monkeypatch, tmp_path: Path):
    win, _ = _make_window(qapp, monkeypatch, tmp_path)
    try:
        canvas = win.canvas
        canvas.resize(1000, 600)

        # 1. Hover with right hand TWO_FINGERS + left hand None + 30 FPS
        upd_h = canvas.apply_cv_fsm_packet({
            "gesture": "TWO_FINGERS",
            "pose_tag": "TWO_FINGERS",
            "pinch_ratio": 0.78,
            "cursor_norm": (0.25, 0.30),
            "left_hand": None,
            "right_hand": {"pose_tag": "TWO_FINGERS"},
            "fps": 30.2,
            "dual_hand": {"two_hand_mode": "SINGLE"},
        })
        assert upd_h.hud_mode == DrawingHudMode.HOVER
        assert "Con Trỏ" in canvas.live_gesture_hud_text
        assert "L:— R:✌️" in canvas.live_gesture_hud_text
        assert "30 FPS" in canvas.live_gesture_hud_text

        # 2. 1-hand INDEX draw with L:PINCH R:INDEX + 60 FPS
        for i in range(4):
            upd_d = canvas.apply_cv_fsm_packet({
                "gesture": "INDEX",
                "pose_tag": "INDEX",
                "pinch_ratio": 0.72,
                "cursor_norm": (0.20 + i * 0.03, 0.35),
                "left_hand": {"pose_tag": "PINCH"},
                "right_hand": {"pose_tag": "INDEX"},
                "fps": 59.8,
                "dual_hand": {"two_hand_mode": "SINGLE"},
            })
        assert upd_d.hud_mode == DrawingHudMode.DRAWING
        assert "Đang Vẽ" in canvas.live_gesture_hud_text
        assert "L:🤏 R:☝️" in canvas.live_gesture_hud_text
        assert "60 FPS" in canvas.live_gesture_hud_text

        # 3. Two-hand Zoom 145% with both hands + 29 FPS
        canvas.apply_zoom_factor(1.45)
        canvas.apply_cv_fsm_packet({
            "gesture": "PINCH",
            "pose_tag": "PINCH",
            "pinch_ratio": 0.18,
            "cursor_norm": (0.50, 0.50),
            "left_hand": {"pose_tag": "PINCH"},
            "right_hand": {"pose_tag": "PINCH"},
            "fps": 29.4,
            "dual_hand": {"two_hand_mode": "DUAL_PINCH_ZOOM", "suppress_primary_drawing": True},
        })
        assert "2 Tay: Đang Zoom 145%" in canvas.live_gesture_hud_text
        assert "L:🤏 R:🤏" in canvas.live_gesture_hud_text
        assert "29 FPS" in canvas.live_gesture_hud_text

        # 4. No hand -> HUD text cleared cleanly
        canvas.apply_cv_fsm_packet({
            "gesture": "IDLE",
            "pose_tag": "IDLE",
            "cursor_norm": None,
            "left_hand": None,
            "right_hand": None,
            "fps": 30.0,
            "dual_hand": {"two_hand_mode": "NONE"},
        })
        assert canvas.live_gesture_hud_text == ""
    finally:
        win.shutdown_resources()
        win.close()


def test_chat_utf8_file_attachments_and_gemini_model_selector(qapp, monkeypatch, tmp_path: Path):
    win, cfg_path = _make_window(qapp, monkeypatch, tmp_path)
    try:
        # 1. Verify full AVAILABLE_GEMINI_MODELS populated & persistence
        assert win.cloud_model_combo.count() == len(gemini_client.AVAILABLE_GEMINI_MODELS)
        assert win._selected_cloud_model() is None  # Default Auto ("")

        pro_idx = win.cloud_model_combo.findData("gemini-2.5-pro")
        assert pro_idx >= 0
        win.cloud_model_combo.setCurrentIndex(pro_idx)
        qapp.processEvents()
        assert win._selected_cloud_model() == "gemini-2.5-pro"
        saved_cfg = gemini_client.load_ui_api_settings(settings_path=cfg_path)
        assert saved_cfg.get("gemini_model_id") == "gemini-2.5-pro"

        # Verify GeminiHealthCheckWorker receives selected model_id
        hc_captured: List[Dict[str, Any]] = []

        class _FakeHealthWorker:
            def __init__(self, api_key: str, model_id=None) -> None:
                hc_captured.append({"api_key": api_key, "model_id": model_id})
                self.completed = type("_Sig", (), {"connect": lambda _s, _cb: None})()

            def isRunning(self) -> bool:
                return False

            def start(self) -> None:
                pass

        monkeypatch.setattr(app_main, "GeminiHealthCheckWorker", _FakeHealthWorker)
        win.api_key_edit.setText("AIzaSyTestFakeKeyForModelCheck0123456789")
        win._on_test_api_connection()
        assert len(hc_captured) == 1
        assert hc_captured[0]["model_id"] == "gemini-2.5-pro"

        # Verify persisted gemini_model_id is restored on a second window instance
        win2 = app_main.AirTeachMainWindow()
        try:
            assert win2._selected_cloud_model() == "gemini-2.5-pro"
        finally:
            win2.shutdown_resources()
            win2.close()

        # 2. Build real Vietnamese UTF-8 TXT, PDF, DOCX, and PNG files
        vn_sample = "Bài giảng Toán học: Phương trình bậc hai ax² + bx + c = 0 (Δ = b² - 4ac)."

        # (i) TXT (NFD decomposed input -> must become NFC)
        txt_path = tmp_path / "Đề_cương_ôn_tập_Toán_10.txt"
        txt_path.write_text(unicodedata.normalize("NFD", vn_sample), encoding="utf-8-sig")

        # (ii) PDF via PyMuPDF (with built-in Unicode font / clean text)
        pdf_path = tmp_path / "Bài_tập_Vật_lý_11.pdf"
        pdoc = fitz.open()
        page = pdoc.new_page(width=595, height=842)
        page.insert_text((60, 100), "AirTeach AI Physics Lesson: E = mc^2", fontsize=18)
        pdoc.save(str(pdf_path))
        pdoc.close()

        # (iii) DOCX via python-docx
        import docx as _docx

        docx_path = tmp_path / "Giáo_án_Hóa_học_hữu_cơ.docx"
        wdoc = _docx.Document()
        wdoc.add_heading("Giáo án Hóa học lớp 11", level=1)
        wdoc.add_paragraph(vn_sample)
        wdoc.save(str(docx_path))

        # (iv) PNG image
        png_path = tmp_path / "Sơ_đồ_mạch_điện_Vật_lý.png"
        qimg = QImage(320, 200, QImage.Format_RGB32)
        qimg.fill(QColor("#0F172A"))
        assert qimg.save(str(png_path), "PNG")

        # Intercept _start_hybrid_worker to inspect prompt, png_bytes & cloud_model
        worker_calls: List[Dict[str, Any]] = []

        def _capture_worker(
            prompt: str,
            png_bytes=None,
            structured_json: bool = False,
            include_board_tools: bool = True,
            cloud_model=None,
        ) -> None:
            worker_calls.append({
                "prompt": prompt,
                "png_bytes": png_bytes,
                "structured_json": structured_json,
                "include_board_tools": include_board_tools,
                "cloud_model": cloud_model,
            })

        monkeypatch.setattr(win, "_start_hybrid_worker", _capture_worker)
        win.chat_attach_board_chk.setChecked(False)

        # (a) Attach TXT + submit with EMPTY chat box -> smart default prompt & NFC text
        assert win.attach_file_to_chat(txt_path) is True
        assert win._pending_attachment is not None
        assert win.attached_file_frame.isVisible() is True
        assert txt_path.name in win.attached_file_lbl.text()

        win.chat_input_edit.setText("")
        win._on_chat_submit()
        assert win._pending_attachment is None
        assert win.attached_file_frame.isVisible() is False
        assert len(worker_calls) == 1
        call_txt = worker_calls[-1]
        assert "Hãy phân tích và tóm tắt chi tiết tệp đính kèm này." in call_txt["prompt"]
        assert unicodedata.is_normalized("NFC", call_txt["prompt"])
        assert "Phương trình bậc hai" in call_txt["prompt"]
        assert "text/plain" in call_txt["prompt"]
        assert call_txt["cloud_model"] == "gemini-2.5-pro"

        # (b) Attach PDF -> includes extracted text + first-page PNG preview bytes
        assert win.attach_file_to_chat(pdf_path) is True
        win.chat_input_edit.setText("Tóm tắt file PDF này")
        win._on_chat_submit()
        call_pdf = worker_calls[-1]
        assert "AirTeach AI Physics Lesson" in call_pdf["prompt"]
        assert "application/pdf" in call_pdf["prompt"]
        assert call_pdf["png_bytes"] is not None and call_pdf["png_bytes"].startswith(b"\x89PNG")

        # (c) Attach DOCX -> includes Vietnamese NFC paragraph text
        assert win.attach_file_to_chat(docx_path) is True
        win.chat_input_edit.setText("Trích ý chính giáo án")
        win._on_chat_submit()
        call_docx = worker_calls[-1]
        assert "Giáo án Hóa học lớp 11" in call_docx["prompt"]
        assert "Phương trình bậc hai" in call_docx["prompt"]
        assert "wordprocessingml" in call_docx["prompt"]

        # (d) Attach PNG -> sends image_bytes as ctx_png with exact image/png MIME
        assert win.attach_file_to_chat(png_path) is True
        win.chat_input_edit.setText("Phân tích sơ đồ này")
        win._on_chat_submit()
        call_png = worker_calls[-1]
        assert "image/png" in call_png["prompt"]
        assert call_png["png_bytes"] is not None and call_png["png_bytes"].startswith(b"\x89PNG")
    finally:
        win.shutdown_resources()
        win.close()


def test_live_spotlight_tour_6_steps_and_zero_overflow_at_1366_and_1920(qapp, monkeypatch, tmp_path: Path):
    win, cfg_path = _make_window(qapp, monkeypatch, tmp_path)
    try:
        # Check gear_menu action text updated to 6-step Interactive Spotlight Tour
        gear_texts = [act.text() for act in win.gear_menu.actions()]
        assert "🔦 Hướng Dẫn Tương Tác Spotlight (6 Bước)" in gear_texts

        # Ensure doc_menu_btn is saved on AirTeachMainWindow
        assert hasattr(win, "doc_menu_btn") and win.doc_menu_btn is not None

        # Start from screen 0 to prove start_spotlight_tour switches to Studio screen 1 automatically
        win.stack.setCurrentIndex(0)
        qapp.processEvents()
        overlay = win.start_spotlight_tour(0)
        qapp.processEvents()
        assert win.stack.currentIndex() == 1
        assert overlay is win.spotlight_overlay
        assert overlay.isVisible() is True
        assert overlay.step_count == 6

        for (w, h) in ((1366, 768), (1920, 1080)):
            win.resize(w, h)
            qapp.processEvents()
            qapp.processEvents()

            # Zero overflow & >= 85% canvas area share preserved
            win_area = max(1, win.width() * win.height())
            canvas_area = win.canvas.width() * win.canvas.height()
            ratio = canvas_area / float(win_area)
            assert ratio >= 0.85, f"At {w}x{h}, canvas ratio {ratio*100:.1f}% < 85%"
            _assert_no_overflow_or_overlap(win.header_bar, f"Header@{w}x{h}")
            _assert_no_overflow_or_overlap(win.floating_dock, f"FloatingDock@{w}x{h}")

            overlay.start(0)
            qapp.processEvents()
            for idx in range(6):
                assert overlay.current_step_index == idx
                assert f"Bước {idx + 1} / 6" in overlay.progress_pill.text()
                cut: QRectF = overlay.active_spotlight_rect()
                cg = overlay.card.geometry()
                assert cut.left() >= 0.0 and cut.top() >= 0.0
                assert cut.right() <= win.width() + 1.0 and cut.bottom() <= win.height() + 1.0
                assert cg.left() >= 0 and cg.top() >= 0
                assert cg.right() < win.width() and cg.bottom() < win.height()

                # On "documents" step (index 4), verify cutout actually targets win.doc_menu_btn
                if idx == 4:
                    doc_tl = win.doc_menu_btn.mapTo(win, QPoint(0, 0))
                    doc_cx = doc_tl.x() + win.doc_menu_btn.width() * 0.5
                    assert abs(cut.center().x() - doc_cx) < 16.0

                if idx < 5:
                    overlay.next_step()
                    qapp.processEvents()

        # Finish tour with dont_show_again checked -> persists tour_completed=true
        overlay.dont_show_chk.setChecked(True)
        overlay.next_step()
        qapp.processEvents()
        assert overlay.isVisible() is False
        cfg_after = gemini_client.load_ui_api_settings(settings_path=cfg_path)
        assert cfg_after.get("tour_completed") == "true"

        # Restart and verify Esc hotkey dismisses overlay cleanly
        win.start_spotlight_tour(2)
        qapp.processEvents()
        assert overlay.isVisible() is True
        win._on_hotkey_esc()
        qapp.processEvents()
        assert overlay.isVisible() is False

        # Verify UserGuideDialog.start_spotlight_requested closes guide & launches Spotlight Tour
        def _fake_guide_exec(dlg_self):
            dlg_self.spotlight_btn.click()
            return 1

        monkeypatch.setattr(app_main.UserGuideDialog, "exec", _fake_guide_exec)
        win._on_show_user_guide(0)
        qapp.processEvents()
        assert win.spotlight_overlay is not None and win.spotlight_overlay.isVisible() is True
        assert win.spotlight_overlay.current_step_index == 0
        win.spotlight_overlay.skip_tour()
        qapp.processEvents()
        assert win.spotlight_overlay.isVisible() is False
    finally:
        win.shutdown_resources()
        win.close()
