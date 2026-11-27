#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive End-to-End Architectural Verification Suite (`tests/verify_all_systems.py`)
for AirTeach AI (KHKT National Live Demo Release):
  1. Rust Native Extension (`airteach_core_rs`) import & < 1.0 ms/frame kinematics benchmark
  2. Cold Window Startup (< 1.5s) & Uncluttered Canvas Viewport Ratio (> 80%)
  3. Universal Real Document & Image Rendering (.pdf, .pptx, .docx, .png, .svg, folder deck,
     movable/resizable board images, and annotated PDF export)
  4. Right-Click Radial Pie Menu (8 angles, 20px deadzone cancel, SmartboardCanvas Qt mouse flow)
  5. Board Navigation ([0.25, 4.0] zoom clamp, Shift+Wheel H-pan, Fit Width/Page, LOD zero re-rasterize on pan)
  6. AI Chatbot Rich Markdown/Table/LaTeX + Local AI GGUF Resumable Range Downloader & SHA-256 check

Run directly (`python tests/verify_all_systems.py`) or via `pytest`.
"""

from __future__ import annotations

from functools import partial
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AIRTEACH_NO_MODAL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(["airteach-verify"])
    return app


def test_rust_core_import_and_1khz_benchmark() -> None:
    """Verify native Rust `airteach_core_rs` imports & runs < 1.0 ms per frame."""
    from cv_engine import rust_bridge
    from cv_engine.rust_bridge import (
        HAS_RUST_CORE,
        RUST_CORE_VERSION,
        analyze_frame_luma_fast,
        apply_gamma_contrast_fast,
        catmull_rom_spline_fast,
        classify_hand_3d_fast,
        compute_dual_hand_kinematics_fast,
        compute_roi_crop_box_fast,
        make_kinematic_60hz_extrapolator,
        make_one_euro_2d,
        make_swipe_detector,
        rdp_simplify_fast,
        snap_geometric_shape_fast,
    )
    from cv_engine.shape_recognizer import beautify_stroke

    assert HAS_RUST_CORE is True, "airteach_core_rs native extension must be compiled and installed!"
    assert "airteach_core_rs" in RUST_CORE_VERSION
    assert rust_bridge._rs is not None

    # Realistic 21 3D hand landmarks
    rng = np.random.default_rng(42)
    pts_rel = [(0.0, 0.0, 0.0)]
    for i in range(1, 21):
        pts_rel.append((0.02 * (i % 5 - 2), -0.04 * (1 + i // 5), -0.005 * i))
    pts_a = np.array(pts_rel, dtype=np.float64) + 0.5
    pts_list_a = [(float(p[0]), float(p[1]), float(p[2])) for p in pts_a]

    flt = make_one_euro_2d(0.8, 0.04, 1.0)
    extrap = make_kinematic_60hz_extrapolator(0.8, 0.04, 1.0, 0.16)
    swipe = make_swipe_detector()
    assert flt is not None and extrap is not None and swipe is not None

    # Warmup
    for i in range(25):
        t = i * 0.016
        classify_hand_3d_fast(pts_list_a, None, True, 0.28, 0.42, "right", 0.20)
        flt.filter(0.4 + 0.001 * i, 0.5, t)
        extrap.ingest_vision((0.4, 0.5), t)
        extrap.sample_60hz(t + 0.008)
        compute_dual_hand_kinematics_fast((0.4, 0.5), (0.6, 0.5), 0.30, (0.50, 0.50))
        swipe.push_and_check(0.4 + 0.005 * i, 0.5, t, "POINT")

    iters = 1000
    t0 = time.perf_counter()
    for i in range(iters):
        t = 1.0 + i * 0.016
        cls_res = classify_hand_3d_fast(pts_list_a, None, True, 0.28, 0.42, "right", 0.20)
        fx, fy = flt.filter(0.42 + 0.0005 * (i % 40), 0.51, t)
        extrap.ingest_vision((fx, fy), t)
        _pt_out, _ex = extrap.sample_60hz(t + 0.008)
        _dual = compute_dual_hand_kinematics_fast((fx, fy), (0.72, 0.50), 0.31, (0.55, 0.50))
        _sw = swipe.push_and_check(fx, fy, t, "POINT")
        _roi = compute_roi_crop_box_fast(fx - 0.08, fy - 0.08, fx + 0.08, fy + 0.08, 0.01, 0.0, 2.2, 0.45)
        _ = cls_res
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_ms / iters
    print(f"[BENCH] Rust Core kinematics 1000-frame avg = {avg_ms:.4f} ms/frame ({elapsed_ms:.1f} ms total)", flush=True)
    assert avg_ms < 1.0, f"Rust kinematics too slow: {avg_ms:.4f} ms/frame"

    # Shape + spline + frame luma checks
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    circle_pts = [(float(300.0 + 90.0 * np.cos(a)), float(300.0 + 90.0 * np.sin(a))) for a in theta]
    circle_pts.append(circle_pts[0])
    simplified = rdp_simplify_fast(circle_pts, 2.5)
    assert simplified is not None and 6 <= len(simplified) <= len(circle_pts)
    splined = catmull_rom_spline_fast(simplified, 4)
    assert splined is not None and len(splined) > len(simplified)

    snapped_circle = snap_geometric_shape_fast(circle_pts, 0.18, 0.78)
    assert snapped_circle is not None and snapped_circle[0] == "CIRCLE"

    open_line = beautify_stroke([(10.0, 20.0), (105.0, 21.0), (210.0, 19.5)])
    assert open_line.shape_type == "LINE"

    gray_frame = rng.integers(35, 190, size=(120, 160), dtype=np.uint8)
    luma_stats = analyze_frame_luma_fast(gray_frame, is_rgb=False, stride=2)
    assert luma_stats is not None
    mean_y, std_y, p95_y = luma_stats
    assert 20.0 < mean_y < 230.0 and std_y > 10.0 and p95_y >= mean_y
    rgb_frame = np.stack([gray_frame, gray_frame, gray_frame], axis=-1)
    boosted = apply_gamma_contrast_fast(rgb_frame, gamma=1.15, alpha=1.05, beta=4.0)
    assert boosted is not None and boosted.shape == rgb_frame.shape and boosted.dtype == np.uint8


def test_startup_under_1_5s_and_canvas_share() -> None:
    """Measure cold isolated subprocess startup & verify > 80% canvas viewport area."""
    probe_code = r'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["AIRTEACH_NO_MODAL"] = "1"
t0 = time.perf_counter()
from PySide6.QtWidgets import QApplication
import main
app = QApplication(["startup-bench"])
win = main.AirTeachMainWindow()
win.resize(1400, 820)
win.stack.setCurrentIndex(1)
win.show()
app.processEvents()
elapsed = time.perf_counter() - t0
win_area = max(1, win.width() * win.height())
canvas_area = win.canvas.width() * win.canvas.height()
ratio = canvas_area / win_area
print(f"BENCH_STARTUP={elapsed:.4f} BENCH_CANVAS_RATIO={ratio:.4f}", flush=True)
win.shutdown_resources()
win.close()
app.quit()
'''
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["AIRTEACH_NO_MODAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    line = [ln for ln in proc.stdout.splitlines() if "BENCH_STARTUP=" in ln][-1]
    parts = dict(tok.split("=", 1) for tok in line.strip().split())
    startup_sec = float(parts["BENCH_STARTUP"])
    canvas_ratio = float(parts["BENCH_CANVAS_RATIO"])
    print(f"[BENCH] Cold Window Startup = {startup_sec:.3f}s | Canvas Area Share = {canvas_ratio * 100:.1f}%", flush=True)
    assert startup_sec < 1.5, f"Cold startup exceeded 1.5s budget: {startup_sec:.3f}s"
    assert canvas_ratio > 0.80, f"Canvas area must exceed 80% of window (got {canvas_ratio * 100:.1f}%)"


def _build_real_sample_files(tmp_dir: Path) -> Dict[str, Path]:
    import docx
    from PIL import Image, ImageDraw
    from pptx import Presentation
    from pptx.util import Inches
    import pymupdf as fitz

    out: Dict[str, Path] = {}

    # 1. Real PDF
    pdf_p = tmp_dir / "lesson_sample.pdf"
    pdf_doc = fitz.open()
    pg = pdf_doc.new_page(width=960, height=540)
    pg.draw_rect(fitz.Rect(24, 24, 936, 516), color=(0.05, 0.35, 0.75), fill=(0.94, 0.97, 1.0), width=2)
    pg.draw_circle(fitz.Point(240, 260), 90, color=(0.85, 0.2, 0.3), fill=(0.98, 0.88, 0.90), width=3)
    pg.insert_text((380, 180), "AirTeach AI Physics Demo: F = m * a", fontsize=24, color=(0.06, 0.12, 0.24))
    pg.insert_text((380, 230), "Delta x = v0*t + 0.5*a*t^2", fontsize=18, color=(0.15, 0.25, 0.45))
    pdf_doc.save(str(pdf_p))
    pdf_doc.close()
    out["pdf"] = pdf_p

    # 2. Real PPTX
    pptx_p = tmp_dir / "deck_sample.pptx"
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    tb = sl.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(10.0), Inches(1.4))
    tb.text_frame.text = "Bai giang Toan hoc 12 - Tich phan"
    box = sl.shapes.add_shape(1, Inches(1.2), Inches(2.5), Inches(5.0), Inches(2.6))
    box.text_frame.text = "Int_a^b f(x) dx = F(b) - F(a)"
    prs.save(str(pptx_p))
    out["pptx"] = pptx_p

    # 3. Real DOCX
    docx_p = tmp_dir / "notes_sample.docx"
    wdoc = docx.Document()
    wdoc.add_heading("De cuong On tap Vat Ly", level=1)
    p = wdoc.add_paragraph("Dinh luat Ohm cho doan mach: I = U / R.")
    p.add_run(" Cong suat toa nhiet: P = I^2 * R.").bold = True
    tbl = wdoc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "Dai luong"
    tbl.cell(0, 1).text = "Don vi"
    tbl.cell(1, 0).text = "Cuong do dong dien I"
    tbl.cell(1, 1).text = "Ampe (A)"
    wdoc.save(str(docx_p))
    out["docx"] = docx_p

    # 4. Real PNG
    png_p = tmp_dir / "diagram.png"
    img = Image.new("RGB", (640, 400), (15, 23, 42))
    dr = ImageDraw.Draw(img)
    dr.rectangle([20, 20, 620, 380], outline=(6, 182, 212), width=6)
    dr.ellipse([180, 70, 460, 330], fill=(79, 70, 229), outline=(0, 245, 212), width=5)
    img.save(str(png_p))
    out["png"] = png_p

    # 5. Real SVG
    svg_p = tmp_dir / "circuit.svg"
    svg_p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320" viewBox="0 0 480 320">'
        '<rect x="12" y="12" width="456" height="296" rx="18" fill="#1E293B" stroke="#06B6D4" stroke-width="5"/>'
        '<circle cx="150" cy="160" r="65" fill="#4F46E5" stroke="#00F5D4" stroke-width="4"/>'
        '<path d="M 240 160 L 400 160 L 370 135 M 400 160 L 370 185" stroke="#FEE440" stroke-width="6" fill="none"/>'
        "</svg>",
        encoding="utf-8",
    )
    out["svg"] = svg_p

    # 6. Multi-Image Folder Deck
    folder_p = tmp_dir / "gallery_slides"
    folder_p.mkdir(parents=True, exist_ok=True)
    for idx, bg_rgb in enumerate(((20, 45, 90), (45, 20, 85), (15, 75, 70)), start=1):
        im = Image.new("RGB", (480, 270), bg_rgb)
        d = ImageDraw.Draw(im)
        d.rectangle([25, 25, 455, 245], outline=(255, 255, 255), width=4)
        im.save(str(folder_p / f"slide_{idx}.png"))
    out["folder"] = folder_p

    return out


def _qimage_pixel_std(qimg) -> float:
    rgba = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
    ptr = rgba.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=rgba.width() * rgba.height() * 4).reshape((rgba.height(), rgba.width(), 4))
    return float(np.std(arr[:, :, :3]))


def test_universal_real_document_and_image_rendering() -> None:
    """Verify real rendering of PDF, PPTX, DOCX, PNG, SVG, Image Folder, Movable Image & PDF export."""
    _ensure_qapp()
    import pymupdf as fitz
    import main
    from cv_engine.document_presenter import LectureDocument, export_annotated_lecture_pdf

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        samples = _build_real_sample_files(tmp_path)
        cache_dir = tmp_path / "doc_cache"

        canvas = main.SmartboardCanvas()
        canvas.resize(1200, 720)

        for fmt_key in ("pdf", "pptx", "docx"):
            doc = LectureDocument.open(samples[fmt_key], cache_dir=cache_dir)
            try:
                assert doc.page_count >= 1
                canvas.attach_lecture_document(doc)
                qimg = doc.render_page_qimage(0, scale=2.0)
                std_val = _qimage_pixel_std(qimg)
                assert std_val > 5.0, f"{fmt_key.upper()} rendered blank frame (std={std_val:.2f})"
                snap_bytes, snap_pm = canvas.export_board_png()
                assert len(snap_bytes) > 1000 and not snap_pm.isNull()
            finally:
                doc.close()
                canvas.lecture_doc = None

        # Single PNG & SVG image slide loading
        assert canvas.load_document_file(samples["png"]) is True
        assert canvas.load_document_file(samples["svg"]) is True
        assert _qimage_pixel_std(canvas.slides[canvas.current_slide_idx]["pixmap"].toImage()) > 5.0

        # Multi-image folder slide deck
        assert canvas.load_image_folder(samples["folder"]) is True
        assert len(canvas.slides) == 3
        canvas.set_current_slide(1)

        # Insert movable/resizable image onto canvas + annotate + export PDF
        assert canvas.insert_image_on_board(samples["svg"]) is True
        assert len(canvas.board_images) == 1
        canvas.move_board_image(0, 140.0, 110.0)
        canvas.resize_board_image(0, 320.0, 220.0)
        assert canvas.board_images[0]["x"] == 140.0 and canvas.board_images[0]["w"] == 320.0

        canvas.strokes.append({
            "points": [QPointF(150.0, 150.0), QPointF(300.0, 240.0), QPointF(420.0, 190.0)],
            "color": canvas.ink_color,
            "thickness": 5,
            "is_highlighter": False,
        })

        exported_pdf = export_annotated_lecture_pdf(
            tmp_path / "exported_lecture.pdf",
            canvas.slides,
            canvas.slide_annotations,
            (float(canvas.width()), float(canvas.height())),
            lecture_doc=canvas.lecture_doc,
        )
        assert exported_pdf.exists() and exported_pdf.stat().st_size > 1500
        out_doc = fitz.open(str(exported_pdf))
        try:
            assert out_doc.page_count == len(canvas.slides)
        finally:
            out_doc.close()


def test_radial_pie_menu_angles_deadzone_and_canvas_mouse() -> None:
    """Verify Blender-style Right-Click Radial Pie Menu angle mapping, 20px deadzone & Qt mouse events."""
    _ensure_qapp()
    import main
    from cv_engine.radial_pie_menu import PIE_MENU_SLICES, resolve_pie_slice

    # Deadzone < 20px cancels cleanly
    assert resolve_pie_slice(0.0, 0.0, deadzone_px=20.0) is None
    assert resolve_pie_slice(12.0, -14.0, deadzone_px=20.0) is None

    # All 8 ergonomic directions: N..NW
    expected_dirs = [
        (0.0, -75.0, "PEN"),
        (55.0, -55.0, "HIGHLIGHTER"),
        (75.0, 0.0, "PAN"),
        (55.0, 55.0, "ASK_AI_REGION"),
        (0.0, 75.0, "ERASER"),
        (-55.0, 55.0, "UNDO"),
        (-75.0, 0.0, "LASER"),
        (-55.0, -55.0, "TOGGLE_SMART_SHAPE"),
    ]
    for exp_idx, (dx, dy, exp_action) in enumerate(expected_dirs):
        idx = resolve_pie_slice(dx, dy, deadzone_px=20.0)
        assert idx == exp_idx
        assert PIE_MENU_SLICES[idx].action_id == exp_action

    # Live Qt RightButton Press -> Move -> Release on SmartboardCanvas
    canvas = main.SmartboardCanvas()
    canvas.resize(1000, 700)
    canvas.show()
    center = QPointF(500.0, 350.0)

    def _mouse_ev(etype, pos, btn, btns):
        return QMouseEvent(etype, pos, pos, btn, btns, Qt.NoModifier)

    # 1. Select ERASER (South: dy = +80)
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    assert canvas.pie_menu.isVisible() is True
    canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, QPointF(500.0, 430.0), Qt.NoButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, QPointF(500.0, 430.0), Qt.RightButton, Qt.NoButton))
    assert canvas.pie_menu.isVisible() is False
    assert canvas.active_tool == "ERASER"

    # 2. Deadzone cancel (< 20px move) keeps ERASER unchanged
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, QPointF(506.0, 355.0), Qt.NoButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, QPointF(506.0, 355.0), Qt.RightButton, Qt.NoButton))
    assert canvas.active_tool == "ERASER"

    # 3. Toggle Smart Shape (North-West)
    prev_ss = canvas.smart_shape_enabled
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, QPointF(440.0, 290.0), Qt.RightButton, Qt.NoButton))
    assert canvas.smart_shape_enabled is (not prev_ss)
    canvas.close()


def test_navigation_zoom_range_and_lod_cache() -> None:
    """Verify [25%..400%] zoom clamp, Shift+Wheel H-pan, Fit Width/Page & zero page re-rasterize on pan."""
    _ensure_qapp()
    import pymupdf as fitz
    import main
    from cv_engine.document_presenter import LectureDocument

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        pdf_p = tmp_p / "nav_test.pdf"
        d = fitz.open()
        pg = d.new_page(width=800, height=600)
        pg.insert_text((80, 120), "LOD Pan Cache Test Page", fontsize=24)
        d.save(str(pdf_p))
        d.close()

        canvas = main.SmartboardCanvas()
        canvas.resize(1100, 700)
        canvas.show()
        ldoc = LectureDocument.open(pdf_p, cache_dir=tmp_p / "cache")
        try:
            canvas.attach_lecture_document(ldoc)

            canvas.apply_zoom_factor(10.0)
            assert abs(canvas.zoom_scale - 4.0) < 1e-3
            canvas.apply_zoom_factor(0.01)
            assert abs(canvas.zoom_scale - 0.25) < 1e-3

            canvas.fit_to_width()
            assert 0.25 <= canvas.zoom_scale <= 4.0
            canvas.fit_to_page()
            assert 0.25 <= canvas.zoom_scale <= 4.0
            canvas.reset_viewport()
            assert abs(canvas.zoom_scale - 1.0) < 1e-3 and canvas.pan_x == 0.0 and canvas.pan_y == 0.0

            # Shift+Wheel horizontal scroll
            old_pan_x = canvas.pan_x
            w_ev = QWheelEvent(
                QPointF(500.0, 350.0),
                QPointF(500.0, 350.0),
                QPointF(0, 0).toPoint(),
                QPointF(0, 120).toPoint(),
                Qt.NoButton,
                Qt.ShiftModifier,
                Qt.ScrollUpdate,
                False,
            )
            canvas.wheelEvent(w_ev)
            assert canvas.pan_x != old_pan_x

            # Verify panning uses offscreen cache with zero re-rasterization
            canvas.repaint()
            cache_len_before = len(canvas._scaled_page_pixmap_cache)
            orig_render = ldoc.render_page_qimage
            render_calls = 0

            def _counting_render(*args, **kwargs):
                nonlocal render_calls
                render_calls += 1
                return orig_render(*args, **kwargs)

            ldoc.render_page_qimage = _counting_render  # type: ignore[method-assign]
            canvas.apply_pan_delta_norm(0.12, -0.08)
            canvas.repaint()
            assert render_calls == 0, f"Panning must not re-rasterize document pages (calls={render_calls})"
            assert len(canvas._scaled_page_pixmap_cache) == cache_len_before
        finally:
            ldoc.close()
            canvas.lecture_doc = None
            canvas.close()


class _RangeBinaryHandler(BaseHTTPRequestHandler):
    payload: bytes = b""
    saw_range_206: bool = False
    drop_after_bytes: Optional[int] = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        data = self.payload
        total = len(data)
        range_hdr = self.headers.get("Range")
        if range_hdr and range_hdr.startswith("bytes="):
            start_s = range_hdr[6:].split("-", 1)[0].strip()
            start = int(start_s) if start_s.isdigit() else 0
            if start >= total:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
            self.__class__.saw_range_206 = True
            slice_bytes = data[start:]
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
            self.send_header("Content-Length", str(len(slice_bytes)))
            self.end_headers()
            self.wfile.write(slice_bytes)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.end_headers()
        if self.drop_after_bytes is not None and self.drop_after_bytes < total:
            self.wfile.write(data[: self.drop_after_bytes])
            return
        self.wfile.write(data)


def test_local_ai_manager_and_chatbot_rich_rendering() -> None:
    """Verify max_output_tokens=4096, rich Markdown/Table/LaTeX HTML, catalog & HTTP Range GGUF resume."""
    from cv_engine.gemini_client import DEFAULT_MAX_OUTPUT_TOKENS
    from cv_engine.gguf_model_manager import (
        CURATED_GGUF_CATALOG,
        DownloadProgress,
        ResumableModelDownloader,
        list_downloaded_models,
        normalize_model_download_url,
    )
    from cv_engine.output_sanitizer import OutputSanitizer

    assert DEFAULT_MAX_OUTPUT_TOKENS >= 4096

    # Rich educational Markdown + Pipe Table + List + Formula + Vector Arrows
    sample_md = (
        "## Định Luật II Newton\n"
        "- Gia tốc tỉ lệ thuận với hợp lực $\\vec{F}$ và tỉ lệ nghịch với khối lượng $m$.\n"
        "| Đại lượng | Ký hiệu | Đơn vị |\n"
        "| --- | --- | --- |\n"
        "| Hợp lực | F | Newton (N) |\n"
        "| Khối lượng | m | Kilogam (kg) |\n"
        "Suy ra: a -> F / m\n"
        "$$F = m \\cdot a$$"
    )
    rich_html = OutputSanitizer.to_rich_html(sample_md, dark_mode=True, render_math_images=True)
    plain = OutputSanitizer.sanitize_plain_text(sample_md)
    assert "<table" in rich_html and "<th" in rich_html and "<td" in rich_html
    assert "Định Luật II Newton" in rich_html
    assert "→" in plain and "$" not in plain

    assert len(CURATED_GGUF_CATALOG) >= 5
    for spec in CURATED_GGUF_CATALOG:
        assert spec.download_url.startswith("https://") and spec.filename.endswith(".gguf")

    assert "/resolve/main/" in normalize_model_download_url(
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    assert "/raw/v1.0/" in normalize_model_download_url(
        "https://github.com/org/repo/blob/v1.0/model.gguf"
    )

    # Resumable HTTP Range + SHA-256 verification against live ThreadingHTTPServer
    synthetic_gguf = b"GGUF" + (b"AirTeachAI-LocalModel-Chunk-" * 9200)
    expected_sha = hashlib.sha256(synthetic_gguf).hexdigest().lower()

    handler_cls = type("_ScopedRangeHandler", (_RangeBinaryHandler,), {})
    handler_cls.payload = synthetic_gguf
    handler_cls.saw_range_206 = False
    handler_cls.drop_after_bytes = len(synthetic_gguf) // 2

    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        port = srv.server_address[1]
        url = f"http://127.0.0.1:{port}/test-qwen.gguf"
        downloader = ResumableModelDownloader()

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            # Step 1: Simulate interrupted first download leaving .part
            interrupted = False
            try:
                downloader.download(url=url, dest_dir=models_dir, filename="test-qwen.gguf", verify_gguf=True)
            except IOError:
                interrupted = True
            assert interrupted is True
            part_file = models_dir / "test-qwen.gguf.part"
            assert part_file.exists() and part_file.stat().st_size == handler_cls.drop_after_bytes

            # Step 2: Disable server drop & resume via HTTP 206 Partial Content
            handler_cls.drop_after_bytes = None
            progs: List[DownloadProgress] = []
            final_path, sha_hex = downloader.download(
                url=url,
                dest_dir=models_dir,
                filename="test-qwen.gguf",
                expected_sha256=expected_sha,
                verify_gguf=True,
                chunk_size=16384,
                progress_cb=progs.append,
            )
            assert handler_cls.saw_range_206 is True
            assert any(p.resumed for p in progs)
            assert final_path.exists() and not part_file.exists()
            assert sha_hex == expected_sha
            listed = list_downloaded_models(models_dir)
            assert len(listed) == 1 and listed[0].valid_gguf and listed[0].sha256 == expected_sha
    finally:
        srv.shutdown()
        srv.server_close()


def main() -> int:
    tests = [
        ("Rust Native Core & 1000-Frame Benchmark", test_rust_core_import_and_1khz_benchmark),
        ("Cold Startup (<1.5s) & Canvas Space (>80%)", test_startup_under_1_5s_and_canvas_share),
        ("Universal Real Document & Image Engine", test_universal_real_document_and_image_rendering),
        ("Right-Click Radial Pie Menu (8 Slices + Deadzone)", test_radial_pie_menu_angles_deadzone_and_canvas_mouse),
        ("Board Navigation, Zoom [25%-400%] & LOD Cache", test_navigation_zoom_range_and_lod_cache),
        ("AI Chatbot Rich Rendering & GGUF Range Resume", test_local_ai_manager_and_chatbot_rich_rendering),
    ]
    for label, fn in tests:
        print(f"=== Running: {label} ===", flush=True)
        fn()
        print(f"[PASS] {label}", flush=True)
    print("ALL_SYSTEMS_VERIFIED_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
