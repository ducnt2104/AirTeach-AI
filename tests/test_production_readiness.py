"""Production readiness & zero-placeholder verification suite for AirTeach AI.

Verifies:
1. Zero fake/placeholder document readers anywhere in Python or Rust sources.
2. Real multi-page PDF, PowerPoint (.pptx), and Word (.docx) high-DPI rendering
   (>= 1280px width, non-uniform pixel buffers, extracted text, < 0.2s SHA-256 CACHE_HIT,
   headless CLI rasterization, Continuous Vertical Scroll Mode, and Annotated PDF Export).
3. 60Hz Kinematic60HzExtrapolator benchmark (< 0.5ms/tick), Active Workspace Margin Mapping
   [12%..88% -> 0%..100%], wide full-frame + 45% velocity ROI acquisition near frame corners,
   scale-invariant 3D gestures (0.8m..3.2m), horizontal swipe (650ms cooldown), vertical scroll
   swipes (450ms cooldown) & two-finger drag, Dual-Hand Zoom (50%..300%), Dual-Hand Pan, and
   Left-Hand Fist LINE_LOCK / Two-Finger SHAPE_LOCK modifiers while Right Hand draws.
4. Slide-aware AI context capture and headless AirTeachMainWindow stability.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import docx
from docx.shared import Inches as DocxInches
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches as PptxInches
import pymupdf as fitz
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QWheelEvent
from PySide6.QtWidgets import QApplication

from cv_engine.document_presenter import (
    LectureDocument,
    convert_presentation_to_cached_pdf,
    convert_word_to_cached_pdf,
    export_annotated_lecture_pdf,
    rasterize_document_to_png_dir,
)
from cv_engine.drawing_fsm import ZeroStrayStrokeDrawingFsm
from cv_engine.hand_gesture import (
    DualHandGestureCoordinator,
    HandGestureEngine,
    Kinematic60HzExtrapolator,
    SlideNavigationGestureDetector,
    classify_hand_3d_vectors,
    map_active_workspace_norm,
)


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _qimage_to_rgb_array(img: QImage) -> np.ndarray:
    conv = img.convertToFormat(QImage.Format_RGB888)
    ptr = conv.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=conv.sizeInBytes())
    stride = conv.bytesPerLine()
    mat = arr.reshape((conv.height(), stride))
    return mat[:, : conv.width() * 3].reshape((conv.height(), conv.width(), 3)).copy()


def _make_skeleton(
    *,
    cx: float = 0.50,
    cy: float = 0.65,
    scale: float = 1.0,
    pinch_ratio_target: str = "OPEN",  # "OPEN" | "PINCH" | "PALM" | "FIST" | "TWO_FINGERS" | "INDEX"
) -> List[List[float]]:
    lm = [[cx, cy, 0.0] for _ in range(21)]

    def pt(dx: float, dy: float, dz: float = 0.0) -> List[float]:
        return [cx + dx * scale, cy + dy * scale, dz * scale]

    lm[0] = pt(0.0, 0.14, 0.0)
    lm[1] = pt(-0.04, 0.09, -0.01)
    lm[2] = pt(-0.06, 0.04, -0.02)
    lm[3] = pt(-0.07, -0.01, -0.025)

    idx_up = pinch_ratio_target in ("OPEN", "PALM", "TWO_FINGERS", "INDEX")
    mid_up = pinch_ratio_target in ("PALM", "TWO_FINGERS")
    ring_up = pinch_ratio_target == "PALM"
    pinky_up = pinch_ratio_target == "PALM"

    if pinch_ratio_target == "PINCH":
        lm[4] = pt(-0.02, -0.08, -0.03)
        lm[5] = pt(-0.03, -0.04, 0.0)
        lm[6] = pt(-0.03, -0.09, -0.015)
        lm[7] = pt(-0.024, -0.085, -0.025)
        lm[8] = pt(-0.018, -0.08, -0.03)
    else:
        lm[4] = pt(-0.11, -0.02, -0.03)
        lm[5] = pt(-0.03, -0.04, 0.0)
        if idx_up:
            lm[6] = pt(-0.03, -0.12, -0.01)
            lm[7] = pt(-0.03, -0.18, -0.015)
            lm[8] = pt(-0.03, -0.24, -0.02)
        else:
            lm[6] = pt(-0.03, -0.07, -0.015)
            lm[7] = pt(-0.02, -0.02, -0.025)
            lm[8] = pt(-0.015, 0.02, -0.02)

    lm[9] = pt(0.0, -0.05, 0.0)
    if mid_up:
        lm[10] = pt(0.0, -0.14, -0.01)
        lm[11] = pt(0.0, -0.21, -0.015)
        lm[12] = pt(0.0, -0.27, -0.02)
    else:
        lm[10] = pt(0.0, -0.07, -0.02)
        lm[11] = pt(0.0, -0.01, -0.03)
        lm[12] = pt(0.0, 0.03, -0.02)

    lm[13] = pt(0.03, -0.04, 0.0)
    if ring_up:
        lm[14] = pt(0.035, -0.12, -0.01)
        lm[15] = pt(0.04, -0.19, -0.015)
        lm[16] = pt(0.045, -0.25, -0.02)
    else:
        lm[14] = pt(0.03, -0.06, -0.02)
        lm[15] = pt(0.03, 0.0, -0.03)
        lm[16] = pt(0.025, 0.035, -0.02)

    lm[17] = pt(0.06, -0.02, 0.0)
    if pinky_up:
        lm[18] = pt(0.075, -0.10, -0.005)
        lm[19] = pt(0.09, -0.18, -0.01)
        lm[20] = pt(0.105, -0.25, -0.015)
    else:
        lm[18] = pt(0.06, -0.04, -0.02)
        lm[19] = pt(0.055, 0.01, -0.03)
        lm[20] = pt(0.05, 0.04, -0.02)

    return lm


def test_zero_placeholder_codebase_audit() -> None:
    forbidden = "Loaded via AirTeach Embedded PDF Reader"
    for folder in (PROJECT_ROOT / "cv_engine", PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".rs"):
                txt = p.read_text(encoding="utf-8", errors="ignore")
                assert forbidden not in txt, f"Placeholder PDF reader found in {p}"
    assert forbidden not in (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")


def test_real_pdf_pptx_docx_rendering_scroll_and_export() -> None:
    app = _ensure_qapp()
    from main import SmartboardCanvas

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        d = Path(tmp)
        cache_dir = d / "doc_cache"

        # Asset image embedded inside both PPTX and DOCX
        pic_path = d / "wave_figure.png"
        im = Image.new("RGB", (400, 220), (15, 45, 95))
        drw = ImageDraw.Draw(im)
        drw.rectangle((20, 20, 380, 200), outline=(56, 189, 248), width=6)
        drw.line((40, 110, 360, 110), fill=(250, 204, 21), width=4)
        im.save(pic_path)

        # 1. Real multi-page PDF
        pdf_path = d / "lecture_sample.pdf"
        pdf = fitz.open()
        for idx in range(2):
            pg = pdf.new_page(width=960, height=540)
            pg.insert_text((48, 84), f"Bai giang Toan Giai Tich - Trang {idx + 1}", fontsize=26)
            pg.draw_circle((480, 300), 90, color=(0.05, 0.55, 0.85), width=4)
        pdf.save(str(pdf_path))
        pdf.close()

        # 2. Real PPTX
        pptx_path = d / "slides_sample.pptx"
        prs = Presentation()
        prs.slide_width = PptxInches(13.333)
        prs.slide_height = PptxInches(7.5)
        s0 = prs.slides.add_slide(prs.slide_layouts[1])
        s0.shapes.title.text = "Bài giảng Dao động Cơ"
        s0.placeholders[1].text = "Phương trình: x = A cos(ωt + φ)"
        s0.shapes.add_picture(str(pic_path), PptxInches(6.8), PptxInches(2.2), width=PptxInches(5.0))
        s1 = prs.slides.add_slide(prs.slide_layouts[1])
        s1.shapes.title.text = "Ví dụ minh họa"
        s1.placeholders[1].text = "Tính chu kỳ T = 2π / ω"
        prs.save(str(pptx_path))

        # 3. Real multi-page DOCX (Heading, Vietnamese paragraphs, table, embedded image)
        docx_path = d / "lesson_notes.docx"
        wdoc = docx.Document()
        wdoc.add_heading("Giáo án Vật lý 12 — Sóng cơ và Giao thoa", level=1)
        p = wdoc.add_paragraph("Bước sóng λ = v · T = v / f là quãng đường sóng truyền trong một chu kỳ.")
        p.runs[0].bold = True
        tbl = wdoc.add_table(rows=2, cols=2)
        tbl.cell(0, 0).text = "Đại lượng"
        tbl.cell(0, 1).text = "Đơn vị"
        tbl.cell(1, 0).text = "Tần số góc ω"
        tbl.cell(1, 1).text = "rad/s"
        wdoc.add_picture(str(pic_path), width=DocxInches(4.2))
        for k in range(36):
            wdoc.add_paragraph(
                f"Mục {k + 1}: Phương trình giao thoa tại điểm M với độ lệch pha Δφ = 2π(d2 - d1)/λ."
            )
        wdoc.save(str(docx_path))

        # Verify PDF, PPTX, and DOCX produce real pixel buffers (width >= 1280, std > 8.0) & cache hits
        for doc_file, expected_type, expected_word in (
            (pdf_path, "PDF", "Bai giang"),
            (pptx_path, "PPTX", "Dao động"),
            (docx_path, "WORD", "Sóng cơ"),
        ):
            ldoc = LectureDocument.open(doc_file, cache_dir=cache_dir)
            try:
                assert ldoc.doc_type == expected_type
                assert ldoc.page_count >= 2
                qimg = ldoc.render_page_qimage(0, scale=2.25)
                assert not qimg.isNull()
                assert qimg.width() >= 1280, f"{doc_file.name} rendered too narrow: {qimg.width()}"
                arr = _qimage_to_rgb_array(qimg)
                assert float(np.std(arr)) > 8.0, f"{doc_file.name} rendered uniform/blank buffer"
                txt0 = ldoc.get_page_text(0)
                assert expected_word.lower() in txt0.lower() or len(txt0) > 15
            finally:
                ldoc.close()

        # Verify <0.2s SHA-256 CACHE_HIT on DOCX and PPTX re-open
        t_c0 = time.perf_counter()
        _, _, back_word = convert_word_to_cached_pdf(docx_path, cache_dir=cache_dir)
        assert back_word == "CACHE_HIT"
        assert (time.perf_counter() - t_c0) < 0.2

        t_c1 = time.perf_counter()
        _, _, back_pptx = convert_presentation_to_cached_pdf(pptx_path, cache_dir=cache_dir)
        assert back_pptx == "CACHE_HIT"
        assert (time.perf_counter() - t_c1) < 0.2

        # Verify headless CLI rasterization (used by Rust viewer) produces non-blank >=1600px PNGs
        ras_dir = d / "cli_ras"
        manifest_path = rasterize_document_to_png_dir(docx_path, ras_dir, target_min_width=1600)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["doc_type"] == "WORD"
        assert manifest["page_count"] >= 2
        png0 = Image.open(ras_dir / manifest["pages"][0]["image_file"])
        assert png0.width >= 1280
        assert float(np.std(np.asarray(png0))) > 8.0

        # Verify SmartboardCanvas Continuous Vertical Scroll Mode for Word + per-page vector ink + PDF Export
        canvas = SmartboardCanvas()
        canvas.resize(1280, 760)
        word_doc = LectureDocument.open(docx_path, cache_dir=cache_dir)
        canvas.attach_lecture_document(word_doc)
        assert canvas.continuous_scroll_mode is True
        assert canvas.doc_scroll_y == 0.0

        # Draw red stroke on Page 0 while in Continuous Scroll Mode
        canvas.strokes.append({
            "points": [QPointF(320.0, 240.0), QPointF(560.0, 280.0), QPointF(620.0, 350.0)],
            "color": QColor("#EF4444"),
            "thickness": 6.0,
            "is_highlighter": False,
        })
        assert len(canvas.slide_annotations[0]["strokes"]) == 1

        # Mouse wheel down scrolls vertically smoothly
        wheel_down = QWheelEvent(
            QPointF(500, 400),
            QPointF(500, 400),
            QPoint(0, 0),
            QPoint(0, -480),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        canvas.wheelEvent(wheel_down)
        assert canvas.doc_scroll_y > 0.0

        # Export Annotated PDF with ink burned onto actual rendered DOCX page
        out_pdf = d / "word_annotated.pdf"
        saved = export_annotated_lecture_pdf(
            out_pdf,
            canvas.slides,
            canvas.slide_annotations,
            (canvas.width(), canvas.height()),
            lecture_doc=canvas.lecture_doc,
        )
        verify = fitz.open(str(saved))
        try:
            assert verify.page_count == word_doc.page_count
            pix0 = verify[0].get_pixmap(matrix=fitz.Matrix(1, 1))
            p_arr = np.frombuffer(pix0.samples, dtype=np.uint8).reshape(pix0.height, pix0.width, pix0.n)
            red_px = np.sum((p_arr[:, :, 0] > 180) & (p_arr[:, :, 1] < 100) & (p_arr[:, :, 2] < 100))
            assert red_px > 20
        finally:
            verify.close()
            canvas.lecture_doc.close()
        app.processEvents()


def test_60hz_extrapolator_margin_mapping_roi_and_dual_hand_gestures() -> None:
    # 1. Active Workspace Margin Mapping [12%..88%] -> [0%..100%]
    assert map_active_workspace_norm(0.12, 0.12) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert map_active_workspace_norm(0.88, 0.88) == pytest.approx((1.0, 1.0), abs=1e-6)
    assert map_active_workspace_norm(0.50, 0.50) == pytest.approx((0.5, 0.5), abs=1e-6)
    assert map_active_workspace_norm(0.05, 0.95) == (0.0, 1.0)

    # 2. 60Hz Extrapolator + Adaptive One-Euro Filter (min_cutoff=0.8, beta=0.04) performance & smoothness
    extrap = Kinematic60HzExtrapolator(min_cutoff=0.8, beta=0.04, d_cutoff=1.0)
    extrap.ingest_vision((0.20, 0.40), t_sec=0.0)
    extrap.sample_60hz(t_sec=1.0 / 60.0)
    n_steps = 300
    t_bench0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        t_ai = step * (1.0 / 30.0)
        p_in = extrap.ingest_vision((0.20 + (step % 25) * 0.02, 0.40 + (step % 25) * 0.005), t_sec=t_ai)
        assert p_in is not None
        p_mid, is_ext = extrap.sample_60hz(t_sec=t_ai + (1.0 / 60.0))
        assert p_mid is not None and is_ext is True
        assert 0.0 <= p_mid[0] <= 1.0 and 0.0 <= p_mid[1] <= 1.0
    elapsed_ms = (time.perf_counter() - t_bench0) * 1000.0
    assert (elapsed_ms / (n_steps * 2.0)) < 0.5, f"Extrapolator too slow: {elapsed_ms / (n_steps * 2.0):.3f}ms/tick"

    # 3. Full-Frame Edge Acquisition + 45% Velocity-Predicted ROI Box (corner hand at 0.87, 0.84)
    engine = HandGestureEngine(
        model_path=PROJECT_ROOT / "models" / "hand_landmarker.task",
        num_hands=2,
        roi_padding_ratio=0.45,
        one_euro_min_cutoff=0.8,
        one_euro_beta=0.04,
    )
    assert engine._has_detector()
    # Close native C++ HandLandmarker cleanly before installing deterministic synthetic frame mock
    engine.close()
    edge_lm = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in _make_skeleton(cx=0.86, cy=0.80, scale=0.65)]
    engine.tasks_detector = SimpleNamespace(
        detect=lambda _mp_img: SimpleNamespace(
            hand_landmarks=[edge_lm],
            handedness=[[SimpleNamespace(score=0.90, category_name="Right")]],
        )
    )
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    _, right_h = engine.detect_hands(frame, face_boxes=[], search_size=(640, 480))
    assert right_h is not None
    assert len(engine._last_hand_roi_boxes) == 1
    roi_w = engine._last_hand_roi_boxes[0][2]
    assert roi_w >= 0.12
    engine.close()

    # 4. Scale-Invariant 3D Gestures across 0.8m (large) & 3.2m (small) + Pinch 0.32/0.42 FSM
    for scale in (1.35, 0.32):
        pinch_info = classify_hand_3d_vectors(
            _make_skeleton(scale=scale, pinch_ratio_target="PINCH"),
            pinch_enter_ratio=0.32,
            pinch_exit_ratio=0.42,
        )
        assert pinch_info["pose_tag"] == "PINCH"
        palm_info = classify_hand_3d_vectors(_make_skeleton(scale=scale, pinch_ratio_target="PALM"))
        assert palm_info["pose_tag"] == "OPEN_PALM"

    fsm = ZeroStrayStrokeDrawingFsm(pinch_enter=0.32, pinch_exit=0.42, tail_trim_ms=100.0, allow_index_point_draw=False)
    for i in range(12):
        fsm.step(100.0 + i * 33.0, True, (200.0 + i * 16.0, 260.0), 0.30, "PINCH")
    rel = fsm.step(520.0, True, (410.0, 260.0), 0.45, "INDEX")
    assert rel.action == "COMMIT_STROKE" and len(rel.committed_points) >= 2

    # 5. Horizontal Swipe (650ms cooldown) & Vertical Scroll Swipe (450ms cooldown + 2-finger drag)
    nav = SlideNavigationGestureDetector(swipe_cooldown_sec=0.65, vertical_cooldown_sec=0.45)
    nav.update({"cursor_norm": (0.68, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.00)
    nav.update({"cursor_norm": (0.54, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.07)
    r_next = nav.update({"cursor_norm": (0.40, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.14)
    assert r_next["nav_event"] == "SLIDE_NEXT" and r_next["swipe_direction"] == "LEFT"

    # Rapid swipe within 650ms suppressed
    nav.update({"cursor_norm": (0.68, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.30)
    nav.update({"cursor_norm": (0.52, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.36)
    r_sup = nav.update({"cursor_norm": (0.36, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=1.42)
    assert r_sup["nav_event"] is None

    # Vertical scroll down swipe (hand sweeps upward: y goes 0.68 -> 0.46)
    nav.update({"cursor_norm": (0.50, 0.68), "pose_tag": "OPEN_PALM"}, t_sec=2.50)
    nav.update({"cursor_norm": (0.50, 0.56), "pose_tag": "OPEN_PALM"}, t_sec=2.57)
    r_scroll = nav.update({"cursor_norm": (0.50, 0.44), "pose_tag": "OPEN_PALM"}, t_sec=2.64)
    assert r_scroll["nav_event"] == "SCROLL_DOWN" and r_scroll["swipe_direction"] == "UP"

    # Vertical scroll up swipe after 450ms cooldown
    nav.update({"cursor_norm": (0.50, 0.36), "pose_tag": "OPEN_PALM"}, t_sec=3.25)
    nav.update({"cursor_norm": (0.50, 0.50), "pose_tag": "OPEN_PALM"}, t_sec=3.32)
    r_scroll_up = nav.update({"cursor_norm": (0.50, 0.64), "pose_tag": "OPEN_PALM"}, t_sec=3.39)
    assert r_scroll_up["nav_event"] == "SCROLL_UP" and r_scroll_up["swipe_direction"] == "DOWN"

    # Two-finger smooth drag scroll
    nav.update({"cursor_norm": (0.50, 0.60), "pose_tag": "TWO_FINGERS"}, t_sec=4.00)
    r_drag = nav.update({"cursor_norm": (0.50, 0.54), "pose_tag": "TWO_FINGERS"}, t_sec=4.06)
    assert r_drag["scroll_delta_norm"] > 0.0

    # 6. Dual-Hand Zoom, Pan, and Left-Hand Fist LINE_LOCK / Two-Finger SHAPE_LOCK Modifiers
    coord = DualHandGestureCoordinator(primary_pref="right")
    z0 = coord.update(
        {"pose_tag": "PINCH", "pinch_ratio": 0.22, "cursor_norm": (0.40, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.22, "cursor_norm": (0.60, 0.50)},
        t_sec=1.0,
    )
    assert z0.two_hand_mode == "DUAL_PINCH_ZOOM"
    z1 = coord.update(
        {"pose_tag": "PINCH", "pinch_ratio": 0.22, "cursor_norm": (0.30, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.22, "cursor_norm": (0.70, 0.50)},
        t_sec=1.05,
    )
    assert z1.two_hand_mode == "DUAL_PINCH_ZOOM" and z1.zoom_factor > 1.05

    p0 = coord.update(
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.40, 0.50)},
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.60, 0.50)},
        t_sec=2.0,
    )
    p1 = coord.update(
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.46, 0.54)},
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.66, 0.54)},
        t_sec=2.05,
    )
    assert p0.two_hand_mode == "DUAL_PALM_PAN" and p1.pan_delta_norm[0] > 0.0

    m_line = coord.update(
        {"pose_tag": "FIST", "pinch_ratio": 0.70, "cursor_norm": (0.25, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.72, 0.50)},
        t_sec=3.0,
    )
    assert m_line.two_hand_mode == "MODIFIER_DRAW" and m_line.modifier_mode == "LINE_LOCK"

    m_shape = coord.update(
        {"pose_tag": "TWO_FINGERS", "pinch_ratio": 0.75, "cursor_norm": (0.25, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.72, 0.50)},
        t_sec=3.1,
    )
    assert m_shape.two_hand_mode == "MODIFIER_DRAW" and m_shape.modifier_mode == "SHAPE_LOCK"


def test_slide_aware_ai_and_headless_window(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _ensure_qapp()
    import main

    captured_calls = []
    monkeypatch.setattr(main.CameraHandWorker, "start", lambda self: None)
    monkeypatch.setattr(
        main.AirTeachMainWindow,
        "_start_hybrid_worker",
        lambda self, prompt, png_bytes=None, structured_json=False, include_board_tools=True: captured_calls.append(
            (prompt, png_bytes, structured_json)
        ),
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        pdf_path = Path(tmp) / "ai_slide.pdf"
        doc = fitz.open()
        pg = doc.new_page(width=960, height=540)
        pg.insert_text((60, 120), "Giai phuong trinh bac hai x^2 - 5x + 6 = 0", fontsize=24)
        doc.save(str(pdf_path))
        doc.close()

        win = main.AirTeachMainWindow()
        try:
            win.resize(1280, 780)
            ldoc = LectureDocument.open(pdf_path, cache_dir=Path(tmp) / "cache")
            win.canvas.attach_lecture_document(ldoc)

            # Verify zoom clamping [50%..300%] & floating toast
            win.canvas.apply_zoom_factor(5.0)
            assert win.canvas.zoom_scale == pytest.approx(3.0, abs=1e-3)
            win.canvas.apply_zoom_factor(0.05)
            assert win.canvas.zoom_scale == pytest.approx(0.50, abs=1e-3)
            win.canvas.apply_zoom_factor(2.0)

            # Verify 60Hz cursor extrapolator + canvas update
            win.extrapolator.ingest_vision((0.40, 0.40), t_sec=time.perf_counter() - 0.02)
            win.canvas.apply_60hz_extrapolated_cursor((0.55, 0.52))
            assert win.canvas.cursor_pos is not None

            # Verify _analyze_full_board attaches real slide image + extracted slide text
            win._analyze_full_board("1. Giải & Chữa Bài Trên Bảng (JSON)", structured_json=False)
            assert len(captured_calls) == 1
            sent_prompt, sent_png, _ = captured_calls[0]
            assert "x^2 - 5x + 6 = 0" in sent_prompt
            assert sent_png is not None and len(sent_png) > 500
        finally:
            win.close()
            win.shutdown_resources()
            app.processEvents()
            del win
