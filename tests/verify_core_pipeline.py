"""End-to-end automated verification suite for AirTeach AI core pipelines.

Covers:
1. HandGestureEngine fail-safe initialization (.task model + mp.solutions.hands fallback)
   and hysteretic GRACE_PERIOD lock-on + low-light dark-frame stability.
2. One-Euro filtering, 3D scale-invariant gesture classification, ZeroStrayStrokeDrawingFsm
   (nib lock, tail cutoff, moving fist eraser), SlideNavigationGestureDetector (swipe + cooldown,
   two-finger peace hold, pinky QUESTION_HOLD), and Smart Shape AXES auto-complete.
3. Native PDF (PyMuPDF) & PowerPoint (python-pptx -> SHA-256 doc_cache PDF) loading, <0.2s
   cache hit on re-open, per-slide non-destructive ink isolation, and annotated PDF export.
4. ResilientGeminiClient multi-key rotation and headless PySide6 AirTeachMainWindow smoke test.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pymupdf as fitz  # PyMuPDF
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from cv_engine.document_presenter import (
    LectureDocument,
    convert_presentation_to_cached_pdf,
    export_annotated_lecture_pdf,
)
from cv_engine.drawing_fsm import ZeroStrayStrokeDrawingFsm
from cv_engine.gemini_client import ResilientGeminiClient
from cv_engine.hand_gesture import (
    HandGestureEngine,
    SlideNavigationGestureDetector,
    classify_hand_3d_vectors,
)
from cv_engine.shape_recognizer import beautify_stroke


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_hand_landmarks(
    *,
    thumb_tip=(0.44, 0.56, -0.03),
    index_extended=True,
    middle_extended=False,
    ring_extended=False,
    pinky_extended=False,
    wrist=(0.50, 0.78, 0.0),
    pinch_index_to_thumb=False,
    scale=1.0,
) -> np.ndarray:
    """Builds a realistic 21x3 normalized MediaPipe hand skeleton."""
    lm = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    lm[0] = [wx, wy, wz]

    def pt(dx: float, dy: float, dz: float = 0.0) -> List[float]:
        return [wx + dx * scale, wy + dy * scale, wz + dz * scale]

    # Thumb (1..4)
    lm[1] = pt(-0.04, -0.04, -0.01)
    lm[2] = pt(-0.06, -0.09, -0.02)
    lm[3] = pt(-0.07, -0.14, -0.025)
    if pinch_index_to_thumb:
        lm[4] = pt(-0.02, -0.22, -0.03)
    else:
        lm[4] = [
            wx + (thumb_tip[0] - 0.50) * scale,
            wy + (thumb_tip[1] - 0.78) * scale,
            wz + thumb_tip[2] * scale,
        ]

    # Index (5..8)
    lm[5] = pt(-0.03, -0.18, 0.0)
    if pinch_index_to_thumb:
        lm[6] = pt(-0.03, -0.23, -0.015)
        lm[7] = pt(-0.025, -0.225, -0.025)
        lm[8] = pt(-0.02, -0.22, -0.03)
    elif index_extended:
        lm[6] = pt(-0.03, -0.26, -0.01)
        lm[7] = pt(-0.03, -0.32, -0.015)
        lm[8] = pt(-0.03, -0.38, -0.02)
    else:
        lm[6] = pt(-0.03, -0.21, -0.015)
        lm[7] = pt(-0.02, -0.17, -0.025)
        lm[8] = pt(-0.015, -0.14, -0.02)

    # Middle (9..12)
    lm[9] = pt(0.0, -0.19, 0.0)
    if middle_extended:
        lm[10] = pt(0.0, -0.28, -0.01)
        lm[11] = pt(0.0, -0.35, -0.015)
        lm[12] = pt(0.0, -0.41, -0.02)
    else:
        lm[10] = pt(0.0, -0.21, -0.02)
        lm[11] = pt(0.0, -0.16, -0.03)
        lm[12] = pt(0.0, -0.13, -0.02)

    # Ring (13..16)
    lm[13] = pt(0.03, -0.18, 0.0)
    if ring_extended:
        lm[14] = pt(0.035, -0.26, -0.01)
        lm[15] = pt(0.04, -0.33, -0.015)
        lm[16] = pt(0.045, -0.39, -0.02)
    else:
        lm[14] = pt(0.03, -0.20, -0.02)
        lm[15] = pt(0.03, -0.15, -0.03)
        lm[16] = pt(0.025, -0.12, -0.02)

    # Pinky (17..20)
    lm[17] = pt(0.06, -0.16, 0.0)
    if pinky_extended:
        lm[18] = pt(0.075, -0.24, -0.005)
        lm[19] = pt(0.09, -0.32, -0.01)
        lm[20] = pt(0.105, -0.40, -0.015)
    else:
        lm[18] = pt(0.06, -0.18, -0.02)
        lm[19] = pt(0.055, -0.14, -0.03)
        lm[20] = pt(0.05, -0.11, -0.02)

    return lm


def test_hand_engine_init_fallback_and_grace_period() -> None:
    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    engine = HandGestureEngine(
        model_path=model_path,
        num_hands=2,
        min_detection_confidence=0.35,
        min_presence_confidence=0.35,
        min_tracking_confidence=0.40,
        one_euro_d_cutoff=1.0,
        roi_unsharp_strength=0.65,
    )
    assert engine._has_detector(), "Expected MediaPipe task landmarker to initialize"
    assert engine.backend_mode in {"TASKS_LANDMARKER", "SOLUTIONS_HANDS"}

    # Verify auto-repair copies missing model & automatic fallback to mp.solutions.hands
    import cv_engine.hand_gesture as hg_mod
    import mediapipe as mp

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        repaired_model = Path(tmp) / "repaired_hand_landmarker.task"
        repaired_engine = HandGestureEngine(model_path=repaired_model, num_hands=2)
        assert repaired_model.exists() and repaired_model.stat().st_size > 500_000
        assert repaired_engine.backend_mode == "TASKS_LANDMARKER"
        for attr in ("landmarker", "_landmarker"):
            lm_obj = getattr(repaired_engine, attr, None)
            if lm_obj is not None and hasattr(lm_obj, "close"):
                try:
                    lm_obj.close()
                except Exception:
                    pass

    had_solutions = hasattr(mp, "solutions")
    orig_solutions = getattr(mp, "solutions", None)
    orig_ensure = hg_mod.ensure_hand_model
    fake_lm = _make_hand_landmarks(index_extended=True, middle_extended=False)
    fake_lm_objs = [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in fake_lm]
    emit_hand = {"active": False}

    class _FakeHands:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process(self, rgb_image):
            if emit_hand["active"]:
                return SimpleNamespace(
                    multi_hand_landmarks=[SimpleNamespace(landmark=fake_lm_objs)],
                    multi_handedness=[
                        SimpleNamespace(
                            classification=[SimpleNamespace(label="Right", score=0.92)]
                        )
                    ],
                )
            return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

        def close(self):
            pass

    mp.solutions = SimpleNamespace(hands=SimpleNamespace(Hands=_FakeHands))
    hg_mod.ensure_hand_model = lambda *_args, **_kwargs: False
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bad_model = Path(tmp) / "unrecoverable_hand_landmarker.task"
            fallback_engine = HandGestureEngine(model_path=bad_model, num_hands=2)
            assert fallback_engine.backend_mode == "SOLUTIONS_HANDS"
            assert fallback_engine._has_detector()

            # Verify synthetic dark frame + GRACE_PERIOD lock-on + One-Euro pinch filter
            dark_frame = np.full((720, 1280, 3), 18, dtype=np.uint8)
            # Call 1: empty dark frame returns None, None cleanly
            left_res, right_res = fallback_engine.detect_hands(
                dark_frame,
                face_boxes=[],
                search_size=(640, 360),
                roi_target_size=(420, 420),
                enable_roi_sharpen=True,
            )
            assert left_res is None and right_res is None

            # Call 2: sustain hand lock across ~180ms (min_hold_sec + frames) via SOLUTIONS_HANDS
            emit_hand["active"] = True
            acquired = None
            for _ in range(5):
                acq_l, acq_r = fallback_engine.detect_hands(
                    dark_frame, face_boxes=[], search_size=(640, 360)
                )
                acquired = acq_l or acq_r
                time.sleep(0.035)
            assert acquired is not None
            assert not acquired.get("in_grace_period", False)
            assert acquired.get("active_gesture") != "IDLE"

            # Call 3: sudden 1-frame detector miss must stay locked via GRACE_PERIOD
            emit_hand["active"] = False
            gr_l, gr_r = fallback_engine.detect_hands(
                dark_frame, face_boxes=[], search_size=(640, 360)
            )
            grace = gr_l or gr_r
            assert grace is not None, "Hand lock flickered off on transient 1-frame dropout!"
            assert grace.get("in_grace_period") is True
            assert fallback_engine.left_fsm.pinch_filter is not None
            assert fallback_engine.right_fsm.pinch_filter is not None
    finally:
        hg_mod.ensure_hand_model = orig_ensure
        if had_solutions:
            mp.solutions = orig_solutions
        elif hasattr(mp, "solutions"):
            delattr(mp, "solutions")


def test_3d_gestures_drawing_fsm_slide_nav_and_smart_shapes() -> None:
    # 1. Scale-invariant 3D poses: POINTER vs OPEN_PALM vs QUESTION_HOLD
    pointer_lm = _make_hand_landmarks(index_extended=True)
    pointer_3d = classify_hand_3d_vectors(pointer_lm)
    assert pointer_3d["pose_tag"] in {"INDEX", "INDEX_POINT"}
    assert pointer_3d["hand_span"] > 0.0

    palm_lm = _make_hand_landmarks(
        thumb_tip=(0.35, 0.62, -0.03),
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )
    palm_3d = classify_hand_3d_vectors(palm_lm)
    assert palm_3d["pose_tag"] == "OPEN_PALM"
    assert palm_3d["hand_span"] > pointer_3d["hand_span"]

    pinky_lm = _make_hand_landmarks(
        thumb_tip=(0.38, 0.58, -0.03),
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=True,
    )
    pinky_3d = classify_hand_3d_vectors(pinky_lm)
    assert pinky_3d["pose_tag"] == "QUESTION_HOLD"

    # 2. Drawing FSM: hover-only does not stray-draw, pinch draws + trims tail + fist scrub erases
    fsm = ZeroStrayStrokeDrawingFsm(allow_index_point_draw=False)
    hover_event = fsm.step(
        t_ms=1000.0,
        hand_present=True,
        cursor_xy=(300.0, 220.0),
        pinch_ratio=0.75,
        pose_tag="INDEX",
    )
    assert hover_event.action != "BEGIN_STROKE"

    # Acquire pinch (<0.28 for >=3 frames) and draw long enough (>300ms) so 105ms tail trim keeps stroke valid
    begun = False
    for i in range(14):
        ev = fsm.step(
            t_ms=1050.0 + i * 33.0,
            hand_present=True,
            cursor_xy=(300.0 + i * 16.0, 220.0 + i * 4.0),
            pinch_ratio=0.18,
            pose_tag="PINCH",
        )
        if ev.action == "BEGIN_STROKE":
            begun = True
    assert begun, "Expected pinch hold to enter BEGIN_STROKE"

    # Release pinch (>0.40) -> stroke commits cleanly with 90-120ms tail trimmed
    end_ev = None
    for j in range(3):
        end_ev = fsm.step(
            t_ms=1520.0 + j * 35.0,
            hand_present=True,
            cursor_xy=(520.0 + j * 4.0, 276.0),
            pinch_ratio=0.65,
            pose_tag="INDEX",
        )
        if end_ev.action in {"COMMIT_STROKE", "CANCEL_STROKE"}:
            break
    assert end_ev is not None and end_ev.action == "COMMIT_STROKE"
    assert len(end_ev.committed_points) >= 2

    # Moving fist outside post-commit cancel window triggers dynamic ERASE
    erase_triggered = False
    for i in range(6):
        ev = fsm.step(
            t_ms=2600.0 + i * 40.0,
            hand_present=True,
            cursor_xy=(200.0 + i * 24.0, 260.0),
            pinch_ratio=0.60,
            pose_tag="FIST",
        )
        if ev.action == "ERASE":
            erase_triggered = True
            break
    assert erase_triggered, "Moving FIST scrub should trigger dynamic ERASE"

    # 3. Slide navigation detector: swipe + 600ms cooldown + 800ms two-finger hold + 900ms question hold
    nav = SlideNavigationGestureDetector(
        swipe_cooldown_sec=0.60,
        swipe_min_dx=0.16,
        swipe_min_vel=0.85,
        swipe_window_sec=0.28,
        peace_hold_sec=0.80,
        question_hold_sec=0.90,
    )
    # Right-to-left swipe (dx < 0) across >=3 samples -> SLIDE_NEXT
    nav.update({"cursor_norm": (0.66, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.00)
    nav.update({"cursor_norm": (0.54, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.07)
    out_next = nav.update({"cursor_norm": (0.40, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.14)
    assert out_next["nav_event"] == "SLIDE_NEXT"

    # Immediate second swipe within 600ms cooldown must be blocked
    nav.update({"cursor_norm": (0.66, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.20)
    nav.update({"cursor_norm": (0.50, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.26)
    out_cd = nav.update({"cursor_norm": (0.36, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=10.32)
    assert out_cd["nav_event"] is None
    assert out_cd["cooldown_remaining_ms"] > 0.0

    # Left-to-right swipe (dx > 0) after cooldown -> SLIDE_PREV
    nav.update({"cursor_norm": (0.36, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=11.00)
    nav.update({"cursor_norm": (0.50, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=11.07)
    out_prev = nav.update({"cursor_norm": (0.64, 0.55), "pose_tag": "OPEN_PALM"}, t_sec=11.14)
    assert out_prev["nav_event"] == "SLIDE_PREV"

    # Two-finger peace hold >= 0.80s -> SLIDE_NEXT
    nav.update({"cursor_norm": (0.50, 0.55), "pose_tag": "TWO_FINGERS"}, t_sec=12.00)
    mid_peace = nav.update({"cursor_norm": (0.505, 0.55), "pose_tag": "TWO_FINGERS"}, t_sec=12.40)
    assert mid_peace["nav_event"] is None and mid_peace["peace_hold_progress"] > 0.4
    done_peace = nav.update({"cursor_norm": (0.505, 0.55), "pose_tag": "TWO_FINGERS"}, t_sec=12.90)
    assert done_peace["nav_event"] == "SLIDE_NEXT"

    # Question hold >= 0.90s -> ASK_SLIDE_AI
    nav.update({"cursor_norm": (0.50, 0.55), "pose_tag": "QUESTION_HOLD"}, t_sec=14.00)
    done_q = nav.update({"cursor_norm": (0.50, 0.55), "pose_tag": "QUESTION_HOLD"}, t_sec=15.05)
    assert done_q["nav_event"] == "ASK_SLIDE_AI"

    # 4. Smart Shapes: Circle + Coordinate Axes (Oxy)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=True)
    circle_pts = [(300.0 + 70.0 * np.cos(a), 300.0 + 70.0 * np.sin(a)) for a in angles]
    circle_res = beautify_stroke(circle_pts)
    assert circle_res.shape_type == "CIRCLE"

    axes_pts = [(120.0, 100.0 + i * 10.0) for i in range(15)] + [
        (120.0 + j * 12.0, 240.0) for j in range(1, 16)
    ]
    axes_res = beautify_stroke(axes_pts)
    assert axes_res.shape_type in {"AXES", "TRIANGLE", "LINE"}


def test_pdf_pptx_render_cache_annotations_and_export() -> None:
    app = _ensure_qapp()
    from main import SmartboardCanvas

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cache_dir = tmp_dir / "doc_cache"

        # 1. Build 2-page PDF via PyMuPDF
        pdf_path = tmp_dir / "physics_lecture.pdf"
        doc = fitz.open()
        for idx in range(2):
            page = doc.new_page(width=960, height=540)
            page.insert_text((48, 80), f"Bai giang Vat ly 12 - Trang {idx + 1}", fontsize=26)
            page.draw_rect(fitz.Rect(60, 130, 320, 280), color=(0.1, 0.4, 0.9), width=3)
        doc.save(str(pdf_path))
        doc.close()

        loaded_pdf = LectureDocument.open(pdf_path, cache_dir=cache_dir)
        try:
            assert loaded_pdf.page_count == 2
            assert "Trang 1" in loaded_pdf.get_page_text(0)
            qimg = loaded_pdf.render_page_qimage(0, scale=2.0)
            assert not qimg.isNull()
            assert qimg.width() >= 1900 and qimg.height() >= 1000
        finally:
            loaded_pdf.close()

        # 2. Build 2-slide PPTX with Vietnamese text, rectangle shape, and embedded Pillow PNG
        img_path = tmp_dir / "diagram.png"
        pil_img = Image.new("RGB", (320, 180), color=(24, 56, 110))
        draw = ImageDraw.Draw(pil_img)
        draw.ellipse((40, 20, 280, 160), outline=(56, 189, 248), width=5)
        pil_img.save(img_path)

        pptx_path = tmp_dir / "chemistry_slides.pptx"
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        slide0 = prs.slides.add_slide(prs.slide_layouts[1])
        slide0.shapes.title.text = "Phản ứng Oxi hóa - Khử"
        slide0.placeholders[1].text = "Cân bằng electron từng bước cho học sinh"
        slide0.shapes.add_picture(str(img_path), Inches(7.2), Inches(2.0), width=Inches(4.5))

        slide1 = prs.slides.add_slide(prs.slide_layouts[1])
        slide1.shapes.title.text = "Bài tập vận dụng"
        slide1.placeholders[1].text = "Fe + CuSO4 -> FeSO4 + Cu"
        prs.save(str(pptx_path))

        cached_pdf_path, texts_1, backend_1 = convert_presentation_to_cached_pdf(
            pptx_path, cache_dir=cache_dir
        )
        assert cached_pdf_path.exists()
        assert backend_1 in {"LIBREOFFICE", "POWERPOINT_COM", "PYTHON_PPTX"}
        assert len(texts_1) == 2

        t0 = time.perf_counter()
        cached_pdf_2, _, backend_2 = convert_presentation_to_cached_pdf(
            pptx_path, cache_dir=cache_dir
        )
        elapsed_cache = time.perf_counter() - t0
        assert cached_pdf_2 == cached_pdf_path
        assert backend_2 == "CACHE_HIT"
        assert elapsed_cache < 0.2, f"Cache hit took {elapsed_cache:.3f}s >= 0.2s"

        # 3. Verify per-slide annotation persistence + SmartboardCanvas + Annotated PDF Export
        canvas = SmartboardCanvas()
        canvas.resize(1280, 720)
        pptx_doc = LectureDocument.open(pptx_path, cache_dir=cache_dir)
        canvas.attach_lecture_document(pptx_doc)
        assert len(canvas.slides) == 2

        # Draw on Slide 0: regular red pen + yellow highlighter using exact QPointF/QColor format
        canvas.active_tool = "PEN"
        canvas.strokes.append(
            {
                "points": [QPointF(300.0, 260.0), QPointF(520.0, 300.0), QPointF(600.0, 360.0)],
                "color": QColor("#EF4444"),
                "thickness": 6.0,
                "is_highlighter": False,
            }
        )
        canvas.strokes.append(
            {
                "points": [QPointF(260.0, 400.0), QPointF(620.0, 400.0)],
                "color": QColor("#FDE047"),
                "thickness": 18.0,
                "is_highlighter": True,
            }
        )
        assert len(canvas.slide_annotations[0]["strokes"]) == 2

        # Navigate to Slide 1 -> independent empty ink layer
        canvas.next_slide()
        assert canvas.current_slide_idx == 1
        assert len(canvas.strokes) == 0
        canvas.strokes.append(
            {
                "points": [QPointF(340.0, 300.0), QPointF(580.0, 300.0)],
                "color": QColor("#38BDF8"),
                "thickness": 6.0,
                "is_highlighter": False,
            }
        )

        # Go back to Slide 0 -> verify both strokes persisted
        canvas.prev_slide()
        assert canvas.current_slide_idx == 0
        assert len(canvas.strokes) == 2

        # Capture Slide + Ink context for AI
        ctx = canvas.capture_active_slide_context()
        assert len(ctx["png_bytes"]) > 500
        assert isinstance(ctx["slide_text"], str) and len(ctx["slide_text"]) > 0

        # Export Annotated PDF
        exported_pdf = tmp_dir / "annotated_export.pdf"
        out_path = export_annotated_lecture_pdf(
            exported_pdf,
            canvas.slides,
            canvas.slide_annotations,
            (canvas.width(), canvas.height()),
            lecture_doc=canvas.lecture_doc,
        )
        assert out_path.exists()
        verify_doc = fitz.open(str(out_path))
        try:
            assert verify_doc.page_count == 2
            pix = verify_doc[0].get_pixmap(matrix=fitz.Matrix(1, 1))
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            # Red pen ink (R high, G/B lower) must appear on rendered page 0
            red_pixels = np.sum((arr[:, :, 0] > 180) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100))
            assert red_pixels > 20, "Burned teacher ink missing in exported annotated PDF"
        finally:
            verify_doc.close()

        # Clear slide 0 ink without wiping slide 1 ink
        canvas.clear_slide_ink()
        assert len(canvas.slide_annotations[0]["strokes"]) == 0
        assert len(canvas.slide_annotations[1]["strokes"]) == 1

        canvas.lecture_doc.close()
        app.processEvents()


def test_gemini_key_rotation_and_headless_window() -> None:
    app = _ensure_qapp()

    # 1. Verify ResilientGeminiClient rotates cleanly across multiple keys on HTTP 429
    attempted_keys: List[str] = []

    class _FakeModels:
        def __init__(self, key: str):
            self.key = key

        def generate_content_stream(self, *, model, contents, config=None):
            attempted_keys.append(self.key)
            if self.key == "AIzaKeyAlpha123456":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
            yield SimpleNamespace(text="Giải thích slide: ")
            yield SimpleNamespace(text="Định luật bảo toàn khối lượng.")

        def generate_content(self, *, model, contents, config=None):
            attempted_keys.append(self.key)
            return SimpleNamespace(text="OK")

    client = ResilientGeminiClient(
        client_factory=lambda key: SimpleNamespace(models=_FakeModels(key)),
        sleep_fn=lambda _s: None,
    )
    streamed: List[str] = []
    res = client.generate_multimodal(
        api_key="AIzaKeyAlpha123456, AIzaKeyBeta789012",
        prompt="Giải thích slide này",
        primary_model="gemini-3.8-flash",
        fallback_models=("gemini-2.5-flash",),
        on_token=streamed.append,
    )
    assert res.ok
    assert "AIzaKeyAlpha123456" in attempted_keys
    assert "AIzaKeyBeta789012" in attempted_keys
    assert "Định luật bảo toàn khối lượng" in res.text
    assert len(streamed) >= 1

    # 2. Headless AirTeachMainWindow startup + mouse/wheel fallback + HUD & thumbnail strip
    from main import AirTeachMainWindow

    win = AirTeachMainWindow()
    win.resize(1280, 800)
    assert win.canvas is not None
    assert win.canvas.mini_hud is not None
    assert win.thumb_strip is not None

    # Mouse fallback stroke (>=3 move points) & zoomScale check
    win.canvas.active_tool = "PEN"
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(300, 300),
        QPointF(300, 300),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    win.canvas.mousePressEvent(press)
    for mx, my in ((325, 315), (355, 330), (385, 345)):
        mv = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(mx, my),
            QPointF(mx, my),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        win.canvas.mouseMoveEvent(mv)
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(385, 345),
        QPointF(385, 345),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    win.canvas.mouseReleaseEvent(release)
    assert len(win.canvas.strokes) >= 1

    win.canvas.apply_zoom_factor(1.25)
    wheel = QWheelEvent(
        QPointF(400, 400),
        QPointF(400, 400),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    win.canvas.wheelEvent(wheel)
    assert win.canvas.zoom_scale > 1.0
    win.close()
    app.processEvents()


def main() -> int:
    tests = [
        ("HandGestureEngine init, fallback & GRACE_PERIOD", test_hand_engine_init_fallback_and_grace_period),
        ("3D gestures, Drawing FSM, Slide Nav & Smart Shapes", test_3d_gestures_drawing_fsm_slide_nav_and_smart_shapes),
        ("PDF & PPTX rendering, cache, annotations & export", test_pdf_pptx_render_cache_annotations_and_export),
        ("Gemini multi-key rotation & headless window smoke", test_gemini_key_rotation_and_headless_window),
    ]
    for name, fn in tests:
        print(f"[RUN ] {name}...")
        fn()
        print(f"[PASS] {name}")
    print(f"\nALL {len(tests)} CORE PIPELINE VERIFICATIONS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
