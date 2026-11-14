#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Verification Suite for AirTeach AI (DNT - Victer):
1. Hybrid Shape Recognition & Auto-Beautification (10 shape classes + Freehand non-snap).
2. 4-Tier Zero-Stray-Stroke Drawing FSM (Schmitt trigger, 105ms tail trim, cancel + 300ms lock, 400ms Undo).
3. Adaptive LAB Lighting Preprocessing + Laplacian Blur Quality Gate.
4. 3D Vector Hand Gesture Classification & 5-Point 112x112 Aligned Face ID.
5. Main App Security, Clean-Slate Whiteboard, Branding SVG & Gemini 3.5 Flash Offline Readiness.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cv_engine.drawing_fsm import (  # noqa: E402
    DrawingHudMode,
    ZeroStrayStrokeDrawingFsm,
)
from cv_engine.face_id import (  # noqa: E402
    ARCFACE_TEMPLATE_112,
    ArcFace512Extractor,
    align_face_5pt_112x112,
    multi_template_cosine_similarity,
)
from cv_engine.hand_gesture import classify_hand_3d_vectors  # noqa: E402
from cv_engine.preprocessing import preprocess_adaptive_frame  # noqa: E402
from cv_engine.shape_recognizer import beautify_stroke  # noqa: E402


def _interp_segment(p0: Tuple[float, float], p1: Tuple[float, float], steps: int = 16) -> List[Tuple[float, float]]:
    return [
        (
            p0[0] + (p1[0] - p0[0]) * (i / steps),
            p0[1] + (p1[1] - p0[1]) * (i / steps),
        )
        for i in range(steps)
    ]


def _polygon_stroke(
    vertices: List[Tuple[float, float]],
    steps_per_edge: int = 18,
    jitter: float = 0.8,
    leave_gap: bool = False,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    n = len(vertices)
    rng = np.random.default_rng(42)
    for idx in range(n):
        a = vertices[idx]
        b = vertices[(idx + 1) % n]
        seg = _interp_segment(a, b, steps_per_edge)
        if leave_gap and idx == n - 1:
            seg = seg[:-2]
        for x, y in seg:
            pts.append((float(x + rng.normal(0.0, jitter)), float(y + rng.normal(0.0, jitter))))
    if not leave_gap:
        pts.append(pts[0])
    return pts


def _make_3d_hand_landmarks(pose: str, pitch_deg: float = 0.0, invert: bool = False) -> List[List[float]]:
    base = [
        [0.50, 0.72, 0.00],  # 0 wrist
        [0.44, 0.67, -0.02],  # 1 thumb_cmc
        [0.40, 0.61, -0.03],  # 2 thumb_mcp
        [0.37, 0.56, -0.03],  # 3 thumb_ip
        [0.35, 0.51, -0.04],  # 4 thumb_tip
        [0.45, 0.54, 0.00],  # 5 index_mcp
        [0.44, 0.44, -0.01],  # 6 index_pip
        [0.44, 0.36, -0.01],  # 7 index_dip
        [0.44, 0.28, -0.01],  # 8 index_tip
        [0.50, 0.53, 0.00],  # 9 middle_mcp
        [0.50, 0.42, 0.00],  # 10 middle_pip
        [0.50, 0.34, 0.00],  # 11 middle_dip
        [0.50, 0.26, 0.00],  # 12 middle_tip
        [0.55, 0.54, 0.01],  # 13 ring_mcp
        [0.56, 0.44, 0.01],  # 14 ring_pip
        [0.56, 0.36, 0.01],  # 15 ring_dip
        [0.56, 0.29, 0.01],  # 16 ring_tip
        [0.60, 0.56, 0.02],  # 17 pinky_mcp
        [0.61, 0.47, 0.02],  # 18 pinky_pip
        [0.62, 0.41, 0.02],  # 19 pinky_dip
        [0.63, 0.35, 0.02],  # 20 pinky_tip
    ]

    def fold_finger(idxs: Tuple[int, int, int, int]) -> None:
        mcp, pip, dip, tip = idxs
        mx, my, mz = base[mcp]
        base[pip] = [mx, my - 0.02, mz - 0.03]
        base[dip] = [mx, my + 0.03, mz - 0.04]
        base[tip] = [mx, my + 0.06, mz - 0.03]

    if pose == "PINCH_DRAW":
        fold_finger((9, 10, 11, 12))
        fold_finger((13, 14, 15, 16))
        fold_finger((17, 18, 19, 20))
        base[4] = [0.46, 0.34, -0.02]
        base[8] = [0.465, 0.335, -0.02]
    elif pose == "INDEX_HOVER":
        fold_finger((9, 10, 11, 12))
        fold_finger((13, 14, 15, 16))
        fold_finger((17, 18, 19, 20))
    elif pose == "TWO_FINGERS":
        fold_finger((13, 14, 15, 16))
        fold_finger((17, 18, 19, 20))
    elif pose == "THREE_FINGERS":
        fold_finger((17, 18, 19, 20))
    elif pose == "FIST":
        for idxs in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            fold_finger(idxs)
        base[4] = [0.51, 0.56, -0.03]

    if abs(pitch_deg) > 1e-3:
        rad = math.radians(pitch_deg)
        c, s = math.cos(rad), math.sin(rad)
        for lm in base:
            dy = lm[1] - 0.5
            dz = lm[2]
            lm[1] = 0.5 + dy * c - dz * s
            lm[2] = dy * s + dz * c

    if invert:
        for lm in base:
            lm[0] = 1.0 - lm[0]
            lm[1] = 1.0 - lm[1]

    return base


# =============================================================================
# 1. HYBRID SHAPE RECOGNITION & AUTO-BEAUTIFICATION TESTS (ALL 10 CLASSES)
# =============================================================================
def test_shape_line_and_angle_snap() -> None:
    # Slightly slanted (~2.5 deg) stroke should snap cleanly to 0 deg horizontal LINE
    pts = [(100.0 + i * 10.0, 200.0 + i * 0.4 + math.sin(i) * 0.4) for i in range(30)]
    res = beautify_stroke(pts)
    assert res.shape_type == "LINE"
    assert res.confidence >= 0.78
    assert abs(res.beautified_points[-1][1] - res.beautified_points[0][1]) < 1e-3
    assert len(res.raw_points) == len(pts)


def test_shape_arrow() -> None:
    shaft = _interp_segment((80.0, 220.0), (340.0, 220.0), steps=28)
    wing1 = _interp_segment((340.0, 220.0), (305.0, 194.0), steps=10)
    back = _interp_segment((305.0, 194.0), (340.0, 220.0), steps=8)
    wing2 = _interp_segment((340.0, 220.0), (305.0, 246.0), steps=10)
    res = beautify_stroke(shaft + wing1 + back + wing2)
    assert res.shape_type == "ARROW"
    assert res.confidence >= 0.78
    assert len(res.beautified_points) >= 5


def test_shape_circle() -> None:
    pts = [
        (300.0 + (90.0 + 1.8 * math.sin(k * 0.25)) * math.cos(t), 240.0 + (90.0 - 1.5 * math.cos(k * 0.3)) * math.sin(t))
        for k, t in enumerate(np.linspace(0.0, 1.93 * math.pi, 72))
    ]
    res = beautify_stroke(pts)
    assert res.shape_type == "CIRCLE"
    assert res.confidence >= 0.78
    assert res.is_closed


def test_shape_rotated_ellipse() -> None:
    angle = math.radians(28.0)
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for t in np.linspace(0.0, 2.0 * math.pi, 80, endpoint=False):
        lx = 135.0 * math.cos(t)
        ly = 58.0 * math.sin(t)
        pts.append((320.0 + lx * ca - ly * sa, 260.0 + lx * sa + ly * ca))
    res = beautify_stroke(pts)
    assert res.shape_type == "ELLIPSE"
    assert res.confidence >= 0.78
    assert res.is_closed


def test_shape_triangle() -> None:
    tri = _polygon_stroke([(200.0, 100.0), (330.0, 320.0), (70.0, 320.0)], jitter=0.7, leave_gap=True)
    res = beautify_stroke(tri)
    assert res.shape_type == "TRIANGLE"
    assert res.confidence >= 0.78
    assert len(res.beautified_points) == 4


def test_shape_square_and_rotated_rectangle() -> None:
    sq = _polygon_stroke([(120.0, 120.0), (280.0, 120.0), (280.0, 280.0), (120.0, 280.0)], jitter=0.6)
    res_sq = beautify_stroke(sq)
    assert res_sq.shape_type == "SQUARE"
    assert len(res_sq.beautified_points) == 5

    # Rotated 25 deg rectangle (aspect ratio 2:1)
    th = math.radians(25.0)
    ca, sa = math.cos(th), math.sin(th)
    rect_verts = []
    for lx, ly in [(-120.0, -55.0), (120.0, -55.0), (120.0, 55.0), (-120.0, 55.0)]:
        rect_verts.append((350.0 + lx * ca - ly * sa, 260.0 + lx * sa + ly * ca))
    rect = _polygon_stroke(rect_verts, jitter=0.6)
    res_rect = beautify_stroke(rect)
    assert res_rect.shape_type == "RECTANGLE"
    assert res_rect.confidence >= 0.78


def test_shape_rhombus_and_parallelogram() -> None:
    rhombus = _polygon_stroke([(250.0, 95.0), (335.0, 240.0), (250.0, 385.0), (165.0, 240.0)], jitter=0.5)
    res_rh = beautify_stroke(rhombus)
    assert res_rh.shape_type == "RHOMBUS"
    assert res_rh.confidence >= 0.78

    para = _polygon_stroke([(140.0, 150.0), (380.0, 150.0), (320.0, 290.0), (80.0, 290.0)], jitter=0.5)
    res_p = beautify_stroke(para)
    assert res_p.shape_type == "PARALLELOGRAM"
    assert res_p.confidence >= 0.78


def test_freehand_handwriting_or_sine_wave_never_mis_snaps() -> None:
    pts = [(60.0 + x * 4.0, 250.0 + 48.0 * math.sin(x * 0.28) + 18.0 * math.cos(x * 0.14)) for x in range(85)]
    res = beautify_stroke(pts, confidence_gate=0.78)
    assert res.shape_type == "FREEHAND"
    assert not res.is_geometric
    assert len(res.beautified_points) > len(pts)


# =============================================================================
# 2. 4-TIER ZERO-STRAY-STROKE DRAWING FSM TESTS
# =============================================================================
def test_4tier_fsm_schmitt_trigger_and_tail_trimming() -> None:
    fsm = ZeroStrayStrokeDrawingFsm()

    # Frames 1 & 2 (<0.28) must remain in HOVER
    for idx in range(2):
        u = fsm.step(idx * 16.0, True, (100.0 + idx * 4.0, 200.0), 0.22, "PINCH")
        assert u.hud_mode == DrawingHudMode.HOVER
        assert u.action == "NONE"

    # Frame 3 enters DRAWING
    u3 = fsm.step(32.0, True, (112.0, 200.0), 0.22, "PINCH")
    assert u3.hud_mode == DrawingHudMode.DRAWING
    assert u3.action == "BEGIN_STROKE"

    # Hysteresis band [0.28, 0.40] keeps drawing
    for step_idx, t_ms in enumerate([64.0, 96.0, 128.0, 160.0]):
        uh = fsm.step(t_ms, True, (130.0 + step_idx * 15.0, 200.0), 0.35, "PINCH")
        assert uh.hud_mode == DrawingHudMode.DRAWING
        assert uh.action == "APPEND_POINT"

    # Simulate upward tail hook during final ~90ms before releasing
    for k, t_ms in enumerate([195.0, 225.0, 255.0]):
        fsm.step(t_ms, True, (190.0 + k * 8.0, 200.0 - (k + 1) * 35.0), 0.37, "PINCH")

    # Release (>0.40) commits trimmed stroke without hook points
    rel = fsm.step(275.0, True, (220.0, 60.0), 0.52, "TWO_FINGERS")
    assert rel.action == "COMMIT_STROKE"
    assert rel.hud_mode == DrawingHudMode.HOVER
    assert len(rel.committed_points) >= 2
    assert max(p[1] for p in rel.committed_points) <= 200.5
    assert min(p[1] for p in rel.committed_points) >= 199.5


def test_4tier_fsm_cancel_open_palm_fist_esc_cooldown_and_undo() -> None:
    fsm = ZeroStrayStrokeDrawingFsm()
    for idx in range(5):
        fsm.step(idx * 25.0, True, (100.0 + idx * 10.0, 150.0), 0.21, "PINCH")
    assert fsm.hud_mode == DrawingHudMode.DRAWING

    # Hold OPEN_PALM < 150ms should not immediately cancel or leave stray commit
    u_hold = fsm.step(140.0, True, (150.0, 150.0), 0.72, "OPEN_PALM")
    assert u_hold.action != "CANCEL_STROKE"

    # Hold OPEN_PALM >= 150ms cancels and locks pen for 300ms
    u_cancel = fsm.step(305.0, True, (150.0, 150.0), 0.72, "OPEN_PALM")
    assert u_cancel.action == "CANCEL_STROKE"
    assert u_cancel.hud_mode == DrawingHudMode.LOCKED_COOLDOWN
    assert u_cancel.cooldown_remaining_ms >= 295.0

    # Pinching inside 300ms lock must be rejected
    for t_ms in (350.0, 420.0, 500.0, 580.0):
        u_blk = fsm.step(t_ms, True, (160.0, 150.0), 0.19, "PINCH")
        assert u_blk.hud_mode == DrawingHudMode.LOCKED_COOLDOWN
        assert u_blk.action != "BEGIN_STROKE"

    # After 300ms cooldown expires, pinching >=3 frames begins normally; Esc cancels immediately
    for idx in range(3):
        fsm.step(640.0 + idx * 20.0, True, (200.0 + idx * 8.0, 160.0), 0.20, "PINCH")
    assert fsm.hud_mode == DrawingHudMode.DRAWING

    u_esc = fsm.step(700.0, True, (220.0, 160.0), 0.20, "PINCH", esc_pressed=True)
    assert u_esc.action == "CANCEL_STROKE"
    assert u_esc.cancel_reason == "ESC"
    assert u_esc.hud_mode == DrawingHudMode.LOCKED_COOLDOWN

    # 400ms Undo cooldown
    u1 = fsm.step(1050.0, False, None, None, "IDLE", undo_requested=True)
    assert u1.action == "UNDO"
    u2 = fsm.step(1250.0, False, None, None, "IDLE", undo_requested=True)
    assert u2.action == "NONE"
    u3 = fsm.step(1470.0, False, None, None, "IDLE", undo_requested=True)
    assert u3.action == "UNDO"


# =============================================================================
# 3. LIGHTING PREPROCESSING + BLUR GATE + 3D GESTURES + 5-PT ALIGNED FACE ID
# =============================================================================
def test_adaptive_lighting_and_laplacian_blur_gate() -> None:
    dark = np.full((480, 640, 3), 24, dtype=np.uint8)
    cv2.rectangle(dark, (180, 120), (420, 360), (48, 42, 44), -1)
    cv2.putText(dark, "DNT-VICTER", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
    out_dark, diag_dark = preprocess_adaptive_frame(dark, is_rgb=False, quality_gate_min=22.0)
    assert out_dark.shape == (480, 640, 3)
    assert diag_dark.lighting_mode == "LOW_LIGHT"
    assert diag_dark.gamma_applied < 1.0

    glare = np.full((480, 640, 3), 252, dtype=np.uint8)
    cv2.circle(glare, (320, 240), 90, (120, 110, 100), -1)
    _, diag_glare = preprocess_adaptive_frame(glare, is_rgb=False, quality_gate_min=22.0)
    assert diag_glare.p95 > 235.0
    assert diag_glare.lighting_mode == "GLARE"

    blurred = cv2.GaussianBlur(out_dark, (21, 21), 8.0)
    _, diag_blur = preprocess_adaptive_frame(blurred, is_rgb=False, quality_gate_min=35.0)
    assert not diag_blur.quality_gate_passed


def test_3d_vector_gestures_rotation_and_hysteresis() -> None:
    res_pinch = classify_hand_3d_vectors(_make_3d_hand_landmarks("PINCH_DRAW", pitch_deg=42.0), was_pinched=False)
    assert res_pinch["pose_tag"] == "PINCH"
    assert float(res_pinch["pinch_ratio"]) < 0.28

    res_draw_idx = classify_hand_3d_vectors(_make_3d_hand_landmarks("INDEX_HOVER", pitch_deg=42.0), was_pinched=False)
    assert res_draw_idx["gesture"] == "DRAW"
    assert res_draw_idx["pose_tag"] == "INDEX"

    res_inv = classify_hand_3d_vectors(_make_3d_hand_landmarks("PINCH_DRAW", invert=True), was_pinched=False)
    assert res_inv["pose_tag"] == "PINCH"
    assert float(res_inv["pinch_ratio"]) < 0.28

    res_palm = classify_hand_3d_vectors(_make_3d_hand_landmarks("OPEN_PALM"), was_pinched=True)
    assert res_palm["pose_tag"] == "OPEN_PALM"
    assert res_palm["gesture"] == "ERASE"

    res_fist = classify_hand_3d_vectors(_make_3d_hand_landmarks("FIST"), was_pinched=True)
    assert res_fist["pose_tag"] == "FIST"


def test_5pt_aligned_arcface_multi_template_matching() -> None:
    img_a = np.full((240, 240, 3), 130, dtype=np.uint8)
    cv2.circle(img_a, (120, 120), 62, (175, 150, 140), -1)
    cv2.circle(img_a, (98, 106), 7, (35, 30, 30), -1)
    cv2.circle(img_a, (142, 106), 7, (35, 30, 30), -1)
    cv2.line(img_a, (120, 108), (120, 132), (80, 65, 65), 3)
    cv2.line(img_a, (104, 150), (136, 150), (90, 55, 55), 3)

    img_b = np.full((240, 240, 3), 115, dtype=np.uint8)
    cv2.rectangle(img_b, (68, 56), (172, 182), (95, 115, 165), -1)
    cv2.circle(img_b, (98, 106), 11, (20, 20, 20), -1)
    cv2.circle(img_b, (142, 106), 11, (20, 20, 20), -1)
    cv2.rectangle(img_b, (100, 144), (140, 162), (40, 40, 90), -1)

    pts_5 = np.array(
        [[98, 106], [142, 106], [120, 126], [104, 150], [136, 150]],
        dtype=np.float32,
    )
    aligned_a = align_face_5pt_112x112(img_a, pts_5)
    assert aligned_a.shape == (112, 112, 3)

    extractor = ArcFace512Extractor(models_dir=ROOT_DIR / "models")
    _, _, emb_a1 = extractor.extract_from_5pt_aligned(img_a, pts_5, (60, 60, 120, 120))
    _, _, emb_a2 = extractor.extract_from_5pt_aligned(img_a, pts_5 + 0.6, (60, 60, 120, 120))
    _, _, emb_b = extractor.extract_from_5pt_aligned(img_b, pts_5, (60, 60, 120, 120))

    same_sim = multi_template_cosine_similarity(emb_a2, [emb_a1, emb_a1], mode="top2")
    diff_sim = multi_template_cosine_similarity(emb_b, [emb_a1], mode="top2")
    assert same_sim >= 0.85
    assert diff_sim < same_sim - 0.15
    assert ARCFACE_TEMPLATE_112.shape == (5, 2)


# =============================================================================
# 4. MAIN MODULE, SVG LOGO, SECURITY, CLEAN TUTORIAL & GEMINI OFFLINE CHECKS
# =============================================================================
def test_main_security_brand_svg_and_clean_whiteboard() -> None:
    import main  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    _ = QApplication.instance() or QApplication([])

    svg_path = ROOT_DIR / "assets" / "branding" / "airteach_logo.svg"
    assert svg_path.exists()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "DNT - Victer" in svg_text
    ET.fromstring(svg_text)

    h = main.hash_pin_pbkdf2("246810")
    assert h.startswith("pbkdf2_sha256$")
    assert main.verify_pin_hash("246810", h)
    assert not main.verify_pin_hash("000000", h)

    canvas = main.SmartboardCanvas()
    assert canvas.smart_shape_enabled is True
    assert len(canvas.slides) == 1
    assert canvas.slides[0].get("is_tutorial") is True
    assert "DNT - Victer" in canvas.slides[0]["title"]

    canvas.dismiss_tutorial_slides()
    assert len(canvas.slides) == 0

    # Gemini offline check executes gracefully without network/API key
    msgs: List[str] = []
    worker = main.GeminiVisionWorker(api_key="", prompt="Test offline board", png_bytes=None, structured_json=True)
    worker.finished_text.connect(msgs.append)
    worker.run()
    assert len(msgs) == 1
    assert "Offline" in msgs[0] or "GEMINI_API_KEY" in msgs[0]
