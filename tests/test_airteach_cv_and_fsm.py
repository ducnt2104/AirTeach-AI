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
from typing import Any, List, Tuple

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


# =============================================================================
# 5. UPGRADED VISION, STROKE COMPLETION, 2-STEP UNDO & RESILIENT GEMINI CLIENT
# =============================================================================
def test_backlit_and_normal_fastpath_preprocessing() -> None:
    # Backlit scene: bright window glare (p95 > 235) + dark foreground shadows (p5 < 48)
    backlit = np.full((180, 240, 3), 24, dtype=np.uint8)
    backlit[:, 140:] = 250
    cv2.putText(backlit, "A", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (55, 55, 55), 3)
    enh_b, diag_b = preprocess_adaptive_frame(
        backlit, is_rgb=False, quality_gate_min=5.0, auto_lightweight_normal=True
    )
    assert diag_b.lighting_mode == "BACKLIT"
    assert diag_b.gamma_applied < 1.0
    assert float(np.mean(enh_b[:, :120])) > float(np.mean(backlit[:, :120]))

    # Balanced lighting scene: fast path returns cleanly when auto_lightweight_normal=True
    normal = np.full((160, 220, 3), 128, dtype=np.uint8)
    normal[:, :60] = 60
    normal[:, 160:] = 200
    cv2.line(normal, (70, 30), (150, 140), (185, 185, 185), 3)
    enh_n, diag_n = preprocess_adaptive_frame(
        normal, is_rgb=False, quality_gate_min=5.0, auto_lightweight_normal=True
    )
    assert diag_n.lighting_mode == "NORMAL"
    assert diag_n.gamma_applied == 1.0
    assert enh_n.shape == normal.shape


def test_one_euro_tip_smoothing_and_deadzone_fsm() -> None:
    from cv_engine.hand_gesture import GestureFsmEngine, GestureFsmState  # noqa: E402

    fsm = GestureFsmEngine(
        min_hold_frames=4,
        min_hold_sec=0.120,
        grace_period_sec=0.150,
        frame_size=(640, 480),
        deadzone_px=3.0,
    )
    base_draw = _make_3d_hand_landmarks("PINCH_DRAW")

    # 1. Brief 1-frame noisy flicker to OPEN_PALM/HOVER must NOT promote from IDLE to ACTIVE
    hover_lm = _make_3d_hand_landmarks("INDEX_HOVER")
    snap_flicker = fsm.update(hover_lm, t_sec=0.010)
    assert snap_flicker.fsm_state != GestureFsmState.ACTIVE

    # 2. Holding PINCH across >= 4 frames and >= 120ms promotes to ACTIVE; let tip converge
    t = 0.050
    for _ in range(12):
        snap = fsm.update(base_draw, t_sec=t)
        t += 0.035
    assert snap.fsm_state == GestureFsmState.ACTIVE
    c0 = snap.cursor_norm
    assert c0 is not None

    # 3. Stationary micro-tremor (< ~2px) is suppressed by normalized deadzone
    jitter_lm = [list(pt) for pt in base_draw]
    jitter_lm[8][0] += 0.0005
    jitter_lm[8][1] -= 0.0004
    snap_jitter = fsm.update(jitter_lm, t_sec=t)
    assert snap_jitter.cursor_norm == c0


def test_drawing_fsm_grace_period_gap_bridging_and_max_jump_guard() -> None:
    fsm = ZeroStrayStrokeDrawingFsm()

    # Enter DRAWING via 3 pinch frames
    for idx in range(3):
        upd = fsm.step(
            idx * 20.0,
            hand_present=True,
            cursor_xy=(80.0 + idx * 8.0, 120.0),
            pinch_ratio=0.19,
            pose_tag="PINCH",
        )
    assert upd.hud_mode == DrawingHudMode.DRAWING
    assert upd.action == "BEGIN_STROKE"

    # Brief occlusion (<= 150ms Grace Period) retains DRAWING via HOLD_GRACE
    grace = fsm.step(
        110.0,
        hand_present=False,
        cursor_xy=None,
        pinch_ratio=None,
        pose_tag="IDLE",
    )
    assert grace.hud_mode == DrawingHudMode.DRAWING
    assert grace.action == "HOLD_GRACE"
    assert grace.in_grace_period is True

    # Reacquire within max_jump_distance (<= 260px): velocity bridge fills intermediate points smoothly
    resumed = fsm.step(
        145.0,
        hand_present=True,
        cursor_xy=(216.0, 120.0),
        pinch_ratio=0.21,
        pose_tag="PINCH",
    )
    assert resumed.hud_mode == DrawingHudMode.DRAWING
    assert resumed.action == "APPEND_POINT"
    assert resumed.gap_bridged is True
    assert len(resumed.active_points) >= 12

    # Release -> tail cutoff trims last ~105ms but keeps main bridged body
    fsm.step(300.0, hand_present=True, cursor_xy=(225.0, 120.0), pinch_ratio=0.21, pose_tag="PINCH")
    fsm.step(340.0, hand_present=True, cursor_xy=(230.0, 190.0), pinch_ratio=0.22, pose_tag="PINCH")
    commit = fsm.step(360.0, hand_present=True, cursor_xy=(235.0, 210.0), pinch_ratio=0.50, pose_tag="HOVER")
    assert commit.action == "COMMIT_STROKE"
    assert len(commit.committed_points) >= 6
    assert max(y for (_, y) in commit.committed_points) < 160.0

    # Large jump (> 260px) while drawing must split cleanly via JUMP_GUARD instead of crossing screen
    for idx in range(3):
        fsm.step(500.0 + idx * 20.0, hand_present=True, cursor_xy=(60.0 + idx * 5.0, 80.0), pinch_ratio=0.20, pose_tag="PINCH")
    jump = fsm.step(600.0, hand_present=True, cursor_xy=(760.0, 520.0), pinch_ratio=0.20, pose_tag="PINCH")
    assert jump.hud_mode == DrawingHudMode.HOVER
    assert jump.cancel_reason == "JUMP_GUARD"


def test_smartboard_canvas_2step_ctrl_z_undo_and_snapshot_compression() -> None:
    import main  # noqa: E402
    from PySide6.QtCore import QPointF  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402
    from cv_engine.gemini_client import compress_snapshot_png  # noqa: E402

    _ = QApplication.instance() or QApplication([])
    canvas = main.SmartboardCanvas()
    canvas.smart_shape_enabled = True
    canvas.hold_to_snap_350ms = False

    # Draw a noisy circle & finalize -> snaps to geometric CIRCLE
    cx, cy, r = 240.0, 220.0, 65.0
    pts = []
    for i in range(42):
        th = 2.0 * math.pi * i / 41.0
        jitter = 1.5 * math.sin(th * 5.0)
        pts.append(QPointF(cx + (r + jitter) * math.cos(th), cy + (r + jitter) * math.sin(th)))
    canvas.current_stroke = pts
    canvas._finalize_current_stroke(now_ms=1000.0)
    assert len(canvas.strokes) == 1
    assert canvas.strokes[0]["is_beautified"] is True
    assert canvas.strokes[0]["shape_type"] == "CIRCLE"

    # Ctrl+Z Step 1: reverts geometric shape back to smoothed hand-drawn stroke
    canvas.undo()
    assert len(canvas.strokes) == 1
    assert canvas.strokes[0]["is_beautified"] is False
    assert canvas.strokes[0]["shape_type"] == "FREEHAND"

    # Redo restores snapped geometric shape cleanly
    canvas.redo()
    assert len(canvas.strokes) == 1
    assert canvas.strokes[0]["is_beautified"] is True
    assert canvas.strokes[0]["shape_type"] == "CIRCLE"

    # Ctrl+Z x2 deletes the stroke completely
    canvas.undo()
    assert canvas.strokes[0]["is_beautified"] is False
    canvas.undo()
    assert len(canvas.strokes) == 0

    # Snapshot compression caps >1024px PNG down proportionally
    large_img = np.full((1200, 1800, 3), 240, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", large_img)
    assert ok
    compressed = compress_snapshot_png(encoded.tobytes(), max_dim=1024)
    assert compressed is not None
    dec = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert dec is not None
    assert max(dec.shape[:2]) == 1024


def test_gemini_client_layered_keys_masking_health_check_backoff_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cv_engine.gemini_client import (  # noqa: E402
        ApiConnectionStatus,
        ResilientGeminiClient,
        clear_ui_api_key,
        mask_api_key,
        resolve_api_key_layered,
        save_ui_api_settings,
    )

    assert mask_api_key("AIzaSyD1234567890abcd") == "AIza••••••abcd"
    assert mask_api_key("") == ""

    settings_file = tmp_path / "api_settings.json"
    env_file = tmp_path / ".env"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    # 1. Test Connection before any key is configured -> UNCONFIGURED
    res_unconf = ResilientGeminiClient().test_connection(api_key="")
    assert res_unconf.status == ApiConnectionStatus.UNCONFIGURED

    env_file.write_text("GEMINI_API_KEY=AIzaDotEnvSecret1234\n", encoding="utf-8")

    # Layer (2) .env used when no UI key saved
    res_env = resolve_api_key_layered(settings_path=settings_file, env_path=env_file)
    assert res_env.source == "env"
    assert res_env.key == "AIzaDotEnvSecret1234"
    assert res_env.masked == "AIza••••••1234"

    # Layer (1) UI key overrides .env
    save_ui_api_settings(ui_api_key="AIzaUiOverrideKey9876", settings_path=settings_file)
    res_ui = resolve_api_key_layered(settings_path=settings_file, env_path=env_file)
    assert res_ui.source == "ui"
    assert res_ui.key == "AIzaUiOverrideKey9876"

    # Clear UI key falls back cleanly to .env
    clear_ui_api_key(settings_path=settings_file)
    res_after_del = resolve_api_key_layered(settings_path=settings_file, env_path=env_file)
    assert res_after_del.source == "env"

    class FakeResp:
        def __init__(self, text: str) -> None:
            self.text = text

    # 2. Live health check with 404 fallback -> primary fails 404, secondary returns OK
    calls: List[Tuple[str, Any]] = []

    class FakeModels404ThenOk:
        def generate_content(self, **kwargs):
            calls.append((kwargs["model"], kwargs.get("config")))
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("404 NOT_FOUND: model not found")
            return FakeResp("OK")

    class FakeClient404:
        models = FakeModels404ThenOk()

    hc_ok = ResilientGeminiClient(client_factory=lambda _k: FakeClient404()).test_connection(
        api_key="AIzaFakeKey1234567890",
        primary_model="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash",),
    )
    assert hc_ok.status == ApiConnectionStatus.CONNECTED
    assert hc_ok.model_used == "gemini-3.5-flash"

    # 3. HTTP 401/403 Auth error -> fails fast without wasting retries
    auth_calls = 0

    class FakeModels401:
        def generate_content(self, **kwargs):
            nonlocal auth_calls
            auth_calls += 1
            raise RuntimeError("403 PERMISSION_DENIED: API_KEY_INVALID")

    class FakeClient401:
        models = FakeModels401()

    r_auth = ResilientGeminiClient(client_factory=lambda _k: FakeClient401()).generate_multimodal(
        api_key="AIzaBadKey000000",
        prompt="Hi",
        primary_model="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash",),
    )
    assert not r_auth.ok
    assert auth_calls == 1
    assert "API Key" in r_auth.error_message_vi

    # 4. HTTP 429 Quota -> Exponential Backoff [1.0, 2.0, 4.0] then succeeds cleanly
    sleeps: List[float] = []
    attempt_count = 0

    class FakeModels429ThenOk:
        def generate_content(self, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")
            return FakeResp("Đáp án x = 2")

    class FakeClient429:
        models = FakeModels429ThenOk()

    r_429 = ResilientGeminiClient(
        client_factory=lambda _k: FakeClient429(),
        sleep_fn=sleeps.append,
    ).generate_multimodal(
        api_key="AIzaQuotaTest1234",
        prompt="Giải phương trình",
        primary_model="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash",),
    )
    assert r_429.ok
    assert r_429.text == "Đáp án x = 2"
    assert sleeps == [1.0, 2.0, 4.0]

    # 5. Network failure never crashes
    class FakeModelsNetErr:
        def generate_content(self, **kwargs):
            raise ConnectionError("NameResolutionError: Failed to resolve generativelanguage.googleapis.com")

    class FakeClientNet:
        models = FakeModelsNetErr()

    r_net = ResilientGeminiClient(
        client_factory=lambda _k: FakeClientNet(),
        sleep_fn=lambda _s: None,
    ).generate_multimodal(
        api_key="AIzaOfflineTest1234",
        prompt="Test offline safety",
        primary_model="gemini-3.8-flash",
        fallback_models=(),
    )
    assert not r_net.ok
    assert "mạng" in r_net.error_message_vi.lower()


# =============================================================================
# 6. MASTER UPGRADE: SANITIZER, AGENTIC WHITEBOARD, LASSO-TO-AI, LOCAL AI & PERF
# =============================================================================
def test_output_sanitizer_streaming_and_math_rendering() -> None:
    from cv_engine.output_sanitizer import (  # noqa: E402
        OutputSanitizer,
        StreamingSanitizer,
        arrow_svg_data_uri,
        render_latex_to_png_bytes,
    )

    dirty = r"Bước 1: $$$2x + 1 = 3$$$ <arrow> $x = \frac{3 - 1}{2} = 1$ <check> $x^2 + \sqrt{9} \geq 3$"
    plain = OutputSanitizer.sanitize_plain_text(dirty)
    assert "$" not in plain
    assert "$$$" not in plain
    assert "<arrow>" not in plain
    assert "<check>" not in plain
    assert "→" in plain
    assert "✓" in plain
    assert "√(9)" in plain
    assert "x²" in plain

    html_out = OutputSanitizer.to_rich_html(dirty, dark_mode=True)
    assert "<arrow>" not in html_out
    assert "$$$" not in html_out
    assert "data:image/svg+xml;base64," in html_out

    svg_uri = arrow_svg_data_uri("right")
    assert svg_uri.startswith("data:image/svg+xml;base64,")

    png = render_latex_to_png_bytes(r"\frac{-b \pm \sqrt{\Delta}}{2a}")
    if png is not None:
        assert png.startswith(b"\x89PNG")

    # Streaming chunk buffer holds broken delimiters/tags until complete
    st = StreamingSanitizer()
    p1 = st.feed("Kết quả <arr")
    assert "<arr" not in p1
    p2 = st.feed("ow> $2x")
    assert "$" not in p2
    assert "→" in p2
    p3 = st.feed(" + 1 = 3$")
    tail = st.flush()
    combined = p1 + p2 + p3 + tail
    assert "$" not in combined and "<arrow>" not in combined


def test_whiteboard_agent_fast_intent_and_free_region_canvas() -> None:
    import main  # noqa: E402
    import time as pytime  # noqa: E402
    from PySide6.QtCore import QPointF  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402
    from cv_engine.whiteboard_agent import WhiteboardAgent  # noqa: E402

    _ = QApplication.instance() or QApplication([])
    t0 = pytime.perf_counter()
    act = WhiteboardAgent.parse_fast_intent("ghi lên bảng hộ tôi 2x + 1 = 3 đi")
    elapsed_ms = (pytime.perf_counter() - t0) * 1000.0
    assert act is not None
    assert elapsed_ms < 5.0
    assert act.tool == "write_on_board"
    assert act.args["content"] == "2x + 1 = 3"

    act_py = WhiteboardAgent.parse_fast_intent("viết công thức định lý Pytago lên bảng")
    assert act_py is not None and "a^2 + b^2 = c^2" in act_py.args["content"]

    act_axis = WhiteboardAgent.parse_fast_intent("vẽ trục tọa độ")
    assert act_axis is not None and act_axis.tool == "draw_shape" and act_axis.args["type"] == "axis"

    canvas = main.SmartboardCanvas()
    canvas.resize(1100, 720)
    canvas.slide_visible = False
    # Place existing ink stroke occupying top-left zone
    canvas.strokes.append({
        "raw_points": [QPointF(20.0, 50.0), QPointF(360.0, 260.0)],
        "points": [QPointF(20.0, 50.0), QPointF(360.0, 260.0)],
        "color": main.QColor("#00F5D4"),
        "thickness": 4,
    })

    report = canvas.execute_whiteboard_actions([act])
    assert "2x + 1 = 3" in report
    assert len(canvas.annotations) == 1
    ann = canvas.annotations[0]
    # Non-overlapping guarantee: must not collide with (20..360, 50..260)
    assert ann["x"] >= 350.0 or ann["y"] >= 250.0

    # AI Draw shape vector + undo_last_ai_action
    canvas.execute_whiteboard_actions([act_axis])
    assert len(canvas.strokes) > 1
    assert canvas.undo_last_ai_action() is True
    assert len(canvas.strokes) == 1


def test_lasso_ai_service_bbox_enhance_and_beautify_replace() -> None:
    import main  # noqa: E402
    from PySide6.QtCore import QPointF  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402
    from cv_engine.lasso_ai_service import (  # noqa: E402
        LassoQuickAction,
        build_lasso_prompt,
        compute_lasso_bounding_box,
        enhance_lasso_crop_png,
        stroke_inside_lasso,
    )

    _ = QApplication.instance() or QApplication([])
    poly = [(95.0, 95.0), (260.0, 95.0), (260.0, 190.0), (95.0, 190.0), (95.0, 100.0)]
    reg = compute_lasso_bounding_box(poly, (1000.0, 700.0), padding=18.0)
    assert reg is not None
    assert reg.bbox_x == 77.0 and reg.bbox_y == 77.0
    assert stroke_inside_lasso([(130.0, 140.0), (190.0, 140.0)], reg) is True
    assert stroke_inside_lasso([(600.0, 500.0), (650.0, 520.0)], reg) is False
    assert "whiteboard_actions" in build_lasso_prompt(LassoQuickAction.WRITE_SOLUTION_TO_BOARD)

    sample = np.full((120, 240, 3), 20, dtype=np.uint8)
    cv2.putText(sample, "2x+1=3", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (230, 245, 245), 3)
    ok, raw_enc = cv2.imencode(".png", sample)
    assert ok
    boosted = enhance_lasso_crop_png(raw_enc.tobytes(), max_dim=1024)
    assert boosted is not None and boosted.startswith(b"\x89PNG")

    canvas = main.SmartboardCanvas()
    canvas.strokes.append({
        "raw_points": [QPointF(120.0, 130.0), QPointF(210.0, 130.0)],
        "points": [QPointF(120.0, 130.0), QPointF(210.0, 130.0)],
        "color": main.QColor("#00F5D4"),
        "thickness": 4,
    })
    canvas.strokes.append({
        "raw_points": [QPointF(550.0, 480.0), QPointF(650.0, 480.0)],
        "points": [QPointF(550.0, 480.0), QPointF(650.0, 480.0)],
        "color": main.QColor("#00F5D4"),
        "thickness": 4,
    })
    canvas.replace_lasso_with_formula(reg, "$$2x + 1 = 3$$")
    # Inner handwritten stroke replaced by clean annotation, outer stroke preserved!
    assert len(canvas.strokes) == 1
    assert len(canvas.annotations) == 1
    assert "2x + 1 = 3" in canvas.annotations[0]["text"]


def test_local_ai_manager_mock_discovery_sse_cache_and_hybrid(tmp_path: Path) -> None:
    import http.server  # noqa: E402
    import json as _json  # noqa: E402
    import threading  # noqa: E402
    from cv_engine.local_ai_manager import (  # noqa: E402
        AiRoutingMode,
        LocalAIManager,
        LocalResponseCache,
    )

    class MockOpenAILocalHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/v1/models":
                body = _json.dumps({"data": [{"id": "qwen3:4b-q4_K_M"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if self.path == "/v1/chat/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = ["Ta có $$$2x ", "+ 1 = 3$$$ <arr", "ow> $x = 1$"]
                for ch in chunks:
                    line = f"data: {_json.dumps({'choices': [{'delta': {'content': ch}}]})}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_response(404)
                self.end_headers()

    srv = http.server.HTTPServer(("127.0.0.1", 0), MockOpenAILocalHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        cache = LocalResponseCache(db_path=tmp_path / "cache.sqlite")
        mgr = LocalAIManager(
            routing_mode=AiRoutingMode.LOCAL,
            preferred_base_url=f"http://127.0.0.1:{port}",
            cache=cache,
        )
        found = mgr.discover_local_engines(timeout=1.5)
        assert any("qwen3:4b-q4_K_M" in s.models for s in found)
        mgr.select_local_model(f"http://127.0.0.1:{port}", "qwen3:4b-q4_K_M", "LM Studio")

        streamed_chunks: List[str] = []
        r1 = mgr.generate_hybrid("Giải 2x+1=3", on_token=streamed_chunks.append)
        assert r1.ok is True
        assert r1.engine_used == "Local"
        assert "$" not in r1.sanitized_text
        assert "<arrow>" not in r1.sanitized_text
        assert "→" in r1.sanitized_text

        # Second call hits 0ms Local Cache
        r2 = mgr.generate_hybrid("Giải 2x+1=3")
        assert r2.ok is True and r2.from_cache is True and r2.engine_used == "Cache"
    finally:
        srv.shutdown()
        srv.server_close()


def test_performance_pipeline_scaling_and_dirty_cache() -> None:
    from cv_engine.performance_pipeline import (  # noqa: E402
        DirtyCanvasBackingTracker,
        PerformancePipeline,
        compute_cv_tracking_size,
    )

    assert compute_cv_tracking_size(1920, 1080, 640, 480) == (640, 360)
    assert compute_cv_tracking_size(640, 480, 640, 480) == (640, 480)

    tracker = DirtyCanvasBackingTracker()
    key1 = (1280, 720, True, 0)
    assert tracker.needs_rebuild(key1) is True
    tracker.mark_clean(key1)
    assert tracker.needs_rebuild(key1) is False
    tracker.mark_dirty()
    assert tracker.needs_rebuild(key1) is True

    pipe = PerformancePipeline()
    pipe.update_cv_fps(60.0)
    pipe.update_ai_metrics(ai_mode="Local", active_model="qwen3:4b", tokens_per_sec=48.5)
    pill = pipe.telemetry.status_pill_text()
    assert "Local" in pill and "48.5 tok/s" in pill and "60 FPS" in pill


# =============================================================================
# 6. 4-TIER VISION ENGINE UPGRADE TESTS (LIGHTING BYPASS, TEACHER LOCK, ROI 2X, 3D + KALMAN)
# =============================================================================
def test_tier1_adaptive_exposure_n_frame_sampling_and_bypass() -> None:
    from cv_engine.preprocessing import AdaptiveExposureController  # noqa: E402

    ctrl = AdaptiveExposureController(check_interval_n=4)
    # Normal room-lit gradient frame (contrast_span >= 65, mean ~125)
    y_vals = np.linspace(45, 205, 480, dtype=np.uint8)[:, None]
    normal = np.repeat(np.repeat(y_vals, 640, axis=1)[:, :, None], 3, axis=2)

    out0, d0 = ctrl.process(normal, is_rgb=True, enabled=True)
    assert d0.lighting_mode == "NORMAL"
    assert d0.clahe_bypassed is True
    assert d0.sampled_frame is True
    assert out0 is normal  # zero-copy true bypass!

    # Intermediate frame 1..3 must NOT re-sample exposure
    out1, d1 = ctrl.process(normal, is_rgb=True, enabled=True)
    assert d1.clahe_bypassed is True
    assert d1.sampled_frame is False
    assert out1 is normal

    # Step through interval to dark frame on check frame #4
    ctrl.process(normal, is_rgb=True, enabled=True)
    ctrl.process(normal, is_rgb=True, enabled=True)
    dark = np.full((480, 640, 3), 26, dtype=np.uint8)
    out4, d4 = ctrl.process(dark, is_rgb=True, enabled=True)
    assert d4.lighting_mode == "LOW_LIGHT"
    assert d4.clahe_bypassed is False
    assert d4.sampled_frame is True
    assert float(np.mean(out4)) > float(np.mean(dark)) + 15.0


def test_tier2_teacher_lock_anti_interference_and_grace_period() -> None:
    from cv_engine.subject_tracker import PrimaryTeacherLockTracker  # noqa: E402

    tracker = PrimaryTeacherLockTracker(teacher_grace_sec=1.5)
    teacher_face = [0.40, 0.20, 0.16, 0.22]
    passerby_face = [0.76, 0.24, 0.11, 0.15]

    snap = tracker.update(
        face_boxes=[teacher_face, passerby_face],
        now_sec=10.0,
    )
    assert snap.locked_teacher_id is not None
    locked_id = snap.locked_teacher_id
    assert len(snap.ignored_face_bboxes) == 1
    assert snap.mode == "FACE_TORSO_LOCK"

    # Build a synthetic hand near passerby's torso (x=0.80, y=0.45) -> MUST be rejected!
    passerby_hand = [[0.80, 0.48, 0.0]] * 21
    passerby_hand[9] = [0.80, 0.43, 0.0]
    assert tracker.is_hand_owned_by_locked_teacher(passerby_hand, snap) is False

    # Build teacher hand inside teacher reach zone (x=0.35, y=0.50) -> MUST pass!
    teacher_hand = [[0.35, 0.52, 0.0]] * 21
    teacher_hand[9] = [0.35, 0.46, 0.0]
    assert tracker.is_hand_owned_by_locked_teacher(teacher_hand, snap) is True

    # Teacher briefly blocked for 1.1s (< 1.5s grace), only passerby visible -> MUST hold teacher ID!
    snap_grace = tracker.update(
        face_boxes=[passerby_face],
        now_sec=11.1,
    )
    assert snap_grace.locked_teacher_id == locked_id
    assert snap_grace.teacher_in_grace_period is True
    assert snap_grace.mode == "GRACE_HOLD"

    # Proactive Stage-2 Pose Wrist ROIs generated around locked teacher
    rois = tracker.get_proactive_wrist_roi_boxes(snap_grace, roi_expand_factor=2.2)
    assert len(rois) >= 2


def test_tier3_long_range_hand_validation_and_roi_backprojection() -> None:
    from cv_engine.hand_gesture import HandGestureEngine  # noqa: E402

    # Tiny distant hand (~2.5m-3m, 4.5% x 5.0% = 0.00225 area) MUST pass pre-zoom validation
    tiny_lm = _make_3d_hand_landmarks("INDEX_HOVER")
    tiny_lm = [[0.50 + (p[0] - 0.50) * 0.22, 0.50 + (p[1] - 0.50) * 0.14, p[2] * 0.22] for p in tiny_lm]
    assert HandGestureEngine.validate_real_hand(tiny_lm, score=0.55, face_boxes=[]) is True

    # ROI crop [x0=200, y0=100, cw=160, ch=120] on 640x480 frame maps crop center (0.5, 0.5) accurately
    mapped = HandGestureEngine.map_roi_landmarks_to_full_frame(
        [[0.5, 0.5, -0.1]],
        crop_box_px=(200, 100, 160, 120),
        full_size_wh=(640, 480),
    )
    assert abs(mapped[0][0] - (280.0 / 640.0)) < 1e-3
    assert abs(mapped[0][1] - (160.0 / 480.0)) < 1e-3


def test_tier4_3d_rotation_scale_palm_normal_kalman_and_config() -> None:
    from cv_engine.hand_gesture import (  # noqa: E402
        FingerFlexionState,
        GestureFsmEngine,
        KalmanVelocityPredictor2D,
        compute_palm_normal_3d,
        compute_palm_scale_3d,
    )
    from cv_engine.vision_config import load_vision_config  # noqa: E402

    cfg = load_vision_config()
    assert cfg.gesture_3d.pinch_enter_ratio == pytest.approx(0.28, abs=1e-3)
    assert cfg.gesture_3d.voting_window_size == 7
    assert cfg.hud.debug_hud_default_enabled is False

    # Test 0, 90, 180 degree planar rotation + scale invariance (0.25x & 2.0x)
    base_idx = _make_3d_hand_landmarks("INDEX_HOVER")
    for angle_deg in (0.0, 90.0, 180.0):
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        rot_lm = [
            [
                0.5 + (p[0] - 0.5) * ca - (p[1] - 0.5) * sa,
                0.5 + (p[0] - 0.5) * sa + (p[1] - 0.5) * ca,
                p[2],
            ]
            for p in base_idx
        ]
        res_rot = classify_hand_3d_vectors(rot_lm)
        assert res_rot["pose_tag"] == "INDEX"
        assert res_rot["finger_states"][1] == FingerFlexionState.EXTENDED.value
        assert res_rot["finger_states"][2] in (FingerFlexionState.CURLED.value, FingerFlexionState.HALF_BENT.value)

    scaled_small = [[0.5 + (p[0] - 0.5) * 0.25, 0.5 + (p[1] - 0.5) * 0.25, p[2] * 0.25] for p in _make_3d_hand_landmarks("PINCH_DRAW")]
    res_small = classify_hand_3d_vectors(scaled_small)
    assert res_small["pose_tag"] == "PINCH"
    assert float(res_small["pinch_ratio"]) < 0.28

    # Zero-division protection on degenerate zero-length landmarks
    degenerate = [[0.5, 0.5, 0.0]] * 21
    assert compute_palm_scale_3d(degenerate) > 0.0
    pn_deg = compute_palm_normal_3d(degenerate)
    assert pn_deg["facing_camera"] is False
    res_deg = classify_hand_3d_vectors(degenerate)
    assert math.isfinite(float(res_deg["pinch_ratio"]))

    # Palm Normal differentiates frontal vs sideways (edge-on)
    pn_front = compute_palm_normal_3d(base_idx, min_facing_z=0.20)
    assert pn_front["facing_camera"] is True
    sideways = [[0.5 + p[2], p[1], p[0] - 0.5] for p in base_idx]
    pn_side = compute_palm_normal_3d(sideways, min_facing_z=0.20)
    assert float(pn_side["facing_score"]) < float(pn_front["facing_score"])

    # Kalman Grace Period velocity prediction keeps moving smoothly across dropout
    fsm = GestureFsmEngine(min_hold_frames=5, min_hold_sec=0.140, grace_period_sec=0.160, voting_window_size=7, voting_min_votes=5)
    t0 = 1.0
    for k in range(6):
        shifted = [[p[0] + k * 0.02, p[1], p[2]] for p in base_idx]
        snap = fsm.update(shifted, t_sec=t0 + k * 0.032)
    assert snap.active_gesture == "DRAW"
    last_x = snap.cursor_norm[0]
    grace_snap = fsm.update(None, t_sec=t0 + 6 * 0.032 + 0.060)
    assert grace_snap.in_grace_period is True
    assert grace_snap.kalman_predicted is True
    assert grace_snap.cursor_norm is not None and grace_snap.cursor_norm[0] > last_x

    # Kalman standalone predictor sanity
    kp = KalmanVelocityPredictor2D()
    kp.update(0.2, 0.5, 0.0)
    kp.update(0.3, 0.5, 0.1)
    pred_x, _ = kp.predict_dropout(0.15)
    assert pred_x > 0.30



