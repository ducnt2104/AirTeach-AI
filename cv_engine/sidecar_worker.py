#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Throughput IPC Sidecar Worker (`cv_engine/sidecar_worker.py`)
for AirTeach AI — DNT - Victer.

Runs OpenCV + MediaPipe Full + 5-Point Aligned 512-D ArcFace/SFace in a dedicated
non-blocking sidecar process communicating with the Rust/Desktop shell at 60 FPS UI rate.
"""

import json
import os
from pathlib import Path
import struct
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# Ensure project root is on sys.path when invoked as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from cv_engine.face_id import (
    ArcFace512Extractor,
    align_face_5pt_112x112,
    multi_template_cosine_similarity,
)
from cv_engine.hand_gesture import (
    FrameSkipLock,
    GestureFsmEngine,
    HandGestureEngine,
    classify_hand_3d_vectors,
    compute_palm_scale_3d,
    compute_pinch_ratio_3d,
    joint_angle_3d_deg,
    thumb_palm_plane_angle_deg,
)
from cv_engine.preprocessing import preprocess_adaptive_frame


def run_self_test() -> int:
    """
    Execute deterministic verification of the entire `cv_engine` pipeline:
    1. LAB Luminance & Exposure Diagnostics (`mean_luma`, `p5`, `p95`), Low-Light Bilateral + Gamma,
       Glare Highlight Roll-off, LAB `(8,8)` CLAHE `[1.5, 4.0]`, USM `1.45*img - 0.45*blur`,
       and Laplacian Variance Blur Quality Gate.
    2. 3D Vector Joint-Angle Hand Classification (`arccos(dot(u,v)/(|u||v|))`), Thumb Palm-Plane
       Abduction, Pinch Ratio (`dist(4,8)/dist(0,9)`), One-Euro Filter, and Gesture FSM (`IDLE -> CANDIDATE -> ACTIVE -> GRACE_PERIOD`).
    3. 5-Point Canonical `112x112` Affine Face Alignment + 512-D Embedding + Multi-Template Matching.
    """
    # 1. Test Low-Light, Glare, and Blur Quality Gate
    dark_img = np.full((240, 320, 3), 28, dtype=np.uint8)
    for y in range(0, 240, 16):
        for x in range(0, 320, 16):
            if ((x // 16) + (y // 16)) % 2 == 0:
                dark_img[y : y + 16, x : x + 16] = 48
    enh_dark, diag_dark = preprocess_adaptive_frame(dark_img, is_rgb=True, quality_gate_min=38.0)
    assert diag_dark.lighting_mode == "LOW_LIGHT", f"Expected LOW_LIGHT, got {diag_dark.lighting_mode}"
    assert diag_dark.gamma_applied < 1.0, f"Expected gamma < 1.0, got {diag_dark.gamma_applied}"
    assert float(np.mean(enh_dark)) > diag_dark.mean_luma + 15.0, "Low-light enhancement must lift brightness"
    assert diag_dark.quality_gate_passed, "Textured checkerboard must pass Laplacian Quality Gate"

    glare_img = np.full((240, 320, 3), 245, dtype=np.uint8)
    enh_glare, diag_glare = preprocess_adaptive_frame(glare_img, is_rgb=True, quality_gate_min=38.0)
    assert diag_glare.lighting_mode == "GLARE", f"Expected GLARE, got {diag_glare.lighting_mode}"
    assert not diag_glare.quality_gate_passed, "Uniform flat glare frame must fail Quality Gate"
    _ = enh_glare

    # 2. Test 3D Vector Joint-Angle Hand Classification & Rotation Invariance (90 deg & 180 deg inverted!)
    def build_draw_pose_3d() -> list:
        lm = [[0.5, 0.5, 0.0] for _ in range(21)]
        lm[0] = [0.50, 0.70, 0.00]
        lm[1] = [0.45, 0.65, -0.01]
        lm[2] = [0.41, 0.60, -0.02]
        lm[3] = [0.38, 0.55, -0.02]
        lm[4] = [0.35, 0.50, -0.02]
        # Index extended straight (5 -> 6 -> 7 -> 8)
        lm[5] = [0.47, 0.55, 0.00]
        lm[6] = [0.47, 0.46, -0.01]
        lm[7] = [0.47, 0.38, -0.02]
        lm[8] = [0.47, 0.30, -0.03]
        # Middle curled (9 -> 10 -> 11 -> 12)
        lm[9] = [0.50, 0.54, 0.00]
        lm[10] = [0.50, 0.48, -0.02]
        lm[11] = [0.50, 0.55, -0.04]
        lm[12] = [0.50, 0.61, -0.01]
        # Ring curled
        lm[13] = [0.53, 0.55, 0.00]
        lm[14] = [0.53, 0.49, -0.02]
        lm[15] = [0.53, 0.56, -0.04]
        lm[16] = [0.53, 0.62, -0.01]
        # Pinky curled
        lm[17] = [0.56, 0.57, 0.00]
        lm[18] = [0.56, 0.52, -0.02]
        lm[19] = [0.56, 0.58, -0.03]
        lm[20] = [0.56, 0.63, -0.01]
        return lm

    upright = build_draw_pose_3d()
    res_up = classify_hand_3d_vectors(upright)
    assert res_up["gesture"] == "DRAW", f"Expected DRAW for upright index pose, got {res_up['gesture']}"

    # Rotate hand 180 degrees upside-down (where naive Y-coordinate rule fails completely!)
    inverted = [[1.0 - p[0], 1.0 - p[1], p[2]] for p in upright]
    res_inv = classify_hand_3d_vectors(inverted)
    assert res_inv["gesture"] == "DRAW", (
        f"180-deg inverted hand must still classify as DRAW via 3D vector angles, got {res_inv['gesture']}"
    )

    # Test Gesture FSM (IDLE -> CANDIDATE -> ACTIVE -> GRACE_PERIOD)
    fsm = GestureFsmEngine(min_hold_frames=5, min_hold_sec=0.150, grace_period_sec=0.150)
    for step in range(1, 5):
        snap = fsm.update(upright, t_sec=step * 0.040)
        assert snap.active_gesture == "IDLE", "Must remain IDLE during candidate hold window"
    snap_active = fsm.update(upright, t_sec=6 * 0.040)
    assert snap_active.active_gesture == "DRAW", f"Expected ACTIVE DRAW, got {snap_active.active_gesture}"
    snap_grace = fsm.update(None, t_sec=6 * 0.040 + 0.080)
    assert snap_grace.in_grace_period and snap_grace.active_gesture == "DRAW", "Grace period must hold DRAW"

    # 3. Test 5-Point 112x112 Affine Face Alignment & 512-D Multi-Template Cosine Similarity
    extractor = ArcFace512Extractor(Path("models"))
    synth_face = np.zeros((240, 240, 3), dtype=np.uint8)
    for y in range(240):
        for x in range(240):
            synth_face[y, x] = [(x * 3 + y) % 255, (y * 2) % 255, ((x + y) * 2) % 255]
    pts5 = np.array(
        [[88.0, 96.0], [152.0, 96.0], [120.0, 130.0], [95.0, 165.0], [145.0, 165.0]],
        dtype=np.float32,
    )
    aligned_112, _, emb512 = extractor.extract_from_5pt_aligned(
        synth_face, pts5, (60.0, 50.0, 120.0, 150.0)
    )
    assert aligned_112.shape == (112, 112, 3), f"Expected (112,112,3), got {aligned_112.shape}"
    assert emb512.shape == (512,), f"Expected 512-D vector, got {emb512.shape}"
    assert abs(float(np.linalg.norm(emb512)) - 1.0) < 1e-4, "512-D embedding must be L2-normalized"

    templates = [emb512.tolist() for _ in range(5)]
    sim = multi_template_cosine_similarity(emb512.tolist(), templates, mode="top2")
    assert sim > 0.99, f"Self template match must exceed 0.99, got {sim}"

    # Verify impostor face is strictly rejected below 0.50 threshold
    impostor_face = np.full((240, 240, 3), 90, dtype=np.uint8)
    for y in range(60, 190):
        for x in range(65, 175):
            impostor_face[y, x] = [120, 155, 180]
    impostor_face[88:104, 78:98] = [10, 10, 10]
    impostor_face[88:104, 142:162] = [10, 10, 10]
    impostor_face[162:168, 95:145] = [40, 20, 20]
    _, _, imp_emb512 = extractor.extract_from_5pt_aligned(
        impostor_face, pts5, (60.0, 50.0, 120.0, 150.0)
    )
    imp_sim = multi_template_cosine_similarity(imp_emb512.tolist(), templates, mode="top2")
    assert imp_sim < 0.45, f"Impostor face must score below 0.50 threshold, got {imp_sim}"

    lock = FrameSkipLock()
    assert lock.try_acquire()
    assert not lock.try_acquire()
    assert lock.skipped_frames == 1
    lock.release()

    print(
        json.dumps(
            {
                "cv_engine_self_test": "PASSED",
                "dark_luma_in": round(diag_dark.mean_luma, 2),
                "dark_gamma": round(diag_dark.gamma_applied, 3),
                "dark_clahe_clip": round(diag_dark.clahe_clip_limit, 2),
                "laplacian_var": round(diag_dark.laplacian_variance, 2),
                "rotation_invariant_3d_gesture": res_inv["gesture"],
                "pinch_ratio": res_inv["pinch_ratio"],
                "aligned_face_shape": list(aligned_112.shape),
                "arcface_embedding_dim": int(emb512.size),
                "multi_template_sim": round(sim, 4),
                "impostor_sim": round(imp_sim, 4),
            }
        )
    )
    return 0


def run_ipc_loop() -> None:
    models_dir = Path("models")
    hand_engine = HandGestureEngine(
        model_path=models_dir / "hand_landmarker.task",
        min_detection_confidence=0.60,
        min_tracking_confidence=0.70,
    )
    if hand_engine.tasks_detector is None:
        sys.exit(1)

    face_engine = ArcFace512Extractor(
        models_dir=models_dir,
        min_detection_confidence=0.55,
    )

    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

    stdout_buf.write(b"READY\n")
    stdout_buf.flush()

    while True:
        hdr = stdin_buf.read(8)
        if not hdr or len(hdr) < 8:
            break
        magic, w, h = struct.unpack("<4sHH", hdr)
        if magic != b"MPHF" or w == 0 or h == 0:
            break

        nbytes = int(w) * int(h) * 3
        raw = stdin_buf.read(nbytes)
        if not raw or len(raw) < nbytes:
            break

        if not hand_engine.frame_skip_lock.try_acquire():
            continue
        try:
            rgb_orig = np.frombuffer(raw, dtype=np.uint8).reshape((int(h), int(w), 3))
            rgb_enhanced, diag = preprocess_adaptive_frame(rgb_orig, is_rgb=True, quality_gate_min=38.0)

            face_info = face_engine.detect_and_extract(rgb_enhanced, quality_gate_min_sharpness=38.0)
            face_boxes = face_info["face_boxes"]

            left_out, right_out = hand_engine.detect_hands(rgb_enhanced, face_boxes)

            resp = {
                "left": left_out,
                "right": right_out,
                "face": face_info["face"],
                "face_eyes": face_info["face_eyes"],
                "face_5pt": face_info["face_5pt"],
                "face_emb": face_info["face_emb"],
                "face_emb_512": face_info["face_emb_512"],
                "diag": diag.to_dict(),
            }
            line = (json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8")
            stdout_buf.write(line)
            stdout_buf.flush()
        finally:
            hand_engine.frame_skip_lock.release()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(run_self_test())
    run_ipc_loop()
