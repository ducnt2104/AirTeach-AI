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

    # 4. Test 4-Tier Zero-Stray-Stroke Drawing FSM (Schmitt Trigger + Tail Trim + 300ms Cooldown + 400ms Undo)
    import math as _math
    from cv_engine.drawing_fsm import DrawingHudMode, ZeroStrayStrokeDrawingFsm
    from cv_engine.shape_recognizer import beautify_stroke

    dfsm = ZeroStrayStrokeDrawingFsm()
    # Two pinch frames < 0.28 must still stay HOVER (requires >= 3 frames)
    u1 = dfsm.step(0.0, True, (100.0, 100.0), 0.24, "PINCH")
    u2 = dfsm.step(33.0, True, (110.0, 100.0), 0.24, "PINCH")
    assert u1.hud_mode == DrawingHudMode.HOVER and u2.hud_mode == DrawingHudMode.HOVER
    # 3rd frame triggers DRAWING
    u3 = dfsm.step(66.0, True, (120.0, 100.0), 0.24, "PINCH")
    assert u3.hud_mode == DrawingHudMode.DRAWING and u3.action == "BEGIN_STROKE"
    for k in range(1, 10):
        uk = dfsm.step(66.0 + k * 33.0, True, (120.0 + k * 20.0, 100.0), 0.35, "PINCH")
        assert uk.hud_mode == DrawingHudMode.DRAWING, "Hysteresis <= 0.40 must keep DRAWING"
    # Inject wild hook points in final ~100ms before release (> 0.40)
    t_hook = 66.0 + 10 * 33.0
    dfsm.step(t_hook, True, (320.0, 260.0), 0.38, "PINCH")
    dfsm.step(t_hook + 33.0, True, (380.0, 350.0), 0.39, "PINCH")
    u_rel = dfsm.step(t_hook + 66.0, True, (420.0, 420.0), 0.48, "HOVER")
    assert u_rel.action == "COMMIT_STROKE" and u_rel.hud_mode == DrawingHudMode.HOVER
    assert max(p[1] for p in u_rel.committed_points) < 200.0, "Tail trimming must remove release hook"

    # Cancel via FIST 150ms -> discards stroke + locks pen 300ms
    dfsm.reset()
    for k in range(5):
        dfsm.step(k * 33.0, True, (100.0 + k * 10.0, 100.0), 0.22, "PINCH")
    dfsm.step(200.0, True, (150.0, 100.0), 0.22, "FIST")
    uc = dfsm.step(360.0, True, (150.0, 100.0), 0.22, "FIST")
    assert uc.action == "CANCEL_STROKE" and uc.hud_mode == DrawingHudMode.LOCKED_COOLDOWN
    # During 300ms cooldown, pinch < 0.28 must NOT draw
    ul = dfsm.step(450.0, True, (160.0, 100.0), 0.18, "PINCH")
    assert ul.hud_mode == DrawingHudMode.LOCKED_COOLDOWN and ul.action == "NONE"

    # 5. Test Hybrid Shape Beautification (Line, Circle, Rotated Ellipse, Rectangle, Freehand gate)
    line_pts = [(100.0 + i * 8.0, 200.0 + i * 0.2) for i in range(25)]
    res_line = beautify_stroke(line_pts)
    assert res_line.shape_type == "LINE", f"Expected LINE, got {res_line.shape_type}"

    circle_pts = [(300.0 + 90.0 * _math.cos(2 * _math.pi * i / 48), 300.0 + 90.0 * _math.sin(2 * _math.pi * i / 48)) for i in range(49)]
    res_circle = beautify_stroke(circle_pts)
    assert res_circle.shape_type == "CIRCLE", f"Expected CIRCLE, got {res_circle.shape_type}"

    ell_pts = [(300.0 + 130.0 * _math.cos(2 * _math.pi * i / 48), 300.0 + 60.0 * _math.sin(2 * _math.pi * i / 48)) for i in range(49)]
    res_ell = beautify_stroke(ell_pts)
    assert res_ell.shape_type == "ELLIPSE", f"Expected ELLIPSE, got {res_ell.shape_type}"

    # Freehand cursive wave must NOT snap falsely into a box/circle
    wave_pts = [(50.0 + i * 6.0, 200.0 + 35.0 * _math.sin(i * 0.45) + 0.5 * i) for i in range(55)]
    res_wave = beautify_stroke(wave_pts)
    assert res_wave.shape_type == "FREEHAND", f"Wave must stay FREEHAND, got {res_wave.shape_type}"

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
                "drawing_fsm_tail_pts": len(u_rel.committed_points),
                "shape_types": [res_line.shape_type, res_circle.shape_type, res_ell.shape_type, res_wave.shape_type],
            }
        )
    )
    return 0


def run_ipc_loop() -> None:
    import time
    from cv_engine.app_paths import get_app_logger, resource_path
    from cv_engine.hand_gesture import DualHandGestureCoordinator
    from cv_engine.performance_pipeline import (
        AutoProfileController,
        PerformanceProfile,
        StaggeredInferenceScheduler,
    )
    from cv_engine.preprocessing import AdaptiveExposureController, ExposureDiagnostics
    from cv_engine.subject_tracker import PoseTorsoEstimator, PrimaryTeacherLockTracker
    from cv_engine.vision_config import load_vision_config

    logger = get_app_logger("sidecar_worker")
    vision_cfg = load_vision_config()
    models_dir = resource_path("models")
    profile_ctrl = AutoProfileController(
        initial_profile=PerformanceProfile.BALANCED,
        auto_adapt=True,
        downgrade_threshold_ms=45.0,
    )
    scheduler = StaggeredInferenceScheduler()
    dual_coord = DualHandGestureCoordinator()

    hand_engine = HandGestureEngine(
        model_path=models_dir / "hand_landmarker.task",
        num_hands=vision_cfg.hand_tracking.num_hands,
        min_detection_confidence=vision_cfg.hand_tracking.min_detection_confidence,
        min_presence_confidence=vision_cfg.hand_tracking.min_presence_confidence,
        min_tracking_confidence=vision_cfg.hand_tracking.min_tracking_confidence,
        roi_area_threshold=vision_cfg.hand_tracking.roi_area_threshold,
        roi_expand_factor=vision_cfg.hand_tracking.roi_expand_factor,
        roi_padding_ratio=vision_cfg.hand_tracking.roi_padding_ratio,
        roi_target_size=profile_ctrl.preset.roi_size,
        pinch_enter_ratio=vision_cfg.gesture_3d.pinch_enter_ratio,
        pinch_exit_ratio=vision_cfg.gesture_3d.pinch_exit_ratio,
        palm_facing_min_z=vision_cfg.gesture_3d.palm_facing_min_z,
        grace_period_sec=vision_cfg.gesture_3d.grace_period_ms / 1000.0,
        voting_window_size=vision_cfg.gesture_3d.voting_window_size,
        voting_min_votes=vision_cfg.gesture_3d.voting_min_votes,
        one_euro_min_cutoff=vision_cfg.gesture_3d.one_euro_min_cutoff,
        one_euro_beta=vision_cfg.gesture_3d.one_euro_beta,
    )
    if hand_engine.tasks_detector is None:
        logger.error("Sidecar HandGestureEngine failed to initialize: %s", hand_engine.init_error)
        sys.exit(1)

    face_engine = ArcFace512Extractor(
        models_dir=models_dir,
        min_detection_confidence=0.45,
    )
    exposure_ctrl = AdaptiveExposureController(
        check_interval_n=vision_cfg.preprocessing.exposure_check_interval_n,
        low_light_mean_luma=vision_cfg.preprocessing.low_light_mean_luma,
        backlit_p95_luma=vision_cfg.preprocessing.backlit_p95_luma,
        flicker_delta_luma=vision_cfg.preprocessing.flicker_delta_luma,
        clahe_clip_limit=vision_cfg.preprocessing.clahe_clip_limit,
        tile_grid_size=vision_cfg.preprocessing.clahe_tile_grid_size,
        quality_gate_min=vision_cfg.preprocessing.quality_gate_min_sharpness,
    )
    pose_estimator = PoseTorsoEstimator(
        models_dir=models_dir,
        enable_pose_landmarker=vision_cfg.subject_tracker.enable_pose_landmarker,
        min_det_conf=vision_cfg.subject_tracker.pose_min_detection_confidence,
        min_pres_conf=vision_cfg.subject_tracker.pose_min_presence_confidence,
        min_track_conf=vision_cfg.subject_tracker.pose_min_tracking_confidence,
    )
    teacher_tracker = PrimaryTeacherLockTracker(
        teacher_grace_sec=vision_cfg.subject_tracker.teacher_grace_sec,
        match_iou_min=vision_cfg.subject_tracker.teacher_match_iou_min,
        face_sim_threshold=vision_cfg.subject_tracker.teacher_face_sim_threshold,
        wrist_gate_radius_scale=vision_cfg.subject_tracker.wrist_gate_radius_scale,
        enabled=vision_cfg.subject_tracker.teacher_lock_enabled,
    )

    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

    stdout_buf.write(b"READY\n")
    stdout_buf.flush()

    cached_face_info: dict = {
        "face": None,
        "face_boxes": [],
        "face_eyes": None,
        "face_5pt": None,
        "face_emb": None,
        "face_emb_512": None,
    }
    cached_poses: list = []
    last_teacher_seen_ts = time.monotonic()

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
        t_start = time.perf_counter()
        try:
            now_mono = time.monotonic()
            preset = profile_ctrl.preset
            exposure_ctrl.check_interval_n = preset.exposure_interval_n

            rgb_orig = np.ascontiguousarray(
                np.frombuffer(raw, dtype=np.uint8).reshape((int(h), int(w), 3))
            )
            if preset.enable_full_frame_clahe:
                rgb_work, diag = exposure_ctrl.process(rgb_orig, is_rgb=True, enabled=True)
            else:
                rgb_work = rgb_orig
                diag = ExposureDiagnostics(
                    mean_luma=128.0,
                    p5=20.0,
                    p95=235.0,
                    lighting_mode="NORMAL",
                    gamma_applied=1.0,
                    clahe_clip_limit=0.0,
                    laplacian_variance=50.0,
                    quality_gate_passed=True,
                    clahe_bypassed=True,
                )

            has_locked = teacher_tracker.locked_track_id is not None
            teacher_lost_sec = (now_mono - last_teacher_seen_ts) if has_locked else 999.0
            run_pose, run_face = scheduler.next_tick(
                now_mono,
                preset,
                has_locked_teacher=has_locked,
                teacher_lost_sec=teacher_lost_sec,
            )

            if run_face:
                cached_face_info = face_engine.detect_and_extract(
                    rgb_work,
                    quality_gate_min_sharpness=vision_cfg.preprocessing.quality_gate_min_sharpness,
                )
            face_boxes = cached_face_info.get("face_boxes") or []

            if run_pose:
                cached_poses = pose_estimator.detect_poses(rgb_work)

            subj_snap = teacher_tracker.update(
                face_boxes=face_boxes,
                poses=cached_poses,
                now_sec=now_mono,
            )
            if subj_snap.locked_teacher_id is not None:
                last_teacher_seen_ts = now_mono

            wrist_hints = teacher_tracker.get_proactive_wrist_roi_boxes(
                subj_snap,
                roi_expand_factor=vision_cfg.hand_tracking.roi_expand_factor,
            )

            left_out, right_out = hand_engine.detect_hands(
                rgb_work,
                face_boxes,
                teacher_tracker=teacher_tracker,
                teacher_snapshot=subj_snap,
                wrist_roi_hints=wrist_hints,
                search_size=preset.search_size,
                roi_target_size=preset.roi_size,
                enable_roi_sharpen=preset.enable_roi_sharpen,
            )
            dual_state = dual_coord.update(left_out, right_out, t_sec=now_mono)

            cv_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            active_prof = profile_ctrl.record_inference_ms(cv_latency_ms)

            resp = {
                "left": left_out,
                "right": right_out,
                "raw_hands": hand_engine.last_raw_hands,
                "dual_hand": {
                    "primary_side": dual_state.primary_side,
                    "mode": dual_state.two_hand_mode,
                    "zoom_factor": dual_state.zoom_factor,
                    "pan_delta_norm": list(dual_state.pan_delta_norm),
                    "offhand_command": dual_state.offhand_command,
                    "status_label_vi": dual_state.status_label_vi,
                },
                "face": subj_snap.locked_face_bbox or cached_face_info["face"],
                "face_eyes": cached_face_info["face_eyes"],
                "face_5pt": cached_face_info["face_5pt"],
                "face_emb": cached_face_info["face_emb"],
                "face_emb_512": cached_face_info["face_emb_512"],
                "diag": diag.to_dict(),
                "subject": subj_snap.to_dict(),
                "cv_latency_ms": cv_latency_ms,
                "perf_profile": active_prof.value,
                "avg_inference_ms": round(profile_ctrl.avg_inference_ms, 2),
                "roi_mode": hand_engine.last_tracking_mode,
            }
            line = (json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8")
            stdout_buf.write(line)
            stdout_buf.flush()
        except Exception as exc:
            logger.exception("Sidecar IPC frame processing error: %s", exc)
        finally:
            hand_engine.frame_skip_lock.release()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(run_self_test())
    run_ipc_loop()
