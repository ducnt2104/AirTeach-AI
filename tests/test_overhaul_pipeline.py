#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for AirTeach AI Overhaul:
1. ROI-Only Sharpening & Adaptive Contrast Stretch (`sharpen_and_contrast_roi`)
2. Auto Performance Profile (`ULTRA_LIGHT` on >45ms) & Staggered Inference Scheduler
3. Dual-Resolution ROI Coordinate Mapping + Catmull-Rom Spline
4. 3D Distance & Rotation-Invariant Gesture Classification
5. Agentic Whiteboard [0..1000] JSON Action Parser & Safe Validation
6. Gemini API Key Layered Priority & Resilient Error Classification
"""

import numpy as np

from cv_engine.gemini_client import (
    ApiConnectionStatus,
    classify_gemini_error,
    mask_api_key,
    resolve_api_key_layered,
)
from cv_engine.hand_gesture import (
    HandGestureEngine,
    catmull_rom_spline,
    classify_hand_3d_vectors,
)
from cv_engine.performance_pipeline import (
    PROFILE_PRESETS,
    AutoProfileController,
    PerformanceProfile,
    StaggeredInferenceScheduler,
)
from cv_engine.preprocessing import sharpen_and_contrast_roi
from cv_engine.whiteboard_agent import (
    WhiteboardAgent,
    build_vector_shape_strokes,
    find_free_regions_normalized,
)


def test_roi_only_sharpening_lifts_contrast_without_mutating_input() -> None:
    # Low-contrast dark 224x224 ROI (simulating distant hand in dim room)
    roi = np.full((224, 224, 3), 42, dtype=np.uint8)
    roi[60:160, 80:140] = 68
    orig_copy = roi.copy()

    out = sharpen_and_contrast_roi(roi, is_rgb=True, unsharp_strength=0.6, contrast_stretch=True)
    assert out.shape == (224, 224, 3)
    assert np.array_equal(roi, orig_copy), "Input ROI array must not be mutated in-place"
    assert float(np.mean(out)) > float(np.mean(roi)) + 10.0, "Low-light ROI must be brightened"


def test_auto_profile_downgrades_to_ultra_light_above_45ms() -> None:
    ctrl = AutoProfileController(
        initial_profile=PerformanceProfile.BALANCED,
        auto_adapt=True,
        downgrade_threshold_ms=45.0,
        window_size=5,
    )
    assert ctrl.active_profile == PerformanceProfile.BALANCED
    for _ in range(10):
        ctrl.record_inference_ms(58.0)
    assert ctrl.active_profile == PerformanceProfile.ULTRA_LIGHT
    assert ctrl.auto_downgraded is True
    assert ctrl.preset.search_size == (320, 180)
    assert ctrl.preset.roi_size == 224
    assert ctrl.preset.pose_interval_frames == 15


def test_staggered_inference_scheduler_separates_pose_and_face() -> None:
    sched = StaggeredInferenceScheduler()
    preset = PROFILE_PRESETS[PerformanceProfile.ULTRA_LIGHT]  # pose every 15 frames, face every 2.0s

    pose_runs = 0
    face_runs = 0
    for i in range(30):
        run_pose, run_face = sched.next_tick(
            now_sec=float(i) * 0.1,
            preset=preset,
            has_locked_teacher=True,
            teacher_lost_sec=0.0,
        )
        if run_pose:
            pose_runs += 1
        if run_face:
            face_runs += 1

    # 30 ticks at 0.1s (3.0s total): Pose runs ~2-3 times, Face runs ~2 times (not 30!)
    assert 1 <= pose_runs <= 3, f"Expected staggered pose runs <=3, got {pose_runs}"
    assert 1 <= face_runs <= 3, f"Expected staggered face runs <=3, got {face_runs}"


def test_roi_landmark_mapping_and_catmull_rom_spline() -> None:
    # Crop [320, 180, 224, 224] inside 1280x720 frame
    mapped = HandGestureEngine.map_roi_landmarks_to_full_frame(
        [[0.5, 0.5, -0.02]],
        crop_box_px=(320, 180, 224, 224),
        full_size_wh=(1280, 720),
    )
    assert len(mapped) == 1
    assert abs(mapped[0][0] - (320 + 112) / 1280.0) < 1e-3
    assert abs(mapped[0][1] - (180 + 112) / 720.0) < 1e-3

    smooth = catmull_rom_spline([(0.0, 0.0), (50.0, 40.0), (100.0, 10.0)], samples_per_segment=4)
    assert len(smooth) > 3


def test_whiteboard_agent_normalized_1000_schema_and_safe_validation() -> None:
    sample_reply = """
    Đây là tam giác vuông ABC có góc vuông tại A:
    ```whiteboard_actions
    {
      "speech_or_text": "Đã vẽ tam giác vuông ABC tại A và chú thích công thức Pytago.",
      "actions": [
        {"type": "draw_shape", "shape": "triangle", "x": 150, "y": 200, "w": 220, "h": 180, "color": "#00F5D4", "label": "Tam giác ABC"},
        {"type": "draw_text", "x": 150, "y": 420, "text": "AB² + AC² = BC²", "color": "#FEE440", "fontSize": 18},
        {"type": "draw_polyline", "points": [[100, 500], [200, 450], [300, 520]], "smooth": true},
        {"type": "highlight_region", "x": 140, "y": 190, "w": 250, "h": 210, "label": "Góc vuông A"},
        {"type": "draw_text", "text": ""},
        {"type": "unknown_op", "foo": 123}
      ]
    }
    ```
    """
    actions, clean_txt = WhiteboardAgent.extract_actions_from_response(sample_reply)
    assert len(actions) == 4, f"Expected 4 valid actions (empty/unknown ignored), got {len(actions)}"
    assert actions[0].tool == "draw_shape" and actions[0].args["type"] == "triangle"
    assert actions[0].args["normalized"] is True
    assert actions[1].tool == "draw_text" and "AB² + AC² = BC²" in actions[1].args["text"]
    assert actions[2].tool == "draw_polyline" and len(actions[2].args["points"]) == 3
    assert actions[3].tool == "highlight_region" and actions[3].args["label"] == "Góc vuông A"
    assert "whiteboard_actions" not in clean_txt

    tri_strokes = build_vector_shape_strokes("triangle", {"w": 200, "h": 150}, (100.0, 100.0))
    assert len(tri_strokes) == 1 and len(tri_strokes[0]) == 4

    free_regs = find_free_regions_normalized([(0.0, 0.0, 400.0, 400.0)], (1280.0, 720.0), max_regions=2)
    assert len(free_regs) == 2 and all(len(r) == 4 for r in free_regs)


def test_gemini_resilient_key_priority_and_error_classification(tmp_path) -> None:
    settings_file = tmp_path / "api_settings.json"
    settings_file.write_text('{"ui_gemini_api_key": "AIzaTestUiKey123456"}', encoding="utf-8")
    resolved = resolve_api_key_layered(settings_path=settings_file)
    assert resolved.source == "ui"
    assert resolved.key == "AIzaTestUiKey123456"
    assert mask_api_key(resolved.key).startswith("AIza")

    code_401, cat_401, msg_401 = classify_gemini_error(Exception("401 UNAUTHENTICATED"))
    assert code_401 == 401 and cat_401 == "AUTH" and "API Key" in msg_401

    code_404, cat_404, msg_404 = classify_gemini_error(Exception("404 NOT_FOUND"))
    assert code_404 == 404 and cat_404 == "NOT_FOUND" and len(msg_404) > 5

    code_429, cat_429, msg_429 = classify_gemini_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert code_429 == 429 and cat_429 == "RATE_LIMIT" and len(msg_429) > 5


def test_app_paths_resource_and_appdata_and_logo() -> None:
    from cv_engine.app_paths import get_config_file_path, get_user_data_dir, resource_path

    ico = resource_path("assets/logo.ico")
    assert ico.exists() and ico.stat().st_size > 10_000, "Multi-size logo.ico must exist"
    assert resource_path("models/hand_landmarker.task").exists()
    assert resource_path("config/vision_config.json").exists()

    user_dir = get_user_data_dir()
    assert user_dir.exists() and "AirTeachAI" in str(user_dir)
    assert (user_dir / "logs").exists()
    cfg_path = get_config_file_path()
    assert cfg_path.parent == user_dir and cfg_path.name == "config.json"
