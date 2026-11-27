#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test suite for AirTeach AI Root Cause Fixes (Stages 1–4):
1. Fail-Open 2-Hand Detection, 1-Frame ROI Fallback, Left/Right preservation, 2nd-hand discovery.
2. Teacher Lock auto-bypass for <= 1 person and multi-person filtering.
3. Dual-Hand Coordinator (Dual-Pinch Zoom, Dual-Palm Pan, Off-hand Tool/Color/Undo).
4. Drawing FSM (`INDEX` 1-finger point drawing + 2-hand `suppress_drawing`).
5. PySide6 GUI & CameraHandWorker zero-crash instantiation, Pan/Zoom canvas, JSON actions + Undo.
6. `%APPDATA%\\AirTeachAI` paths & Gemini API config/fallback resilience.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cv_engine.app_paths import (
    get_config_file_path,
    get_local_cache_db_path,
    get_user_data_dir,
)
from cv_engine.drawing_fsm import DrawingHudMode, ZeroStrayStrokeDrawingFsm
from cv_engine.gemini_client import (
    DEFAULT_FALLBACK_MODELS,
    ApiConnectionStatus,
    GeminiHealthCheckResult,
    ResolvedApiKey,
    load_ui_api_settings,
    save_ui_api_settings,
)
from cv_engine.hand_gesture import DualHandGestureCoordinator, HandGestureEngine
from cv_engine.subject_tracker import PrimaryTeacherLockTracker
from cv_engine.vision_config import load_vision_config
from cv_engine.whiteboard_agent import WhiteboardAgent


def _make_hand_landmarks(
    cx: float = 0.5,
    cy: float = 0.5,
    scale: float = 0.14,
    *,
    pinch: bool = False,
    open_palm: bool = True,
    index_only: bool = False,
    two_fingers: bool = False,
    fist: bool = False,
) -> List[List[float]]:
    """Build 21 anatomically plausible normalized landmarks [x, y, z] for deterministic tests."""
    pts = [[cx, cy, 0.0] for _ in range(21)]
    # Wrist and palm core
    pts[0] = [cx, cy + scale * 0.70, 0.0]
    pts[1] = [cx - scale * 0.25, cy + scale * 0.45, -0.01]
    pts[2] = [cx - scale * 0.42, cy + scale * 0.22, -0.015]
    pts[5] = [cx - scale * 0.28, cy + scale * 0.05, -0.01]
    pts[9] = [cx, cy, -0.01]
    pts[13] = [cx + scale * 0.24, cy + scale * 0.05, -0.01]
    pts[17] = [cx + scale * 0.45, cy + scale * 0.14, -0.01]

    def set_finger(mcp_i: int, base_x: float, base_y: float, extended: bool, len_scale: float = 1.0):
        pts[mcp_i] = [base_x, base_y, -0.01]
        if extended:
            pts[mcp_i + 1] = [base_x, base_y - scale * 0.30 * len_scale, -0.012]
            pts[mcp_i + 2] = [base_x, base_y - scale * 0.56 * len_scale, -0.014]
            pts[mcp_i + 3] = [base_x, base_y - scale * 0.80 * len_scale, -0.016]
        else:
            pts[mcp_i + 1] = [base_x, base_y - scale * 0.12, -0.01]
            pts[mcp_i + 2] = [base_x, base_y + scale * 0.10, -0.02]
            pts[mcp_i + 3] = [base_x, base_y + scale * 0.22, -0.02]

    if fist:
        pts[3] = [cx - scale * 0.25, cy + scale * 0.08, -0.02]
        pts[4] = [cx - scale * 0.10, cy + scale * 0.05, -0.02]
        set_finger(5, cx - scale * 0.28, cy + scale * 0.05, False)
        set_finger(9, cx, cy, False)
        set_finger(13, cx + scale * 0.24, cy + scale * 0.05, False)
        set_finger(17, cx + scale * 0.45, cy + scale * 0.14, False)
    elif pinch:
        pts[3] = [cx - scale * 0.30, cy - scale * 0.20, -0.018]
        pts[5] = [cx - scale * 0.24, cy, -0.01]
        pts[6] = [cx - scale * 0.25, cy - scale * 0.32, -0.015]
        pts[7] = [cx - scale * 0.27, cy - scale * 0.52, -0.018]
        pts[8] = [cx - scale * 0.26, cy - scale * 0.66, -0.02]
        pts[4] = [cx - scale * 0.26, cy - scale * 0.65, -0.02]
        set_finger(9, cx, cy, False)
        set_finger(13, cx + scale * 0.24, cy + scale * 0.05, False)
        set_finger(17, cx + scale * 0.45, cy + scale * 0.14, False)
    elif index_only:
        pts[3] = [cx - scale * 0.25, cy + scale * 0.10, -0.015]
        pts[4] = [cx - scale * 0.10, cy + scale * 0.08, -0.02]
        set_finger(5, cx - scale * 0.28, cy + scale * 0.05, True, 1.0)
        set_finger(9, cx, cy, False)
        set_finger(13, cx + scale * 0.24, cy + scale * 0.05, False)
        set_finger(17, cx + scale * 0.45, cy + scale * 0.14, False)
    elif two_fingers:
        pts[3] = [cx - scale * 0.25, cy + scale * 0.10, -0.015]
        pts[4] = [cx - scale * 0.12, cy + scale * 0.08, -0.02]
        set_finger(5, cx - scale * 0.28, cy + scale * 0.05, True, 1.0)
        set_finger(9, cx, cy, True, 1.05)
        set_finger(13, cx + scale * 0.24, cy + scale * 0.05, False)
        set_finger(17, cx + scale * 0.45, cy + scale * 0.14, False)
    else:
        # Open palm 5 fingers
        pts[3] = [cx - scale * 0.56, cy - scale * 0.02, -0.015]
        pts[4] = [cx - scale * 0.72, cy - scale * 0.22, -0.02]
        set_finger(5, cx - scale * 0.28, cy + scale * 0.05, True, 0.95)
        set_finger(9, cx, cy, True, 1.05)
        set_finger(13, cx + scale * 0.24, cy + scale * 0.05, True, 0.95)
        set_finger(17, cx + scale * 0.45, cy + scale * 0.14, True, 0.80)
    return pts


def _to_mp_hand_objects(pts: List[List[float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]


class _FakeTasksDetector:
    """Scripted MediaPipe HandLandmarker mock to test ROI fallback and 2-hand discovery."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: List[Tuple[int, int]] = []

    def detect(self, mp_image):
        arr = mp_image.numpy_view()
        h, w = int(arr.shape[0]), int(arr.shape[1])
        self.calls.append((w, h))
        return self.handler(len(self.calls), w, h)

    def close(self) -> None:
        pass


def test_vision_config_and_engine_failopen_defaults(tmp_path: Path, monkeypatch):
    cfg = load_vision_config()
    assert cfg.hand_tracking.num_hands == 2
    assert cfg.hand_tracking.min_detection_confidence <= 0.35
    assert cfg.hand_tracking.min_presence_confidence <= 0.40
    assert cfg.hand_tracking.min_tracking_confidence <= 0.40
    assert cfg.hand_tracking.roi_target_size >= 256

    monkeypatch.setattr("cv_engine.hand_gesture.ensure_hand_model", lambda *a, **k: False)
    eng = HandGestureEngine(model_path=tmp_path / "dummy.task")
    assert eng.num_hands == 2
    assert eng.min_detection_confidence <= 0.35


def test_real_mediapipe_init_after_main_import():
    """Regression test: Python 3.15 lazy_import on concurrent.futures must never break MediaPipe init."""
    import main  # noqa: F401
    from cv_engine.face_id import ArcFace512Extractor

    hand_model = Path("models/hand_landmarker.task")
    if hand_model.exists():
        eng = HandGestureEngine(model_path=hand_model)
        assert eng.init_error == ""
        assert eng.tasks_detector is not None
        eng.close()

    face_ext = ArcFace512Extractor(models_dir=Path("models"))
    if (Path("models") / "blaze_face_short_range.tflite").exists():
        assert face_ext.face_detector is not None
    face_ext.close()


def test_single_left_hand_not_collapsed_and_immediate_roi_fallback(tmp_path: Path, monkeypatch):
    left_pts = _make_hand_landmarks(cx=0.32, cy=0.52, scale=0.14, open_palm=True)

    def handler(call_idx: int, w: int, h: int):
        # Call 1 comes from Stage A stale ROI crop (size ~256x256): simulate miss!
        if call_idx == 1:
            return SimpleNamespace(hand_landmarks=[], handedness=[])
        # Call 2 immediately runs Stage B Full-Frame scan and finds Left hand!
        return SimpleNamespace(
            hand_landmarks=[_to_mp_hand_objects(left_pts)],
            handedness=[[SimpleNamespace(score=0.92, category_name="Left")]],
        )

    monkeypatch.setattr("cv_engine.hand_gesture.ensure_hand_model", lambda *a, **k: False)
    eng = HandGestureEngine(model_path=tmp_path / "dummy.task")
    eng.tasks_detector = _FakeTasksDetector(handler)
    # Plant a stale ROI cache far from the actual hand
    eng._last_hand_roi_boxes = [[0.75, 0.70, 0.18, 0.18]]

    rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
    left_h, right_h = eng.detect_hands(rgb, face_boxes=[])

    assert eng.last_tracking_mode in ("FALLBACK_FULL", "FULL_SCAN")
    assert left_h is not None
    assert right_h is None
    assert len(eng.last_raw_hands) == 1
    assert eng.last_raw_hands[0]["handedness"] == "left"


def test_two_hand_discovery_finds_second_hand_while_roi_tracks_first(tmp_path: Path, monkeypatch):
    right_pts = _make_hand_landmarks(cx=0.72, cy=0.55, scale=0.22, pinch=True)
    left_pts = _make_hand_landmarks(cx=0.25, cy=0.55, scale=0.22, open_palm=True)

    def handler(call_idx: int, w: int, h: int):
        # ROI crop scans use square 256x256 input
        if w == 256 and h == 256:
            local_r = _make_hand_landmarks(cx=0.50, cy=0.50, scale=0.45, pinch=True)
            return SimpleNamespace(
                hand_landmarks=[_to_mp_hand_objects(local_r)],
                handedness=[[SimpleNamespace(score=0.92, category_name="Right")]],
            )
        # Stage B full-frame scan discovers BOTH Right (deduped) and Left hands
        return SimpleNamespace(
            hand_landmarks=[_to_mp_hand_objects(right_pts), _to_mp_hand_objects(left_pts)],
            handedness=[
                [SimpleNamespace(score=0.90, category_name="Right")],
                [SimpleNamespace(score=0.88, category_name="Left")],
            ],
        )

    monkeypatch.setattr("cv_engine.hand_gesture.ensure_hand_model", lambda *a, **k: False)
    eng = HandGestureEngine(model_path=tmp_path / "dummy.task")
    eng.tasks_detector = _FakeTasksDetector(handler)
    # Tick 3 -> +1 = 4 triggers periodic 2nd-hand full-frame scan
    eng._detect_tick = 3
    eng._last_hand_roi_boxes = [[0.58, 0.35, 0.28, 0.40]]

    rgb = np.full((720, 1280, 3), 120, dtype=np.uint8)
    left_h, right_h = eng.detect_hands(rgb, face_boxes=[])
    assert left_h is not None
    assert right_h is not None
    assert len(eng.last_raw_hands) == 2


def test_teacher_lock_auto_bypasses_single_person_and_filters_second_person():
    tracker = PrimaryTeacherLockTracker(enabled=True)
    assert tracker.locked_track_id is None

    # 1 person -> Fail-Open bypass active: high raised hand outside face must pass!
    snap_1p = tracker.update(face_boxes=[[0.42, 0.30, 0.14, 0.18]], now_sec=1.0)
    assert snap_1p.bypass_active is True
    assert snap_1p.locked_teacher_id == 1
    assert tracker.locked_track_id == 1
    high_hand = _make_hand_landmarks(cx=0.10, cy=0.12, scale=0.10)
    assert tracker.is_hand_owned_by_locked_teacher(high_hand, snap_1p) is True

    # 2 people -> Teacher stays on left [0.42, 0.30], bystander walks in on right [0.76, 0.28]
    snap_2p = tracker.update(
        face_boxes=[[0.42, 0.30, 0.14, 0.18], [0.76, 0.28, 0.12, 0.16]],
        now_sec=1.5,
    )
    assert snap_2p.person_count == 2
    assert snap_2p.bypass_active is False

    # Bystander hand near ignored face center [0.82, 0.36] should be filtered when locked on teacher
    bystander_hand = _make_hand_landmarks(cx=0.80, cy=0.45, scale=0.11)
    teacher_hand = _make_hand_landmarks(cx=0.46, cy=0.52, scale=0.13)
    assert tracker.is_hand_owned_by_locked_teacher(teacher_hand, snap_2p) is True
    assert tracker.is_hand_owned_by_locked_teacher(bystander_hand, snap_2p) is False

    # Turning Teacher Lock OFF allows all hands
    tracker.set_enabled(False)
    assert tracker.is_hand_owned_by_locked_teacher(bystander_hand, snap_2p) is True


def test_dual_hand_coordinator_zoom_pan_and_offhand_commands():
    coord = DualHandGestureCoordinator()
    coord.set_primary_preference("Right")

    def h_dict(gesture: str, x: float, y: float) -> dict:
        return {
            "gesture": gesture,
            "pose_tag": gesture,
            "cursor_norm": (x, y),
            "pinch_ratio": 0.18 if gesture == "PINCH" else 0.72,
        }

    # 1. Dual Pinch Zoom Out / Zoom In
    st0 = coord.update(h_dict("PINCH", 0.30, 0.50), h_dict("PINCH", 0.60, 0.50), t_sec=1.0)
    assert st0.two_hand_mode == "DUAL_PINCH_ZOOM"
    assert st0.suppress_primary_drawing is True
    assert st0.zoom_factor == pytest.approx(1.0, abs=1e-4)

    st1 = coord.update(h_dict("PINCH", 0.20, 0.50), h_dict("PINCH", 0.70, 0.50), t_sec=1.05)
    assert st1.two_hand_mode == "DUAL_PINCH_ZOOM"
    assert st1.zoom_factor > 1.05

    # 2. Dual Palm Pan
    p0 = coord.update(h_dict("ERASE_PALM", 0.30, 0.50), h_dict("ERASE_PALM", 0.70, 0.50), t_sec=2.0)
    assert p0.two_hand_mode == "DUAL_PALM_PAN"
    p1 = coord.update(h_dict("ERASE_PALM", 0.38, 0.56), h_dict("ERASE_PALM", 0.78, 0.56), t_sec=2.05)
    assert p1.pan_delta_norm[0] > 0.05
    assert p1.pan_delta_norm[1] > 0.04

    # 3. Off-hand fist toggles Pen/Eraser (2-frame stability + 700ms cooldown)
    coord.update(h_dict("HOVER", 0.25, 0.60), h_dict("INDEX", 0.65, 0.50), t_sec=2.9)
    o0 = coord.update(h_dict("FIST_CANCEL", 0.25, 0.60), h_dict("INDEX", 0.65, 0.50), t_sec=3.0)
    assert o0.offhand_command is None
    o1 = coord.update(h_dict("FIST_CANCEL", 0.25, 0.60), h_dict("INDEX", 0.65, 0.50), t_sec=3.05)
    assert o1.offhand_command == "TOGGLE_PEN_ERASER"
    # Immediate next frames are rate-limited by 0.70s cooldown
    o_repeat = coord.update(h_dict("FIST_CANCEL", 0.25, 0.60), h_dict("INDEX", 0.66, 0.50), t_sec=3.15)
    assert o_repeat.offhand_command is None

    # Release and trigger THREE_FINGERS -> UNDO after cooldown
    coord.update(h_dict("HOVER", 0.25, 0.60), h_dict("INDEX", 0.66, 0.50), t_sec=3.5)
    coord.update(h_dict("THREE_FINGERS", 0.25, 0.60), h_dict("INDEX", 0.66, 0.50), t_sec=4.15)
    o_undo = coord.update(h_dict("THREE_FINGERS", 0.25, 0.60), h_dict("INDEX", 0.66, 0.50), t_sec=4.22)
    assert o_undo.offhand_command == "UNDO"


def test_drawing_fsm_supports_index_point_and_two_hand_suppress():
    fsm = ZeroStrayStrokeDrawingFsm(allow_index_point_draw=True)
    # 1. Start drawing via INDEX pose after 3-frame hold debounce (PINCH_ENTER_MIN_FRAMES=3) without pinch
    u1 = fsm.step(100.0, True, (200.0, 220.0), 0.75, "INDEX")
    assert u1.action == "NONE"
    u2 = fsm.step(118.0, True, (215.0, 225.0), 0.75, "INDEX")
    assert u2.action == "NONE"
    u3 = fsm.step(136.0, True, (230.0, 232.0), 0.75, "INDEX")
    assert u3.action == "BEGIN_STROKE"
    assert u3.hud_mode == DrawingHudMode.DRAWING

    # Add points well outside the 105ms tail-trim window before suppressing with 2 hands
    u4a = fsm.step(200.0, True, (250.0, 240.0), 0.75, "INDEX")
    u4b = fsm.step(260.0, True, (270.0, 248.0), 0.75, "INDEX")
    u4c = fsm.step(320.0, True, (285.0, 252.0), 0.75, "INDEX")
    assert u4a.action == "APPEND_POINT"
    assert u4b.action == "APPEND_POINT"
    assert u4c.action == "APPEND_POINT"

    # 2. Two-hand gesture suppresses primary drawing cleanly and preserves older stroke pts
    u5 = fsm.step(380.0, True, (295.0, 255.0), 0.20, "PINCH", suppress_drawing=True)
    assert u5.action == "COMMIT_STROKE"
    assert u5.hud_mode == DrawingHudMode.PAN_ERASE


def test_appdata_paths_and_api_settings_roundtrip(tmp_path: Path):
    assert get_user_data_dir().name == "AirTeachAI"
    assert get_config_file_path().name == "config.json"
    assert get_local_cache_db_path().name == "local_ai_cache.sqlite"
    assert "gemini-2.5-flash" in DEFAULT_FALLBACK_MODELS

    cfg_file = tmp_path / "config.json"
    save_ui_api_settings(ui_api_key="AIzaTestKey1234567890", settings_path=cfg_file, onboarding_skipped=False)
    loaded = load_ui_api_settings(cfg_file)
    assert loaded["ui_gemini_api_key"] == "AIzaTestKey1234567890"
    assert loaded["onboarding_skipped"] == "false"

    rk = ResolvedApiKey(
        key="AIzaTestKey1234567890",
        masked="AIza...7890",
        source="ui",
        ui_key_present=True,
        env_key_present=False,
    )
    assert rk.api_key == rk.key

    hc_ok = GeminiHealthCheckResult(status=ApiConnectionStatus.CONNECTED, model_used="gemini-2.5-flash", latency_ms=42.0, message_vi="OK")
    hc_err = GeminiHealthCheckResult(status=ApiConnectionStatus.ERROR, model_used="gemini-2.5-flash", latency_ms=0.0, message_vi="Err")
    assert hc_ok.ok is True
    assert hc_err.ok is False


def test_whiteboard_agent_json_actions_and_gui_pan_zoom_undo(qapp, monkeypatch):
    import main

    # Avoid spawning a second background LiteRT/WebGPU QThread inside the same pytest process
    # where test_real_mediapipe_init_after_main_import already exercised real model init on Thread 0.
    monkeypatch.setattr(main.CameraHandWorker, "start", lambda self: None)
    win = main.AirTeachMainWindow()
    try:
        # Execute JSON commands normalized [0..1000]
        raw = """Đã vẽ tam giác vuông lên bảng:
```json
{
  "speech_or_text": "Tam giác ABC vuông tại A",
  "actions": [
    {"tool": "draw_shape", "args": {"shape": "triangle", "center_x": 450, "center_y": 500, "radius": 120, "label": "Tam giác ABC"}},
    {"tool": "draw_text", "args": {"content": "$a^2 + b^2 = c^2$", "x": 620, "y": 420}},
    {"tool": "highlight_region", "args": {"x": 350, "y": 380, "w": 200, "h": 200}}
  ]
}
```"""
        actions, speech = WhiteboardAgent.extract_actions_from_response(raw)
        assert "Tam giác ABC" in speech
        assert len(actions) == 3

        before_strokes = len(win.canvas.strokes)
        before_anns = len(win.canvas.annotations)
        win.canvas.execute_whiteboard_actions(actions)
        assert len(win.canvas.strokes) > before_strokes
        assert len(win.canvas.annotations) > before_anns

        # Two-step Undo restores AI drawings/annotations safely
        win.canvas.undo()
        assert len(win.canvas.strokes) == before_strokes
        assert len(win.canvas.annotations) == before_anns

        # Pan / Zoom & Reset Viewport
        win.canvas.apply_zoom_factor(1.4)
        assert win.canvas.zoom_scale > 1.2
        win.canvas.apply_pan_delta_norm(0.10, -0.05)
        assert abs(win.canvas.pan_x) > 1.0
        win.canvas.reset_viewport()
        assert win.canvas.zoom_scale == pytest.approx(1.0)
        assert win.canvas.pan_x == pytest.approx(0.0)
    finally:
        win.shutdown_resources()
        win.deleteLater()
        qapp.processEvents()
