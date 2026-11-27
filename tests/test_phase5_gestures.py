#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5 Verification Suite: Fast 640x480 CV Config, 1-Hand & 2-Hand Gestures,
1-Hand Pinch Object Grab & Drag, Dual-Fist & Dual-Palm Pan/Scroll, Live Gesture HUD.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

import main as app_main
from cv_engine.drawing_fsm import DrawingHudMode
from cv_engine.hand_gesture import DualHandGestureCoordinator, SlideNavigationGestureDetector
from cv_engine.vision_config import load_vision_config


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_camera_default_640x480_and_rust_dual_hand_gestures():
    vcfg = load_vision_config()
    assert vcfg.camera.width == 640
    assert vcfg.camera.height == 480
    assert vcfg.camera.prefer_hd is False
    assert vcfg.hand_tracking.num_hands == 2

    # Verify Dual Pinch Zoom AND Dual Fist Pan AND Dual Open Palm Pan (Rust-backed)
    coord = DualHandGestureCoordinator(primary_pref="right")

    # 1. Dual Pinch Zoom Out -> Zoom In
    st1 = coord.update(
        {"pose_tag": "PINCH", "cursor_norm": (0.35, 0.50), "pinch_ratio": 0.20, "handedness": "left"},
        {"pose_tag": "PINCH", "cursor_norm": (0.65, 0.50), "pinch_ratio": 0.20, "handedness": "right"},
        t_sec=1.0,
    )
    assert st1.two_hand_mode == "DUAL_PINCH_ZOOM"
    st2 = coord.update(
        {"pose_tag": "PINCH", "cursor_norm": (0.25, 0.50), "pinch_ratio": 0.20, "handedness": "left"},
        {"pose_tag": "PINCH", "cursor_norm": (0.75, 0.50), "pinch_ratio": 0.20, "handedness": "right"},
        t_sec=1.05,
    )
    assert st2.two_hand_mode == "DUAL_PINCH_ZOOM"
    assert st2.zoom_factor > 1.05
    assert st2.suppress_primary_drawing is True

    # 2. Dual Fist Parallel Pan
    coord.reset()
    f1 = coord.update(
        {"pose_tag": "FIST", "cursor_norm": (0.35, 0.50), "pinch_ratio": 0.60, "handedness": "left"},
        {"pose_tag": "FIST", "cursor_norm": (0.65, 0.50), "pinch_ratio": 0.60, "handedness": "right"},
        t_sec=2.0,
    )
    assert f1.two_hand_mode == "DUAL_PALM_PAN"
    f2 = coord.update(
        {"pose_tag": "FIST", "cursor_norm": (0.45, 0.42), "pinch_ratio": 0.60, "handedness": "left"},
        {"pose_tag": "FIST", "cursor_norm": (0.75, 0.42), "pinch_ratio": 0.60, "handedness": "right"},
        t_sec=2.05,
    )
    assert f2.two_hand_mode == "DUAL_PALM_PAN"
    assert f2.pan_delta_norm[0] > 0.03 and f2.pan_delta_norm[1] < -0.02

    # 3. Swipe Left / Right with cooldown anti-double-flip
    nav = SlideNavigationGestureDetector(swipe_cooldown_sec=0.65)
    ev = None
    for i, x in enumerate([0.25, 0.42, 0.59, 0.78]):
        res = nav.update(
            {"pose_tag": "OPEN_PALM", "cursor_norm": (x, 0.50), "pinch_ratio": 0.90},
            suppress=False,
            is_drawing=False,
            t_sec=10.0 + i * 0.04,
        )
        if res.get("nav_event"):
            ev = res["nav_event"]
    assert ev in ("SLIDE_NEXT", "SLIDE_PREV")


def test_canvas_1hand_draw_hover_pinch_object_grab_and_live_hud(qapp):
    canvas = app_main.SmartboardCanvas()
    canvas.resize(1000, 600)
    assert canvas.drawing_fsm.allow_index_point_draw is True

    # Place a Shape object at (400, 260, 140, 100) -> center (470, 310) -> norm (0.47, 0.517)
    shp = canvas.add_shape_object("rect", 400.0, 260.0, 140.0, 100.0, select=False)

    # 1. Two-finger V -> Hover/Pointer mode (no drawing)
    upd_v = canvas.apply_cv_fsm_packet({
        "gesture": "TWO_FINGERS",
        "pose_tag": "TWO_FINGERS",
        "pinch_ratio": 0.75,
        "cursor_norm": (0.20, 0.30),
        "dual_hand": {"two_hand_mode": "SINGLE"},
    })
    assert upd_v.hud_mode == DrawingHudMode.HOVER
    assert "Con Trỏ" in canvas.live_gesture_hud_text

    # 2. 1-Hand Pinch over `shp` -> Grabs & Moves `shp` without drawing stray ink!
    strokes_before = len(canvas.strokes)
    canvas.apply_cv_fsm_packet({
        "gesture": "PINCH",
        "pose_tag": "PINCH",
        "pinch_ratio": 0.18,
        "cursor_norm": (0.47, 0.517),
        "dual_hand": {"two_hand_mode": "SINGLE"},
    })
    assert canvas._cv_grab_obj_id == shp.id
    assert "Kéo Object" in canvas.live_gesture_hud_text

    canvas.apply_cv_fsm_packet({
        "gesture": "PINCH",
        "pose_tag": "PINCH",
        "pinch_ratio": 0.18,
        "cursor_norm": (0.62, 0.60),
        "dual_hand": {"two_hand_mode": "SINGLE"},
    })
    moved = canvas.scene.get_by_id(shp.id)
    assert moved is not None and moved.x > 520.0
    assert len(canvas.strokes) == strokes_before  # No stray ink created

    # Release pinch
    canvas.apply_cv_fsm_packet({
        "gesture": "TWO_FINGERS",
        "pose_tag": "TWO_FINGERS",
        "pinch_ratio": 0.75,
        "cursor_norm": (0.62, 0.60),
        "dual_hand": {"two_hand_mode": "SINGLE"},
    })
    assert canvas._cv_grab_obj_id is None

    # 3. 1-Finger INDEX draw starts drawing after debounce frames
    for step_i in range(4):
        upd_idx = canvas.apply_cv_fsm_packet({
            "gesture": "INDEX",
            "pose_tag": "INDEX",
            "pinch_ratio": 0.72,
            "cursor_norm": (0.15 + step_i * 0.03, 0.25),
            "dual_hand": {"two_hand_mode": "SINGLE"},
        })
    assert upd_idx.hud_mode == DrawingHudMode.DRAWING
    assert "Đang Vẽ" in canvas.live_gesture_hud_text

    # 4. Two-Hand Zoom HUD text
    canvas.apply_zoom_factor(1.45)
    canvas.apply_cv_fsm_packet({
        "gesture": "PINCH",
        "pose_tag": "PINCH",
        "pinch_ratio": 0.20,
        "cursor_norm": (0.50, 0.50),
        "dual_hand": {"two_hand_mode": "DUAL_PINCH_ZOOM", "suppress_primary_drawing": True},
    })
    assert "2 Tay: Đang Zoom 145%" in canvas.live_gesture_hud_text
    canvas.close()
