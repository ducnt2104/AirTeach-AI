#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive End-to-End Hard Verification Suite (`tests/verify_all_systems.py`)
for AirTeach AI (Strict Audit & Zero-Skip Enforcement):
  1. Rust Native Extension (`airteach_core_rs` v2.0.0) import & < 1.0 ms/frame kinematics benchmark
     (One-Euro + 3D dual-hand gesture classification + Catmull-Rom spline + angle-snapped LINE & ARROW)
  2. Synthetic 1-Hand, Air-Swipe & Dual-Hand Gesture State Transitions wired to AirTeachMainWindow & SmartboardCanvas:
     HOVER, DRAW, ERASE, SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN, TWO_HAND_ZOOM, TWO_HAND_PAN
  3. Right-Click Radial Pie Menu: all 8 sectors (N..NW), 20px cancel deadzone, action_triggered signal, Esc cancel
  4. Universal Real Document & Image Rendering (.pdf, .pptx, .docx, .png, .svg, image folder) asserting
     width >= 1000 & pixel std_dev > 5.0 + movable/resizable board image + per-page ink + valid annotated PDF export
  5. Board Navigation ([25%-400%] zoom clamp, Middle/Space+drag pan, Shift+Wheel H-pan, Fit Width/Page, LOD cache)
  6. Local AI Manager & Hybrid Router (HTTPS catalog, custom URL normalization, HTTP 206 Range resume + SHA-256,
     clean LOCAL/CLOUD/HYBRID mode switching & GGUF routing priority)
  7. AI Chatbot Deep Response & Streaming (max_output_tokens >= 4096, system_instruction + history forwarded,
     LaTeX preserved through WhiteboardAgent, multi-paragraph Markdown+Table+LaTeX streaming without truncation)
  8. PySide6 Headless/Offscreen GUI Smoke Test (Cold startup < 1.5s, Canvas > 80%, toggle ALL collapsible panels,
     open 4-tab [? Hướng Dẫn] UserGuideDialog & [Quản Lý AI Local] LocalAiManagerDialog, clean exit 0)

Run directly (`python tests/verify_all_systems.py`) or via `pytest`.
"""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
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


# =============================================================================
# 1. RUST NATIVE CORE BENCHMARK (< 1.0 ms / frame)
# =============================================================================
def test_rust_core_import_and_1khz_benchmark() -> None:
    """Verify native Rust `airteach_core_rs` v2.0.0 imports & runs < 1.0 ms per frame."""
    import airteach_core_rs
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
    assert "2.0.0" in RUST_CORE_VERSION, f"Expected airteach_core_rs 2.0.0, got {RUST_CORE_VERSION}"
    assert hasattr(airteach_core_rs, "backend_info")
    assert hasattr(airteach_core_rs, "joint_angle_3d_deg")
    assert hasattr(airteach_core_rs, "cosine_similarity_vec")
    assert rust_bridge._rs is not None

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

    sample_poly = [(10.0 + k * 8.0, 20.0 + math.sin(k * 0.4) * 14.0) for k in range(20)]

    # Warmup
    for i in range(25):
        t = i * 0.016
        classify_hand_3d_fast(pts_list_a, None, True, 0.28, 0.42, "right", 0.20)
        flt.filter(0.4 + 0.001 * i, 0.5, t)
        extrap.ingest_vision((0.4, 0.5), t)
        extrap.sample_60hz(t + 0.008)
        compute_dual_hand_kinematics_fast((0.4, 0.5), (0.6, 0.5), 0.30, (0.50, 0.50))
        swipe.push_and_check(0.4 + 0.005 * i, 0.5, t, "OPEN_PALM")
        catmull_rom_spline_fast(sample_poly, 4)

    iters = 1000
    t0 = time.perf_counter()
    for i in range(iters):
        t = 1.0 + i * 0.016
        _cls = classify_hand_3d_fast(pts_list_a, None, True, 0.28, 0.42, "right", 0.20)
        fx, fy = flt.filter(0.42 + 0.0005 * (i % 40), 0.51, t)
        extrap.ingest_vision((fx, fy), t)
        _pt_out, _ex = extrap.sample_60hz(t + 0.008)
        _dual = compute_dual_hand_kinematics_fast((fx, fy), (0.72, 0.50), 0.31, (0.55, 0.50))
        _sw = swipe.push_and_check(fx, fy, t, "OPEN_PALM")
        _roi = compute_roi_crop_box_fast(fx - 0.08, fy - 0.08, fx + 0.08, fy + 0.08, 0.01, 0.0, 2.2, 0.45)
        _sp = catmull_rom_spline_fast(sample_poly, 4)
        _ = (_cls, _pt_out, _ex, _dual, _sw, _roi, _sp)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_ms / iters
    print(f"[BENCH] Rust Core full pipeline 1000-frame avg = {avg_ms:.4f} ms/frame ({elapsed_ms:.1f} ms total)", flush=True)
    assert avg_ms < 1.0, f"Rust kinematics+spline too slow: {avg_ms:.4f} ms/frame"

    # Shape + angle snapping + arrow non-stealing + spline + frame luma checks
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    circle_pts = [(float(300.0 + 90.0 * np.cos(a)), float(300.0 + 90.0 * np.sin(a))) for a in theta]
    circle_pts.append(circle_pts[0])
    simplified = rdp_simplify_fast(circle_pts, 2.5)
    assert simplified is not None and 6 <= len(simplified) <= len(circle_pts)
    splined = catmull_rom_spline_fast(simplified, 4)
    assert splined is not None and len(splined) > len(simplified)

    snapped_circle = snap_geometric_shape_fast(circle_pts, 0.18, 0.78)
    assert snapped_circle is not None and snapped_circle[0] == "CIRCLE"

    # Slightly slanted horizontal line must snap cleanly to exact 0 degrees
    slanted_line = [(100.0 + i * 10.0, 200.0 + i * 0.4 + math.sin(i) * 0.4) for i in range(30)]
    res_line = beautify_stroke(slanted_line)
    assert res_line.shape_type == "LINE"
    assert abs(res_line.beautified_points[-1][1] - res_line.beautified_points[0][1]) < 1e-3

    # Arrow stroke must NOT be stolen as LINE
    def _seg(p0: Tuple[float, float], p1: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
        return [(p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n) for k in range(n + 1)]

    arrow_pts = (
        _seg((80.0, 220.0), (340.0, 220.0), 28)
        + _seg((340.0, 220.0), (305.0, 194.0), 10)
        + _seg((305.0, 194.0), (340.0, 220.0), 8)
        + _seg((340.0, 220.0), (305.0, 246.0), 10)
    )
    res_arrow = beautify_stroke(arrow_pts)
    assert res_arrow.shape_type == "ARROW", f"Expected ARROW, got {res_arrow.shape_type}"

    gray_frame = rng.integers(35, 190, size=(120, 160), dtype=np.uint8)
    luma_stats = analyze_frame_luma_fast(gray_frame, is_rgb=False, stride=2)
    assert luma_stats is not None
    mean_y, p5_y, p95_y = luma_stats
    assert 20.0 < mean_y < 230.0 and p95_y >= mean_y >= p5_y
    rgb_frame = np.stack([gray_frame, gray_frame, gray_frame], axis=-1)
    boosted = apply_gamma_contrast_fast(rgb_frame, gamma=1.15, alpha=1.05, beta=4.0)
    assert boosted is not None and boosted.shape == rgb_frame.shape and boosted.dtype == np.uint8


# =============================================================================
# 2. SYNTHETIC 1-HAND, AIR-SWIPE & 2-HAND GESTURE STATE TRANSITIONS
# =============================================================================
def test_synthetic_1hand_swipe_and_dual_hand_gestures() -> None:
    """Verify HOVER, DRAW, ERASE, SWIPE_LEFT/RIGHT/UP/DOWN, TWO_HAND_ZOOM & TWO_HAND_PAN end-to-end."""
    _ensure_qapp()
    import pymupdf as fitz
    import main
    from cv_engine.document_presenter import LectureDocument
    from cv_engine.drawing_fsm import DrawingHudMode, ZeroStrayStrokeDrawingFsm
    from cv_engine.hand_gesture import DualHandGestureCoordinator, SlideNavigationGestureDetector

    # Part A: Drawing FSM deterministic state transitions (HOVER -> DRAW -> COMMIT -> ERASE)
    fsm = ZeroStrayStrokeDrawingFsm(pinch_enter=0.32, pinch_exit=0.42, pinch_enter_frames=3, tail_trim_ms=100.0)
    u_hover = fsm.step(100.0, True, (200.0, 200.0), 0.72, "INDEX")
    assert u_hover.hud_mode == DrawingHudMode.HOVER

    # Enter DRAW via 3 consecutive pinch frames
    for idx in range(3):
        u_draw = fsm.step(140.0 + idx * 20.0, True, (200.0 + idx * 12.0, 220.0), 0.22, "PINCH")
    assert u_draw.hud_mode == DrawingHudMode.DRAWING and u_draw.action == "BEGIN_STROKE"
    for k in range(1, 10):
        u_app = fsm.step(200.0 + k * 25.0, True, (236.0 + k * 14.0, 220.0 + k * 4.0), 0.24, "PINCH")
        assert u_app.hud_mode == DrawingHudMode.DRAWING
    u_commit = fsm.step(520.0, True, (390.0, 265.0), 0.65, "INDEX")
    assert u_commit.action == "COMMIT_STROKE" and u_commit.hud_mode == DrawingHudMode.HOVER
    assert len(u_commit.committed_points) >= 2

    # Transition to ERASE via 5-finger OPEN_PALM (past post-commit cancel window)
    u_erase = fsm.step(1500.0, True, (300.0, 230.0), 0.85, "OPEN_PALM")
    assert u_erase.hud_mode == DrawingHudMode.PAN_ERASE and u_erase.action == "ERASE"

    # Part B: Air-Swipe Detector (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN)
    nav = SlideNavigationGestureDetector(swipe_cooldown_sec=0.40, vertical_cooldown_sec=0.35)

    def _run_swipe(coords: List[Tuple[float, float]], t0: float, pose: str = "OPEN_PALM") -> Optional[Dict[str, Any]]:
        fired: Optional[Dict[str, Any]] = None
        for idx, xy in enumerate(coords):
            out = nav.update({"pose_tag": pose, "pinch_ratio": 0.85, "cursor_norm": xy}, t_sec=t0 + idx * 0.04)
            if out.get("nav_event"):
                fired = out
                break
        return fired

    ev_left = _run_swipe([(0.78, 0.50), (0.66, 0.50), (0.54, 0.50), (0.42, 0.50)], t0=1.0)
    assert ev_left is not None and ev_left["nav_event"] == "SLIDE_NEXT" and ev_left["swipe_direction"] == "LEFT"

    ev_right = _run_swipe([(0.25, 0.50), (0.38, 0.50), (0.52, 0.50), (0.66, 0.50)], t0=2.0)
    assert ev_right is not None and ev_right["nav_event"] == "SLIDE_PREV" and ev_right["swipe_direction"] == "RIGHT"

    ev_up = _run_swipe([(0.50, 0.72), (0.50, 0.60), (0.50, 0.48), (0.50, 0.36)], t0=3.0)
    assert ev_up is not None and ev_up["nav_event"] == "SCROLL_DOWN" and ev_up["swipe_direction"] == "UP"

    ev_down = _run_swipe([(0.50, 0.28), (0.50, 0.40), (0.50, 0.52), (0.50, 0.64)], t0=4.0)
    assert ev_down is not None and ev_down["nav_event"] == "SCROLL_UP" and ev_down["swipe_direction"] == "DOWN"

    # Part C: Dual-Hand Coordinator (TWO_HAND_ZOOM & TWO_HAND_PAN)
    coord = DualHandGestureCoordinator(primary_pref="right")
    z1 = coord.update(
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.35, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.65, 0.50)},
        t_sec=5.0,
    )
    assert z1.two_hand_mode == "DUAL_PINCH_ZOOM" and z1.suppress_primary_drawing is True
    z2 = coord.update(
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.25, 0.50)},
        {"pose_tag": "PINCH", "pinch_ratio": 0.20, "cursor_norm": (0.75, 0.50)},
        t_sec=5.05,
    )
    assert z2.two_hand_mode == "DUAL_PINCH_ZOOM" and z2.zoom_factor > 1.05

    p1 = coord.update(
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.35, 0.50)},
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.65, 0.50)},
        t_sec=5.10,
    )
    assert p1.two_hand_mode == "DUAL_PALM_PAN"
    p2 = coord.update(
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.45, 0.42)},
        {"pose_tag": "OPEN_PALM", "pinch_ratio": 0.85, "cursor_norm": (0.75, 0.42)},
        t_sec=5.15,
    )
    assert p2.two_hand_mode == "DUAL_PALM_PAN" and p2.pan_delta_norm[0] > 0.05 and p2.pan_delta_norm[1] < -0.05

    # Part D: Live MainWindow._on_camera_packet wiring test (slide flip, scroll, 2-hand zoom/pan, zero accidental erase!)
    orig_start = main.CameraHandWorker.start
    main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
    win = main.AirTeachMainWindow()
    try:
        win.resize(1280, 760)
        win.stack.setCurrentIndex(1)
        win.show()
        with tempfile.TemporaryDirectory() as tmp:
            pdf_p = Path(tmp) / "multipage_nav.pdf"
            pdf_d = fitz.open()
            for pg_i in range(3):
                pg = pdf_d.new_page(width=600, height=840)
                pg.insert_text((70, 120), f"Page {pg_i + 1} AirTeach Gesture Test", fontsize=24)
            pdf_d.save(str(pdf_p))
            pdf_d.close()

            ldoc = LectureDocument.open(pdf_p, cache_dir=Path(tmp) / "cache")
            try:
                win.canvas.attach_lecture_document(ldoc)
                # Put a real ink stroke on Page 0 to confirm SWIPE_LEFT with OPEN_PALM does NOT erase it!
                win.canvas.strokes.append({
                    "points": [QPointF(500.0, 350.0), QPointF(560.0, 360.0)],
                    "color": win.canvas.ink_color,
                    "thickness": 4,
                    "is_highlighter": False,
                })
                assert len(win.canvas.slide_annotations[0]["strokes"]) == 1

                # Swipe Left -> Slide 0 -> Slide 1 + HUD toast + Page 0 stroke preserved!
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "ERASE",
                    "pose_tag": "OPEN_PALM",
                    "pinch_ratio": 0.85,
                    "cursor_norm": (0.45, 0.50),
                    "nav_event": "SLIDE_NEXT",
                    "swipe_direction": "LEFT",
                    "scroll_delta_norm": 0.0,
                    "dual_hand": {"two_hand_mode": "NONE"},
                })
                assert win.canvas.current_slide_idx == 1
                assert win.canvas.gesture_toast is not None and "Tiếp Theo" in win.canvas.gesture_toast["text"]
                assert len(win.canvas.slide_annotations[0]["strokes"]) == 1, "Air-swipe must NOT erase ink!"

                # Swipe Right -> Slide 1 -> Slide 0
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "ERASE",
                    "pose_tag": "OPEN_PALM",
                    "pinch_ratio": 0.85,
                    "cursor_norm": (0.65, 0.50),
                    "nav_event": "SLIDE_PREV",
                    "swipe_direction": "RIGHT",
                    "scroll_delta_norm": 0.0,
                    "dual_hand": {"two_hand_mode": "NONE"},
                })
                assert win.canvas.current_slide_idx == 0

                # Continuous Scroll mode: SWIPE_UP -> scrolls down, SWIPE_DOWN -> scrolls up
                win.canvas.set_continuous_scroll_mode(True)
                y0 = win.canvas.doc_scroll_y
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "ERASE",
                    "pose_tag": "OPEN_PALM",
                    "pinch_ratio": 0.85,
                    "cursor_norm": (0.50, 0.36),
                    "nav_event": "SCROLL_DOWN",
                    "swipe_direction": "UP",
                    "scroll_delta_norm": 0.38,
                    "dual_hand": {"two_hand_mode": "NONE"},
                })
                y1 = win.canvas.doc_scroll_y
                assert y1 > y0 + 100.0
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "ERASE",
                    "pose_tag": "OPEN_PALM",
                    "pinch_ratio": 0.85,
                    "cursor_norm": (0.50, 0.64),
                    "nav_event": "SCROLL_UP",
                    "swipe_direction": "DOWN",
                    "scroll_delta_norm": -0.38,
                    "dual_hand": {"two_hand_mode": "NONE"},
                })
                assert win.canvas.doc_scroll_y < y1

                # Two-Hand Zoom & Two-Hand Pan via _on_camera_packet
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "CLICK",
                    "pose_tag": "PINCH",
                    "pinch_ratio": 0.22,
                    "cursor_norm": (0.50, 0.50),
                    "dual_hand": {
                        "two_hand_mode": "DUAL_PINCH_ZOOM",
                        "zoom_factor": 1.15,
                        "zoom_center_norm": (0.50, 0.50),
                        "suppress_primary_drawing": True,
                        "status_label_vi": "2 Tay: Thu/Phóng",
                    },
                })
                assert abs(win.canvas.zoom_scale - 1.15) < 0.02
                assert win.canvas.cursor_mode == "TWO_HAND_NAV"

                px_before, py_before = win.canvas.pan_x, win.canvas.pan_y
                win._on_camera_packet({
                    "ok": True,
                    "gesture": "ERASE",
                    "pose_tag": "OPEN_PALM",
                    "pinch_ratio": 0.85,
                    "cursor_norm": (0.55, 0.45),
                    "dual_hand": {
                        "two_hand_mode": "DUAL_PALM_PAN",
                        "pan_delta_norm": (0.08, -0.06),
                        "suppress_primary_drawing": True,
                        "status_label_vi": "2 Tay: Kéo trượt bảng",
                    },
                })
                assert win.canvas.pan_x > px_before + 30.0 and win.canvas.pan_y < py_before - 20.0
                assert win.canvas.cursor_mode == "TWO_HAND_NAV"
            finally:
                ldoc.close()
                win.canvas.lecture_doc = None
    finally:
        main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
        win.shutdown_resources()
        win.close()


# =============================================================================
# 3. RIGHT-CLICK RADIAL PIE MENU (ALL 8 SECTORS + 20PX DEADZONE + SIGNAL)
# =============================================================================
def test_radial_pie_menu_angles_deadzone_and_canvas_mouse() -> None:
    """Verify Right-Click Radial Pie Menu across all 8 sectors, 22px deadzone, outer sub-ring & Qt signals."""
    _ensure_qapp()
    import main
    from cv_engine.radial_pie_menu import PIE_MENU_SLICES, resolve_pie_slice, resolve_pie_sub_item

    assert resolve_pie_slice(0.0, 0.0, deadzone_px=22.0) is None
    assert resolve_pie_slice(12.0, -14.0, deadzone_px=22.0) is None

    expected_dirs = [
        (0.0, -85.0, "PEN"),
        (62.0, -62.0, "ERASER"),
        (85.0, 0.0, "SELECT"),
        (62.0, 62.0, "PAN"),
        (0.0, 85.0, "TEXT"),
        (-62.0, 62.0, "SHAPES"),
        (-85.0, 0.0, "UNDO"),
        (-62.0, -62.0, "CLEAR_OR_NEW"),
    ]
    for exp_idx, (dx, dy, exp_action) in enumerate(expected_dirs):
        idx = resolve_pie_slice(dx, dy, deadzone_px=22.0)
        assert idx == exp_idx
        assert PIE_MENU_SLICES[idx].action_id == exp_action

    # Verify outer sub-ring resolution (e.g., PEN -> center color #F43F5E at North, r=170px)
    sub_pen = resolve_pie_sub_item(0.0, -170.0, 0)
    assert sub_pen is not None and sub_pen.action_id.startswith("COLOR:")

    canvas = main.SmartboardCanvas()
    canvas.resize(1000, 700)
    canvas.show()
    center = QPointF(500.0, 350.0)

    emitted_actions: List[str] = []
    emitted_canvas_tools: List[str] = []
    canvas.pie_menu.action_triggered.connect(emitted_actions.append)
    canvas.pie_tool_selected.connect(emitted_canvas_tools.append)

    def _mouse_ev(etype, pos, btn, btns):
        return QMouseEvent(etype, pos, pos, btn, btns, Qt.NoModifier)

    # Test ALL 8 directional sectors end-to-end through Qt mouse Press -> Move -> Release
    for dx, dy, exp_action in expected_dirs:
        emitted_actions.clear()
        target_pt = QPointF(center.x() + dx, center.y() + dy)
        canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
        assert canvas.pie_menu.isVisible() is True
        canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, target_pt, Qt.NoButton, Qt.RightButton))
        canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, target_pt, Qt.RightButton, Qt.NoButton))
        assert canvas.pie_menu.isVisible() is False
        assert emitted_actions == [exp_action], f"Sector ({dx},{dy}) expected {exp_action}, got {emitted_actions}"
        if exp_action in ("PEN", "ERASER", "SELECT", "PAN", "TEXT"):
            assert canvas.active_tool == exp_action
        elif exp_action == "SHAPES":
            assert canvas.active_tool == "SHAPE"

    assert len(emitted_canvas_tools) == 8

    # Test outer sub-ring trigger via mouse drag (North -> r=172px -> quick color change)
    emitted_actions.clear()
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, QPointF(center.x(), center.y() - 90.0), Qt.NoButton, Qt.RightButton))
    canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, QPointF(center.x(), center.y() - 172.0), Qt.NoButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, QPointF(center.x(), center.y() - 172.0), Qt.RightButton, Qt.NoButton))
    assert emitted_actions and emitted_actions[0].startswith("COLOR:")

    # Deadzone cancel (< 22px move) emits NOTHING and preserves active tool
    canvas.active_tool = "ERASER"
    emitted_actions.clear()
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    canvas.mouseMoveEvent(_mouse_ev(QMouseEvent.Type.MouseMove, QPointF(508.0, 357.0), Qt.NoButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse_ev(QMouseEvent.Type.MouseButtonRelease, QPointF(508.0, 357.0), Qt.RightButton, Qt.NoButton))
    assert canvas.pie_menu.isVisible() is False
    assert emitted_actions == []
    assert canvas.active_tool == "ERASER"

    # Esc cancels an open Pie Menu cleanly
    canvas.mousePressEvent(_mouse_ev(QMouseEvent.Type.MouseButtonPress, center, Qt.RightButton, Qt.RightButton))
    assert canvas.pie_menu.isVisible() is True
    canvas.trigger_esc_cancel()
    assert canvas.pie_menu.isVisible() is False
    canvas.close()


# =============================================================================
# 4. REAL DOCUMENT & IMAGE ENGINE (width >= 1000, std_dev > 5.0, PDF EXPORT)
# =============================================================================
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

    # 4. Real High-Res PNG (>= 1280px wide)
    png_p = tmp_dir / "diagram.png"
    img = Image.new("RGB", (1280, 720), (15, 23, 42))
    dr = ImageDraw.Draw(img)
    dr.rectangle([40, 40, 1240, 680], outline=(6, 182, 212), width=8)
    dr.ellipse([360, 120, 920, 600], fill=(79, 70, 229), outline=(0, 245, 212), width=6)
    img.save(str(png_p))
    out["png"] = png_p

    # 5. Real SVG
    svg_p = tmp_dir / "circuit.svg"
    svg_p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" viewBox="0 0 600 400">'
        '<rect x="12" y="12" width="576" height="376" rx="18" fill="#1E293B" stroke="#06B6D4" stroke-width="5"/>'
        '<circle cx="190" cy="200" r="85" fill="#4F46E5" stroke="#00F5D4" stroke-width="4"/>'
        '<path d="M 300 200 L 500 200 L 460 165 M 500 200 L 460 235" stroke="#FEE440" stroke-width="6" fill="none"/>'
        "</svg>",
        encoding="utf-8",
    )
    out["svg"] = svg_p

    # 6. Multi-Image Folder Deck
    folder_p = tmp_dir / "gallery_slides"
    folder_p.mkdir(parents=True, exist_ok=True)
    for idx, bg_rgb in enumerate(((20, 45, 90), (45, 20, 85), (15, 75, 70)), start=1):
        im = Image.new("RGB", (1280, 720), bg_rgb)
        d = ImageDraw.Draw(im)
        d.rectangle([40, 40, 1240, 680], outline=(255, 255, 255), width=6)
        im.save(str(folder_p / f"slide_{idx}.png"))
    out["folder"] = folder_p

    return out


def _qimage_pixel_std(qimg) -> float:
    rgba = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
    ptr = rgba.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=rgba.width() * rgba.height() * 4).reshape((rgba.height(), rgba.width(), 4))
    return float(np.std(arr[:, :, :3]))


def test_universal_real_document_and_image_rendering() -> None:
    """Verify real rendering of PDF, PPTX, DOCX, PNG, SVG (width >= 1000, std_dev > 5.0), board images & PDF export."""
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
                qimg = doc.render_page_qimage(0, scale=2.25)
                std_val = _qimage_pixel_std(qimg)
                print(
                    f"[DOC] {fmt_key.upper()} backend={doc.backend_used} size={qimg.width()}x{qimg.height()} std={std_val:.2f}",
                    flush=True,
                )
                assert qimg.width() >= 1000, f"{fmt_key.upper()} rendered width too small: {qimg.width()}"
                assert std_val > 5.0, f"{fmt_key.upper()} rendered blank frame (std={std_val:.2f})"
                snap_bytes, snap_pm = canvas.export_board_png()
                assert len(snap_bytes) > 1000 and not snap_pm.isNull()
            finally:
                doc.close()
                canvas.lecture_doc = None

        # Single PNG & SVG image slide loading (assert width >= 1000 & std_dev > 5.0)
        assert canvas.load_document_file(samples["png"]) is True
        png_img = canvas.slides[canvas.current_slide_idx]["pixmap"].toImage()
        png_std = _qimage_pixel_std(png_img)
        print(f"[DOC] PNG size={png_img.width()}x{png_img.height()} std={png_std:.2f}", flush=True)
        assert png_img.width() >= 1000 and png_std > 5.0

        assert canvas.load_document_file(samples["svg"]) is True
        svg_img = canvas.slides[canvas.current_slide_idx]["pixmap"].toImage()
        svg_std = _qimage_pixel_std(svg_img)
        assert svg_img.width() >= 1000 and svg_std > 5.0

        # Multi-image folder slide deck + per-page ink isolation
        assert canvas.load_image_folder(samples["folder"]) is True
        assert len(canvas.slides) == 3
        canvas.strokes.append({
            "points": [QPointF(100.0, 100.0), QPointF(260.0, 180.0)],
            "color": canvas.ink_color,
            "thickness": 4,
            "is_highlighter": False,
        })
        canvas.set_current_slide(1)
        assert len(canvas.strokes) == 0, "Slide 1 ink must be isolated from Slide 0"

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


# =============================================================================
# 5. BOARD NAVIGATION, ZOOM [25%-400%], SPACE+DRAG & LOD ZERO-RERASTER CACHE
# =============================================================================
def test_navigation_zoom_range_and_lod_cache() -> None:
    """Verify [25%..400%] zoom clamp, Middle/Space+drag pan, Shift+Wheel H-pan, Fit Width/Page & LOD cache."""
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

            # Middle-mouse drag panning
            canvas.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(400.0, 300.0), QPointF(400.0, 300.0), Qt.MiddleButton, Qt.MiddleButton, Qt.NoModifier))
            canvas.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(480.0, 350.0), QPointF(480.0, 350.0), Qt.NoButton, Qt.MiddleButton, Qt.NoModifier))
            canvas.mouseReleaseEvent(QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(480.0, 350.0), QPointF(480.0, 350.0), Qt.MiddleButton, Qt.NoButton, Qt.NoModifier))
            assert abs(canvas.pan_x - 80.0) < 1e-3 and abs(canvas.pan_y - 50.0) < 1e-3

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


# =============================================================================
# 6. LOCAL AI MANAGER, HYBRID ROUTER & AI CHATBOT DEEP STREAMING
# =============================================================================
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
    """Verify max_output_tokens >= 4096, deep streaming, Markdown+LaTeX, catalog, Range resume & Hybrid router."""
    _ensure_qapp()
    import main
    from cv_engine.gemini_client import DEFAULT_MAX_OUTPUT_TOKENS, ResilientGeminiClient
    from cv_engine.gguf_model_manager import (
        CURATED_GGUF_CATALOG,
        DownloadProgress,
        ResumableModelDownloader,
        list_downloaded_models,
        normalize_model_download_url,
    )
    from cv_engine.local_ai_manager import AiRoutingMode, LocalAIManager, LocalResponseCache
    from cv_engine.output_sanitizer import OutputSanitizer
    from cv_engine.whiteboard_agent import WhiteboardAgent

    assert DEFAULT_MAX_OUTPUT_TOKENS >= 4096

    # Verify catalog includes required Vietnamese-capable & STEM GGUF models with valid HTTPS URLs
    catalog_ids = {s.model_id for s in CURATED_GGUF_CATALOG}
    assert "qwen2.5-1.5b-instruct-q4_k_m" in catalog_ids
    assert "qwen2.5-3b-instruct-q4_k_m" in catalog_ids
    assert "phi-3.5-mini-instruct-q4_k_m" in catalog_ids
    for spec in CURATED_GGUF_CATALOG:
        assert spec.download_url.startswith("https://") and spec.filename.endswith(".gguf")

    # Custom URL validation & normalization
    assert "/resolve/main/" in normalize_model_download_url(
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    assert "/raw/v1.0/" in normalize_model_download_url(
        "https://github.com/org/repo/blob/v1.0/model.gguf"
    )
    invalid_rejected = False
    try:
        normalize_model_download_url("ftp://malicious.invalid/model.gguf")
    except ValueError:
        invalid_rejected = True
    assert invalid_rejected is True

    # Multi-paragraph Markdown + Table + LaTeX streaming test via instrumented Gemini client
    deep_chunks = [
        "## 1. Phân tích Định Luật II Newton\n\n",
        "Gia tốc của một vật cùng hướng với hợp lực $\\vec{F}$ tác dụng lên vật và có độ lớn:\n\n",
        "$$\\vec{a} = \\frac{\\vec{F}}{m}$$\n\n",
        "## 2. Bảng Tổng Hợp Đại Lượng\n",
        "| Đại lượng | Ký hiệu | Đơn vị chuẩn |\n",
        "| --- | --- | --- |\n",
        "| Hợp lực | $F$ | Newton (N) |\n",
        "| Khối lượng | $m$ | Kilogam (kg) |\n",
        "| Gia tốc | $a$ | $m/s^2$ |\n\n",
        "## 3. Ví Dụ Minh Họa & Lưu Ý Sai Lầm\n",
        "- Khi $F = 20\\text{ N}$ và $m = 4\\text{ kg}$ -> $a = 5\\text{ m/s}^2$.\n",
        "- Lưu ý: Luôn chiếu phương trình vectơ lên hệ trục tọa độ $Oxy$ trước khi thế số.\n",
    ]
    captured_calls: List[Dict[str, Any]] = []

    class _FakeStreamEvent:
        def __init__(self, txt: str) -> None:
            self.text = txt

    class _FakeModels:
        def generate_content_stream(self, **kwargs):
            captured_calls.append(kwargs)
            for c in deep_chunks:
                yield _FakeStreamEvent(c)

        def generate_content(self, **kwargs):
            captured_calls.append(kwargs)
            return _FakeStreamEvent("".join(deep_chunks))

    class _FakeClient:
        models = _FakeModels()

    fake_cloud = ResilientGeminiClient(client_factory=lambda _k: _FakeClient(), sleep_fn=lambda _s: None)
    with tempfile.TemporaryDirectory() as tmp:
        cache_db = Path(tmp) / "ai_cache.sqlite"
        mgr = LocalAIManager(
            routing_mode=AiRoutingMode.CLOUD,
            cache=LocalResponseCache(db_path=cache_db),
            cloud_client=fake_cloud,
        )

        # Verify Hybrid router switches cleanly across CLOUD -> LOCAL -> HYBRID
        streamed_pieces: List[str] = []
        cloud_res = mgr.generate_hybrid(
            "Giải thích chi tiết Định luật II Newton và lập bảng đơn vị.",
            history=[{"role": "user", "content": "Hôm nay học Vật Lý 10"}],
            cloud_api_key="AIzaFakeTestKey0000000000000000000000",
            on_token=streamed_pieces.append,
            use_cache=False,
        )
        assert cloud_res.ok is True and cloud_res.engine_used == "Cloud"
        assert len(captured_calls) >= 1
        cfg_sent = captured_calls[0].get("config")
        max_tok_sent = getattr(cfg_sent, "max_output_tokens", None) if not isinstance(cfg_sent, dict) else cfg_sent.get("max_output_tokens")
        sys_inst_sent = getattr(cfg_sent, "system_instruction", None) if not isinstance(cfg_sent, dict) else cfg_sent.get("system_instruction")
        assert max_tok_sent is not None and int(max_tok_sent) >= 4096, f"max_output_tokens must be >= 4096, got {max_tok_sent}"
        assert sys_inst_sent and "AirTeach AI" in str(sys_inst_sent), "Deep pedagogical system_instruction must be passed!"

        joined_stream = "".join(streamed_pieces)
        assert len(joined_stream) > 350 and "Định Luật II Newton" in joined_stream and "Lưu Ý Sai Lầm" in joined_stream

        # Verify WhiteboardAgent preserves LaTeX delimiters for Rich HTML rendering
        _, preserved_reply = WhiteboardAgent.extract_actions_from_response(cloud_res.text)
        assert "$$" in preserved_reply and "\\frac{\\vec{F}}{m}" in preserved_reply
        rich_html = OutputSanitizer.to_rich_html(preserved_reply, dark_mode=True, render_math_images=True)
        assert "<table" in rich_html and "<th" in rich_html and "<td" in rich_html
        assert "Định Luật II Newton" in rich_html and "Lưu Ý Sai Lầm" in rich_html

        # Verify AirTeachMainWindow streaming handler renders full multi-paragraph output without truncation
        orig_start = main.CameraHandWorker.start
        main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
        win = main.AirTeachMainWindow()
        try:
            win._stream_raw_accum = ""
            win._stream_last_render_t = 0.0
            for piece in streamed_pieces:
                win._on_hybrid_token(piece)
            win._on_hybrid_finished(cloud_res)
            rendered_plain = win.ai_output.toPlainText()
            assert "Định Luật II Newton" in rendered_plain
            assert "Bảng Tổng Hợp Đại Lượng" in rendered_plain
            assert "Lưu Ý Sai Lầm" in rendered_plain
        finally:
            main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
            win.shutdown_resources()
            win.close()

        # Verify LOCAL mode fails loudly with clear Vietnamese guidance when no local engine is running
        mgr.routing_mode = AiRoutingMode.LOCAL
        mgr.active_base_url = ""
        mgr.active_model = ""
        local_err = mgr.generate_hybrid("Xin chào", use_cache=False)
        assert local_err.ok is False and "Local AI" in local_err.error_message_vi

        # Verify HYBRID mode automatically falls back smoothly to Cloud when Local is absent
        mgr.routing_mode = AiRoutingMode.HYBRID
        hyb_res = mgr.generate_hybrid(
            "Tóm tắt lực hấp dẫn",
            cloud_api_key="AIzaFakeTestKey0000000000000000000000",
            use_cache=False,
        )
        assert hyb_res.ok is True and hyb_res.engine_used == "Cloud"

        # Resumable HTTP 206 Range + SHA-256 verification against live ThreadingHTTPServer
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
            models_dir = Path(tmp) / "models"

            interrupted = False
            try:
                downloader.download(url=url, dest_dir=models_dir, filename="test-qwen.gguf", verify_gguf=True)
            except IOError:
                interrupted = True
            assert interrupted is True
            part_file = models_dir / "test-qwen.gguf.part"
            assert part_file.exists() and part_file.stat().st_size == handler_cls.drop_after_bytes

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


# =============================================================================
# 7. HEADLESS OFFSCREEN GUI SMOKE TEST (STARTUP < 1.5s, > 80% CANVAS, PANELS, GUIDE)
# =============================================================================
def test_headless_gui_smoke_startup_panels_and_guide() -> None:
    """Measure cold startup (<1.5s), Canvas >80%, toggle ALL panels, open [? Hướng Dẫn] & Local AI dialog, exit 0."""
    probe_code = r'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["AIRTEACH_NO_MODAL"] = "1"
t0 = time.perf_counter()
from PySide6.QtWidgets import QApplication
import main
from cv_engine.user_guide_dialog import UserGuideDialog
from cv_engine.gguf_model_manager import LocalAiManagerDialog

app = QApplication(["startup-smoke"])
win = main.AirTeachMainWindow()
win.resize(1400, 820)
win.stack.setCurrentIndex(1)
win.show()
app.processEvents()
elapsed = time.perf_counter() - t0

win_area = max(1, win.width() * win.height())
canvas_area = win.canvas.width() * win.canvas.height()
ratio = canvas_area / win_area

# Toggle ALL collapsible panels ON and OFF cleanly
assert win.left_dock.isVisible() is False
win._on_toggle_studio_camera()
assert win.left_dock.isVisible() is True
win._on_toggle_studio_camera()
assert win.left_dock.isVisible() is False

assert win.right_dock.isVisible() is False
win._on_toggle_ai_drawer()
assert win.right_dock.isVisible() is True
win._on_toggle_ai_drawer()
assert win.right_dock.isVisible() is False

assert win.thumb_strip.isVisible() is False
win._on_toggle_thumb_strip()
assert win.thumb_strip.isVisible() is True
win._on_toggle_thumb_strip()
assert win.thumb_strip.isVisible() is False

assert win.canvas.mini_hud.isVisible() is True
win._on_toggle_mini_hud()
assert win.canvas.mini_hud.isVisible() is False
win._on_toggle_mini_hud()
assert win.canvas.mini_hud.isVisible() is True

# Open [? Hướng Dẫn] 5-tab visual guide dialog (including Live Gesture Sandbox)
guide = UserGuideDialog(win)
guide.show()
app.processEvents()
assert guide.tabs.count() == 5
tab_titles = [guide.tabs.tabText(i) for i in range(5)]
assert any("1 Tay" in t for t in tab_titles)
assert any("2 Tay" in t for t in tab_titles)
assert any("Bảng Tròn" in t for t in tab_titles)
assert any("AI Local" in t for t in tab_titles)
assert any("Sandbox" in t or "Thử Cử Chỉ" in t for t in tab_titles)
guide.close()

# Open [Quản Lý AI Local] dialog
local_dlg = LocalAiManagerDialog(parent=win)
local_dlg.show()
app.processEvents()
assert "ONNX Runtime" in local_dlg._runtime_status_lbl.text()
local_dlg.close()

print(f"BENCH_STARTUP={elapsed:.4f} BENCH_CANVAS_RATIO={ratio:.4f} GUI_SMOKE=OK", flush=True)
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
        timeout=35,
        check=True,
    )
    assert proc.returncode == 0
    line = [ln for ln in proc.stdout.splitlines() if "BENCH_STARTUP=" in ln][-1]
    parts = dict(tok.split("=", 1) for tok in line.strip().split())
    startup_sec = float(parts["BENCH_STARTUP"])
    canvas_ratio = float(parts["BENCH_CANVAS_RATIO"])
    assert parts.get("GUI_SMOKE") == "OK"
    print(
        f"[BENCH] Cold Window Startup = {startup_sec:.3f}s | Canvas Area Share = {canvas_ratio * 100:.1f}% | Panels & 4-Tab Guide OK",
        flush=True,
    )
    assert startup_sec < 1.5, f"Cold startup exceeded 1.5s budget: {startup_sec:.3f}s"
    assert canvas_ratio > 0.80, f"Canvas area must exceed 80% of window (got {canvas_ratio * 100:.1f}%)"


def main() -> int:
    tests = [
        ("1. Rust Native Core & 1000-Frame Benchmark", test_rust_core_import_and_1khz_benchmark),
        ("2. Synthetic 1-Hand, Swipe & Dual-Hand Gestures", test_synthetic_1hand_swipe_and_dual_hand_gestures),
        ("3. Right-Click Radial Pie Menu (8 Slices + Deadzone + Signals)", test_radial_pie_menu_angles_deadzone_and_canvas_mouse),
        ("4. Universal Real Document & Image Engine (>=1000px + Export)", test_universal_real_document_and_image_rendering),
        ("5. Board Navigation, Zoom [25%-400%] & LOD Cache", test_navigation_zoom_range_and_lod_cache),
        ("6. Local AI Manager, Hybrid Router & Deep Streaming Chatbot", test_local_ai_manager_and_chatbot_rich_rendering),
        ("7. PySide6 Headless GUI Smoke, Panels & 4-Tab Guide", test_headless_gui_smoke_startup_panels_and_guide),
    ]
    for label, fn in tests:
        print(f"=== Running: {label} ===", flush=True)
        fn()
        print(f"[PASS] {label}", flush=True)
    print("ALL_SYSTEMS_VERIFIED_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
