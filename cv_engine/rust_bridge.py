#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin Native Rust Bridge (`cv_engine/rust_bridge.py`)
Imports `airteach_core_rs` (PyO3 `--release` native extension) first and falls
back gracefully to pure-Python/NumPy only if the native module is absent.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cv_engine.app_paths import get_app_logger

_LOGGER = get_app_logger("rust_bridge")

try:
    import airteach_core_rs as _rs  # type: ignore

    RUST_CORE_VERSION: str = str(_rs.version())
    HAS_RUST_CORE: bool = True
except Exception as exc:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    HAS_RUST_CORE = False
    RUST_CORE_VERSION = f"python-fallback ({exc})"
    _LOGGER.warning("airteach_core_rs unavailable, falling back to Python/NumPy: %s", exc)


Point2D = Tuple[float, float]


def make_one_euro_1d(min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0):
    if HAS_RUST_CORE and hasattr(_rs, "RustOneEuroFilter1D"):
        return _rs.RustOneEuroFilter1D(float(min_cutoff), float(beta), float(d_cutoff))
    return None


def make_one_euro_2d(min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0):
    if HAS_RUST_CORE and hasattr(_rs, "RustOneEuroFilter2D"):
        return _rs.RustOneEuroFilter2D(float(min_cutoff), float(beta), float(d_cutoff))
    return None


def make_one_euro_3d(min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0):
    if HAS_RUST_CORE and hasattr(_rs, "RustOneEuroFilter3D"):
        return _rs.RustOneEuroFilter3D(float(min_cutoff), float(beta), float(d_cutoff))
    return None


def make_kinematic_60hz_extrapolator(
    min_cutoff: float = 0.8,
    beta: float = 0.04,
    d_cutoff: float = 1.0,
    max_stale_sec: float = 0.16,
):
    if HAS_RUST_CORE and hasattr(_rs, "RustKinematic60HzExtrapolator"):
        return _rs.RustKinematic60HzExtrapolator(
            float(min_cutoff), float(beta), float(d_cutoff), float(max_stale_sec)
        )
    return None


def make_swipe_detector(
    swipe_cooldown_sec: float = 0.60,
    swipe_min_dx: float = 0.16,
    swipe_min_vel: float = 0.85,
    swipe_window_sec: float = 0.28,
    vertical_cooldown_sec: float = 0.45,
    vertical_min_dy: float = 0.14,
    vertical_min_vel: float = 0.75,
):
    if HAS_RUST_CORE and hasattr(_rs, "RustSwipeDetector"):
        return _rs.RustSwipeDetector(
            float(swipe_cooldown_sec),
            float(swipe_min_dx),
            float(swipe_min_vel),
            float(swipe_window_sec),
            float(vertical_cooldown_sec),
            float(vertical_min_dy),
            float(vertical_min_vel),
        )
    return None


def classify_hand_3d_fast(
    lm: Sequence[Sequence[float]],
    prev_fingers: Optional[Sequence[bool]] = None,
    was_pinched: bool = False,
    pinch_enter_ratio: float = 0.28,
    pinch_exit_ratio: float = 0.42,
    handedness: str = "right",
    palm_facing_min_z: float = 0.20,
) -> Optional[Dict[str, object]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "classify_hand_3d_rs")):
        return None
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in lm] if lm is not None else []
    pf = [bool(x) for x in prev_fingers] if prev_fingers is not None else None
    return dict(
        _rs.classify_hand_3d_rs(
            pts,
            pf,
            bool(was_pinched),
            float(pinch_enter_ratio),
            float(pinch_exit_ratio),
            str(handedness),
            float(palm_facing_min_z),
        )
    )


def compute_dual_hand_kinematics_fast(
    p_xy: Point2D,
    s_xy: Point2D,
    prev_pinch_dist: Optional[float] = None,
    prev_pan_center: Optional[Point2D] = None,
) -> Tuple[float, Point2D, float, Point2D, Point2D]:
    if HAS_RUST_CORE and hasattr(_rs, "compute_dual_hand_kinematics_rs"):
        return _rs.compute_dual_hand_kinematics_rs(
            (float(p_xy[0]), float(p_xy[1])),
            (float(s_xy[0]), float(s_xy[1])),
            float(prev_pinch_dist) if prev_pinch_dist is not None else None,
            (float(prev_pan_center[0]), float(prev_pan_center[1]))
            if prev_pan_center is not None
            else None,
        )
    dist = max(0.03, math.hypot(p_xy[0] - s_xy[0], p_xy[1] - s_xy[1]))
    mid = (0.5 * (p_xy[0] + s_xy[0]), 0.5 * (p_xy[1] + s_xy[1]))
    zf = 1.0
    if prev_pinch_dist is not None and abs(dist - prev_pinch_dist) > 0.005:
        zf = max(0.85, min(1.15, dist / max(0.03, prev_pinch_dist)))
    dx, dy = 0.0, 0.0
    if prev_pan_center is not None:
        rdx = mid[0] - prev_pan_center[0]
        rdy = mid[1] - prev_pan_center[1]
        if math.hypot(rdx, rdy) > 0.003:
            dx = max(-0.15, min(0.15, rdx))
            dy = max(-0.15, min(0.15, rdy))
    return (round(zf, 4), mid, dist, (round(dx, 5), round(dy, 5)), mid)


def catmull_rom_spline_fast(
    pts: Sequence[Point2D],
    subdivisions: int = 4,
    alpha: float = 0.5,
) -> Optional[List[Point2D]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "catmull_rom_spline_rs")):
        return None
    in_pts = [(float(p[0]), float(p[1])) for p in pts]
    return list(_rs.catmull_rom_spline_rs(in_pts, int(subdivisions), float(alpha)))


def rdp_simplify_fast(pts: Sequence[Point2D], epsilon: float) -> Optional[List[Point2D]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "rdp_simplify_rs")):
        return None
    in_pts = [(float(p[0]), float(p[1])) for p in pts]
    return list(_rs.rdp_simplify_rs(in_pts, float(epsilon)))


def resample_arc_length_fast(pts: Sequence[Point2D], n: int = 64) -> Optional[List[Point2D]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "resample_arc_length_rs")):
        return None
    in_pts = [(float(p[0]), float(p[1])) for p in pts]
    return list(_rs.resample_arc_length_rs(in_pts, int(n)))


def snap_geometric_shape_fast(
    pts: Sequence[Point2D],
    closed_gap_ratio: float = 0.18,
    confidence_gate: float = 0.78,
) -> Optional[Tuple[str, float, bool, List[Point2D]]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "snap_geometric_shape_rs")):
        return None
    in_pts = [(float(p[0]), float(p[1])) for p in pts]
    res = _rs.snap_geometric_shape_rs(
        in_pts, float(closed_gap_ratio), float(confidence_gate)
    )
    if res is None:
        return None
    shape_type, conf, is_closed, out_pts = res
    return str(shape_type), float(conf), bool(is_closed), list(out_pts)


def analyze_frame_luma_fast(
    frame: np.ndarray,
    *,
    is_rgb: bool = True,
    stride: int = 4,
) -> Optional[Tuple[float, float, float]]:
    if not (HAS_RUST_CORE and hasattr(_rs, "analyze_frame_luma_rs")) or frame is None or frame.size == 0:
        return None
    contig = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = contig.shape[:2]
    ch = 1 if contig.ndim == 2 else int(contig.shape[2])
    return _rs.analyze_frame_luma_rs(contig.tobytes(), int(w), int(h), int(ch), bool(is_rgb), int(stride))


def apply_gamma_contrast_fast(
    frame: np.ndarray,
    *,
    gamma: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Optional[np.ndarray]:
    if not (HAS_RUST_CORE and hasattr(_rs, "apply_gamma_contrast_rs")) or frame is None or frame.size == 0:
        return None
    contig = np.ascontiguousarray(frame, dtype=np.uint8)
    raw_out = _rs.apply_gamma_contrast_rs(contig.tobytes(), float(gamma), float(alpha), float(beta))
    return np.frombuffer(raw_out, dtype=np.uint8).reshape(contig.shape).copy()


def compute_roi_crop_box_fast(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    pred_dx: float = 0.0,
    pred_dy: float = 0.0,
    expand_factor: float = 2.2,
    padding_ratio: float = 0.45,
) -> List[float]:
    if HAS_RUST_CORE and hasattr(_rs, "compute_roi_crop_box_rs"):
        rx0, ry0, rw, rh = _rs.compute_roi_crop_box_rs(
            float(min_x),
            float(min_y),
            float(max_x),
            float(max_y),
            float(pred_dx),
            float(pred_dy),
            float(expand_factor),
            float(padding_ratio),
        )
        return [float(rx0), float(ry0), float(rw), float(rh)]
    bw = max(0.06, max_x - min_x)
    bh = max(0.06, max_y - min_y)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    expand = max(expand_factor, 1.0 + 2.0 * max(0.45, padding_ratio))
    side = max(bw, bh) * expand
    pred_cx = max(0.0, min(1.0, cx + pred_dx * 0.5))
    pred_cy = max(0.0, min(1.0, cy + pred_dy * 0.5))
    rx0 = max(0.0, pred_cx - 0.5 * side)
    ry0 = max(0.0, pred_cy - 0.5 * side)
    return [rx0, ry0, min(1.0 - rx0, side), min(1.0 - ry0, side)]
