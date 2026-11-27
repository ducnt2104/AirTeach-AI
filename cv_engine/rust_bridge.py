#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin Native Rust Bridge (`cv_engine/rust_bridge.py`)
Imports `airteach_core_rs` (PyO3 `--release` native extension) first and falls
back gracefully to pure-Python/NumPy only if the native module is absent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def make_kalman_predictor_2d(process_noise: float = 0.04, meas_noise: float = 0.0015):
    if HAS_RUST_CORE and hasattr(_rs, "RustKalmanVelocityPredictor2D"):
        return _rs.RustKalmanVelocityPredictor2D(float(process_noise), float(meas_noise))
    return None


def make_gesture_state_machine(
    *,
    min_hold_frames: int = 3,
    min_hold_sec: float = 0.080,
    grace_period_sec: float = 0.160,
    min_cutoff: float = 1.45,
    beta: float = 0.020,
    d_cutoff: float = 1.0,
    frame_w: int = 640,
    frame_h: int = 480,
    deadzone_px: float = 2.0,
    voting_window_size: int = 7,
    voting_min_votes: int = 3,
    pinch_enter_ratio: float = 0.28,
    pinch_exit_ratio: float = 0.42,
    palm_facing_min_z: float = 0.20,
):
    if HAS_RUST_CORE and hasattr(_rs, "RustGestureStateMachine"):
        return _rs.RustGestureStateMachine(
            int(min_hold_frames),
            float(min_hold_sec),
            float(grace_period_sec),
            float(min_cutoff),
            float(beta),
            float(d_cutoff),
            int(frame_w),
            int(frame_h),
            float(deadzone_px),
            int(voting_window_size),
            int(voting_min_votes),
            float(pinch_enter_ratio),
            float(pinch_exit_ratio),
            float(palm_facing_min_z),
        )
    return None


def make_dual_hand_state_machine():
    if HAS_RUST_CORE and hasattr(_rs, "RustDualHandStateMachine"):
        return _rs.RustDualHandStateMachine()
    return None


def _py_perp_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    lsq = dx * dx + dy * dy
    if lsq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / lsq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def hit_test_polyline_fast(
    px: float,
    py: float,
    pts: Sequence[Point2D],
    radius: float = 10.0,
) -> bool:
    if not pts:
        return False
    if HAS_RUST_CORE and hasattr(_rs, "hit_test_polyline_rs"):
        in_pts = [(float(p[0]), float(p[1])) for p in pts]
        return bool(_rs.hit_test_polyline_rs(float(px), float(py), in_pts, float(radius)))
    r = max(1.0, float(radius))
    if len(pts) == 1:
        return math.hypot(px - pts[0][0], py - pts[0][1]) <= r
    for i in range(len(pts) - 1):
        if _py_perp_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) <= r:
            return True
    return False


def hit_test_objects_fast(
    px: float,
    py: float,
    objects: Sequence[Tuple[str, str, float, float, float, float, float, int, Sequence[Point2D]]],
    tolerance: float = 10.0,
) -> Optional[str]:
    """Hit-test Canvas objects in Rust (descending z_index) with automatic Python fallback."""
    if not objects:
        return None
    if HAS_RUST_CORE and hasattr(_rs, "hit_test_objects_rs"):
        payload = [
            (
                str(o[0]),
                str(o[1]),
                float(o[2]),
                float(o[3]),
                float(o[4]),
                float(o[5]),
                float(o[6]),
                int(o[7]),
                [(float(p[0]), float(p[1])) for p in o[8]],
            )
            for o in objects
        ]
        res = _rs.hit_test_objects_rs(float(px), float(py), payload, float(tolerance))
        return str(res) if res is not None else None

    tol = max(2.0, float(tolerance))
    ordered = sorted(enumerate(objects), key=lambda item: (int(item[1][7]), item[0]), reverse=True)
    for _, o in ordered:
        oid, otype, x, y, w, h, rot_deg, _, pts = o
        bw, bh = max(4.0, float(w)), max(4.0, float(h))
        cx, cy = x + 0.5 * bw, y + 0.5 * bh
        if abs(rot_deg) > 1e-4:
            rad = math.radians(-rot_deg)
            c, s = math.cos(rad), math.sin(rad)
            dx, dy = px - cx, py - cy
            lx, ly = cx + dx * c - dy * s, cy + dx * s + dy * c
        else:
            lx, ly = px, py
        if lx < x - tol or lx > x + bw + tol or ly < y - tol or ly > y + bh + tol:
            continue
        if otype in ("stroke", "shape") and pts:
            if hit_test_polyline_fast(lx, ly, pts, tol):
                return str(oid)
            if otype == "shape" and len(pts) >= 4:
                if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 12.0:
                    inside = False
                    j = len(pts) - 1
                    for i in range(len(pts)):
                        xi, yi = pts[i]
                        xj, yj = pts[j]
                        if ((yi > ly) != (yj > ly)) and (lx < (xj - xi) * (ly - yi) / ((yj - yi) + 1e-12) + xi):
                            inside = not inside
                        j = i
                    if inside:
                        return str(oid)
        else:
            return str(oid)
    return None


def marquee_select_objects_fast(
    rx: float,
    ry: float,
    rw: float,
    rh: float,
    objects: Sequence[Tuple[str, float, float, float, float, float]],
) -> List[str]:
    """Select all objects intersecting marquee rectangle (Rust accelerated with Python fallback)."""
    if not objects:
        return []
    if HAS_RUST_CORE and hasattr(_rs, "marquee_select_objects_rs"):
        payload = [
            (str(o[0]), float(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]))
            for o in objects
        ]
        return list(_rs.marquee_select_objects_rs(float(rx), float(ry), float(rw), float(rh), payload))

    mx0, mx1 = min(rx, rx + rw), max(rx, rx + rw)
    my0, my1 = min(ry, ry + rh), max(ry, ry + rh)
    hits: List[str] = []
    for oid, x, y, w, h, rot_deg in objects:
        bw, bh = max(2.0, float(w)), max(2.0, float(h))
        if abs(rot_deg) < 1e-4:
            ox0, oy0, ox1, oy1 = x, y, x + bw, y + bh
        else:
            cx, cy = x + 0.5 * bw, y + 0.5 * bh
            rad = math.radians(rot_deg)
            c, s = math.cos(rad), math.sin(rad)
            corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
            wxs = [cx + (px - cx) * c - (py - cy) * s for px, py in corners]
            wys = [cy + (px - cx) * s + (py - cy) * c for px, py in corners]
            ox0, oy0, ox1, oy1 = min(wxs), min(wys), max(wxs), max(wys)
        if ox1 >= mx0 and ox0 <= mx1 and oy1 >= my0 and oy0 <= my1:
            hits.append(str(oid))
    return hits


# =============================================================================
# 2D QUADTREE SPATIAL INDEX (RustSpatialIndex + Pure-Python Quadtree Fallback)
# =============================================================================


class _PyQuadNode:
    __slots__ = ("min_x", "min_y", "max_x", "max_y", "depth", "items", "children")

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        depth: int,
    ) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.depth = depth
        self.items: List[str] = []
        self.children: Optional[Tuple["_PyQuadNode", "_PyQuadNode", "_PyQuadNode", "_PyQuadNode"]] = None

    def child_quadrant(self, x0: float, y0: float, x1: float, y1: float) -> Optional[int]:
        if x0 < self.min_x or x1 > self.max_x or y0 < self.min_y or y1 > self.max_y:
            return None
        mid_x = 0.5 * (self.min_x + self.max_x)
        mid_y = 0.5 * (self.min_y + self.max_y)
        if x1 <= mid_x:
            col = 0
        elif x0 >= mid_x:
            col = 1
        else:
            return None
        if y1 <= mid_y:
            row = 0
        elif y0 >= mid_y:
            row = 2
        else:
            return None
        return row + col


class PythonSpatialIndex:
    """Pure-Python 2D Quadtree fallback matching `RustSpatialIndex` API."""

    def __init__(
        self,
        min_x: float = -50000.0,
        min_y: float = -50000.0,
        max_x: float = 50000.0,
        max_y: float = 50000.0,
        max_items: int = 12,
        max_depth: int = 8,
    ) -> None:
        x0, x1 = (float(min_x), float(max_x)) if min_x <= max_x else (float(max_x), float(min_x))
        y0, y1 = (float(min_y), float(max_y)) if min_y <= max_y else (float(max_y), float(min_y))
        if abs(x1 - x0) < 1.0:
            x1 = x0 + 1000.0
        if abs(y1 - y0) < 1.0:
            y1 = y0 + 1000.0
        self._default_bounds = (x0, y0, x1, y1)
        self._max_items = max(2, int(max_items))
        self._max_depth = max(1, min(16, int(max_depth)))
        self._root = _PyQuadNode(x0, y0, x1, y1, 0)
        # id -> (min_x, min_y, max_x, max_y, z_index, order, node_ref)
        self._entries: Dict[str, Tuple[float, float, float, float, int, int, _PyQuadNode]] = {}
        self._next_order = 0

    def clear(self) -> None:
        x0, y0, x1, y1 = self._default_bounds
        self._root = _PyQuadNode(x0, y0, x1, y1, 0)
        self._entries.clear()
        self._next_order = 0

    def _split_node(self, node: _PyQuadNode) -> None:
        if node.children is not None or node.depth >= self._max_depth:
            return
        mid_x = 0.5 * (node.min_x + node.max_x)
        mid_y = 0.5 * (node.min_y + node.max_y)
        cd = node.depth + 1
        children = (
            _PyQuadNode(node.min_x, node.min_y, mid_x, mid_y, cd),
            _PyQuadNode(mid_x, node.min_y, node.max_x, mid_y, cd),
            _PyQuadNode(node.min_x, mid_y, mid_x, node.max_y, cd),
            _PyQuadNode(mid_x, mid_y, node.max_x, node.max_y, cd),
        )
        node.children = children
        old_items = node.items
        keep_here: List[str] = []
        for oid in old_items:
            rec = self._entries.get(oid)
            if rec is None:
                continue
            x0, y0, x1, y1, z, order, _ = rec
            q = node.child_quadrant(x0, y0, x1, y1)
            if q is not None:
                cnode = children[q]
                cnode.items.append(oid)
                self._entries[oid] = (x0, y0, x1, y1, z, order, cnode)
            else:
                keep_here.append(oid)
                self._entries[oid] = (x0, y0, x1, y1, z, order, node)
        node.items = keep_here

    def rebuild(self, items: Sequence[Tuple[str, float, float, float, float, int]]) -> None:
        rx0, ry0, rx1, ry1 = self._default_bounds
        for _, min_x, min_y, max_x, max_y, _ in items:
            x0, x1 = (float(min_x), float(max_x)) if min_x <= max_x else (float(max_x), float(min_x))
            y0, y1 = (float(min_y), float(max_y)) if min_y <= max_y else (float(max_y), float(min_y))
            if x0 < rx0:
                rx0 = x0 - 64.0
            if y0 < ry0:
                ry0 = y0 - 64.0
            if x1 > rx1:
                rx1 = x1 + 64.0
            if y1 > ry1:
                ry1 = y1 + 64.0
        self._root = _PyQuadNode(rx0, ry0, rx1, ry1, 0)
        self._entries.clear()
        self._next_order = 0
        for oid, min_x, min_y, max_x, max_y, z_index in items:
            self.upsert(str(oid), float(min_x), float(min_y), float(max_x), float(max_y), int(z_index))

    def upsert(
        self,
        id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        z_index: int,
    ) -> None:
        oid = str(id)
        if oid in self._entries:
            self.remove(oid)
        x0, x1 = (float(min_x), float(max_x)) if min_x <= max_x else (float(max_x), float(min_x))
        y0, y1 = (float(min_y), float(max_y)) if min_y <= max_y else (float(max_y), float(min_y))
        z = int(z_index)
        order = self._next_order
        self._next_order += 1

        cur = self._root
        while cur.children is not None:
            q = cur.child_quadrant(x0, y0, x1, y1)
            if q is not None:
                cur = cur.children[q]
            else:
                break
        cur.items.append(oid)
        self._entries[oid] = (x0, y0, x1, y1, z, order, cur)
        if cur.children is None and len(cur.items) > self._max_items and cur.depth < self._max_depth:
            self._split_node(cur)

    def remove(self, id: str) -> bool:
        oid = str(id)
        rec = self._entries.pop(oid, None)
        if rec is None:
            return False
        node = rec[6]
        try:
            node.items.remove(oid)
        except ValueError:
            pass
        return True

    def _query_aabb(self, qx0: float, qy0: float, qx1: float, qy1: float) -> List[Tuple[int, int, str]]:
        if not self._entries:
            return []
        hits: List[Tuple[int, int, str]] = []
        stack: List[_PyQuadNode] = [self._root]
        while stack:
            node = stack.pop()
            for oid in node.items:
                rec = self._entries.get(oid)
                if rec is not None:
                    ex0, ey0, ex1, ey1, z, order, _ = rec
                    if ex1 >= qx0 and ex0 <= qx1 and ey1 >= qy0 and ey0 <= qy1:
                        hits.append((z, order, oid))
            if node.children is not None:
                for child in node.children:
                    if child.max_x >= qx0 and child.min_x <= qx1 and child.max_y >= qy0 and child.min_y <= qy1:
                        stack.append(child)
        return hits

    def query_point(self, px: float, py: float, tolerance: float = 10.0) -> List[str]:
        tol = max(0.0, float(tolerance))
        hits = self._query_aabb(float(px) - tol, float(py) - tol, float(px) + tol, float(py) + tol)
        hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in hits]

    def query_rect(self, rx: float, ry: float, rw: float, rh: float) -> List[str]:
        qx0 = min(float(rx), float(rx) + float(rw))
        qx1 = max(float(rx), float(rx) + float(rw))
        qy0 = min(float(ry), float(ry) + float(rh))
        qy1 = max(float(ry), float(ry) + float(rh))
        hits = self._query_aabb(qx0, qy0, qx1, qy1)
        hits.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in hits]

    def len(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def make_spatial_index(
    min_x: float = -50000.0,
    min_y: float = -50000.0,
    max_x: float = 50000.0,
    max_y: float = 50000.0,
    max_items: int = 12,
    max_depth: int = 8,
) -> Any:
    """Create native Rust 2D Quadtree (`RustSpatialIndex`) with pure-Python Quadtree fallback."""
    if HAS_RUST_CORE and hasattr(_rs, "RustSpatialIndex"):
        return _rs.RustSpatialIndex(
            float(min_x),
            float(min_y),
            float(max_x),
            float(max_y),
            int(max_items),
            int(max_depth),
        )
    return PythonSpatialIndex(
        float(min_x),
        float(min_y),
        float(max_x),
        float(max_y),
        int(max_items),
        int(max_depth),
    )


def _py_resize_sharpen_rgb(
    arr: np.ndarray,
    dst_w: int,
    dst_h: int,
    sharpen_amount: float,
) -> np.ndarray:
    """Pure NumPy/Pillow fallback for RGB24 resize + 3x3 Gaussian unsharp mask."""
    sh, sw = arr.shape[:2]
    if sw == dst_w and sh == dst_h:
        resized = arr.astype(np.float32, copy=True)
    else:
        try:
            from PIL import Image

            resample = Image.Resampling.BOX if (sw > dst_w * 2 or sh > dst_h * 2) else Image.Resampling.BILINEAR
            pil_out = Image.fromarray(arr, mode="RGB").resize((dst_w, dst_h), resample=resample)
            resized = np.asarray(pil_out, dtype=np.float32)
        except Exception:
            xs = np.clip((np.arange(dst_w, dtype=np.float32) + 0.5) * (sw / float(dst_w)) - 0.5, 0.0, max(0, sw - 1))
            ys = np.clip((np.arange(dst_h, dtype=np.float32) + 0.5) * (sh / float(dst_h)) - 0.5, 0.0, max(0, sh - 1))
            x0 = np.floor(xs).astype(np.int32)
            x1 = np.minimum(x0 + 1, sw - 1)
            wx = (xs - x0)[None, :, None]
            y0 = np.floor(ys).astype(np.int32)
            y1 = np.minimum(y0 + 1, sh - 1)
            wy = (ys - y0)[:, None, None]
            src_f = arr.astype(np.float32)
            top = src_f[y0][:, x0] * (1.0 - wx) + src_f[y0][:, x1] * wx
            bot = src_f[y1][:, x0] * (1.0 - wx) + src_f[y1][:, x1] * wx
            resized = top * (1.0 - wy) + bot * wy

    if sharpen_amount > 0.01:
        amt = max(0.0, min(3.0, float(sharpen_amount)))
        padded = np.pad(resized, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) * (1.0 / 16.0)
        resized = resized + amt * (resized - blur)
    return np.clip(np.rint(resized), 0.0, 255.0).astype(np.uint8)


def preprocess_image_resize_sharpen_fast(
    rgb_np_or_bytes: Union[np.ndarray, bytes, bytearray, memoryview],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    sharpen_amount: float = 0.35,
) -> Union[np.ndarray, bytes]:
    """
    Resize RGB24 image to `(dst_w, dst_h)` and apply 3x3 unsharp mask in native multi-threaded Rust
    (GIL released via `py.allow_threads` + `std::thread::scope`) with NumPy/Pillow fallback.

    Returns an `(dst_h, dst_w, 3)` `uint8` NumPy array when given a NumPy array,
    or raw RGB24 `bytes` of length `dst_w * dst_h * 3` when given bytes/buffer.
    """
    sw, sh = int(src_w), int(src_h)
    dw, dh = int(dst_w), int(dst_h)
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        raise ValueError("src_w, src_h, dst_w, and dst_h must all be > 0")
    amt = float(sharpen_amount)

    is_numpy = isinstance(rgb_np_or_bytes, np.ndarray)
    if is_numpy:
        arr = np.asarray(rgb_np_or_bytes, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB uint8 ndarray, got shape {arr.shape}")
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        raw_bytes = arr.tobytes()
    else:
        raw_bytes = bytes(rgb_np_or_bytes)
        if len(raw_bytes) < sw * sh * 3:
            raise ValueError(f"rgb_bytes too short: need {sw * sh * 3}, got {len(raw_bytes)}")

    if HAS_RUST_CORE and hasattr(_rs, "preprocess_image_resize_sharpen_rs"):
        out_bytes = _rs.preprocess_image_resize_sharpen_rs(raw_bytes, sw, sh, dw, dh, amt)
        if is_numpy:
            return np.frombuffer(out_bytes, dtype=np.uint8).reshape((dh, dw, 3)).copy()
        return bytes(out_bytes)

    src_arr = np.frombuffer(raw_bytes[: sw * sh * 3], dtype=np.uint8).reshape((sh, sw, 3))
    out_arr = _py_resize_sharpen_rgb(src_arr, dw, dh, amt)
    if is_numpy:
        return out_arr
    return out_arr.tobytes()
