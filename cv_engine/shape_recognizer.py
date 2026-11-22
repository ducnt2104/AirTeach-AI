"""
AirTeach AI (DNT - Victer) — Hybrid Shape Recognition & Auto-Beautification Engine
==================================================================================
Pipeline 3 tầng chuẩn xác tối đa (Zero Bounding-Box Shortcuts):
1. Tiền xử lý & Chuẩn hóa Nét Vẽ:
   - Ramer-Douglas-Peucker (RDP) khử nhiễu răng cưa.
   - Arc-Length Resampling đều N = 64 điểm theo độ dài cung.
   - Closed-Loop Gap Auto-Closure (< 18% tổng chiều dài hoặc đường chéo bbox).
2. Phân loại Lai Đa Tầng (Geometric Invariants + $1 / $P Template Matching):
   - PCA 2x2 Covariance (lambda_2 / lambda_1 < 0.035) + Snap góc 0°/45°/90° (< 6°) -> Line / Arrow.
   - Hệ số tròn đẳng chu (4*pi*A / P^2) + std(r)/mean(r) < 0.14 + Kasa Fit -> Circle.
   - cv2.fitEllipse + kiểm tra độ chữ nhật (rectangularity) -> Ellipse xoay chính xác.
   - Đa ngưỡng cv2.approxPolyDP + kiểm tra góc nội (80°-100°) -> Tam giác, Hình vuông, Hình chữ nhật, Hình thoi / Bình hành.
   - Bộ nhận dạng mẫu $1 & Point-Cloud ($P) bất biến điểm bắt đầu và chiều vẽ.
3. Ngưỡng tin cậy < 0.78 -> Không nắn nhầm chữ viết / đồ thị; xuất Catmull-Rom Spline mượt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

RESAMPLE_N = 64
CONFIDENCE_GATE = 0.78
CLOSED_LOOP_GAP_RATIO = 0.18
PCA_LINE_EIG_RATIO = 0.035
ANGLE_SNAP_TOL_DEG = 6.0
CIRCLE_RADIAL_CV_MAX = 0.14
GOLDEN_PHI = 0.5 * (-1.0 + math.sqrt(5.0))

SHAPE_LABELS_VI: Dict[str, str] = {
    "FREEHAND": "Nét mượt tự do",
    "LINE": "Đoạn thẳng",
    "ARROW": "Mũi tên",
    "CIRCLE": "Hình tròn",
    "ELLIPSE": "Hình elip",
    "TRIANGLE": "Hình tam giác",
    "SQUARE": "Hình vuông",
    "RECTANGLE": "Hình chữ nhật",
    "RHOMBUS": "Hình thoi",
    "PARALLELOGRAM": "Hình bình hành",
}

Point2D = Tuple[float, float]


@dataclass
class BeautifiedShapeResult:
    shape_type: str
    shape_label_vi: str
    confidence: float
    is_closed: bool
    raw_points: List[Point2D]
    beautified_points: List[Point2D]
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def is_geometric(self) -> bool:
        return self.shape_type != "FREEHAND"


def _dist(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(pts: Sequence[Point2D]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        total += _dist(pts[i], pts[i + 1])
    return total


def bbox_and_diag(pts: Sequence[Point2D]) -> Tuple[float, float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x, max_y, math.hypot(max_x - min_x, max_y - min_y)


def deduplicate_points(pts: Sequence[Point2D], min_step: float = 0.6) -> List[Point2D]:
    if not pts:
        return []
    out: List[Point2D] = [(float(pts[0][0]), float(pts[0][1]))]
    min_sq = min_step * min_step
    for p in pts[1:]:
        dx = float(p[0]) - out[-1][0]
        dy = float(p[1]) - out[-1][1]
        if dx * dx + dy * dy >= min_sq:
            out.append((float(p[0]), float(p[1])))
    return out


def _perpendicular_distance(p: Point2D, a: Point2D, b: Point2D) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq))
    px = a[0] + t * dx
    py = a[1] + t * dy
    return math.hypot(p[0] - px, p[1] - py)


def rdp_simplify(pts: Sequence[Point2D], epsilon: float) -> List[Point2D]:
    """Iterative Ramer-Douglas-Peucker simplification."""
    n = len(pts)
    if n <= 2 or epsilon <= 0.0:
        return list(pts)
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        max_d = -1.0
        idx = lo
        a = pts[lo]
        b = pts[hi]
        for i in range(lo + 1, hi):
            d = _perpendicular_distance(pts[i], a, b)
            if d > max_d:
                max_d = d
                idx = i
        if max_d > epsilon and idx != lo and idx != hi:
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return [pts[i] for i in range(n) if keep[i]]


def resample_arc_length(pts: Sequence[Point2D], n: int = RESAMPLE_N) -> List[Point2D]:
    """Resample polyline into exactly N equidistant points by cumulative arc length."""
    if not pts:
        return []
    if len(pts) == 1 or n <= 1:
        return [pts[0]] * max(1, n)

    work = [tuple(p) for p in pts]
    total_len = polyline_length(work)
    if total_len < 1e-6:
        return [work[0]] * n

    step = total_len / float(n - 1)
    cum = 0.0
    out: List[Point2D] = [work[0]]
    i = 1
    prev = work[0]
    while i < len(work):
        cur = work[i]
        seg = _dist(prev, cur)
        if cum + seg >= step:
            t = (step - cum) / max(seg, 1e-9)
            nx = prev[0] + t * (cur[0] - prev[0])
            ny = prev[1] + t * (cur[1] - prev[1])
            q = (nx, ny)
            out.append(q)
            prev = q
            cum = 0.0
        else:
            cum += seg
            prev = cur
            i += 1
    while len(out) < n:
        out.append(work[-1])
    return out[:n]


def catmull_rom_spline(pts: Sequence[Point2D], subdivisions: int = 8) -> List[Point2D]:
    """Centripetal-safe Catmull-Rom spline for smooth freehand curves / handwriting."""
    clean = deduplicate_points(pts, min_step=1.0)
    if len(clean) < 3:
        return clean
    out: List[Point2D] = [clean[0]]
    n = len(clean)
    sub = max(3, subdivisions)
    for i in range(n - 1):
        p0 = clean[i - 1] if i > 0 else clean[i]
        p1 = clean[i]
        p2 = clean[i + 1]
        p3 = clean[i + 2] if (i + 2) < n else clean[n - 1]
        for s in range(1, sub + 1):
            t = s / float(sub)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    return out


def _pca_2d(pts: Sequence[Point2D]) -> Tuple[Point2D, float, float, Point2D]:
    """Return (centroid, lambda1, lambda2, principal_unit_vec) via 2x2 covariance."""
    arr = np.asarray(pts, dtype=np.float64)
    mean = arr.mean(axis=0)
    centered = arr - mean
    n = max(1, len(arr) - 1)
    cov = (centered.T @ centered) / float(n)
    cxx = float(cov[0, 0])
    cyy = float(cov[1, 1])
    cxy = float(cov[0, 1])
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, 0.25 * trace * trace - det))
    l1 = max(0.0, 0.5 * trace + disc)
    l2 = max(0.0, 0.5 * trace - disc)
    if abs(cxy) > 1e-12:
        vx, vy = l1 - cyy, cxy
    elif cxx >= cyy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    vnorm = math.hypot(vx, vy)
    if vnorm < 1e-12:
        vx, vy = 1.0, 0.0
    else:
        vx /= vnorm
        vy /= vnorm
    return (float(mean[0]), float(mean[1])), l1, l2, (vx, vy)


def _snap_angle_deg(angle_deg: float, tol_deg: float = ANGLE_SNAP_TOL_DEG) -> float:
    """Snap angle to nearest multiple of 45° when within tol_deg."""
    step = 45.0
    nearest = round(angle_deg / step) * step
    if abs(angle_deg - nearest) <= tol_deg:
        return nearest
    return angle_deg


# =============================================================================
# $1 & $P HYBRID TEMPLATE RECOGNIZER
# =============================================================================

def _template_circle(n: int = RESAMPLE_N) -> List[Point2D]:
    return [
        (math.cos(2.0 * math.pi * i / (n - 1)), math.sin(2.0 * math.pi * i / (n - 1)))
        for i in range(n)
    ]


def _template_ellipse(aspect: float = 1.85, n: int = RESAMPLE_N) -> List[Point2D]:
    pts = [
        (aspect * math.cos(2.0 * math.pi * i / (n - 1)), math.sin(2.0 * math.pi * i / (n - 1)))
        for i in range(n)
    ]
    return resample_arc_length(pts, n)


def _template_polygon(vertices: Sequence[Point2D], n: int = RESAMPLE_N) -> List[Point2D]:
    closed = list(vertices) + [vertices[0]]
    return resample_arc_length(closed, n)


def _normalize_dollar_one(pts: Sequence[Point2D], n: int = RESAMPLE_N, rotate_indicative: bool = True) -> np.ndarray:
    res = resample_arc_length(pts, n)
    arr = np.asarray(res, dtype=np.float64)
    c = arr.mean(axis=0)
    arr = arr - c
    if rotate_indicative:
        theta = math.atan2(arr[0, 1], arr[0, 0])
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        arr = arr @ rot.T
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    w = max(float(max_xy[0] - min_xy[0]), 1e-6)
    h = max(float(max_xy[1] - min_xy[1]), 1e-6)
    scale = 250.0 / max(w, h)
    arr = arr * scale
    arr = arr - arr.mean(axis=0)
    return arr


def _rotate_pts_np(arr: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rx = arr[:, 0] * c - arr[:, 1] * s
    ry = arr[:, 0] * s + arr[:, 1] * c
    return np.column_stack((rx, ry))


def _dollar_one_distance_at_angle(pts: np.ndarray, tmpl: np.ndarray, angle_rad: float) -> float:
    rot = _rotate_pts_np(pts, angle_rad)
    return float(np.mean(np.linalg.norm(rot - tmpl, axis=1)))


def _dollar_one_best_score(pts: np.ndarray, tmpl: np.ndarray) -> float:
    a = math.radians(-45.0)
    b = math.radians(45.0)
    thresh = math.radians(2.0)
    x1 = GOLDEN_PHI * a + (1.0 - GOLDEN_PHI) * b
    f1 = _dollar_one_distance_at_angle(pts, tmpl, x1)
    x2 = (1.0 - GOLDEN_PHI) * a + GOLDEN_PHI * b
    f2 = _dollar_one_distance_at_angle(pts, tmpl, x2)
    while abs(b - a) > thresh:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = GOLDEN_PHI * a + (1.0 - GOLDEN_PHI) * b
            f1 = _dollar_one_distance_at_angle(pts, tmpl, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - GOLDEN_PHI) * a + GOLDEN_PHI * b
            f2 = _dollar_one_distance_at_angle(pts, tmpl, x2)
    best_d = min(f1, f2)
    half_diag = 0.5 * math.sqrt(250.0 * 250.0 + 250.0 * 250.0)
    return max(0.0, min(1.0, 1.0 - best_d / half_diag))


def _point_cloud_chamfer_score(pts_norm: np.ndarray, tmpl_norm: np.ndarray) -> float:
    """$P-style rotation-aligned bidirectional point-cloud similarity (start-point & direction invariant)."""
    # Align orientation by testing principal axes (0, 90, 180, 270 + indicative rotations)
    best_d = 1e9
    for deg in range(0, 360, 15):
        rot = _rotate_pts_np(pts_norm, math.radians(float(deg)))
        # Pairwise L2 matrix [N x M]
        diff = rot[:, None, :] - tmpl_norm[None, :, :]
        dmat = np.linalg.norm(diff, axis=2)
        d_fwd = float(np.mean(np.min(dmat, axis=1)))
        d_bwd = float(np.mean(np.min(dmat, axis=0)))
        chamfer = 0.5 * (d_fwd + d_bwd)
        if chamfer < best_d:
            best_d = chamfer
    # Empirically calibrated for normalized 250x250 bounding square
    return max(0.0, min(1.0, 1.0 - best_d / 65.0))


class HybridTemplateBank:
    """Precomputed $1 and $P canonical templates for closed geometric shapes."""

    def __init__(self) -> None:
        raw_templates: Dict[str, List[List[Point2D]]] = {
            "CIRCLE": [_template_circle()],
            "ELLIPSE": [_template_ellipse(1.65), _template_ellipse(2.2)],
            "TRIANGLE": [
                _template_polygon([(0.0, -1.0), (0.866, 0.5), (-0.866, 0.5)]),
                _template_polygon([(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]),
            ],
            "SQUARE": [_template_polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])],
            "RECTANGLE": [
                _template_polygon([(-1.6, -0.8), (1.6, -0.8), (1.6, 0.8), (-1.6, 0.8)]),
                _template_polygon([(-0.7, -1.5), (0.7, -1.5), (0.7, 1.5), (-0.7, 1.5)]),
            ],
            "RHOMBUS": [_template_polygon([(0.0, -1.3), (0.9, 0.0), (0.0, 1.3), (-0.9, 0.0)])],
            "PARALLELOGRAM": [_template_polygon([(-0.9, -0.7), (1.3, -0.7), (0.9, 0.7), (-1.3, 0.7)])],
        }
        self._d1_bank: Dict[str, List[np.ndarray]] = {}
        self._pc_bank: Dict[str, List[np.ndarray]] = {}
        for label, tmpls in raw_templates.items():
            d1_list: List[np.ndarray] = []
            pc_list: List[np.ndarray] = []
            for t in tmpls:
                d1_list.append(_normalize_dollar_one(t, RESAMPLE_N, rotate_indicative=True))
                d1_list.append(_normalize_dollar_one(list(reversed(t)), RESAMPLE_N, rotate_indicative=True))
                pc_list.append(_normalize_dollar_one(t, 32, rotate_indicative=False))
            self._d1_bank[label] = d1_list
            self._pc_bank[label] = pc_list

    def score_all(self, closed_pts: Sequence[Point2D]) -> Dict[str, float]:
        q_d1 = _normalize_dollar_one(closed_pts, RESAMPLE_N, rotate_indicative=True)
        q_pc = _normalize_dollar_one(closed_pts, 32, rotate_indicative=False)
        scores: Dict[str, float] = {}
        for label in self._d1_bank:
            best_d1 = max(_dollar_one_best_score(q_d1, t) for t in self._d1_bank[label])
            best_pc = max(_point_cloud_chamfer_score(q_pc, t) for t in self._pc_bank[label])
            scores[label] = 0.45 * best_d1 + 0.55 * best_pc
        return scores


_TEMPLATE_BANK = HybridTemplateBank()


# =============================================================================
# GEOMETRIC INVARIANTS & SHAPE RECONSTRUCTION
# =============================================================================

def _fit_circle_kasa(pts: Sequence[Point2D]) -> Optional[Tuple[float, float, float, float]]:
    arr = np.asarray(pts, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + cx * cx + cy * cy
    if r_sq <= 1.0:
        return None
    r = math.sqrt(r_sq)
    dists = np.hypot(x - cx, y - cy)
    cv = float(np.std(dists) / max(float(np.mean(dists)), 1e-6))
    return cx, cy, r, cv


def _sample_circle(cx: float, cy: float, r: float, segments: int = 72) -> List[Point2D]:
    return [
        (cx + r * math.cos(2.0 * math.pi * i / segments), cy + r * math.sin(2.0 * math.pi * i / segments))
        for i in range(segments + 1)
    ]


def _sample_ellipse(cx: float, cy: float, rx: float, ry: float, angle_deg: float, segments: int = 72) -> List[Point2D]:
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    out: List[Point2D] = []
    for i in range(segments + 1):
        t = 2.0 * math.pi * i / segments
        lx = rx * math.cos(t)
        ly = ry * math.sin(t)
        out.append((cx + lx * ca - ly * sa, cy + lx * sa + ly * ca))
    return out


def _point_to_polyline_rms(pts: Sequence[Point2D], poly_closed: Sequence[Point2D]) -> float:
    if len(poly_closed) < 2 or not pts:
        return 1e9
    total_sq = 0.0
    m = len(poly_closed)
    for p in pts:
        best = 1e9
        for i in range(m - 1):
            d = _perpendicular_distance(p, poly_closed[i], poly_closed[i + 1])
            if d < best:
                best = d
        total_sq += best * best
    return math.sqrt(total_sq / float(len(pts)))


def _ellipse_normalized_rms(pts: Sequence[Point2D], cx: float, cy: float, rx: float, ry: float, angle_deg: float) -> float:
    if rx < 2.0 or ry < 2.0:
        return 1e9
    ang = math.radians(angle_deg)
    ca, sa = math.cos(-ang), math.sin(-ang)
    errs: List[float] = []
    for x, y in pts:
        dx = x - cx
        dy = y - cy
        u = dx * ca - dy * sa
        v = dx * sa + dy * ca
        norm_r = math.hypot(u / rx, v / ry)
        errs.append((norm_r - 1.0) ** 2)
    return math.sqrt(sum(errs) / max(1, len(errs)))


def _interior_angles_deg(verts: Sequence[Point2D]) -> List[float]:
    n = len(verts)
    out: List[float] = []
    for i in range(n):
        p_prev = verts[(i - 1) % n]
        p_curr = verts[i]
        p_next = verts[(i + 1) % n]
        ux, uy = p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]
        vx, vy = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        nu = math.hypot(ux, uy)
        nv = math.hypot(vx, vy)
        if nu < 1e-6 or nv < 1e-6:
            out.append(180.0)
            continue
        cos_a = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
        out.append(math.degrees(math.acos(cos_a)))
    return out


def _order_ccw(verts: Sequence[Point2D]) -> List[Point2D]:
    cx = sum(p[0] for p in verts) / float(len(verts))
    cy = sum(p[1] for p in verts) / float(len(verts))
    return sorted(verts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _prune_collinear_vertices(verts: Sequence[Point2D], diag: float) -> List[Point2D]:
    pts = list(verts)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        # Merge abnormally short edges first
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            if _dist(pts[i], pts[j]) < max(8.0, 0.08 * diag):
                mid = (0.5 * (pts[i][0] + pts[j][0]), 0.5 * (pts[i][1] + pts[j][1]))
                pts[i] = mid
                pts.pop(j)
                changed = True
                break
        if changed:
            continue
        angles = _interior_angles_deg(pts)
        for i, ang in enumerate(angles):
            if ang > 162.0:
                pts.pop(i)
                changed = True
                break
    return pts


def _extract_stable_polygon(pts: Sequence[Point2D], diag: float) -> Optional[List[Point2D]]:
    cnt = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    peri = float(cv2.arcLength(cnt, True))
    if peri < 1e-3:
        return None
    candidates: List[List[Point2D]] = []
    for frac in (0.025, 0.035, 0.045, 0.055, 0.068, 0.08):
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        raw_v = [(float(p[0][0]), float(p[0][1])) for p in approx]
        pruned = _prune_collinear_vertices(raw_v, diag)
        if 3 <= len(pruned) <= 5:
            candidates.append(pruned)
    if not candidates:
        return None
    counts: Dict[int, List[List[Point2D]]] = {}
    for c in candidates:
        counts.setdefault(len(c), []).append(c)
    # Prefer vertex count with lowest polynomial residual & high stability count
    best_poly: Optional[List[Point2D]] = None
    best_cost = 1e9
    for k, group in counts.items():
        if k not in (3, 4):
            continue
        rep = group[len(group) // 2]
        ordered = _order_ccw(rep)
        rms = _point_to_polyline_rms(pts, ordered + [ordered[0]]) / max(diag, 1.0)
        stability_bonus = 0.008 * len(group)
        cost = rms - stability_bonus
        if cost < best_cost:
            best_cost = cost
            best_poly = ordered
    return best_poly


def _try_detect_arrow(pts: Sequence[Point2D], diag: float) -> Optional[Tuple[List[Point2D], float]]:
    """Detect Line + Arrowhead barb at tip using shaft PCA (< 0.04) and sharp hook reversal."""
    n = len(pts)
    if n < 8 or diag < 24.0:
        return None
    # Split off the last 36% of arc-resampled points to inspect shaft vs barb
    split_idx = max(5, int(n * 0.65))
    shaft_pts = pts[:split_idx]
    _, s_l1, s_l2, s_vec = _pca_2d(shaft_pts)
    shaft_ratio = s_l2 / max(s_l1, 1e-9)
    if shaft_ratio >= 0.045:
        return None

    # Locate furthest tip along shaft direction starting near pts[0]
    p0 = pts[0]
    if (pts[split_idx - 1][0] - p0[0]) * s_vec[0] + (pts[split_idx - 1][1] - p0[1]) * s_vec[1] < 0.0:
        s_vec = (-s_vec[0], -s_vec[1])
    projs = [(p[0] - p0[0]) * s_vec[0] + (p[1] - p0[1]) * s_vec[1] for p in pts]
    tip_idx = int(np.argmax(projs))
    if tip_idx < 3 or tip_idx >= n - 2:
        return None

    shaft_len = projs[tip_idx] - min(projs[:tip_idx + 1])
    if shaft_len < 22.0:
        return None

    tail_pts = pts[tip_idx:]
    perp_devs = [abs((p[0] - pts[tip_idx][0]) * (-s_vec[1]) + (p[1] - pts[tip_idx][1]) * s_vec[0]) for p in tail_pts]
    max_perp = max(perp_devs) if perp_devs else 0.0
    if max_perp < max(6.0, 0.06 * shaft_len) or max_perp > 0.45 * shaft_len:
        return None

    # Verify tail goes meaningfully back toward start rather than continuing forward
    tip_proj = projs[tip_idx]
    backtrack = tip_proj - min(projs[tip_idx:])
    if backtrack < max(5.0, 0.05 * shaft_len):
        return None

    start = pts[0]
    tip = pts[tip_idx]
    ang_deg = math.degrees(math.atan2(tip[1] - start[1], tip[0] - start[0]))
    snapped_deg = _snap_angle_deg(ang_deg, ANGLE_SNAP_TOL_DEG)
    rad = math.radians(snapped_deg)
    length = _dist(start, tip)
    mid = (0.5 * (start[0] + tip[0]), 0.5 * (start[1] + tip[1]))
    ux, uy = math.cos(rad), math.sin(rad)
    s_snap = (mid[0] - 0.5 * length * ux, mid[1] - 0.5 * length * uy)
    e_snap = (mid[0] + 0.5 * length * ux, mid[1] + 0.5 * length * uy)

    wing = max(14.0, min(34.0, 0.14 * length))
    barb_angle = math.radians(28.0)
    ca, sa = math.cos(barb_angle), math.sin(barb_angle)
    bx, by = -ux, -uy
    left_wing = (e_snap[0] + wing * (bx * ca - by * sa), e_snap[1] + wing * (bx * sa + by * ca))
    right_wing = (e_snap[0] + wing * (bx * ca + by * sa), e_snap[1] + wing * (-bx * sa + by * ca))
    arrow_poly = [s_snap, e_snap, left_wing, e_snap, right_wing]
    conf = max(0.80, min(0.97, 0.96 - shaft_ratio * 2.5))
    return arrow_poly, conf


def _reconstruct_triangle(verts: Sequence[Point2D]) -> Tuple[List[Point2D], str]:
    pts = _order_ccw(verts)
    sides = [_dist(pts[i], pts[(i + 1) % 3]) for i in range(3)]
    angles = _interior_angles_deg(pts)
    side_cv = float(np.std(sides) / max(float(np.mean(sides)), 1e-6))

    # Snap flat horizontal base if any edge is within < 6 deg of horizontal/vertical
    # First check equilateral
    if all(abs(a - 60.0) <= 12.0 for a in angles) and side_cv < 0.18:
        cx = sum(p[0] for p in pts) / 3.0
        cy = sum(p[1] for p in pts) / 3.0
        r = sum(_dist(p, (cx, cy)) for p in pts) / 3.0
        # Orient upward vertex cleanly if close to vertical
        p_top = min(pts, key=lambda p: p[1])
        base_ang = math.degrees(math.atan2(p_top[1] - cy, p_top[0] - cx))
        base_ang = _snap_angle_deg(base_ang, 10.0)
        tri = [
            (
                cx + r * math.cos(math.radians(base_ang + 120.0 * k)),
                cy + r * math.sin(math.radians(base_ang + 120.0 * k)),
            )
            for k in range(3)
        ]
        return tri + [tri[0]], "TRIANGLE"

    # Check right triangle (one angle 80°..100°)
    right_idx = next((i for i, a in enumerate(angles) if 80.0 <= a <= 100.0), None)
    if right_idx is not None:
        o = pts[right_idx]
        p_prev = pts[(right_idx - 1) % 3]
        p_next = pts[(right_idx + 1) % 3]
        len1 = _dist(o, p_prev)
        len2 = _dist(o, p_next)
        ang1 = _snap_angle_deg(math.degrees(math.atan2(p_prev[1] - o[1], p_prev[0] - o[0])), ANGLE_SNAP_TOL_DEG)
        r1 = math.radians(ang1)
        # Ensure perpendicular second leg preserving handedness
        cross = (p_prev[0] - o[0]) * (p_next[1] - o[1]) - (p_prev[1] - o[1]) * (p_next[0] - o[0])
        r2 = r1 + (0.5 * math.pi if cross >= 0 else -0.5 * math.pi)
        q_prev = (o[0] + len1 * math.cos(r1), o[1] + len1 * math.sin(r1))
        q_next = (o[0] + len2 * math.cos(r2), o[1] + len2 * math.sin(r2))
        tri = [o, q_prev, q_next]
        return tri + [tri[0]], "TRIANGLE"

    # General / isosceles triangle
    tri = list(pts)
    return tri + [tri[0]], "TRIANGLE"


def _reconstruct_quad(verts: Sequence[Point2D]) -> Tuple[List[Point2D], str, float]:
    pts = _order_ccw(verts)
    edges = [_dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    angles = _interior_angles_deg(pts)
    min_e, max_e = min(edges), max(edges)
    side_uniformity = min_e / max(max_e, 1e-6)
    opp_ratio_a = min(edges[0], edges[2]) / max(edges[0], edges[2], 1e-6)
    opp_ratio_b = min(edges[1], edges[3]) / max(edges[1], edges[3], 1e-6)
    right_count = sum(1 for a in angles if 78.0 <= a <= 102.0)

    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    # Oriented bounding rectangle via cv2.minAreaRect
    rect = cv2.minAreaRect(np.asarray(pts, dtype=np.float32))
    (rcx, rcy), (rw, rh), r_ang = rect
    r_ang_snapped = _snap_angle_deg(float(r_ang), ANGLE_SNAP_TOL_DEG)

    if right_count >= 3:
        aspect = min(rw, rh) / max(rw, rh, 1e-6)
        if aspect >= 0.83 and side_uniformity >= 0.80:
            side = 0.5 * (rw + rh)
            box = cv2.boxPoints(((rcx, rcy), (side, side), r_ang_snapped))
            quad = [(float(p[0]), float(p[1])) for p in box]
            return quad + [quad[0]], "SQUARE", 0.94
        box = cv2.boxPoints(((rcx, rcy), (rw, rh), r_ang_snapped))
        quad = [(float(p[0]), float(p[1])) for p in box]
        return quad + [quad[0]], "RECTANGLE", 0.93

    # Non-right angles: Rhombus vs Parallelogram vs Rectangle fallback
    # Diagonals check
    u_x, u_y = pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]
    v_x, v_y = pts[3][0] - pts[1][0], pts[3][1] - pts[1][1]
    du = math.hypot(u_x, u_y)
    dv = math.hypot(v_x, v_y)
    if du > 1e-6 and dv > 1e-6:
        diag_cos = abs((u_x * v_x + u_y * v_y) / (du * dv))
        if diag_cos < 0.30 and side_uniformity >= 0.78:
            # Exact orthogonal-diagonal Rhombus centered at (cx, cy)
            ang_u = _snap_angle_deg(math.degrees(math.atan2(u_y, u_x)), ANGLE_SNAP_TOL_DEG)
            ru = math.radians(ang_u)
            rv = ru + 0.5 * math.pi
            hu, hv = 0.5 * du, 0.5 * dv
            quad = [
                (cx - hu * math.cos(ru), cy - hu * math.sin(ru)),
                (cx - hv * math.cos(rv), cy - hv * math.sin(rv)),
                (cx + hu * math.cos(ru), cy + hu * math.sin(ru)),
                (cx + hv * math.cos(rv), cy + hv * math.sin(rv)),
            ]
            return quad + [quad[0]], "RHOMBUS", 0.89

    if opp_ratio_a >= 0.76 and opp_ratio_b >= 0.76:
        # Enforce exact parallelogram pairs around centroid
        p0, p1 = pts[0], pts[1]
        p2, p3 = pts[2], pts[3]
        vx1 = 0.5 * ((p1[0] - p0[0]) + (p2[0] - p3[0]))
        vy1 = 0.5 * ((p1[1] - p0[1]) + (p2[1] - p3[1]))
        vx2 = 0.5 * ((p3[0] - p0[0]) + (p2[0] - p1[0]))
        vy2 = 0.5 * ((p3[1] - p0[1]) + (p2[1] - p1[1]))
        quad = [
            (cx - 0.5 * vx1 - 0.5 * vx2, cy - 0.5 * vy1 - 0.5 * vy2),
            (cx + 0.5 * vx1 - 0.5 * vx2, cy + 0.5 * vy1 - 0.5 * vy2),
            (cx + 0.5 * vx1 + 0.5 * vx2, cy + 0.5 * vy1 + 0.5 * vy2),
            (cx - 0.5 * vx1 + 0.5 * vx2, cy - 0.5 * vy1 + 0.5 * vy2),
        ]
        return quad + [quad[0]], "PARALLELOGRAM", 0.86

    # Fallback: if reasonably rectangular angles >=2 and solid rectangularity
    quad = list(pts)
    return quad + [quad[0]], "RECTANGLE", 0.74


def beautify_stroke(
    raw_points: Sequence[Point2D],
    auto_shape_enabled: bool = True,
    confidence_gate: float = CONFIDENCE_GATE,
    closed_gap_ratio: float = CLOSED_LOOP_GAP_RATIO,
) -> BeautifiedShapeResult:
    """
    Full 3-step Hybrid Shape Recognition & Beautification pipeline.
    Preserves `raw_points` verbatim for non-destructive rollback.
    """
    raw_copy = [(float(p[0]), float(p[1])) for p in raw_points]
    clean = deduplicate_points(raw_copy, min_step=1.0)
    if len(clean) < 3:
        return BeautifiedShapeResult(
            shape_type="FREEHAND",
            shape_label_vi=SHAPE_LABELS_VI["FREEHAND"],
            confidence=0.0,
            is_closed=False,
            raw_points=raw_copy,
            beautified_points=raw_copy,
        )

    if not auto_shape_enabled:
        smooth = catmull_rom_spline(clean, subdivisions=6)
        return BeautifiedShapeResult(
            shape_type="FREEHAND",
            shape_label_vi=SHAPE_LABELS_VI["FREEHAND"],
            confidence=0.0,
            is_closed=False,
            raw_points=raw_copy,
            beautified_points=smooth,
        )

    _, _, _, _, diag = bbox_and_diag(clean)
    total_len = polyline_length(clean)
    if total_len < 16.0 or diag < 14.0:
        smooth = catmull_rom_spline(clean, subdivisions=5)
        return BeautifiedShapeResult(
            shape_type="FREEHAND",
            shape_label_vi=SHAPE_LABELS_VI["FREEHAND"],
            confidence=0.0,
            is_closed=False,
            raw_points=raw_copy,
            beautified_points=smooth,
        )

    # Step 1: RDP Denoise + Closed-Loop Gap Auto-Closure + Resample N = 64
    eps = max(2.0, 0.014 * diag)
    simplified = rdp_simplify(clean, eps)
    gap = _dist(clean[0], clean[-1])
    gap_ratio = max(0.05, min(0.35, float(closed_gap_ratio)))
    is_closed = len(clean) >= 6 and (
        gap < gap_ratio * total_len or gap < gap_ratio * diag
    )

    work_pts = list(clean)
    if is_closed and gap > 1.0:
        work_pts.append(clean[0])
    resampled = resample_arc_length(work_pts, RESAMPLE_N)

    # Step 2A: Open Shapes (PCA Line & Arrow)
    if not is_closed:
        arrow_match = _try_detect_arrow(resampled, diag)
        if arrow_match is not None:
            arrow_pts, arrow_conf = arrow_match
            if arrow_conf >= confidence_gate:
                return BeautifiedShapeResult(
                    shape_type="ARROW",
                    shape_label_vi=SHAPE_LABELS_VI["ARROW"],
                    confidence=round(arrow_conf, 3),
                    is_closed=False,
                    raw_points=raw_copy,
                    beautified_points=arrow_pts,
                    metadata={"diag": round(diag, 2)},
                )

        centroid, l1, l2, uvec = _pca_2d(resampled)
        eig_ratio = l2 / max(l1, 1e-9)
        if eig_ratio < PCA_LINE_EIG_RATIO and len(simplified) <= 6:
            projs = [
                (p[0] - centroid[0]) * uvec[0] + (p[1] - centroid[1]) * uvec[1]
                for p in resampled
            ]
            t_min, t_max = min(projs), max(projs)
            # Keep start -> end ordering
            p0_proj = (clean[0][0] - centroid[0]) * uvec[0] + (clean[0][1] - centroid[1]) * uvec[1]
            p1_proj = (clean[-1][0] - centroid[0]) * uvec[0] + (clean[-1][1] - centroid[1]) * uvec[1]
            if p1_proj < p0_proj:
                uvec = (-uvec[0], -uvec[1])
            line_len = t_max - t_min
            raw_ang = math.degrees(math.atan2(uvec[1], uvec[0]))
            snap_ang = _snap_angle_deg(raw_ang, ANGLE_SNAP_TOL_DEG)
            rad = math.radians(snap_ang)
            mid_proj = 0.5 * (t_min + t_max)
            mid_pt = (
                centroid[0] + mid_proj * math.cos(math.atan2(uvec[1], uvec[0])),
                centroid[1] + mid_proj * math.sin(math.atan2(uvec[1], uvec[0])),
            )
            ux, uy = math.cos(rad), math.sin(rad)
            p_start = (mid_pt[0] - 0.5 * line_len * ux, mid_pt[1] - 0.5 * line_len * uy)
            p_end = (mid_pt[0] + 0.5 * line_len * ux, mid_pt[1] + 0.5 * line_len * uy)
            line_conf = max(0.80, min(0.99, 1.0 - (eig_ratio / PCA_LINE_EIG_RATIO) * 0.18))
            return BeautifiedShapeResult(
                shape_type="LINE",
                shape_label_vi=SHAPE_LABELS_VI["LINE"],
                confidence=round(line_conf, 3),
                is_closed=False,
                raw_points=raw_copy,
                beautified_points=[p_start, p_end],
                metadata={
                    "pca_eig_ratio": round(eig_ratio, 5),
                    "snapped_angle_deg": round(snap_ang, 2),
                },
            )

        smooth = catmull_rom_spline(clean, subdivisions=6)
        return BeautifiedShapeResult(
            shape_type="FREEHAND",
            shape_label_vi=SHAPE_LABELS_VI["FREEHAND"],
            confidence=round(max(0.0, 0.70 - eig_ratio), 3),
            is_closed=False,
            raw_points=raw_copy,
            beautified_points=smooth,
            metadata={"pca_eig_ratio": round(eig_ratio, 5)},
        )

    # Step 2B: Closed Shapes (Circle, Ellipse, Polygons + $1/$P Template Scores)
    tmpl_scores = _TEMPLATE_BANK.score_all(resampled)
    cnt_np = np.asarray(resampled, dtype=np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(cnt_np)))
    peri = float(cv2.arcLength(cnt_np, True))
    circularity = (4.0 * math.pi * area) / max(peri * peri, 1e-6)

    # Candidate scoring list: (shape_type, confidence, beautified_pts, meta)
    candidates: List[Tuple[str, float, List[Point2D], Dict[str, float]]] = []

    # 1) Circle Fit (Circularity + Radial CV < 0.14 + Kasa LS)
    kasa = _fit_circle_kasa(resampled)
    if kasa is not None:
        cx, cy, r, radial_cv = kasa
        circle_tmpl = tmpl_scores.get("CIRCLE", 0.0)
        if radial_cv < CIRCLE_RADIAL_CV_MAX and circularity >= 0.76:
            geom_circle = max(0.0, min(1.0, 1.0 - (radial_cv / CIRCLE_RADIAL_CV_MAX) * 0.24))
            conf = 0.68 * geom_circle + 0.32 * circle_tmpl
            candidates.append((
                "CIRCLE",
                conf,
                _sample_circle(cx, cy, r),
                {"circularity": round(circularity, 3), "radial_cv": round(radial_cv, 4), "radius": round(r, 2)},
            ))

    # Check Polygon stability first so elongated Rectangles never masquerade as Ellipses
    poly = _extract_stable_polygon(resampled, diag)
    poly_rms_norm = 1e9
    if poly is not None:
        poly_rms_norm = _point_to_polyline_rms(resampled, poly + [poly[0]]) / max(diag, 1.0)

    rect_min = cv2.minAreaRect(cnt_np)
    rect_area = max(1.0, float(rect_min[1][0] * rect_min[1][1]))
    rectangularity = area / rect_area

    # 2) Ellipse Fit (cv2.fitEllipse)
    if len(resampled) >= 5:
        try:
            (ecx, ecy), (w_ax, h_ax), e_ang = cv2.fitEllipse(cnt_np)
            rx, ry = 0.5 * max(w_ax, 1.0), 0.5 * max(h_ax, 1.0)
            ell_rms = _ellipse_normalized_rms(resampled, ecx, ecy, rx, ry, e_ang)
            aspect = max(rx, ry) / max(min(rx, ry), 1e-6)
            if aspect >= 1.15 and ell_rms < 0.125 and math.isfinite(ell_rms):
                # Penalize ellipse if the stroke clearly shows sharp polygonal corners & high rectangularity
                is_boxy_quad = (poly is not None and len(poly) == 4 and poly_rms_norm < 0.036 and rectangularity > 0.87)
                if not is_boxy_quad:
                    ell_geom = max(0.0, min(1.0, 1.0 - ell_rms * 2.2))
                    ell_tmpl = tmpl_scores.get("ELLIPSE", 0.0)
                    conf = 0.65 * ell_geom + 0.35 * ell_tmpl
                    snap_e_ang = _snap_angle_deg(float(e_ang), ANGLE_SNAP_TOL_DEG)
                    candidates.append((
                        "ELLIPSE",
                        conf,
                        _sample_ellipse(ecx, ecy, rx, ry, snap_e_ang),
                        {"ellipse_rms": round(ell_rms, 4), "aspect": round(aspect, 2), "rectangularity": round(rectangularity, 3)},
                    ))
        except cv2.error as err:
            _ = err

    # 3) Triangle / Square / Rectangle / Rhombus / Parallelogram
    if poly is not None and area > 64.0 and poly_rms_norm < 0.068:
        poly_geom = max(0.0, min(1.0, 1.0 - (poly_rms_norm / 0.068) * 0.28))
        if len(poly) == 3:
            tri_pts, tri_kind = _reconstruct_triangle(poly)
            conf = 0.65 * poly_geom + 0.35 * tmpl_scores.get("TRIANGLE", 0.0)
            candidates.append((
                tri_kind,
                conf,
                tri_pts,
                {"poly_rms_norm": round(poly_rms_norm, 4), "vertices": 3},
            ))
        elif len(poly) == 4:
            quad_pts, quad_kind, q_prior = _reconstruct_quad(poly)
            q_tmpl = tmpl_scores.get(quad_kind, tmpl_scores.get("RECTANGLE", 0.0))
            conf = 0.45 * poly_geom + 0.25 * q_prior + 0.30 * q_tmpl
            candidates.append((
                quad_kind,
                conf,
                quad_pts,
                {"poly_rms_norm": round(poly_rms_norm, 4), "rectangularity": round(rectangularity, 3), "vertices": 4},
            ))

    if candidates:
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_type, best_conf, best_pts, best_meta = candidates[0]
        if best_conf >= confidence_gate:
            return BeautifiedShapeResult(
                shape_type=best_type,
                shape_label_vi=SHAPE_LABELS_VI.get(best_type, best_type),
                confidence=round(best_conf, 3),
                is_closed=True,
                raw_points=raw_copy,
                beautified_points=best_pts,
                metadata=best_meta,
            )

    # Step 3: Confidence < 0.78 -> Non-destructive smooth Catmull-Rom spline (never mis-snap handwriting)
    smooth = catmull_rom_spline(clean, subdivisions=6)
    max_rejected = round(candidates[0][1], 3) if candidates else 0.0
    return BeautifiedShapeResult(
        shape_type="FREEHAND",
        shape_label_vi=SHAPE_LABELS_VI["FREEHAND"],
        confidence=max_rejected,
        is_closed=is_closed,
        raw_points=raw_copy,
        beautified_points=smooth,
        metadata={"circularity": round(circularity, 3), "poly_rms_norm": round(poly_rms_norm, 4)},
    )
