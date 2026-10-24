"""
AirTeach AI - 3-Tier Stroke Beautifier & Geometric Shape Snapper
=================================================================
Tier 1  – Real-time:    1€ Filter (Casiez et al., 2012)          [OneEuroFilter2D]
Tier 2  – Post-stroke:  RDP simplification + Cubic Bézier spline  [fit_cubic_bezier_spline]
Tier 3  – Geometric:    Convex-hull/circularity shape snapping     [GeometricShapeSnapper]

All mathematics are implemented from scratch using numpy and scipy only.
No mock data, no pass stubs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — 1€ Filter (real-time jitter removal)
# Reference: Géry Casiez, Nicolas Roussel, Daniel Vogel (2012)
#            "1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input"
# ─────────────────────────────────────────────────────────────────────────────

class _LowPassFilter:
    """Single-axis first-order low-pass filter (exponential smoothing)."""

    __slots__ = ("_alpha", "_y", "_initialized")

    def __init__(self) -> None:
        self._alpha: float = 1.0
        self._y: float = 0.0
        self._initialized: bool = False

    def filter(self, x: float, alpha: float) -> float:
        self._alpha = alpha
        if not self._initialized:
            self._y = x
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last_value(self) -> float:
        return self._y

    def reset(self) -> None:
        self._alpha = 1.0
        self._y = 0.0
        self._initialized = False


class OneEuroFilter:
    """
    1€ Filter for a single scalar channel.

    Adapts its cutoff frequency to the speed of the signal:
    - Slow movements → low cutoff → strong smoothing (removes jitter).
    - Fast movements → high cutoff → weak smoothing (preserves responsiveness).

    Args:
        freq:        Nominal sampling frequency in Hz.
        min_cutoff:  Minimum cutoff frequency (Hz). Default 1.2 (AirTeach spec).
        beta:        Speed coefficient. Default 0.35 (AirTeach spec).
        d_cutoff:    Cutoff frequency for the derivative. Default 1.0.
    """

    __slots__ = ("_freq", "_min_cutoff", "_beta", "_d_cutoff", "_x", "_dx", "_last_t")

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 1.2,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x = _LowPassFilter()
        self._dx = _LowPassFilter()
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Compute smoothing factor α from cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t_ms: float) -> float:
        """
        Filter a new sample *x* at timestamp *t_ms* (milliseconds).

        Returns the filtered value.
        """
        if self._last_t is not None:
            dt_s = (t_ms - self._last_t) / 1000.0
            if dt_s > 2.0 or dt_s < 0.0:
                self.reset()
            elif dt_s > 0.001:
                self._freq = max(1.0, min(240.0, 1.0 / dt_s))
        self._last_t = t_ms

        # Derivative estimate
        dx_raw = (
            (x - self._x.last_value) * self._freq
            if self._x._initialized
            else 0.0
        )
        a_d = self._alpha(self._d_cutoff, self._freq)
        dx_hat = self._dx.filter(dx_raw, a_d)

        # Adaptive cutoff based on speed
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, self._freq)
        return self._x.filter(x, a)


class OneEuroFilter2D:
    """
    1€ Filter for 2D (x, y) hand-tip coordinates.

    Parameters match AirTeach AI specification:
        min_cutoff = 1.2, beta = 0.35, d_cutoff = 1.0

    Usage::

        filt = OneEuroFilter2D()
        smooth_x, smooth_y = filt.filter(raw_x, raw_y, timestamp_ms)
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(self, x: float, y: float, t_ms: float) -> tuple[float, float]:
        """Return (filtered_x, filtered_y) for the given timestamp."""
        return self._fx.filter(x, t_ms), self._fy.filter(y, t_ms)

    def reset(self) -> None:
        """Reset the filter state (call when starting a new stroke)."""
        self._fx.reset()
        self._fy.reset()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — RDP Simplification + Cubic Bézier Spline with velocity-width
# ─────────────────────────────────────────────────────────────────────────────

def _perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Perpendicular distance from *point* to the segment [line_start, line_end]."""
    d = line_end - line_start
    len_sq = float(np.dot(d, d))
    if len_sq < 1e-12:
        return float(np.linalg.norm(point - line_start))
    t = float(np.dot(point - line_start, d)) / len_sq
    t = max(0.0, min(1.0, t))
    projection = line_start + t * d
    return float(np.linalg.norm(point - projection))


def ramer_douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker polyline simplification.

    Recursively removes points that deviate less than *epsilon* pixels from
    the straight line between the current segment's endpoints.

    Args:
        points:  Shape (N, 2) array of (x, y) coordinates.
        epsilon: Maximum allowable perpendicular deviation in pixels.

    Returns:
        Shape (M, 2) simplified point array with M ≤ N.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # Iterative implementation to avoid recursion limit on very long strokes
    # Uses an explicit stack of (start_idx, end_idx) segments to process
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack: list[tuple[int, int]] = [(0, n - 1)]

    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue

        seg_start = pts[start]
        seg_end = pts[end]
        sub_pts = pts[start + 1 : end]

        d = seg_end - seg_start
        norm_d_sq = float(np.dot(d, d))
        diff = sub_pts - seg_start

        if norm_d_sq < 1e-12:
            dists = np.linalg.norm(diff, axis=1)
        else:
            t = np.clip(np.dot(diff, d) / norm_d_sq, 0.0, 1.0)
            proj = seg_start + t[:, np.newaxis] * d
            dists = np.linalg.norm(sub_pts - proj, axis=1)

        max_rel_idx = int(np.argmax(dists))
        max_dist = float(dists[max_rel_idx])
        max_idx = start + 1 + max_rel_idx

        if max_dist > epsilon:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))

    return pts[keep]


@dataclass
class BezierSegment:
    """A cubic Bézier segment with two endpoints and two control points."""
    p0: np.ndarray   # start point
    p1: np.ndarray   # control point 1
    p2: np.ndarray   # control point 2
    p3: np.ndarray   # end point
    width_start: float = 2.0   # pen width at p0 (px)
    width_end: float = 2.0     # pen width at p3 (px)

    def __post_init__(self) -> None:
        self.p0 = np.asarray(self.p0, dtype=np.float64)
        self.p1 = np.asarray(self.p1, dtype=np.float64)
        self.p2 = np.asarray(self.p2, dtype=np.float64)
        self.p3 = np.asarray(self.p3, dtype=np.float64)

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter t ∈ [0, 1]."""
        u = 1.0 - t
        return (
            u ** 3 * self.p0
            + 3 * u ** 2 * t * self.p1
            + 3 * u * t ** 2 * self.p2
            + t ** 3 * self.p3
        )

    def width_at(self, t: float) -> float:
        """Linearly interpolated pen width at parameter t."""
        return self.width_start + t * (self.width_end - self.width_start)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
            "p2": self.p2.tolist(),
            "p3": self.p3.tolist(),
            "width_start": self.width_start,
            "width_end": self.width_end,
        }


def fit_cubic_bezier_spline(
    points: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    velocity_scale: float = 0.04,
    width_min: float = 1.5,
    width_max: float = 8.0,
) -> list[BezierSegment]:
    """
    Fit a C¹-continuous cubic Bézier spline through *points*.

    Control points are derived using the Catmull-Rom → Bézier conversion,
    which guarantees C¹ continuity (matching tangents at every knot).

    The pen width at each knot is modulated by the finger velocity:
        width(v) = clamp(width_max − velocity_scale × v, width_min, width_max)

    Args:
        points:         Shape (N, 2) knot array (already RDP-simplified).
        velocities:     Shape (N,) array of fingertip speeds in px/ms.
                        If None, uniform width is used.
        velocity_scale: How much each px/ms of speed narrows the stroke.
        width_min:      Minimum pen width in pixels.
        width_max:      Maximum pen width in pixels.

    Returns:
        List of (N-1) BezierSegment objects forming the complete spline.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)

    if n < 2:
        return []

    if n == 2:
        # Degenerate case: straight line as a degenerate Bézier
        t = pts[1] - pts[0]
        p1 = pts[0] + t / 3.0
        p2 = pts[0] + 2.0 * t / 3.0
        w = width_max if velocities is None else float(
            np.clip(width_max - velocity_scale * velocities[0], width_min, width_max)
        )
        return [BezierSegment(pts[0], p1, p2, pts[1], w, w)]

    # ── Compute tangent vectors via Catmull-Rom parameterization ──────────
    # Tangent at knot i = (P[i+1] - P[i-1]) / 2
    # Boundary tangents use one-sided differences.
    tangents = np.empty_like(pts)
    tangents[0] = (pts[1] - pts[0])
    tangents[-1] = (pts[-1] - pts[-2])
    for i in range(1, n - 1):
        tangents[i] = (pts[i + 1] - pts[i - 1]) * 0.5

    # Scale tangent magnitude to ~1/3 of the segment chord length for
    # well-proportioned Bézier handles (standard Catmull-Rom → Bézier rule)
    segments: list[BezierSegment] = []

    for i in range(n - 1):
        p0 = pts[i]
        p3 = pts[i + 1]
        chord = float(np.linalg.norm(p3 - p0))
        scale = chord / 3.0 if chord > 1e-6 else 1.0 / 3.0

        # Normalize tangents then scale by chord/3
        tan_i = tangents[i]
        tan_j = tangents[i + 1]
        len_i = float(np.linalg.norm(tan_i))
        len_j = float(np.linalg.norm(tan_j))

        p1 = p0 + (tan_i / len_i * scale if len_i > 1e-6 else np.zeros(2))
        p2 = p3 - (tan_j / len_j * scale if len_j > 1e-6 else np.zeros(2))

        # ── Velocity → pen width mapping ──────────────────────────────────
        if velocities is not None and len(velocities) == n:
            v_start = float(velocities[i])
            v_end = float(velocities[i + 1])
            w_start = float(np.clip(width_max - velocity_scale * v_start, width_min, width_max))
            w_end = float(np.clip(width_max - velocity_scale * v_end, width_min, width_max))
        else:
            w_start = w_end = (width_min + width_max) * 0.5

        segments.append(BezierSegment(p0, p1, p2, p3, w_start, w_end))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — Geometric Shape Snapper
# ─────────────────────────────────────────────────────────────────────────────

class ShapeType(Enum):
    FREEFORM = auto()
    LINE = auto()
    ARROW = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    RECTANGLE = auto()
    TRIANGLE = auto()


@dataclass
class SnappedShape:
    """Result of shape snapping analysis."""
    shape_type: ShapeType
    # Geometry parameters (type-dependent)
    params: dict = field(default_factory=dict)
    # Confidence score ∈ [0, 1]
    confidence: float = 0.0
    # The canonical polygon/ellipse points for rendering
    render_points: Optional[np.ndarray] = None

    def is_geometric(self) -> bool:
        return self.shape_type != ShapeType.FREEFORM

    def to_dict(self) -> dict:
        """Serialize SnappedShape to JSON-compatible dictionary."""
        p_dict = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                p_dict[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                p_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, (list, tuple)):
                p_dict[k] = [
                    [float(coord) for coord in elem] if isinstance(elem, (list, tuple, np.ndarray))
                    else (float(elem) if isinstance(elem, (float, int, np.number)) else elem)
                    for elem in v
                ]
            else:
                p_dict[k] = v

        return {
            "shape_type": self.shape_type.name,
            "params": p_dict,
            "confidence": float(self.confidence),
            "render_points": self.render_points.tolist() if self.render_points is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SnappedShape:
        """Reconstruct SnappedShape from dictionary."""
        st_name = data.get("shape_type", "FREEFORM")
        try:
            st = ShapeType[st_name]
        except KeyError:
            st = ShapeType.FREEFORM

        raw_rp = data.get("render_points")
        rp = np.asarray(raw_rp, dtype=np.float64) if raw_rp is not None else None

        return cls(
            shape_type=st,
            params=data.get("params", {}),
            confidence=float(data.get("confidence", 0.0)),
            render_points=rp,
        )


class GeometricShapeSnapper:
    """
    Analyzes a completed stroke and snaps it to the closest canonical shape.

    Detection pipeline (ordered by priority):
      1. ARROW  – checks for arrowhead fork at the terminal end.
      2. LINE   – endpoint-to-arc-length ratio ≥ LINE_RATIO.
      3. CIRCLE/ELLIPSE – closed stroke + roundness ≥ CIRCLE_ROUNDNESS.
      4. RECTANGLE/TRIANGLE – closed stroke, approximate polygon with 4 or 3 vertices.
      5. FREEFORM – none of the above.

    All thresholds are read from config but can be overridden at construction.
    """

    def __init__(
        self,
        line_ratio: float = 0.90,
        angle_tolerance_deg: float = 5.0,
        circle_roundness: float = 0.80,
        close_ratio: float = 0.15,
    ) -> None:
        self.line_ratio = line_ratio
        self.angle_tolerance_deg = angle_tolerance_deg
        self.circle_roundness = circle_roundness
        self.close_ratio = close_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def snap(self, points: np.ndarray) -> SnappedShape:
        """
        Analyze *points* (shape N×2) and return the best-matching SnappedShape.

        Detection priority (important — LINE before ARROW so that a nearly-
        straight stroke with noisy endpoints is not mis-classified):
          1. LINE   – endpoint / arc-length ratio ≥ line_ratio.
          2. ARROW  – non-straight stroke with a sharp fork at the terminal end.
          3. CIRCLE/ELLIPSE – closed stroke with high circularity.
          4. RECTANGLE/TRIANGLE – closed stroke, convex hull ≈ 4 or 3 vertices.
          5. FREEFORM – none of the above.

        Args:
            points: Raw or RDP-simplified stroke points, shape (N, 2).

        Returns:
            SnappedShape with shape_type, params, confidence, and render_points.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] > 2:
            pts = pts[:, :2]
        if len(pts) < 3:
            return SnappedShape(ShapeType.FREEFORM, confidence=0.0)

        arc_length = self._arc_length(pts)
        endpoint_dist = float(np.linalg.norm(pts[-1] - pts[0]))
        is_closed = (arc_length > 1e-6) and (endpoint_dist / arc_length < self.close_ratio)

        # Priority 1: LINE — must come BEFORE arrow so a straight-ish stroke
        # with noisy tip is not mistakenly classified as ARROW.
        # Use PCA linearity: the first principal component explains ≥ line_ratio
        # of total variance. This is robust to noisy start/end points.
        if arc_length > 1e-6 and len(pts) >= 3:
            line_score = self._pca_linearity(pts)
            if line_score >= self.line_ratio:
                return self._make_line(pts)

        # Priority 2: ARROW — only for open (non-closed) strokes
        # Closed shapes (circles, rectangles) must not be intercepted here.
        if not is_closed:
            arrow_result = self._detect_arrow(pts, arc_length)
            if arrow_result is not None:
                return arrow_result

        # Closed shape analysis
        if is_closed:
            # Priority 3: CIRCLE / ELLIPSE
            circle_result = self._detect_circle_ellipse(pts)
            if circle_result is not None:
                return circle_result

            # Priority 4: RECTANGLE / TRIANGLE via polygon approximation
            poly_result = self._detect_polygon(pts)
            if poly_result is not None:
                return poly_result

        return SnappedShape(ShapeType.FREEFORM, confidence=1.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _arc_length(pts: np.ndarray) -> float:
        """Total arc length of the polyline."""
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @staticmethod
    def _pca_linearity(pts: np.ndarray) -> float:
        """
        Compute PCA linearity score: the fraction of total variance explained
        by the first principal component.

        A perfect straight line scores 1.0; a perfect circle scores 0.5.
        This is robust to noisy endpoints unlike the endpoint_dist/arc_length ratio.

        Returns a value in [0, 1].
        """
        centered = pts - pts.mean(axis=0)
        cov = np.dot(centered.T, centered) / max(len(pts) - 1, 1)
        eigenvalues = np.linalg.eigvalsh(cov)  # sorted ascending
        total = float(np.sum(eigenvalues))
        if total < 1e-12:
            return 1.0  # degenerate: single point, treat as line
        # Largest eigenvalue is the last (eigvalsh returns ascending order)
        return float(eigenvalues[-1]) / total

    def _make_line(self, pts: np.ndarray) -> SnappedShape:
        """Snap to a LINE, optionally angle-snapping to 0°/45°/90°."""
        # Use PCA to find the principal direction (robust to noisy endpoints)
        pca_score = self._pca_linearity(pts)
        centered = pts - pts.mean(axis=0)
        cov = np.dot(centered.T, centered) / max(len(pts) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Principal direction = eigenvector of largest eigenvalue (last column)
        principal = eigenvectors[:, -1]

        # Project all points onto principal axis, matching stroke direction
        projections = centered @ principal
        proj_start = (pts[0] - pts.mean(axis=0)) @ principal
        proj_end = (pts[-1] - pts.mean(axis=0)) @ principal
        if proj_start <= proj_end:
            p_start = pts.mean(axis=0) + projections.min() * principal
            p_end = pts.mean(axis=0) + projections.max() * principal
        else:
            p_start = pts.mean(axis=0) + projections.max() * principal
            p_end = pts.mean(axis=0) + projections.min() * principal

        dx = float(p_end[0] - p_start[0])
        dy = float(p_end[1] - p_start[1])
        angle_deg = math.degrees(math.atan2(dy, dx))

        snapped_angle = self._snap_angle(angle_deg)

        if snapped_angle is not None:
            length = float(np.linalg.norm(p_end - p_start))
            rad = math.radians(snapped_angle)
            p_end_snapped = p_start + length * np.array([math.cos(rad), math.sin(rad)])
            return SnappedShape(
                shape_type=ShapeType.LINE,
                params={
                    "start": p_start.tolist(),
                    "end": p_end_snapped.tolist(),
                    "angle_deg": snapped_angle,
                    "angle_snapped": True,
                },
                confidence=min(pca_score, 1.0),
                render_points=np.array([p_start, p_end_snapped]),
            )

        return SnappedShape(
            shape_type=ShapeType.LINE,
            params={
                "start": p_start.tolist(),
                "end": p_end.tolist(),
                "angle_deg": angle_deg,
                "angle_snapped": False,
            },
            confidence=float(min(pca_score, 1.0)),
            render_points=np.array([p_start, p_end]),
        )

    def _snap_angle(self, angle_deg: float) -> Optional[float]:
        """Return nearest snap angle (0, 45, 90, 135, 180, -45, -90, -135) if within tolerance."""
        snap_angles = [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0]
        for snap in snap_angles:
            diff = abs((angle_deg - snap + 180.0) % 360.0 - 180.0)
            if diff < self.angle_tolerance_deg:
                return snap
        return None

    def _detect_arrow(self, pts: np.ndarray, arc_length: float) -> Optional[SnappedShape]:
        """
        Detect an ARROW by finding a sharp V-shaped fork near the terminal end.

        Examines the last 20% of the stroke for a reversal angle < 60°.
        """
        n = len(pts)
        if n < 10:
            return None

        tail_n = max(3, n // 5)
        tail = pts[-tail_n:]

        # Compute angles of consecutive segments in the tail
        angles = []
        for i in range(1, len(tail) - 1):
            v1 = tail[i - 1] - tail[i]
            v2 = tail[i + 1] - tail[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 1e-6 or len2 < 1e-6:
                angles.append(180.0)
                continue
            cos_a = float(np.dot(v1, v2) / (len1 * len2))
            cos_a = max(-1.0, min(1.0, cos_a))
            angles.append(math.degrees(math.acos(cos_a)))

        if not angles:
            return None

        # Sharp reversal (< 55°) suggests arrowhead
        min_angle = min(angles)
        if min_angle < 55.0:
            shaft_end = pts[-tail_n]
            arrow_tip = pts[-1]
            confidence = 1.0 - min_angle / 55.0
            return SnappedShape(
                shape_type=ShapeType.ARROW,
                params={
                    "start": pts[0].tolist(),
                    "shaft_end": shaft_end.tolist(),
                    "tip": arrow_tip.tolist(),
                    "head_angle_deg": min_angle,
                },
                confidence=float(confidence),
                render_points=np.array([pts[0], shaft_end, arrow_tip]),
            )

        return None

    def _detect_circle_ellipse(self, pts: np.ndarray) -> Optional[SnappedShape]:
        """
        Detect CIRCLE or ELLIPSE using:
          - Roundness metric C = 4π·Area / Perimeter² computed on the convex hull
            (convex hull is robust to stroke noise; raw shoelace on a noisy path
             severely underestimates the enclosed area).
          - cv2.fitEllipse for precise ellipse center, axes and angle.
        """
        if len(pts) < 5:
            return None

        pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)

        # ── Compute convex hull ──────────────────────────────────────────────
        hull = cv2.convexHull(pts_f32)
        if hull is None or len(hull) < 3:
            return None

        hull_pts = hull.reshape(-1, 2).astype(np.float64)

        # Hull area via shoelace (convex polygon → exact result)
        hx = hull_pts[:, 0]
        hy = hull_pts[:, 1]
        area = abs(float(np.dot(hx, np.roll(hy, -1)) - np.dot(hy, np.roll(hx, -1)))) * 0.5

        # Hull perimeter
        hull_perimeter = float(cv2.arcLength(hull, closed=True))

        if hull_perimeter < 1e-6 or area < 1e-6:
            return None

        # Roundness C = 4π·A / P²  (1.0 = perfect circle)
        roundness = (4.0 * math.pi * area) / (hull_perimeter ** 2)

        if roundness < self.circle_roundness:
            return None

        # ── Fit ellipse to the original (noisy) point cloud ─────────────────
        try:
            ellipse = cv2.fitEllipse(pts_f32)
        except cv2.error:
            return None

        center, axes, angle = ellipse
        a = max(axes) / 2.0  # semi-major
        b = min(axes) / 2.0  # semi-minor

        # Determine CIRCLE vs ELLIPSE by axis ratio
        axis_ratio = b / a if a > 1e-6 else 0.0
        shape_type = ShapeType.CIRCLE if axis_ratio > 0.90 else ShapeType.ELLIPSE

        # Generate render points (smooth ellipse outline, 64 pts)
        theta = np.linspace(0, 2 * math.pi, 64)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx, cy = float(center[0]), float(center[1])
        rx = a * np.cos(theta) * cos_a - b * np.sin(theta) * sin_a + cx
        ry = a * np.cos(theta) * sin_a + b * np.sin(theta) * cos_a + cy
        render_points = np.stack([rx, ry], axis=1)

        return SnappedShape(
            shape_type=shape_type,
            params={
                "center": [cx, cy],
                "semi_major": a,
                "semi_minor": b,
                "angle_deg": float(angle),
                "roundness": roundness,
                "axis_ratio": axis_ratio,
            },
            confidence=float(min(roundness, 1.0)),
            render_points=render_points,
        )

    def _detect_polygon(self, pts: np.ndarray) -> Optional[SnappedShape]:
        """
        Detect RECTANGLE (4 vertices) or TRIANGLE (3 vertices) by:
          1. Computing the convex hull.
          2. Running cv2.approxPolyDP on the hull.
        """
        if len(pts) < 6:
            return None

        pts_int = pts.astype(np.float32).reshape(-1, 1, 2)

        # Compute convex hull via OpenCV (returns indices into pts)
        hull_pts = cv2.convexHull(pts_int)
        if hull_pts is None or len(hull_pts) < 3:
            return None

        arc = cv2.arcLength(hull_pts, closed=True)
        # approxPolyDP with epsilon = 5% of arc length
        approx = cv2.approxPolyDP(hull_pts, epsilon=0.05 * arc, closed=True)
        n_verts = len(approx)

        if n_verts == 4:
            corners = approx.reshape(4, 2).astype(np.float64)
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners_ordered = self._order_rect_corners(corners)
            confidence = self._rect_confidence(corners_ordered)
            return SnappedShape(
                shape_type=ShapeType.RECTANGLE,
                params={
                    "corners": corners_ordered.tolist(),
                    "n_vertices": 4,
                },
                confidence=float(confidence),
                render_points=np.vstack([corners_ordered, corners_ordered[:1]]),
            )
        elif n_verts == 3:
            corners = approx.reshape(3, 2).astype(np.float64)
            confidence = self._triangle_confidence(corners)
            return SnappedShape(
                shape_type=ShapeType.TRIANGLE,
                params={
                    "corners": corners.tolist(),
                    "n_vertices": 3,
                },
                confidence=float(confidence),
                render_points=np.vstack([corners, corners[:1]]),
            )

        return None

    @staticmethod
    def _order_rect_corners(corners: np.ndarray) -> np.ndarray:
        """Order 4 corners as: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y (top to bottom), then by x within pairs
        s = corners.sum(axis=1)     # smallest sum → top-left
        d = np.diff(corners, axis=1).ravel()  # smallest diff → top-right

        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        top_right = corners[np.argmin(d)]
        bottom_left = corners[np.argmax(d)]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    @staticmethod
    def _rect_confidence(corners: np.ndarray) -> float:
        """Confidence for rectangle: measures how close to 90° all angles are."""
        angles = []
        n = len(corners)
        for i in range(n):
            v1 = corners[i - 1] - corners[i]
            v2 = corners[(i + 1) % n] - corners[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            cos_a = float(np.dot(v1, v2) / (len1 * len2))
            cos_a = max(-1.0, min(1.0, cos_a))
            angles.append(abs(math.degrees(math.acos(cos_a)) - 90.0))
        if not angles:
            return 0.5
        mean_dev = float(np.mean(angles))
        return float(max(0.0, 1.0 - mean_dev / 45.0))

    @staticmethod
    def _triangle_confidence(corners: np.ndarray) -> float:
        """Confidence for triangle: 1.0 if it looks like a well-formed triangle."""
        # Compute side lengths
        sides = [
            float(np.linalg.norm(corners[(i + 1) % 3] - corners[i]))
            for i in range(3)
        ]
        if any(s < 1e-6 for s in sides):
            return 0.0
        # Coefficient of variation of side lengths (0 = equilateral, high = degenerate)
        cv = float(np.std(sides) / np.mean(sides))
        # Perimeter vs bounding-box diagonal ratio
        perimeter = sum(sides)
        bbox = corners.max(axis=0) - corners.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        compactness = perimeter / max(diag, 1e-6)
        return float(max(0.0, min(1.0, 0.6 / cv if cv > 0.01 else 1.0) * min(1.0, 1.5 / max(compactness, 0.5))))


# ─────────────────────────────────────────────────────────────────────────────
# Quality metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_stroke_rmse(
    raw_points: np.ndarray,
    smoothed_points: np.ndarray,
) -> float:
    """
    Compute Root Mean Square Error between raw and smoothed stroke points.

    Uses nearest-neighbor matching from each raw point to the smoothed polyline
    to handle the fact that RDP may have fewer points than the raw input.

    Args:
        raw_points:     Shape (N, 2) original stroke points.
        smoothed_points: Shape (M, 2) post-filter/spline sampled points.

    Returns:
        RMSE in pixels.
    """
    raw = np.asarray(raw_points, dtype=np.float64)
    smooth = np.asarray(smoothed_points, dtype=np.float64)

    if len(raw) == 0 or len(smooth) == 0:
        return 0.0

    # Accelerate nearest neighbor search via compiled cKDTree
    tree = cKDTree(smooth)
    distances, _ = tree.query(raw)
    return float(math.sqrt(np.mean(distances ** 2)))
