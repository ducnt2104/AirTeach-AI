//! Pure Rust Vector Math & Smoothing Engine:
//! - 1€ Filter (Casiez et al.) for real-time low-latency jitter suppression
//! - Ramer-Douglas-Peucker (RDP) polyline simplification
//! - C1-Continuous Cubic Bézier Spline interpolation
//! - Geometric Shape Snapper (Line, Circle, Rectangle, Triangle) using Convex Hull & Circularity Index
//!
//! Designed with zero-allocation buffer reuse for the inner 60-120 FPS rendering loop.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dist(self, other: Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    #[inline]
    pub fn dist_sq(self, other: Self) -> f32 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// First-order low-pass filter used inside the 1€ Filter.
#[derive(Debug, Clone, Copy)]
struct LowPassFilter {
    y: f32,
    s: f32,
    initialized: bool,
}

impl LowPassFilter {
    #[inline]
    const fn new() -> Self {
        Self {
            y: 0.0,
            s: 0.0,
            initialized: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.initialized = false;
        self.y = 0.0;
        self.s = 0.0;
    }

    #[inline]
    fn filter(&mut self, value: f32, alpha: f32) -> f32 {
        let result = if self.initialized {
            alpha * value + (1.0 - alpha) * self.s
        } else {
            self.initialized = true;
            value
        };
        self.y = value;
        self.s = result;
        result
    }

    #[inline]
    fn last_raw(&self) -> f32 {
        self.y
    }

    #[inline]
    fn has_last_raw(&self) -> bool {
        self.initialized
    }
}

/// 1D 1€ Filter for adaptive low-latency signal smoothing.
#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilter {
    pub min_cutoff: f32,
    pub beta: f32,
    pub d_cutoff: f32,
    x_filter: LowPassFilter,
    dx_filter: LowPassFilter,
    last_time: Option<f64>,
}

impl OneEuroFilter {
    pub const fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_filter: LowPassFilter::new(),
            dx_filter: LowPassFilter::new(),
            last_time: None,
        }
    }

    #[inline]
    fn alpha(dt: f32, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * PI * cutoff.max(1e-4));
        1.0 / (1.0 + tau / dt.max(1e-5))
    }

    pub fn reset(&mut self) {
        self.x_filter.reset();
        self.dx_filter.reset();
        self.last_time = None;
    }

    /// Filter a 1D sample at timestamp `t_sec` (in seconds).
    #[inline]
    pub fn filter(&mut self, value: f32, t_sec: f64) -> f32 {
        let dt = match self.last_time {
            Some(prev) if t_sec > prev => (t_sec - prev) as f32,
            _ => 1.0 / 60.0,
        };
        self.last_time = Some(t_sec);

        let dvalue = if self.x_filter.has_last_raw() {
            (value - self.x_filter.last_raw()) / dt
        } else {
            0.0
        };

        let edvalue = self
            .dx_filter
            .filter(dvalue, Self::alpha(dt, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * edvalue.abs();
        self.x_filter.filter(value, Self::alpha(dt, cutoff))
    }
}

/// 2D Adaptive Exponential Moving Average (EMA) filter:
/// - Slow hand movement (writing small letters/digits): alpha = 0.25 (maximum anti-jitter)
/// - Fast hand movement (long strokes / cursor travel): alpha scales up to 0.65 (zero lag)
#[derive(Debug, Clone, Copy)]
pub struct EmaFilter2D {
    pub alpha_min: f32,
    pub alpha_max: f32,
    prev: Option<Point2D>,
}

impl EmaFilter2D {
    pub const fn new(alpha_min: f32) -> Self {
        Self {
            alpha_min,
            alpha_max: 0.65,
            prev: None,
        }
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }

    #[inline]
    pub fn filter(&mut self, pt: Point2D) -> Point2D {
        let next = match self.prev {
            Some(p) => {
                let speed = pt.dist(p);
                // Normalize speed across [2px .. 28px] per frame to interpolate alpha in [0.25 .. 0.65]
                let t = ((speed - 2.0) / 26.0).clamp(0.0, 1.0);
                let a = (self.alpha_min + t * (self.alpha_max - self.alpha_min)).clamp(0.15, 0.85);
                Point2D {
                    x: a * pt.x + (1.0 - a) * p.x,
                    y: a * pt.y + (1.0 - a) * p.y,
                }
            }
            None => pt,
        };
        self.prev = Some(next);
        next
    }
}

/// 2D 1€ Filter for fingertip coordinates `(x, y)`.
#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilter2D {
    fx: OneEuroFilter,
    fy: OneEuroFilter,
}

impl OneEuroFilter2D {
    pub const fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            fx: OneEuroFilter::new(min_cutoff, beta, d_cutoff),
            fy: OneEuroFilter::new(min_cutoff, beta, d_cutoff),
        }
    }

    pub fn set_params(&mut self, min_cutoff: f32, beta: f32, d_cutoff: f32) {
        self.fx.min_cutoff = min_cutoff;
        self.fx.beta = beta;
        self.fx.d_cutoff = d_cutoff;
        self.fy.min_cutoff = min_cutoff;
        self.fy.beta = beta;
        self.fy.d_cutoff = d_cutoff;
    }

    pub fn reset(&mut self) {
        self.fx.reset();
        self.fy.reset();
    }

    #[inline]
    pub fn filter(&mut self, pt: Point2D, t_sec: f64) -> Point2D {
        Point2D {
            x: self.fx.filter(pt.x, t_sec),
            y: self.fy.filter(pt.y, t_sec),
        }
    }
}

/// Compute perpendicular distance from `pt` to segment `line_start -> line_end`.
#[inline]
pub fn perpendicular_distance(pt: Point2D, line_start: Point2D, line_end: Point2D) -> f32 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let mag_sq = dx * dx + dy * dy;
    if mag_sq < 1e-8 {
        return pt.dist(line_start);
    }
    let num = (dy * pt.x - dx * pt.y + line_end.x * line_start.y - line_end.y * line_start.x).abs();
    num / mag_sq.sqrt()
}

/// Iterative Ramer-Douglas-Peucker (RDP) simplification using pre-allocated scratch buffers
/// to avoid recursive stack frames or dynamic allocations in hot paths.
pub fn rdp_simplify_into(
    points: &[Point2D],
    epsilon: f32,
    keep_flags: &mut Vec<bool>,
    stack: &mut Vec<(usize, usize)>,
    out: &mut Vec<Point2D>,
) {
    out.clear();
    let n = points.len();
    if n <= 2 {
        out.extend_from_slice(points);
        return;
    }

    keep_flags.clear();
    keep_flags.resize(n, false);
    keep_flags[0] = true;
    keep_flags[n - 1] = true;

    stack.clear();
    stack.push((0, n - 1));

    while let Some((start_idx, end_idx)) = stack.pop() {
        if end_idx <= start_idx + 1 {
            continue;
        }
        let mut dmax = 0.0_f32;
        let mut index = start_idx;
        let start_pt = points[start_idx];
        let end_pt = points[end_idx];

        for i in (start_idx + 1)..end_idx {
            let d = perpendicular_distance(points[i], start_pt, end_pt);
            if d > dmax {
                index = i;
                dmax = d;
            }
        }

        if dmax > epsilon {
            keep_flags[index] = true;
            stack.push((start_idx, index));
            stack.push((index, end_idx));
        }
    }

    for (i, &keep) in keep_flags.iter().enumerate() {
        if keep {
            out.push(points[i]);
        }
    }
}

/// Evaluate a single Cubic Bézier curve at parameter `t` in `[0, 1]`:
/// B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
#[inline]
pub fn evaluate_cubic_bezier(
    p0: Point2D,
    p1: Point2D,
    p2: Point2D,
    p3: Point2D,
    t: f32,
) -> Point2D {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;

    let x = uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x;
    let y = uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y;
    Point2D { x, y }
}

/// Generate a C1-continuous Cubic Bézier spline through `control_points` into `out_vertices`
/// without allocating a new `Vec`.
pub fn generate_cubic_bezier_spline_into(
    control_points: &[Point2D],
    subdivisions_per_segment: usize,
    out_vertices: &mut Vec<Point2D>,
) {
    out_vertices.clear();
    let n = control_points.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out_vertices.push(control_points[0]);
        return;
    }
    if n == 2 {
        out_vertices.push(control_points[0]);
        out_vertices.push(control_points[1]);
        return;
    }

    let steps = subdivisions_per_segment.max(2);
    out_vertices.push(control_points[0]);

    for i in 0..(n - 1) {
        let p_prev = if i == 0 {
            control_points[0]
        } else {
            control_points[i - 1]
        };
        let p_curr = control_points[i];
        let p_next = control_points[i + 1];
        let p_next2 = if i + 2 < n {
            control_points[i + 2]
        } else {
            control_points[n - 1]
        };

        // Catmull-Rom to Cubic Bézier tangents (tension = 1/6 for C1 continuity)
        let b1 = Point2D {
            x: p_curr.x + (p_next.x - p_prev.x) / 6.0,
            y: p_curr.y + (p_next.y - p_prev.y) / 6.0,
        };
        let b2 = Point2D {
            x: p_next.x - (p_next2.x - p_curr.x) / 6.0,
            y: p_next.y - (p_next2.y - p_curr.y) / 6.0,
        };

        for step in 1..=steps {
            let t = step as f32 / steps as f32;
            out_vertices.push(evaluate_cubic_bezier(p_curr, b1, b2, p_next, t));
        }
    }
}

/// Fine-grained quadrilateral subtype from polygon angle & edge invariant solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuadKind {
    Square,
    Rectangle,
    Rhombus,
    Parallelogram,
}

/// Recognized geometric shape when user lifts pen / finalizes stroke.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SnappedShape {
    Freehand,
    Line {
        start: Point2D,
        end: Point2D,
    },
    Arrow {
        start: Point2D,
        end: Point2D,
        head_left: Point2D,
        head_right: Point2D,
    },
    Circle {
        center: Point2D,
        radius: f32,
    },
    Ellipse {
        center: Point2D,
        rx: f32,
        ry: f32,
        angle_rad: f32,
    },
    Rectangle {
        min: Point2D,
        max: Point2D,
    },
    Quad {
        corners: [Point2D; 4],
        kind: QuadKind,
    },
    Triangle {
        vertices: [Point2D; 3],
    },
}

impl SnappedShape {
    pub const fn label_vi(&self) -> &'static str {
        match self {
            Self::Freehand => "Nét mượt tự do",
            Self::Line { .. } => "Đoạn thẳng",
            Self::Arrow { .. } => "Mũi tên",
            Self::Circle { .. } => "Hình tròn",
            Self::Ellipse { .. } => "Hình elip",
            Self::Rectangle { .. } => "Hình chữ nhật",
            Self::Quad {
                kind: QuadKind::Square,
                ..
            } => "Hình vuông",
            Self::Quad {
                kind: QuadKind::Rectangle,
                ..
            } => "Hình chữ nhật",
            Self::Quad {
                kind: QuadKind::Rhombus,
                ..
            } => "Hình thoi",
            Self::Quad {
                kind: QuadKind::Parallelogram,
                ..
            } => "Hình bình hành",
            Self::Triangle { .. } => "Hình tam giác",
        }
    }

    pub const fn is_geometric(&self) -> bool {
        !matches!(self, Self::Freehand)
    }
}

/// 2D cross product of OA and OB vectors.
#[inline]
fn cross_product(o: Point2D, a: Point2D, b: Point2D) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Compute Monotone Chain Convex Hull into `hull_out`.
pub fn convex_hull_into(points: &[Point2D], sorted_buf: &mut Vec<Point2D>, hull_out: &mut Vec<Point2D>) {
    hull_out.clear();
    if points.len() <= 2 {
        hull_out.extend_from_slice(points);
        return;
    }

    sorted_buf.clear();
    sorted_buf.extend_from_slice(points);
    sorted_buf.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Lower hull
    for &p in sorted_buf.iter() {
        while hull_out.len() >= 2
            && cross_product(hull_out[hull_out.len() - 2], hull_out[hull_out.len() - 1], p) <= 0.0
        {
            hull_out.pop();
        }
        hull_out.push(p);
    }

    // Upper hull
    let lower_len = hull_out.len() + 1;
    for &p in sorted_buf.iter().rev() {
        while hull_out.len() >= lower_len
            && cross_product(hull_out[hull_out.len() - 2], hull_out[hull_out.len() - 1], p) <= 0.0
        {
            hull_out.pop();
        }
        hull_out.push(p);
    }
    hull_out.pop();
}

/// Compute polygon area (Shoelace formula) and perimeter.
#[inline]
fn polygon_area_and_perimeter(poly: &[Point2D]) -> (f32, f32) {
    let n = poly.len();
    if n < 3 {
        return (0.0, 0.0);
    }
    let mut area2 = 0.0_f32;
    let mut perim = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
        perim += poly[i].dist(poly[j]);
    }
    (0.5 * area2.abs(), perim)
}

const RESAMPLE_N: usize = 64;
const CONFIDENCE_GATE: f32 = 0.78;
const CLOSED_LOOP_GAP_RATIO: f32 = 0.18;
const PCA_LINE_EIG_RATIO: f32 = 0.035;
const ANGLE_SNAP_TOL_DEG: f32 = 6.0;
const CIRCLE_RADIAL_CV_MAX: f32 = 0.14;

/// Pre-allocated hybrid stroke processor executing EMA + 1€ smoothing, RDP denoising,
/// Arc-Length Resampling (N=64), Geometric + $1/$P Shape Beautification, and Bézier splines.
pub struct StrokeEngine {
    pub ema: EmaFilter2D,
    pub filter: OneEuroFilter2D,
    pub rdp_epsilon: f32,
    pub bezier_subdivisions: usize,
    pub auto_shape_snap: bool,
    pub last_confidence: f32,
    // Pre-allocated scratch buffers
    keep_flags_buf: Vec<bool>,
    rdp_stack_buf: Vec<(usize, usize)>,
    rdp_out_buf: Vec<Point2D>,
    hull_sorted_buf: Vec<Point2D>,
    hull_out_buf: Vec<Point2D>,
    hull_simplified_buf: Vec<Point2D>,
    resampled_buf: Vec<Point2D>,
}

impl StrokeEngine {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32, rdp_epsilon: f32) -> Self {
        Self {
            ema: EmaFilter2D::new(0.25),
            filter: OneEuroFilter2D::new(min_cutoff, beta, d_cutoff),
            rdp_epsilon,
            bezier_subdivisions: 6,
            auto_shape_snap: false,
            last_confidence: 0.0,
            keep_flags_buf: Vec::with_capacity(2048),
            rdp_stack_buf: Vec::with_capacity(256),
            rdp_out_buf: Vec::with_capacity(512),
            hull_sorted_buf: Vec::with_capacity(512),
            hull_out_buf: Vec::with_capacity(256),
            hull_simplified_buf: Vec::with_capacity(64),
            resampled_buf: Vec::with_capacity(128),
        }
    }

    pub fn reset_filter(&mut self) {
        self.ema.reset();
        self.filter.reset();
    }

    #[inline]
    pub fn filter_point(&mut self, raw: Point2D, t_sec: f64) -> Point2D {
        let ema_pt = self.ema.filter(raw);
        self.filter.filter(ema_pt, t_sec)
    }

    /// Finalize stroke cleanly and emit display polyline + confidence score `[0.0, 1.0]`.
    pub fn finalize_stroke_with_confidence(
        &mut self,
        raw_points: &[Point2D],
        out_display_points: &mut Vec<Point2D>,
    ) -> (SnappedShape, f32) {
        let shape = self.finalize_stroke(raw_points, out_display_points);
        (shape, self.last_confidence)
    }

    /// Finalize a completed stroke on pen lift / fist clench:
    /// 1. Run RDP downsampling + Closed-Loop Gap Auto-Closure + N=64 Arc-Length Resample
    /// 2. Hybrid Geometric Invariants (PCA, Kasa Circle, Least-Squares Ellipse, Multi-RDP Poly) + $1/$P Matcher
    /// 3. Render snapped geometry (>= 0.78 confidence) or C1 Cubic Bézier spline into `out_display_points`
    pub fn finalize_stroke(
        &mut self,
        raw_points: &[Point2D],
        out_display_points: &mut Vec<Point2D>,
    ) -> SnappedShape {
        self.last_confidence = 0.0;
        if raw_points.len() < 3 {
            out_display_points.clear();
            out_display_points.extend_from_slice(raw_points);
            return SnappedShape::Freehand;
        }

        rdp_simplify_into(
            raw_points,
            self.rdp_epsilon,
            &mut self.keep_flags_buf,
            &mut self.rdp_stack_buf,
            &mut self.rdp_out_buf,
        );

        if self.auto_shape_snap {
            let (snapped, conf) = self.detect_shape_hybrid(raw_points);
            self.last_confidence = conf;
            match snapped {
                SnappedShape::Line { start, end } => {
                    out_display_points.clear();
                    out_display_points.push(start);
                    out_display_points.push(end);
                    return snapped;
                }
                SnappedShape::Arrow {
                    start,
                    end,
                    head_left,
                    head_right,
                } => {
                    out_display_points.clear();
                    out_display_points.push(start);
                    out_display_points.push(end);
                    out_display_points.push(head_left);
                    out_display_points.push(end);
                    out_display_points.push(head_right);
                    return snapped;
                }
                SnappedShape::Circle { center, radius } => {
                    out_display_points.clear();
                    let segments = 72;
                    for i in 0..=segments {
                        let theta = (i as f32 / segments as f32) * 2.0 * PI;
                        out_display_points.push(Point2D::new(
                            center.x + radius * theta.cos(),
                            center.y + radius * theta.sin(),
                        ));
                    }
                    return snapped;
                }
                SnappedShape::Ellipse {
                    center,
                    rx,
                    ry,
                    angle_rad,
                } => {
                    out_display_points.clear();
                    let segments = 72;
                    let ca = angle_rad.cos();
                    let sa = angle_rad.sin();
                    for i in 0..=segments {
                        let theta = (i as f32 / segments as f32) * 2.0 * PI;
                        let lx = rx * theta.cos();
                        let ly = ry * theta.sin();
                        out_display_points.push(Point2D::new(
                            center.x + lx * ca - ly * sa,
                            center.y + lx * sa + ly * ca,
                        ));
                    }
                    return snapped;
                }
                SnappedShape::Rectangle { min, max } => {
                    out_display_points.clear();
                    out_display_points.push(Point2D::new(min.x, min.y));
                    out_display_points.push(Point2D::new(max.x, min.y));
                    out_display_points.push(Point2D::new(max.x, max.y));
                    out_display_points.push(Point2D::new(min.x, max.y));
                    out_display_points.push(Point2D::new(min.x, min.y));
                    return snapped;
                }
                SnappedShape::Quad { corners, .. } => {
                    out_display_points.clear();
                    for c in corners {
                        out_display_points.push(c);
                    }
                    out_display_points.push(corners[0]);
                    return snapped;
                }
                SnappedShape::Triangle { vertices } => {
                    out_display_points.clear();
                    out_display_points.push(vertices[0]);
                    out_display_points.push(vertices[1]);
                    out_display_points.push(vertices[2]);
                    out_display_points.push(vertices[0]);
                    return snapped;
                }
                SnappedShape::Freehand => {}
            }
        }

        generate_cubic_bezier_spline_into(
            &self.rdp_out_buf,
            self.bezier_subdivisions,
            out_display_points,
        );
        SnappedShape::Freehand
    }

    fn detect_shape_hybrid(&mut self, raw_points: &[Point2D]) -> (SnappedShape, f32) {
        let clean = deduplicate_points(raw_points, 1.0);
        if clean.len() < 3 {
            return (SnappedShape::Freehand, 0.0);
        }

        let (min_x, min_y, max_x, max_y, diag) = bbox_and_diag(&clean);
        let total_len = polyline_length(&clean);
        if total_len < 16.0 || diag < 14.0 {
            return (SnappedShape::Freehand, 0.0);
        }

        let eps = (0.014 * diag).max(2.0);
        rdp_simplify_into(
            &clean,
            eps,
            &mut self.keep_flags_buf,
            &mut self.rdp_stack_buf,
            &mut self.rdp_out_buf,
        );

        let first = clean[0];
        let last = clean[clean.len() - 1];
        let gap = first.dist(last);
        let is_closed = clean.len() >= 6
            && (gap < CLOSED_LOOP_GAP_RATIO * total_len || gap < CLOSED_LOOP_GAP_RATIO * diag);

        let mut work = clean.clone();
        if is_closed && gap > 1.0 {
            work.push(first);
        }
        resample_arc_length_into(&work, RESAMPLE_N, &mut self.resampled_buf);

        if !is_closed {
            if let Some((arrow, conf)) = try_detect_arrow(&self.resampled_buf, diag) {
                if conf >= CONFIDENCE_GATE {
                    return (arrow, conf);
                }
            }

            let (centroid, l1, l2, mut uvec) = pca_2d(&self.resampled_buf);
            let eig_ratio = l2 / l1.max(1e-9);
            if eig_ratio < PCA_LINE_EIG_RATIO && self.rdp_out_buf.len() <= 6 {
                let mut t_min = f32::MAX;
                let mut t_max = f32::MIN;
                for &p in &self.resampled_buf {
                    let proj = (p.x - centroid.x) * uvec.x + (p.y - centroid.y) * uvec.y;
                    t_min = t_min.min(proj);
                    t_max = t_max.max(proj);
                }
                let p0_proj = (first.x - centroid.x) * uvec.x + (first.y - centroid.y) * uvec.y;
                let p1_proj = (last.x - centroid.x) * uvec.x + (last.y - centroid.y) * uvec.y;
                if p1_proj < p0_proj {
                    uvec = Point2D::new(-uvec.x, -uvec.y);
                    std::mem::swap(&mut t_min, &mut t_max);
                    t_min = -t_min;
                    t_max = -t_max;
                }
                let line_len = (t_max - t_min).abs();
                let raw_ang_deg = uvec.y.atan2(uvec.x).to_degrees();
                let snap_deg = snap_angle_deg(raw_ang_deg, ANGLE_SNAP_TOL_DEG);
                let rad = snap_deg.to_radians();
                let mid_proj = 0.5 * (t_min + t_max);
                let orig_ang = uvec.y.atan2(uvec.x);
                let mid_pt = Point2D::new(
                    centroid.x + mid_proj * orig_ang.cos(),
                    centroid.y + mid_proj * orig_ang.sin(),
                );
                let ux = rad.cos();
                let uy = rad.sin();
                let p_start = Point2D::new(mid_pt.x - 0.5 * line_len * ux, mid_pt.y - 0.5 * line_len * uy);
                let p_end = Point2D::new(mid_pt.x + 0.5 * line_len * ux, mid_pt.y + 0.5 * line_len * uy);
                let conf = (1.0 - (eig_ratio / PCA_LINE_EIG_RATIO) * 0.18).clamp(0.80, 0.99);
                return (
                    SnappedShape::Line {
                        start: p_start,
                        end: p_end,
                    },
                    conf,
                );
            }
            return (SnappedShape::Freehand, (0.70 - eig_ratio).max(0.0));
        }

        let (contour_area, contour_peri) = polygon_area_and_perimeter(&self.resampled_buf);
        if contour_peri < 1e-3 || contour_area < 64.0 {
            return (SnappedShape::Freehand, 0.0);
        }
        let circularity = (4.0 * PI * contour_area) / (contour_peri * contour_peri);
        let tmpl_scores = score_closed_templates(&self.resampled_buf);

        let mut candidates: Vec<(SnappedShape, f32)> = Vec::new();

        if let Some((cx, cy, r, radial_cv)) = fit_circle_kasa(&self.resampled_buf) {
            if radial_cv < CIRCLE_RADIAL_CV_MAX && circularity >= 0.76 {
                let geom = (1.0 - (radial_cv / CIRCLE_RADIAL_CV_MAX) * 0.24).clamp(0.0, 1.0);
                let conf = 0.68 * geom + 0.32 * tmpl_scores.circle;
                candidates.push((
                    SnappedShape::Circle {
                        center: Point2D::new(cx, cy),
                        radius: r,
                    },
                    conf,
                ));
            }
        }

        let poly_opt = self.extract_stable_polygon(diag, contour_peri);
        let poly_rms_norm = if let Some(ref poly) = poly_opt {
            let mut closed_poly = poly.clone();
            closed_poly.push(poly[0]);
            point_to_polyline_rms(&self.resampled_buf, &closed_poly) / diag.max(1.0)
        } else {
            f32::MAX
        };

        convex_hull_into(
            &self.resampled_buf,
            &mut self.hull_sorted_buf,
            &mut self.hull_out_buf,
        );
        let min_rect = min_area_rotated_rect(&self.hull_out_buf);
        let rectangularity = contour_area / (min_rect.w * min_rect.h).max(1.0);

        if let Some((cx, cy, rx, ry, angle_rad, ell_rms)) = fit_ellipse_pca(&self.resampled_buf) {
            let aspect = (rx / ry.max(1e-3)).max(ry / rx.max(1e-3));
            let is_boxy_quad = poly_opt
                .as_ref()
                .map(|p| p.len() == 4 && poly_rms_norm < 0.036 && rectangularity > 0.87)
                .unwrap_or(false);
            if aspect >= 1.15 && ell_rms < 0.125 && !is_boxy_quad {
                let geom = (1.0 - ell_rms * 2.2).clamp(0.0, 1.0);
                let conf = 0.65 * geom + 0.35 * tmpl_scores.ellipse;
                let snapped_ang = snap_angle_deg(angle_rad.to_degrees(), ANGLE_SNAP_TOL_DEG).to_radians();
                candidates.push((
                    SnappedShape::Ellipse {
                        center: Point2D::new(cx, cy),
                        rx,
                        ry,
                        angle_rad: snapped_ang,
                    },
                    conf,
                ));
            }
        }

        if let Some(poly) = poly_opt {
            if poly_rms_norm < 0.068 {
                let poly_geom = (1.0 - (poly_rms_norm / 0.068) * 0.28).clamp(0.0, 1.0);
                if poly.len() == 3 {
                    let tri = reconstruct_triangle(&poly);
                    let conf = 0.65 * poly_geom + 0.35 * tmpl_scores.triangle;
                    candidates.push((SnappedShape::Triangle { vertices: tri }, conf));
                } else if poly.len() == 4 {
                    let (corners, kind, prior) = reconstruct_quad(&poly, &min_rect);
                    let tmpl_q = match kind {
                        QuadKind::Square => tmpl_scores.square,
                        QuadKind::Rectangle => tmpl_scores.rectangle,
                        QuadKind::Rhombus => tmpl_scores.rhombus,
                        QuadKind::Parallelogram => tmpl_scores.parallelogram,
                    };
                    let conf = 0.45 * poly_geom + 0.25 * prior + 0.30 * tmpl_q;
                    candidates.push((SnappedShape::Quad { corners, kind }, conf));
                }
            }
        }

        let _ = (min_x, min_y, max_x, max_y);
        if let Some((best_shape, best_conf)) = candidates
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if best_conf >= CONFIDENCE_GATE {
                return (best_shape, best_conf);
            }
            return (SnappedShape::Freehand, best_conf);
        }

        (SnappedShape::Freehand, 0.0)
    }

    fn extract_stable_polygon(&mut self, diag: f32, peri: f32) -> Option<Vec<Point2D>> {
        let fracs = [0.025_f32, 0.035, 0.045, 0.055, 0.068, 0.080];
        let mut group3: Vec<Vec<Point2D>> = Vec::new();
        let mut group4: Vec<Vec<Point2D>> = Vec::new();

        for frac in fracs {
            let eps = (frac * peri).max(4.0);
            rdp_simplify_into(
                &self.resampled_buf,
                eps,
                &mut self.keep_flags_buf,
                &mut self.rdp_stack_buf,
                &mut self.hull_simplified_buf,
            );
            if self.hull_simplified_buf.len() >= 2 {
                let last_idx = self.hull_simplified_buf.len() - 1;
                if self.hull_simplified_buf[0].dist(self.hull_simplified_buf[last_idx]) < eps.max(10.0) {
                    self.hull_simplified_buf.pop();
                }
            }
            let pruned = prune_collinear_vertices(&self.hull_simplified_buf, diag);
            if pruned.len() == 3 {
                group3.push(order_ccw(&pruned));
            } else if pruned.len() == 4 {
                group4.push(order_ccw(&pruned));
            }
        }

        let mut best: Option<Vec<Point2D>> = None;
        let mut best_cost = f32::MAX;
        for group in [&group3, &group4] {
            if group.is_empty() {
                continue;
            }
            let rep = &group[group.len() / 2];
            let mut closed = rep.clone();
            closed.push(rep[0]);
            let rms = point_to_polyline_rms(&self.resampled_buf, &closed) / diag.max(1.0);
            let cost = rms - 0.008 * (group.len() as f32);
            if cost < best_cost {
                best_cost = cost;
                best = Some(rep.clone());
            }
        }
        best
    }
}

fn polyline_length(pts: &[Point2D]) -> f32 {
    if pts.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for w in pts.windows(2) {
        sum += w[0].dist(w[1]);
    }
    sum
}

fn bbox_and_diag(pts: &[Point2D]) -> (f32, f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &p in pts {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    (min_x, min_y, max_x, max_y, diag)
}

fn deduplicate_points(pts: &[Point2D], min_step: f32) -> Vec<Point2D> {
    if pts.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pts.len());
    out.push(pts[0]);
    let min_sq = min_step * min_step;
    for &p in &pts[1..] {
        if out.last().unwrap().dist_sq(p) >= min_sq {
            out.push(p);
        }
    }
    out
}

pub fn resample_arc_length_into(pts: &[Point2D], n: usize, out: &mut Vec<Point2D>) {
    out.clear();
    if pts.is_empty() || n == 0 {
        return;
    }
    if pts.len() == 1 || n == 1 {
        for _ in 0..n {
            out.push(pts[0]);
        }
        return;
    }
    let total_len = polyline_length(pts);
    if total_len < 1e-5 {
        for _ in 0..n {
            out.push(pts[0]);
        }
        return;
    }
    let step = total_len / (n - 1) as f32;
    let mut cum = 0.0_f32;
    out.push(pts[0]);
    let mut prev = pts[0];
    let mut i = 1usize;
    while i < pts.len() && out.len() < n {
        let cur = pts[i];
        let seg = prev.dist(cur);
        if cum + seg >= step {
            let t = (step - cum) / seg.max(1e-6);
            let q = Point2D::new(prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y));
            out.push(q);
            prev = q;
            cum = 0.0;
        } else {
            cum += seg;
            prev = cur;
            i += 1;
        }
    }
    while out.len() < n {
        out.push(*pts.last().unwrap());
    }
}

fn pca_2d(pts: &[Point2D]) -> (Point2D, f32, f32, Point2D) {
    let n = pts.len().max(1) as f32;
    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    for &p in pts {
        sx += p.x;
        sy += p.y;
    }
    let cx = sx / n;
    let cy = sy / n;
    let denom = (pts.len().saturating_sub(1)).max(1) as f32;
    let (mut cxx, mut cyy, mut cxy) = (0.0_f32, 0.0_f32, 0.0_f32);
    for &p in pts {
        let dx = p.x - cx;
        let dy = p.y - cy;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= denom;
    cyy /= denom;
    cxy /= denom;
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (0.25 * trace * trace - det).max(0.0).sqrt();
    let l1 = (0.5 * trace + disc).max(0.0);
    let l2 = (0.5 * trace - disc).max(0.0);
    let (mut vx, mut vy) = if cxy.abs() > 1e-8 {
        (l1 - cyy, cxy)
    } else if cxx >= cyy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let vnorm = (vx * vx + vy * vy).sqrt();
    if vnorm > 1e-8 {
        vx /= vnorm;
        vy /= vnorm;
    } else {
        vx = 1.0;
        vy = 0.0;
    }
    (Point2D::new(cx, cy), l1, l2, Point2D::new(vx, vy))
}

#[inline]
fn snap_angle_deg(angle_deg: f32, tol_deg: f32) -> f32 {
    let nearest = (angle_deg / 45.0).round() * 45.0;
    if (angle_deg - nearest).abs() <= tol_deg {
        nearest
    } else {
        angle_deg
    }
}

fn try_detect_arrow(pts: &[Point2D], diag: f32) -> Option<(SnappedShape, f32)> {
    let n = pts.len();
    if n < 8 || diag < 24.0 {
        return None;
    }
    let split = ((n as f32 * 0.65) as usize).max(5).min(n - 2);
    let (_, s_l1, s_l2, mut s_vec) = pca_2d(&pts[..split]);
    let shaft_ratio = s_l2 / s_l1.max(1e-9);
    if shaft_ratio >= 0.045 {
        return None;
    }
    let p0 = pts[0];
    if (pts[split - 1].x - p0.x) * s_vec.x + (pts[split - 1].y - p0.y) * s_vec.y < 0.0 {
        s_vec = Point2D::new(-s_vec.x, -s_vec.y);
    }
    let mut best_idx = 0usize;
    let mut best_proj = f32::MIN;
    let mut min_proj = f32::MAX;
    for (i, &p) in pts.iter().enumerate() {
        let pr = (p.x - p0.x) * s_vec.x + (p.y - p0.y) * s_vec.y;
        if pr > best_proj {
            best_proj = pr;
            best_idx = i;
        }
        if i <= split {
            min_proj = min_proj.min(pr);
        }
    }
    if best_idx < 3 || best_idx >= n - 2 {
        return None;
    }
    let shaft_len = best_proj - min_proj;
    if shaft_len < 22.0 {
        return None;
    }
    let tip = pts[best_idx];
    let mut max_perp = 0.0_f32;
    let mut min_tail_proj = f32::MAX;
    for &p in &pts[best_idx..] {
        let perp = ((p.x - tip.x) * (-s_vec.y) + (p.y - tip.y) * s_vec.x).abs();
        max_perp = max_perp.max(perp);
        let pr = (p.x - p0.x) * s_vec.x + (p.y - p0.y) * s_vec.y;
        min_tail_proj = min_tail_proj.min(pr);
    }
    if max_perp < (0.06 * shaft_len).max(5.5) || max_perp > 0.45 * shaft_len {
        return None;
    }
    let backtrack = best_proj - min_tail_proj;
    if backtrack < (0.05 * shaft_len).max(5.0) {
        return None;
    }
    let raw_deg = (tip.y - p0.y).atan2(tip.x - p0.x).to_degrees();
    let rad = snap_angle_deg(raw_deg, ANGLE_SNAP_TOL_DEG).to_radians();
    let len = p0.dist(tip);
    let mid = Point2D::new(0.5 * (p0.x + tip.x), 0.5 * (p0.y + tip.y));
    let (ux, uy) = (rad.cos(), rad.sin());
    let start = Point2D::new(mid.x - 0.5 * len * ux, mid.y - 0.5 * len * uy);
    let end = Point2D::new(mid.x + 0.5 * len * ux, mid.y + 0.5 * len * uy);

    let wing = (0.14 * len).clamp(14.0, 34.0);
    let barb = 28.0_f32.to_radians();
    let (ca, sa) = (barb.cos(), barb.sin());
    let (bx, by) = (-ux, -uy);
    let head_left = Point2D::new(end.x + wing * (bx * ca - by * sa), end.y + wing * (bx * sa + by * ca));
    let head_right = Point2D::new(end.x + wing * (bx * ca + by * sa), end.y + wing * (-bx * sa + by * ca));
    let conf = (0.96 - shaft_ratio * 2.5).clamp(0.80, 0.97);
    Some((
        SnappedShape::Arrow {
            start,
            end,
            head_left,
            head_right,
        },
        conf,
    ))
}

fn fit_circle_kasa(pts: &[Point2D]) -> Option<(f32, f32, f32, f32)> {
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [0.0_f64; 3];
    for &p in pts {
        let px = p.x as f64;
        let py = p.y as f64;
        let row = [2.0 * px, 2.0 * py, 1.0];
        let bi = px * px + py * py;
        for r in 0..3 {
            atb[r] += row[r] * bi;
            for c in 0..3 {
                ata[r][c] += row[r] * row[c];
            }
        }
    }
    let sol = solve_3x3(ata, atb)?;
    let (cx, cy, c_val) = (sol[0], sol[1], sol[2]);
    let r_sq = c_val + cx * cx + cy * cy;
    if r_sq <= 1.0 {
        return None;
    }
    let r = r_sq.sqrt() as f32;
    let cx = cx as f32;
    let cy = cy as f32;
    let center = Point2D::new(cx, cy);
    let mut sum_d = 0.0_f32;
    for &p in pts {
        sum_d += p.dist(center);
    }
    let mean_d = (sum_d / pts.len().max(1) as f32).max(1e-5);
    let mut var_d = 0.0_f32;
    for &p in pts {
        let diff = p.dist(center) - mean_d;
        var_d += diff * diff;
    }
    let cv = (var_d / pts.len().max(1) as f32).sqrt() / mean_d;
    Some((cx, cy, r, cv))
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        for row in (col + 1)..3 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / div;
            b[row] -= factor * b[col];
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    let mut x = [0.0_f64; 3];
    for row in (0..3).rev() {
        let mut s = b[row];
        for k in (row + 1)..3 {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

fn fit_ellipse_pca(pts: &[Point2D]) -> Option<(f32, f32, f32, f32, f32, f32)> {
    if pts.len() < 5 {
        return None;
    }
    let (centroid, l1, l2, uvec) = pca_2d(pts);
    let _ = (l1, l2);
    let angle_rad = uvec.y.atan2(uvec.x);
    let ca = (-angle_rad).cos();
    let sa = (-angle_rad).sin();

    let mut u2_sum = 0.0_f32;
    let mut v2_sum = 0.0_f32;
    for &p in pts {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let u = dx * ca - dy * sa;
        let v = dx * sa + dy * ca;
        u2_sum += u * u;
        v2_sum += v * v;
    }
    let n = pts.len() as f32;
    let rx = (2.0 * u2_sum / n).sqrt().max(3.0);
    let ry = (2.0 * v2_sum / n).sqrt().max(3.0);

    let mut err_sq = 0.0_f32;
    for &p in pts {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let u = (dx * ca - dy * sa) / rx;
        let v = (dx * sa + dy * ca) / ry;
        let nr = (u * u + v * v).sqrt();
        err_sq += (nr - 1.0) * (nr - 1.0);
    }
    let rms = (err_sq / n).sqrt();
    Some((centroid.x, centroid.y, rx, ry, angle_rad, rms))
}

#[derive(Clone, Copy)]
struct MinAreaRect {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle_rad: f32,
}

fn min_area_rotated_rect(hull: &[Point2D]) -> MinAreaRect {
    if hull.len() < 3 {
        return MinAreaRect {
            cx: 0.0,
            cy: 0.0,
            w: 1.0,
            h: 1.0,
            angle_rad: 0.0,
        };
    }
    let mut best_area = f32::MAX;
    let mut best = MinAreaRect {
        cx: 0.0,
        cy: 0.0,
        w: 1.0,
        h: 1.0,
        angle_rad: 0.0,
    };
    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let ang = (b.y - a.y).atan2(b.x - a.x);
        let ca = (-ang).cos();
        let sa = (-ang).sin();
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &p in hull {
            let u = p.x * ca - p.y * sa;
            let v = p.x * sa + p.y * ca;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = (max_u - min_u).max(1.0);
        let h = (max_v - min_v).max(1.0);
        let area = w * h;
        if area < best_area {
            best_area = area;
            let cu = 0.5 * (min_u + max_u);
            let cv = 0.5 * (min_v + max_v);
            let rca = ang.cos();
            let rsa = ang.sin();
            best = MinAreaRect {
                cx: cu * rca - cv * rsa,
                cy: cu * rsa + cv * rca,
                w,
                h,
                angle_rad: ang,
            };
        }
    }
    best
}

fn interior_angles_deg(verts: &[Point2D]) -> Vec<f32> {
    let n = verts.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = verts[(i + n - 1) % n];
        let curr = verts[i];
        let next = verts[(i + 1) % n];
        let ux = prev.x - curr.x;
        let uy = prev.y - curr.y;
        let vx = next.x - curr.x;
        let vy = next.y - curr.y;
        let nu = (ux * ux + uy * uy).sqrt();
        let nv = (vx * vx + vy * vy).sqrt();
        if nu < 1e-5 || nv < 1e-5 {
            out.push(180.0);
        } else {
            let c = ((ux * vx + uy * vy) / (nu * nv)).clamp(-1.0, 1.0);
            out.push(c.acos().to_degrees());
        }
    }
    out
}

fn order_ccw(verts: &[Point2D]) -> Vec<Point2D> {
    let n = verts.len().max(1) as f32;
    let cx = verts.iter().map(|p| p.x).sum::<f32>() / n;
    let cy = verts.iter().map(|p| p.y).sum::<f32>() / n;
    let mut out = verts.to_vec();
    out.sort_by(|a, b| {
        let aa = (a.y - cy).atan2(a.x - cx);
        let ab = (b.y - cy).atan2(b.x - cx);
        aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn prune_collinear_vertices(verts: &[Point2D], diag: f32) -> Vec<Point2D> {
    let mut pts = verts.to_vec();
    let mut changed = true;
    while changed && pts.len() > 3 {
        changed = false;
        for i in 0..pts.len() {
            let j = (i + 1) % pts.len();
            if pts[i].dist(pts[j]) < (0.08 * diag).max(8.0) {
                pts[i] = Point2D::new(0.5 * (pts[i].x + pts[j].x), 0.5 * (pts[i].y + pts[j].y));
                pts.remove(j);
                changed = true;
                break;
            }
        }
        if changed {
            continue;
        }
        let angles = interior_angles_deg(&pts);
        for (i, &ang) in angles.iter().enumerate() {
            if ang > 162.0 {
                pts.remove(i);
                changed = true;
                break;
            }
        }
    }
    pts
}

fn point_to_polyline_rms(pts: &[Point2D], poly_closed: &[Point2D]) -> f32 {
    if pts.is_empty() || poly_closed.len() < 2 {
        return f32::MAX;
    }
    let mut sum_sq = 0.0_f32;
    for &p in pts {
        let mut best = f32::MAX;
        for seg in poly_closed.windows(2) {
            best = best.min(perpendicular_distance(p, seg[0], seg[1]));
        }
        sum_sq += best * best;
    }
    (sum_sq / pts.len() as f32).sqrt()
}

fn reconstruct_triangle(verts: &[Point2D]) -> [Point2D; 3] {
    let pts = order_ccw(verts);
    let sides = [
        pts[0].dist(pts[1]),
        pts[1].dist(pts[2]),
        pts[2].dist(pts[0]),
    ];
    let angles = interior_angles_deg(&pts);
    let mean_s = (sides[0] + sides[1] + sides[2]) / 3.0;
    let side_cv = ((sides.iter().map(|s| (s - mean_s).powi(2)).sum::<f32>()) / 3.0).sqrt()
        / mean_s.max(1e-5);

    if angles.iter().all(|&a| (a - 60.0).abs() <= 12.0) && side_cv < 0.18 {
        let cx = (pts[0].x + pts[1].x + pts[2].x) / 3.0;
        let cy = (pts[0].y + pts[1].y + pts[2].y) / 3.0;
        let c = Point2D::new(cx, cy);
        let r = (pts[0].dist(c) + pts[1].dist(c) + pts[2].dist(c)) / 3.0;
        let top = *pts.iter().min_by(|a, b| a.y.partial_cmp(&b.y).unwrap()).unwrap();
        let base = snap_angle_deg((top.y - cy).atan2(top.x - cx).to_degrees(), 10.0);
        return [
            Point2D::new(cx + r * base.to_radians().cos(), cy + r * base.to_radians().sin()),
            Point2D::new(
                cx + r * (base + 120.0).to_radians().cos(),
                cy + r * (base + 120.0).to_radians().sin(),
            ),
            Point2D::new(
                cx + r * (base + 240.0).to_radians().cos(),
                cy + r * (base + 240.0).to_radians().sin(),
            ),
        ];
    }

    if let Some((idx, _)) = angles.iter().enumerate().find(|(_, &a)| (80.0..=100.0).contains(&a)) {
        let o = pts[idx];
        let p_prev = pts[(idx + 2) % 3];
        let p_next = pts[(idx + 1) % 3];
        let l1 = o.dist(p_prev);
        let l2 = o.dist(p_next);
        let r1 = snap_angle_deg((p_prev.y - o.y).atan2(p_prev.x - o.x).to_degrees(), ANGLE_SNAP_TOL_DEG)
            .to_radians();
        let cross = (p_prev.x - o.x) * (p_next.y - o.y) - (p_prev.y - o.y) * (p_next.x - o.x);
        let r2 = r1 + if cross >= 0.0 { 0.5 * PI } else { -0.5 * PI };
        return [
            o,
            Point2D::new(o.x + l1 * r1.cos(), o.y + l1 * r1.sin()),
            Point2D::new(o.x + l2 * r2.cos(), o.y + l2 * r2.sin()),
        ];
    }

    [pts[0], pts[1], pts[2]]
}

fn box_corners(cx: f32, cy: f32, w: f32, h: f32, angle_rad: f32) -> [Point2D; 4] {
    let hw = 0.5 * w;
    let hh = 0.5 * h;
    let ca = angle_rad.cos();
    let sa = angle_rad.sin();
    let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
    [
        Point2D::new(cx + local[0].0 * ca - local[0].1 * sa, cy + local[0].0 * sa + local[0].1 * ca),
        Point2D::new(cx + local[1].0 * ca - local[1].1 * sa, cy + local[1].0 * sa + local[1].1 * ca),
        Point2D::new(cx + local[2].0 * ca - local[2].1 * sa, cy + local[2].0 * sa + local[2].1 * ca),
        Point2D::new(cx + local[3].0 * ca - local[3].1 * sa, cy + local[3].0 * sa + local[3].1 * ca),
    ]
}

fn reconstruct_quad(verts: &[Point2D], min_rect: &MinAreaRect) -> ([Point2D; 4], QuadKind, f32) {
    let pts = order_ccw(verts);
    let edges = [
        pts[0].dist(pts[1]),
        pts[1].dist(pts[2]),
        pts[2].dist(pts[3]),
        pts[3].dist(pts[0]),
    ];
    let angles = interior_angles_deg(&pts);
    let min_e = edges.iter().cloned().fold(f32::MAX, f32::min);
    let max_e = edges.iter().cloned().fold(1e-5_f32, f32::max);
    let side_uniformity = min_e / max_e;
    let opp_a = edges[0].min(edges[2]) / edges[0].max(edges[2]).max(1e-5);
    let opp_b = edges[1].min(edges[3]) / edges[1].max(edges[3]).max(1e-5);
    let right_count = angles.iter().filter(|&&a| (78.0..=102.0).contains(&a)).count();

    let snapped_rect_ang = snap_angle_deg(min_rect.angle_rad.to_degrees(), ANGLE_SNAP_TOL_DEG).to_radians();
    if right_count >= 3 {
        let aspect = min_rect.w.min(min_rect.h) / min_rect.w.max(min_rect.h).max(1e-5);
        if aspect >= 0.83 && side_uniformity >= 0.80 {
            let side = 0.5 * (min_rect.w + min_rect.h);
            return (
                box_corners(min_rect.cx, min_rect.cy, side, side, snapped_rect_ang),
                QuadKind::Square,
                0.94,
            );
        }
        return (
            box_corners(min_rect.cx, min_rect.cy, min_rect.w, min_rect.h, snapped_rect_ang),
            QuadKind::Rectangle,
            0.93,
        );
    }

    let cx = pts.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let cy = pts.iter().map(|p| p.y).sum::<f32>() / 4.0;
    let ux = pts[2].x - pts[0].x;
    let uy = pts[2].y - pts[0].y;
    let vx = pts[3].x - pts[1].x;
    let vy = pts[3].y - pts[1].y;
    let du = (ux * ux + uy * uy).sqrt();
    let dv = (vx * vx + vy * vy).sqrt();
    if du > 1e-5 && dv > 1e-5 {
        let diag_cos = ((ux * vx + uy * vy) / (du * dv)).abs();
        if diag_cos < 0.30 && side_uniformity >= 0.78 {
            let ru = snap_angle_deg(uy.atan2(ux).to_degrees(), ANGLE_SNAP_TOL_DEG).to_radians();
            let rv = ru + 0.5 * PI;
            let (hu, hv) = (0.5 * du, 0.5 * dv);
            return (
                [
                    Point2D::new(cx - hu * ru.cos(), cy - hu * ru.sin()),
                    Point2D::new(cx - hv * rv.cos(), cy - hv * rv.sin()),
                    Point2D::new(cx + hu * ru.cos(), cy + hu * ru.sin()),
                    Point2D::new(cx + hv * rv.cos(), cy + hv * rv.sin()),
                ],
                QuadKind::Rhombus,
                0.89,
            );
        }
    }

    if opp_a >= 0.76 && opp_b >= 0.76 {
        let vx1 = 0.5 * ((pts[1].x - pts[0].x) + (pts[2].x - pts[3].x));
        let vy1 = 0.5 * ((pts[1].y - pts[0].y) + (pts[2].y - pts[3].y));
        let vx2 = 0.5 * ((pts[3].x - pts[0].x) + (pts[2].x - pts[1].x));
        let vy2 = 0.5 * ((pts[3].y - pts[0].y) + (pts[2].y - pts[1].y));
        return (
            [
                Point2D::new(cx - 0.5 * vx1 - 0.5 * vx2, cy - 0.5 * vy1 - 0.5 * vy2),
                Point2D::new(cx + 0.5 * vx1 - 0.5 * vx2, cy + 0.5 * vy1 - 0.5 * vy2),
                Point2D::new(cx + 0.5 * vx1 + 0.5 * vx2, cy + 0.5 * vy1 + 0.5 * vy2),
                Point2D::new(cx - 0.5 * vx1 + 0.5 * vx2, cy - 0.5 * vy1 + 0.5 * vy2),
            ],
            QuadKind::Parallelogram,
            0.86,
        );
    }

    ([pts[0], pts[1], pts[2], pts[3]], QuadKind::Rectangle, 0.74)
}

struct ClosedTemplateScores {
    circle: f32,
    ellipse: f32,
    triangle: f32,
    square: f32,
    rectangle: f32,
    rhombus: f32,
    parallelogram: f32,
}

fn normalize_dollar_one(pts: &[Point2D], n: usize, rotate_indicative: bool) -> Vec<Point2D> {
    let mut res = Vec::with_capacity(n);
    resample_arc_length_into(pts, n, &mut res);
    if res.is_empty() {
        return res;
    }
    let nf = res.len() as f32;
    let cx = res.iter().map(|p| p.x).sum::<f32>() / nf;
    let cy = res.iter().map(|p| p.y).sum::<f32>() / nf;
    for p in &mut res {
        p.x -= cx;
        p.y -= cy;
    }
    if rotate_indicative {
        let theta = res[0].y.atan2(res[0].x);
        let (c, s) = ((-theta).cos(), (-theta).sin());
        for p in &mut res {
            let (ox, oy) = (p.x, p.y);
            p.x = ox * c - oy * s;
            p.y = ox * s + oy * c;
        }
    }
    let (min_x, min_y, max_x, max_y, _) = bbox_and_diag(&res);
    let scale = 250.0 / (max_x - min_x).max(max_y - min_y).max(1e-5);
    for p in &mut res {
        p.x *= scale;
        p.y *= scale;
    }
    let mx = res.iter().map(|p| p.x).sum::<f32>() / nf;
    let my = res.iter().map(|p| p.y).sum::<f32>() / nf;
    for p in &mut res {
        p.x -= mx;
        p.y -= my;
    }
    res
}

fn dollar_one_dist_at(pts: &[Point2D], tmpl: &[Point2D], rad: f32) -> f32 {
    let (c, s) = (rad.cos(), rad.sin());
    let mut sum = 0.0_f32;
    for (p, t) in pts.iter().zip(tmpl.iter()) {
        let rx = p.x * c - p.y * s;
        let ry = p.x * s + p.y * c;
        sum += ((rx - t.x).powi(2) + (ry - t.y).powi(2)).sqrt();
    }
    sum / pts.len().max(1) as f32
}

fn dollar_one_best_score(pts: &[Point2D], tmpl: &[Point2D]) -> f32 {
    let phi = 0.5 * (-1.0_f32 + 5.0_f32.sqrt());
    let mut a = (-45.0_f32).to_radians();
    let mut b = 45.0_f32.to_radians();
    let tol = 2.0_f32.to_radians();
    let mut x1 = phi * a + (1.0 - phi) * b;
    let mut f1 = dollar_one_dist_at(pts, tmpl, x1);
    let mut x2 = (1.0 - phi) * a + phi * b;
    let mut f2 = dollar_one_dist_at(pts, tmpl, x2);
    while (b - a).abs() > tol {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = phi * a + (1.0 - phi) * b;
            f1 = dollar_one_dist_at(pts, tmpl, x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = (1.0 - phi) * a + phi * b;
            f2 = dollar_one_dist_at(pts, tmpl, x2);
        }
    }
    let best_d = f1.min(f2);
    let half_diag = 0.5 * (250.0_f32 * 250.0 + 250.0 * 250.0).sqrt();
    (1.0 - best_d / half_diag).clamp(0.0, 1.0)
}

fn point_cloud_chamfer_score(q_pc: &[Point2D], t_pc: &[Point2D]) -> f32 {
    let mut best = f32::MAX;
    for step in 0..24 {
        let ang = (step as f32 * 15.0).to_radians();
        let (c, s) = (ang.cos(), ang.sin());
        let mut d_fwd = 0.0_f32;
        for q in q_pc {
            let rx = q.x * c - q.y * s;
            let ry = q.x * s + q.y * c;
            let mut m = f32::MAX;
            for t in t_pc {
                m = m.min(((rx - t.x).powi(2) + (ry - t.y).powi(2)).sqrt());
            }
            d_fwd += m;
        }
        d_fwd /= q_pc.len().max(1) as f32;
        let mut d_bwd = 0.0_f32;
        for t in t_pc {
            let mut m = f32::MAX;
            for q in q_pc {
                let rx = q.x * c - q.y * s;
                let ry = q.x * s + q.y * c;
                m = m.min(((rx - t.x).powi(2) + (ry - t.y).powi(2)).sqrt());
            }
            d_bwd += m;
        }
        d_bwd /= t_pc.len().max(1) as f32;
        best = best.min(0.5 * (d_fwd + d_bwd));
    }
    (1.0 - best / 65.0).clamp(0.0, 1.0)
}

fn score_against_templates(q_d1: &[Point2D], q_pc: &[Point2D], raw_tmpls: &[Vec<Point2D>]) -> f32 {
    let mut best_d1 = 0.0_f32;
    let mut best_pc = 0.0_f32;
    for raw in raw_tmpls {
        let t_fwd = normalize_dollar_one(raw, RESAMPLE_N, true);
        let mut rev = raw.clone();
        rev.reverse();
        let t_rev = normalize_dollar_one(&rev, RESAMPLE_N, true);
        best_d1 = best_d1
            .max(dollar_one_best_score(q_d1, &t_fwd))
            .max(dollar_one_best_score(q_d1, &t_rev));
        let t_pc = normalize_dollar_one(raw, 28, false);
        best_pc = best_pc.max(point_cloud_chamfer_score(q_pc, &t_pc));
    }
    0.45 * best_d1 + 0.55 * best_pc
}

fn poly_closed_template(verts: &[Point2D]) -> Vec<Point2D> {
    let mut v = verts.to_vec();
    v.push(verts[0]);
    v
}

fn score_closed_templates(closed_pts: &[Point2D]) -> ClosedTemplateScores {
    let q_d1 = normalize_dollar_one(closed_pts, RESAMPLE_N, true);
    let q_pc = normalize_dollar_one(closed_pts, 28, false);

    let circle_t = vec![(0..RESAMPLE_N)
        .map(|i| {
            let a = 2.0 * PI * (i as f32) / (RESAMPLE_N - 1) as f32;
            Point2D::new(a.cos(), a.sin())
        })
        .collect::<Vec<_>>()];
    let ell_t = vec![
        (0..RESAMPLE_N)
            .map(|i| {
                let a = 2.0 * PI * (i as f32) / (RESAMPLE_N - 1) as f32;
                Point2D::new(1.7 * a.cos(), a.sin())
            })
            .collect::<Vec<_>>(),
        (0..RESAMPLE_N)
            .map(|i| {
                let a = 2.0 * PI * (i as f32) / (RESAMPLE_N - 1) as f32;
                Point2D::new(2.2 * a.cos(), a.sin())
            })
            .collect::<Vec<_>>(),
    ];
    let tri_t = vec![
        poly_closed_template(&[
            Point2D::new(0.0, -1.0),
            Point2D::new(0.866, 0.5),
            Point2D::new(-0.866, 0.5),
        ]),
        poly_closed_template(&[
            Point2D::new(-1.0, -1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(-1.0, 1.0),
        ]),
    ];
    let sq_t = vec![poly_closed_template(&[
        Point2D::new(-1.0, -1.0),
        Point2D::new(1.0, -1.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(-1.0, 1.0),
    ])];
    let rect_t = vec![
        poly_closed_template(&[
            Point2D::new(-1.6, -0.8),
            Point2D::new(1.6, -0.8),
            Point2D::new(1.6, 0.8),
            Point2D::new(-1.6, 0.8),
        ]),
        poly_closed_template(&[
            Point2D::new(-0.7, -1.5),
            Point2D::new(0.7, -1.5),
            Point2D::new(0.7, 1.5),
            Point2D::new(-0.7, 1.5),
        ]),
    ];
    let rhom_t = vec![poly_closed_template(&[
        Point2D::new(0.0, -1.3),
        Point2D::new(0.9, 0.0),
        Point2D::new(0.0, 1.3),
        Point2D::new(-0.9, 0.0),
    ])];
    let para_t = vec![poly_closed_template(&[
        Point2D::new(-0.9, -0.7),
        Point2D::new(1.3, -0.7),
        Point2D::new(0.9, 0.7),
        Point2D::new(-1.3, 0.7),
    ])];

    ClosedTemplateScores {
        circle: score_against_templates(&q_d1, &q_pc, &circle_t),
        ellipse: score_against_templates(&q_d1, &q_pc, &ell_t),
        triangle: score_against_templates(&q_d1, &q_pc, &tri_t),
        square: score_against_templates(&q_d1, &q_pc, &sq_t),
        rectangle: score_against_templates(&q_d1, &q_pc, &rect_t),
        rhombus: score_against_templates(&q_d1, &q_pc, &rhom_t),
        parallelogram: score_against_templates(&q_d1, &q_pc, &para_t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_euro_filter_converges() {
        let mut f = OneEuroFilter2D::new(1.2, 0.007, 1.0);
        let p0 = f.filter(Point2D::new(100.0, 200.0), 0.0);
        assert_eq!(p0, Point2D::new(100.0, 200.0));
        let p1 = f.filter(Point2D::new(102.0, 198.0), 0.016);
        assert!(p1.x > 100.0 && p1.x < 102.0);
    }

    #[test]
    fn test_rdp_simplifies_collinear_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.1),
            Point2D::new(20.0, -0.1),
            Point2D::new(30.0, 0.0),
        ];
        let mut keep = Vec::new();
        let mut stack = Vec::new();
        let mut out = Vec::new();
        rdp_simplify_into(&pts, 1.0, &mut keep, &mut stack, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], Point2D::new(0.0, 0.0));
        assert_eq!(out[1], Point2D::new(30.0, 0.0));
    }

    #[test]
    fn test_hybrid_shape_recognition_suite() {
        let mut engine = StrokeEngine::new(1.0, 0.01, 1.0, 2.0);
        engine.auto_shape_snap = true;
        let mut out = Vec::new();

        // 1. PCA Line + 6 deg horizontal snap
        let line_pts: Vec<Point2D> = (0..25)
            .map(|i| Point2D::new(100.0 + i as f32 * 8.0, 200.0 + i as f32 * 0.35))
            .collect();
        let (snapped_line, conf_line) = engine.finalize_stroke_with_confidence(&line_pts, &mut out);
        assert!(matches!(snapped_line, SnappedShape::Line { .. }), "Got {snapped_line:?}");
        assert!(conf_line >= 0.78);
        if let SnappedShape::Line { start, end } = snapped_line {
            assert!((start.y - end.y).abs() < 1.0, "Should snap near horizontal");
        }

        // 2. Circle
        let circle_pts: Vec<Point2D> = (0..=48)
            .map(|i| {
                let a = (i as f32 / 48.0) * 2.0 * PI;
                Point2D::new(260.0 + 85.0 * a.cos(), 260.0 + 85.0 * a.sin())
            })
            .collect();
        let (snapped_c, conf_c) = engine.finalize_stroke_with_confidence(&circle_pts, &mut out);
        assert!(matches!(snapped_c, SnappedShape::Circle { .. }), "Got {snapped_c:?}");
        assert!(conf_c >= 0.78);

        // 3. Rotated Ellipse (30 deg)
        let rot = 30.0_f32.to_radians();
        let ell_pts: Vec<Point2D> = (0..=48)
            .map(|i| {
                let a = (i as f32 / 48.0) * 2.0 * PI;
                let lx = 135.0 * a.cos();
                let ly = 60.0 * a.sin();
                Point2D::new(
                    320.0 + lx * rot.cos() - ly * rot.sin(),
                    280.0 + lx * rot.sin() + ly * rot.cos(),
                )
            })
            .collect();
        let (snapped_e, conf_e) = engine.finalize_stroke_with_confidence(&ell_pts, &mut out);
        assert!(matches!(snapped_e, SnappedShape::Ellipse { .. }), "Got {snapped_e:?}");
        assert!(conf_e >= 0.78);

        // 4. Rotated Rectangle
        let corners = [(-110.0_f32, -55.0), (110.0, -55.0), (110.0, 55.0), (-110.0, 55.0)];
        let r_ang = 25.0_f32.to_radians();
        let mut rect_pts = Vec::new();
        for side in 0..4 {
            let a = corners[side];
            let b = corners[(side + 1) % 4];
            for k in 0..10 {
                let t = k as f32 / 10.0;
                let lx = a.0 + (b.0 - a.0) * t;
                let ly = a.1 + (b.1 - a.1) * t;
                rect_pts.push(Point2D::new(
                    300.0 + lx * r_ang.cos() - ly * r_ang.sin(),
                    300.0 + lx * r_ang.sin() + ly * r_ang.cos(),
                ));
            }
        }
        rect_pts.push(rect_pts[0]);
        let (snapped_r, _) = engine.finalize_stroke_with_confidence(&rect_pts, &mut out);
        assert!(
            matches!(
                snapped_r,
                SnappedShape::Quad {
                    kind: QuadKind::Rectangle,
                    ..
                }
            ),
            "Got {snapped_r:?}"
        );

        // 5. Triangle
        let tri_corners = [(200.0_f32, 120.0), (310.0, 290.0), (90.0, 290.0)];
        let mut tri_pts = Vec::new();
        for s in 0..3 {
            let a = tri_corners[s];
            let b = tri_corners[(s + 1) % 3];
            for k in 0..12 {
                let t = k as f32 / 12.0;
                tri_pts.push(Point2D::new(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
            }
        }
        tri_pts.push(tri_pts[0]);
        let (snapped_t, _) = engine.finalize_stroke_with_confidence(&tri_pts, &mut out);
        assert!(matches!(snapped_t, SnappedShape::Triangle { .. }), "Got {snapped_t:?}");

        // 6. Freehand Sine Wave / Writing must stay Freehand (Confidence Gate < 0.78)
        let wave: Vec<Point2D> = (0..55)
            .map(|i| Point2D::new(60.0 + i as f32 * 6.0, 200.0 + 38.0 * (i as f32 * 0.45).sin()))
            .collect();
        let (snapped_w, _) = engine.finalize_stroke_with_confidence(&wave, &mut out);
        assert!(matches!(snapped_w, SnappedShape::Freehand), "Wave snapped falsely: {snapped_w:?}");
    }
}
