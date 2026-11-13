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

/// Recognized geometric shape when user lifts pen / clenches fist.
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
    Rectangle {
        min: Point2D,
        max: Point2D,
    },
    Triangle {
        vertices: [Point2D; 3],
    },
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

/// Pre-allocated stroke processor that executes EMA (alpha=0.35) + 1€ smoothing, RDP simplification,
/// Cubic Bézier generation, and Shape Snapping without inner-loop heap allocations.
pub struct StrokeEngine {
    pub ema: EmaFilter2D,
    pub filter: OneEuroFilter2D,
    pub rdp_epsilon: f32,
    pub bezier_subdivisions: usize,
    pub auto_shape_snap: bool,
    // Pre-allocated scratch buffers
    keep_flags_buf: Vec<bool>,
    rdp_stack_buf: Vec<(usize, usize)>,
    rdp_out_buf: Vec<Point2D>,
    hull_sorted_buf: Vec<Point2D>,
    hull_out_buf: Vec<Point2D>,
    hull_simplified_buf: Vec<Point2D>,
}

impl StrokeEngine {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32, rdp_epsilon: f32) -> Self {
        Self {
            ema: EmaFilter2D::new(0.25),
            filter: OneEuroFilter2D::new(min_cutoff, beta, d_cutoff),
            rdp_epsilon,
            bezier_subdivisions: 6,
            auto_shape_snap: false,
            keep_flags_buf: Vec::with_capacity(2048),
            rdp_stack_buf: Vec::with_capacity(256),
            rdp_out_buf: Vec::with_capacity(512),
            hull_sorted_buf: Vec::with_capacity(512),
            hull_out_buf: Vec::with_capacity(256),
            hull_simplified_buf: Vec::with_capacity(64),
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

    /// Finalize a completed stroke on pen lift / fist clench:
    /// 1. Run RDP downsampling
    /// 2. Attempt geometric shape snapping (Line, Circle, Triangle, Rectangle)
    /// 3. Render snapped shape or C1 Cubic Bézier spline into `out_display_points`
    pub fn finalize_stroke(
        &mut self,
        raw_points: &[Point2D],
        out_display_points: &mut Vec<Point2D>,
    ) -> SnappedShape {
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
            let snapped = self.detect_shape(raw_points);
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
                    let segments = 64;
                    for i in 0..=segments {
                        let theta = (i as f32 / segments as f32) * 2.0 * PI;
                        out_display_points.push(Point2D {
                            x: center.x + radius * theta.cos(),
                            y: center.y + radius * theta.sin(),
                        });
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

    fn detect_shape(&mut self, points: &[Point2D]) -> SnappedShape {
        if points.len() < 5 {
            return SnappedShape::Freehand;
        }

        let first = points[0];
        let last = points[points.len() - 1];
        let mut path_len = 0.0_f32;
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for (i, &p) in points.iter().enumerate() {
            if i > 0 {
                path_len += p.dist(points[i - 1]);
            }
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        if path_len < 35.0 {
            return SnappedShape::Freehand;
        }

        let direct_dist = first.dist(last);
        // MODULE 3.3: Straight Line Check — only snap if stroke length > 80px AND
        // maximum perpendicular deviation from the start-to-end line is < 8.0px.
        if path_len > 80.0 && direct_dist > 80.0 {
            let mut max_dev = 0.0_f32;
            for &p in &points[1..points.len() - 1] {
                let d = perpendicular_distance(p, first, last);
                if d > max_dev {
                    max_dev = d;
                }
            }
            if max_dev < 8.0 && (path_len / direct_dist) < 1.12 {
                return SnappedShape::Line {
                    start: first,
                    end: last,
                };
            }
        }

        // 1b. Arrow Check: Long straight shaft followed by an angled arrowhead hook at the end
        if self.rdp_out_buf.len() >= 3 && self.rdp_out_buf.len() <= 5 && direct_dist / path_len > 0.62 {
            let tip = self.rdp_out_buf[self.rdp_out_buf.len() - 2];
            let tail_end = self.rdp_out_buf[self.rdp_out_buf.len() - 1];
            let shaft_len = first.dist(tip);
            let hook_len = tip.dist(tail_end);
            if shaft_len > 45.0 && hook_len > 10.0 && hook_len < shaft_len * 0.48 {
                let dx = tip.x - first.x;
                let dy = tip.y - first.y;
                let angle = dy.atan2(dx);
                let head_len = (shaft_len * 0.22).clamp(14.0, 36.0);
                let wing_angle = 2.62_f32; // ~150 degrees back from shaft direction
                let head_left = Point2D::new(
                    tip.x + head_len * (angle + wing_angle).cos(),
                    tip.y + head_len * (angle + wing_angle).sin(),
                );
                let head_right = Point2D::new(
                    tip.x + head_len * (angle - wing_angle).cos(),
                    tip.y + head_len * (angle - wing_angle).sin(),
                );
                return SnappedShape::Arrow {
                    start: first,
                    end: tip,
                    head_left,
                    head_right,
                };
            }
        }

        let bbox_diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
        // Check if stroke forms a closed loop
        let is_closed = direct_dist < (bbox_diag * 0.35).max(40.0);
        if !is_closed {
            return SnappedShape::Freehand;
        }

        convex_hull_into(points, &mut self.hull_sorted_buf, &mut self.hull_out_buf);
        if self.hull_out_buf.len() < 3 {
            return SnappedShape::Freehand;
        }

        let (hull_area, hull_perim) = polygon_area_and_perimeter(&self.hull_out_buf);
        if hull_perim < 1e-3 || hull_area < 200.0 {
            return SnappedShape::Freehand;
        }

        // Circularity Index: C = (4 * pi * Area) / Perimeter^2
        let circularity = (4.0 * PI * hull_area) / (hull_perim * hull_perim);
        let bbox_w = (max_x - min_x).max(1.0);
        let bbox_h = (max_y - min_y).max(1.0);
        let aspect = (bbox_w / bbox_h).max(bbox_h / bbox_w);

        // 2. Circle Check: Circularity = (4 * pi * Area) / Perimeter^2 > 0.78
        if circularity >= 0.78 && aspect <= 1.45 {
            let center = Point2D::new(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
            let radius = 0.25 * (bbox_w + bbox_h);
            return SnappedShape::Circle { center, radius };
        }

        // Simplify the closed stroke with epsilon = 0.04 * Perimeter to count polygon vertices
        let corner_eps = (hull_perim * 0.04).max(8.0);
        rdp_simplify_into(
            points,
            corner_eps,
            &mut self.keep_flags_buf,
            &mut self.rdp_stack_buf,
            &mut self.hull_simplified_buf,
        );

        // Remove duplicate closing vertex if present
        if self.hull_simplified_buf.len() >= 2 {
            let n = self.hull_simplified_buf.len();
            if self.hull_simplified_buf[0].dist(self.hull_simplified_buf[n - 1]) < corner_eps * 1.5
            {
                self.hull_simplified_buf.pop();
            }
        }

        // 3. Triangle Check
        if self.hull_simplified_buf.len() == 3 {
            return SnappedShape::Triangle {
                vertices: [
                    self.hull_simplified_buf[0],
                    self.hull_simplified_buf[1],
                    self.hull_simplified_buf[2],
                ],
            };
        }

        // 4. Rectangle Check
        let bbox_area = bbox_w * bbox_h;
        let rectangularity = hull_area / bbox_area;
        if (self.hull_simplified_buf.len() == 4 && rectangularity > 0.72) || rectangularity > 0.84 {
            return SnappedShape::Rectangle {
                min: Point2D::new(min_x, min_y),
                max: Point2D::new(max_x, max_y),
            };
        }

        SnappedShape::Freehand
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
    fn test_shape_snapper_line_and_circle() {
        let mut engine = StrokeEngine::new(1.0, 0.01, 1.0, 2.0);
        engine.auto_shape_snap = true;
        let mut out = Vec::new();

        // Line
        let line_pts: Vec<Point2D> = (0..20)
            .map(|i| Point2D::new(i as f32 * 10.0, i as f32 * 5.0))
            .collect();
        let snapped = engine.finalize_stroke(&line_pts, &mut out);
        assert!(matches!(snapped, SnappedShape::Line { .. }));

        // Circle
        let circle_pts: Vec<Point2D> = (0..=40)
            .map(|i| {
                let a = (i as f32 / 40.0) * 2.0 * PI;
                Point2D::new(200.0 + 80.0 * a.cos(), 200.0 + 80.0 * a.sin())
            })
            .collect();
        let snapped_c = engine.finalize_stroke(&circle_pts, &mut out);
        assert!(matches!(snapped_c, SnappedShape::Circle { .. }));
    }
}
