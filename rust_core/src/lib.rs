use pyo3::prelude::*;

fn smoothing_factor(t_e: f64, cutoff: f64) -> f64 {
    let r = 2.0 * std::f64::consts::PI * cutoff * t_e;
    r / (r + 1.0)
}

fn exponential_smoothing(a: f64, x: f64, x_prev: f64) -> f64 {
    a * x + (1.0 - a) * x_prev
}

#[pyclass]
pub struct RustOneEuroFilter1D {
    #[pyo3(get, set)]
    pub min_cutoff: f64,
    #[pyo3(get, set)]
    pub beta: f64,
    #[pyo3(get, set)]
    pub d_cutoff: f64,
    x_prev: Option<f64>,
    dx_prev: f64,
    t_prev: Option<f64>,
}

#[pymethods]
impl RustOneEuroFilter1D {
    #[new]
    #[pyo3(signature = (min_cutoff = 1.35, beta = 0.015, d_cutoff = 1.0))]
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = None;
    }

    fn filter(&mut self, x: f64, t_sec: f64) -> f64 {
        let (Some(xp), Some(tp)) = (self.x_prev, self.t_prev) else {
            self.x_prev = Some(x);
            self.t_prev = Some(t_sec);
            self.dx_prev = 0.0;
            return x;
        };
        let t_e = (t_sec - tp).max(1e-4);
        self.t_prev = Some(t_sec);
        let a_d = smoothing_factor(t_e, self.d_cutoff);
        let dx = (x - xp) / t_e;
        let dx_hat = exponential_smoothing(a_d, dx, self.dx_prev);
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = smoothing_factor(t_e, cutoff);
        let x_hat = exponential_smoothing(a, x, xp);
        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        x_hat
    }
}

#[pyclass]
pub struct RustOneEuroFilter2D {
    fx: RustOneEuroFilter1D,
    fy: RustOneEuroFilter1D,
}

#[pymethods]
impl RustOneEuroFilter2D {
    #[new]
    #[pyo3(signature = (min_cutoff = 1.35, beta = 0.015, d_cutoff = 1.0))]
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            fx: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            fy: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
        }
    }

    fn reset(&mut self) {
        self.fx.reset();
        self.fy.reset();
    }

    fn filter(&mut self, pt: (f64, f64), t_sec: f64) -> (f64, f64) {
        (self.fx.filter(pt.0, t_sec), self.fy.filter(pt.1, t_sec))
    }
}

#[pyclass]
pub struct RustOneEuroFilter3D {
    fx: RustOneEuroFilter1D,
    fy: RustOneEuroFilter1D,
    fz: RustOneEuroFilter1D,
}

#[pymethods]
impl RustOneEuroFilter3D {
    #[new]
    #[pyo3(signature = (min_cutoff = 1.35, beta = 0.015, d_cutoff = 1.0))]
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            fx: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            fy: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            fz: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
        }
    }

    fn reset(&mut self) {
        self.fx.reset();
        self.fy.reset();
        self.fz.reset();
    }

    fn filter(&mut self, pt: (f64, f64, f64), t_sec: f64) -> (f64, f64, f64) {
        (
            self.fx.filter(pt.0, t_sec),
            self.fy.filter(pt.1, t_sec),
            self.fz.filter(pt.2, t_sec),
        )
    }
}

fn perpendicular_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len_sq).clamp(0.0, 1.0);
    let px = a.0 + t * dx;
    let py = a.1 + t * dy;
    (p.0 - px).hypot(p.1 - py)
}

#[pyfunction]
fn rdp_simplify_rs(points: Vec<(f64, f64)>, epsilon: f64) -> Vec<(f64, f64)> {
    let n = points.len();
    if n <= 2 || epsilon <= 0.0 {
        return points;
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let a = points[start];
        let b = points[end];
        let mut best_idx = None;
        let mut best_dist = -1.0_f64;
        for (idx, pt) in points.iter().enumerate().take(end).skip(start + 1) {
            let d = perpendicular_distance(*pt, a, b);
            if d > best_dist {
                best_dist = d;
                best_idx = Some(idx);
            }
        }
        if let Some(split) = best_idx {
            if best_dist > epsilon {
                keep[split] = true;
                stack.push((start, split));
                stack.push((split, end));
            }
        }
    }

    points
        .into_iter()
        .zip(keep)
        .filter_map(|(pt, k)| if k { Some(pt) } else { None })
        .collect()
}

#[pyfunction]
fn rdp_simplify_2d(points: Vec<(f64, f64)>, epsilon: f64) -> Vec<(f64, f64)> {
    rdp_simplify_rs(points, epsilon)
}

#[pyfunction]
#[pyo3(signature = (pts, n = 64))]
fn resample_arc_length_rs(pts: Vec<(f64, f64)>, n: usize) -> Vec<(f64, f64)> {
    if pts.is_empty() {
        return Vec::new();
    }
    if pts.len() == 1 || n <= 1 {
        return vec![pts[0]; n.max(1)];
    }
    let mut total_len = 0.0_f64;
    for w in pts.windows(2) {
        total_len += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
    }
    if total_len < 1e-6 {
        return vec![pts[0]; n];
    }

    let step = total_len / ((n - 1) as f64);
    let mut cum = 0.0_f64;
    let mut out = Vec::with_capacity(n);
    out.push(pts[0]);
    let mut i = 1usize;
    let mut prev = pts[0];
    while i < pts.len() && out.len() < n {
        let cur = pts[i];
        let seg = (cur.0 - prev.0).hypot(cur.1 - prev.1);
        if cum + seg >= step {
            let t = (step - cum) / seg.max(1e-9);
            let q = (prev.0 + t * (cur.0 - prev.0), prev.1 + t * (cur.1 - prev.1));
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
    out.truncate(n);
    out
}

#[pyfunction]
#[pyo3(signature = (pts, subdivisions = 4, alpha = 0.5))]
fn catmull_rom_spline_rs(
    pts: Vec<(f64, f64)>,
    subdivisions: usize,
    alpha: f64,
) -> Vec<(f64, f64)> {
    let _ = alpha;
    if pts.is_empty() {
        return Vec::new();
    }
    let mut clean = vec![pts[0]];
    for p in pts.into_iter().skip(1) {
        let last = clean.last().unwrap();
        let dx = p.0 - last.0;
        let dy = p.1 - last.1;
        if dx * dx + dy * dy >= 1.0 {
            clean.push(p);
        }
    }
    if clean.len() < 3 {
        return clean;
    }
    let sub = subdivisions.max(3);
    let n = clean.len();
    let mut out = Vec::with_capacity((n - 1) * sub + 1);
    out.push(clean[0]);
    for i in 0..(n - 1) {
        let p0 = if i > 0 { clean[i - 1] } else { clean[i] };
        let p1 = clean[i];
        let p2 = clean[i + 1];
        let p3 = if i + 2 < n { clean[i + 2] } else { clean[n - 1] };
        for s in 1..=sub {
            let t = (s as f64) / (sub as f64);
            let t2 = t * t;
            let t3 = t2 * t;
            let x = 0.5
                * ((2.0 * p1.0)
                    + (-p0.0 + p2.0) * t
                    + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                    + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);
            let y = 0.5
                * ((2.0 * p1.1)
                    + (-p0.1 + p2.1) * t
                    + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                    + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);
            out.push((x, y));
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (p_xy, s_xy, prev_pinch_dist = None, prev_pan_center = None))]
fn compute_dual_hand_kinematics_rs(
    p_xy: (f64, f64),
    s_xy: (f64, f64),
    prev_pinch_dist: Option<f64>,
    prev_pan_center: Option<(f64, f64)>,
) -> (f64, (f64, f64), f64, (f64, f64), (f64, f64)) {
    let dist = (p_xy.0 - s_xy.0).hypot(p_xy.1 - s_xy.1).max(0.03);
    let mid = (0.5 * (p_xy.0 + s_xy.0), 0.5 * (p_xy.1 + s_xy.1));
    let mut zf = 1.0_f64;
    if let Some(prev_d) = prev_pinch_dist {
        if (dist - prev_d).abs() > 0.005 {
            zf = (dist / prev_d.max(0.03)).clamp(0.85, 1.15);
        }
    }
    let mut dx = 0.0_f64;
    let mut dy = 0.0_f64;
    if let Some(prev_mid) = prev_pan_center {
        let rdx = mid.0 - prev_mid.0;
        let rdy = mid.1 - prev_mid.1;
        if rdx.hypot(rdy) > 0.003 {
            dx = rdx.clamp(-0.15, 0.15);
            dy = rdy.clamp(-0.15, 0.15);
        }
    }
    let zf_r = (zf * 10_000.0).round() / 10_000.0;
    let dx_r = (dx * 100_000.0).round() / 100_000.0;
    let dy_r = (dy * 100_000.0).round() / 100_000.0;
    (zf_r, mid, dist, (dx_r, dy_r), mid)
}

#[pyfunction]
#[pyo3(signature = (min_x, min_y, max_x, max_y, pred_dx = 0.0, pred_dy = 0.0, expand_factor = 2.2, padding_ratio = 0.45))]
#[allow(clippy::too_many_arguments)]
fn compute_roi_crop_box_rs(
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    pred_dx: f64,
    pred_dy: f64,
    expand_factor: f64,
    padding_ratio: f64,
) -> (f64, f64, f64, f64) {
    let bw = (max_x - min_x).max(0.06);
    let bh = (max_y - min_y).max(0.06);
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    let expand = expand_factor.max(1.0 + 2.0 * padding_ratio.max(0.45));
    let side = bw.max(bh) * expand;
    let pred_cx = (cx + pred_dx * 0.5).clamp(0.0, 1.0);
    let pred_cy = (cy + pred_dy * 0.5).clamp(0.0, 1.0);
    let rx0 = (pred_cx - 0.5 * side).max(0.0);
    let ry0 = (pred_cy - 0.5 * side).max(0.0);
    (rx0, ry0, side.min(1.0 - rx0), side.min(1.0 - ry0))
}

#[pyfunction]
fn joint_angle_3d_deg(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> f64 {
    let u = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
    let v = (c.0 - b.0, c.1 - b.1, c.2 - b.2);
    let nu = (u.0 * u.0 + u.1 * u.1 + u.2 * u.2).sqrt();
    let nv = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if nu < 1e-8 || nv < 1e-8 {
        return 0.0;
    }
    let cos_theta = ((u.0 * v.0 + u.1 * v.1 + u.2 * v.2) / (nu * nv)).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

#[pyfunction]
fn cosine_similarity_vec(a: Vec<f32>, b: Vec<f32>) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0) as f32
}

#[pyfunction]
fn version() -> &'static str {
    "2.0.0"
}

#[pyfunction]
fn backend_info() -> &'static str {
    "airteach_core_rs 2.0.0 (Rust PyO3 native)"
}

#[pymodule]
fn airteach_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustOneEuroFilter1D>()?;
    m.add_class::<RustOneEuroFilter2D>()?;
    m.add_class::<RustOneEuroFilter3D>()?;
    m.add_function(wrap_pyfunction!(rdp_simplify_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_simplify_2d, m)?)?;
    m.add_function(wrap_pyfunction!(resample_arc_length_rs, m)?)?;
    m.add_function(wrap_pyfunction!(catmull_rom_spline_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dual_hand_kinematics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_roi_crop_box_rs, m)?)?;
    m.add_function(wrap_pyfunction!(joint_angle_3d_deg, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_vec, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}
