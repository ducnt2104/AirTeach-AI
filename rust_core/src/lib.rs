use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::VecDeque;

const WRIST: usize = 0;
const THUMB_CMC: usize = 1;
const THUMB_MCP: usize = 2;
const THUMB_IP: usize = 3;
const THUMB_TIP: usize = 4;
const INDEX_MCP: usize = 5;
const INDEX_PIP: usize = 6;
const INDEX_DIP: usize = 7;
const INDEX_TIP: usize = 8;
const MIDDLE_MCP: usize = 9;
const MIDDLE_PIP: usize = 10;
const MIDDLE_DIP: usize = 11;
const MIDDLE_TIP: usize = 12;
const RING_MCP: usize = 13;
const RING_PIP: usize = 14;
const RING_DIP: usize = 15;
const RING_TIP: usize = 16;
const PINKY_MCP: usize = 17;
const PINKY_PIP: usize = 18;
const PINKY_DIP: usize = 19;
const PINKY_TIP: usize = 20;

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

    fn filter(&mut self, x: f64, y: f64, t_sec: f64) -> (f64, f64) {
        (self.fx.filter(x, t_sec), self.fy.filter(y, t_sec))
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

    fn filter(&mut self, x: f64, y: f64, z: f64, t_sec: f64) -> (f64, f64, f64) {
        (
            self.fx.filter(x, t_sec),
            self.fy.filter(y, t_sec),
            self.fz.filter(z, t_sec),
        )
    }
}

#[pyclass]
pub struct RustKinematic60HzExtrapolator {
    fx: RustOneEuroFilter1D,
    fy: RustOneEuroFilter1D,
    max_stale_sec: f64,
    anchor_xy: Option<(f64, f64)>,
    anchor_t: f64,
    #[pyo3(get)]
    pub velocity: (f64, f64),
    last_output: Option<(f64, f64)>,
}

#[pymethods]
impl RustKinematic60HzExtrapolator {
    #[new]
    #[pyo3(signature = (min_cutoff = 0.8, beta = 0.04, d_cutoff = 1.0, max_stale_sec = 0.16))]
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64, max_stale_sec: f64) -> Self {
        Self {
            fx: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            fy: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            max_stale_sec: max_stale_sec.max(0.05),
            anchor_xy: None,
            anchor_t: 0.0,
            velocity: (0.0, 0.0),
            last_output: None,
        }
    }

    fn reset(&mut self) {
        self.fx.reset();
        self.fy.reset();
        self.anchor_xy = None;
        self.anchor_t = 0.0;
        self.velocity = (0.0, 0.0);
        self.last_output = None;
    }

    #[pyo3(signature = (cursor_norm = None, t_sec = 0.0))]
    fn ingest_vision(
        &mut self,
        cursor_norm: Option<(f64, f64)>,
        t_sec: f64,
    ) -> Option<(f64, f64)> {
        let Some(pt) = cursor_norm else {
            self.reset();
            return None;
        };
        let cx = pt.0.clamp(0.0, 1.0);
        let cy = pt.1.clamp(0.0, 1.0);
        if let Some(anc) = self.anchor_xy {
            if self.anchor_t > 0.0 {
                let dt = (t_sec - self.anchor_t).max(1e-3);
                if dt <= 0.25 {
                    let inst_vx = ((cx - anc.0) / dt).clamp(-3.5, 3.5);
                    let inst_vy = ((cy - anc.1) / dt).clamp(-3.5, 3.5);
                    self.velocity = (
                        0.55 * self.velocity.0 + 0.45 * inst_vx,
                        0.55 * self.velocity.1 + 0.45 * inst_vy,
                    );
                } else {
                    self.velocity = (0.0, 0.0);
                    self.fx.reset();
                    self.fy.reset();
                }
            }
        } else {
            self.velocity = (0.0, 0.0);
        }
        self.anchor_xy = Some((cx, cy));
        self.anchor_t = t_sec;
        let ox = self.fx.filter(cx, t_sec).clamp(0.0, 1.0);
        let oy = self.fy.filter(cy, t_sec).clamp(0.0, 1.0);
        self.last_output = Some((ox, oy));
        self.last_output
    }

    fn sample_60hz(&mut self, t_sec: f64) -> (Option<(f64, f64)>, bool) {
        let Some(anc) = self.anchor_xy else {
            return (None, false);
        };
        let elapsed = t_sec - self.anchor_t;
        if elapsed > self.max_stale_sec {
            return (None, false);
        }
        let dt = elapsed.clamp(0.0, 0.045);
        let decay = (-10.0 * dt).exp();
        let px = (anc.0 + self.velocity.0 * dt * decay).clamp(0.0, 1.0);
        let py = (anc.1 + self.velocity.1 * dt * decay).clamp(0.0, 1.0);
        let ox = self.fx.filter(px, t_sec).clamp(0.0, 1.0);
        let oy = self.fy.filter(py, t_sec).clamp(0.0, 1.0);
        self.last_output = Some((ox, oy));
        (self.last_output, dt > 0.004)
    }
}

#[pyclass]
pub struct RustSwipeDetector {
    swipe_cooldown_sec: f64,
    swipe_min_dx: f64,
    swipe_min_vel: f64,
    swipe_window_sec: f64,
    vertical_cooldown_sec: f64,
    vertical_min_dy: f64,
    vertical_min_vel: f64,
    trail: VecDeque<(f64, f64, f64)>,
    last_trigger_ts: f64,
    last_vert_trigger_ts: f64,
}

#[pymethods]
impl RustSwipeDetector {
    #[new]
    #[pyo3(signature = (
        swipe_cooldown_sec = 0.60,
        swipe_min_dx = 0.16,
        swipe_min_vel = 0.85,
        swipe_window_sec = 0.28,
        vertical_cooldown_sec = 0.45,
        vertical_min_dy = 0.14,
        vertical_min_vel = 0.75
    ))]
    fn new(
        swipe_cooldown_sec: f64,
        swipe_min_dx: f64,
        swipe_min_vel: f64,
        swipe_window_sec: f64,
        vertical_cooldown_sec: f64,
        vertical_min_dy: f64,
        vertical_min_vel: f64,
    ) -> Self {
        Self {
            swipe_cooldown_sec: swipe_cooldown_sec.max(0.25),
            swipe_min_dx: swipe_min_dx.max(0.08),
            swipe_min_vel: swipe_min_vel.max(0.35),
            swipe_window_sec: swipe_window_sec.max(0.12),
            vertical_cooldown_sec: vertical_cooldown_sec.max(0.25),
            vertical_min_dy: vertical_min_dy.max(0.07),
            vertical_min_vel: vertical_min_vel.max(0.30),
            trail: VecDeque::with_capacity(24),
            last_trigger_ts: -1e9,
            last_vert_trigger_ts: -1e9,
        }
    }

    fn reset(&mut self) {
        self.trail.clear();
    }

    fn push_and_check(
        &mut self,
        x: f64,
        y: f64,
        now: f64,
        pose_tag: &str,
    ) -> (Option<&'static str>, Option<&'static str>, f64) {
        if self.trail.len() >= 24 {
            self.trail.pop_front();
        }
        self.trail.push_back((x, y, now));
        while let Some(&(_, _, ot)) = self.trail.front() {
            if now - ot > self.swipe_window_sec {
                self.trail.pop_front();
            } else {
                break;
            }
        }

        let cooldown_ok = (now - self.last_trigger_ts) >= self.swipe_cooldown_sec;
        let vert_ok = (now - self.last_vert_trigger_ts) >= self.vertical_cooldown_sec;
        let pose_up = pose_tag.to_ascii_uppercase();
        let allow_vert =
            pose_up == "OPEN_PALM" || pose_up == "TWO_FINGERS" || pose_up == "HOVER";

        if self.trail.len() >= 3 {
            for &(ox, oy, ot) in &self.trail {
                let dt = now - ot;
                if dt < 0.04 || dt > self.swipe_window_sec {
                    continue;
                }
                let dx = x - ox;
                let dy = y - oy;
                let vel_x = dx.abs() / dt;
                let vel_y = dy.abs() / dt;
                if cooldown_ok
                    && dx.abs() >= self.swipe_min_dx
                    && vel_x >= self.swipe_min_vel
                    && dx.abs() >= 1.8 * dy.abs()
                {
                    self.last_trigger_ts = now;
                    self.trail.clear();
                    return if dx < 0.0 {
                        (Some("SLIDE_NEXT"), Some("LEFT"), 0.0)
                    } else {
                        (Some("SLIDE_PREV"), Some("RIGHT"), 0.0)
                    };
                }
                if allow_vert
                    && vert_ok
                    && dy.abs() >= self.vertical_min_dy
                    && vel_y >= self.vertical_min_vel
                    && dy.abs() >= 1.8 * dx.abs()
                {
                    self.last_vert_trigger_ts = now;
                    self.trail.clear();
                    return if dy < 0.0 {
                        (Some("SCROLL_DOWN"), Some("UP"), 0.38)
                    } else {
                        (Some("SCROLL_UP"), Some("DOWN"), -0.38)
                    };
                }
            }
        }
        (None, None, 0.0)
    }
}

#[inline]
fn pt3(lm: &[Vec<f64>], idx: usize) -> (f64, f64, f64) {
    let p = &lm[idx];
    (p[0], p[1], if p.len() > 2 { p[2] } else { 0.0 })
}

#[inline]
fn vec3(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (b.0 - a.0, b.1 - a.1, b.2 - a.2)
}

#[inline]
fn dist3(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn dot3(u: (f64, f64, f64), v: (f64, f64, f64)) -> f64 {
    u.0 * v.0 + u.1 * v.1 + u.2 * v.2
}

#[inline]
fn norm3(u: (f64, f64, f64)) -> f64 {
    (u.0 * u.0 + u.1 * u.1 + u.2 * u.2).sqrt()
}

#[inline]
fn cross3(u: (f64, f64, f64), v: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        u.1 * v.2 - u.2 * v.1,
        u.2 * v.0 - u.0 * v.2,
        u.0 * v.1 - u.1 * v.0,
    )
}

fn vec_angle_deg(u: (f64, f64, f64), v: (f64, f64, f64)) -> f64 {
    let nu = norm3(u);
    let nv = norm3(v);
    if nu < 1e-7 || nv < 1e-7 {
        return 0.0;
    }
    let c = (dot3(u, v) / (nu * nv)).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

fn finger_angles_3d(
    lm: &[Vec<f64>],
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
) -> (f64, f64, f64) {
    let p_mcp = pt3(lm, mcp);
    let p_pip = pt3(lm, pip);
    let p_dip = pt3(lm, dip);
    let p_tip = pt3(lm, tip);
    let u1 = vec3(p_mcp, p_pip);
    let u2 = vec3(p_pip, p_dip);
    let u3 = vec3(p_dip, p_tip);
    let u_ov = vec3(p_pip, p_tip);
    (
        vec_angle_deg(u1, u2),
        vec_angle_deg(u2, u3),
        vec_angle_deg(u1, u_ov),
    )
}

fn classify_finger_flexion(
    lm: &[Vec<f64>],
    palm_scale: f64,
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
) -> (&'static str, f64) {
    let wrist = pt3(lm, WRIST);
    let (pip_ang, dip_ang, ov_ang) = finger_angles_3d(lm, mcp, pip, dip, tip);
    let r_pip = dist3(wrist, pt3(lm, pip)) / palm_scale;
    let r_tip = dist3(wrist, pt3(lm, tip)) / palm_scale;
    let radial_ratio = r_tip / r_pip.max(0.15);
    let eff = pip_ang.max(ov_ang * 0.92);
    let eff_r = (eff * 10.0).round() / 10.0;
    if pip_ang < 42.0
        && dip_ang < 48.0
        && ov_ang < 42.0
        && radial_ratio > 1.12
        && r_tip > 1.18
    {
        return ("EXTENDED", eff_r);
    }
    if eff < 80.0 && radial_ratio > 0.92 && r_tip > 0.95 {
        return ("HALF_BENT", eff_r);
    }
    ("CURLED", eff_r)
}

fn is_finger_ext(
    lm: &[Vec<f64>],
    palm_scale: f64,
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
    was_ext: bool,
) -> bool {
    let wrist = pt3(lm, WRIST);
    let (pip_ang, dip_ang, ov_ang) = finger_angles_3d(lm, mcp, pip, dip, tip);
    let r_pip = dist3(wrist, pt3(lm, pip)) / palm_scale;
    let r_tip = dist3(wrist, pt3(lm, tip)) / palm_scale;
    let radial_ratio = r_tip / r_pip.max(0.15);
    if was_ext {
        pip_ang < 54.0 && dip_ang < 58.0 && ov_ang < 48.0 && radial_ratio > 1.05 && r_tip > 1.05
    } else {
        pip_ang < 42.0 && dip_ang < 46.0 && ov_ang < 36.0 && radial_ratio > 1.14 && r_tip > 1.22
    }
}

fn thumb_geom(lm: &[Vec<f64>], palm_scale: f64) -> (f64, f64, f64) {
    let wrist = pt3(lm, WRIST);
    let v_idx = vec3(wrist, pt3(lm, INDEX_MCP));
    let v_pinky = vec3(wrist, pt3(lm, PINKY_MCP));
    let palm_n = cross3(v_idx, v_pinky);
    let n_norm = norm3(palm_n);
    let v_thumb = vec3(pt3(lm, THUMB_MCP), pt3(lm, THUMB_TIP));
    let t_norm = norm3(v_thumb);
    let mut abd = vec_angle_deg(v_thumb, v_idx);
    if n_norm > 1e-7 && t_norm > 1e-7 {
        let sin_el = (dot3(v_thumb, palm_n).abs() / (t_norm * n_norm)).clamp(0.0, 1.0);
        abd = abd.max(sin_el.asin().to_degrees());
    }
    let ip_deg = vec_angle_deg(
        vec3(pt3(lm, THUMB_CMC), pt3(lm, THUMB_MCP)),
        vec3(pt3(lm, THUMB_IP), pt3(lm, THUMB_TIP)),
    );
    let spread = dist3(pt3(lm, THUMB_TIP), pt3(lm, INDEX_MCP)) / palm_scale;
    (abd, ip_deg, spread)
}

#[inline]
fn map_workspace(x: f64, y: f64) -> (f64, f64) {
    let span = 0.88 - 0.12;
    (
        ((x - 0.12) / span).clamp(0.0, 1.0),
        ((y - 0.12) / span).clamp(0.0, 1.0),
    )
}

#[inline]
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

#[pyfunction]
#[pyo3(signature = (
    lm,
    prev_fingers = None,
    was_pinched = false,
    pinch_enter_ratio = 0.28,
    pinch_exit_ratio = 0.42,
    handedness = "right",
    palm_facing_min_z = 0.20
))]
#[allow(clippy::too_many_arguments)]
fn classify_hand_3d_rs<'py>(
    py: Python<'py>,
    lm: Vec<Vec<f64>>,
    prev_fingers: Option<Vec<bool>>,
    was_pinched: bool,
    pinch_enter_ratio: f64,
    pinch_exit_ratio: f64,
    handedness: &str,
    palm_facing_min_z: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if lm.len() < 21 || lm.iter().any(|p| p.len() < 2) {
        out.set_item("gesture", "IDLE")?;
        out.set_item("pose_tag", "IDLE")?;
        out.set_item("cursor_norm", (0.5_f64, 0.5_f64))?;
        out.set_item("pinch_ratio", 1.0_f64)?;
        out.set_item("palm_scale", 0.1_f64)?;
        out.set_item("fingers", vec![false; 5])?;
        out.set_item("finger_states", vec!["CURLED"; 5])?;
        out.set_item("finger_angles_deg", vec![0.0_f64; 5])?;
        out.set_item("palm_normal", (0.0_f64, 0.0_f64, 0.0_f64))?;
        out.set_item("palm_facing_score", 0.0_f64)?;
        out.set_item("palm_facing_ok", false)?;
        out.set_item("palm_orientation", "SIDE")?;
        return Ok(out);
    }

    let state: [bool; 5] = match prev_fingers {
        Some(ref pf) if pf.len() == 5 => [pf[0], pf[1], pf[2], pf[3], pf[4]],
        _ => [false; 5],
    };

    let wrist = pt3(&lm, WRIST);
    let mid_mcp = pt3(&lm, MIDDLE_MCP);
    let palm_scale = dist3(wrist, mid_mcp).max(1e-4);
    let (t_abd, t_ip, t_spread) = thumb_geom(&lm, palm_scale);

    let thumb_up = if state[0] {
        t_ip < 62.0 && t_spread > 0.36 && t_abd > 14.0
    } else {
        t_ip < 48.0 && t_spread > 0.45 && t_abd > 18.0
    };
    let idx_up = is_finger_ext(
        &lm, palm_scale, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, state[1],
    );
    let mid_up = is_finger_ext(
        &lm, palm_scale, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, state[2],
    );
    let ring_up = is_finger_ext(
        &lm, palm_scale, RING_MCP, RING_PIP, RING_DIP, RING_TIP, state[3],
    );
    let pinky_up = is_finger_ext(
        &lm, palm_scale, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, state[4],
    );

    let t_ang_r = (t_ip * 10.0).round() / 10.0;
    let t_state = if t_ip < 48.0 && t_spread > 0.45 && t_abd > 18.0 {
        "EXTENDED"
    } else if t_ip < 76.0 && t_spread > 0.30 {
        "HALF_BENT"
    } else {
        "CURLED"
    };
    let (i_state, i_ang) =
        classify_finger_flexion(&lm, palm_scale, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP);
    let (m_state, m_ang) =
        classify_finger_flexion(&lm, palm_scale, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP);
    let (r_state, r_ang) =
        classify_finger_flexion(&lm, palm_scale, RING_MCP, RING_PIP, RING_DIP, RING_TIP);
    let (p_state, p_ang) =
        classify_finger_flexion(&lm, palm_scale, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP);

    let thumb_tip = pt3(&lm, THUMB_TIP);
    let index_tip = pt3(&lm, INDEX_TIP);
    let mid_tip = pt3(&lm, MIDDLE_TIP);
    let pinky_tip = pt3(&lm, PINKY_TIP);

    let pinch_ratio = dist3(thumb_tip, index_tip) / palm_scale;
    let pinch_thresh = if was_pinched {
        pinch_exit_ratio
    } else {
        pinch_enter_ratio
    };
    let index_reach = dist3(wrist, index_tip) / palm_scale;
    let hand_span = palm_scale.max(dist3(thumb_tip, pinky_tip));

    let v_idx = vec3(wrist, pt3(&lm, INDEX_MCP));
    let v_pinky = vec3(wrist, pt3(&lm, PINKY_MCP));
    let cvec = cross3(v_idx, v_pinky);
    let cnorm = norm3(cvec);
    let _ = handedness;
    let (palm_normal, facing_score, facing_ok, orientation) = if cnorm < 1e-7 {
        ((0.0_f64, 0.0_f64, 0.0_f64), 0.0_f64, false, "SIDE")
    } else {
        let ux = cvec.0 / cnorm;
        let uy = cvec.1 / cnorm;
        let uz = cvec.2 / cnorm;
        let fs = uz.abs();
        let ori = if fs < (palm_facing_min_z * 0.7).max(0.08) {
            "SIDE"
        } else if fs >= palm_facing_min_z {
            "FRONT"
        } else {
            "ANGLED"
        };
        (
            (round4(ux), round4(uy), round4(uz)),
            round4(fs),
            fs >= palm_facing_min_z,
            ori,
        )
    };

    out.set_item("pinch_ratio", round4(pinch_ratio))?;
    out.set_item("palm_scale", round4(palm_scale))?;
    out.set_item("hand_span", round4(hand_span))?;
    out.set_item("fingers", vec![thumb_up, idx_up, mid_up, ring_up, pinky_up])?;
    out.set_item(
        "finger_states",
        vec![t_state, i_state, m_state, r_state, p_state],
    )?;
    out.set_item(
        "finger_angles_deg",
        vec![t_ang_r, i_ang, m_ang, r_ang, p_ang],
    )?;
    out.set_item("palm_normal", palm_normal)?;
    out.set_item("palm_facing_score", facing_score)?;
    out.set_item("palm_facing_ok", facing_ok)?;
    out.set_item("palm_orientation", orientation)?;

    let (gesture, pose_tag, cursor) = if idx_up && mid_up && ring_up && pinky_up && pinch_ratio > 0.38
    {
        ("ERASE", "OPEN_PALM", map_workspace(mid_mcp.0, mid_mcp.1))
    } else if !idx_up && !mid_up && !ring_up && !pinky_up && index_reach < 1.15 {
        ("FIST", "FIST", map_workspace(mid_mcp.0, mid_mcp.1))
    } else if pinch_ratio < pinch_thresh {
        let px = 0.5 * (thumb_tip.0 + index_tip.0);
        let py = 0.5 * (thumb_tip.1 + index_tip.1);
        ("CLICK", "PINCH", map_workspace(px, py))
    } else if idx_up && mid_up && ring_up && !pinky_up && pinch_ratio > 0.40 {
        let cx = 0.5 * (index_tip.0 + mid_tip.0);
        let cy = 0.5 * (index_tip.1 + mid_tip.1);
        ("UNDO", "THREE_FINGERS", map_workspace(cx, cy))
    } else if idx_up && mid_up && !ring_up {
        let cx = 0.5 * (index_tip.0 + mid_tip.0);
        let cy = 0.5 * (index_tip.1 + mid_tip.1);
        ("HOVER", "TWO_FINGERS", map_workspace(cx, cy))
    } else if idx_up && !mid_up && !ring_up {
        ("DRAW", "INDEX", map_workspace(index_tip.0, index_tip.1))
    } else if pinky_up && !idx_up && !mid_up && !ring_up && pinch_ratio > 0.40 {
        ("HOVER", "QUESTION_HOLD", map_workspace(pinky_tip.0, pinky_tip.1))
    } else if !idx_up && !mid_up && !ring_up && !pinky_up {
        ("FIST", "FIST", map_workspace(mid_mcp.0, mid_mcp.1))
    } else {
        ("HOVER", "HOVER", map_workspace(index_tip.0, index_tip.1))
    };

    out.set_item("gesture", gesture)?;
    out.set_item("pose_tag", pose_tag)?;
    out.set_item("cursor_norm", cursor)?;
    Ok(out)
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
#[pyo3(signature = (pts, closed_gap_ratio = 0.18, confidence_gate = 0.78))]
fn snap_geometric_shape_rs(
    pts: Vec<(f64, f64)>,
    closed_gap_ratio: f64,
    confidence_gate: f64,
) -> (String, f64, bool, Vec<(f64, f64)>) {
    if pts.len() < 3 {
        return ("FREEHAND".to_string(), 0.0, false, pts);
    }
    let mut clean = vec![pts[0]];
    for p in pts.into_iter().skip(1) {
        let last = clean.last().unwrap();
        let dx = p.0 - last.0;
        let dy = p.1 - last.1;
        if dx * dx + dy * dy >= 0.36 {
            clean.push(p);
        }
    }
    if clean.len() < 3 {
        return ("FREEHAND".to_string(), 0.0, false, clean);
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in &clean {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let bbox_diag = (max_x - min_x).hypot(max_y - min_y);
    if bbox_diag < 18.0 {
        return ("FREEHAND".to_string(), 0.0, false, clean);
    }
    let mut total_len = 0.0_f64;
    for w in clean.windows(2) {
        total_len += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
    }
    let gap = (clean[0].0 - clean.last().unwrap().0).hypot(clean[0].1 - clean.last().unwrap().1);
    let is_closed = gap <= closed_gap_ratio * total_len.min(bbox_diag * 2.2) || gap <= 0.22 * bbox_diag;

    if !is_closed {
        let resampled = resample_arc_length_rs(clean.clone(), 64);
        let rn = resampled.len() as f64;
        let mx = resampled.iter().map(|p| p.0).sum::<f64>() / rn;
        let my = resampled.iter().map(|p| p.1).sum::<f64>() / rn;
        let mut sxx = 0.0_f64;
        let mut syy = 0.0_f64;
        let mut sxy = 0.0_f64;
        for &(x, y) in &resampled {
            let dx = x - mx;
            let dy = y - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        sxx /= rn;
        syy /= rn;
        sxy /= rn;
        let trace = sxx + syy;
        let disc = ((sxx - syy).powi(2) + 4.0 * sxy * sxy).sqrt();
        let l1 = 0.5 * (trace + disc);
        let l2 = (0.5 * (trace - disc)).max(0.0);
        let eig_ratio = l2 / l1.max(1e-9);
        let simplified = rdp_simplify_rs(clean.clone(), (bbox_diag * 0.025).max(3.0));
        if eig_ratio < 0.035 && simplified.len() <= 6 {
            let line_conf = (1.0 - (eig_ratio / 0.035) * 0.18).clamp(0.80, 0.99);
            if line_conf >= confidence_gate {
                return (
                    "LINE".to_string(),
                    line_conf,
                    false,
                    vec![clean[0], *clean.last().unwrap()],
                );
            }
        }
        return ("FREEHAND".to_string(), 0.0, false, clean);
    }

    let resampled = resample_arc_length_rs(clean.clone(), 64);
    let n = resampled.len();
    let mut area2 = 0.0_f64;
    let mut peri = 0.0_f64;
    for i in 0..n {
        let a = resampled[i];
        let b = resampled[(i + 1) % n];
        area2 += a.0 * b.1 - b.0 * a.1;
        peri += (b.0 - a.0).hypot(b.1 - a.1);
    }
    let area = (area2 * 0.5).abs();
    let circularity = (4.0 * std::f64::consts::PI * area) / (peri * peri).max(1e-6);

    let cx = resampled.iter().map(|p| p.0).sum::<f64>() / (n as f64);
    let cy = resampled.iter().map(|p| p.1).sum::<f64>() / (n as f64);
    let radii: Vec<f64> = resampled
        .iter()
        .map(|p| (p.0 - cx).hypot(p.1 - cy))
        .collect();
    let r = radii.iter().sum::<f64>() / (n as f64);
    if r > 4.0 {
        let var_r = radii.iter().map(|rv| (rv - r) * (rv - r)).sum::<f64>() / (n as f64);
        let radial_cv = var_r.sqrt() / r.max(1e-6);
        if radial_cv < 0.14 && circularity >= 0.76 {
            let conf = (1.0 - (radial_cv / 0.14) * 0.22).clamp(0.80, 0.99);
            if conf >= confidence_gate {
                let circle_pts = (0..=64)
                    .map(|k| {
                        let ang = 2.0 * std::f64::consts::PI * (k as f64) / 64.0;
                        (cx + r * ang.cos(), cy + r * ang.sin())
                    })
                    .collect();
                return ("CIRCLE".to_string(), conf, true, circle_pts);
            }
        }
    }
    ("FREEHAND".to_string(), 0.0, true, clean)
}

#[pyfunction]
#[pyo3(signature = (data, width, height, channels, is_rgb = true, stride = 4))]
fn analyze_frame_luma_rs(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    is_rgb: bool,
    stride: usize,
) -> (f64, f64, f64) {
    if width == 0 || height == 0 || channels == 0 || data.is_empty() {
        return (120.0, 30.0, 210.0);
    }
    let step = stride.max(1);
    let row_bytes = width * channels;
    let mut hist = [0u32; 256];
    let mut sum = 0u64;
    let mut count = 0u64;

    let mut y = 0usize;
    while y < height {
        let row_off = y * row_bytes;
        if row_off + row_bytes > data.len() {
            break;
        }
        let mut x = 0usize;
        while x < width {
            let idx = row_off + x * channels;
            let lum = if channels >= 3 {
                let (r, g, b) = if is_rgb {
                    (data[idx] as u32, data[idx + 1] as u32, data[idx + 2] as u32)
                } else {
                    (data[idx + 2] as u32, data[idx + 1] as u32, data[idx] as u32)
                };
                ((77 * r + 150 * g + 29 * b) >> 8).min(255) as usize
            } else {
                data[idx] as usize
            };
            hist[lum] += 1;
            sum += lum as u64;
            count += 1;
            x += step;
        }
        y += step;
    }

    if count == 0 {
        return (120.0, 30.0, 210.0);
    }
    let mean_luma = (sum as f64) / (count as f64);
    let p5_rank = ((count as f64) * 0.05).ceil() as u64;
    let p95_rank = ((count as f64) * 0.95).ceil() as u64;
    let mut cum = 0u64;
    let mut p5 = 0.0_f64;
    let mut p95 = 255.0_f64;
    let mut found_p5 = false;
    for (val, &c) in hist.iter().enumerate() {
        cum += c as u64;
        if !found_p5 && cum >= p5_rank.max(1) {
            p5 = val as f64;
            found_p5 = true;
        }
        if cum >= p95_rank.max(1) {
            p95 = val as f64;
            break;
        }
    }
    (mean_luma, p5, p95)
}

#[pyfunction]
#[pyo3(signature = (data, gamma = 1.0, alpha = 1.0, beta = 0.0))]
fn apply_gamma_contrast_rs<'py>(
    py: Python<'py>,
    data: &[u8],
    gamma: f64,
    alpha: f64,
    beta: f64,
) -> Bound<'py, PyBytes> {
    let mut lut = [0u8; 256];
    let use_gamma = (gamma - 1.0).abs() > 1e-4;
    for (i, slot) in lut.iter_mut().enumerate() {
        let g = if use_gamma {
            ((i as f64) / 255.0).powf(gamma) * 255.0
        } else {
            i as f64
        };
        *slot = (g * alpha + beta).round().clamp(0.0, 255.0) as u8;
    }
    let out: Vec<u8> = data.iter().map(|&b| lut[b as usize]).collect();
    PyBytes::new(py, &out)
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
    vec_angle_deg(vec3(b, a), vec3(b, c))
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
    "airteach_core_rs 2.0.0"
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
    m.add_class::<RustKinematic60HzExtrapolator>()?;
    m.add_class::<RustSwipeDetector>()?;
    m.add_function(wrap_pyfunction!(classify_hand_3d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_simplify_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_simplify_2d, m)?)?;
    m.add_function(wrap_pyfunction!(resample_arc_length_rs, m)?)?;
    m.add_function(wrap_pyfunction!(catmull_rom_spline_rs, m)?)?;
    m.add_function(wrap_pyfunction!(snap_geometric_shape_rs, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_frame_luma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gamma_contrast_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dual_hand_kinematics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_roi_crop_box_rs, m)?)?;
    m.add_function(wrap_pyfunction!(joint_angle_3d_deg, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_vec, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}
