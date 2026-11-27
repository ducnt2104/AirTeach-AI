use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, VecDeque};

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

        // 1A. Detect Arrow (straight shaft + sharp hook barb near tip) BEFORE plain LINE
        if resampled.len() >= 10 && bbox_diag >= 24.0 {
            let split_idx = ((resampled.len() as f64) * 0.65).floor() as usize;
            let split_idx = split_idx.clamp(5, resampled.len() - 3);
            let shaft = &resampled[..split_idx];
            let sn = shaft.len() as f64;
            let smx = shaft.iter().map(|p| p.0).sum::<f64>() / sn;
            let smy = shaft.iter().map(|p| p.1).sum::<f64>() / sn;
            let mut ssxx = 0.0_f64;
            let mut ssyy = 0.0_f64;
            let mut ssxy = 0.0_f64;
            for &(x, y) in shaft {
                let dx = x - smx;
                let dy = y - smy;
                ssxx += dx * dx;
                ssyy += dy * dy;
                ssxy += dx * dy;
            }
            ssxx /= sn;
            ssyy /= sn;
            ssxy /= sn;
            let strace = ssxx + ssyy;
            let sdisc = ((ssxx - ssyy).powi(2) + 4.0 * ssxy * ssxy).sqrt();
            let sl1 = 0.5 * (strace + sdisc);
            let sl2 = (0.5 * (strace - sdisc)).max(0.0);
            let shaft_ratio = sl2 / sl1.max(1e-9);
            if shaft_ratio < 0.045 {
                let (mut sux, mut suy) = if ssxy.abs() > 1e-12 {
                    let vx = sl1 - ssyy;
                    let vy = ssxy;
                    let nrm = vx.hypot(vy).max(1e-12);
                    (vx / nrm, vy / nrm)
                } else if ssxx >= ssyy {
                    (1.0, 0.0)
                } else {
                    (0.0, 1.0)
                };
                let p0 = resampled[0];
                if (resampled[split_idx - 1].0 - p0.0) * sux
                    + (resampled[split_idx - 1].1 - p0.1) * suy
                    < 0.0
                {
                    sux = -sux;
                    suy = -suy;
                }
                let aprojs: Vec<f64> = resampled
                    .iter()
                    .map(|p| (p.0 - p0.0) * sux + (p.1 - p0.1) * suy)
                    .collect();
                let mut tip_idx = 0usize;
                let mut tip_proj = f64::NEG_INFINITY;
                for (i, &val) in aprojs.iter().enumerate() {
                    if val > tip_proj {
                        tip_proj = val;
                        tip_idx = i;
                    }
                }
                if tip_idx >= 3 && tip_idx + 2 < resampled.len() {
                    let min_before = aprojs[..=tip_idx]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    let shaft_len = tip_proj - min_before;
                    let tip_pt = resampled[tip_idx];
                    let max_perp = resampled[tip_idx..]
                        .iter()
                        .map(|p| ((p.0 - tip_pt.0) * (-suy) + (p.1 - tip_pt.1) * sux).abs())
                        .fold(0.0_f64, f64::max);
                    let min_tail_proj = aprojs[tip_idx..]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    let backtrack = tip_proj - min_tail_proj;
                    if shaft_len >= 22.0
                        && max_perp >= (0.06 * shaft_len).max(6.0)
                        && max_perp <= 0.45 * shaft_len
                        && backtrack >= (0.05 * shaft_len).max(5.0)
                    {
                        let raw_ang = (tip_pt.1 - p0.1).atan2(tip_pt.0 - p0.0).to_degrees();
                        let mut snap_ang = raw_ang;
                        for &anchor in
                            &[-180.0_f64, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
                        {
                            if (raw_ang - anchor).abs() <= 6.0 {
                                snap_ang = anchor;
                                break;
                            }
                        }
                        let rad = snap_ang.to_radians();
                        let len = (tip_pt.0 - p0.0).hypot(tip_pt.1 - p0.1);
                        let mid = (0.5 * (p0.0 + tip_pt.0), 0.5 * (p0.1 + tip_pt.1));
                        let (ux, uy) = (rad.cos(), rad.sin());
                        let s_snap = (mid.0 - 0.5 * len * ux, mid.1 - 0.5 * len * uy);
                        let e_snap = (mid.0 + 0.5 * len * ux, mid.1 + 0.5 * len * uy);
                        let wing = (0.14 * len).clamp(14.0, 34.0);
                        let barb = 28.0_f64.to_radians();
                        let (ca, sa) = (barb.cos(), barb.sin());
                        let (bx, by) = (-ux, -uy);
                        let left_w = (
                            e_snap.0 + wing * (bx * ca - by * sa),
                            e_snap.1 + wing * (bx * sa + by * ca),
                        );
                        let right_w = (
                            e_snap.0 + wing * (bx * ca + by * sa),
                            e_snap.1 + wing * (-bx * sa + by * ca),
                        );
                        let arrow_conf = (0.96 - shaft_ratio * 2.5).clamp(0.80, 0.97);
                        if arrow_conf >= confidence_gate {
                            return (
                                "ARROW".to_string(),
                                arrow_conf,
                                false,
                                vec![s_snap, e_snap, left_w, e_snap, right_w],
                            );
                        }
                    }
                }
            }
        }

        let simplified = rdp_simplify_rs(clean.clone(), (bbox_diag * 0.025).max(3.0));
        if eig_ratio < 0.035 && simplified.len() <= 6 {
            let (mut ux, mut uy) = if sxy.abs() > 1e-12 {
                let vx = l1 - syy;
                let vy = sxy;
                let nrm = vx.hypot(vy).max(1e-12);
                (vx / nrm, vy / nrm)
            } else if sxx >= syy {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            };
            let p0_proj = (clean[0].0 - mx) * ux + (clean[0].1 - my) * uy;
            let p1_proj = (clean.last().unwrap().0 - mx) * ux + (clean.last().unwrap().1 - my) * uy;
            if p1_proj < p0_proj {
                ux = -ux;
                uy = -uy;
            }
            let projs: Vec<f64> = resampled
                .iter()
                .map(|p| (p.0 - mx) * ux + (p.1 - my) * uy)
                .collect();
            let t_min = projs.iter().copied().fold(f64::INFINITY, f64::min);
            let t_max = projs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let line_len = (t_max - t_min).max(1e-6);
            let mut max_backtrack = 0.0_f64;
            let mut running_max = projs[0];
            for &tp in &projs[1..] {
                if tp > running_max {
                    running_max = tp;
                } else {
                    max_backtrack = max_backtrack.max(running_max - tp);
                }
            }
            if max_backtrack <= 0.08 * line_len {
                let raw_ang = uy.atan2(ux).to_degrees();
                let mut snap_ang = raw_ang;
                for &anchor in &[-180.0_f64, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0] {
                    if (raw_ang - anchor).abs() <= 6.0 {
                        snap_ang = anchor;
                        break;
                    }
                }
                let mid_proj = 0.5 * (t_min + t_max);
                let mid_x = mx + mid_proj * ux;
                let mid_y = my + mid_proj * uy;
                let rad = snap_ang.to_radians();
                let sx = rad.cos();
                let sy = rad.sin();
                let p_start = (mid_x - 0.5 * line_len * sx, mid_y - 0.5 * line_len * sy);
                let p_end = (mid_x + 0.5 * line_len * sx, mid_y + 0.5 * line_len * sy);
                let line_conf = (1.0 - (eig_ratio / 0.035) * 0.18).clamp(0.80, 0.99);
                if line_conf >= confidence_gate {
                    return ("LINE".to_string(), line_conf, false, vec![p_start, p_end]);
                }
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
        if radial_cv < 0.088 && circularity >= 0.855 {
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

    // Closed Polygon Recognition in Rust: TRIANGLE, SQUARE, RECTANGLE
    let mut closed_loop = clean.clone();
    if let (Some(&first), Some(&last)) = (closed_loop.first(), closed_loop.last()) {
        if (first.0 - last.0).hypot(first.1 - last.1) > 1.0 {
            closed_loop.push(first);
        }
    }
    let mut best_poly: Option<Vec<(f64, f64)>> = None;
    let mut best_rms = f64::INFINITY;
    for &frac in &[0.032_f64, 0.045, 0.058, 0.072] {
        let simp = rdp_simplify_rs(closed_loop.clone(), (peri * frac).max(5.0));
        let mut verts = simp;
        if verts.len() >= 2 {
            let f = verts[0];
            let l = *verts.last().unwrap();
            if (f.0 - l.0).hypot(f.1 - l.1) < (bbox_diag * 0.12).max(8.0) {
                verts.pop();
            }
        }
        // Prune near-collinear or tiny edges
        let mut changed = true;
        while changed && verts.len() > 3 {
            changed = false;
            for i in 0..verts.len() {
                let j = (i + 1) % verts.len();
                if (verts[i].0 - verts[j].0).hypot(verts[i].1 - verts[j].1)
                    < (bbox_diag * 0.08).max(8.0)
                {
                    verts[i] = (
                        0.5 * (verts[i].0 + verts[j].0),
                        0.5 * (verts[i].1 + verts[j].1),
                    );
                    verts.remove(j);
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }
            let vn = verts.len();
            for i in 0..vn {
                let prev = verts[(i + vn - 1) % vn];
                let cur = verts[i];
                let nxt = verts[(i + 1) % vn];
                let ang = vec_angle_deg(
                    (prev.0 - cur.0, prev.1 - cur.1, 0.0),
                    (nxt.0 - cur.0, nxt.1 - cur.1, 0.0),
                );
                if ang > 160.0 {
                    verts.remove(i);
                    changed = true;
                    break;
                }
            }
        }
        if verts.len() == 3 || verts.len() == 4 {
            // Order CCW around centroid
            let vcx = verts.iter().map(|p| p.0).sum::<f64>() / (verts.len() as f64);
            let vcy = verts.iter().map(|p| p.1).sum::<f64>() / (verts.len() as f64);
            verts.sort_by(|a, b| {
                let aa = (a.1 - vcy).atan2(a.0 - vcx);
                let ba = (b.1 - vcy).atan2(b.0 - vcx);
                aa.partial_cmp(&ba).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sum_sq = 0.0_f64;
            for &p in &resampled {
                let mut md = f64::INFINITY;
                for i in 0..verts.len() {
                    let d = perpendicular_distance(p, verts[i], verts[(i + 1) % verts.len()]);
                    md = md.min(d);
                }
                sum_sq += md * md;
            }
            let rms = (sum_sq / (resampled.len() as f64)).sqrt() / bbox_diag.max(1.0);
            if rms < best_rms {
                best_rms = rms;
                best_poly = Some(verts);
            }
        }
    }

    if let Some(verts) = best_poly {
        if best_rms <= 0.055 {
            if verts.len() == 3 {
                let conf = (0.97 - best_rms * 2.8).clamp(0.80, 0.97);
                if conf >= confidence_gate {
                    let mut out = verts.clone();
                    out.push(verts[0]);
                    return ("TRIANGLE".to_string(), conf, true, out);
                }
            } else if verts.len() == 4 {
                let mut angles = [0.0_f64; 4];
                let mut edges = [0.0_f64; 4];
                for i in 0..4 {
                    let prev = verts[(i + 3) % 4];
                    let cur = verts[i];
                    let nxt = verts[(i + 1) % 4];
                    angles[i] = vec_angle_deg(
                        (prev.0 - cur.0, prev.1 - cur.1, 0.0),
                        (nxt.0 - cur.0, nxt.1 - cur.1, 0.0),
                    );
                    edges[i] = (nxt.0 - cur.0).hypot(nxt.1 - cur.1);
                }
                let right_cnt = angles.iter().filter(|&&a| (78.0..=102.0).contains(&a)).count();
                if right_cnt >= 3 {
                    let min_e = edges.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_e = edges.iter().copied().fold(0.0_f64, f64::max);
                    let uniformity = min_e / max_e.max(1e-6);
                    // Reconstruct exact orthogonal rectangle/square along edge[0] orientation
                    let e0 = (verts[1].0 - verts[0].0, verts[1].1 - verts[0].1);
                    let mut ang_deg = e0.1.atan2(e0.0).to_degrees();
                    for &anchor in
                        &[-180.0_f64, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
                    {
                        if (ang_deg - anchor).abs() <= 6.0 {
                            ang_deg = anchor;
                            break;
                        }
                    }
                    let rad = ang_deg.to_radians();
                    let (ux, uy) = (rad.cos(), rad.sin());
                    let (vx, vy) = (-uy, ux);
                    let qcx = verts.iter().map(|p| p.0).sum::<f64>() * 0.25;
                    let qcy = verts.iter().map(|p| p.1).sum::<f64>() * 0.25;
                    let mut w_avg = 0.5 * (edges[0] + edges[2]);
                    let mut h_avg = 0.5 * (edges[1] + edges[3]);
                    let is_square = uniformity >= 0.82
                        && (w_avg.min(h_avg) / w_avg.max(h_avg).max(1e-6)) >= 0.83;
                    if is_square {
                        let s = 0.5 * (w_avg + h_avg);
                        w_avg = s;
                        h_avg = s;
                    }
                    let (hw, hh) = (0.5 * w_avg, 0.5 * h_avg);
                    let c0 = (qcx - hw * ux - hh * vx, qcy - hw * uy - hh * vy);
                    let c1 = (qcx + hw * ux - hh * vx, qcy + hw * uy - hh * vy);
                    let c2 = (qcx + hw * ux + hh * vx, qcy + hw * uy + hh * vy);
                    let c3 = (qcx - hw * ux + hh * vx, qcy - hw * uy + hh * vy);
                    let conf = (0.96 - best_rms * 2.5).clamp(0.80, 0.96);
                    if conf >= confidence_gate {
                        let label = if is_square { "SQUARE" } else { "RECTANGLE" };
                        return (label.to_string(), conf, true, vec![c0, c1, c2, c3, c0]);
                    }
                }
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

// =============================================================================
// KALMAN 2D VELOCITY PREDICTOR, 1-HAND & 2-HAND GESTURE FSM, & HIT-TESTING ENGINE
// =============================================================================

#[pyclass]
pub struct RustKalmanVelocityPredictor2D {
    q: f64,
    r: f64,
    #[pyo3(get)]
    pub initialized: bool,
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub vx: f64,
    #[pyo3(get)]
    pub vy: f64,
    p_pos: f64,
    p_vel: f64,
    last_t: Option<f64>,
    last_meas: Option<(f64, f64)>,
}

#[pymethods]
impl RustKalmanVelocityPredictor2D {
    #[new]
    #[pyo3(signature = (process_noise = 0.04, meas_noise = 0.0015))]
    fn new(process_noise: f64, meas_noise: f64) -> Self {
        Self {
            q: process_noise,
            r: meas_noise,
            initialized: false,
            x: 0.5,
            y: 0.5,
            vx: 0.0,
            vy: 0.0,
            p_pos: 1.0,
            p_vel: 1.0,
            last_t: None,
            last_meas: None,
        }
    }

    fn reset(&mut self) {
        self.initialized = false;
        self.vx = 0.0;
        self.vy = 0.0;
        self.p_pos = 1.0;
        self.p_vel = 1.0;
        self.last_t = None;
        self.last_meas = None;
    }

    fn update(&mut self, meas_x: f64, meas_y: f64, t_sec: f64) -> (f64, f64) {
        if !self.initialized || self.last_t.is_none() {
            self.initialized = true;
            self.x = meas_x;
            self.y = meas_y;
            self.vx = 0.0;
            self.vy = 0.0;
            self.p_pos = 0.02;
            self.p_vel = 0.1;
            self.last_t = Some(t_sec);
            self.last_meas = Some((meas_x, meas_y));
            return (meas_x, meas_y);
        }
        let dt = (t_sec - self.last_t.unwrap()).clamp(1e-3, 0.25);
        self.last_t = Some(t_sec);
        let (prev_mx, prev_my) = self.last_meas.unwrap_or((self.x, self.y));
        self.last_meas = Some((meas_x, meas_y));

        let px = self.x + self.vx * dt;
        let py = self.y + self.vy * dt;
        self.p_pos = self.p_pos + self.p_vel * (dt * dt) + self.q * dt;
        self.p_vel += self.q;

        let k_pos = self.p_pos / (self.p_pos + self.r).max(1e-6);
        self.x = px + k_pos * (meas_x - px);
        self.y = py + k_pos * (meas_y - py);
        let raw_vx = (meas_x - prev_mx) / dt;
        let raw_vy = (meas_y - prev_my) / dt;
        self.vx = (0.35 * self.vx + 0.65 * raw_vx).clamp(-2.5, 2.5);
        self.vy = (0.35 * self.vy + 0.65 * raw_vy).clamp(-2.5, 2.5);
        self.p_pos = (1.0 - k_pos) * self.p_pos;

        (self.x.clamp(0.0, 1.0), self.y.clamp(0.0, 1.0))
    }

    #[pyo3(signature = (t_sec, damping = 0.86))]
    fn predict_dropout(&mut self, t_sec: f64, damping: f64) -> Option<(f64, f64)> {
        if !self.initialized {
            return None;
        }
        let last_t = self.last_t?;
        let dt = (t_sec - last_t).clamp(0.0, 0.12);
        self.last_t = Some(t_sec);
        self.x = (self.x + self.vx * dt).clamp(0.0, 1.0);
        self.y = (self.y + self.vy * dt).clamp(0.0, 1.0);
        self.vx *= damping;
        self.vy *= damping;
        Some((self.x, self.y))
    }
}

#[pyclass]
pub struct RustGestureStateMachine {
    min_hold_frames: usize,
    min_hold_sec: f64,
    grace_period_sec: f64,
    frame_w: usize,
    frame_h: usize,
    deadzone_px: f64,
    voting_window_size: usize,
    voting_min_votes: usize,
    pinch_enter_ratio: f64,
    pinch_exit_ratio: f64,
    palm_facing_min_z: f64,
    cursor_filter: RustOneEuroFilter2D,
    thumb_tip_filter: RustOneEuroFilter3D,
    index_tip_filter: RustOneEuroFilter3D,
    pinch_filter: RustOneEuroFilter1D,
    kalman: RustKalmanVelocityPredictor2D,
    fsm_state: String,
    active_gesture: String,
    candidate_gesture: String,
    candidate_count: usize,
    candidate_since_sec: Option<f64>,
    last_seen_sec: Option<f64>,
    last_cursor: Option<(f64, f64)>,
    deadzone_cursor: Option<(f64, f64)>,
    prev_fingers: [bool; 5],
    vote_window: VecDeque<String>,
    raw_first_seen: HashMap<String, f64>,
}

#[pymethods]
impl RustGestureStateMachine {
    #[new]
    #[pyo3(signature = (
        min_hold_frames = 3,
        min_hold_sec = 0.080,
        grace_period_sec = 0.160,
        min_cutoff = 1.45,
        beta = 0.020,
        d_cutoff = 1.0,
        frame_w = 640,
        frame_h = 480,
        deadzone_px = 2.0,
        voting_window_size = 7,
        voting_min_votes = 3,
        pinch_enter_ratio = 0.28,
        pinch_exit_ratio = 0.42,
        palm_facing_min_z = 0.20
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_hold_frames: usize,
        min_hold_sec: f64,
        grace_period_sec: f64,
        min_cutoff: f64,
        beta: f64,
        d_cutoff: f64,
        frame_w: usize,
        frame_h: usize,
        deadzone_px: f64,
        voting_window_size: usize,
        voting_min_votes: usize,
        pinch_enter_ratio: f64,
        pinch_exit_ratio: f64,
        palm_facing_min_z: f64,
    ) -> Self {
        let win = voting_window_size.max(3);
        Self {
            min_hold_frames: min_hold_frames.max(1),
            min_hold_sec: min_hold_sec.max(0.050),
            grace_period_sec: grace_period_sec.max(0.080),
            frame_w: frame_w.max(1),
            frame_h: frame_h.max(1),
            deadzone_px: deadzone_px.max(0.0),
            voting_window_size: win,
            voting_min_votes: voting_min_votes.clamp(2, win),
            pinch_enter_ratio,
            pinch_exit_ratio,
            palm_facing_min_z,
            cursor_filter: RustOneEuroFilter2D::new(min_cutoff, beta, d_cutoff),
            thumb_tip_filter: RustOneEuroFilter3D::new(min_cutoff, beta, d_cutoff),
            index_tip_filter: RustOneEuroFilter3D::new(min_cutoff, beta, d_cutoff),
            pinch_filter: RustOneEuroFilter1D::new(min_cutoff, beta, d_cutoff),
            kalman: RustKalmanVelocityPredictor2D::new(0.04, 0.0015),
            fsm_state: "IDLE".to_string(),
            active_gesture: "IDLE".to_string(),
            candidate_gesture: "IDLE".to_string(),
            candidate_count: 0,
            candidate_since_sec: None,
            last_seen_sec: None,
            last_cursor: None,
            deadzone_cursor: None,
            prev_fingers: [false; 5],
            vote_window: VecDeque::with_capacity(win + 1),
            raw_first_seen: HashMap::new(),
        }
    }

    fn set_frame_size(&mut self, w: usize, h: usize) {
        self.frame_w = w.max(1);
        self.frame_h = h.max(1);
    }

    fn reset(&mut self) {
        self.fsm_state = "IDLE".to_string();
        self.active_gesture = "IDLE".to_string();
        self.candidate_gesture = "IDLE".to_string();
        self.candidate_count = 0;
        self.candidate_since_sec = None;
        self.last_seen_sec = None;
        self.last_cursor = None;
        self.deadzone_cursor = None;
        self.prev_fingers = [false; 5];
        self.vote_window.clear();
        self.raw_first_seen.clear();
        self.cursor_filter.reset();
        self.thumb_tip_filter.reset();
        self.index_tip_filter.reset();
        self.pinch_filter.reset();
        self.kalman.reset();
    }

    #[pyo3(signature = (landmarks, t_sec, handedness = "right"))]
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        landmarks: Option<Vec<Vec<f64>>>,
        t_sec: f64,
        handedness: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        if let Some(mut lm) = landmarks {
            if lm.len() >= 21 && !lm.iter().any(|p| p.len() < 3) {
                self.last_seen_sec = Some(t_sec);
                let t_filt = self.thumb_tip_filter.filter(
                    lm[THUMB_TIP][0],
                    lm[THUMB_TIP][1],
                    lm[THUMB_TIP][2],
                    t_sec,
                );
                let i_filt = self.index_tip_filter.filter(
                    lm[INDEX_TIP][0],
                    lm[INDEX_TIP][1],
                    lm[INDEX_TIP][2],
                    t_sec,
                );
                lm[THUMB_TIP] = vec![t_filt.0, t_filt.1, t_filt.2];
                lm[INDEX_TIP] = vec![i_filt.0, i_filt.1, i_filt.2];

                let was_pinched = self.active_gesture == "CLICK";
                let palm_scale = dist3(pt3(&lm, WRIST), pt3(&lm, MIDDLE_MCP)).max(1e-4);
                let raw_pr = dist3(pt3(&lm, THUMB_TIP), pt3(&lm, INDEX_TIP)) / palm_scale;
                let smoothed_pr = self.pinch_filter.filter(raw_pr, t_sec).clamp(0.01, 3.0);

                let cls = classify_hand_3d_rs(
                    py,
                    lm,
                    Some(self.prev_fingers.to_vec()),
                    was_pinched,
                    self.pinch_enter_ratio,
                    self.pinch_exit_ratio,
                    handedness,
                    self.palm_facing_min_z,
                )?;

                let mut raw_gesture: String = cls.get_item("gesture")?.unwrap().extract()?;
                let mut pose_tag: String = cls.get_item("pose_tag")?.unwrap().extract()?;
                let raw_cursor: (f64, f64) = cls.get_item("cursor_norm")?.unwrap().extract()?;
                let fingers: Vec<bool> = cls.get_item("fingers")?.unwrap().extract()?;
                if fingers.len() == 5 {
                    self.prev_fingers = [fingers[0], fingers[1], fingers[2], fingers[3], fingers[4]];
                }

                let pinch_thresh = if was_pinched {
                    self.pinch_exit_ratio
                } else {
                    self.pinch_enter_ratio
                };
                if smoothed_pr < pinch_thresh
                    && pose_tag != "OPEN_PALM"
                    && pose_tag != "FIST"
                    && pose_tag != "THREE_FINGERS"
                    && pose_tag != "QUESTION_HOLD"
                {
                    raw_gesture = "CLICK".to_string();
                    pose_tag = "PINCH".to_string();
                    cls.set_item("gesture", "CLICK")?;
                    cls.set_item("pose_tag", "PINCH")?;
                }
                cls.set_item("pinch_ratio", round4(smoothed_pr))?;

                if self.vote_window.len() >= self.voting_window_size {
                    self.vote_window.pop_front();
                }
                self.vote_window.push_back(raw_gesture.clone());
                if raw_gesture != self.active_gesture {
                    self.raw_first_seen.entry(raw_gesture.clone()).or_insert(t_sec);
                }

                let mut counts: HashMap<String, usize> = HashMap::new();
                for g in &self.vote_window {
                    *counts.entry(g.clone()).or_insert(0) += 1;
                }
                let mut voted_gesture = raw_gesture.clone();
                let mut voted_count = 0usize;
                for (k, &v) in &counts {
                    if v > voted_count || (v == voted_count && k == &raw_gesture) {
                        voted_count = v;
                        voted_gesture = k.clone();
                    }
                }

                let filt_2d = self.cursor_filter.filter(raw_cursor.0, raw_cursor.1, t_sec);
                let kal_2d = self.kalman.update(filt_2d.0, filt_2d.1, t_sec);

                // Deadzone suppression
                let px_thresh = self.deadzone_px / (self.frame_w.max(self.frame_h) as f64);
                let palm_thresh = (palm_scale * 0.022).clamp(0.002, 0.009);
                let thresh = px_thresh.max(palm_thresh);
                let smoothed_cursor = match self.deadzone_cursor {
                    Some(prev) if (kal_2d.0 - prev.0).hypot(kal_2d.1 - prev.1) < thresh => prev,
                    _ => {
                        self.deadzone_cursor = Some(kal_2d);
                        kal_2d
                    }
                };
                self.last_cursor = Some(smoothed_cursor);

                if voted_gesture == self.active_gesture {
                    self.candidate_gesture = voted_gesture.clone();
                    self.candidate_count = voted_count;
                    self.candidate_since_sec = None;
                    self.fsm_state = if self.active_gesture == "IDLE" {
                        "IDLE".to_string()
                    } else {
                        "ACTIVE".to_string()
                    };
                } else {
                    if voted_gesture == self.candidate_gesture {
                        self.candidate_count += 1;
                    } else {
                        self.candidate_gesture = voted_gesture.clone();
                        self.candidate_count = voted_count;
                        self.candidate_since_sec =
                            Some(*self.raw_first_seen.get(&voted_gesture).unwrap_or(&t_sec));
                    }
                    let since = *self
                        .raw_first_seen
                        .get(&voted_gesture)
                        .unwrap_or(&self.candidate_since_sec.unwrap_or(t_sec));
                    let elapsed = t_sec - since;
                    self.fsm_state = "CANDIDATE".to_string();
                    let frames_ok = if self.min_hold_frames <= self.voting_window_size {
                        voted_count >= self.min_hold_frames
                    } else {
                        self.candidate_count >= self.min_hold_frames
                    };
                    if voted_count >= self.voting_min_votes
                        && frames_ok
                        && elapsed >= self.min_hold_sec
                    {
                        self.active_gesture = voted_gesture.clone();
                        self.candidate_count = 0;
                        self.candidate_since_sec = None;
                        self.raw_first_seen.clear();
                        self.fsm_state = if voted_gesture == "IDLE" {
                            "IDLE".to_string()
                        } else {
                            "ACTIVE".to_string()
                        };
                    }
                }

                let counts_py = PyDict::new(py);
                for (k, v) in counts {
                    counts_py.set_item(k, v)?;
                }
                out.set_item("fsm_state", self.fsm_state.as_str())?;
                out.set_item("active_gesture", self.active_gesture.as_str())?;
                out.set_item("candidate_gesture", self.candidate_gesture.as_str())?;
                out.set_item("cursor_norm", smoothed_cursor)?;
                out.set_item("hold_frames", self.candidate_count)?;
                out.set_item("in_grace_period", false)?;
                out.set_item("kalman_predicted", false)?;
                out.set_item("voting_counts", counts_py)?;
                out.set_item("pinch_ratio", round4(smoothed_pr))?;
                out.set_item("pose_tag", pose_tag)?;
                out.set_item("classify_meta", cls)?;
                return Ok(out);
            }
        }

        if let Some(last_seen) = self.last_seen_sec {
            if self.active_gesture != "IDLE" && (t_sec - last_seen) <= self.grace_period_sec {
                self.fsm_state = "GRACE_PERIOD".to_string();
                let predicted = self.kalman.predict_dropout(t_sec, 0.86);
                let pred_cursor = predicted.or(self.last_cursor);
                self.last_cursor = pred_cursor;
                let counts_py = PyDict::new(py);
                let mut counts: HashMap<String, usize> = HashMap::new();
                for g in &self.vote_window {
                    *counts.entry(g.clone()).or_insert(0) += 1;
                }
                for (k, v) in counts {
                    counts_py.set_item(k, v)?;
                }
                out.set_item("fsm_state", "GRACE_PERIOD")?;
                out.set_item("active_gesture", self.active_gesture.as_str())?;
                out.set_item("candidate_gesture", self.candidate_gesture.as_str())?;
                out.set_item("cursor_norm", pred_cursor)?;
                out.set_item("hold_frames", self.candidate_count)?;
                out.set_item("in_grace_period", true)?;
                out.set_item("kalman_predicted", predicted.is_some())?;
                out.set_item("voting_counts", counts_py)?;
                return Ok(out);
            }
        }

        self.reset();
        out.set_item("fsm_state", "IDLE")?;
        out.set_item("active_gesture", "IDLE")?;
        out.set_item("candidate_gesture", "IDLE")?;
        out.set_item("cursor_norm", py.None())?;
        out.set_item("hold_frames", 0usize)?;
        out.set_item("in_grace_period", false)?;
        out.set_item("kalman_predicted", false)?;
        out.set_item("voting_counts", PyDict::new(py))?;
        Ok(out)
    }
}

#[pyclass]
pub struct RustDualHandStateMachine {
    prev_pinch_dist: Option<f64>,
    smoothed_dist: Option<f64>,
    prev_pan_center: Option<(f64, f64)>,
    prev_fist_pan_xy: Option<(f64, f64)>,
    fist_pan_until: f64,
    offhand_cand: String,
    offhand_cand_count: usize,
    last_offhand_cmd_ts: f64,
}

#[pymethods]
impl RustDualHandStateMachine {
    #[new]
    fn new() -> Self {
        Self {
            prev_pinch_dist: None,
            smoothed_dist: None,
            prev_pan_center: None,
            prev_fist_pan_xy: None,
            fist_pan_until: 0.0,
            offhand_cand: String::new(),
            offhand_cand_count: 0,
            last_offhand_cmd_ts: -1e9,
        }
    }

    fn reset(&mut self) {
        self.prev_pinch_dist = None;
        self.smoothed_dist = None;
        self.prev_pan_center = None;
        self.prev_fist_pan_xy = None;
        self.fist_pan_until = 0.0;
        self.offhand_cand.clear();
        self.offhand_cand_count = 0;
    }

    #[pyo3(signature = (p_pose, p_pr, p_xy, s_pose, s_pr, s_xy, now_sec))]
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        p_pose: &str,
        p_pr: f64,
        p_xy: (f64, f64),
        s_pose: &str,
        s_pr: f64,
        s_xy: (f64, f64),
        now_sec: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let p_up = p_pose.to_ascii_uppercase();
        let s_up = s_pose.to_ascii_uppercase();

        // 1. Dual Pinch Zoom with smooth EMA distance filter
        let both_pinch = (p_up == "PINCH" || p_pr < 0.33) && (s_up == "PINCH" || s_pr < 0.33);
        if both_pinch {
            self.prev_pan_center = None;
            self.prev_fist_pan_xy = None;
            self.offhand_cand.clear();
            self.offhand_cand_count = 0;

            let raw_dist = (p_xy.0 - s_xy.0).hypot(p_xy.1 - s_xy.1).max(0.03);
            let dist = match self.smoothed_dist {
                Some(sd) => 0.45 * sd + 0.55 * raw_dist,
                None => raw_dist,
            };
            self.smoothed_dist = Some(dist);
            let mid = (0.5 * (p_xy.0 + s_xy.0), 0.5 * (p_xy.1 + s_xy.1));
            let mut zf = 1.0_f64;
            if let Some(prev_d) = self.prev_pinch_dist {
                if (dist - prev_d).abs() > 0.004 {
                    zf = (dist / prev_d.max(0.03)).clamp(0.84, 1.18);
                }
            }
            self.prev_pinch_dist = Some(dist);

            out.set_item("two_hand_mode", "DUAL_PINCH_ZOOM")?;
            out.set_item("modifier_mode", "NONE")?;
            out.set_item("zoom_factor", round4(zf))?;
            out.set_item("zoom_center_norm", mid)?;
            out.set_item("pan_delta_norm", (0.0_f64, 0.0_f64))?;
            out.set_item("offhand_command", py.None())?;
            out.set_item("suppress_primary_drawing", true)?;
            out.set_item("status_label_vi", "2 Tay: Thu/Phóng bảng (Dual Pinch)")?;
            return Ok(out);
        }
        self.prev_pinch_dist = None;
        self.smoothed_dist = None;

        // 2. Dual Open Palm Pan OR Dual Fist Pan (both hands moving in parallel)
        if (p_up == "OPEN_PALM" && s_up == "OPEN_PALM") || (p_up == "FIST" && s_up == "FIST") {
            self.prev_fist_pan_xy = None;
            self.offhand_cand.clear();
            self.offhand_cand_count = 0;
            let mid = (0.5 * (p_xy.0 + s_xy.0), 0.5 * (p_xy.1 + s_xy.1));
            let mut dx = 0.0_f64;
            let mut dy = 0.0_f64;
            if let Some(prev_mid) = self.prev_pan_center {
                let rdx = mid.0 - prev_mid.0;
                let rdy = mid.1 - prev_mid.1;
                if rdx.hypot(rdy) > 0.0025 {
                    dx = rdx.clamp(-0.15, 0.15);
                    dy = rdy.clamp(-0.15, 0.15);
                }
            }
            self.prev_pan_center = Some(mid);
            let lbl = if p_up == "FIST" {
                "2 Tay: Kéo trượt bảng (2 Tay Nắm)"
            } else {
                "2 Tay: Kéo trượt bảng (2 Tay Xòe)"
            };
            out.set_item("two_hand_mode", "DUAL_PALM_PAN")?;
            out.set_item("modifier_mode", "NONE")?;
            out.set_item("zoom_factor", 1.0_f64)?;
            out.set_item("zoom_center_norm", py.None())?;
            out.set_item(
                "pan_delta_norm",
                (
                    (dx * 100_000.0).round() / 100_000.0,
                    (dy * 100_000.0).round() / 100_000.0,
                ),
            )?;
            out.set_item("offhand_command", py.None())?;
            out.set_item("suppress_primary_drawing", true)?;
            out.set_item("status_label_vi", lbl)?;
            return Ok(out);
        }
        self.prev_pan_center = None;

        // 2A. Off-Hand Fist + Primary Hand Move (non-pinching) -> Dual Pan
        let p_is_drawing = p_up == "PINCH" || p_pr < 0.34;
        if s_up == "FIST" && !p_is_drawing {
            if let Some(prev_xy) = self.prev_fist_pan_xy {
                let fdx = p_xy.0 - prev_xy.0;
                let fdy = p_xy.1 - prev_xy.1;
                let fstep = fdx.hypot(fdy);
                let is_moving =
                    fstep > 0.010 || (now_sec <= self.fist_pan_until && fstep > 0.002);
                if is_moving {
                    self.prev_fist_pan_xy = Some(p_xy);
                    self.fist_pan_until = now_sec + 0.28;
                    self.offhand_cand.clear();
                    self.offhand_cand_count = 0;
                    out.set_item("two_hand_mode", "DUAL_PALM_PAN")?;
                    out.set_item("modifier_mode", "NONE")?;
                    out.set_item("zoom_factor", 1.0_f64)?;
                    out.set_item("zoom_center_norm", py.None())?;
                    out.set_item(
                        "pan_delta_norm",
                        (
                            (fdx.clamp(-0.15, 0.15) * 100_000.0).round() / 100_000.0,
                            (fdy.clamp(-0.15, 0.15) * 100_000.0).round() / 100_000.0,
                        ),
                    )?;
                    out.set_item("offhand_command", py.None())?;
                    out.set_item("suppress_primary_drawing", true)?;
                    out.set_item("status_label_vi", "2 Tay: Kéo trượt bảng (Tay phụ Nắm + Kéo)")?;
                    return Ok(out);
                }
            }
            self.prev_fist_pan_xy = Some(p_xy);
        } else {
            self.prev_fist_pan_xy = None;
            self.fist_pan_until = 0.0;
        }

        // 2B. Modifier draw lock
        if p_is_drawing && (s_up == "FIST" || s_up == "TWO_FINGERS") {
            self.offhand_cand.clear();
            self.offhand_cand_count = 0;
            let (mod_str, lbl) = if s_up == "FIST" {
                ("LINE_LOCK", "2 Tay: Khóa Nét Thẳng (Tay phụ Nắm + Tay chính Vẽ)")
            } else {
                (
                    "SHAPE_LOCK",
                    "2 Tay: Khóa Hình Học Chuẩn (Tay phụ 2 Ngón + Tay chính Vẽ)",
                )
            };
            out.set_item("two_hand_mode", "MODIFIER_DRAW")?;
            out.set_item("modifier_mode", mod_str)?;
            out.set_item("zoom_factor", 1.0_f64)?;
            out.set_item("zoom_center_norm", py.None())?;
            out.set_item("pan_delta_norm", (0.0_f64, 0.0_f64))?;
            out.set_item("offhand_command", py.None())?;
            out.set_item("suppress_primary_drawing", false)?;
            out.set_item("status_label_vi", lbl)?;
            return Ok(out);
        }

        // 3. Off-hand quick tool command
        let mapped_cmd = match s_up.as_str() {
            "FIST" => Some("TOGGLE_PEN_ERASER"),
            "TWO_FINGERS" => Some("CYCLE_COLOR"),
            "THREE_FINGERS" => Some("UNDO"),
            "PINCH" => Some("SET_PEN"),
            _ => None,
        };
        let mut triggered_cmd: Option<&'static str> = None;
        if let Some(cmd) = mapped_cmd {
            if self.offhand_cand == cmd {
                self.offhand_cand_count += 1;
            } else {
                self.offhand_cand = cmd.to_string();
                self.offhand_cand_count = 1;
            }
            if self.offhand_cand_count >= 2 && (now_sec - self.last_offhand_cmd_ts) >= 0.70 {
                self.last_offhand_cmd_ts = now_sec;
                triggered_cmd = Some(cmd);
                self.offhand_cand_count = 0;
            }
        } else {
            self.offhand_cand.clear();
            self.offhand_cand_count = 0;
        }

        let mode_str = if triggered_cmd.is_some() {
            "OFFHAND_COMMAND"
        } else {
            "NONE"
        };
        let status = match triggered_cmd {
            Some(c) => format!("2 Tay: Chính={} | Phụ={} -> {}", p_up, s_up, c),
            None => format!("2 Tay: Chính={} | Phụ={}", p_up, s_up),
        };
        out.set_item("two_hand_mode", mode_str)?;
        out.set_item("modifier_mode", "NONE")?;
        out.set_item("zoom_factor", 1.0_f64)?;
        out.set_item("zoom_center_norm", py.None())?;
        out.set_item("pan_delta_norm", (0.0_f64, 0.0_f64))?;
        out.set_item("offhand_command", triggered_cmd)?;
        out.set_item("suppress_primary_drawing", false)?;
        out.set_item("status_label_vi", status)?;
        Ok(out)
    }
}

#[inline]
fn world_to_local_rotated(
    px: f64,
    py: f64,
    cx: f64,
    cy: f64,
    rotation_deg: f64,
) -> (f64, f64) {
    if rotation_deg.abs() < 1e-4 {
        return (px, py);
    }
    let rad = (-rotation_deg).to_radians();
    let (c, s) = (rad.cos(), rad.sin());
    let dx = px - cx;
    let dy = py - cy;
    (cx + dx * c - dy * s, cy + dx * s + dy * c)
}

#[pyfunction]
#[pyo3(signature = (px, py, pts, radius = 10.0))]
fn hit_test_polyline_rs(px: f64, py: f64, pts: Vec<(f64, f64)>, radius: f64) -> bool {
    if pts.is_empty() {
        return false;
    }
    let r = radius.max(1.0);
    if pts.len() == 1 {
        return (px - pts[0].0).hypot(py - pts[0].1) <= r;
    }
    for w in pts.windows(2) {
        if perpendicular_distance((px, py), w[0], w[1]) <= r {
            return true;
        }
    }
    false
}

/// Fast O(N) Canvas Object hit-testing sorted by descending z_index.
/// Each object tuple: `(id, obj_type, x, y, w, h, rotation_deg, z_index, polyline_pts)`.
#[pyfunction]
#[pyo3(signature = (px, py, objects, tolerance = 10.0))]
fn hit_test_objects_rs(
    px: f64,
    py: f64,
    objects: Vec<(String, String, f64, f64, f64, f64, f64, i64, Vec<(f64, f64)>)>,
    tolerance: f64,
) -> Option<String> {
    let tol = tolerance.max(2.0);
    let mut order: Vec<usize> = (0..objects.len()).collect();
    order.sort_by(|&a, &b| objects[b].7.cmp(&objects[a].7).then_with(|| b.cmp(&a)));

    for idx in order {
        let (ref id, ref obj_type, x, y, w, h, rot_deg, _, ref pts) = objects[idx];
        let bw = w.max(4.0);
        let bh = h.max(4.0);
        let cx = x + 0.5 * bw;
        let cy = y + 0.5 * bh;
        let (lx, ly) = world_to_local_rotated(px, py, cx, cy, rot_deg);

        // Quick local AABB reject
        if lx < x - tol || lx > x + bw + tol || ly < y - tol || ly > y + bh + tol {
            continue;
        }

        if (obj_type == "stroke" || obj_type == "shape") && !pts.is_empty() {
            if hit_test_polyline_rs(lx, ly, pts.clone(), tol) {
                return Some(id.clone());
            }
            // For closed shapes, also allow clicking inside interior when reasonable
            if obj_type == "shape" && pts.len() >= 4 {
                let first = pts[0];
                let last = *pts.last().unwrap();
                if (first.0 - last.0).hypot(first.1 - last.1) <= 12.0 {
                    // Ray-casting point-in-polygon
                    let mut inside = false;
                    let mut j = pts.len() - 1;
                    for i in 0..pts.len() {
                        let (xi, yi) = pts[i];
                        let (xj, yj) = pts[j];
                        if ((yi > ly) != (yj > ly))
                            && (lx < (xj - xi) * (ly - yi) / ((yj - yi) + 1e-12) + xi)
                        {
                            inside = !inside;
                        }
                        j = i;
                    }
                    if inside {
                        return Some(id.clone());
                    }
                }
            }
        } else {
            return Some(id.clone());
        }
    }
    None
}

/// Fast marquee rectangle selection returning all object IDs whose rotated AABB intersects `(rx, ry, rw, rh)`.
/// Each object tuple: `(id, x, y, w, h, rotation_deg)`.
#[pyfunction]
fn marquee_select_objects_rs(
    rx: f64,
    ry: f64,
    rw: f64,
    rh: f64,
    objects: Vec<(String, f64, f64, f64, f64, f64)>,
) -> Vec<String> {
    let mx0 = rx.min(rx + rw);
    let my0 = ry.min(ry + rh);
    let mx1 = rx.max(rx + rw);
    let my1 = ry.max(ry + rh);
    let mut hits = Vec::new();
    for (id, x, y, w, h, rot_deg) in objects {
        let bw = w.max(2.0);
        let bh = h.max(2.0);
        let (ox0, oy0, ox1, oy1) = if rot_deg.abs() < 1e-4 {
            (x, y, x + bw, y + bh)
        } else {
            let cx = x + 0.5 * bw;
            let cy = y + 0.5 * bh;
            let rad = rot_deg.to_radians();
            let (c, s) = (rad.cos(), rad.sin());
            let corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)];
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for (px, py) in corners {
                let dx = px - cx;
                let dy = py - cy;
                let wx = cx + dx * c - dy * s;
                let wy = cy + dx * s + dy * c;
                min_x = min_x.min(wx);
                min_y = min_y.min(wy);
                max_x = max_x.max(wx);
                max_y = max_y.max(wy);
            }
            (min_x, min_y, max_x, max_y)
        };
        if ox1 >= mx0 && ox0 <= mx1 && oy1 >= my0 && oy0 <= my1 {
            hits.push(id);
        }
    }
    hits
}

// =============================================================================
// 2D QUADTREE SPATIAL INDEXING ENGINE (RustSpatialIndex)
// =============================================================================

#[derive(Clone, Debug)]
struct SpatialEntry {
    id: String,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    z_index: i64,
    order: u64,
    node_idx: usize,
    alive: bool,
}

#[derive(Clone, Debug)]
struct QuadNode {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    depth: usize,
    items: Vec<usize>,
    children: Option<[usize; 4]>,
}

impl QuadNode {
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64, depth: usize) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
            depth,
            items: Vec::new(),
            children: None,
        }
    }

    #[inline]
    fn child_quadrant(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> Option<usize> {
        if x0 < self.min_x || x1 > self.max_x || y0 < self.min_y || y1 > self.max_y {
            return None;
        }
        let mid_x = 0.5 * (self.min_x + self.max_x);
        let mid_y = 0.5 * (self.min_y + self.max_y);
        let col = if x1 <= mid_x {
            0
        } else if x0 >= mid_x {
            1
        } else {
            return None;
        };
        let row = if y1 <= mid_y {
            0
        } else if y0 >= mid_y {
            2
        } else {
            return None;
        };
        Some(row + col)
    }
}

#[inline]
fn normalize_bounds(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> (f64, f64, f64, f64) {
    let (x0, x1) = if min_x <= max_x {
        (min_x, max_x)
    } else {
        (max_x, min_x)
    };
    let (y0, y1) = if min_y <= max_y {
        (min_y, max_y)
    } else {
        (max_y, min_y)
    };
    (x0, y0, x1, y1)
}

#[pyclass]
pub struct RustSpatialIndex {
    default_bounds: (f64, f64, f64, f64),
    max_items: usize,
    max_depth: usize,
    nodes: Vec<QuadNode>,
    entries: Vec<SpatialEntry>,
    free_slots: Vec<usize>,
    id_to_slot: HashMap<String, usize>,
    next_order: u64,
}

impl RustSpatialIndex {
    fn reset_tree_with_bounds(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        self.nodes.clear();
        self.entries.clear();
        self.free_slots.clear();
        self.id_to_slot.clear();
        self.next_order = 0;
        self.nodes.push(QuadNode::new(min_x, min_y, max_x, max_y, 0));
    }

    fn split_node(&mut self, node_idx: usize) {
        let (min_x, min_y, max_x, max_y, depth) = {
            let n = &self.nodes[node_idx];
            if n.children.is_some() || n.depth >= self.max_depth {
                return;
            }
            (n.min_x, n.min_y, n.max_x, n.max_y, n.depth)
        };
        let mid_x = 0.5 * (min_x + max_x);
        let mid_y = 0.5 * (min_y + max_y);
        let child_depth = depth + 1;
        let base = self.nodes.len();
        self.nodes
            .push(QuadNode::new(min_x, min_y, mid_x, mid_y, child_depth));
        self.nodes
            .push(QuadNode::new(mid_x, min_y, max_x, mid_y, child_depth));
        self.nodes
            .push(QuadNode::new(min_x, mid_y, mid_x, max_y, child_depth));
        self.nodes
            .push(QuadNode::new(mid_x, mid_y, max_x, max_y, child_depth));
        let children = [base, base + 1, base + 2, base + 3];
        self.nodes[node_idx].children = Some(children);

        let old_items = std::mem::take(&mut self.nodes[node_idx].items);
        let mut keep_here = Vec::new();
        for slot in old_items {
            let (alive, x0, y0, x1, y1) = {
                let e = &self.entries[slot];
                (e.alive, e.min_x, e.min_y, e.max_x, e.max_y)
            };
            if !alive {
                continue;
            }
            if let Some(q) = self.nodes[node_idx].child_quadrant(x0, y0, x1, y1) {
                let child_idx = children[q];
                self.entries[slot].node_idx = child_idx;
                self.nodes[child_idx].items.push(slot);
            } else {
                self.entries[slot].node_idx = node_idx;
                keep_here.push(slot);
            }
        }
        self.nodes[node_idx].items = keep_here;
    }

    fn insert_slot(&mut self, slot: usize) {
        let (x0, y0, x1, y1) = {
            let e = &self.entries[slot];
            (e.min_x, e.min_y, e.max_x, e.max_y)
        };
        let mut cur = 0usize;
        loop {
            if let Some(children) = self.nodes[cur].children {
                if let Some(q) = self.nodes[cur].child_quadrant(x0, y0, x1, y1) {
                    cur = children[q];
                    continue;
                }
                self.entries[slot].node_idx = cur;
                self.nodes[cur].items.push(slot);
                return;
            }
            self.entries[slot].node_idx = cur;
            self.nodes[cur].items.push(slot);
            if self.nodes[cur].items.len() > self.max_items
                && self.nodes[cur].depth < self.max_depth
            {
                self.split_node(cur);
            }
            return;
        }
    }

    fn remove_by_key(&mut self, id: &str) -> bool {
        let Some(slot) = self.id_to_slot.remove(id) else {
            return false;
        };
        let node_idx = self.entries[slot].node_idx;
        self.entries[slot].alive = false;
        if let Some(node) = self.nodes.get_mut(node_idx) {
            if let Some(pos) = node.items.iter().position(|&idx| idx == slot) {
                node.items.swap_remove(pos);
            }
        }
        self.free_slots.push(slot);
        true
    }

    fn query_aabb_slots(&self, qx0: f64, qy0: f64, qx1: f64, qy1: f64) -> Vec<usize> {
        if self.nodes.is_empty() || self.id_to_slot.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut stack = Vec::with_capacity(self.max_depth * 3 + 4);
        stack.push(0usize);
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            for &slot in &node.items {
                let e = &self.entries[slot];
                if e.alive
                    && e.max_x >= qx0
                    && e.min_x <= qx1
                    && e.max_y >= qy0
                    && e.min_y <= qy1
                {
                    out.push(slot);
                }
            }
            if let Some(children) = node.children {
                for &c_idx in &children {
                    let c = &self.nodes[c_idx];
                    if c.max_x >= qx0 && c.min_x <= qx1 && c.max_y >= qy0 && c.min_y <= qy1 {
                        stack.push(c_idx);
                    }
                }
            }
        }
        out
    }
}

#[pymethods]
impl RustSpatialIndex {
    #[new]
    #[pyo3(signature = (
        min_x = -50000.0,
        min_y = -50000.0,
        max_x = 50000.0,
        max_y = 50000.0,
        max_items = 12,
        max_depth = 8
    ))]
    fn new(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        max_items: usize,
        max_depth: usize,
    ) -> Self {
        let (x0, y0, mut x1, mut y1) = normalize_bounds(min_x, min_y, max_x, max_y);
        if (x1 - x0).abs() < 1.0 {
            x1 = x0 + 1000.0;
        }
        if (y1 - y0).abs() < 1.0 {
            y1 = y0 + 1000.0;
        }
        let mut idx = Self {
            default_bounds: (x0, y0, x1, y1),
            max_items: max_items.max(2),
            max_depth: max_depth.clamp(1, 16),
            nodes: Vec::with_capacity(64),
            entries: Vec::new(),
            free_slots: Vec::new(),
            id_to_slot: HashMap::new(),
            next_order: 0,
        };
        idx.nodes.push(QuadNode::new(x0, y0, x1, y1, 0));
        idx
    }

    fn clear(&mut self) {
        let (x0, y0, x1, y1) = self.default_bounds;
        self.reset_tree_with_bounds(x0, y0, x1, y1);
    }

    fn rebuild(&mut self, items: Vec<(String, f64, f64, f64, f64, i64)>) {
        let (mut rx0, mut ry0, mut rx1, mut ry1) = self.default_bounds;
        for &(_, min_x, min_y, max_x, max_y, _) in &items {
            let (x0, y0, x1, y1) = normalize_bounds(min_x, min_y, max_x, max_y);
            if x0 < rx0 {
                rx0 = x0 - 64.0;
            }
            if y0 < ry0 {
                ry0 = y0 - 64.0;
            }
            if x1 > rx1 {
                rx1 = x1 + 64.0;
            }
            if y1 > ry1 {
                ry1 = y1 + 64.0;
            }
        }
        self.reset_tree_with_bounds(rx0, ry0, rx1, ry1);
        self.entries.reserve(items.len());
        self.id_to_slot.reserve(items.len());
        for (id, min_x, min_y, max_x, max_y, z_index) in items {
            self.upsert(id, min_x, min_y, max_x, max_y, z_index);
        }
    }

    fn upsert(
        &mut self,
        id: String,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        z_index: i64,
    ) {
        if self.id_to_slot.contains_key(&id) {
            self.remove_by_key(&id);
        }
        let (x0, y0, x1, y1) = normalize_bounds(min_x, min_y, max_x, max_y);
        let order = self.next_order;
        self.next_order = self.next_order.wrapping_add(1);
        let entry = SpatialEntry {
            id: id.clone(),
            min_x: x0,
            min_y: y0,
            max_x: x1,
            max_y: y1,
            z_index,
            order,
            node_idx: 0,
            alive: true,
        };
        let slot = if let Some(free) = self.free_slots.pop() {
            self.entries[free] = entry;
            free
        } else {
            let idx = self.entries.len();
            self.entries.push(entry);
            idx
        };
        self.id_to_slot.insert(id, slot);
        self.insert_slot(slot);
    }

    fn remove(&mut self, id: String) -> bool {
        self.remove_by_key(&id)
    }

    #[pyo3(signature = (px, py, tolerance = 10.0))]
    fn query_point(&self, px: f64, py: f64, tolerance: f64) -> Vec<String> {
        let tol = tolerance.max(0.0);
        let mut slots = self.query_aabb_slots(px - tol, py - tol, px + tol, py + tol);
        slots.sort_by(|&a, &b| {
            let ea = &self.entries[a];
            let eb = &self.entries[b];
            eb.z_index
                .cmp(&ea.z_index)
                .then_with(|| eb.order.cmp(&ea.order))
        });
        slots
            .into_iter()
            .map(|s| self.entries[s].id.clone())
            .collect()
    }

    fn query_rect(&self, rx: f64, ry: f64, rw: f64, rh: f64) -> Vec<String> {
        let qx0 = rx.min(rx + rw);
        let qx1 = rx.max(rx + rw);
        let qy0 = ry.min(ry + rh);
        let qy1 = ry.max(ry + rh);
        let mut slots = self.query_aabb_slots(qx0, qy0, qx1, qy1);
        slots.sort_by(|&a, &b| {
            let ea = &self.entries[a];
            let eb = &self.entries[b];
            ea.z_index
                .cmp(&eb.z_index)
                .then_with(|| ea.order.cmp(&eb.order))
        });
        slots
            .into_iter()
            .map(|s| self.entries[s].id.clone())
            .collect()
    }

    fn len(&self) -> usize {
        self.id_to_slot.len()
    }

    fn __len__(&self) -> usize {
        self.id_to_slot.len()
    }
}

// =============================================================================
// MULTI-THREADED RGB24 IMAGE RESIZE + 3x3 UNSHARP MASK PREPROCESSOR
// =============================================================================

#[inline(always)]
fn round_clamp_u8(v: f32) -> u8 {
    (v.clamp(0.0, 255.0) + 0.5).min(255.0) as u8
}

fn resize_band_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    x_lut: &[(usize, usize, f32)],
    band_y0: usize,
    dst_band: &mut [u8],
) {
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;
    let band_rows = dst_band.len() / dst_stride;
    let scale_y = (src_h as f32) / (dst_h as f32);
    let max_sy = (src_h - 1) as f32;

    for local_y in 0..band_rows {
        let dy = band_y0 + local_y;
        let (y0, y1, wy) = if src_h == 1 {
            (0usize, 0usize, 0.0_f32)
        } else {
            let sy = (((dy as f32) + 0.5) * scale_y - 0.5).clamp(0.0, max_sy);
            let iy0 = sy as usize;
            let iy1 = (iy0 + 1).min(src_h - 1);
            (iy0, iy1, sy - (iy0 as f32))
        };
        let wy0 = 1.0_f32 - wy;
        let row0 = &src[y0 * src_stride..(y0 + 1) * src_stride];
        let row1 = &src[y1 * src_stride..(y1 + 1) * src_stride];
        let out_row = &mut dst_band[local_y * dst_stride..(local_y + 1) * dst_stride];

        for (dx, &(x0, x1, wx)) in x_lut.iter().enumerate() {
            let wx0 = 1.0_f32 - wx;
            let w00 = wx0 * wy0;
            let w10 = wx * wy0;
            let w01 = wx0 * wy;
            let w11 = wx * wy;
            let p0 = x0 * 3;
            let p1 = x1 * 3;
            let dst_off = dx * 3;
            out_row[dst_off] = round_clamp_u8(
                (row0[p0] as f32) * w00
                    + (row0[p1] as f32) * w10
                    + (row1[p0] as f32) * w01
                    + (row1[p1] as f32) * w11,
            );
            out_row[dst_off + 1] = round_clamp_u8(
                (row0[p0 + 1] as f32) * w00
                    + (row0[p1 + 1] as f32) * w10
                    + (row1[p0 + 1] as f32) * w01
                    + (row1[p1 + 1] as f32) * w11,
            );
            out_row[dst_off + 2] = round_clamp_u8(
                (row0[p0 + 2] as f32) * w00
                    + (row0[p1 + 2] as f32) * w10
                    + (row1[p0 + 2] as f32) * w01
                    + (row1[p1 + 2] as f32) * w11,
            );
        }
    }
}

fn resize_band_box(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    x_spans: &[(usize, usize, f32, f32)],
    band_y0: usize,
    dst_band: &mut [u8],
) {
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;
    let band_rows = dst_band.len() / dst_stride;
    let scale_y = (src_h as f32) / (dst_h as f32);

    for local_y in 0..band_rows {
        let dy = band_y0 + local_y;
        let fy0 = (dy as f32) * scale_y;
        let fy1 = ((dy + 1) as f32) * scale_y;
        let iy0 = (fy0 as usize).min(src_h - 1);
        let iy1 = (fy1.ceil() as usize)
            .saturating_sub(1)
            .clamp(iy0, src_h - 1);
        let out_row = &mut dst_band[local_y * dst_stride..(local_y + 1) * dst_stride];

        for (dx, &(ix0, ix1, fx0, fx1)) in x_spans.iter().enumerate() {
            let mut r_acc = 0.0_f32;
            let mut g_acc = 0.0_f32;
            let mut b_acc = 0.0_f32;
            let mut w_acc = 0.0_f32;

            for sy in iy0..=iy1 {
                let wy = (fy1.min((sy + 1) as f32) - fy0.max(sy as f32)).max(0.0);
                if wy <= 0.0 {
                    continue;
                }
                let row = &src[sy * src_stride..(sy + 1) * src_stride];
                for sx in ix0..=ix1 {
                    let wx = (fx1.min((sx + 1) as f32) - fx0.max(sx as f32)).max(0.0);
                    let w = wx * wy;
                    if w <= 0.0 {
                        continue;
                    }
                    let off = sx * 3;
                    r_acc += (row[off] as f32) * w;
                    g_acc += (row[off + 1] as f32) * w;
                    b_acc += (row[off + 2] as f32) * w;
                    w_acc += w;
                }
            }
            let inv_w = if w_acc > 1e-6 { 1.0_f32 / w_acc } else { 0.0_f32 };
            let dst_off = dx * 3;
            out_row[dst_off] = round_clamp_u8(r_acc * inv_w);
            out_row[dst_off + 1] = round_clamp_u8(g_acc * inv_w);
            out_row[dst_off + 2] = round_clamp_u8(b_acc * inv_w);
        }
    }
}

fn unsharp_mask_band(
    src: &[u8],
    w: usize,
    h: usize,
    amount: f32,
    band_y0: usize,
    dst_band: &mut [u8],
) {
    let stride = w * 3;
    let band_rows = dst_band.len() / stride;
    let last_x = w - 1;
    let last_y = h - 1;
    let k_orig = 1.0_f32 + amount;
    let k_blur = amount * (1.0_f32 / 16.0_f32);

    for local_y in 0..band_rows {
        let y = band_y0 + local_y;
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(last_y);
        let r_top = &src[ym1 * stride..(ym1 + 1) * stride];
        let r_mid = &src[y * stride..(y + 1) * stride];
        let r_bot = &src[yp1 * stride..(yp1 + 1) * stride];
        let out_row = &mut dst_band[local_y * stride..(local_y + 1) * stride];

        for x in 0..w {
            let xm1 = x.saturating_sub(1) * 3;
            let x0 = x * 3;
            let xp1 = (x + 1).min(last_x) * 3;
            for c in 0..3 {
                let sum16 = (r_top[xm1 + c] as u32)
                    + 2 * (r_top[x0 + c] as u32)
                    + (r_top[xp1 + c] as u32)
                    + 2 * (r_mid[xm1 + c] as u32)
                    + 4 * (r_mid[x0 + c] as u32)
                    + 2 * (r_mid[xp1 + c] as u32)
                    + (r_bot[xm1 + c] as u32)
                    + 2 * (r_bot[x0 + c] as u32)
                    + (r_bot[xp1 + c] as u32);
                let orig = r_mid[x0 + c] as f32;
                out_row[x0 + c] = round_clamp_u8(orig * k_orig - (sum16 as f32) * k_blur);
            }
        }
    }
}

/// GIL-free, zero-dependency multi-threaded RGB24 image resize (bilinear / area box)
/// + crisp 3x3 Gaussian unsharp mask preprocessing.
#[pyfunction]
#[pyo3(signature = (rgb_bytes, src_w, src_h, dst_w, dst_h, sharpen_amount = 0.35))]
fn preprocess_image_resize_sharpen_rs(
    py: Python<'_>,
    rgb_bytes: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    sharpen_amount: f32,
) -> PyResult<Py<PyBytes>> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err(PyValueError::new_err(
            "src_w, src_h, dst_w, and dst_h must all be > 0",
        ));
    }
    let expected_src = src_w
        .checked_mul(src_h)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| PyValueError::new_err("source image dimensions overflow"))?;
    if rgb_bytes.len() < expected_src {
        return Err(PyValueError::new_err(format!(
            "rgb_bytes too short: expected at least {} bytes for {}x{} RGB24, got {}",
            expected_src,
            src_w,
            src_h,
            rgb_bytes.len()
        )));
    }
    let dst_len = dst_w
        .checked_mul(dst_h)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| PyValueError::new_err("destination image dimensions overflow"))?;

    let src = &rgb_bytes[..expected_src];

    let result_buf = py.allow_threads(|| {
        let dst_pixels = dst_w.saturating_mul(dst_h);
        let use_parallel = dst_pixels >= 65_536 && dst_h >= 16;
        let num_threads = if use_parallel {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(2, 8)
                .min(dst_h)
        } else {
            1
        };
        let dst_stride = dst_w * 3;
        let rows_per_band = (dst_h + num_threads - 1) / num_threads;

        // 1. Resize (identity copy / box area downscale / bilinear)
        let mut resized = vec![0u8; dst_len];
        if src_w == dst_w && src_h == dst_h {
            resized.copy_from_slice(src);
        } else if src_w > dst_w * 2 || src_h > dst_h * 2 {
            let scale_x = (src_w as f32) / (dst_w as f32);
            let x_spans: Vec<(usize, usize, f32, f32)> = (0..dst_w)
                .map(|dx| {
                    let fx0 = (dx as f32) * scale_x;
                    let fx1 = ((dx + 1) as f32) * scale_x;
                    let ix0 = (fx0.floor() as usize).min(src_w - 1);
                    let ix1 = (fx1.ceil() as usize)
                        .saturating_sub(1)
                        .clamp(ix0, src_w - 1);
                    (ix0, ix1, fx0, fx1)
                })
                .collect();
            if use_parallel {
                let x_spans_ref = &x_spans;
                std::thread::scope(|s| {
                    for (band_idx, band) in resized.chunks_mut(rows_per_band * dst_stride).enumerate()
                    {
                        let band_y0 = band_idx * rows_per_band;
                        s.spawn(move || {
                            resize_band_box(
                                src,
                                src_w,
                                src_h,
                                dst_w,
                                dst_h,
                                x_spans_ref,
                                band_y0,
                                band,
                            );
                        });
                    }
                });
            } else {
                resize_band_box(src, src_w, src_h, dst_w, dst_h, &x_spans, 0, &mut resized);
            }
        } else {
            let scale_x = (src_w as f32) / (dst_w as f32);
            let max_sx = (src_w - 1) as f32;
            let x_lut: Vec<(usize, usize, f32)> = (0..dst_w)
                .map(|dx| {
                    if src_w == 1 {
                        (0usize, 0usize, 0.0_f32)
                    } else {
                        let sx = (((dx as f32) + 0.5) * scale_x - 0.5).clamp(0.0, max_sx);
                        let ix0 = sx.floor() as usize;
                        let ix1 = (ix0 + 1).min(src_w - 1);
                        (ix0, ix1, sx - (ix0 as f32))
                    }
                })
                .collect();
            if use_parallel {
                let x_lut_ref = &x_lut;
                std::thread::scope(|s| {
                    for (band_idx, band) in resized.chunks_mut(rows_per_band * dst_stride).enumerate()
                    {
                        let band_y0 = band_idx * rows_per_band;
                        s.spawn(move || {
                            resize_band_bilinear(
                                src, src_w, src_h, dst_w, dst_h, x_lut_ref, band_y0, band,
                            );
                        });
                    }
                });
            } else {
                resize_band_bilinear(src, src_w, src_h, dst_w, dst_h, &x_lut, 0, &mut resized);
            }
        }

        // 2. Crisp 3x3 Unsharp Mask when sharpen_amount > 0.01
        if sharpen_amount > 0.01 {
            let amount = sharpen_amount.clamp(0.0, 3.0);
            let mut sharpened = vec![0u8; dst_len];
            if use_parallel {
                let resized_ref = &resized;
                std::thread::scope(|s| {
                    for (band_idx, band) in
                        sharpened.chunks_mut(rows_per_band * dst_stride).enumerate()
                    {
                        let band_y0 = band_idx * rows_per_band;
                        s.spawn(move || {
                            unsharp_mask_band(resized_ref, dst_w, dst_h, amount, band_y0, band);
                        });
                    }
                });
            } else {
                unsharp_mask_band(&resized, dst_w, dst_h, amount, 0, &mut sharpened);
            }
            sharpened
        } else {
            resized
        }
    });

    Ok(PyBytes::new(py, &result_buf).unbind())
}

#[pyfunction]
fn version() -> &'static str {
    "airteach_core_rs 2.0.0"
}

#[pyfunction]
fn backend_info() -> &'static str {
    "airteach_core_rs 2.0.0 (Rust PyO3 native: One-Euro + Kalman + GestureFSM + Geometry + Quadtree + HitTest + ImagePreprocess)"
}

#[pymodule]
fn airteach_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustOneEuroFilter1D>()?;
    m.add_class::<RustOneEuroFilter2D>()?;
    m.add_class::<RustOneEuroFilter3D>()?;
    m.add_class::<RustKinematic60HzExtrapolator>()?;
    m.add_class::<RustSwipeDetector>()?;
    m.add_class::<RustKalmanVelocityPredictor2D>()?;
    m.add_class::<RustGestureStateMachine>()?;
    m.add_class::<RustDualHandStateMachine>()?;
    m.add_class::<RustSpatialIndex>()?;
    m.add_function(wrap_pyfunction!(classify_hand_3d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_simplify_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_simplify_2d, m)?)?;
    m.add_function(wrap_pyfunction!(resample_arc_length_rs, m)?)?;
    m.add_function(wrap_pyfunction!(catmull_rom_spline_rs, m)?)?;
    m.add_function(wrap_pyfunction!(snap_geometric_shape_rs, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_frame_luma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gamma_contrast_rs, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_image_resize_sharpen_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dual_hand_kinematics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_roi_crop_box_rs, m)?)?;
    m.add_function(wrap_pyfunction!(hit_test_polyline_rs, m)?)?;
    m.add_function(wrap_pyfunction!(hit_test_objects_rs, m)?)?;
    m.add_function(wrap_pyfunction!(marquee_select_objects_rs, m)?)?;
    m.add_function(wrap_pyfunction!(joint_angle_3d_deg, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_vec, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}
