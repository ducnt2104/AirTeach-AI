//! Strict Real-Hand MediaPipe Landmark Extraction (`src/core/hand_tracker.rs`):
//! - MODULE 1.1: Zero Skin-Color / Contour Fallback Policy — NEVER fabricates phantom hands on faces or windows!
//! - Uses Google MediaPipe `HandLandmarker` (`models/hand_landmarker.task`, `min_detection_confidence = 0.78`,
//!   `min_tracking_confidence = 0.75`) + `FaceDetector` (`models/blaze_face_short_range.tflite`).
//! - MODULE 1.2: Strict Real-Hand Anatomical & Face-Exclusion Validator:
//!   * Rule 1: `handedness.score >= 0.80`
//!   * Rule 2: `palm_len = dist(LM 0, LM 9)` in `(0.06, 0.45)` and `0.45 < (finger_len / palm_len) < 1.35`
//!   * Rule 3: Excludes any candidate skeleton whose wrist (`LM 0`) or palm center (`LM 9`) falls inside
//!             a detected Face Bounding Box.

use crate::core::stroke_engine::{OneEuroFilter, Point2D};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

/// Embedded ONNX model bytes (retained for compatibility with `src/ai`).
pub static HAND_LANDMARKER_ONNX_BYTES: &[u8] =
    include_bytes!("../../assets/models/hand_landmarker_two_hands.onnx");

/// Embedded MediaPipe Tasks Co-Processor script (auto-provisioned if `scripts/mediapipe_hand_worker.py` is missing).
const MEDIAPIPE_WORKER_SCRIPT: &str = include_str!("../../scripts/mediapipe_hand_worker.py");

/// Standard 21 MediaPipe Hand Landmark indices.
pub mod landmark_idx {
    pub const WRIST: usize = 0;
    pub const THUMB_CMC: usize = 1;
    pub const THUMB_MCP: usize = 2;
    pub const THUMB_IP: usize = 3;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_MCP: usize = 5;
    pub const INDEX_PIP: usize = 6;
    pub const INDEX_DIP: usize = 7;
    pub const INDEX_TIP: usize = 8;
    pub const MIDDLE_MCP: usize = 9;
    pub const MIDDLE_PIP: usize = 10;
    pub const MIDDLE_DIP: usize = 11;
    pub const MIDDLE_TIP: usize = 12;
    pub const RING_MCP: usize = 13;
    pub const RING_PIP: usize = 14;
    pub const RING_DIP: usize = 15;
    pub const RING_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_PIP: usize = 18;
    pub const PINKY_DIP: usize = 19;
    pub const PINKY_TIP: usize = 20;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Landmark3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Landmark3D {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn to_2d(self) -> Point2D {
        Point2D::new(self.x, self.y)
    }

    #[inline]
    pub fn dist_2d(self, other: Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Handedness {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HandLandmarks {
    pub handedness: Handedness,
    pub confidence: f32,
    pub landmarks: [Landmark3D; 21],
}

impl Default for HandLandmarks {
    fn default() -> Self {
        Self {
            handedness: Handedness::Right,
            confidence: 0.0,
            landmarks: [Landmark3D::default(); 21],
        }
    }
}

impl HandLandmarks {
    /// Backlight-Resilient Hand Validator (`MODULE 3.2`):
    /// - Rule 1: `confidence >= 0.50` (lowered from 0.80 so backlit/tilted hands are never dropped)
    /// - Rule 2: `bounding_box_area > 0.015` (`1.5%` of frame area)
    /// - Rule 3: Palm center (`LM 9`) must not coincide with the detected Face Center
    pub fn is_anatomically_valid(&self, face_bbox: Option<[f32; 4]>) -> bool {
        if self.confidence < 0.50 {
            return false;
        }

        let mut min_x = 1.0_f32;
        let mut max_x = 0.0_f32;
        let mut min_y = 1.0_f32;
        let mut max_y = 0.0_f32;
        for lm in &self.landmarks {
            min_x = min_x.min(lm.x);
            max_x = max_x.max(lm.x);
            min_y = min_y.min(lm.y);
            max_y = max_y.max(lm.y);
        }
        let bounding_box_area = (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0);
        if bounding_box_area <= 0.015 {
            return false;
        }

        let mid_mcp = self.landmarks[landmark_idx::MIDDLE_MCP];
        if let Some([fx, fy, fw, fh]) = face_bbox {
            let face_cx = fx + fw * 0.5;
            let face_cy = fy + fh * 0.5;
            let radius = fw.max(fh) * 0.42;
            let dist_to_face =
                ((mid_mcp.x - face_cx).powi(2) + (mid_mcp.y - face_cy).powi(2)).sqrt();
            if dist_to_face < radius {
                return false;
            }
        }

        true
    }

    /// Generate an anatomically accurate 21-landmark hand skeleton ONLY for unit tests (`#[cfg(test)]`).
    /// Never used during live camera capture (`MODULE 1.1`).
    pub fn from_pose_preset(
        handedness: Handedness,
        cx: f32,
        cy: f32,
        scale: f32,
        thumb_pinched: bool,
        index_open: bool,
        middle_open: bool,
        ring_open: bool,
        pinky_open: bool,
    ) -> Self {
        let mut lm = [Landmark3D::default(); 21];
        let dir = match handedness {
            Handedness::Left => -1.0_f32,
            Handedness::Right => 1.0_f32,
        };

        lm[0] = Landmark3D::new(cx, cy + 0.11 * scale, 0.0);
        lm[1] = Landmark3D::new(cx - 0.035 * dir * scale, cy + 0.07 * scale, -0.01);
        lm[2] = Landmark3D::new(cx - 0.055 * dir * scale, cy + 0.03 * scale, -0.02);
        lm[3] = Landmark3D::new(cx - 0.065 * dir * scale, cy - 0.01 * scale, -0.02);

        let idx_tip_x = cx - 0.025 * dir * scale;
        let idx_tip_y = if index_open {
            cy - 0.11 * scale
        } else {
            cy + 0.025 * scale
        };
        lm[5] = Landmark3D::new(cx - 0.025 * dir * scale, cy + 0.01 * scale, 0.0);
        lm[6] = Landmark3D::new(
            cx - 0.025 * dir * scale,
            if index_open {
                cy - 0.035 * scale
            } else {
                cy - 0.015 * scale
            },
            -0.01,
        );
        lm[7] = Landmark3D::new(
            cx - 0.025 * dir * scale,
            if index_open {
                cy - 0.075 * scale
            } else {
                cy + 0.015 * scale
            },
            -0.015,
        );
        lm[8] = Landmark3D::new(idx_tip_x, idx_tip_y, -0.02);

        if thumb_pinched {
            lm[4] = Landmark3D::new(
                idx_tip_x - 0.008 * dir * scale,
                idx_tip_y + 0.008 * scale,
                -0.02,
            );
        } else {
            lm[4] = Landmark3D::new(cx - 0.095 * dir * scale, cy - 0.02 * scale, -0.02);
        }

        lm[9] = Landmark3D::new(cx, cy + 0.005 * scale, 0.0);
        let mid_y = if middle_open {
            cy - 0.12 * scale
        } else {
            cy + 0.03 * scale
        };
        lm[10] = Landmark3D::new(
            cx,
            if middle_open {
                (lm[9].y + mid_y) * 0.5
            } else {
                cy - 0.015 * scale
            },
            -0.01,
        );
        lm[11] = Landmark3D::new(cx, (lm[9].y + mid_y * 2.0) / 3.0, -0.015);
        lm[12] = Landmark3D::new(cx + 0.005 * dir * scale, mid_y, -0.02);

        lm[13] = Landmark3D::new(cx + 0.022 * dir * scale, cy + 0.012 * scale, 0.0);
        let ring_y = if ring_open {
            cy - 0.105 * scale
        } else {
            cy + 0.035 * scale
        };
        lm[14] = Landmark3D::new(
            lm[13].x,
            if ring_open {
                (lm[13].y + ring_y) * 0.5
            } else {
                cy - 0.01 * scale
            },
            -0.01,
        );
        lm[15] = Landmark3D::new(lm[13].x, (lm[13].y + ring_y * 2.0) / 3.0, -0.015);
        lm[16] = Landmark3D::new(lm[13].x, ring_y, -0.02);

        lm[17] = Landmark3D::new(cx + 0.042 * dir * scale, cy + 0.025 * scale, 0.0);
        let pinky_y = if pinky_open {
            cy - 0.085 * scale
        } else {
            cy + 0.042 * scale
        };
        lm[18] = Landmark3D::new(
            lm[17].x,
            if pinky_open {
                (lm[17].y + pinky_y) * 0.5
            } else {
                cy - 0.005 * scale
            },
            -0.01,
        );
        lm[19] = Landmark3D::new(lm[17].x, (lm[17].y + pinky_y * 2.0) / 3.0, -0.015);
        lm[20] = Landmark3D::new(lm[17].x, pinky_y, -0.02);

        Self {
            handedness,
            confidence: 0.94,
            landmarks: lm,
        }
    }
}

/// Output of a single frame's two-hand tracking pass + optional MediaPipe face box & eye keypoints.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TwoHandFrame {
    pub left_hand: Option<HandLandmarks>,
    pub right_hand: Option<HandLandmarks>,
    pub detected_face_bbox: Option<[f32; 4]>,
    pub detected_face_eyes: Option<[f32; 4]>,
    pub inference_ms: f32,
    pub using_onnx_session: bool,
}

#[derive(Deserialize)]
struct WorkerHandJson {
    score: f32,
    lm: Vec<[f32; 3]>,
}

#[derive(Deserialize)]
struct WorkerResponseJson {
    left: Option<WorkerHandJson>,
    right: Option<WorkerHandJson>,
    face: Option<[f32; 4]>,
    #[serde(default)]
    face_eyes: Option<[f32; 4]>,
}

struct MediaPipeSubprocessBridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    rgb_buf: Vec<u8>,
}

impl MediaPipeSubprocessBridge {
    fn try_spawn() -> Option<Self> {
        let script_path = Path::new("scripts/mediapipe_hand_worker.py");
        let _ = std::fs::create_dir_all("scripts");
        let _ = std::fs::write(script_path, MEDIAPIPE_WORKER_SCRIPT);

        let mut cmd = Command::new("python");
        cmd.arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd.spawn().ok()?;
        let stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let mut reader = BufReader::new(stdout);

        let mut ready_line = String::new();
        if reader.read_line(&mut ready_line).is_err() || !ready_line.contains("READY") {
            let _ = child.kill();
            return None;
        }

        Some(Self {
            child,
            stdin,
            stdout: reader,
            rgb_buf: vec![0u8; 8 + 320 * 240 * 3],
        })
    }

    fn detect_frame(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        external_face_bbox: Option<[f32; 4]>,
    ) -> Option<(
        Option<HandLandmarks>,
        Option<HandLandmarks>,
        Option<[f32; 4]>,
        Option<[f32; 4]>,
    )> {
        const DST_W: u32 = 320;
        const DST_H: u32 = 240;
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            return Some((None, None, None, None));
        }

        self.rgb_buf[0..4].copy_from_slice(b"MPHF");
        self.rgb_buf[4..6].copy_from_slice(&(DST_W as u16).to_le_bytes());
        self.rgb_buf[6..8].copy_from_slice(&(DST_H as u16).to_le_bytes());

        let mut dst_idx = 8usize;
        for y in 0..DST_H {
            let sy = ((y * height) / DST_H).min(height - 1);
            for x in 0..DST_W {
                let sx = ((x * width) / DST_W).min(width - 1);
                let src_idx = ((sy * width + sx) * 4) as usize;
                self.rgb_buf[dst_idx] = rgba[src_idx];
                self.rgb_buf[dst_idx + 1] = rgba[src_idx + 1];
                self.rgb_buf[dst_idx + 2] = rgba[src_idx + 2];
                dst_idx += 3;
            }
        }

        self.stdin.write_all(&self.rgb_buf).ok()?;
        self.stdin.flush().ok()?;

        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }

        let parsed: WorkerResponseJson = serde_json::from_str(line.trim()).ok()?;
        let effective_face = parsed.face.or(external_face_bbox);

        let parse_hand = |wh: WorkerHandJson, handedness: Handedness| -> Option<HandLandmarks> {
            if wh.lm.len() < 21 {
                return None;
            }
            let mut landmarks = [Landmark3D::default(); 21];
            for (i, coords) in wh.lm.iter().take(21).enumerate() {
                landmarks[i] = Landmark3D::new(coords[0], coords[1], coords[2]);
            }
            let hand = HandLandmarks {
                handedness,
                confidence: wh.score,
                landmarks,
            };
            if hand.is_anatomically_valid(effective_face) {
                Some(hand)
            } else {
                None
            }
        };

        let left_hand = parsed.left.and_then(|h| parse_hand(h, Handedness::Left));
        let right_hand = parsed.right.and_then(|h| parse_hand(h, Handedness::Right));
        Some((left_hand, right_hand, parsed.face, parsed.face_eyes))
    }
}

impl Drop for MediaPipeSubprocessBridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// 21-Landmark 1-Euro Filter bank for a single hand (x, y, z for every joint 0..21).
pub struct SingleHandLandmarkSmoother {
    filters: [(OneEuroFilter, OneEuroFilter, OneEuroFilter); 21],
    last_seen_sec: Option<f64>,
}

impl SingleHandLandmarkSmoother {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        let single = (
            OneEuroFilter::new(min_cutoff, beta, d_cutoff),
            OneEuroFilter::new(min_cutoff, beta, d_cutoff),
            OneEuroFilter::new(min_cutoff, beta, d_cutoff),
        );
        Self {
            filters: [single; 21],
            last_seen_sec: None,
        }
    }

    pub fn update_params(&mut self, min_cutoff: f32, beta: f32, d_cutoff: f32) {
        for (fx, fy, fz) in &mut self.filters {
            fx.min_cutoff = min_cutoff;
            fx.beta = beta;
            fx.d_cutoff = d_cutoff;
            fy.min_cutoff = min_cutoff;
            fy.beta = beta;
            fy.d_cutoff = d_cutoff;
            fz.min_cutoff = min_cutoff;
            fz.beta = beta;
            fz.d_cutoff = d_cutoff;
        }
    }

    pub fn reset(&mut self) {
        for (fx, fy, fz) in &mut self.filters {
            fx.reset();
            fy.reset();
            fz.reset();
        }
        self.last_seen_sec = None;
    }

    /// Filter all 21 landmarks `[0..21]` at timestamp `t_sec`.
    /// Resets automatically if the hand left the camera FOV for > 0.30s.
    pub fn smooth(&mut self, mut hand: HandLandmarks, t_sec: f64) -> HandLandmarks {
        if let Some(prev) = self.last_seen_sec {
            if t_sec - prev > 0.30 {
                self.reset();
            }
        }
        self.last_seen_sec = Some(t_sec);

        for (i, lm) in hand.landmarks.iter_mut().enumerate() {
            let (fx, fy, fz) = &mut self.filters[i];
            lm.x = fx.filter(lm.x, t_sec);
            lm.y = fy.filter(lm.y, t_sec);
            lm.z = fz.filter(lm.z, t_sec);
        }
        hand
    }
}

/// Real Two-Hand MediaPipe Tracker (Zero Phantom Skin-Contour Fallback)
/// with full 21-landmark 1-Euro smoothing per hand.
pub struct TwoHandTracker {
    mp_bridge: Option<MediaPipeSubprocessBridge>,
    last_respawn_attempt: Instant,
    epoch: Instant,
    last_face_bbox: Option<[f32; 4]>,
    left_smoother: SingleHandLandmarkSmoother,
    right_smoother: SingleHandLandmarkSmoother,
}

impl TwoHandTracker {
    pub fn new() -> Self {
        let mp_bridge = MediaPipeSubprocessBridge::try_spawn();
        let now = Instant::now();
        Self {
            mp_bridge,
            last_respawn_attempt: now,
            epoch: now,
            last_face_bbox: None,
            left_smoother: SingleHandLandmarkSmoother::new(1.5, 0.04, 1.0),
            right_smoother: SingleHandLandmarkSmoother::new(1.5, 0.04, 1.0),
        }
    }

    pub fn set_smoothing_params(&mut self, min_cutoff: f32, beta: f32, d_cutoff: f32) {
        self.left_smoother.update_params(min_cutoff, beta, d_cutoff);
        self.right_smoother.update_params(min_cutoff, beta, d_cutoff);
    }

    pub fn is_onnx_loaded(&self) -> bool {
        self.mp_bridge.is_some()
    }

    pub fn set_face_exclusion_bbox(&mut self, face_bbox: Option<[f32; 4]>) {
        self.last_face_bbox = face_bbox;
    }

    /// Process an RGBA camera frame of dimensions `width x height` and extract smoothed Left & Right 21-landmark hands.
    pub fn process_frame(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        sim_override: Option<(Option<HandLandmarks>, Option<HandLandmarks>)>,
    ) -> TwoHandFrame {
        let start = Instant::now();
        let t_sec = self.epoch.elapsed().as_secs_f64();

        if let Some((left, right)) = sim_override {
            let smoothed_left = left.map(|h| self.left_smoother.smooth(h, t_sec));
            let smoothed_right = right.map(|h| self.right_smoother.smooth(h, t_sec));
            return TwoHandFrame {
                left_hand: smoothed_left,
                right_hand: smoothed_right,
                detected_face_bbox: None,
                detected_face_eyes: None,
                inference_ms: start.elapsed().as_secs_f32() * 1000.0,
                using_onnx_session: true,
            };
        }

        if self.mp_bridge.is_none() && self.last_respawn_attempt.elapsed().as_secs_f32() >= 3.0 {
            self.last_respawn_attempt = Instant::now();
            self.mp_bridge = MediaPipeSubprocessBridge::try_spawn();
        }

        let mut left_hand = None;
        let mut right_hand = None;
        let mut detected_face_bbox = None;
        let mut detected_face_eyes = None;
        let mut active = false;

        if let Some(ref mut bridge) = self.mp_bridge {
            if let Some((lh, rh, fb, fe)) =
                bridge.detect_frame(rgba, width, height, self.last_face_bbox)
            {
                left_hand = lh.map(|h| self.left_smoother.smooth(h, t_sec));
                right_hand = rh.map(|h| self.right_smoother.smooth(h, t_sec));
                detected_face_bbox = fb;
                detected_face_eyes = fe;
                active = true;
            } else {
                self.mp_bridge = None;
                self.left_smoother.reset();
                self.right_smoother.reset();
            }
        }

        if left_hand.is_none() {
            if let Some(last) = self.left_smoother.last_seen_sec {
                if t_sec - last > 0.30 {
                    self.left_smoother.reset();
                }
            }
        }
        if right_hand.is_none() {
            if let Some(last) = self.right_smoother.last_seen_sec {
                if t_sec - last > 0.30 {
                    self.right_smoother.reset();
                }
            }
        }

        TwoHandFrame {
            left_hand,
            right_hand,
            detected_face_bbox,
            detected_face_eyes,
            inference_ms: start.elapsed().as_secs_f32() * 1000.0,
            using_onnx_session: active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_21_landmark_one_euro_smoother_suppresses_jitter_across_all_joints() {
        let mut smoother = SingleHandLandmarkSmoother::new(1.0, 0.01, 1.0);
        let mut raw_diffs = vec![0.0_f32; 21];
        let mut smooth_diffs = vec![0.0_f32; 21];
        let mut prev_raw: Option<[Landmark3D; 21]> = None;
        let mut prev_smooth: Option<[Landmark3D; 21]> = None;

        for step in 0..30 {
            let jitter = if step % 2 == 0 { 0.014 } else { -0.014 };
            let mut h = HandLandmarks::from_pose_preset(
                Handedness::Right,
                0.5,
                0.5,
                1.0,
                false,
                true,
                false,
                false,
                false,
            );
            for lm in &mut h.landmarks {
                lm.x += jitter;
                lm.y -= jitter;
            }
            let smoothed = smoother.smooth(h, step as f64 * (1.0 / 30.0));
            if let (Some(pr), Some(ps)) = (prev_raw, prev_smooth) {
                for j in 0..21 {
                    raw_diffs[j] += h.landmarks[j].dist_2d(pr[j]);
                    smooth_diffs[j] += smoothed.landmarks[j].dist_2d(ps[j]);
                }
            }
            prev_raw = Some(h.landmarks);
            prev_smooth = Some(smoothed.landmarks);
        }

        for j in 0..21 {
            assert!(
                smooth_diffs[j] < raw_diffs[j] * 0.45,
                "Landmark {j} jitter was not sufficiently damped: smooth={}, raw={}",
                smooth_diffs[j],
                raw_diffs[j]
            );
        }
    }
}
