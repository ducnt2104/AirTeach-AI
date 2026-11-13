//! Two-Hand Gesture Recognizer & Distance-Invariant FSM (`src/core/gestures.rs`):
//! - Distance-invariant palm normalization (`palm_len = dist(WRIST, MIDDLE_MCP)`) so gestures remain
//!   consistent at any distance from the webcam (far or close).
//! - Orientation-aware extension detection combining wrist radial ratio + joint straightness cosine
//!   with asymmetric hysteresis (harder to enter, smoother to maintain).
//! - Pointer Deadzone (`~2.5%` palm scale) preventing stationary micro-jitter ink blobs.
//! - Stability Hold FSM requiring 5-8 frames (150-250ms) before triggering discrete transitions.

use crate::core::hand_tracker::{landmark_idx::*, HandLandmarks, Landmark3D};
use crate::core::stroke_engine::Point2D;
use serde::{Deserialize, Serialize};

/// Recommended stability window in frames (~180ms at 30 FPS; strictly within 5-8 frames).
pub const DEFAULT_GESTURE_HOLD_FRAMES: u8 = 6;
/// Recommended minimum physical hold duration in seconds (150-250ms window).
pub const DEFAULT_MIN_HOLD_SEC: f64 = 0.175;
/// Grace period preserving active gesture state across brief tracking dropouts (150ms).
pub const DEFAULT_GRACE_PERIOD_SEC: f64 = 0.150;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeftGesture {
    Idle,
    PinchDragZoom,
    PalmOpenAnchor,
}

impl LeftGesture {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "Nghỉ (Idle)",
            Self::PinchDragZoom => "Chụm tay (Kéo / Thu phóng)",
            Self::PalmOpenAnchor => "Xòe tay (Neo tọa độ)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RightGesture {
    Idle,
    DrawStroke,
    HoverTwoFingers,
    PinchToolSelect,
    PalmEraser,
    LaserPointer,
    FistClench,
    SelectBoxVShape,
}

impl RightGesture {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "Chờ tay",
            Self::DrawStroke => "Đang Vẽ (1 Ngón)",
            Self::HoverTwoFingers => "Di chuyển (2 Ngón)",
            Self::PinchToolSelect => "Click (Chụm tay)",
            Self::PalmEraser => "Tẩy (5 Ngón)",
            Self::LaserPointer => "Con trỏ Laser",
            Self::FistClench => "Ngắt nét (Nắm tay)",
            Self::SelectBoxVShape => "Khoanh vùng AI",
        }
    }

    pub const fn short_status(self) -> &'static str {
        match self {
            Self::Idle => "Trạng thái: Chờ tay",
            Self::DrawStroke => "Trạng thái: Đang Vẽ",
            Self::HoverTwoFingers | Self::LaserPointer | Self::FistClench => {
                "Trạng thái: Di chuyển"
            }
            Self::PinchToolSelect | Self::SelectBoxVShape => "Trạng thái: Click",
            Self::PalmEraser => "Trạng thái: Tẩy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightTransitionEvent {
    None,
    BeganStroke,
    EndedStrokeAndSnap,
    BeganSelection,
    CompletedSelection,
}

#[derive(Debug, Clone, Copy)]
pub struct DualHandGestureState {
    pub left_gesture: LeftGesture,
    pub right_gesture: RightGesture,
    pub left_cursor_norm: Option<Point2D>,
    pub right_cursor_norm: Option<Point2D>,
    pub left_pinch_span: f32,
    pub right_event: RightTransitionEvent,
    pub no_hand_idle: bool,
    pub in_grace_period: bool,
    pub right_hold_progress: f32,
}

/// Normalize hand reference scale as distance from Wrist (LM 0) to Middle MCP (LM 9).
#[inline]
pub fn compute_palm_scale(lm: &[Landmark3D; 21]) -> f32 {
    lm[WRIST].dist_2d(lm[MIDDLE_MCP]).max(1e-4)
}

/// Scale- and rotation-invariant finger extension metric using wrist radial ratio,
/// multi-bone dot-product joint angles (`MCP->PIP · PIP->DIP` and `PIP->DIP · DIP->TIP`),
/// and palm-scaled forward reach, with asymmetric hysteresis (`was_extended`).
#[inline]
pub fn is_finger_extended_hysteresis(
    lm: &[Landmark3D; 21],
    mcp_idx: usize,
    pip_idx: usize,
    tip_idx: usize,
    palm_scale: f32,
    was_extended: bool,
) -> bool {
    let wrist = lm[WRIST];
    let mcp = lm[mcp_idx];
    let pip = lm[pip_idx];
    let dip_idx = (tip_idx.saturating_sub(1)).max(pip_idx);
    let dip = lm[dip_idx];
    let tip = lm[tip_idx];

    let r_pip = wrist.dist_2d(pip) / palm_scale;
    let r_tip = wrist.dist_2d(tip) / palm_scale;
    let radial_ratio = r_tip / r_pip.max(0.15);

    // Direction vectors for MCP->PIP, PIP->DIP, and DIP->TIP bone segments
    let v1x = pip.x - mcp.x;
    let v1y = pip.y - mcp.y;
    let len1 = (v1x * v1x + v1y * v1y).sqrt().max(1e-5);

    let v2x = dip.x - pip.x;
    let v2y = dip.y - pip.y;
    let len2 = (v2x * v2x + v2y * v2y).sqrt().max(1e-5);

    let v3x = tip.x - dip.x;
    let v3y = tip.y - dip.y;
    let len3 = (v3x * v3x + v3y * v3y).sqrt().max(1e-5);

    let cos_pip = (v1x * v2x + v1y * v2y) / (len1 * len2);
    let cos_dip = (v2x * v3x + v2y * v3y) / (len2 * len3);
    let cos_overall = (v1x * (tip.x - pip.x) + v1y * (tip.y - pip.y))
        / (len1 * ((tip.x - pip.x).hypot(tip.y - pip.y).max(1e-5)));
    let cos_min_chain = cos_pip.min(cos_dip).max(cos_overall * 0.9);
    let forward_reach = wrist.dist_2d(tip) / palm_scale;

    if was_extended {
        // Hysteresis hold band: tolerates slight natural jitter without flickering out
        radial_ratio > 1.05 && cos_min_chain > 0.22 && cos_overall > 0.28 && forward_reach > 1.05
    } else {
        // Strict enter band: clearly straightened along full bone chain & pointing outward
        radial_ratio > 1.15 && cos_min_chain > 0.45 && cos_overall > 0.50 && forward_reach > 1.24
    }
}

/// Rotation- and scale-invariant thumb extension metric using CMC->MCP->IP dot product & spread.
#[inline]
pub fn is_thumb_extended_hysteresis(
    lm: &[Landmark3D; 21],
    palm_scale: f32,
    was_extended: bool,
) -> bool {
    let cmc = lm[THUMB_CMC];
    let mcp = lm[THUMB_MCP];
    let ip = lm[THUMB_IP];
    let tip = lm[THUMB_TIP];

    let v1x = mcp.x - cmc.x;
    let v1y = mcp.y - cmc.y;
    let len1 = v1x.hypot(v1y).max(1e-5);

    let v2x = tip.x - ip.x;
    let v2y = tip.y - ip.y;
    let len2 = v2x.hypot(v2y).max(1e-5);

    let cos_thumb = (v1x * v2x + v1y * v2y) / (len1 * len2);
    let spread_index = tip.dist_2d(lm[INDEX_MCP]) / palm_scale;

    if was_extended {
        cos_thumb > 0.30 && spread_index > 0.36
    } else {
        cos_thumb > 0.48 && spread_index > 0.46
    }
}

/// Stateless distance-invariant finger check helper.
#[inline]
pub fn is_finger_extended(
    lm: &[Landmark3D; 21],
    mcp_idx: usize,
    pip_idx: usize,
    tip_idx: usize,
) -> bool {
    let palm_scale = compute_palm_scale(lm);
    is_finger_extended_hysteresis(lm, mcp_idx, pip_idx, tip_idx, palm_scale, false)
}

/// Classify raw Left Hand (Navigation) pose using normalized palm coordinates + hysteresis.
pub fn classify_left_hand_with_hysteresis(
    hand: &HandLandmarks,
    prev_fingers: &mut [bool; 4],
    was_pinched: bool,
) -> (LeftGesture, Point2D, f32) {
    let lm = &hand.landmarks;
    let palm_scale = compute_palm_scale(lm);

    prev_fingers[0] = is_finger_extended_hysteresis(
        lm, INDEX_MCP, INDEX_PIP, INDEX_TIP, palm_scale, prev_fingers[0],
    );
    prev_fingers[1] = is_finger_extended_hysteresis(
        lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, palm_scale, prev_fingers[1],
    );
    prev_fingers[2] = is_finger_extended_hysteresis(
        lm, RING_MCP, RING_PIP, RING_TIP, palm_scale, prev_fingers[2],
    );
    prev_fingers[3] = is_finger_extended_hysteresis(
        lm, PINKY_MCP, PINKY_PIP, PINKY_TIP, palm_scale, prev_fingers[3],
    );

    let thumb_index_dist = lm[THUMB_TIP].dist_2d(lm[INDEX_TIP]);
    let pinch_ratio = thumb_index_dist / palm_scale;
    let palm_center = Point2D::new(lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y);

    let pinch_enter = if was_pinched { 0.42 } else { 0.30 };
    if prev_fingers.iter().all(|&f| f) && pinch_ratio > 0.48 {
        (LeftGesture::PalmOpenAnchor, palm_center, pinch_ratio)
    } else if pinch_ratio < pinch_enter {
        let pinch_mid = Point2D::new(
            0.5 * (lm[THUMB_TIP].x + lm[INDEX_TIP].x),
            0.5 * (lm[THUMB_TIP].y + lm[INDEX_TIP].y),
        );
        (LeftGesture::PinchDragZoom, pinch_mid, pinch_ratio)
    } else {
        (LeftGesture::Idle, palm_center, pinch_ratio)
    }
}

/// Stateless Left Hand classification wrapper.
pub fn classify_left_hand_raw(hand: &HandLandmarks) -> (LeftGesture, Point2D, f32) {
    let mut tmp = [false; 4];
    classify_left_hand_with_hysteresis(hand, &mut tmp, false)
}

/// Classify Right Hand using distance-invariant palm normalization + finger hysteresis.
pub fn classify_right_hand_with_hysteresis(
    hand: &HandLandmarks,
    prev_fingers: &mut [bool; 4],
    was_pinched: bool,
) -> (RightGesture, Point2D, f32) {
    let lm = &hand.landmarks;
    let palm_scale = compute_palm_scale(lm);

    prev_fingers[0] = is_finger_extended_hysteresis(
        lm, INDEX_MCP, INDEX_PIP, INDEX_TIP, palm_scale, prev_fingers[0],
    );
    prev_fingers[1] = is_finger_extended_hysteresis(
        lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, palm_scale, prev_fingers[1],
    );
    prev_fingers[2] = is_finger_extended_hysteresis(
        lm, RING_MCP, RING_PIP, RING_TIP, palm_scale, prev_fingers[2],
    );
    prev_fingers[3] = is_finger_extended_hysteresis(
        lm, PINKY_MCP, PINKY_PIP, PINKY_TIP, palm_scale, prev_fingers[3],
    );

    let [index_up, middle_up, ring_up, pinky_up] = *prev_fingers;
    let thumb_index_ratio = lm[THUMB_TIP].dist_2d(lm[INDEX_TIP]) / palm_scale;
    let is_thumb_pinched = thumb_index_ratio < if was_pinched { 0.36 } else { 0.27 };
    let index_tip_pt = lm[INDEX_TIP].to_2d();

    // 1. Open palm (4-5 fingers extended): Palm Eraser at palm center (LM 9)
    if index_up && middle_up && ring_up && pinky_up {
        return (RightGesture::PalmEraser, lm[MIDDLE_MCP].to_2d(), palm_scale);
    }

    // 2. Index + Middle up, Ring & Pinky curled: Hover cursor (cuts stroke smoothly)
    if index_up && middle_up && !ring_up {
        let mid_cursor = Point2D::new(
            0.5 * (lm[INDEX_TIP].x + lm[MIDDLE_TIP].x),
            0.5 * (lm[INDEX_TIP].y + lm[MIDDLE_TIP].y),
        );
        return (RightGesture::HoverTwoFingers, mid_cursor, palm_scale);
    }

    // 3. Only Index finger up: Draw Stroke at Index Tip (LM 8)
    if index_up && !middle_up && !ring_up {
        return (RightGesture::DrawStroke, index_tip_pt, palm_scale);
    }

    // 4. All curled -> Pinch Click or Fist Clench
    if !index_up && !middle_up && !ring_up && !pinky_up {
        if is_thumb_pinched {
            let pinch_pt = Point2D::new(
                0.5 * (lm[THUMB_TIP].x + lm[INDEX_TIP].x),
                0.5 * (lm[THUMB_TIP].y + lm[INDEX_TIP].y),
            );
            return (RightGesture::PinchToolSelect, pinch_pt, palm_scale);
        }
        return (RightGesture::FistClench, lm[MIDDLE_MCP].to_2d(), palm_scale);
    }

    (RightGesture::HoverTwoFingers, index_tip_pt, palm_scale)
}

/// Stateless Right Hand classification wrapper.
pub fn classify_right_hand_raw(hand: &HandLandmarks) -> (RightGesture, Point2D) {
    let mut tmp = [false; 4];
    let (g, pt, _) = classify_right_hand_with_hysteresis(hand, &mut tmp, false);
    (g, pt)
}

/// Finite State Machine with:
/// - Scale- & rotation-invariant hysteresis per finger state
/// - Palm-scaled pointer Deadzone (prevents micro-oscillation while holding steady)
/// - Configurable stability hold (`5-8 frames` / `150-250ms` in production)
/// - `150ms` Grace Period holding active state across brief tracking dropouts
pub struct GestureDebounceFsm {
    pub debounce_frames: u8,
    pub min_hold_sec: f64,
    pub grace_period_sec: f64,
    stable_left: LeftGesture,
    candidate_left: LeftGesture,
    left_counter: u8,
    left_candidate_since_sec: Option<f64>,
    last_left_seen_sec: Option<f64>,
    left_fingers: [bool; 4],
    left_deadzone_cursor: Option<Point2D>,

    stable_right: RightGesture,
    candidate_right: RightGesture,
    right_counter: u8,
    right_candidate_since_sec: Option<f64>,
    last_right_seen_sec: Option<f64>,
    right_fingers: [bool; 4],
    right_deadzone_cursor: Option<Point2D>,
    no_hand_frames: u8,
    frame_clock_sec: f64,
}

impl GestureDebounceFsm {
    /// Standard production constructor: 6 frames hold (~180ms), 175ms dwell, 150ms tracking grace period.
    pub const fn production() -> Self {
        Self {
            debounce_frames: DEFAULT_GESTURE_HOLD_FRAMES,
            min_hold_sec: DEFAULT_MIN_HOLD_SEC,
            grace_period_sec: DEFAULT_GRACE_PERIOD_SEC,
            stable_left: LeftGesture::Idle,
            candidate_left: LeftGesture::Idle,
            left_counter: 0,
            left_candidate_since_sec: None,
            last_left_seen_sec: None,
            left_fingers: [false; 4],
            left_deadzone_cursor: None,
            stable_right: RightGesture::Idle,
            candidate_right: RightGesture::Idle,
            right_counter: 0,
            right_candidate_since_sec: None,
            last_right_seen_sec: None,
            right_fingers: [false; 4],
            right_deadzone_cursor: None,
            no_hand_frames: 3,
            frame_clock_sec: 0.0,
        }
    }

    /// Custom frame-count constructor (sets zero grace/dwell for deterministic frame tests).
    pub const fn new(debounce_frames: u8) -> Self {
        let frames = if debounce_frames == 0 { 1 } else { debounce_frames };
        Self {
            debounce_frames: frames,
            min_hold_sec: 0.0,
            grace_period_sec: 0.0,
            stable_left: LeftGesture::Idle,
            candidate_left: LeftGesture::Idle,
            left_counter: 0,
            left_candidate_since_sec: None,
            last_left_seen_sec: None,
            left_fingers: [false; 4],
            left_deadzone_cursor: None,
            stable_right: RightGesture::Idle,
            candidate_right: RightGesture::Idle,
            right_counter: 0,
            right_candidate_since_sec: None,
            last_right_seen_sec: None,
            right_fingers: [false; 4],
            right_deadzone_cursor: None,
            no_hand_frames: 3,
            frame_clock_sec: 0.0,
        }
    }

    #[inline]
    fn required_frames_for_right(&self, candidate: RightGesture) -> u8 {
        match candidate {
            RightGesture::PalmEraser => self.debounce_frames.max(6),
            RightGesture::Idle => self.debounce_frames.max(5),
            _ => self.debounce_frames,
        }
    }

    #[inline]
    fn apply_deadzone(
        raw_pt: Option<Point2D>,
        palm_scale: f32,
        slot: &mut Option<Point2D>,
    ) -> Option<Point2D> {
        let Some(pt) = raw_pt else {
            *slot = None;
            return None;
        };
        let deadzone = (palm_scale * 0.025).clamp(0.002, 0.009);
        match *slot {
            Some(prev) if pt.dist(prev) < deadzone => Some(prev),
            _ => {
                *slot = Some(pt);
                Some(pt)
            }
        }
    }

    /// Advance FSM advancing internal virtual clock at 30 FPS (`+1/30s` per call).
    pub fn update(
        &mut self,
        left_hand: Option<&HandLandmarks>,
        right_hand: Option<&HandLandmarks>,
    ) -> DualHandGestureState {
        self.frame_clock_sec += 1.0 / 30.0;
        let t = self.frame_clock_sec;
        self.update_at(left_hand, right_hand, t)
    }

    /// Advance FSM with explicit monotonic timestamp `t_sec` (enforces both frame count AND `min_hold_sec`).
    pub fn update_at(
        &mut self,
        left_hand: Option<&HandLandmarks>,
        right_hand: Option<&HandLandmarks>,
        t_sec: f64,
    ) -> DualHandGestureState {
        // --- Left Hand with 150ms Grace Period, Hysteresis & Deadzone ---
        let mut left_in_grace = false;
        let (left_cursor_norm, left_pinch_span) = if let Some(lh) = left_hand {
            self.last_left_seen_sec = Some(t_sec);
            let was_pinch = self.stable_left == LeftGesture::PinchDragZoom;
            let (raw_left, raw_left_pt, span) =
                classify_left_hand_with_hysteresis(lh, &mut self.left_fingers, was_pinch);
            let cursor = Self::apply_deadzone(
                Some(raw_left_pt),
                compute_palm_scale(&lh.landmarks),
                &mut self.left_deadzone_cursor,
            );
            if raw_left == self.stable_left {
                self.candidate_left = raw_left;
                self.left_counter = 0;
                self.left_candidate_since_sec = None;
            } else if raw_left == self.candidate_left {
                self.left_counter = self.left_counter.saturating_add(1);
                let elapsed = t_sec - self.left_candidate_since_sec.unwrap_or(t_sec);
                if self.left_counter >= self.debounce_frames && elapsed >= self.min_hold_sec {
                    self.stable_left = raw_left;
                    self.left_counter = 0;
                    self.left_candidate_since_sec = None;
                }
            } else {
                self.candidate_left = raw_left;
                self.left_counter = 1;
                self.left_candidate_since_sec = Some(t_sec);
            }
            (cursor, span)
        } else if let Some(last_l) = self.last_left_seen_sec {
            if self.grace_period_sec > 0.0 && (t_sec - last_l) <= self.grace_period_sec {
                left_in_grace = self.stable_left != LeftGesture::Idle;
                (self.left_deadzone_cursor, 0.0)
            } else {
                self.left_fingers = [false; 4];
                self.left_deadzone_cursor = None;
                if LeftGesture::Idle == self.stable_left {
                    self.candidate_left = LeftGesture::Idle;
                    self.left_counter = 0;
                    self.left_candidate_since_sec = None;
                } else if LeftGesture::Idle == self.candidate_left {
                    self.left_counter = self.left_counter.saturating_add(1);
                    let elapsed = t_sec - self.left_candidate_since_sec.unwrap_or(t_sec);
                    if self.left_counter >= self.debounce_frames && elapsed >= self.min_hold_sec {
                        self.stable_left = LeftGesture::Idle;
                        self.left_counter = 0;
                        self.left_candidate_since_sec = None;
                    }
                } else {
                    self.candidate_left = LeftGesture::Idle;
                    self.left_counter = 1;
                    self.left_candidate_since_sec = Some(t_sec);
                }
                (None, 0.0)
            }
        } else {
            self.left_fingers = [false; 4];
            self.left_deadzone_cursor = None;
            (None, 0.0)
        };

        // --- Right Hand with 150ms Grace Period, Hysteresis & Deadzone ---
        let prev_right = self.stable_right;
        let mut right_in_grace = false;
        let right_cursor_norm = if let Some(rh) = right_hand {
            self.last_right_seen_sec = Some(t_sec);
            let was_pinch = self.stable_right == RightGesture::PinchToolSelect;
            let (raw_right, raw_right_pt, right_palm) =
                classify_right_hand_with_hysteresis(rh, &mut self.right_fingers, was_pinch);
            let cursor = Self::apply_deadzone(
                Some(raw_right_pt),
                right_palm,
                &mut self.right_deadzone_cursor,
            );
            let needed = self.required_frames_for_right(raw_right);
            if raw_right == self.stable_right {
                self.candidate_right = raw_right;
                self.right_counter = 0;
                self.right_candidate_since_sec = None;
            } else if raw_right == self.candidate_right {
                self.right_counter = self.right_counter.saturating_add(1);
                let elapsed = t_sec - self.right_candidate_since_sec.unwrap_or(t_sec);
                let hold_ok = raw_right == RightGesture::Idle || elapsed >= self.min_hold_sec;
                if self.right_counter >= needed && hold_ok {
                    self.stable_right = raw_right;
                    self.right_counter = 0;
                    self.right_candidate_since_sec = None;
                }
            } else {
                self.candidate_right = raw_right;
                self.right_counter = 1;
                self.right_candidate_since_sec = Some(t_sec);
                if needed <= 1 {
                    self.stable_right = raw_right;
                }
            }
            cursor
        } else if let Some(last_r) = self.last_right_seen_sec {
            if self.grace_period_sec > 0.0 && (t_sec - last_r) <= self.grace_period_sec {
                // Grace period active: hold gesture and cursor across brief dropout
                right_in_grace = self.stable_right != RightGesture::Idle;
                self.right_deadzone_cursor
            } else {
                self.right_fingers = [false; 4];
                self.right_deadzone_cursor = None;
                let needed = self.required_frames_for_right(RightGesture::Idle);
                if self.stable_right == RightGesture::Idle {
                    self.candidate_right = RightGesture::Idle;
                    self.right_counter = 0;
                    self.right_candidate_since_sec = None;
                } else if self.candidate_right == RightGesture::Idle {
                    self.right_counter = self.right_counter.saturating_add(1);
                    if self.right_counter >= needed {
                        self.stable_right = RightGesture::Idle;
                        self.right_counter = 0;
                        self.right_candidate_since_sec = None;
                    }
                } else {
                    self.candidate_right = RightGesture::Idle;
                    self.right_counter = 1;
                    self.right_candidate_since_sec = Some(t_sec);
                    if needed <= 1 {
                        self.stable_right = RightGesture::Idle;
                    }
                }
                None
            }
        } else {
            self.right_fingers = [false; 4];
            self.right_deadzone_cursor = None;
            None
        };

        let in_grace_period = left_in_grace || right_in_grace;
        if left_hand.is_none() && right_hand.is_none() && !in_grace_period {
            self.no_hand_frames = self.no_hand_frames.saturating_add(1);
        } else {
            self.no_hand_frames = 0;
        }

        if self.no_hand_frames >= 5 && !in_grace_period {
            self.stable_left = LeftGesture::Idle;
            self.stable_right = RightGesture::Idle;
        }

        let right_hold_progress = if self.candidate_right == self.stable_right {
            1.0
        } else {
            let needed = self.required_frames_for_right(self.candidate_right).max(1) as f32;
            (self.right_counter as f32 / needed).clamp(0.0, 1.0)
        };

        let right_event = match (prev_right, self.stable_right) {
            (a, RightGesture::DrawStroke) if a != RightGesture::DrawStroke => {
                RightTransitionEvent::BeganStroke
            }
            (RightGesture::DrawStroke, b) if b != RightGesture::DrawStroke => {
                RightTransitionEvent::EndedStrokeAndSnap
            }
            (a, RightGesture::SelectBoxVShape) if a != RightGesture::SelectBoxVShape => {
                RightTransitionEvent::BeganSelection
            }
            (RightGesture::SelectBoxVShape, b) if b != RightGesture::SelectBoxVShape => {
                RightTransitionEvent::CompletedSelection
            }
            _ => RightTransitionEvent::None,
        };

        DualHandGestureState {
            left_gesture: self.stable_left,
            right_gesture: self.stable_right,
            left_cursor_norm,
            right_cursor_norm,
            left_pinch_span,
            right_event,
            no_hand_idle: self.no_hand_frames >= 5 && !in_grace_period,
            in_grace_period,
            right_hold_progress,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::hand_tracker::Handedness;

    #[test]
    fn test_distance_invariant_classification_across_scales() {
        for scale in [0.30_f32, 0.75, 1.0, 2.1] {
            let draw = HandLandmarks::from_pose_preset(
                Handedness::Right,
                0.5,
                0.5,
                scale,
                true,
                true,
                false,
                false,
                false,
            );
            assert_eq!(
                classify_right_hand_raw(&draw).0,
                RightGesture::DrawStroke,
                "Draw gesture must remain invariant at scale {scale}"
            );

            let hover = HandLandmarks::from_pose_preset(
                Handedness::Right,
                0.5,
                0.5,
                scale,
                false,
                true,
                true,
                false,
                false,
            );
            assert_eq!(
                classify_right_hand_raw(&hover).0,
                RightGesture::HoverTwoFingers,
                "Hover gesture must remain invariant at scale {scale}"
            );

            let erase = HandLandmarks::from_pose_preset(
                Handedness::Right,
                0.5,
                0.5,
                scale,
                false,
                true,
                true,
                true,
                true,
            );
            assert_eq!(
                classify_right_hand_raw(&erase).0,
                RightGesture::PalmEraser,
                "Eraser gesture must remain invariant at scale {scale}"
            );
        }
    }

    #[test]
    fn test_production_fsm_requires_hold_window_and_blocks_transient_glitches() {
        let mut fsm = GestureDebounceFsm::production();
        let draw_hand = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.7,
            0.5,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );

        // First 4 frames (<150ms & <6 frames) must NOT trigger DrawStroke prematurely
        for i in 1..=5 {
            let st = fsm.update_at(None, Some(&draw_hand), i as f64 * 0.030);
            assert_eq!(st.right_gesture, RightGesture::Idle, "Triggered too early at frame {i}");
        }

        // Frame 7 reaches 6 candidate repeats AND >=175ms elapsed -> confirmed transition!
        let s6 = fsm.update_at(None, Some(&draw_hand), 7.0 * 0.030);
        assert_eq!(s6.right_gesture, RightGesture::DrawStroke);
        assert_eq!(s6.right_event, RightTransitionEvent::BeganStroke);

        // Single-frame accidental eraser spike while drawing must be rejected
        let erase_spike = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.7,
            0.5,
            1.0,
            false,
            true,
            true,
            true,
            true,
        );
        let spike_state = fsm.update_at(None, Some(&erase_spike), 8.0 * 0.030);
        assert_eq!(spike_state.right_gesture, RightGesture::DrawStroke);
    }

    #[test]
    fn test_deadzone_ignores_stationary_micro_tremor() {
        let mut fsm = GestureDebounceFsm::new(2);
        let hand1 = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.500,
            0.500,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );
        let s1 = fsm.update_at(None, Some(&hand1), 0.1);
        let c1 = s1.right_cursor_norm.unwrap();

        // Micro tremor of 0.001 (below palm-scaled deadzone)
        let hand_tremor = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.501,
            0.501,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );
        let s2 = fsm.update_at(None, Some(&hand_tremor), 0.13);
        let c2 = s2.right_cursor_norm.unwrap();
        assert_eq!(c1, c2, "Micro tremor below deadzone must keep exact stationary cursor");

        // Larger intentional motion exceeds deadzone and updates cursor
        let hand_move = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.54,
            0.53,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );
        let s3 = fsm.update_at(None, Some(&hand_move), 0.16);
        let c3 = s3.right_cursor_norm.unwrap();
        assert_ne!(c1, c3, "Intentional motion above deadzone must move cursor");
    }

    #[test]
    fn test_rotation_invariant_tilted_hand_classification() {
        let mut draw_tilted = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.5,
            0.5,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );
        // Rotate entire 21-landmark skeleton by +40 degrees around (0.5, 0.5)
        let angle = 40.0_f32.to_radians();
        let (sin_a, cos_a) = angle.sin_cos();
        for lm in &mut draw_tilted.landmarks {
            let dx = lm.x - 0.5;
            let dy = lm.y - 0.5;
            lm.x = 0.5 + dx * cos_a - dy * sin_a;
            lm.y = 0.5 + dx * sin_a + dy * cos_a;
        }
        assert_eq!(
            classify_right_hand_raw(&draw_tilted).0,
            RightGesture::DrawStroke,
            "Rotated 40-deg Draw pose must still classify via joint dot-products"
        );
    }

    #[test]
    fn test_grace_period_150ms_survives_brief_dropout_then_releases() {
        let mut fsm = GestureDebounceFsm::production();
        let draw_hand = HandLandmarks::from_pose_preset(
            Handedness::Right,
            0.65,
            0.52,
            1.0,
            true,
            true,
            false,
            false,
            false,
        );
        for i in 1..=7 {
            fsm.update_at(None, Some(&draw_hand), i as f64 * 0.030);
        }
        let locked = fsm.update_at(None, Some(&draw_hand), 0.240);
        assert_eq!(locked.right_gesture, RightGesture::DrawStroke);

        // Brief 110ms tracking dropout (< 150ms grace period) must preserve DrawStroke
        let dropout = fsm.update_at(None, None, 0.350);
        assert_eq!(dropout.right_gesture, RightGesture::DrawStroke);
        assert!(dropout.in_grace_period);
        assert!(!dropout.no_hand_idle);

        // Once > 150ms passes cleanly without hand + idle hold finishes, release cleanly to Idle
        let mut t = 0.420;
        let mut released = dropout;
        for _ in 0..8 {
            released = fsm.update_at(None, None, t);
            t += 0.035;
        }
        assert_eq!(released.right_gesture, RightGesture::Idle);
    }
}
