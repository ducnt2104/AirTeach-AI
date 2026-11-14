#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Accurate 3D Hand Gesture Engine (`cv_engine/hand_gesture.py`)
for AirTeach AI (DNT - Victer).

Strictly implements:
1. MediaPipe Hands / Hand Landmarker (`model_complexity=1` Full),
   `min_detection_confidence=0.6`, `min_tracking_confidence=0.7`.
2. ROI Dynamic Zoom Crop:
   - When hand bounding box area `< 8%` (`< 0.08`) of frame area, crops expanded ROI around the hand
     with `+35%` padding (`pad_ratio=0.35`), upscales to `384x384` using `cv2.INTER_CUBIC`,
     and re-runs landmark extraction on the ROI to recover pinpoint 21-joint 3D coordinates.
3. 3D Vector Geometry Gesture Classification (Rotation- & Distance-Invariant, ZERO raw Y-coordinate comparison):
   - Computes inter-bone joint flexion angles via `arccos(dot(u, v) / (|u| * |v|))` across MCP-PIP-DIP-TIP.
   - Computes thumb abduction angle relative to the palm plane (`cross(LM5 - LM0, LM17 - LM0)`).
   - Normalizes all distances by palm scale (`dist(landmark[0], landmark[9])`), including
     `Pinch Ratio = dist(4, 8) / dist(0, 9)`.
4. One-Euro Filter (`OneEuroFilter1D`, `OneEuroFilter2D`, `OneEuroFilter3D`) with configurable `min_cutoff` & `beta`.
5. Gesture FSM: `IDLE -> CANDIDATE (>= 5 frames / >= 150ms) -> ACTIVE -> GRACE_PERIOD (150ms) -> IDLE`.
6. Frame Skipping (`FrameSkipLock` / `is_processing` lock) so worker never queues stale frames.
"""

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Standard 21 MediaPipe Hand Landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def ensure_hand_model(path: Path, url: str = HAND_MODEL_URL) -> bool:
    if path.exists() and path.stat().st_size > 50_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        return path.exists() and path.stat().st_size > 50_000
    except OSError as exc:
        _ = exc
        return False


# =============================================================================
# 3D VECTOR GEOMETRY HELPERS (ROTATION- & DISTANCE-INVARIANT)
# =============================================================================
def _vec3(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Return 3D vector from landmark `a` to landmark `b` (`b - a`)."""
    return np.array(
        [float(b[0]) - float(a[0]), float(b[1]) - float(a[1]), float(b[2]) - float(a[2])],
        dtype=np.float64,
    )


def dist_3d(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_palm_scale_3d(lm: Sequence[Sequence[float]]) -> float:
    """Reference palm length `dist(landmark[0], landmark[9])`."""
    if len(lm) < 10:
        return 1e-4
    return max(1e-4, dist_3d(lm[WRIST], lm[MIDDLE_MCP]))


def compute_pinch_ratio_3d(lm: Sequence[Sequence[float]]) -> float:
    """Palm-normalized Pinch Ratio `dist(4, 8) / dist(0, 9)`."""
    if len(lm) < 10:
        return 1.0
    palm = compute_palm_scale_3d(lm)
    return dist_3d(lm[THUMB_TIP], lm[INDEX_TIP]) / palm


def joint_angle_3d_deg(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Compute 3D angle in degrees between bone vectors `u` and `v`:
    `degrees(arccos(clip(dot(u, v) / (|u| * |v|), -1.0, 1.0)))`.
    `0 deg` = perfectly straight bone chain; `> 55 deg` = flexed/curled joint.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    nu = float(np.linalg.norm(u_arr))
    nv = float(np.linalg.norm(v_arr))
    if nu < 1e-7 or nv < 1e-7:
        return 0.0
    cos_theta = float(np.dot(u_arr, v_arr) / (nu * nv))
    cos_clamped = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_clamped))


def finger_joint_angles_deg(
    lm: Sequence[Sequence[float]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
) -> Tuple[float, float, float]:
    """
    Return `(pip_angle_deg, dip_angle_deg, overall_angle_deg)` for a 4-joint finger chain
    (`MCP -> PIP -> DIP -> TIP`) using 3D `arccos(dot(u, v) / (|u||v|))`.
    """
    u1 = _vec3(lm[mcp_idx], lm[pip_idx])
    u2 = _vec3(lm[pip_idx], lm[dip_idx])
    u3 = _vec3(lm[dip_idx], lm[tip_idx])
    u_overall = _vec3(lm[pip_idx], lm[tip_idx])
    return (
        joint_angle_3d_deg(u1, u2),
        joint_angle_3d_deg(u2, u3),
        joint_angle_3d_deg(u1, u_overall),
    )


def is_finger_extended_3d(
    lm: Sequence[Sequence[float]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
    was_extended: bool = False,
) -> bool:
    """
    Rotation- and distance-invariant 3D finger extension test using:
    1. Bone vector joint angles (`MCP-PIP-DIP-TIP`) via `arccos(dot(u, v) / (|u||v|))`
    2. Palm-normalized 3D radial reach `dist(0, TIP) / dist(0, PIP)` and `dist(0, TIP) / dist(0, 9)`.
    Never compares raw Y coordinates.
    """
    palm_scale = compute_palm_scale_3d(lm)
    pip_ang, dip_ang, overall_ang = finger_joint_angles_deg(lm, mcp_idx, pip_idx, dip_idx, tip_idx)

    r_pip = dist_3d(lm[WRIST], lm[pip_idx]) / palm_scale
    r_tip = dist_3d(lm[WRIST], lm[tip_idx]) / palm_scale
    radial_ratio = r_tip / max(0.15, r_pip)

    if was_extended:
        return (
            pip_ang < 54.0
            and dip_ang < 58.0
            and overall_ang < 48.0
            and radial_ratio > 1.05
            and r_tip > 1.05
        )
    return (
        pip_ang < 42.0
        and dip_ang < 46.0
        and overall_ang < 36.0
        and radial_ratio > 1.14
        and r_tip > 1.22
    )


def thumb_palm_plane_angle_deg(lm: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """
    Compute thumb 3D geometry relative to the palm plane:
    - `palm_normal = cross(LM5 - LM0, LM17 - LM0)`
    - `abduction_palm_deg`: angle of thumb vector `(LM4 - LM2)` across the palm plane & index base
    - `ip_flexion_deg`: inter-bone flexion angle `arccos(dot(LM2-LM1, LM4-LM3) / ...)`
    - `spread_ratio`: palm-normalized distance `dist(LM4, LM5) / dist(LM0, LM9)`
    """
    palm_scale = compute_palm_scale_3d(lm)
    v_index = _vec3(lm[WRIST], lm[INDEX_MCP])
    v_pinky = _vec3(lm[WRIST], lm[PINKY_MCP])
    palm_normal = np.cross(v_index, v_pinky)
    n_norm = float(np.linalg.norm(palm_normal))

    v_thumb = _vec3(lm[THUMB_MCP], lm[THUMB_TIP])
    t_norm = float(np.linalg.norm(v_thumb))

    # Angle between thumb vector and palm index axis (in-plane + out-of-plane abduction)
    abduction_deg = joint_angle_3d_deg(v_thumb, v_index)
    if n_norm > 1e-7 and t_norm > 1e-7:
        sin_elev = abs(float(np.dot(v_thumb, palm_normal)) / (t_norm * n_norm))
        elev_deg = math.degrees(math.asin(max(0.0, min(1.0, sin_elev))))
        abduction_deg = max(abduction_deg, elev_deg)

    u_cmc_mcp = _vec3(lm[THUMB_CMC], lm[THUMB_MCP])
    u_ip_tip = _vec3(lm[THUMB_IP], lm[THUMB_TIP])
    ip_flexion_deg = joint_angle_3d_deg(u_cmc_mcp, u_ip_tip)
    spread_ratio = dist_3d(lm[THUMB_TIP], lm[INDEX_MCP]) / palm_scale
    return abduction_deg, ip_flexion_deg, spread_ratio


def is_thumb_extended_3d(lm: Sequence[Sequence[float]], was_extended: bool = False) -> bool:
    abduction_deg, ip_flexion_deg, spread_ratio = thumb_palm_plane_angle_deg(lm)
    if was_extended:
        return ip_flexion_deg < 62.0 and spread_ratio > 0.36 and abduction_deg > 14.0
    return ip_flexion_deg < 48.0 and spread_ratio > 0.45 and abduction_deg > 18.0


def classify_hand_3d_vectors(
    lm: Sequence[Sequence[float]],
    prev_fingers: Optional[List[bool]] = None,
    was_pinched: bool = False,
) -> Dict[str, object]:
    """
    Classify 21-landmark 3D hand skeleton using purely rotation- and scale-invariant 3D vector
    joint angles (`arccos(dot(u,v)/(|u||v|))`) and palm-normalized Pinch Ratio (`dist(4,8)/dist(0,9)`).
    Returns:
      `{"gesture": str, "cursor_norm": (float, float), "pinch_ratio": float, "palm_scale": float, "fingers": List[bool]}`
    """
    if len(lm) < 21:
        return {
            "gesture": "IDLE",
            "cursor_norm": (0.5, 0.5),
            "pinch_ratio": 1.0,
            "palm_scale": 0.1,
            "fingers": [False, False, False, False, False],
        }

    state = list(prev_fingers) if (prev_fingers and len(prev_fingers) == 5) else [False] * 5
    thumb_up = is_thumb_extended_3d(lm, state[0])
    idx_up = is_finger_extended_3d(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, state[1])
    mid_up = is_finger_extended_3d(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, state[2])
    ring_up = is_finger_extended_3d(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, state[3])
    pinky_up = is_finger_extended_3d(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, state[4])
    fingers = [thumb_up, idx_up, mid_up, ring_up, pinky_up]

    palm_scale = compute_palm_scale_3d(lm)
    pinch_ratio = compute_pinch_ratio_3d(lm)
    pinch_thresh = 0.40 if was_pinched else 0.28
    index_reach = dist_3d(lm[WRIST], lm[INDEX_TIP]) / palm_scale

    def map_box(x: float, y: float) -> Tuple[float, float]:
        margin = 0.12
        mx = min(1.0, max(0.0, (float(x) - margin) / (1.0 - 2.0 * margin)))
        my = min(1.0, max(0.0, (float(y) - margin) / (1.0 - 2.0 * margin)))
        return (mx, my)

    # 1. Open Palm (all 4 fingers up, not pinched): CANCEL while drawing / ERASE when idle
    if idx_up and mid_up and ring_up and pinky_up and pinch_ratio > 0.38:
        return {
            "gesture": "ERASE",
            "pose_tag": "OPEN_PALM",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 2. Tight Fist (all 4 fingers curled tightly into palm & not forward pinching): CANCEL
    is_tight_fist = (not idx_up and not mid_up and not ring_up and not pinky_up) and index_reach < 1.15
    if is_tight_fist:
        return {
            "gesture": "FIST",
            "pose_tag": "FIST",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 3. Thumb-Index Pinch (`dist(4,8)/dist(0,9) < 0.28` enter, `> 0.40` exit via Schmitt Trigger)
    if pinch_ratio < pinch_thresh:
        px = 0.5 * (float(lm[THUMB_TIP][0]) + float(lm[INDEX_TIP][0]))
        py = 0.5 * (float(lm[THUMB_TIP][1]) + float(lm[INDEX_TIP][1]))
        return {
            "gesture": "CLICK",
            "pose_tag": "PINCH",
            "cursor_norm": map_box(px, py),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 4. Three Fingers (Index + Middle + Ring up, Pinky curled, unpinched): UNDO Trigger (220ms)
    if idx_up and mid_up and ring_up and not pinky_up and pinch_ratio > 0.40:
        cx = 0.5 * (float(lm[INDEX_TIP][0]) + float(lm[MIDDLE_TIP][0]))
        cy = 0.5 * (float(lm[INDEX_TIP][1]) + float(lm[MIDDLE_TIP][1]))
        return {
            "gesture": "UNDO",
            "pose_tag": "THREE_FINGERS",
            "cursor_norm": map_box(cx, cy),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 5. Two Fingers (Index + Middle up, Ring curled): HOVER cursor
    if idx_up and mid_up and not ring_up:
        cx = 0.5 * (float(lm[INDEX_TIP][0]) + float(lm[MIDDLE_TIP][0]))
        cy = 0.5 * (float(lm[INDEX_TIP][1]) + float(lm[MIDDLE_TIP][1]))
        return {
            "gesture": "HOVER",
            "pose_tag": "TWO_FINGERS",
            "cursor_norm": map_box(cx, cy),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 6. Index Extended (unpinched): INDEX pose (points / HOVER under Pinch Schmitt FSM)
    if idx_up and not mid_up and not ring_up:
        return {
            "gesture": "DRAW",
            "pose_tag": "INDEX",
            "cursor_norm": map_box(lm[INDEX_TIP][0], lm[INDEX_TIP][1]),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    # 7. Loose curl fallback
    if not idx_up and not mid_up and not ring_up and not pinky_up:
        return {
            "gesture": "FIST",
            "pose_tag": "FIST",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
            "pinch_ratio": round(pinch_ratio, 4),
            "palm_scale": round(palm_scale, 4),
            "fingers": fingers,
        }

    return {
        "gesture": "HOVER",
        "pose_tag": "HOVER",
        "cursor_norm": map_box(lm[INDEX_TIP][0], lm[INDEX_TIP][1]),
        "pinch_ratio": round(pinch_ratio, 4),
        "palm_scale": round(palm_scale, 4),
        "fingers": fingers,
    }


# =============================================================================
# ONE-EURO FILTER (1D / 2D / 3D) & GESTURE FSM + FRAME SKIPPING LOCK
# =============================================================================
class OneEuroFilter1D:
    """Adaptive low-pass One-Euro Filter tuning cutoff frequency by instantaneous speed."""

    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-5, dt))

    def filter(self, x: float, t_sec: Optional[float] = None) -> float:
        now = time.perf_counter() if t_sec is None else float(t_sec)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = float(x)
            self._t_prev = now
            return float(x)

        dt = max(1e-4, min(0.25, now - self._t_prev))
        self._t_prev = now

        dx = (float(x) - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a_x = self._alpha(cutoff, dt)
        x_hat = a_x * float(x) + (1.0 - a_x) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def filter(self, pt: Tuple[float, float], t_sec: Optional[float] = None) -> Tuple[float, float]:
        return (self.fx.filter(pt[0], t_sec), self.fy.filter(pt[1], t_sec))


class OneEuroFilter3D:
    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()

    def filter(self, pt: Sequence[float], t_sec: Optional[float] = None) -> List[float]:
        return [
            self.fx.filter(float(pt[0]), t_sec),
            self.fy.filter(float(pt[1]), t_sec),
            self.fz.filter(float(pt[2]), t_sec),
        ]


class GestureFsmState(str, Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    GRACE_PERIOD = "GRACE_PERIOD"


@dataclass
class GestureFsmSnapshot:
    fsm_state: GestureFsmState
    active_gesture: str
    candidate_gesture: str
    cursor_norm: Optional[Tuple[float, float]]
    hold_frames: int
    in_grace_period: bool


class GestureFsmEngine:
    """
    Production Gesture Finite State Machine:
    `IDLE -> CANDIDATE (hold >= 5 frames / >= 150ms) -> ACTIVE -> GRACE_PERIOD (150ms) -> IDLE`
    """

    def __init__(
        self,
        min_hold_frames: int = 5,
        min_hold_sec: float = 0.150,
        grace_period_sec: float = 0.150,
        min_cutoff: float = 1.35,
        beta: float = 0.015,
    ) -> None:
        self.min_hold_frames = max(1, int(min_hold_frames))
        self.min_hold_sec = max(0.050, float(min_hold_sec))
        self.grace_period_sec = max(0.050, float(grace_period_sec))
        self.cursor_filter = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)
        self.fsm_state = GestureFsmState.IDLE
        self.active_gesture = "IDLE"
        self.candidate_gesture = "IDLE"
        self.candidate_count = 0
        self.candidate_since_sec: Optional[float] = None
        self.last_seen_sec: Optional[float] = None
        self.last_cursor: Optional[Tuple[float, float]] = None
        self.prev_fingers: List[bool] = [False] * 5

    def update(
        self,
        landmarks: Optional[Sequence[Sequence[float]]],
        t_sec: Optional[float] = None,
    ) -> GestureFsmSnapshot:
        now = time.perf_counter() if t_sec is None else float(t_sec)

        if landmarks is not None and len(landmarks) >= 21:
            self.last_seen_sec = now
            was_pinched = self.active_gesture == "CLICK"
            info = classify_hand_3d_vectors(landmarks, self.prev_fingers, was_pinched=was_pinched)
            self.prev_fingers = list(info["fingers"])  # type: ignore[arg-type]
            raw_gesture = str(info["gesture"])
            raw_cursor = info["cursor_norm"]  # type: ignore[assignment]
            smoothed_cursor = self.cursor_filter.filter(raw_cursor, now)  # type: ignore[arg-type]
            self.last_cursor = smoothed_cursor

            if raw_gesture == self.active_gesture:
                self.candidate_gesture = raw_gesture
                self.candidate_count = 0
                self.candidate_since_sec = None
                self.fsm_state = (
                    GestureFsmState.IDLE if self.active_gesture == "IDLE" else GestureFsmState.ACTIVE
                )
            elif raw_gesture == self.candidate_gesture:
                self.candidate_count += 1
                elapsed = now - (self.candidate_since_sec if self.candidate_since_sec is not None else now)
                self.fsm_state = GestureFsmState.CANDIDATE
                if self.candidate_count >= self.min_hold_frames and elapsed >= self.min_hold_sec:
                    self.active_gesture = raw_gesture
                    self.candidate_count = 0
                    self.candidate_since_sec = None
                    self.fsm_state = (
                        GestureFsmState.IDLE if raw_gesture == "IDLE" else GestureFsmState.ACTIVE
                    )
            else:
                self.candidate_gesture = raw_gesture
                self.candidate_count = 1
                self.candidate_since_sec = now
                self.fsm_state = GestureFsmState.CANDIDATE

            return GestureFsmSnapshot(
                fsm_state=self.fsm_state,
                active_gesture=self.active_gesture,
                candidate_gesture=self.candidate_gesture,
                cursor_norm=smoothed_cursor,
                hold_frames=self.candidate_count,
                in_grace_period=False,
            )

        # Hand temporarily lost -> check 150ms GRACE_PERIOD
        if (
            self.last_seen_sec is not None
            and self.active_gesture != "IDLE"
            and (now - self.last_seen_sec) <= self.grace_period_sec
        ):
            self.fsm_state = GestureFsmState.GRACE_PERIOD
            return GestureFsmSnapshot(
                fsm_state=self.fsm_state,
                active_gesture=self.active_gesture,
                candidate_gesture=self.candidate_gesture,
                cursor_norm=self.last_cursor,
                hold_frames=self.candidate_count,
                in_grace_period=True,
            )

        self.fsm_state = GestureFsmState.IDLE
        self.active_gesture = "IDLE"
        self.candidate_gesture = "IDLE"
        self.candidate_count = 0
        self.candidate_since_sec = None
        self.prev_fingers = [False] * 5
        self.last_cursor = None
        self.cursor_filter.reset()
        return GestureFsmSnapshot(
            fsm_state=GestureFsmState.IDLE,
            active_gesture="IDLE",
            candidate_gesture="IDLE",
            cursor_norm=None,
            hold_frames=0,
            in_grace_period=False,
        )


class FrameSkipLock:
    """
    Non-blocking `isProcessing` atomic lock ensuring only the newest camera frame is processed
    when the CV worker is idle; never queues frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.skipped_frames = 0

    def try_acquire(self) -> bool:
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            self.skipped_frames += 1
        return acquired

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()


# =============================================================================
# MEDIAPIPE HAND ENGINE WITH ROI DYNAMIC ZOOM CROP (< 8% -> +35% -> 384x384)
# =============================================================================
class HandGestureEngine:
    """
    High-Precision Hand Landmark & Gesture Engine:
    - MediaPipe Full (`model_complexity=1`), `min_detection_confidence=0.6`, `min_tracking_confidence=0.7`.
    - ROI Dynamic Zoom Crop (`bbox_area < 0.08`, `+35%` padding, `384x384` `cv2.INTER_CUBIC`).
    """

    ROI_AREA_THRESHOLD = 0.08
    ROI_PADDING_RATIO = 0.35
    ROI_TARGET_SIZE = 384

    def __init__(
        self,
        model_path: Path = Path("models/hand_landmarker.task"),
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.70,
    ) -> None:
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.model_complexity = 1
        self.solutions_hands = None
        self.tasks_detector = None
        self.left_fsm = GestureFsmEngine(min_hold_frames=5, min_hold_sec=0.150, grace_period_sec=0.150)
        self.right_fsm = GestureFsmEngine(min_hold_frames=5, min_hold_sec=0.150, grace_period_sec=0.150)
        self.frame_skip_lock = FrameSkipLock()

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.solutions_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            except RuntimeError as exc:
                _ = exc
                self.solutions_hands = None

        if ensure_hand_model(model_path, HAND_MODEL_URL):
            try:
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.tasks_detector = mp_vision.HandLandmarker.create_from_options(opts)
            except RuntimeError as exc:
                _ = exc
                self.tasks_detector = None

    @staticmethod
    def validate_real_hand(
        lm_list: Sequence[Sequence[float]],
        score: float,
        face_boxes: Sequence[Sequence[float]],
        min_score: float = 0.45,
    ) -> bool:
        if score < min_score or len(lm_list) < 21:
            return False

        xs = [float(p[0]) for p in lm_list]
        ys = [float(p[1]) for p in lm_list]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        if (bbox_w * bbox_h) <= 0.008:
            return False

        px, py = float(lm_list[MIDDLE_MCP][0]), float(lm_list[MIDDLE_MCP][1])
        for fx, fy, fw, fh in face_boxes:
            face_cx = float(fx) + float(fw) * 0.5
            face_cy = float(fy) + float(fh) * 0.5
            radius = max(float(fw), float(fh)) * 0.42
            if math.hypot(px - face_cx, py - face_cy) < radius:
                return False

        return True

    def refine_hand_roi_zoom(
        self,
        rgb_frame: np.ndarray,
        lm_coords: List[List[float]],
        orig_score: float,
        orig_cat: str,
        face_boxes: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], float, str]:
        """
        ROI Dynamic Zoom Crop:
        When hand bounding box area `< 8%` (`< 0.08`) of frame area, crop around the hand with `+35%`
        padding, resize to `384x384` via `cv2.INTER_CUBIC`, and re-extract 21 landmarks.
        """
        if self.tasks_detector is None:
            return lm_coords, orig_score, orig_cat

        h, w = rgb_frame.shape[:2]
        xs = [float(p[0]) for p in lm_coords]
        ys = [float(p[1]) for p in lm_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max_x - min_x
        bh = max_y - min_y
        bbox_area = bw * bh

        if bbox_area >= self.ROI_AREA_THRESHOLD or bw <= 0.01 or bh <= 0.01:
            return lm_coords, orig_score, orig_cat

        pad_x = max(bw, bh) * self.ROI_PADDING_RATIO
        pad_y = max(bw, bh) * self.ROI_PADDING_RATIO
        x0 = max(0, int(math.floor((min_x - pad_x) * w)))
        y0 = max(0, int(math.floor((min_y - pad_y) * h)))
        x1 = min(w, int(math.ceil((max_x + pad_x) * w)))
        y1 = min(h, int(math.ceil((max_y + pad_y) * h)))
        cw = x1 - x0
        ch = y1 - y0
        if cw < 20 or ch < 20:
            return lm_coords, orig_score, orig_cat

        roi = rgb_frame[y0:y1, x0:x1]
        roi_384 = cv2.resize(
            roi,
            (self.ROI_TARGET_SIZE, self.ROI_TARGET_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        try:
            roi_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_384)
            roi_res = self.tasks_detector.detect(roi_mp)
            if not roi_res or not roi_res.hand_landmarks or not roi_res.handedness:
                return lm_coords, orig_score, orig_cat

            orig_cx = float(lm_coords[MIDDLE_MCP][0])
            orig_cy = float(lm_coords[MIDDLE_MCP][1])
            max_center_drift = max(bw, bh) * 0.85

            best_candidate: Optional[Tuple[List[List[float]], float, str]] = None
            for lms, handed_list in zip(roi_res.hand_landmarks, roi_res.handedness):
                if not handed_list:
                    continue
                top_cat = handed_list[0]
                r_score = float(top_cat.score)
                r_cat = (top_cat.category_name or "").lower() or orig_cat
                mapped = [
                    [
                        round(max(0.0, min(1.0, (x0 + float(p.x) * cw) / float(w))), 4),
                        round(max(0.0, min(1.0, (y0 + float(p.y) * ch) / float(h))), 4),
                        round(float(p.z) * (float(cw) / float(w)), 4),
                    ]
                    for p in lms
                ]
                if not self.validate_real_hand(mapped, r_score, face_boxes):
                    continue
                cand_cx = float(mapped[MIDDLE_MCP][0])
                cand_cy = float(mapped[MIDDLE_MCP][1])
                if math.hypot(cand_cx - orig_cx, cand_cy - orig_cy) > max_center_drift:
                    continue
                if r_score >= orig_score * 0.90:
                    if best_candidate is None or r_score > best_candidate[1]:
                        best_candidate = (mapped, r_score, r_cat)

            if best_candidate is not None:
                return best_candidate
        except RuntimeError as exc:
            _ = exc

        return lm_coords, orig_score, orig_cat

    def detect_hands(
        self,
        rgb_frame: np.ndarray,
        face_boxes: Sequence[Sequence[float]],
    ) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        """
        Detect Left and Right hands in an RGB frame, applying ROI Dynamic Zoom Crop (`384x384` `INTER_CUBIC`)
        when bbox area `< 8%`, and attaching 3D vector gesture metadata.
        """
        left_out: Optional[Dict[str, object]] = None
        right_out: Optional[Dict[str, object]] = None

        if self.tasks_detector is None:
            return None, None

        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            h_res = self.tasks_detector.detect(mp_img)
            if h_res and h_res.hand_landmarks and h_res.handedness:
                valid_hands: List[Tuple[str, Dict[str, object]]] = []
                for lms, handed_list in zip(h_res.hand_landmarks, h_res.handedness):
                    if not handed_list:
                        continue
                    top_cat = handed_list[0]
                    score = float(top_cat.score)
                    cat_name = (top_cat.category_name or "").lower()
                    lm_coords = [
                        [round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)]
                        for p in lms
                    ]
                    if not self.validate_real_hand(lm_coords, score, face_boxes):
                        continue
                    lm_coords, score, cat_name = self.refine_hand_roi_zoom(
                        rgb_frame, lm_coords, score, cat_name, face_boxes
                    )
                    vec_info = classify_hand_3d_vectors(lm_coords)
                    hand_obj: Dict[str, object] = {
                        "score": round(score, 4),
                        "lm": lm_coords,
                        "gesture_3d": vec_info["gesture"],
                        "pose_tag": vec_info.get("pose_tag", vec_info["gesture"]),
                        "pinch_ratio": vec_info["pinch_ratio"],
                    }
                    valid_hands.append((cat_name, hand_obj))

                if len(valid_hands) == 1:
                    right_out = valid_hands[0][1]
                else:
                    for cat_name, hand_obj in valid_hands:
                        if cat_name == "left" and left_out is None:
                            left_out = hand_obj
                        else:
                            right_out = hand_obj

                # Update per-hand GestureFsmEngine (IDLE -> CANDIDATE -> ACTIVE -> GRACE_PERIOD)
                now_sec = time.perf_counter()
                left_snap = self.left_fsm.update(
                    left_out["lm"] if left_out is not None else None,  # type: ignore[arg-type]
                    t_sec=now_sec,
                )
                right_snap = self.right_fsm.update(
                    right_out["lm"] if right_out is not None else None,  # type: ignore[arg-type]
                    t_sec=now_sec,
                )
                if left_out is not None:
                    left_out["fsm_state"] = left_snap.fsm_state.value
                    left_out["active_gesture"] = left_snap.active_gesture
                    if left_snap.cursor_norm is not None:
                        left_out["cursor_norm"] = [
                            round(left_snap.cursor_norm[0], 4),
                            round(left_snap.cursor_norm[1], 4),
                        ]
                if right_out is not None:
                    right_out["fsm_state"] = right_snap.fsm_state.value
                    right_out["active_gesture"] = right_snap.active_gesture
                    if right_snap.cursor_norm is not None:
                        right_out["cursor_norm"] = [
                            round(right_snap.cursor_norm[0], 4),
                            round(right_snap.cursor_norm[1], 4),
                        ]
            else:
                now_sec = time.perf_counter()
                self.left_fsm.update(None, t_sec=now_sec)
                self.right_fsm.update(None, t_sec=now_sec)
        except RuntimeError as exc:
            _ = exc

        return left_out, right_out
