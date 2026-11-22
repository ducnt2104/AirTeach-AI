"""
AirTeach AI — Tier 3 & Tier 4 Real-Time 3D Hand Tracking & Gesture Engine (`cv_engine/hand_gesture.py`).

Upgrades implemented:
1. Tier 3 — 2-Stage Hand Tracking (Long-Range 1.5m–3m & Anti-Motion-Blur):
   - Distinct `min_detection_confidence`, `min_presence_confidence`, `min_tracking_confidence`.
   - Accepts small hands (`bbox_area >= 0.0012`) instead of rejecting distant hands prematurely.
   - Automatic Stage-2 ROI Crop-Zoom (`384x384` `cv2.INTER_CUBIC`) both when detected bbox < 8%
     AND proactively around locked Teacher's Pose wrists / previous hand box (`2.2x` padded ROI)
     when full-frame detection drops a distant or fast-moving hand.
2. Tier 4 — Pure 3D Scale/Rotation-Invariant Gesture Classification & Temporal Stability:
   - 3D bone flexion angles classify every finger into `EXTENDED` / `HALF_BENT` / `CURLED`
     (zero raw Y coordinate checks, zero-division guards throughout).
   - `compute_palm_normal_3d`: cross product `cross(LM5 - LM0, LM17 - LM0)` camera-facing score.
   - `KalmanVelocityPredictor2D` paired with `OneEuroFilter3D` to predict smooth cursor motion
     through 120–180ms motion-blur `GRACE_PERIOD` dropouts.
   - Dual-threshold Pinch Hysteresis (`< 0.28` enter, `> 0.42` exit) + Sliding Window Majority
     Voting (`5 of 7` frames) + `150ms` time debounce + `FrameSkipLock`.
"""

from __future__ import annotations

import math
import threading
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cv_engine.preprocessing import sharpen_and_contrast_roi

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore

    HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    mp_python = None  # type: ignore
    mp_vision = None  # type: ignore
    HAS_MEDIAPIPE = False


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe 21-landmark indices
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

FINGER_CHAINS = (
    ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    ("pinky", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)


class FingerFlexionState(str, Enum):
    EXTENDED = "EXTENDED"
    HALF_BENT = "HALF_BENT"
    CURLED = "CURLED"


def ensure_hand_model(path: Path, url: str = HAND_MODEL_URL) -> bool:
    if path.exists() and path.stat().st_size > 500_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "AirTeachAI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) > 500_000:
            path.write_bytes(data)
            return True
    except Exception:
        return False
    return False


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
    """Reference palm length `dist(landmark[0], landmark[9])` with zero-division floor."""
    if not lm or len(lm) < 10:
        return 1e-4
    return max(1e-4, dist_3d(lm[WRIST], lm[MIDDLE_MCP]))


def compute_pinch_ratio_3d(lm: Sequence[Sequence[float]]) -> float:
    """Palm-normalized Pinch Ratio `dist(4, 8) / dist(0, 9)`."""
    if not lm or len(lm) < 10:
        return 1.0
    palm = compute_palm_scale_3d(lm)
    return dist_3d(lm[THUMB_TIP], lm[INDEX_TIP]) / max(1e-4, palm)


def joint_angle_3d_deg(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Compute 3D angle in degrees between bone vectors `u` and `v`:
    `degrees(arccos(clip(dot(u, v) / (|u| * |v|), -1.0, 1.0)))`.
    `0 deg` = straight bone chain; `> 78 deg` = curled joint.
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


def compute_palm_normal_3d(
    lm: Sequence[Sequence[float]],
    handedness: str = "right",
    min_facing_z: float = 0.20,
) -> Dict[str, object]:
    """
    Compute Palm Normal Vector from `cross(LM5 - LM0, LM17 - LM0)` with zero-division protection.
    Measures how directly the palm/back-of-hand faces the camera vs edge-on side profile.
    """
    if not lm or len(lm) <= PINKY_MCP:
        return {
            "normal": (0.0, 0.0, 1.0),
            "facing_score": 1.0,
            "facing_camera": True,
            "orientation": "FRONT",
        }
    v_index = _vec3(lm[WRIST], lm[INDEX_MCP])
    v_pinky = _vec3(lm[WRIST], lm[PINKY_MCP])
    cross_vec = np.cross(v_index, v_pinky)
    norm = float(np.linalg.norm(cross_vec))
    if norm < 1e-7:
        return {
            "normal": (0.0, 0.0, 0.0),
            "facing_score": 0.0,
            "facing_camera": False,
            "orientation": "SIDE",
        }
    unit = cross_vec / norm
    nz_signed = float(unit[2]) if handedness.lower() != "left" else -float(unit[2])
    facing_score = abs(float(unit[2]))
    if facing_score < max(0.08, min_facing_z * 0.7):
        orientation = "SIDE"
    elif facing_score >= min_facing_z:
        orientation = "FRONT"
    else:
        orientation = "ANGLED"
    return {
        "normal": (round(float(unit[0]), 4), round(float(unit[1]), 4), round(float(unit[2]), 4)),
        "facing_score": round(facing_score, 4),
        "signed_nz": round(nz_signed, 4),
        "facing_camera": facing_score >= float(min_facing_z),
        "orientation": orientation,
    }


def classify_finger_flexion_3d(
    lm: Sequence[Sequence[float]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
    extended_max_deg: float = 42.0,
    half_bent_max_deg: float = 80.0,
) -> Tuple[FingerFlexionState, float]:
    """
    Classify finger flexion purely from 3D bone angles + wrist-normalized radial reach into:
      `EXTENDED` | `HALF_BENT` | `CURLED`.
    """
    palm_scale = compute_palm_scale_3d(lm)
    pip_ang, dip_ang, overall_ang = finger_joint_angles_deg(lm, mcp_idx, pip_idx, dip_idx, tip_idx)
    r_pip = dist_3d(lm[WRIST], lm[pip_idx]) / palm_scale
    r_tip = dist_3d(lm[WRIST], lm[tip_idx]) / palm_scale
    radial_ratio = r_tip / max(0.15, r_pip)

    effective_deg = max(pip_ang, overall_ang * 0.92)
    if (
        pip_ang < extended_max_deg
        and dip_ang < (extended_max_deg + 6.0)
        and overall_ang < extended_max_deg
        and radial_ratio > 1.12
        and r_tip > 1.18
    ):
        return FingerFlexionState.EXTENDED, round(effective_deg, 1)
    if effective_deg < half_bent_max_deg and radial_ratio > 0.92 and r_tip > 0.95:
        return FingerFlexionState.HALF_BENT, round(effective_deg, 1)
    return FingerFlexionState.CURLED, round(effective_deg, 1)


def is_finger_extended_3d(
    lm: Sequence[Sequence[float]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
    was_extended: bool = False,
) -> bool:
    """
    Rotation- and distance-invariant 3D finger extension test using 3D bone joint angles
    and palm-normalized radial reach (`dist(0, TIP)/dist(0, PIP)`). Never compares raw Y.
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
    Compute thumb 3D geometry relative to the palm plane (`palm_normal = cross(LM5-LM0, LM17-LM0)`).
    Returns `(abduction_deg, ip_flexion_deg, spread_ratio)`.
    """
    palm_scale = compute_palm_scale_3d(lm)
    v_index = _vec3(lm[WRIST], lm[INDEX_MCP])
    v_pinky = _vec3(lm[WRIST], lm[PINKY_MCP])
    palm_normal = np.cross(v_index, v_pinky)
    n_norm = float(np.linalg.norm(palm_normal))

    v_thumb = _vec3(lm[THUMB_MCP], lm[THUMB_TIP])
    t_norm = float(np.linalg.norm(v_thumb))

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


def classify_thumb_flexion_3d(lm: Sequence[Sequence[float]]) -> Tuple[FingerFlexionState, float]:
    abduction_deg, ip_flexion_deg, spread_ratio = thumb_palm_plane_angle_deg(lm)
    if ip_flexion_deg < 48.0 and spread_ratio > 0.45 and abduction_deg > 18.0:
        return FingerFlexionState.EXTENDED, round(ip_flexion_deg, 1)
    if ip_flexion_deg < 76.0 and spread_ratio > 0.30:
        return FingerFlexionState.HALF_BENT, round(ip_flexion_deg, 1)
    return FingerFlexionState.CURLED, round(ip_flexion_deg, 1)


def is_thumb_extended_3d(lm: Sequence[Sequence[float]], was_extended: bool = False) -> bool:
    abduction_deg, ip_flexion_deg, spread_ratio = thumb_palm_plane_angle_deg(lm)
    if was_extended:
        return ip_flexion_deg < 62.0 and spread_ratio > 0.36 and abduction_deg > 14.0
    return ip_flexion_deg < 48.0 and spread_ratio > 0.45 and abduction_deg > 18.0


def classify_hand_3d_vectors(
    lm: Sequence[Sequence[float]],
    prev_fingers: Optional[List[bool]] = None,
    was_pinched: bool = False,
    pinch_enter_ratio: float = 0.28,
    pinch_exit_ratio: float = 0.42,
    handedness: str = "right",
    palm_facing_min_z: float = 0.20,
) -> Dict[str, object]:
    """
    Classify 21-landmark 3D hand skeleton using purely rotation- and scale-invariant 3D vector
    joint angles (`arccos(dot(u,v)/(|u||v|))`), 3-level finger flexion (`EXTENDED`/`HALF_BENT`/`CURLED`),
    Palm Normal vector orientation check, and palm-normalized Pinch Ratio hysteresis.
    """
    if not lm or len(lm) < 21:
        return {
            "gesture": "IDLE",
            "pose_tag": "IDLE",
            "cursor_norm": (0.5, 0.5),
            "pinch_ratio": 1.0,
            "palm_scale": 0.1,
            "fingers": [False, False, False, False, False],
            "finger_states": ["CURLED"] * 5,
            "finger_angles_deg": [0.0] * 5,
            "palm_normal": (0.0, 0.0, 0.0),
            "palm_facing_score": 0.0,
            "palm_facing_ok": False,
            "palm_orientation": "SIDE",
        }

    state = list(prev_fingers) if (prev_fingers and len(prev_fingers) == 5) else [False] * 5
    thumb_up = is_thumb_extended_3d(lm, state[0])
    idx_up = is_finger_extended_3d(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, state[1])
    mid_up = is_finger_extended_3d(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, state[2])
    ring_up = is_finger_extended_3d(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, state[3])
    pinky_up = is_finger_extended_3d(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, state[4])
    fingers = [thumb_up, idx_up, mid_up, ring_up, pinky_up]

    t_state, t_ang = classify_thumb_flexion_3d(lm)
    i_state, i_ang = classify_finger_flexion_3d(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    m_state, m_ang = classify_finger_flexion_3d(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    r_state, r_ang = classify_finger_flexion_3d(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP)
    p_state, p_ang = classify_finger_flexion_3d(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)
    finger_states = [t_state.value, i_state.value, m_state.value, r_state.value, p_state.value]
    finger_angles = [t_ang, i_ang, m_ang, r_ang, p_ang]

    palm_scale = compute_palm_scale_3d(lm)
    pinch_ratio = compute_pinch_ratio_3d(lm)
    pinch_thresh = float(pinch_exit_ratio) if was_pinched else float(pinch_enter_ratio)
    index_reach = dist_3d(lm[WRIST], lm[INDEX_TIP]) / palm_scale

    palm_norm_info = compute_palm_normal_3d(lm, handedness=handedness, min_facing_z=palm_facing_min_z)

    def map_box(x: float, y: float) -> Tuple[float, float]:
        margin = 0.12
        mx = min(1.0, max(0.0, (float(x) - margin) / (1.0 - 2.0 * margin)))
        my = min(1.0, max(0.0, (float(y) - margin) / (1.0 - 2.0 * margin)))
        return (mx, my)

    base_meta: Dict[str, object] = {
        "pinch_ratio": round(pinch_ratio, 4),
        "palm_scale": round(palm_scale, 4),
        "fingers": fingers,
        "finger_states": finger_states,
        "finger_angles_deg": finger_angles,
        "palm_normal": palm_norm_info["normal"],
        "palm_facing_score": palm_norm_info["facing_score"],
        "palm_facing_ok": palm_norm_info["facing_camera"],
        "palm_orientation": palm_norm_info["orientation"],
    }

    # 1. Open Palm (all 4 fingers up, not pinched)
    if idx_up and mid_up and ring_up and pinky_up and pinch_ratio > 0.38:
        return {
            **base_meta,
            "gesture": "ERASE",
            "pose_tag": "OPEN_PALM",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
        }

    # 2. Tight Fist (all 4 fingers curled tightly into palm & not forward pinching): CANCEL
    is_tight_fist = (not idx_up and not mid_up and not ring_up and not pinky_up) and index_reach < 1.15
    if is_tight_fist:
        return {
            **base_meta,
            "gesture": "FIST",
            "pose_tag": "FIST",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
        }

    # 3. Thumb-Index Pinch (`dist(4,8)/dist(0,9) < 0.28` enter, `> 0.42` exit via Hysteresis)
    if pinch_ratio < pinch_thresh:
        px = 0.5 * (float(lm[THUMB_TIP][0]) + float(lm[INDEX_TIP][0]))
        py = 0.5 * (float(lm[THUMB_TIP][1]) + float(lm[INDEX_TIP][1]))
        return {
            **base_meta,
            "gesture": "CLICK",
            "pose_tag": "PINCH",
            "cursor_norm": map_box(px, py),
        }

    # 4. Three Fingers (Index + Middle + Ring up, Pinky curled, unpinched): UNDO
    if idx_up and mid_up and ring_up and not pinky_up and pinch_ratio > 0.40:
        cx = 0.5 * (float(lm[INDEX_TIP][0]) + float(lm[MIDDLE_TIP][0]))
        cy = 0.5 * (float(lm[INDEX_TIP][1]) + float(lm[MIDDLE_TIP][1]))
        return {
            **base_meta,
            "gesture": "UNDO",
            "pose_tag": "THREE_FINGERS",
            "cursor_norm": map_box(cx, cy),
        }

    # 5. Two Fingers (Index + Middle up, Ring curled): HOVER cursor
    if idx_up and mid_up and not ring_up:
        cx = 0.5 * (float(lm[INDEX_TIP][0]) + float(lm[MIDDLE_TIP][0]))
        cy = 0.5 * (float(lm[INDEX_TIP][1]) + float(lm[MIDDLE_TIP][1]))
        return {
            **base_meta,
            "gesture": "HOVER",
            "pose_tag": "TWO_FINGERS",
            "cursor_norm": map_box(cx, cy),
        }

    # 6. Index Extended (unpinched): INDEX pose (points / HOVER under Pinch Schmitt FSM)
    if idx_up and not mid_up and not ring_up:
        return {
            **base_meta,
            "gesture": "DRAW",
            "pose_tag": "INDEX",
            "cursor_norm": map_box(lm[INDEX_TIP][0], lm[INDEX_TIP][1]),
        }

    # 7. Loose curl fallback
    if not idx_up and not mid_up and not ring_up and not pinky_up:
        return {
            **base_meta,
            "gesture": "FIST",
            "pose_tag": "FIST",
            "cursor_norm": map_box(lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]),
        }

    return {
        **base_meta,
        "gesture": "HOVER",
        "pose_tag": "HOVER",
        "cursor_norm": map_box(lm[INDEX_TIP][0], lm[INDEX_TIP][1]),
    }


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter1D:
    """Adaptive One-Euro Filter suppressing static tremor while maintaining 60 FPS velocity tracking."""

    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t_sec: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t_sec)
            self.dx_prev = 0.0
            return float(x)

        t_e = max(1e-4, float(t_sec) - self.t_prev)
        self.t_prev = float(t_sec)
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (float(x) - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, float(x), self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def filter(self, pt: Tuple[float, float], t_sec: float) -> Tuple[float, float]:
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

    def filter(self, pt: Sequence[float], t_sec: float) -> List[float]:
        return [
            self.fx.filter(float(pt[0]), t_sec),
            self.fy.filter(float(pt[1]), t_sec),
            self.fz.filter(float(pt[2]), t_sec),
        ]


class KalmanVelocityPredictor2D:
    """
    Lightweight scalar 2D Constant-Velocity Kalman filter `[x, y, vx, vy]` with zero NumPy overhead.
    Tracks true cursor velocity so 120–180ms motion-blur dropouts extrapolate smoothly instead
    of freezing at a stale pixel or causing jumps when re-acquired.
    """

    def __init__(self, process_noise: float = 0.04, meas_noise: float = 0.0015) -> None:
        self.q = float(process_noise)
        self.r = float(meas_noise)
        self.initialized = False
        self.x = 0.5
        self.y = 0.5
        self.vx = 0.0
        self.vy = 0.0
        self.p_pos = 1.0
        self.p_vel = 1.0
        self.last_t: Optional[float] = None
        self._last_meas: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.initialized = False
        self.vx = 0.0
        self.vy = 0.0
        self.p_pos = 1.0
        self.p_vel = 1.0
        self.last_t = None
        self._last_meas = None

    def update(self, meas_x: float, meas_y: float, t_sec: float) -> Tuple[float, float]:
        mx = float(meas_x)
        my = float(meas_y)
        if not self.initialized or self.last_t is None:
            self.initialized = True
            self.x = mx
            self.y = my
            self.vx = 0.0
            self.vy = 0.0
            self.p_pos = 0.02
            self.p_vel = 0.1
            self.last_t = float(t_sec)
            self._last_meas = (mx, my)
            return (mx, my)

        dt = max(1e-3, min(0.25, float(t_sec) - self.last_t))
        self.last_t = float(t_sec)
        prev_mx, prev_my = self._last_meas if self._last_meas is not None else (self.x, self.y)
        self._last_meas = (mx, my)

        # Predict step
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        self.p_pos = self.p_pos + self.p_vel * (dt * dt) + self.q * dt
        self.p_vel = self.p_vel + self.q

        # Innovation & Kalman gain
        k_pos = self.p_pos / max(1e-6, self.p_pos + self.r)
        inn_x = mx - px
        inn_y = my - py

        self.x = px + k_pos * inn_x
        self.y = py + k_pos * inn_y
        raw_vx = (mx - prev_mx) / dt
        raw_vy = (my - prev_my) / dt
        self.vx = max(-2.5, min(2.5, 0.35 * self.vx + 0.65 * raw_vx))
        self.vy = max(-2.5, min(2.5, 0.35 * self.vy + 0.65 * raw_vy))
        self.p_pos = (1.0 - k_pos) * self.p_pos

        return (max(0.0, min(1.0, self.x)), max(0.0, min(1.0, self.y)))

    def predict_dropout(self, t_sec: float, damping: float = 0.86) -> Optional[Tuple[float, float]]:
        if not self.initialized or self.last_t is None:
            return None
        dt = max(0.0, min(0.12, float(t_sec) - self.last_t))
        self.last_t = float(t_sec)
        self.x = max(0.0, min(1.0, self.x + self.vx * dt))
        self.y = max(0.0, min(1.0, self.y + self.vy * dt))
        self.vx *= damping
        self.vy *= damping
        return (self.x, self.y)


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
    kalman_predicted: bool = False
    voting_counts: Dict[str, int] = field(default_factory=dict)


class GestureFsmEngine:
    """
    Production Gesture Finite State Machine with Sliding Window Majority Voting:
    - Raw gesture samples pushed into `window_size=7` buffer requiring `>= min_votes=5` majority
      plus time debounce (`min_hold_sec >= 0.150s`) before transitioning state.
    - Combines One-Euro 3D fingertip smoothing with `KalmanVelocityPredictor2D` during
      `120–180ms` motion-blur `GRACE_PERIOD` dropouts so cursor glides smoothly instead of freezing.
    """

    def __init__(
        self,
        min_hold_frames: int = 5,
        min_hold_sec: float = 0.150,
        grace_period_sec: float = 0.160,
        min_cutoff: float = 1.35,
        beta: float = 0.015,
        frame_size: Tuple[int, int] = (640, 480),
        deadzone_px: float = 2.0,
        voting_window_size: int = 7,
        voting_min_votes: int = 5,
        pinch_enter_ratio: float = 0.28,
        pinch_exit_ratio: float = 0.42,
        palm_facing_min_z: float = 0.20,
    ) -> None:
        self.min_hold_frames = max(1, int(min_hold_frames))
        self.min_hold_sec = max(0.050, float(min_hold_sec))
        self.grace_period_sec = max(0.080, float(grace_period_sec))
        self.frame_size = (max(1, int(frame_size[0])), max(1, int(frame_size[1])))
        self.deadzone_px = max(0.0, float(deadzone_px))
        self.voting_window_size = max(3, int(voting_window_size))
        self.voting_min_votes = max(2, min(self.voting_window_size, int(voting_min_votes)))
        self.pinch_enter_ratio = float(pinch_enter_ratio)
        self.pinch_exit_ratio = float(pinch_exit_ratio)
        self.palm_facing_min_z = float(palm_facing_min_z)

        self.cursor_filter = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)
        self.thumb_tip_filter = OneEuroFilter3D(min_cutoff=min_cutoff, beta=beta)
        self.index_tip_filter = OneEuroFilter3D(min_cutoff=min_cutoff, beta=beta)
        self.kalman_predictor = KalmanVelocityPredictor2D()

        self.fsm_state = GestureFsmState.IDLE
        self.active_gesture = "IDLE"
        self.candidate_gesture = "IDLE"
        self.candidate_count = 0
        self.candidate_since_sec: Optional[float] = None
        self.last_seen_sec: Optional[float] = None
        self.last_cursor: Optional[Tuple[float, float]] = None
        self.deadzone_cursor: Optional[Tuple[float, float]] = None
        self.last_smoothed_pinch_ratio: Optional[float] = None
        self.last_classify_meta: Dict[str, object] = {}
        self.prev_fingers: List[bool] = [False] * 5
        self.vote_window: Deque[str] = deque(maxlen=self.voting_window_size)
        self._raw_first_seen: Dict[str, float] = {}

    def _apply_deadzone(
        self,
        pt: Tuple[float, float],
        palm_scale: float,
    ) -> Tuple[float, float]:
        px_thresh = self.deadzone_px / float(max(self.frame_size))
        palm_thresh = max(0.002, min(0.009, palm_scale * 0.022))
        thresh = max(px_thresh, palm_thresh)
        if self.deadzone_cursor is not None:
            dx = pt[0] - self.deadzone_cursor[0]
            dy = pt[1] - self.deadzone_cursor[1]
            if math.hypot(dx, dy) < thresh:
                return self.deadzone_cursor
        self.deadzone_cursor = pt
        return pt

    def update(
        self,
        landmarks: Optional[Sequence[Sequence[float]]],
        t_sec: Optional[float] = None,
        handedness: str = "right",
    ) -> GestureFsmSnapshot:
        now = time.perf_counter() if t_sec is None else float(t_sec)

        if landmarks is not None and len(landmarks) >= 21:
            self.last_seen_sec = now
            lm_filtered = [list(p) for p in landmarks]
            lm_filtered[THUMB_TIP] = self.thumb_tip_filter.filter(landmarks[THUMB_TIP], now)
            lm_filtered[INDEX_TIP] = self.index_tip_filter.filter(landmarks[INDEX_TIP], now)

            was_pinched = self.active_gesture == "CLICK"
            info = classify_hand_3d_vectors(
                lm_filtered,
                self.prev_fingers,
                was_pinched=was_pinched,
                pinch_enter_ratio=self.pinch_enter_ratio,
                pinch_exit_ratio=self.pinch_exit_ratio,
                handedness=handedness,
                palm_facing_min_z=self.palm_facing_min_z,
            )
            self.last_classify_meta = info
            self.prev_fingers = list(info["fingers"])  # type: ignore[arg-type]
            self.last_smoothed_pinch_ratio = float(info["pinch_ratio"])  # type: ignore[arg-type]
            palm_scale = float(info["palm_scale"])  # type: ignore[arg-type]
            raw_gesture = str(info["gesture"])
            raw_cursor = info["cursor_norm"]  # type: ignore[assignment]

            # Push into sliding window & elect majority vote
            self.vote_window.append(raw_gesture)
            if raw_gesture != self.active_gesture and raw_gesture not in self._raw_first_seen:
                self._raw_first_seen[raw_gesture] = now
            counts = Counter(self.vote_window)
            voted_gesture, voted_count = counts.most_common(1)[0]

            filtered_2d = self.cursor_filter.filter((float(raw_cursor[0]), float(raw_cursor[1])), now)
            kalman_2d = self.kalman_predictor.update(filtered_2d[0], filtered_2d[1], now)
            smoothed_cursor = self._apply_deadzone(kalman_2d, palm_scale)
            self.last_cursor = smoothed_cursor

            if voted_gesture == self.active_gesture:
                self.candidate_gesture = voted_gesture
                self.candidate_count = voted_count
                self.candidate_since_sec = None
                self.fsm_state = (
                    GestureFsmState.IDLE if self.active_gesture == "IDLE" else GestureFsmState.ACTIVE
                )
            else:
                if voted_gesture == self.candidate_gesture:
                    self.candidate_count += 1
                else:
                    self.candidate_gesture = voted_gesture
                    self.candidate_count = voted_count
                    self.candidate_since_sec = self._raw_first_seen.get(voted_gesture, now)
                since = self._raw_first_seen.get(
                    voted_gesture,
                    self.candidate_since_sec if self.candidate_since_sec is not None else now,
                )
                elapsed = now - since
                self.fsm_state = GestureFsmState.CANDIDATE
                frames_ok = (
                    voted_count >= self.min_hold_frames
                    if self.min_hold_frames <= self.voting_window_size
                    else self.candidate_count >= self.min_hold_frames
                )
                if voted_count >= self.voting_min_votes and frames_ok and elapsed >= self.min_hold_sec:
                    self.active_gesture = voted_gesture
                    self.candidate_count = 0
                    self.candidate_since_sec = None
                    self._raw_first_seen.clear()
                    self.fsm_state = (
                        GestureFsmState.IDLE if voted_gesture == "IDLE" else GestureFsmState.ACTIVE
                    )

            return GestureFsmSnapshot(
                fsm_state=self.fsm_state,
                active_gesture=self.active_gesture,
                candidate_gesture=self.candidate_gesture,
                cursor_norm=smoothed_cursor,
                hold_frames=self.candidate_count,
                in_grace_period=False,
                kalman_predicted=False,
                voting_counts=dict(counts),
            )

        # Hand temporarily lost -> check 120–180ms GRACE_PERIOD with Kalman velocity prediction
        if (
            self.last_seen_sec is not None
            and self.active_gesture != "IDLE"
            and (now - self.last_seen_sec) <= self.grace_period_sec
        ):
            self.fsm_state = GestureFsmState.GRACE_PERIOD
            predicted = self.kalman_predictor.predict_dropout(now)
            pred_cursor = predicted if predicted is not None else self.last_cursor
            self.last_cursor = pred_cursor
            return GestureFsmSnapshot(
                fsm_state=self.fsm_state,
                active_gesture=self.active_gesture,
                candidate_gesture=self.candidate_gesture,
                cursor_norm=pred_cursor,
                hold_frames=self.candidate_count,
                in_grace_period=True,
                kalman_predicted=(predicted is not None),
                voting_counts=dict(Counter(self.vote_window)),
            )

        self.fsm_state = GestureFsmState.IDLE
        self.active_gesture = "IDLE"
        self.candidate_gesture = "IDLE"
        self.candidate_count = 0
        self.candidate_since_sec = None
        self.prev_fingers = [False] * 5
        self.vote_window.clear()
        self._raw_first_seen.clear()
        self.last_cursor = None
        self.deadzone_cursor = None
        self.last_smoothed_pinch_ratio = None
        self.cursor_filter.reset()
        self.thumb_tip_filter.reset()
        self.index_tip_filter.reset()
        self.kalman_predictor.reset()
        return GestureFsmSnapshot(
            fsm_state=GestureFsmState.IDLE,
            active_gesture="IDLE",
            candidate_gesture="IDLE",
            cursor_norm=None,
            hold_frames=0,
            in_grace_period=False,
            kalman_predicted=False,
        )


class FrameSkipLock:
    """Thread-safe non-blocking frame-skip lock ensuring zero queue buildup on AI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_processing = False
        self.skipped_frames = 0

    def try_acquire(self) -> bool:
        with self._lock:
            if self.is_processing:
                self.skipped_frames += 1
                return False
            self.is_processing = True
            return True

    def release(self) -> None:
        with self._lock:
            self.is_processing = False


class HandGestureEngine:
    """
    2-Stage High-Precision Hand Landmark & Gesture Engine (Tier 3 + Tier 4):
    - Separated confidence gates: `min_detection_confidence`, `min_presence_confidence`, `min_tracking_confidence`.
    - Stage-2 ROI Crop-Zoom (`384x384` `cv2.INTER_CUBIC`) both on small full-frame hands (`bbox_area < 8%`)
      AND proactively around last known hand box / Teacher Pose wrists (`2.2x` padded ROI) on full-frame miss!
    """

    ROI_AREA_THRESHOLD = 0.08
    ROI_PADDING_RATIO = 0.45
    ROI_TARGET_SIZE = 384

    def __init__(
        self,
        model_path: Path = Path("models/hand_landmarker.task"),
        min_detection_confidence: float = 0.55,
        min_presence_confidence: float = 0.52,
        min_tracking_confidence: float = 0.65,
        roi_area_threshold: float = 0.08,
        roi_expand_factor: float = 2.2,
        roi_padding_ratio: float = 0.45,
        roi_target_size: int = 384,
        pinch_enter_ratio: float = 0.28,
        pinch_exit_ratio: float = 0.42,
        palm_facing_min_z: float = 0.20,
        grace_period_sec: float = 0.160,
        voting_window_size: int = 7,
        voting_min_votes: int = 5,
        one_euro_min_cutoff: float = 1.35,
        one_euro_beta: float = 0.015,
    ) -> None:
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.ROI_AREA_THRESHOLD = float(roi_area_threshold)
        self.roi_expand_factor = float(roi_expand_factor)
        self.ROI_PADDING_RATIO = float(roi_padding_ratio)
        self.ROI_TARGET_SIZE = int(roi_target_size)
        self.pinch_enter_ratio = float(pinch_enter_ratio)
        self.pinch_exit_ratio = float(pinch_exit_ratio)
        self.palm_facing_min_z = float(palm_facing_min_z)

        self.model_complexity = 1
        self.solutions_hands = None
        self.tasks_detector = None
        self.left_fsm = GestureFsmEngine(
            min_hold_frames=5,
            min_hold_sec=0.150,
            grace_period_sec=grace_period_sec,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            voting_window_size=voting_window_size,
            voting_min_votes=voting_min_votes,
            pinch_enter_ratio=pinch_enter_ratio,
            pinch_exit_ratio=pinch_exit_ratio,
            palm_facing_min_z=palm_facing_min_z,
        )
        self.right_fsm = GestureFsmEngine(
            min_hold_frames=5,
            min_hold_sec=0.150,
            grace_period_sec=grace_period_sec,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            voting_window_size=voting_window_size,
            voting_min_votes=voting_min_votes,
            pinch_enter_ratio=pinch_enter_ratio,
            pinch_exit_ratio=pinch_exit_ratio,
            palm_facing_min_z=palm_facing_min_z,
        )
        self.frame_skip_lock = FrameSkipLock()
        self._last_hand_roi_boxes: List[List[float]] = []
        self._last_hand_seen_ts: float = 0.0
        self.last_tracking_mode: str = "SEARCH_LOWRES"

        if HAS_MEDIAPIPE and ensure_hand_model(model_path, HAND_MODEL_URL):
            try:
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_presence_confidence,
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
        min_score: float = 0.42,
    ) -> bool:
        if score < min_score or len(lm_list) < 21:
            return False

        xs = [float(p[0]) for p in lm_list]
        ys = [float(p[1]) for p in lm_list]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        # Allow tiny distant hands (~1.5m-3m, e.g. 3.5%x3.5% = 0.0012) to enter Stage-2 ROI Zoom!
        if (bbox_w * bbox_h) <= 0.0012:
            return False

        px, py = float(lm_list[MIDDLE_MCP][0]), float(lm_list[MIDDLE_MCP][1])
        for fb in face_boxes:
            if len(fb) < 4:
                continue
            fx, fy, fw, fh = float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])
            face_cx = fx + fw * 0.5
            face_cy = fy + fh * 0.5
            radius = max(fw, fh) * 0.42
            if math.hypot(px - face_cx, py - face_cy) < radius:
                return False

        return True

    @staticmethod
    def map_roi_landmarks_to_full_frame(
        roi_landmarks: Sequence[Sequence[float]],
        crop_box_px: Tuple[int, int, int, int],
        full_size_wh: Tuple[int, int],
    ) -> List[List[float]]:
        """
        Accurately map normalized landmarks detected inside `[x0, y0, cw, ch]` ROI crop back into
        normalized `[0..1]` full-frame coordinates (preserving relative Z scale).
        """
        x0, y0, cw, ch = crop_box_px
        w, h = max(1, int(full_size_wh[0])), max(1, int(full_size_wh[1]))
        scale_z = float(cw) / float(w)
        mapped: List[List[float]] = []
        for p in roi_landmarks:
            lx = float(p[0]) if not hasattr(p, "x") else float(p.x)  # type: ignore[union-attr]
            ly = float(p[1]) if not hasattr(p, "y") else float(p.y)  # type: ignore[union-attr]
            lz = float(p[2]) if not hasattr(p, "z") else float(p.z)  # type: ignore[union-attr]
            mapped.append(
                [
                    round(max(0.0, min(1.0, (x0 + lx * cw) / float(w))), 4),
                    round(max(0.0, min(1.0, (y0 + ly * ch) / float(h))), 4),
                    round(lz * scale_z, 4),
                ]
            )
        return mapped

    def detect_hand_in_roi_box(
        self,
        rgb_frame: np.ndarray,
        roi_box_norm: Sequence[float],
        face_boxes: Sequence[Sequence[float]],
        *,
        roi_target_size: Optional[int] = None,
        enable_roi_sharpen: bool = True,
    ) -> List[Tuple[List[List[float]], float, str]]:
        """
        Crop high-res square ROI (`224x224` or `256x256`) directly from `rgb_frame` (e.g. 720p),
        apply ROI-only Fast Unsharp Masking + Adaptive Contrast Stretch, run HandLandmarker ONCE,
        and map 21 landmarks back into `[0..1]` full-frame coordinates.
        """
        if self.tasks_detector is None or rgb_frame is None or rgb_frame.size == 0:
            return []
        h, w = rgb_frame.shape[:2]
        rx, ry, rw, rh = (
            float(roi_box_norm[0]),
            float(roi_box_norm[1]),
            float(roi_box_norm[2]),
            float(roi_box_norm[3]),
        )
        x0 = max(0, int(math.floor(rx * w)))
        y0 = max(0, int(math.floor(ry * h)))
        x1 = min(w, int(math.ceil((rx + rw) * w)))
        y1 = min(h, int(math.ceil((ry + rh) * h)))
        cw = x1 - x0
        ch = y1 - y0
        if cw < 24 or ch < 24:
            return []

        target_sz = int(roi_target_size or self.ROI_TARGET_SIZE)
        target_sz = max(160, min(384, target_sz))
        roi = rgb_frame[y0:y1, x0:x1]
        interp = cv2.INTER_CUBIC if (cw < target_sz or ch < target_sz) else cv2.INTER_LINEAR
        roi_resized = cv2.resize(roi, (target_sz, target_sz), interpolation=interp)
        if enable_roi_sharpen:
            roi_resized = sharpen_and_contrast_roi(roi_resized, is_rgb=True)

        results: List[Tuple[List[List[float]], float, str]] = []
        try:
            roi_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_resized)
            roi_res = self.tasks_detector.detect(roi_mp)
            if not roi_res or not roi_res.hand_landmarks or not roi_res.handedness:
                return []
            for lms, handed_list in zip(roi_res.hand_landmarks, roi_res.handedness):
                if not handed_list:
                    continue
                top_cat = handed_list[0]
                r_score = float(top_cat.score)
                r_cat = (top_cat.category_name or "").lower()
                mapped = self.map_roi_landmarks_to_full_frame(lms, (x0, y0, cw, ch), (w, h))
                if self.validate_real_hand(mapped, r_score, face_boxes):
                    results.append((mapped, r_score, r_cat))
        except RuntimeError:
            return []
        return results

    def refine_hand_roi_zoom(
        self,
        rgb_frame: np.ndarray,
        lm_coords: List[List[float]],
        orig_score: float,
        orig_cat: str,
        face_boxes: Sequence[Sequence[float]],
        *,
        roi_target_size: Optional[int] = None,
        enable_roi_sharpen: bool = True,
    ) -> Tuple[List[List[float]], float, str, bool]:
        """
        Stage-2 ROI Dynamic Zoom Crop:
        When hand bbox area `< 8%` (`< 0.08`), crop expanded square ROI (`2.2x` around hand)
        from high-res frame, apply ROI-only sharpening, resize to `224x224`/`256x256`,
        and re-extract 21 landmarks.
        """
        if self.tasks_detector is None:
            return lm_coords, orig_score, orig_cat, False

        xs = [float(p[0]) for p in lm_coords]
        ys = [float(p[1]) for p in lm_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max_x - min_x
        bh = max_y - min_y
        bbox_area = bw * bh

        if bbox_area >= self.ROI_AREA_THRESHOLD or bw <= 0.008 or bh <= 0.008:
            return lm_coords, orig_score, orig_cat, False

        pad_x = max(bw, bh) * max(self.ROI_PADDING_RATIO, (self.roi_expand_factor - 1.0) * 0.5)
        pad_y = max(bw, bh) * max(self.ROI_PADDING_RATIO, (self.roi_expand_factor - 1.0) * 0.5)
        roi_norm = [
            max(0.0, min_x - pad_x),
            max(0.0, min_y - pad_y),
            min(1.0, max_x + pad_x) - max(0.0, min_x - pad_x),
            min(1.0, max_y + pad_y) - max(0.0, min_y - pad_y),
        ]
        cands = self.detect_hand_in_roi_box(
            rgb_frame,
            roi_norm,
            face_boxes,
            roi_target_size=roi_target_size,
            enable_roi_sharpen=enable_roi_sharpen,
        )
        if not cands:
            return lm_coords, orig_score, orig_cat, False

        orig_cx = float(lm_coords[MIDDLE_MCP][0])
        orig_cy = float(lm_coords[MIDDLE_MCP][1])
        max_center_drift = max(bw, bh) * 1.1

        best_candidate: Optional[Tuple[List[List[float]], float, str]] = None
        for mapped, r_score, r_cat in cands:
            cand_cx = float(mapped[MIDDLE_MCP][0])
            cand_cy = float(mapped[MIDDLE_MCP][1])
            if math.hypot(cand_cx - orig_cx, cand_cy - orig_cy) > max_center_drift:
                continue
            if r_score >= orig_score * 0.85:
                if best_candidate is None or r_score > best_candidate[1]:
                    best_candidate = (mapped, r_score, r_cat or orig_cat)

        if best_candidate is not None:
            return best_candidate[0], best_candidate[1], best_candidate[2], True
        return lm_coords, orig_score, orig_cat, False

    def detect_hands(
        self,
        rgb_frame: np.ndarray,
        face_boxes: Sequence[Sequence[float]],
        teacher_tracker: Any = None,
        teacher_snapshot: Any = None,
        wrist_roi_hints: Optional[Sequence[Sequence[float]]] = None,
        *,
        search_size: Tuple[int, int] = (320, 180),
        roi_target_size: Optional[int] = None,
        enable_roi_sharpen: bool = True,
    ) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        """
        Tracking-First Dual-Resolution ROI Pipeline (Potato-PC + 2.5m-3m Long-Range):
        - Tracking Stage: When `_last_hand_roi_boxes` exists, crop `2.2x` square ROI from `rgb_frame`
          (720p), sharpen ONLY the ROI (`224x224`/`256x256`), and run HandLandmarker ONCE.
        - Search Stage (on initial acquisition or loss): Downscale `rgb_frame` to `search_size`
          (`320x180` / `426x240`) for `<5ms` fast hand/wrist discovery, then refine small hands
          via high-res ROI crop + local sharpening.
        """
        left_out: Optional[Dict[str, object]] = None
        right_out: Optional[Dict[str, object]] = None

        if self.tasks_detector is None or rgb_frame is None or rgb_frame.size == 0:
            return None, None

        try:
            raw_candidates: List[Tuple[List[List[float]], float, str, bool]] = []
            now_mono = time.monotonic()

            # Stage A: Fast Tracking-First ROI Crop from full-res frame (skips full-frame inference!)
            if self._last_hand_roi_boxes:
                self.last_tracking_mode = "TRACK_ROI"
                for roi_box in self._last_hand_roi_boxes[:2]:
                    for mapped, r_score, r_cat in self.detect_hand_in_roi_box(
                        rgb_frame,
                        roi_box,
                        face_boxes,
                        roi_target_size=roi_target_size,
                        enable_roi_sharpen=enable_roi_sharpen,
                    ):
                        if teacher_tracker is not None and not teacher_tracker.is_hand_owned_by_locked_teacher(
                            mapped, teacher_snapshot
                        ):
                            continue
                        raw_candidates.append((mapped, r_score, r_cat, True))

            # Stage B: Low-Res Search Stage (320x180 / 426x240) when no tracked ROI succeeded
            if not raw_candidates:
                self.last_tracking_mode = "SEARCH_LOWRES"
                fh, fw = rgb_frame.shape[:2]
                sw, sh = max(160, int(search_size[0])), max(120, int(search_size[1]))
                if fw > sw or fh > sh:
                    search_rgb = cv2.resize(rgb_frame, (sw, sh), interpolation=cv2.INTER_AREA)
                else:
                    search_rgb = rgb_frame
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=search_rgb)
                h_res = self.tasks_detector.detect(mp_img)
                if h_res and h_res.hand_landmarks and h_res.handedness:
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
                        if teacher_tracker is not None and not teacher_tracker.is_hand_owned_by_locked_teacher(
                            lm_coords, teacher_snapshot
                        ):
                            continue
                        lm_coords, score, cat_name, zoomed = self.refine_hand_roi_zoom(
                            rgb_frame,
                            lm_coords,
                            score,
                            cat_name,
                            face_boxes,
                            roi_target_size=roi_target_size,
                            enable_roi_sharpen=enable_roi_sharpen,
                        )
                        raw_candidates.append((lm_coords, score, cat_name, zoomed))

            # Stage C: Proactive Long-Range Recovery via Teacher Pose Wrist Hints
            if not raw_candidates:
                probe_rois: List[List[float]] = []
                if wrist_roi_hints:
                    probe_rois.extend([list(b) for b in wrist_roi_hints])
                elif teacher_tracker is not None and teacher_snapshot is not None:
                    probe_rois.extend(
                        teacher_tracker.get_proactive_wrist_roi_boxes(
                            teacher_snapshot, roi_expand_factor=self.roi_expand_factor
                        )
                    )
                for roi_box in probe_rois[:2]:
                    for mapped, r_score, r_cat in self.detect_hand_in_roi_box(
                        rgb_frame,
                        roi_box,
                        face_boxes,
                        roi_target_size=roi_target_size,
                        enable_roi_sharpen=enable_roi_sharpen,
                    ):
                        if teacher_tracker is not None and not teacher_tracker.is_hand_owned_by_locked_teacher(
                            mapped, teacher_snapshot
                        ):
                            continue
                        raw_candidates.append((mapped, r_score, r_cat, True))

            new_roi_cache: List[List[float]] = []
            valid_hands: List[Tuple[str, Dict[str, object]]] = []
            for lm_coords, score, cat_name, stage2_used in raw_candidates:
                xs = [float(p[0]) for p in lm_coords]
                ys = [float(p[1]) for p in lm_coords]
                bw = max(0.06, max(xs) - min(xs))
                bh = max(0.06, max(ys) - min(ys))
                # 2.2x square ROI expansion around detected hand
                side = max(bw, bh) * self.roi_expand_factor
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                rx0 = max(0.0, cx - 0.5 * side)
                ry0 = max(0.0, cy - 0.5 * side)
                new_roi_cache.append(
                    [
                        rx0,
                        ry0,
                        min(1.0 - rx0, side),
                        min(1.0 - ry0, side),
                    ]
                )

                vec_info = classify_hand_3d_vectors(
                    lm_coords,
                    pinch_enter_ratio=self.pinch_enter_ratio,
                    pinch_exit_ratio=self.pinch_exit_ratio,
                    handedness=cat_name or "right",
                    palm_facing_min_z=self.palm_facing_min_z,
                )
                hand_obj: Dict[str, object] = {
                    "score": round(score, 4),
                    "lm": lm_coords,
                    "gesture_3d": vec_info["gesture"],
                    "pose_tag": vec_info.get("pose_tag", vec_info["gesture"]),
                    "pinch_ratio": vec_info["pinch_ratio"],
                    "finger_states": vec_info["finger_states"],
                    "finger_angles_deg": vec_info["finger_angles_deg"],
                    "palm_normal": vec_info["palm_normal"],
                    "palm_facing_score": vec_info["palm_facing_score"],
                    "palm_facing_ok": vec_info["palm_facing_ok"],
                    "palm_orientation": vec_info["palm_orientation"],
                    "stage2_roi_zoom": stage2_used,
                }
                valid_hands.append((cat_name, hand_obj))

            if new_roi_cache:
                self._last_hand_roi_boxes = new_roi_cache
                self._last_hand_seen_ts = now_mono
            elif (now_mono - self._last_hand_seen_ts) > 0.150:
                self._last_hand_roi_boxes = []

            if valid_hands:
                if len(valid_hands) == 1:
                    right_out = valid_hands[0][1]
                else:
                    for cat_name, hand_obj in valid_hands:
                        if cat_name == "left" and left_out is None:
                            left_out = hand_obj
                        else:
                            right_out = hand_obj

            h_px, w_px = rgb_frame.shape[:2]
            self.left_fsm.frame_size = (max(1, int(w_px)), max(1, int(h_px)))
            self.right_fsm.frame_size = (max(1, int(w_px)), max(1, int(h_px)))
            now_sec = time.perf_counter()
            left_snap = self.left_fsm.update(
                left_out["lm"] if left_out is not None else None,  # type: ignore[arg-type]
                t_sec=now_sec,
                handedness="left",
            )
            right_snap = self.right_fsm.update(
                right_out["lm"] if right_out is not None else None,  # type: ignore[arg-type]
                t_sec=now_sec,
                handedness="right",
            )

            for hand_obj, snap, fsm in (
                (left_out, left_snap, self.left_fsm),
                (right_out, right_snap, self.right_fsm),
            ):
                if hand_obj is None:
                    continue
                hand_obj["fsm_state"] = snap.fsm_state.value
                hand_obj["active_gesture"] = snap.active_gesture
                hand_obj["gesture"] = (
                    snap.active_gesture
                    if snap.active_gesture != "IDLE"
                    else str(hand_obj.get("gesture_3d", "HOVER"))
                )
                hand_obj["kalman_predicted"] = snap.kalman_predicted
                if fsm.last_smoothed_pinch_ratio is not None:
                    hand_obj["pinch_ratio"] = round(fsm.last_smoothed_pinch_ratio, 4)
                if snap.cursor_norm is not None:
                    hand_obj["cursor_norm"] = [
                        round(snap.cursor_norm[0], 4),
                        round(snap.cursor_norm[1], 4),
                    ]
                if fsm.last_classify_meta:
                    hand_obj["finger_states"] = fsm.last_classify_meta.get(
                        "finger_states", hand_obj["finger_states"]
                    )
                    hand_obj["finger_angles_deg"] = fsm.last_classify_meta.get(
                        "finger_angles_deg", hand_obj["finger_angles_deg"]
                    )
                    hand_obj["palm_facing_ok"] = fsm.last_classify_meta.get(
                        "palm_facing_ok", hand_obj["palm_facing_ok"]
                    )
                    hand_obj["palm_facing_score"] = fsm.last_classify_meta.get(
                        "palm_facing_score", hand_obj["palm_facing_score"]
                    )
        except RuntimeError as exc:
            _ = exc

        return left_out, right_out

    def close(self) -> None:
        """Release underlying MediaPipe HandLandmarker resources cleanly."""
        if self.tasks_detector is not None and hasattr(self.tasks_detector, "close"):
            try:
                self.tasks_detector.close()
            except Exception as exc:
                _ = exc
            self.tasks_detector = None
        if self.solutions_hands is not None and hasattr(self.solutions_hands, "close"):
            try:
                self.solutions_hands.close()
            except Exception as exc:
                _ = exc
            self.solutions_hands = None


def catmull_rom_spline(
    points: Sequence[Tuple[float, float]],
    samples_per_segment: int = 4,
    alpha: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Centripetal Catmull-Rom Spline interpolation for 60 FPS smooth stroke rendering
    even when AI Hand Tracking runs at 15-20 FPS on older Intel Core i3 CPUs.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n <= 2 or samples_per_segment <= 1:
        return pts

    out: List[Tuple[float, float]] = [pts[0]]
    ext = [pts[0]] + pts + [pts[-1]]
    steps = max(2, int(samples_per_segment))

    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]

        def _tj(ti: float, pa: Tuple[float, float], pb: Tuple[float, float]) -> float:
            d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            return ti + max(1e-4, d ** alpha)

        t0 = 0.0
        t1 = _tj(t0, p0, p1)
        t2 = _tj(t1, p1, p2)
        t3 = _tj(t2, p2, p3)

        for s in range(1, steps + 1):
            t = t1 + (t2 - t1) * (float(s) / float(steps))
            a1x = (t1 - t) / (t1 - t0) * p0[0] + (t - t0) / (t1 - t0) * p1[0]
            a1y = (t1 - t) / (t1 - t0) * p0[1] + (t - t0) / (t1 - t0) * p1[1]
            a2x = (t2 - t) / (t2 - t1) * p1[0] + (t - t1) / (t2 - t1) * p2[0]
            a2y = (t2 - t) / (t2 - t1) * p1[1] + (t - t1) / (t2 - t1) * p2[1]
            a3x = (t3 - t) / (t3 - t2) * p2[0] + (t - t2) / (t3 - t2) * p3[0]
            a3y = (t3 - t) / (t3 - t2) * p2[1] + (t - t2) / (t3 - t2) * p3[1]

            b1x = (t2 - t) / (t2 - t0) * a1x + (t - t0) / (t2 - t0) * a2x
            b1y = (t2 - t) / (t2 - t0) * a1y + (t - t0) / (t2 - t0) * a2y
            b2x = (t3 - t) / (t3 - t1) * a2x + (t - t1) / (t3 - t1) * a3x
            b2y = (t3 - t) / (t3 - t1) * a2y + (t - t1) / (t3 - t1) * a3y

            cx = (t2 - t) / (t2 - t1) * b1x + (t - t1) / (t2 - t1) * b2x
            cy = (t2 - t) / (t2 - t1) * b1y + (t - t1) / (t2 - t1) * b2y
            out.append((cx, cy))

    return out
