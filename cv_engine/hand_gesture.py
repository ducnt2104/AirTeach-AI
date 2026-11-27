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

import importlib.util
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

from cv_engine.app_paths import get_app_logger
from cv_engine.preprocessing import sharpen_and_contrast_roi
from cv_engine.rust_bridge import (
    HAS_RUST_CORE,
    catmull_rom_spline_fast,
    classify_hand_3d_fast,
    compute_dual_hand_kinematics_fast,
    compute_roi_crop_box_fast,
    make_kinematic_60hz_extrapolator,
    make_one_euro_1d,
    make_swipe_detector,
)

_LOGGER = get_app_logger("hand_gesture")

mp = None  # type: ignore
mp_python = None  # type: ignore
mp_vision = None  # type: ignore
HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None


def _ensure_mediapipe() -> bool:
    global mp, mp_python, mp_vision, HAS_MEDIAPIPE
    if mp is not None:
        return True
    try:
        import mediapipe as _mp  # type: ignore
        from mediapipe.tasks import python as _mp_python  # type: ignore
        from mediapipe.tasks.python import vision as _mp_vision  # type: ignore

        mp = _mp
        mp_python = _mp_python
        mp_vision = _mp_vision
        HAS_MEDIAPIPE = True
        return True
    except Exception as exc:
        _LOGGER.error("MediaPipe import failed: %s", exc)
        mp = None  # type: ignore
        mp_python = None  # type: ignore
        mp_vision = None  # type: ignore
        HAS_MEDIAPIPE = False
        return False


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


def _is_valid_hand_model_file(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size <= 500_000:
            return False
        with path.open("rb") as fh:
            head = fh.read(32)
        return (b"PK" in head) or (b"TFL3" in head)
    except OSError:
        return False


def ensure_hand_model(path: Path, url: str = HAND_MODEL_URL) -> bool:
    if _is_valid_hand_model_file(path):
        return True
    try:
        from cv_engine.app_paths import get_user_models_dir, resource_path

        for cand in (
            resource_path("assets/models/hand_landmarker.task"),
            resource_path("models/hand_landmarker.task"),
            get_user_models_dir() / "hand_landmarker.task",
        ):
            if cand != path and _is_valid_hand_model_file(cand):
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(cand.read_bytes())
                    return True
                except OSError:
                    return True
    except Exception as exc:
        _LOGGER.debug("Local bundled hand model lookup skipped: %s", exc)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AirTeachAI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) > 500_000 and (b"PK" in data[:32] or b"TFL3" in data[:32]):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return True
            except OSError:
                from cv_engine.app_paths import get_user_models_dir

                user_path = get_user_models_dir() / "hand_landmarker.task"
                user_path.parent.mkdir(parents=True, exist_ok=True)
                user_path.write_bytes(data)
                return True
    except Exception as exc:
        _LOGGER.warning("Failed to download hand model to %s: %s", path, exc)
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
    if lm is None or len(lm) < 10:
        return 1e-4
    return max(1e-4, dist_3d(lm[WRIST], lm[MIDDLE_MCP]))


def compute_pinch_ratio_3d(lm: Sequence[Sequence[float]]) -> float:
    """Palm-normalized Pinch Ratio `dist(4, 8) / dist(0, 9)`."""
    if lm is None or len(lm) < 10:
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
    if lm is None or len(lm) <= PINKY_MCP:
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


def map_active_workspace_norm(
    x: float,
    y: float,
    margin_min: float = 0.12,
    margin_max: float = 0.88,
) -> Tuple[float, float]:
    """
    Map inner active camera workspace `[margin_min..margin_max]` (default `12%..88%`)
    linearly onto `[0.0..1.0]` smartboard coordinates so teachers reach all 4 corners comfortably.
    """
    span = max(0.20, float(margin_max) - float(margin_min))
    mx = min(1.0, max(0.0, (float(x) - float(margin_min)) / span))
    my = min(1.0, max(0.0, (float(y) - float(margin_min)) / span))
    return (mx, my)


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
    if lm is None or len(lm) < 21:
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

    rust_res = classify_hand_3d_fast(
        lm,
        prev_fingers=prev_fingers,
        was_pinched=was_pinched,
        pinch_enter_ratio=pinch_enter_ratio,
        pinch_exit_ratio=pinch_exit_ratio,
        handedness=handedness,
        palm_facing_min_z=palm_facing_min_z,
    )
    if rust_res is not None:
        return rust_res

    state = list(prev_fingers) if (prev_fingers is not None and len(prev_fingers) == 5) else [False] * 5
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
        return map_active_workspace_norm(x, y, margin_min=0.12, margin_max=0.88)

    base_meta: Dict[str, object] = {
        "pinch_ratio": round(pinch_ratio, 4),
        "palm_scale": round(palm_scale, 4),
        "hand_span": round(max(palm_scale, dist_3d(lm[THUMB_TIP], lm[PINKY_TIP])), 4),
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

    # 6B. Pinky Extended Alone (unpinched): QUESTION_HOLD (Context Slide AI trigger)
    if pinky_up and not idx_up and not mid_up and not ring_up and pinch_ratio > 0.40:
        return {
            **base_meta,
            "gesture": "HOVER",
            "pose_tag": "QUESTION_HOLD",
            "cursor_norm": map_box(lm[PINKY_TIP][0], lm[PINKY_TIP][1]),
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
    """Adaptive One-Euro Filter (Rust PyO3 accelerated with Python fallback)."""

    def __init__(self, min_cutoff: float = 1.35, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._rs = make_one_euro_1d(self._min_cutoff, self._beta, self._d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @property
    def min_cutoff(self) -> float:
        return self._rs.min_cutoff if self._rs is not None else self._min_cutoff

    @min_cutoff.setter
    def min_cutoff(self, val: float) -> None:
        self._min_cutoff = float(val)
        if self._rs is not None:
            self._rs.min_cutoff = self._min_cutoff

    @property
    def beta(self) -> float:
        return self._rs.beta if self._rs is not None else self._beta

    @beta.setter
    def beta(self, val: float) -> None:
        self._beta = float(val)
        if self._rs is not None:
            self._rs.beta = self._beta

    @property
    def d_cutoff(self) -> float:
        return self._rs.d_cutoff if self._rs is not None else self._d_cutoff

    @d_cutoff.setter
    def d_cutoff(self, val: float) -> None:
        self._d_cutoff = float(val)
        if self._rs is not None:
            self._rs.d_cutoff = self._d_cutoff

    def reset(self) -> None:
        if self._rs is not None:
            self._rs.reset()
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t_sec: float) -> float:
        if self._rs is not None:
            return float(self._rs.filter(float(x), float(t_sec)))
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t_sec)
            self.dx_prev = 0.0
            return float(x)

        t_e = max(1e-4, float(t_sec) - self.t_prev)
        self.t_prev = float(t_sec)
        a_d = _smoothing_factor(t_e, self._d_cutoff)
        dx = (float(x) - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
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


@dataclass
class HandTelemetryPacket:
    """Lightweight 60Hz cursor/gesture telemetry packet emitted cleanly to the PySide6 UI thread."""

    timestamp_sec: float
    cursor_norm: Optional[Tuple[float, float]] = None
    velocity_norm: Tuple[float, float] = (0.0, 0.0)
    gesture: str = "IDLE"
    pose_tag: str = "IDLE"
    pinch_ratio: Optional[float] = None
    hand_span: float = 0.20
    two_hand_mode: str = "NONE"
    modifier_mode: str = "NONE"
    nav_event: Optional[str] = None
    swipe_direction: Optional[str] = None
    scroll_delta_norm: float = 0.0
    extrapolated: bool = False
    cam_fps: float = 0.0
    ai_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": round(self.timestamp_sec, 6),
            "cursor_norm": self.cursor_norm,
            "velocity_norm": self.velocity_norm,
            "gesture": self.gesture,
            "pose_tag": self.pose_tag,
            "pinch_ratio": self.pinch_ratio,
            "hand_span": self.hand_span,
            "two_hand_mode": self.two_hand_mode,
            "modifier_mode": self.modifier_mode,
            "nav_event": self.nav_event,
            "swipe_direction": self.swipe_direction,
            "scroll_delta_norm": self.scroll_delta_norm,
            "extrapolated": self.extrapolated,
            "cam_fps": round(self.cam_fps, 1),
            "ai_fps": round(self.ai_fps, 1),
        }


class Kinematic60HzExtrapolator:
    """
    Decoupled 60Hz Kinematic Velocity Extrapolator + Adaptive One-Euro Filter.
    Uses `airteach_core_rs.RustKinematic60HzExtrapolator` for sub-millisecond updates,
    with seamless pure-Python fallback.
    """

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.04,
        d_cutoff: float = 1.0,
        max_stale_sec: float = 0.16,
    ) -> None:
        self.filter = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.max_stale_sec = max(0.05, float(max_stale_sec))
        self._rs_extrap = make_kinematic_60hz_extrapolator(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            max_stale_sec=self.max_stale_sec,
        )
        self.anchor_xy: Optional[Tuple[float, float]] = None
        self.anchor_t: float = 0.0
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_output: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        if self._rs_extrap is not None:
            self._rs_extrap.reset()
        self.filter.reset()
        self.anchor_xy = None
        self.anchor_t = 0.0
        self.velocity = (0.0, 0.0)
        self.last_output = None

    def ingest_vision(
        self,
        cursor_norm: Optional[Tuple[float, float]],
        t_sec: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        now = time.perf_counter() if t_sec is None else float(t_sec)
        if self._rs_extrap is not None:
            pt = (float(cursor_norm[0]), float(cursor_norm[1])) if cursor_norm is not None else None
            out = self._rs_extrap.ingest_vision(pt, now)
            self.last_output = tuple(out) if out is not None else None
            self.velocity = tuple(self._rs_extrap.velocity)
            if cursor_norm is None:
                self.anchor_xy = None
                self.anchor_t = 0.0
            else:
                self.anchor_xy = (max(0.0, min(1.0, pt[0])), max(0.0, min(1.0, pt[1])))  # type: ignore[index]
                self.anchor_t = now
            return self.last_output

        if cursor_norm is None:
            self.reset()
            return None

        cx = max(0.0, min(1.0, float(cursor_norm[0])))
        cy = max(0.0, min(1.0, float(cursor_norm[1])))
        if self.anchor_xy is not None and self.anchor_t > 0.0:
            dt = max(1e-3, now - self.anchor_t)
            if dt <= 0.25:
                inst_vx = max(-3.5, min(3.5, (cx - self.anchor_xy[0]) / dt))
                inst_vy = max(-3.5, min(3.5, (cy - self.anchor_xy[1]) / dt))
                vx = 0.55 * self.velocity[0] + 0.45 * inst_vx
                vy = 0.55 * self.velocity[1] + 0.45 * inst_vy
                self.velocity = (vx, vy)
            else:
                self.velocity = (0.0, 0.0)
                self.filter.reset()
        else:
            self.velocity = (0.0, 0.0)

        self.anchor_xy = (cx, cy)
        self.anchor_t = now
        out = self.filter.filter((cx, cy), now)
        clamped = (max(0.0, min(1.0, out[0])), max(0.0, min(1.0, out[1])))
        self.last_output = clamped
        return clamped

    def sample_60hz(
        self,
        t_sec: Optional[float] = None,
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        now = time.perf_counter() if t_sec is None else float(t_sec)
        if self._rs_extrap is not None:
            pt, is_ext = self._rs_extrap.sample_60hz(now)
            out = tuple(pt) if pt is not None else None
            self.last_output = out
            return out, bool(is_ext)

        if self.anchor_xy is None:
            return None, False
        elapsed = now - self.anchor_t
        if elapsed > self.max_stale_sec:
            return None, False

        dt = max(0.0, min(0.045, elapsed))
        decay = math.exp(-10.0 * dt)
        px = max(0.0, min(1.0, self.anchor_xy[0] + self.velocity[0] * dt * decay))
        py = max(0.0, min(1.0, self.anchor_xy[1] + self.velocity[1] * dt * decay))
        out = self.filter.filter((px, py), now)
        clamped = (max(0.0, min(1.0, out[0])), max(0.0, min(1.0, out[1])))
        self.last_output = clamped
        return clamped, (dt > 0.004)


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
        min_hold_frames: int = 3,
        min_hold_sec: float = 0.080,
        grace_period_sec: float = 0.160,
        min_cutoff: float = 1.45,
        beta: float = 0.020,
        d_cutoff: float = 1.0,
        frame_size: Tuple[int, int] = (640, 480),
        deadzone_px: float = 2.0,
        voting_window_size: int = 7,
        voting_min_votes: int = 3,
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

        self.cursor_filter = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.thumb_tip_filter = OneEuroFilter3D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.index_tip_filter = OneEuroFilter3D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.pinch_filter = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
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
            raw_pinch_ratio = compute_pinch_ratio_3d(lm_filtered)
            smoothed_pinch_ratio = max(0.01, min(3.0, self.pinch_filter.filter(raw_pinch_ratio, now)))
            info = classify_hand_3d_vectors(
                lm_filtered,
                self.prev_fingers,
                was_pinched=was_pinched,
                pinch_enter_ratio=self.pinch_enter_ratio,
                pinch_exit_ratio=self.pinch_exit_ratio,
                handedness=handedness,
                palm_facing_min_z=self.palm_facing_min_z,
            )
            pinch_thresh = float(self.pinch_exit_ratio) if was_pinched else float(self.pinch_enter_ratio)
            if smoothed_pinch_ratio < pinch_thresh and info.get("pose_tag") not in (
                "OPEN_PALM",
                "FIST",
                "THREE_FINGERS",
                "QUESTION_HOLD",
            ):
                info["gesture"] = "CLICK"
                info["pose_tag"] = "PINCH"
            info["pinch_ratio"] = round(smoothed_pinch_ratio, 4)
            self.last_classify_meta = info
            self.prev_fingers = list(info["fingers"])  # type: ignore[arg-type]
            self.last_smoothed_pinch_ratio = float(smoothed_pinch_ratio)
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
        self.pinch_filter.reset()
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
    Fail-Open 2-Hand High-Precision Landmark & Gesture Engine:
    - Always runs `num_hands = 2` with sensitive thresholds (`0.35 / 0.40 / 0.40`) for 30cm–3m range.
    - Immediate 1-Frame ROI Fallback: if ROI crop misses on frame N, immediately clears cache and
      runs Full-Frame Scan in the same pass + next frame so tracking NEVER gets stuck on empty ROI.
    - Periodic 2nd-hand Full-Frame Discovery when only 1 hand is tracked, plus center deduplication.
    - Stores `last_raw_hands` before any Teacher Lock filtering for live debug verification.
    """

    ROI_AREA_THRESHOLD = 0.08
    ROI_PADDING_RATIO = 0.35
    ROI_TARGET_SIZE = 256

    def __init__(
        self,
        model_path: Path = Path("models/hand_landmarker.task"),
        num_hands: int = 2,
        min_detection_confidence: float = 0.35,
        min_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.40,
        roi_area_threshold: float = 0.08,
        roi_expand_factor: float = 2.2,
        roi_padding_ratio: float = 0.45,
        roi_target_size: int = 256,
        pinch_enter_ratio: float = 0.28,
        pinch_exit_ratio: float = 0.42,
        palm_facing_min_z: float = 0.20,
        grace_period_sec: float = 0.160,
        voting_window_size: int = 7,
        voting_min_votes: int = 3,
        one_euro_min_cutoff: float = 0.8,
        one_euro_beta: float = 0.04,
        one_euro_d_cutoff: float = 1.0,
        roi_unsharp_strength: float = 0.65,
        min_hold_frames: int = 3,
        min_hold_sec: float = 0.080,
        model_complexity: int = 1,
    ) -> None:
        self.num_hands = max(1, min(2, int(num_hands)))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.ROI_AREA_THRESHOLD = float(roi_area_threshold)
        self.roi_expand_factor = float(roi_expand_factor)
        self.ROI_PADDING_RATIO = max(0.45, float(roi_padding_ratio))
        self.ROI_TARGET_SIZE = int(roi_target_size)
        self.roi_unsharp_strength = float(roi_unsharp_strength)
        self.pinch_enter_ratio = float(pinch_enter_ratio)
        self.pinch_exit_ratio = float(pinch_exit_ratio)
        self.palm_facing_min_z = float(palm_facing_min_z)

        self.model_complexity = 0 if int(model_complexity) <= 0 else 1
        self.solutions_hands = None
        self.tasks_detector = None
        self.backend_mode: str = "NONE"
        self.init_error: str = ""
        self.left_fsm = GestureFsmEngine(
            min_hold_frames=min_hold_frames,
            min_hold_sec=min_hold_sec,
            grace_period_sec=grace_period_sec,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
            voting_window_size=voting_window_size,
            voting_min_votes=voting_min_votes,
            pinch_enter_ratio=pinch_enter_ratio,
            pinch_exit_ratio=pinch_exit_ratio,
            palm_facing_min_z=palm_facing_min_z,
        )
        self.right_fsm = GestureFsmEngine(
            min_hold_frames=min_hold_frames,
            min_hold_sec=min_hold_sec,
            grace_period_sec=grace_period_sec,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
            voting_window_size=voting_window_size,
            voting_min_votes=voting_min_votes,
            pinch_enter_ratio=pinch_enter_ratio,
            pinch_exit_ratio=pinch_exit_ratio,
            palm_facing_min_z=palm_facing_min_z,
        )
        self.frame_skip_lock = FrameSkipLock()
        self._last_hand_roi_boxes: List[List[float]] = []
        self._last_hand_seen_ts: float = 0.0
        self._last_valid_left: Optional[Dict[str, object]] = None
        self._last_valid_right: Optional[Dict[str, object]] = None
        self._prev_hand_centers: Dict[str, Tuple[float, float, float]] = {}
        self._detect_tick: int = 0
        self.last_tracking_mode: str = "FULL_SCAN"
        self.last_raw_hands: List[Dict[str, object]] = []
        self.last_rejected_by_teacher_lock: int = 0

        if not _ensure_mediapipe() or not HAS_MEDIAPIPE:
            self.init_error = "MediaPipe library is not installed"
            _LOGGER.error(self.init_error)
            return

        if ensure_hand_model(model_path, HAND_MODEL_URL):
            try:
                from cv_engine.app_paths import ensure_concurrent_futures_eager

                ensure_concurrent_futures_eager()
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_presence_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.tasks_detector = mp_vision.HandLandmarker.create_from_options(opts)
                self.backend_mode = "TASKS_LANDMARKER"
                _LOGGER.info(
                    "HandLandmarker initialized (num_hands=%d, det=%.2f, pres=%.2f, track=%.2f, model=%s)",
                    self.num_hands,
                    self.min_detection_confidence,
                    self.min_presence_confidence,
                    self.min_tracking_confidence,
                    model_path,
                )
            except Exception as exc:
                self.init_error = f"HandLandmarker init failed: {exc}"
                _LOGGER.exception(self.init_error)
                self.tasks_detector = None
        else:
            self.init_error = f"HandLandmarker model missing/unreadable: {model_path}"
            _LOGGER.warning(self.init_error)

        if self.tasks_detector is None and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.solutions_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=self.num_hands,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.backend_mode = "SOLUTIONS_HANDS"
                self.init_error = ""
                _LOGGER.info("HandGestureEngine fallback to mp.solutions.hands initialized cleanly.")
            except Exception as exc:
                self.init_error = f"{self.init_error}; solutions.hands fallback failed: {exc}"
                _LOGGER.error(self.init_error)
                self.solutions_hands = None

    def _has_detector(self) -> bool:
        return (self.tasks_detector is not None) or (self.solutions_hands is not None)

    def _detect_raw_hands(self, rgb_img: np.ndarray) -> List[Tuple[Any, float, str]]:
        out: List[Tuple[Any, float, str]] = []
        if self.tasks_detector is not None:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            res = self.tasks_detector.detect(mp_img)
            if res and res.hand_landmarks and res.handedness:
                for lms, handed_list in zip(res.hand_landmarks, res.handedness):
                    if not handed_list:
                        continue
                    top_cat = handed_list[0]
                    out.append((lms, float(top_cat.score), (top_cat.category_name or "").lower()))
        elif self.solutions_hands is not None:
            res = self.solutions_hands.process(rgb_img)
            lms_list = getattr(res, "multi_hand_landmarks", None) or []
            hd_list = getattr(res, "multi_handedness", None) or []
            for idx, hand_lms in enumerate(lms_list):
                score = 0.85
                cat_name = "right"
                if idx < len(hd_list) and getattr(hd_list[idx], "classification", None):
                    cls0 = hd_list[idx].classification[0]
                    score = float(getattr(cls0, "score", 0.85))
                    cat_name = str(getattr(cls0, "label", "right") or "right").lower()
                out.append((hand_lms.landmark, score, cat_name))
        return out

    @staticmethod
    def validate_real_hand(
        lm_list: Sequence[Sequence[float]],
        score: float,
        face_boxes: Sequence[Sequence[float]],
        min_score: float = 0.25,
    ) -> bool:
        if score < min_score or len(lm_list) < 21:
            return False

        xs = [float(p[0]) for p in lm_list]
        ys = [float(p[1]) for p in lm_list]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        area = bbox_w * bbox_h
        # Allow tiny distant hands (~2.5m-3m, e.g. 2.6%x2.8% = 0.0007) to enter Stage-2 ROI Zoom!
        if area <= 0.0007:
            return False

        # Only reject tiny low-confidence phantom skeletons right inside a face center;
        # close-up hands (area >= 0.015) or high-confidence hands held near chin/face pass!
        if area < 0.015 and score < 0.55:
            px, py = float(lm_list[MIDDLE_MCP][0]), float(lm_list[MIDDLE_MCP][1])
            for fb in face_boxes:
                if len(fb) < 4:
                    continue
                fx, fy, fw, fh = float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])
                face_cx = fx + fw * 0.5
                face_cy = fy + fh * 0.5
                radius = max(fw, fh) * 0.34
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
        if not self._has_detector() or rgb_frame is None or rgb_frame.size == 0:
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
        roi_resized = np.ascontiguousarray(cv2.resize(roi, (target_sz, target_sz), interpolation=interp), dtype=np.uint8)
        if enable_roi_sharpen:
            roi_resized = np.ascontiguousarray(
                sharpen_and_contrast_roi(
                    roi_resized,
                    is_rgb=True,
                    unsharp_strength=self.roi_unsharp_strength,
                    contrast_stretch=True,
                ),
                dtype=np.uint8,
            )

        results: List[Tuple[List[List[float]], float, str]] = []
        try:
            raw_hits = self._detect_raw_hands(roi_resized)
            for lms, r_score, r_cat in raw_hits:
                mapped = self.map_roi_landmarks_to_full_frame(lms, (x0, y0, cw, ch), (w, h))
                if self.validate_real_hand(mapped, r_score, face_boxes):
                    results.append((mapped, r_score, r_cat))
        except Exception as exc:
            _LOGGER.warning("ROI HandLandmarker inference error: %s", exc)
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
        When hand bbox area `< 8%` (`< 0.08`), crop expanded square ROI (`>= 35%` padding around hand)
        from high-res frame, apply ROI-only unsharp masking, resize to `224x224`/`256x256`,
        and re-extract 21 landmarks.
        """
        if not self._has_detector():
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

        pad_x = max(bw, bh) * max(0.45, self.ROI_PADDING_RATIO, (self.roi_expand_factor - 1.0) * 0.5)
        pad_y = max(bw, bh) * max(0.45, self.ROI_PADDING_RATIO, (self.roi_expand_factor - 1.0) * 0.5)
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

    @staticmethod
    def _hand_center_norm(lm_coords: Sequence[Sequence[float]]) -> Tuple[float, float]:
        return (float(lm_coords[MIDDLE_MCP][0]), float(lm_coords[MIDDLE_MCP][1]))

    @classmethod
    def _merge_unique_hand_candidates(
        cls,
        target: List[Tuple[List[List[float]], float, str, bool]],
        incoming: Sequence[Tuple[List[List[float]], float, str, bool]],
        min_center_dist: float = 0.08,
    ) -> None:
        """Merge hand detections while suppressing duplicate detections of the same physical hand."""
        for lm_coords, score, cat_name, zoomed in incoming:
            cx, cy = cls._hand_center_norm(lm_coords)
            dup_idx: Optional[int] = None
            for idx, (ex_lm, ex_sc, _, _) in enumerate(target):
                ecx, ecy = cls._hand_center_norm(ex_lm)
                if math.hypot(cx - ecx, cy - ecy) < min_center_dist:
                    dup_idx = idx
                    break
            if dup_idx is None:
                target.append((lm_coords, score, cat_name, zoomed))
            elif score > target[dup_idx][1]:
                target[dup_idx] = (lm_coords, score, cat_name, zoomed)

    def _run_frame_scan(
        self,
        rgb_frame: np.ndarray,
        target_wh: Tuple[int, int],
        face_boxes: Sequence[Sequence[float]],
        *,
        roi_target_size: Optional[int],
        enable_roi_sharpen: bool,
    ) -> List[Tuple[List[List[float]], float, str, bool]]:
        fh, fw = rgb_frame.shape[:2]
        sw, sh = max(160, int(target_wh[0])), max(120, int(target_wh[1]))
        if fw > sw or fh > sh:
            scan_rgb = np.ascontiguousarray(cv2.resize(rgb_frame, (sw, sh), interpolation=cv2.INTER_AREA), dtype=np.uint8)
        else:
            scan_rgb = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        found: List[Tuple[List[List[float]], float, str, bool]] = []
        raw_hits = self._detect_raw_hands(scan_rgb)
        for lms, score, cat_name in raw_hits:
            lm_coords = [
                [
                    round(float(p.x if hasattr(p, "x") else p[0]), 4),
                    round(float(p.y if hasattr(p, "y") else p[1]), 4),
                    round(float(p.z if hasattr(p, "z") else p[2]), 4),
                ]
                for p in lms
            ]
            if not self.validate_real_hand(lm_coords, score, face_boxes):
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
            found.append((lm_coords, score, cat_name, zoomed))
        return found

    def detect_hands(
        self,
        rgb_frame: np.ndarray,
        face_boxes: Sequence[Sequence[float]],
        teacher_tracker: Any = None,
        teacher_snapshot: Any = None,
        wrist_roi_hints: Optional[Sequence[Sequence[float]]] = None,
        *,
        search_size: Tuple[int, int] = (426, 240),
        roi_target_size: Optional[int] = None,
        enable_roi_sharpen: bool = True,
    ) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        """
        Fail-Open 2-Hand Dual-Resolution Pipeline (30cm–3m & Intel i3 Optimized):
        1. Stage A (ROI Tracking): When `_last_hand_roi_boxes` exists, crop `2.2x` ROI from full-res
           frame and sharpen ONLY the ROI (`224x224`/`256x256`).
           - **Immediate 1-Frame Fallback:** If Stage A misses (0 hands), IMMEDIATELY clear ROI cache
             and execute Full-Frame Scan right away so ROI never gets stuck on an empty box.
           - **2-Hand Discovery:** If tracking only 1 hand (`< num_hands`), periodically run a fast
             frame scan every 4 ticks so raising a 2nd hand is always detected immediately.
        2. Stage B (Full-Frame / Low-Res Search + Higher-Res Fallback): Runs `search_size`, and if
           nothing is found on a larger frame (`>= 480p`), immediately falls back to `640x480` full scan.
        3. Stage C (Proactive Wrist ROI): Checks Teacher Pose wrist hints when still empty.
        4. **Fail-Open Teacher Lock:** Only filters when `teacher_tracker` confirms multi-person gating;
           always saves `self.last_raw_hands` (pre-filter) for camera debug verification.
        """
        left_out: Optional[Dict[str, object]] = None
        right_out: Optional[Dict[str, object]] = None
        self.last_raw_hands = []
        self.last_rejected_by_teacher_lock = 0

        if not self._has_detector() or rgb_frame is None or rgb_frame.size == 0:
            return None, None

        try:
            rgb_contig = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            fh, fw = rgb_contig.shape[:2]
            now_mono = time.monotonic()
            self._detect_tick += 1

            unfiltered_candidates: List[Tuple[List[List[float]], float, str, bool]] = []
            roi_missed = False

            # Stage A: Fast Tracking-First ROI Crop
            if self._last_hand_roi_boxes:
                self.last_tracking_mode = "TRACK_ROI"
                for roi_box in self._last_hand_roi_boxes[: self.num_hands]:
                    roi_hits = [
                        (mapped, r_score, r_cat, True)
                        for mapped, r_score, r_cat in self.detect_hand_in_roi_box(
                            rgb_contig,
                            roi_box,
                            face_boxes,
                            roi_target_size=roi_target_size,
                            enable_roi_sharpen=enable_roi_sharpen,
                        )
                    ]
                    self._merge_unique_hand_candidates(unfiltered_candidates, roi_hits)

                if not unfiltered_candidates:
                    # Immediate 1-Frame Fallback: clear stale ROI boxes immediately!
                    self._last_hand_roi_boxes = []
                    roi_missed = True

            # Stage B: Wide Full-Frame Acquisition (640x480 base + 800x600 HD sweep if distant)
            need_full_scan = (not unfiltered_candidates) or (
                self.num_hands > 1 and len(unfiltered_candidates) < self.num_hands and (self._detect_tick % 4 == 0)
            )
            if need_full_scan:
                self.last_tracking_mode = "FALLBACK_FULL" if roi_missed else "FULL_SCAN"
                wide_wh = (
                    min(fw, max(640, int(search_size[0]))),
                    min(fh, max(480, int(search_size[1]))),
                )
                scan_hits = self._run_frame_scan(
                    rgb_contig,
                    wide_wh,
                    face_boxes,
                    roi_target_size=roi_target_size,
                    enable_roi_sharpen=enable_roi_sharpen,
                )
                self._merge_unique_hand_candidates(unfiltered_candidates, scan_hits)

                # If 640x480 full-frame found 0 hands on a high-res sensor, step up to 800x600 full-frame
                if not unfiltered_candidates and (fw >= 800 or fh >= 600):
                    self.last_tracking_mode = "FULL_FRAME_HD"
                    hd_hits = self._run_frame_scan(
                        rgb_contig,
                        (min(fw, 800), min(fh, 600)),
                        face_boxes,
                        roi_target_size=roi_target_size,
                        enable_roi_sharpen=enable_roi_sharpen,
                    )
                    self._merge_unique_hand_candidates(unfiltered_candidates, hd_hits)

            # Stage C: Proactive Long-Range Recovery via Teacher Pose Wrist Hints
            if not unfiltered_candidates:
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
                    hint_hits = [
                        (mapped, r_score, r_cat, True)
                        for mapped, r_score, r_cat in self.detect_hand_in_roi_box(
                            rgb_contig,
                            roi_box,
                            face_boxes,
                            roi_target_size=roi_target_size,
                            enable_roi_sharpen=enable_roi_sharpen,
                        )
                    ]
                    self._merge_unique_hand_candidates(unfiltered_candidates, hint_hits)

            # Apply Teacher Lock (Fail-Open) while preserving raw_hands for visual verification
            raw_candidates: List[Tuple[List[List[float]], float, str, bool]] = []
            for lm_coords, score, cat_name, stage2_used in unfiltered_candidates:
                owned = True
                if teacher_tracker is not None:
                    owned = bool(teacher_tracker.is_hand_owned_by_locked_teacher(lm_coords, teacher_snapshot))
                self.last_raw_hands.append(
                    {
                        "score": round(score, 4),
                        "handedness": cat_name or "unknown",
                        "lm": lm_coords,
                        "accepted": owned,
                    }
                )
                if owned:
                    raw_candidates.append((lm_coords, score, cat_name, stage2_used))
                else:
                    self.last_rejected_by_teacher_lock += 1

            new_roi_cache: List[List[float]] = []
            valid_hands: List[Tuple[str, Dict[str, object]]] = []
            for lm_coords, score, cat_name, stage2_used in raw_candidates[: self.num_hands]:
                xs = [float(p[0]) for p in lm_coords]
                ys = [float(p[1]) for p in lm_coords]
                bw = max(0.06, max(xs) - min(xs))
                bh = max(0.06, max(ys) - min(ys))
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                side_key = (cat_name or "right").lower()
                pred_dx, pred_dy = 0.0, 0.0
                prev_info = self._prev_hand_centers.get(side_key)
                if prev_info is not None:
                    dt_c = now_mono - prev_info[2]
                    if 0.005 <= dt_c <= 0.14:
                        pred_dx = max(-0.08, min(0.08, (cx - prev_info[0])))
                        pred_dy = max(-0.08, min(0.08, (cy - prev_info[1])))
                self._prev_hand_centers[side_key] = (cx, cy, now_mono)

                # Generous >= 45% padded velocity-predicted square ROI around detected hand (Rust fast math)
                new_roi_cache.append(
                    compute_roi_crop_box_fast(
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                        pred_dx=pred_dx,
                        pred_dy=pred_dy,
                        expand_factor=self.roi_expand_factor,
                        padding_ratio=self.ROI_PADDING_RATIO,
                    )
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
                    "handedness": cat_name or "right",
                    "center_x": round(float(lm_coords[MIDDLE_MCP][0]), 4),
                    "lm": lm_coords,
                    "gesture_3d": vec_info["gesture"],
                    "pose_tag": vec_info.get("pose_tag", vec_info["gesture"]),
                    "pinch_ratio": vec_info["pinch_ratio"],
                    "hand_span": vec_info.get("hand_span", 0.20),
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
            else:
                # Clear immediately on 1-frame miss so next frame always scans full frame
                self._last_hand_roi_boxes = []

            if valid_hands:
                if len(valid_hands) == 1:
                    cat_0, obj_0 = valid_hands[0]
                    if cat_0 == "left":
                        left_out = obj_0
                    else:
                        right_out = obj_0
                else:
                    # Assign distinct Left/Right slots cleanly, never overwriting a 2nd hand!
                    unassigned: List[Dict[str, object]] = []
                    for cat_name, hand_obj in valid_hands[:2]:
                        if cat_name == "left" and left_out is None:
                            left_out = hand_obj
                        elif cat_name == "right" and right_out is None:
                            right_out = hand_obj
                        else:
                            unassigned.append(hand_obj)
                    for hand_obj in unassigned:
                        if left_out is None and right_out is not None:
                            if float(hand_obj.get("center_x", 0.5)) <= float(right_out.get("center_x", 0.5)):
                                left_out = hand_obj
                            else:
                                left_out = right_out
                                right_out = hand_obj
                        elif right_out is None:
                            right_out = hand_obj
                        elif left_out is None:
                            left_out = hand_obj

            h_px, w_px = rgb_contig.shape[:2]
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

            for hand_obj, snap, fsm, side_lbl in (
                (left_out, left_snap, self.left_fsm, "left"),
                (right_out, right_snap, self.right_fsm, "right"),
            ):
                if hand_obj is None:
                    # Hysteretic lock-on across brief 120-180ms motion-blur dropouts
                    cached_prev = self._last_valid_left if side_lbl == "left" else self._last_valid_right
                    if snap.in_grace_period and snap.cursor_norm is not None and cached_prev is not None:
                        synth = dict(cached_prev)
                        synth["slot"] = side_lbl
                        synth["fsm_state"] = snap.fsm_state.value
                        synth["active_gesture"] = snap.active_gesture
                        synth["gesture"] = snap.active_gesture if snap.active_gesture != "IDLE" else synth.get("gesture", "HOVER")
                        synth["in_grace_period"] = True
                        synth["kalman_predicted"] = snap.kalman_predicted
                        synth["cursor_norm"] = [
                            round(snap.cursor_norm[0], 4),
                            round(snap.cursor_norm[1], 4),
                        ]
                        if side_lbl == "left":
                            left_out = synth
                        else:
                            right_out = synth
                    else:
                        if side_lbl == "left":
                            self._last_valid_left = None
                        else:
                            self._last_valid_right = None
                    continue
                hand_obj["slot"] = side_lbl
                hand_obj["fsm_state"] = snap.fsm_state.value
                hand_obj["active_gesture"] = snap.active_gesture
                hand_obj["gesture"] = (
                    snap.active_gesture
                    if snap.active_gesture != "IDLE"
                    else str(hand_obj.get("gesture_3d", "HOVER"))
                )
                hand_obj["in_grace_period"] = False
                hand_obj["kalman_predicted"] = snap.kalman_predicted
                if fsm.last_smoothed_pinch_ratio is not None:
                    hand_obj["pinch_ratio"] = round(fsm.last_smoothed_pinch_ratio, 4)
                if snap.cursor_norm is not None:
                    hand_obj["cursor_norm"] = [
                        round(snap.cursor_norm[0], 4),
                        round(snap.cursor_norm[1], 4),
                    ]
                if fsm.last_classify_meta:
                    hand_obj["pose_tag"] = fsm.last_classify_meta.get(
                        "pose_tag", hand_obj.get("pose_tag", "HOVER")
                    )
                    hand_obj["hand_span"] = fsm.last_classify_meta.get(
                        "hand_span", hand_obj.get("hand_span", 0.20)
                    )
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
                if side_lbl == "left":
                    self._last_valid_left = dict(hand_obj)
                else:
                    self._last_valid_right = dict(hand_obj)
        except Exception as exc:
            _LOGGER.exception("HandGestureEngine.detect_hands failed: %s", exc)
            self._last_hand_roi_boxes = []

        return left_out, right_out

    def close(self) -> None:
        """Release underlying MediaPipe HandLandmarker resources cleanly."""
        if self.tasks_detector is not None and hasattr(self.tasks_detector, "close"):
            try:
                self.tasks_detector.close()
            except Exception as exc:
                _LOGGER.warning("Failed closing HandLandmarker: %s", exc)
            self.tasks_detector = None
        if self.solutions_hands is not None and hasattr(self.solutions_hands, "close"):
            try:
                self.solutions_hands.close()
            except Exception as exc:
                _LOGGER.warning("Failed closing solutions hands: %s", exc)
            self.solutions_hands = None


@dataclass
class DualHandActionState:
    primary_side: str = "none"  # "left" | "right" | "none"
    primary_hand: Optional[Dict[str, object]] = None
    secondary_hand: Optional[Dict[str, object]] = None
    two_hand_mode: str = "NONE"  # "NONE" | "DUAL_PINCH_ZOOM" | "DUAL_PALM_PAN" | "OFFHAND_COMMAND" | "MODIFIER_DRAW"
    modifier_mode: str = "NONE"  # "NONE" | "LINE_LOCK" | "SHAPE_LOCK"
    zoom_factor: float = 1.0
    zoom_center_norm: Optional[Tuple[float, float]] = None
    pan_delta_norm: Tuple[float, float] = (0.0, 0.0)
    offhand_command: Optional[str] = None  # "TOGGLE_PEN_ERASER" | "CYCLE_COLOR" | "UNDO" | "SET_PEN"
    suppress_primary_drawing: bool = False
    status_label_vi: str = "Chờ tay"


class DualHandGestureCoordinator:
    """
    Coordinates 1-hand primary interaction and 2-hand collaborative gestures:
    - Dual Pinch (`PINCH` + `PINCH`) -> smooth Zoom In / Zoom Out around midpoint.
    - Dual Open Palm (`OPEN_PALM` + `OPEN_PALM`) -> smooth Canvas Pan.
    - Off-hand quick command while Primary hand holds position:
      * Off-hand `FIST` -> Toggle Pen / Eraser
      * Off-hand `TWO_FINGERS` -> Cycle Ink Color
      * Off-hand `THREE_FINGERS` -> Undo
      * Off-hand `PINCH` (when primary not pinching) -> Switch back to Pen
    """

    def __init__(self, primary_pref: str = "auto") -> None:
        self.primary_pref = primary_pref if primary_pref in ("auto", "left", "right") else "auto"
        self._sticky_primary: Optional[str] = None
        self._prev_pinch_dist: Optional[float] = None
        self._prev_pan_center: Optional[Tuple[float, float]] = None
        self._prev_fist_pan_xy: Optional[Tuple[float, float]] = None
        self._fist_pan_until: float = 0.0
        self._offhand_cand: str = ""
        self._offhand_cand_count: int = 0
        self._last_offhand_cmd_ts: float = -1e9

    def set_primary_preference(self, pref: str) -> None:
        if pref in ("auto", "left", "right"):
            self.primary_pref = pref
            self._sticky_primary = None

    def reset(self) -> None:
        self._sticky_primary = None
        self._prev_pinch_dist = None
        self._prev_pan_center = None
        self._prev_fist_pan_xy = None
        self._fist_pan_until = 0.0
        self._offhand_cand = ""
        self._offhand_cand_count = 0

    @staticmethod
    def _hand_xy(hand: Dict[str, object]) -> Tuple[float, float]:
        for key in ("cursor_norm", "cursor"):
            cn = hand.get(key)
            if isinstance(cn, (list, tuple)) and len(cn) >= 2:
                return (float(cn[0]), float(cn[1]))
        lm = hand.get("lm")
        if isinstance(lm, (list, tuple)) and len(lm) > MIDDLE_MCP:
            return (float(lm[MIDDLE_MCP][0]), float(lm[MIDDLE_MCP][1]))
        return (0.5, 0.5)

    @staticmethod
    def _norm_pose(hand: Dict[str, object]) -> str:
        raw = str(hand.get("pose_tag") or hand.get("gesture") or "HOVER").upper()
        if raw in ("ERASE_PALM", "FIVE_FINGERS"):
            return "OPEN_PALM"
        if raw in ("FIST_CANCEL",):
            return "FIST"
        if raw in ("DRAW",):
            return "PINCH"
        if raw in ("UNDO_THREE",):
            return "THREE_FINGERS"
        if raw in ("IDLE", "NONE", ""):
            return "HOVER"
        return raw

    def update(
        self,
        left_hand: Optional[Dict[str, object]],
        right_hand: Optional[Dict[str, object]],
        t_sec: Optional[float] = None,
    ) -> DualHandActionState:
        now = time.monotonic() if t_sec is None else float(t_sec)

        if left_hand is None and right_hand is None:
            self.reset()
            return DualHandActionState(status_label_vi="Chờ tay")

        # Select primary vs secondary hand deterministically without stealing cursor
        if left_hand is not None and right_hand is not None:
            if self.primary_pref == "left":
                p_side, primary, secondary = "left", left_hand, right_hand
            elif self.primary_pref == "right":
                p_side, primary, secondary = "right", right_hand, left_hand
            elif self._sticky_primary == "left":
                p_side, primary, secondary = "left", left_hand, right_hand
            else:
                p_side, primary, secondary = "right", right_hand, left_hand
        elif right_hand is not None:
            p_side, primary, secondary = "right", right_hand, None
        else:
            p_side, primary, secondary = "left", left_hand, None

        self._sticky_primary = p_side

        if secondary is None:
            self._prev_pinch_dist = None
            self._prev_pan_center = None
            self._offhand_cand = ""
            self._offhand_cand_count = 0
            pose = str(primary.get("pose_tag", "HOVER"))
            return DualHandActionState(
                primary_side=p_side,
                primary_hand=primary,
                secondary_hand=None,
                two_hand_mode="NONE",
                status_label_vi=f"1 Tay ({p_side.upper()}): {pose}",
            )

        # Both hands active!
        p_pose = self._norm_pose(primary)
        s_pose = self._norm_pose(secondary)
        p_pr = float(primary.get("pinch_ratio") or 1.0)
        s_pr = float(secondary.get("pinch_ratio") or 1.0)
        p_xy = self._hand_xy(primary)
        s_xy = self._hand_xy(secondary)

        # 1. Dual Pinch Zoom (both hands pinching — Rust accelerated kinematics)
        both_pinch = (p_pose == "PINCH" or p_pr < 0.33) and (s_pose == "PINCH" or s_pr < 0.33)
        if both_pinch:
            self._prev_pan_center = None
            self._prev_fist_pan_xy = None
            self._offhand_cand = ""
            self._offhand_cand_count = 0
            zoom_factor, mid, dist, _, _ = compute_dual_hand_kinematics_fast(
                p_xy, s_xy, prev_pinch_dist=self._prev_pinch_dist, prev_pan_center=None
            )
            self._prev_pinch_dist = dist
            return DualHandActionState(
                primary_side=p_side,
                primary_hand=primary,
                secondary_hand=secondary,
                two_hand_mode="DUAL_PINCH_ZOOM",
                zoom_factor=round(zoom_factor, 4),
                zoom_center_norm=mid,
                suppress_primary_drawing=True,
                status_label_vi="2 Tay: Thu/Phóng bảng (Dual Pinch)",
            )
        self._prev_pinch_dist = None

        # 2. Dual Open Palm Pan (both hands open palm — Rust accelerated kinematics)
        if p_pose == "OPEN_PALM" and s_pose == "OPEN_PALM":
            self._prev_fist_pan_xy = None
            self._offhand_cand = ""
            self._offhand_cand_count = 0
            _, _, _, pan_delta, mid = compute_dual_hand_kinematics_fast(
                p_xy, s_xy, prev_pinch_dist=None, prev_pan_center=self._prev_pan_center
            )
            self._prev_pan_center = mid
            return DualHandActionState(
                primary_side=p_side,
                primary_hand=primary,
                secondary_hand=secondary,
                two_hand_mode="DUAL_PALM_PAN",
                pan_delta_norm=(round(pan_delta[0], 5), round(pan_delta[1], 5)),
                suppress_primary_drawing=True,
                status_label_vi="2 Tay: Kéo trượt bảng (Dual Pan)",
            )
        self._prev_pan_center = None

        # 2A. Hold Off-Hand Fist + Move Primary Hand (non-pinching) -> Dual Pan
        p_is_drawing = (p_pose == "PINCH") or (p_pr < 0.34)
        if s_pose == "FIST" and not p_is_drawing:
            if self._prev_fist_pan_xy is not None:
                fdx = p_xy[0] - self._prev_fist_pan_xy[0]
                fdy = p_xy[1] - self._prev_fist_pan_xy[1]
                fstep = math.hypot(fdx, fdy)
                is_moving = fstep > 0.010 or (now <= self._fist_pan_until and fstep > 0.002)
                if is_moving:
                    self._prev_fist_pan_xy = p_xy
                    self._fist_pan_until = now + 0.28
                    self._offhand_cand = ""
                    self._offhand_cand_count = 0
                    return DualHandActionState(
                        primary_side=p_side,
                        primary_hand=primary,
                        secondary_hand=secondary,
                        two_hand_mode="DUAL_PALM_PAN",
                        pan_delta_norm=(
                            round(max(-0.15, min(0.15, fdx)), 5),
                            round(max(-0.15, min(0.15, fdy)), 5),
                        ),
                        suppress_primary_drawing=True,
                        status_label_vi="2 Tay: Kéo trượt bảng (Tay phụ Nắm + Kéo)",
                    )
            self._prev_fist_pan_xy = p_xy
        else:
            self._prev_fist_pan_xy = None
            self._fist_pan_until = 0.0

        # 2B. Left-Hand Modifier Lock while Primary Hand is Pinching/Drawing:
        # Holding Secondary Fist -> lock straight line; Secondary Two-Fingers -> lock geometric shape snap.
        if p_is_drawing and s_pose in ("FIST", "TWO_FINGERS"):
            self._offhand_cand = ""
            self._offhand_cand_count = 0
            mod = "LINE_LOCK" if s_pose == "FIST" else "SHAPE_LOCK"
            lbl = (
                "2 Tay: Khóa Nét Thẳng (Tay phụ Nắm + Tay chính Vẽ)"
                if mod == "LINE_LOCK"
                else "2 Tay: Khóa Hình Học Chuẩn (Tay phụ 2 Ngón + Tay chính Vẽ)"
            )
            return DualHandActionState(
                primary_side=p_side,
                primary_hand=primary,
                secondary_hand=secondary,
                two_hand_mode="MODIFIER_DRAW",
                modifier_mode=mod,
                suppress_primary_drawing=False,
                status_label_vi=lbl,
            )

        # 3. Off-hand quick tool command while primary hand keeps its position/drawing
        cmd_map = {
            "FIST": "TOGGLE_PEN_ERASER",
            "TWO_FINGERS": "CYCLE_COLOR",
            "THREE_FINGERS": "UNDO",
            "PINCH": "SET_PEN",
        }
        mapped_cmd = cmd_map.get(s_pose, "")
        triggered_cmd: Optional[str] = None
        if mapped_cmd:
            if mapped_cmd == self._offhand_cand:
                self._offhand_cand_count += 1
            else:
                self._offhand_cand = mapped_cmd
                self._offhand_cand_count = 1
            if self._offhand_cand_count >= 2 and (now - self._last_offhand_cmd_ts) >= 0.70:
                self._last_offhand_cmd_ts = now
                triggered_cmd = mapped_cmd
                self._offhand_cand_count = 0
        else:
            self._offhand_cand = ""
            self._offhand_cand_count = 0

        return DualHandActionState(
            primary_side=p_side,
            primary_hand=primary,
            secondary_hand=secondary,
            two_hand_mode="OFFHAND_COMMAND" if triggered_cmd else "NONE",
            offhand_command=triggered_cmd,
            suppress_primary_drawing=False,
            status_label_vi=f"2 Tay: Chính={p_pose} | Phụ={s_pose}"
            + (f" -> {triggered_cmd}" if triggered_cmd else ""),
        )


def catmull_rom_spline(
    points: Sequence[Tuple[float, float]],
    samples_per_segment: int = 4,
    alpha: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Centripetal Catmull-Rom Spline interpolation (Rust accelerated with Python fallback).
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n <= 2 or samples_per_segment <= 1:
        return pts
    fast = catmull_rom_spline_fast(pts, max(2, int(samples_per_segment)), float(alpha))
    if fast is not None:
        return fast

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


class SlideNavigationGestureDetector:
    """
    Detects non-drawing classroom presentation air gestures:
    1. Horizontal Air-Swipe (`SLIDE_NEXT` / `SLIDE_PREV`) with velocity threshold + `600–650ms`
       cooldown to prevent double-skipping slides.
    2. Vertical Air-Swipe (`SCROLL_UP` / `SCROLL_DOWN`) with `450ms` cooldown + continuous two-finger
       vertical drag for scrolling long Word (`.docx`) and multi-page PDF documents.
    3. Two-Finger Peace-Sign Hold (`TWO_FINGERS` held steady for `800ms`) -> `SLIDE_NEXT`.
    4. Pinky Question Hold (`QUESTION_HOLD` held steady for `900ms`) -> `ASK_SLIDE_AI`.
    """

    def __init__(
        self,
        swipe_cooldown_sec: float = 0.60,
        swipe_min_dx: float = 0.16,
        swipe_min_vel: float = 0.85,
        swipe_window_sec: float = 0.28,
        vertical_cooldown_sec: float = 0.45,
        vertical_min_dy: float = 0.14,
        vertical_min_vel: float = 0.75,
        peace_hold_sec: float = 0.80,
        question_hold_sec: float = 0.90,
    ) -> None:
        self.swipe_cooldown_sec = max(0.35, float(swipe_cooldown_sec))
        self.swipe_min_dx = max(0.08, float(swipe_min_dx))
        self.swipe_min_vel = max(0.35, float(swipe_min_vel))
        self.swipe_window_sec = max(0.12, float(swipe_window_sec))
        self.vertical_cooldown_sec = max(0.25, float(vertical_cooldown_sec))
        self.vertical_min_dy = max(0.07, float(vertical_min_dy))
        self.vertical_min_vel = max(0.30, float(vertical_min_vel))
        self.peace_hold_sec = max(0.40, float(peace_hold_sec))
        self.question_hold_sec = max(0.45, float(question_hold_sec))

        self._trail: Deque[Tuple[float, float, float]] = deque(maxlen=24)
        self._last_trigger_ts: float = -1e9
        self._last_vert_trigger_ts: float = -1e9
        self._rs_swipe = make_swipe_detector(
            swipe_cooldown_sec=self.swipe_cooldown_sec,
            swipe_min_dx=self.swipe_min_dx,
            swipe_min_vel=self.swipe_min_vel,
            swipe_window_sec=self.swipe_window_sec,
            vertical_cooldown_sec=self.vertical_cooldown_sec,
            vertical_min_dy=self.vertical_min_dy,
            vertical_min_vel=self.vertical_min_vel,
        )
        self._prev_two_finger_y: Optional[float] = None
        self._peace_since_ts: Optional[float] = None
        self._peace_anchor: Optional[Tuple[float, float]] = None
        self._peace_fired: bool = False
        self._question_since_ts: Optional[float] = None
        self._question_anchor: Optional[Tuple[float, float]] = None
        self._question_fired: bool = False

    def reset(self) -> None:
        self._trail.clear()
        if self._rs_swipe is not None:
            self._rs_swipe.reset()
        self._prev_two_finger_y = None
        self._peace_since_ts = None
        self._peace_anchor = None
        self._peace_fired = False
        self._question_since_ts = None
        self._question_anchor = None
        self._question_fired = False

    def update(
        self,
        primary_hand: Optional[Dict[str, object]],
        *,
        suppress: bool = False,
        is_drawing: bool = False,
        t_sec: Optional[float] = None,
    ) -> Dict[str, object]:
        now = time.monotonic() if t_sec is None else float(t_sec)
        cooldown_rem_ms = max(0.0, (self.swipe_cooldown_sec - (now - self._last_trigger_ts)) * 1000.0)
        vert_cooldown_rem_ms = max(0.0, (self.vertical_cooldown_sec - (now - self._last_vert_trigger_ts)) * 1000.0)

        if primary_hand is None or suppress or is_drawing:
            self.reset()
            return {
                "nav_event": None,
                "swipe_direction": None,
                "scroll_delta_norm": 0.0,
                "peace_hold_progress": 0.0,
                "question_hold_progress": 0.0,
                "cooldown_remaining_ms": round(cooldown_rem_ms, 1),
            }

        pose_tag = str(primary_hand.get("pose_tag") or primary_hand.get("gesture") or "HOVER").upper()
        pinch_ratio = float(primary_hand.get("pinch_ratio") or 1.0)
        cn = primary_hand.get("cursor_norm")
        if isinstance(cn, (list, tuple)) and len(cn) >= 2:
            xy = (float(cn[0]), float(cn[1]))
        else:
            lm = primary_hand.get("lm")
            if isinstance(lm, (list, tuple)) and len(lm) > MIDDLE_MCP:
                xy = (float(lm[MIDDLE_MCP][0]), float(lm[MIDDLE_MCP][1]))
            else:
                xy = (0.5, 0.5)

        if pose_tag == "PINCH" or pinch_ratio < 0.34:
            self.reset()
            return {
                "nav_event": None,
                "swipe_direction": None,
                "scroll_delta_norm": 0.0,
                "peace_hold_progress": 0.0,
                "question_hold_progress": 0.0,
                "cooldown_remaining_ms": round(cooldown_rem_ms, 1),
            }

        scroll_delta_norm = 0.0
        # 1. Two-Finger Vertical Drag Scroll & Two-Finger Peace-Sign Hold (800ms -> SLIDE_NEXT)
        peace_prog = 0.0
        if pose_tag == "TWO_FINGERS":
            if self._prev_two_finger_y is not None:
                step_dy = xy[1] - self._prev_two_finger_y
                if 0.008 < abs(step_dy) < 0.12:
                    scroll_delta_norm = round(-step_dy * 1.6, 4)
                    if abs(step_dy) > 0.025:
                        self._peace_since_ts = now
                        self._peace_anchor = xy
                        self._peace_fired = False
            self._prev_two_finger_y = xy[1]

            if self._peace_since_ts is None or self._peace_anchor is None:
                self._peace_since_ts = now
                self._peace_anchor = xy
                self._peace_fired = False
            else:
                drift = math.hypot(xy[0] - self._peace_anchor[0], xy[1] - self._peace_anchor[1])
                if drift > 0.075:
                    self._peace_since_ts = now
                    self._peace_anchor = xy
                    self._peace_fired = False
            held = now - (self._peace_since_ts or now)
            peace_prog = min(1.0, max(0.0, held / self.peace_hold_sec))
            if held >= self.peace_hold_sec and not self._peace_fired and cooldown_rem_ms <= 0.0:
                self._peace_fired = True
                self._last_trigger_ts = now
                self._trail.clear()
                return {
                    "nav_event": "SLIDE_NEXT",
                    "swipe_direction": "LEFT",
                    "scroll_delta_norm": 0.0,
                    "peace_hold_progress": 1.0,
                    "question_hold_progress": 0.0,
                    "cooldown_remaining_ms": round(self.swipe_cooldown_sec * 1000.0, 1),
                }
        else:
            self._prev_two_finger_y = None
            self._peace_since_ts = None
            self._peace_anchor = None
            self._peace_fired = False

        # 2. Pinky Question Hold (900ms -> ASK_SLIDE_AI)
        question_prog = 0.0
        if pose_tag == "QUESTION_HOLD":
            if self._question_since_ts is None or self._question_anchor is None:
                self._question_since_ts = now
                self._question_anchor = xy
                self._question_fired = False
            else:
                drift = math.hypot(xy[0] - self._question_anchor[0], xy[1] - self._question_anchor[1])
                if drift > 0.08:
                    self._question_since_ts = now
                    self._question_anchor = xy
                    self._question_fired = False
            held_q = now - (self._question_since_ts or now)
            question_prog = min(1.0, max(0.0, held_q / self.question_hold_sec))
            if held_q >= self.question_hold_sec and not self._question_fired and cooldown_rem_ms <= 0.0:
                self._question_fired = True
                self._last_trigger_ts = now
                self._trail.clear()
                return {
                    "nav_event": "ASK_SLIDE_AI",
                    "swipe_direction": None,
                    "scroll_delta_norm": 0.0,
                    "peace_hold_progress": round(peace_prog, 3),
                    "question_hold_progress": 1.0,
                    "cooldown_remaining_ms": round(self.swipe_cooldown_sec * 1000.0, 1),
                }
        else:
            self._question_since_ts = None
            self._question_anchor = None
            self._question_fired = False

        # 3. Horizontal Air-Swipe (Next/Prev) + Vertical Air-Swipe (Scroll Up/Down) — Rust accelerated
        if pose_tag in ("OPEN_PALM", "HOVER", "TWO_FINGERS", "INDEX"):
            if self._rs_swipe is not None:
                ev, sdir, sdelta = self._rs_swipe.push_and_check(float(xy[0]), float(xy[1]), now, pose_tag)
                if ev is not None:
                    self._peace_since_ts = None
                    if ev in ("SLIDE_NEXT", "SLIDE_PREV"):
                        self._last_trigger_ts = now
                        return {
                            "nav_event": ev,
                            "swipe_direction": sdir,
                            "scroll_delta_norm": 0.0,
                            "peace_hold_progress": 0.0,
                            "question_hold_progress": round(question_prog, 3),
                            "cooldown_remaining_ms": round(self.swipe_cooldown_sec * 1000.0, 1),
                        }
                    self._last_vert_trigger_ts = now
                    return {
                        "nav_event": ev,
                        "swipe_direction": sdir,
                        "scroll_delta_norm": float(sdelta),
                        "peace_hold_progress": 0.0,
                        "question_hold_progress": round(question_prog, 3),
                        "cooldown_remaining_ms": round(cooldown_rem_ms, 1),
                    }
            else:
                self._trail.append((xy[0], xy[1], now))
                while self._trail and (now - self._trail[0][2]) > self.swipe_window_sec:
                    self._trail.popleft()
                if len(self._trail) >= 3:
                    for ox, oy, ot in self._trail:
                        dt = now - ot
                        if dt < 0.04 or dt > self.swipe_window_sec:
                            continue
                        dx = xy[0] - ox
                        dy = xy[1] - oy
                        vel_x = abs(dx) / dt
                        vel_y = abs(dy) / dt
                        if (
                            cooldown_rem_ms <= 0.0
                            and abs(dx) >= self.swipe_min_dx
                            and vel_x >= self.swipe_min_vel
                            and abs(dx) >= 1.8 * abs(dy)
                        ):
                            ev = "SLIDE_NEXT" if dx < 0.0 else "SLIDE_PREV"
                            sdir = "LEFT" if dx < 0.0 else "RIGHT"
                            self._last_trigger_ts = now
                            self._trail.clear()
                            self._peace_since_ts = None
                            return {
                                "nav_event": ev,
                                "swipe_direction": sdir,
                                "scroll_delta_norm": 0.0,
                                "peace_hold_progress": 0.0,
                                "question_hold_progress": round(question_prog, 3),
                                "cooldown_remaining_ms": round(self.swipe_cooldown_sec * 1000.0, 1),
                            }
                        if (
                            pose_tag in ("OPEN_PALM", "TWO_FINGERS", "HOVER")
                            and vert_cooldown_rem_ms <= 0.0
                            and abs(dy) >= self.vertical_min_dy
                            and vel_y >= self.vertical_min_vel
                            and abs(dy) >= 1.8 * abs(dx)
                        ):
                            ev = "SCROLL_DOWN" if dy < 0.0 else "SCROLL_UP"
                            sdir = "UP" if dy < 0.0 else "DOWN"
                            delta = 0.38 if dy < 0.0 else -0.38
                            self._last_vert_trigger_ts = now
                            self._trail.clear()
                            self._peace_since_ts = None
                            return {
                                "nav_event": ev,
                                "swipe_direction": sdir,
                                "scroll_delta_norm": delta,
                                "peace_hold_progress": 0.0,
                                "question_hold_progress": round(question_prog, 3),
                                "cooldown_remaining_ms": round(cooldown_rem_ms, 1),
                            }
        else:
            self._trail.clear()
            if self._rs_swipe is not None:
                self._rs_swipe.reset()

        return {
            "nav_event": None,
            "swipe_direction": None,
            "scroll_delta_norm": scroll_delta_norm,
            "peace_hold_progress": round(peace_prog, 3),
            "question_hold_progress": round(question_prog, 3),
            "cooldown_remaining_ms": round(cooldown_rem_ms, 1),
        }
