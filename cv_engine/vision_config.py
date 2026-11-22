"""
Externalized Vision Configuration Loader (`cv_engine/vision_config.py`).

Loads tunable parameters from `config/vision_config.json` (plus optional `.env` /
environment overrides) with schema clamping so Python and Rust engines share
identical, deterministic defaults and never fail on missing or malformed config files.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


from cv_engine.app_paths import resource_path

DEFAULT_VISION_CONFIG_PATH = resource_path("config/vision_config.json")


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    target_fps: int = 30
    hd_width: int = 1280
    hd_height: int = 720
    prefer_hd: bool = False

    def active_resolution(self) -> Tuple[int, int]:
        if self.prefer_hd:
            return (int(self.hd_width), int(self.hd_height))
        return (int(self.width), int(self.height))


@dataclass
class PreprocessingConfig:
    exposure_check_interval_n: int = 4
    low_light_mean_luma: float = 72.0
    backlit_p95_luma: float = 232.0
    flicker_delta_luma: float = 14.0
    clahe_clip_limit: float = 2.4
    clahe_tile_grid_size: int = 8
    adaptive_gamma_low: float = 0.62
    adaptive_gamma_high: float = 1.24
    quality_gate_min_sharpness: float = 22.0


@dataclass
class SubjectTrackerConfig:
    teacher_grace_sec: float = 1.5
    teacher_match_iou_min: float = 0.18
    teacher_face_sim_threshold: float = 0.48
    wrist_gate_radius_scale: float = 2.35
    enable_pose_landmarker: bool = True
    pose_min_detection_confidence: float = 0.50
    pose_min_presence_confidence: float = 0.50
    pose_min_tracking_confidence: float = 0.55


@dataclass
class HandTrackingConfig:
    min_detection_confidence: float = 0.55
    min_presence_confidence: float = 0.52
    min_tracking_confidence: float = 0.65
    roi_area_threshold: float = 0.08
    roi_expand_factor: float = 2.2
    roi_padding_ratio: float = 0.45
    roi_target_size: int = 384


@dataclass
class Gesture3DConfig:
    finger_extended_max_deg: float = 40.0
    finger_half_bent_max_deg: float = 78.0
    pinch_enter_ratio: float = 0.28
    pinch_exit_ratio: float = 0.42
    palm_facing_min_z: float = 0.20
    one_euro_min_cutoff: float = 1.35
    one_euro_beta: float = 0.015
    one_euro_d_cutoff: float = 1.0
    voting_window_size: int = 7
    voting_min_votes: int = 5
    debounce_ms: float = 150.0
    grace_period_ms: float = 160.0


@dataclass
class HudConfig:
    debug_hud_default_enabled: bool = False


@dataclass
class VisionConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    subject_tracker: SubjectTrackerConfig = field(default_factory=SubjectTrackerConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    gesture_3d: Gesture3DConfig = field(default_factory=Gesture3DConfig)
    hud: HudConfig = field(default_factory=HudConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return default


def _clamp_float(val: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(val)
        if f != f:  # NaN guard
            return default
        return max(lo, min(hi, f))
    except (TypeError, ValueError):
        return default


def load_vision_config(config_path: Path | str | None = None) -> VisionConfig:
    """Load `VisionConfig` from disk + environment overrides with safe fallbacks."""
    cfg = VisionConfig()
    env_path = os.environ.get("AIRTEACH_VISION_CONFIG_PATH", "").strip()
    target = Path(config_path) if config_path else (Path(env_path) if env_path else DEFAULT_VISION_CONFIG_PATH)

    raw: Dict[str, Any] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    cam = raw.get("camera") or {}
    if isinstance(cam, dict):
        cfg.camera.width = _clamp_int(cam.get("width"), 320, 1920, cfg.camera.width)
        cfg.camera.height = _clamp_int(cam.get("height"), 240, 1080, cfg.camera.height)
        cfg.camera.target_fps = _clamp_int(cam.get("target_fps"), 15, 120, cfg.camera.target_fps)
        cfg.camera.hd_width = _clamp_int(cam.get("hd_width"), 640, 1920, cfg.camera.hd_width)
        cfg.camera.hd_height = _clamp_int(cam.get("hd_height"), 480, 1080, cfg.camera.hd_height)
        cfg.camera.prefer_hd = bool(cam.get("prefer_hd", cfg.camera.prefer_hd))

    prep = raw.get("preprocessing") or {}
    if isinstance(prep, dict):
        cfg.preprocessing.exposure_check_interval_n = _clamp_int(
            prep.get("exposure_check_interval_n"), 1, 30, cfg.preprocessing.exposure_check_interval_n
        )
        cfg.preprocessing.low_light_mean_luma = _clamp_float(
            prep.get("low_light_mean_luma"), 25.0, 120.0, cfg.preprocessing.low_light_mean_luma
        )
        cfg.preprocessing.backlit_p95_luma = _clamp_float(
            prep.get("backlit_p95_luma"), 180.0, 254.0, cfg.preprocessing.backlit_p95_luma
        )
        cfg.preprocessing.flicker_delta_luma = _clamp_float(
            prep.get("flicker_delta_luma"), 4.0, 60.0, cfg.preprocessing.flicker_delta_luma
        )
        cfg.preprocessing.clahe_clip_limit = _clamp_float(
            prep.get("clahe_clip_limit"), 1.0, 6.0, cfg.preprocessing.clahe_clip_limit
        )
        cfg.preprocessing.clahe_tile_grid_size = _clamp_int(
            prep.get("clahe_tile_grid_size"), 4, 16, cfg.preprocessing.clahe_tile_grid_size
        )
        cfg.preprocessing.adaptive_gamma_low = _clamp_float(
            prep.get("adaptive_gamma_low"), 0.35, 1.0, cfg.preprocessing.adaptive_gamma_low
        )
        cfg.preprocessing.adaptive_gamma_high = _clamp_float(
            prep.get("adaptive_gamma_high"), 1.0, 1.8, cfg.preprocessing.adaptive_gamma_high
        )
        cfg.preprocessing.quality_gate_min_sharpness = _clamp_float(
            prep.get("quality_gate_min_sharpness"), 5.0, 120.0, cfg.preprocessing.quality_gate_min_sharpness
        )

    subj = raw.get("subject_tracker") or {}
    if isinstance(subj, dict):
        cfg.subject_tracker.teacher_grace_sec = _clamp_float(
            subj.get("teacher_grace_sec"), 0.5, 4.0, cfg.subject_tracker.teacher_grace_sec
        )
        cfg.subject_tracker.teacher_match_iou_min = _clamp_float(
            subj.get("teacher_match_iou_min"), 0.05, 0.75, cfg.subject_tracker.teacher_match_iou_min
        )
        cfg.subject_tracker.teacher_face_sim_threshold = _clamp_float(
            subj.get("teacher_face_sim_threshold"), 0.25, 0.90, cfg.subject_tracker.teacher_face_sim_threshold
        )
        cfg.subject_tracker.wrist_gate_radius_scale = _clamp_float(
            subj.get("wrist_gate_radius_scale"), 1.2, 4.5, cfg.subject_tracker.wrist_gate_radius_scale
        )
        cfg.subject_tracker.enable_pose_landmarker = bool(
            subj.get("enable_pose_landmarker", cfg.subject_tracker.enable_pose_landmarker)
        )
        cfg.subject_tracker.pose_min_detection_confidence = _clamp_float(
            subj.get("pose_min_detection_confidence"), 0.20, 0.95, cfg.subject_tracker.pose_min_detection_confidence
        )
        cfg.subject_tracker.pose_min_presence_confidence = _clamp_float(
            subj.get("pose_min_presence_confidence"), 0.20, 0.95, cfg.subject_tracker.pose_min_presence_confidence
        )
        cfg.subject_tracker.pose_min_tracking_confidence = _clamp_float(
            subj.get("pose_min_tracking_confidence"), 0.20, 0.95, cfg.subject_tracker.pose_min_tracking_confidence
        )

    ht = raw.get("hand_tracking") or {}
    if isinstance(ht, dict):
        cfg.hand_tracking.min_detection_confidence = _clamp_float(
            ht.get("min_detection_confidence"), 0.25, 0.95, cfg.hand_tracking.min_detection_confidence
        )
        cfg.hand_tracking.min_presence_confidence = _clamp_float(
            ht.get("min_presence_confidence"), 0.25, 0.95, cfg.hand_tracking.min_presence_confidence
        )
        cfg.hand_tracking.min_tracking_confidence = _clamp_float(
            ht.get("min_tracking_confidence"), 0.25, 0.95, cfg.hand_tracking.min_tracking_confidence
        )
        cfg.hand_tracking.roi_area_threshold = _clamp_float(
            ht.get("roi_area_threshold"), 0.02, 0.25, cfg.hand_tracking.roi_area_threshold
        )
        cfg.hand_tracking.roi_expand_factor = _clamp_float(
            ht.get("roi_expand_factor"), 1.3, 3.2, cfg.hand_tracking.roi_expand_factor
        )
        cfg.hand_tracking.roi_padding_ratio = _clamp_float(
            ht.get("roi_padding_ratio"), 0.15, 1.0, cfg.hand_tracking.roi_padding_ratio
        )
        cfg.hand_tracking.roi_target_size = _clamp_int(
            ht.get("roi_target_size"), 224, 512, cfg.hand_tracking.roi_target_size
        )

    g3 = raw.get("gesture_3d") or {}
    if isinstance(g3, dict):
        cfg.gesture_3d.finger_extended_max_deg = _clamp_float(
            g3.get("finger_extended_max_deg"), 15.0, 65.0, cfg.gesture_3d.finger_extended_max_deg
        )
        cfg.gesture_3d.finger_half_bent_max_deg = _clamp_float(
            g3.get("finger_half_bent_max_deg"), 45.0, 115.0, cfg.gesture_3d.finger_half_bent_max_deg
        )
        cfg.gesture_3d.pinch_enter_ratio = _clamp_float(
            g3.get("pinch_enter_ratio"), 0.15, 0.40, cfg.gesture_3d.pinch_enter_ratio
        )
        cfg.gesture_3d.pinch_exit_ratio = _clamp_float(
            g3.get("pinch_exit_ratio"), 0.30, 0.65, cfg.gesture_3d.pinch_exit_ratio
        )
        cfg.gesture_3d.palm_facing_min_z = _clamp_float(
            g3.get("palm_facing_min_z"), 0.05, 0.65, cfg.gesture_3d.palm_facing_min_z
        )
        cfg.gesture_3d.one_euro_min_cutoff = _clamp_float(
            g3.get("one_euro_min_cutoff"), 0.30, 5.0, cfg.gesture_3d.one_euro_min_cutoff
        )
        cfg.gesture_3d.one_euro_beta = _clamp_float(
            g3.get("one_euro_beta"), 0.001, 0.5, cfg.gesture_3d.one_euro_beta
        )
        cfg.gesture_3d.one_euro_d_cutoff = _clamp_float(
            g3.get("one_euro_d_cutoff"), 0.20, 5.0, cfg.gesture_3d.one_euro_d_cutoff
        )
        cfg.gesture_3d.voting_window_size = _clamp_int(
            g3.get("voting_window_size"), 3, 11, cfg.gesture_3d.voting_window_size
        )
        cfg.gesture_3d.voting_min_votes = _clamp_int(
            g3.get("voting_min_votes"), 2, cfg.gesture_3d.voting_window_size, cfg.gesture_3d.voting_min_votes
        )
        cfg.gesture_3d.debounce_ms = _clamp_float(
            g3.get("debounce_ms"), 40.0, 400.0, cfg.gesture_3d.debounce_ms
        )
        cfg.gesture_3d.grace_period_ms = _clamp_float(
            g3.get("grace_period_ms"), 80.0, 400.0, cfg.gesture_3d.grace_period_ms
        )

    hud = raw.get("hud") or {}
    if isinstance(hud, dict):
        cfg.hud.debug_hud_default_enabled = bool(
            hud.get("debug_hud_default_enabled", cfg.hud.debug_hud_default_enabled)
        )

    # Environment overrides
    if os.environ.get("AIRTEACH_CAMERA_HD", "").strip() in ("1", "true", "TRUE"):
        cfg.camera.prefer_hd = True
    if os.environ.get("AIRTEACH_VISION_DEBUG_HUD", "").strip() in ("1", "true", "TRUE"):
        cfg.hud.debug_hud_default_enabled = True
    if os.environ.get("AIRTEACH_HAND_MIN_CONFIDENCE", "").strip():
        cfg.hand_tracking.min_detection_confidence = _clamp_float(
            os.environ["AIRTEACH_HAND_MIN_CONFIDENCE"],
            0.25,
            0.95,
            cfg.hand_tracking.min_detection_confidence,
        )

    return cfg


def save_vision_config(cfg: VisionConfig, config_path: Path | str | None = None) -> Path:
    target = Path(config_path) if config_path else DEFAULT_VISION_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return target
