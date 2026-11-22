#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIRTEACH AI — CLEAN STUDIO DESKTOP APPLICATION (DNT - Victer)
PySide6 + OpenCV + MediaPipe + ArcFace 5-Point 512D + Zero-Stray FSM + Hybrid Shape + Gemini 3.5 Flash
"""

import os
import sys
import time
import math
import hmac
import json
import queue
import threading
import wave
import hashlib
import secrets
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Sequence, Tuple, Dict, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np

from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
    QTimer,
    QPointF,
    QRectF,
    QSize,
    QByteArray,
    QBuffer,
    QIODevice,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QFont,
    QFontDatabase,
    QPainterPath,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextBrowser,
    QFrame,
    QFileDialog,
    QSlider,
    QCheckBox,
    QProgressBar,
    QComboBox,
    QTabWidget,
    QStackedWidget,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
)

try:
    from PySide6.QtSvgWidgets import QSvgWidget
    from PySide6.QtSvg import QSvgRenderer
    HAS_QTSVG = True
except Exception:
    HAS_QTSVG = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI_SDK = True
except Exception:
    HAS_GENAI_SDK = False

from cv_engine.drawing_fsm import (
    DrawingHudMode,
    DrawingFsmUpdate,
    ZeroStrayStrokeDrawingFsm,
    UNDO_COOLDOWN_MS,
    interpolate_bridge_points,
)
from cv_engine.face_id import (
    ArcFace512Extractor,
    DEFAULT_ARCFACE_THRESHOLD,
    multi_template_cosine_similarity,
)
from cv_engine.gemini_client import (
    ApiConnectionStatus,
    DEFAULT_GEMINI_MODEL,
    GeminiHealthCheckResult,
    ResilientGeminiClient,
    build_model_chain,
    clear_ui_api_key,
    compress_snapshot_png,
    mask_api_key,
    resolve_api_key_layered,
    save_ui_api_settings,
)
from cv_engine.hand_gesture import HandGestureEngine, classify_hand_3d_vectors
from cv_engine.preprocessing import (
    AdaptiveExposureController,
    preprocess_adaptive_frame,
)
from cv_engine.subject_tracker import PoseTorsoEstimator, PrimaryTeacherLockTracker
from cv_engine.vision_config import load_vision_config
from cv_engine.lasso_ai_service import (
    LassoQuickAction,
    LassoSelectionRegion,
    build_lasso_prompt,
    compute_lasso_bounding_box,
    enhance_lasso_crop_png,
    extract_clean_beautified_formula,
    stroke_inside_lasso,
)
from cv_engine.local_ai_manager import (
    AiRoutingMode,
    DiscoveredLocalServer,
    LocalAIManager,
    LocalAiRunResult,
)
from cv_engine.output_sanitizer import (
    OutputSanitizer,
    StreamingSanitizer,
    render_latex_to_png_bytes,
)
from cv_engine.app_paths import (
    get_config_file_path,
    get_user_data_dir,
    resource_path,
    set_windows_app_user_model_id,
    setup_app_logging,
)
from cv_engine.performance_pipeline import (
    AutoProfileController,
    DirtyCanvasBackingTracker,
    PerformancePipeline,
    PerformanceProfile,
    StaggeredInferenceScheduler,
    compute_cv_tracking_size,
)
from cv_engine.preprocessing import preprocess_adaptive_frame
from cv_engine.shape_recognizer import beautify_stroke, catmull_rom_spline
from cv_engine.whiteboard_agent import (
    WhiteboardAction,
    WhiteboardAgent,
    build_vector_shape_strokes,
    find_free_board_region,
    find_free_regions_normalized,
)


def _load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except OSError as exc:
        _ = exc


_load_dotenv_if_present()


# =============================================================================
# DESIGN TOKENS & DIRECTORIES
# =============================================================================
BG_DARK = "#0F172A"
SURFACE_DARK = "#1E293B"
BORDER_SLATE = "#334155"
PRIMARY_INDIGO = "#4F46E5"
SECONDARY_TEAL = "#06B6D4"
NEON_CYAN = "#00F5D4"
BRIGHT_YELLOW = "#FEE440"
ALERT_ROSE = "#F43F5E"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

DATA_DIR = get_user_data_dir()
FACE_EMB_DIR = DATA_DIR / "face_embeddings"
USERS_DB_PATH = DATA_DIR / "users_db.json"
AI_CACHE_PATH = DATA_DIR / "ai_cache.json"
MODELS_DIR = resource_path("models")
RECORDINGS_DIR = DATA_DIR / "recordings"
BRAND_LOGO_SVG = resource_path("assets/branding/airteach_logo.svg")
APP_LOGO_ICO = resource_path("assets/logo.ico")

for d in (DATA_DIR, FACE_EMB_DIR, RECORDINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
if not getattr(sys, "frozen", False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def make_brand_logo_widget(size: int = 44) -> QWidget:
    if HAS_QTSVG and BRAND_LOGO_SVG.exists():
        svg = QSvgWidget(str(BRAND_LOGO_SVG))
        svg.setFixedSize(QSize(size, size))
        svg.setStyleSheet("background: transparent;")
        return svg
    lbl = QLabel("AT")
    lbl.setFixedSize(size, size)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: {max(6, size // 5)}px; font-weight: 800;"
    )
    return lbl

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

HAND_BONE_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_model_file(path: Path, url: str) -> bool:
    if path.exists() and path.stat().st_size > 50_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        return path.exists() and path.stat().st_size > 50_000
    except Exception:
        return False


# =============================================================================
# MODULE 1: PERSISTENT USER DATABASE & FACE EMBEDDING STORAGE
# =============================================================================
def sanitize_username(username: str) -> str:
    cleaned = "".join(
        c.lower() if (c.isalnum() or c in ("_", "-")) else "_"
        for c in username.strip()
    )
    return cleaned or "teacher"


def hash_pin_pbkdf2(pin: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.strip().encode("utf-8"), salt.encode("ascii"), 120_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def verify_pin_hash(pin: str, stored_hash: Optional[str]) -> bool:
    if not stored_hash:
        return False
    clean_pin = pin.strip()
    if stored_hash.startswith("pbkdf2_sha256$"):
        parts = stored_hash.split("$")
        if len(parts) != 3:
            return False
        _, salt, expected_hex = parts
        dk = hashlib.pbkdf2_hmac(
            "sha256", clean_pin.encode("utf-8"), salt.encode("ascii"), 120_000
        )
        return hmac.compare_digest(dk.hex(), expected_hex)
    legacy = hashlib.sha256(clean_pin.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, stored_hash)


def load_users_db() -> List[Dict[str, Any]]:
    if not USERS_DB_PATH.exists():
        return []
    try:
        data = json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_users_db(users: List[Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_DB_PATH.write_text(
        json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    n = min(af.size, bf.size)
    na = float(np.linalg.norm(af[:n]))
    nb = float(np.linalg.norm(bf[:n]))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(af[:n], bf[:n]) / (na * nb), -1.0, 1.0))


# =============================================================================
# MODULE 3: BACKLIGHT-RESILIENT CAMERA & MEDIAPIPE HAND TRACKING WORKER
# =============================================================================
def enhance_backlit_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """Automatic Multi-Condition Exposure & Sharpening Recovery via `cv_engine.preprocessing`."""
    enhanced, _ = preprocess_adaptive_frame(bgr_frame, is_rgb=False, quality_gate_min=22.0)
    return enhanced


class CameraHandWorker(QThread):
    frame_ready = Signal(dict)

    def __init__(self, initial_cam_idx: int = 0) -> None:
        super().__init__()
        self.vision_cfg = load_vision_config()
        self.cam_idx = initial_cam_idx
        self.auto_hdr_backlight = True
        self.debug_hud_enabled = bool(self.vision_cfg.hud.debug_hud_default_enabled)
        self.prefer_hd = bool(self.vision_cfg.camera.prefer_hd)
        self.running = True
        self._camera_paused = False
        self._is_processing = False
        self._switch_requested: Optional[int] = None
        self._reopen_requested: bool = False
        self._was_pinched = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._frame_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._cam_fps_smoothed: float = 0.0

    def set_debug_hud(self, enabled: bool) -> None:
        self.debug_hud_enabled = bool(enabled)

    def set_hd_resolution(self, hd: bool) -> None:
        if self.prefer_hd != bool(hd):
            self.prefer_hd = bool(hd)
            self._reopen_requested = True

    def select_camera(self, idx: int) -> None:
        self._camera_paused = False
        self._switch_requested = idx

    def release_camera_hardware(self) -> None:
        """Pause streaming and immediately invoke `cap.release()` on next loop tick."""
        self._camera_paused = True

    def resume_camera_hardware(self) -> None:
        self._camera_paused = False

    def stop(self) -> None:
        self.running = False
        self._frame_event.set()
        self.wait(2000)

    def _open_camera(self, idx: int) -> Optional[cv2.VideoCapture]:
        req_w = self.vision_cfg.camera.hd_width if self.prefer_hd else self.vision_cfg.camera.width
        req_h = self.vision_cfg.camera.hd_height if self.prefer_hd else self.vision_cfg.camera.height
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                cap.set(cv2.CAP_PROP_FPS, self.vision_cfg.camera.target_fps)
                ok, test_frame = cap.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    return cap
                cap.release()
        return None

    def _capture_loop(self) -> None:
        """Dedicated non-blocking camera capture thread keeping only the latest frame (zero queue buildup)."""
        cap = None if self._camera_paused else self._open_camera(self.cam_idx)
        paused_notice_sent = False
        no_cam_notice_sent = False
        cap_last_ts = time.perf_counter()

        while self.running:
            if self._camera_paused:
                if cap is not None:
                    cap.release()
                    cap = None
                    paused_notice_sent = False
                with self._frame_lock:
                    self._latest_frame = None
                if not paused_notice_sent:
                    paused_notice_sent = True
                    self.frame_ready.emit({
                        "ok": False,
                        "paused": True,
                        "cam_idx": self.cam_idx,
                        "fps": 0.0,
                        "qimage": None,
                        "raw_bgr": None,
                        "lighting": {"mean_luma": 0.0, "laplacian_var": 0.0, "condition": "PAUSED"},
                        "face_bbox": None,
                        "face_5pt": None,
                        "face_emb_512": None,
                        "liveness_diff": 0.0,
                        "gesture": "IDLE",
                        "pose_tag": "IDLE",
                        "pinch_ratio": None,
                        "cursor_norm": None,
                        "left_pan_active": False,
                    })
                time.sleep(0.05)
                continue

            paused_notice_sent = False
            if self._switch_requested is not None or self._reopen_requested:
                if self._switch_requested is not None:
                    self.cam_idx = self._switch_requested
                    self._switch_requested = None
                self._reopen_requested = False
                if cap is not None:
                    cap.release()
                cap = self._open_camera(self.cam_idx)
                no_cam_notice_sent = False
                with self._frame_lock:
                    self._latest_frame = None

            if cap is None or not cap.isOpened():
                if not no_cam_notice_sent:
                    no_cam_notice_sent = True
                    self.frame_ready.emit({
                        "ok": False,
                        "paused": False,
                        "cam_idx": self.cam_idx,
                        "fps": 0.0,
                        "qimage": None,
                        "raw_bgr": None,
                        "lighting": {"mean_luma": 0.0, "laplacian_var": 0.0, "condition": "NO_CAM"},
                        "face_bbox": None,
                        "face_5pt": None,
                        "face_emb_512": None,
                        "liveness_diff": 0.0,
                        "gesture": "IDLE",
                        "pose_tag": "IDLE",
                        "pinch_ratio": None,
                        "cursor_norm": None,
                        "left_pan_active": False,
                    })
                time.sleep(0.25)
                cap = self._open_camera(self.cam_idx)
                continue

            no_cam_notice_sent = False
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            now_cap = time.perf_counter()
            cap_dt = max(1e-3, now_cap - cap_last_ts)
            cap_last_ts = now_cap
            inst_cap_fps = min(120.0, 1.0 / cap_dt)
            self._cam_fps_smoothed = (
                inst_cap_fps
                if self._cam_fps_smoothed <= 0.1
                else (0.85 * self._cam_fps_smoothed + 0.15 * inst_cap_fps)
            )

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_seq += 1
            self._frame_event.set()

        if cap is not None:
            cap.release()

    def run(self) -> None:
        vcfg = self.vision_cfg
        face_extractor = ArcFace512Extractor(models_dir=MODELS_DIR, min_detection_confidence=0.55)
        exposure_ctrl = AdaptiveExposureController(
            check_interval_n=vcfg.preprocessing.exposure_check_interval_n,
            low_light_mean_luma=vcfg.preprocessing.low_light_mean_luma,
            backlit_p95_luma=vcfg.preprocessing.backlit_p95_luma,
            flicker_delta_luma=vcfg.preprocessing.flicker_delta_luma,
            clahe_clip_limit=vcfg.preprocessing.clahe_clip_limit,
            tile_grid_size=vcfg.preprocessing.clahe_tile_grid_size,
            quality_gate_min=vcfg.preprocessing.quality_gate_min_sharpness,
        )
        pose_estimator = PoseTorsoEstimator(
            models_dir=MODELS_DIR,
            enable_pose_landmarker=vcfg.subject_tracker.enable_pose_landmarker,
            min_det_conf=vcfg.subject_tracker.pose_min_detection_confidence,
            min_pres_conf=vcfg.subject_tracker.pose_min_presence_confidence,
            min_track_conf=vcfg.subject_tracker.pose_min_tracking_confidence,
        )
        teacher_tracker = PrimaryTeacherLockTracker(
            teacher_grace_sec=vcfg.subject_tracker.teacher_grace_sec,
            match_iou_min=vcfg.subject_tracker.teacher_match_iou_min,
            face_sim_threshold=vcfg.subject_tracker.teacher_face_sim_threshold,
            wrist_gate_radius_scale=vcfg.subject_tracker.wrist_gate_radius_scale,
        )
        hand_engine = HandGestureEngine(
            models_dir=MODELS_DIR,
            min_detection_confidence=vcfg.hand_tracking.min_detection_confidence,
            min_presence_confidence=vcfg.hand_tracking.min_presence_confidence,
            min_tracking_confidence=vcfg.hand_tracking.min_tracking_confidence,
            roi_area_threshold=vcfg.hand_tracking.roi_area_threshold,
            roi_expand_factor=vcfg.hand_tracking.roi_expand_factor,
            roi_padding_ratio=vcfg.hand_tracking.roi_padding_ratio,
            roi_target_size=vcfg.hand_tracking.roi_target_size,
            pinch_enter_ratio=vcfg.gesture_3d.pinch_enter_ratio,
            pinch_exit_ratio=vcfg.gesture_3d.pinch_exit_ratio,
            palm_facing_min_z=vcfg.gesture_3d.palm_facing_min_z,
            grace_period_sec=vcfg.gesture_3d.grace_period_ms / 1000.0,
            voting_window_size=vcfg.gesture_3d.voting_window_size,
            voting_min_votes=vcfg.gesture_3d.voting_min_votes,
            one_euro_min_cutoff=vcfg.gesture_3d.one_euro_min_cutoff,
            one_euro_beta=vcfg.gesture_3d.one_euro_beta,
        )

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="airteach-py-capture", daemon=True
        )
        self._capture_thread.start()

        prev_gray: Optional[np.ndarray] = None
        last_time = time.perf_counter()
        smoothed_fps = 0.0
        last_processed_seq = -1
        profile_ctrl = AutoProfileController(
            initial_profile=PerformanceProfile.BALANCED,
            auto_adapt=True,
            downgrade_threshold_ms=45.0,
        )
        scheduler = StaggeredInferenceScheduler()
        cached_face_info: Dict[str, Any] = {"face": None, "face_boxes": [], "face_eyes": None}
        cached_poses: List[Any] = []
        last_teacher_seen_ts = time.monotonic()

        try:
            while self.running:
                self._frame_event.wait(timeout=0.05)
                self._frame_event.clear()

                if not self.running or self._camera_paused:
                    prev_gray = None
                    continue

                if self._is_processing:
                    continue

                with self._frame_lock:
                    if self._latest_frame is None or self._frame_seq == last_processed_seq:
                        continue
                    frame = self._latest_frame.copy()
                    last_processed_seq = self._frame_seq

                self._is_processing = True
                try:
                    t_cv_start = time.perf_counter()
                    now_mono = time.monotonic()
                    preset = profile_ctrl.preset
                    exposure_ctrl.check_interval_n = preset.exposure_interval_n

                    dt = max(1e-3, t_cv_start - last_time)
                    last_time = t_cv_start
                    inst_fps = min(120.0, 1.0 / dt)
                    smoothed_fps = inst_fps if smoothed_fps <= 0.1 else (0.8 * smoothed_fps + 0.2 * inst_fps)

                    bgr = cv2.flip(frame, 1)
                    if self.auto_hdr_backlight and preset.enable_full_frame_clahe:
                        bgr, light_diag = exposure_ctrl.process(bgr, is_rgb=False, enabled=True)
                        light_metrics = light_diag.to_dict()
                    else:
                        light_metrics = {"lighting_mode": "NORMAL", "clahe_bypassed": True}

                    h, w = bgr.shape[:2]
                    # Strided liveness diff on 1/4 resolution to save CPU on i3
                    gray_sub = cv2.cvtColor(bgr[::4, ::4], cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None and prev_gray.shape == gray_sub.shape:
                        liveness_diff = float(np.mean(cv2.absdiff(gray_sub, prev_gray))) / 12.0
                        liveness_diff = float(np.clip(liveness_diff, 0.0, 1.0))
                    else:
                        liveness_diff = 0.0
                    prev_gray = gray_sub

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    has_locked = teacher_tracker.locked_track_id is not None
                    teacher_lost_sec = (now_mono - last_teacher_seen_ts) if has_locked else 999.0
                    run_pose, run_face = scheduler.next_tick(
                        now_mono,
                        preset,
                        has_locked_teacher=has_locked,
                        teacher_lost_sec=teacher_lost_sec,
                    )

                    if run_face or run_pose:
                        sw, sh = preset.search_size
                        aux_rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA) if (w > sw or h > sh) else rgb
                        if run_face:
                            cached_face_info = face_extractor.detect_and_extract(
                                aux_rgb,
                                quality_gate_min_sharpness=vcfg.preprocessing.quality_gate_min_sharpness,
                            )
                        if run_pose:
                            cached_poses = pose_estimator.detect_poses(aux_rgb)

                    face_boxes = cached_face_info.get("face_boxes") or []
                    subj_snap = teacher_tracker.update(
                        face_boxes=face_boxes,
                        poses=cached_poses,
                        now_sec=now_mono,
                    )
                    if subj_snap.locked_teacher_id is not None:
                        last_teacher_seen_ts = now_mono

                    wrist_hints = teacher_tracker.get_proactive_wrist_roi_boxes(
                        subj_snap,
                        roi_expand_factor=vcfg.hand_tracking.roi_expand_factor,
                    )
                    primary_face = subj_snap.locked_face_bbox or cached_face_info.get("face")
                    face_bbox = tuple(primary_face) if primary_face and len(primary_face) == 4 else None

                    # Tracking-First Dual-Resolution ROI (720p crop -> ROI-only sharpen)
                    left_hand, right_hand = hand_engine.detect_hands(
                        rgb,
                        face_boxes,
                        teacher_tracker=teacher_tracker,
                        teacher_snapshot=subj_snap,
                        wrist_roi_hints=wrist_hints,
                        search_size=preset.search_size,
                        roi_target_size=preset.roi_size,
                        enable_roi_sharpen=preset.enable_roi_sharpen,
                    )
                    cv_latency_ms = (time.perf_counter() - t_cv_start) * 1000.0
                    profile_ctrl.record_inference_ms(cv_latency_ms)

                    display_bgr = bgr.copy()
                    # Visualize Subject Tracking Lock when Debug HUD enabled (or standard subtle primary box)
                    if self.debug_hud_enabled:
                        if subj_snap.locked_torso_bbox:
                            tx, ty, tw, th = subj_snap.locked_torso_bbox
                            cv2.rectangle(
                                display_bgr,
                                (int(tx * w), int(ty * h)),
                                (int((tx + tw) * w), int((ty + th) * h)),
                                (34, 197, 94),
                                2,
                                cv2.LINE_AA,
                            )
                            tid_lbl = f"TEACHER #{subj_snap.locked_teacher_id} ({subj_snap.mode})"
                            cv2.putText(
                                display_bgr,
                                tid_lbl,
                                (int(tx * w) + 4, max(16, int(ty * h) + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (34, 197, 94),
                                1,
                                cv2.LINE_AA,
                            )
                        for ig_box in subj_snap.ignored_face_bboxes:
                            ix, iy, iw, ih = ig_box
                            cv2.rectangle(
                                display_bgr,
                                (int(ix * w), int(iy * h)),
                                (int((ix + iw) * w), int((iy + ih) * h)),
                                (68, 68, 239),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                display_bgr,
                                "IGNORED",
                                (int(ix * w) + 2, max(14, int(iy * h) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,
                                (68, 68, 239),
                                1,
                                cv2.LINE_AA,
                            )
                        for wx, wy in subj_snap.locked_wrist_anchors:
                            cv2.circle(display_bgr, (int(wx * w), int(wy * h)), 5, (255, 220, 0), 2, cv2.LINE_AA)
                    elif face_bbox is not None:
                        fx, fy, fw, fh = face_bbox
                        cv2.rectangle(
                            display_bgr,
                            (int(fx * w), int(fy * h)),
                            (int((fx + fw) * w), int((fy + fh) * h)),
                            (212, 245, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    for h_obj in (left_hand, right_hand):
                        if not h_obj or "lm" not in h_obj:
                            continue
                        pts = h_obj["lm"]
                        for i, j in HAND_BONE_PAIRS:
                            p1 = (int(pts[i][0] * w), int(pts[i][1] * h))
                            p2 = (int(pts[j][0] * w), int(pts[j][1] * h))
                            cv2.line(display_bgr, p1, p2, (212, 245, 0), 2, cv2.LINE_AA)
                        for idx, pt in enumerate(pts):
                            r = 5 if idx in (4, 8, 12, 16, 20) else 3
                            cv2.circle(
                                display_bgr,
                                (int(pt[0] * w), int(pt[1] * h)),
                                r,
                                (64, 228, 254),
                                -1,
                                cv2.LINE_AA,
                            )

                    gesture = "IDLE"
                    pose_tag = "IDLE"
                    pinch_ratio = None
                    cursor_norm = None
                    primary = right_hand or left_hand
                    if primary is not None:
                        gesture = str(primary.get("gesture", "HOVER"))
                        pose_tag = str(primary.get("pose_tag", gesture))
                        pinch_ratio = primary.get("pinch_ratio")
                        cn = primary.get("cursor_norm")
                        if cn is not None and len(cn) >= 2:
                            cursor_norm = (float(cn[0]), float(cn[1]))
                        self._was_pinched = pose_tag == "PINCH"
                    else:
                        self._was_pinched = False

                    left_pan_active = bool(
                        left_hand and left_hand.get("gesture") in ("HOVER", "PAN")
                    )

                    disp_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                    qimg = QImage(
                        disp_rgb.data, w, h, disp_rgb.strides[0], QImage.Format_RGB888
                    ).copy()

                    primary_telemetry = {
                        "cam_fps": round(self._cam_fps_smoothed or smoothed_fps, 1),
                        "ai_fps": round(smoothed_fps, 1),
                        "cv_latency_ms": round(cv_latency_ms, 2),
                        "resolution": f"{w}x{h}",
                        "lighting_mode": light_metrics.get("lighting_mode", "NORMAL"),
                        "clahe_bypassed": bool(light_metrics.get("clahe_bypassed", False)),
                        "teacher_mode": subj_snap.mode,
                        "locked_teacher_id": subj_snap.locked_teacher_id,
                        "ignored_count": len(subj_snap.ignored_face_bboxes),
                        "finger_states": (primary or {}).get("finger_states", ["CURLED"] * 5),
                        "finger_angles_deg": (primary or {}).get("finger_angles_deg", [0.0] * 5),
                        "pinch_ratio": pinch_ratio,
                        "palm_facing_ok": (primary or {}).get("palm_facing_ok", True),
                        "palm_facing_score": (primary or {}).get("palm_facing_score", 0.0),
                        "stage2_roi_zoom": bool((primary or {}).get("stage2_roi_zoom", False)),
                        "kalman_predicted": bool((primary or {}).get("kalman_predicted", False)),
                        "fsm_state": (primary or {}).get("fsm_state", "IDLE"),
                    }

                    self.frame_ready.emit({
                        "ok": True,
                        "paused": False,
                        "cam_idx": self.cam_idx,
                        "fps": smoothed_fps,
                        "cam_fps": self._cam_fps_smoothed or smoothed_fps,
                        "qimage": qimg,
                        "raw_bgr": bgr,
                        "lighting": light_metrics,
                        "face_bbox": face_bbox,
                        "face_5pt": face_info.get("face_5pt"),
                        "face_emb_512": face_info.get("face_emb_512"),
                        "liveness_diff": liveness_diff,
                        "gesture": gesture,
                        "pose_tag": pose_tag,
                        "pinch_ratio": pinch_ratio,
                        "cursor_norm": cursor_norm,
                        "left_pan_active": left_pan_active,
                        "vision_telemetry": primary_telemetry,
                    })
                finally:
                    self._is_processing = False
        finally:
            if self._capture_thread is not None and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.0)
            pose_estimator.close()
            hand_engine.close()
            face_extractor.close()

    @staticmethod
    def _is_valid_hand(
        pts: List[Tuple[float, float, float]],
        face_bbox: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        if len(pts) < 21:
            return False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area <= 0.008:
            return False
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            fcx = fx + fw * 0.5
            fcy = fy + fh * 0.5
            px, py = pts[9][0], pts[9][1]
            if math.hypot(px - fcx, py - fcy) < max(fw, fh) * 0.42:
                return False
        return True

    @staticmethod
    def _classify_gesture(
        pts: List[Tuple[float, float, float]],
        was_pinched: bool = False,
    ) -> Tuple[str, Tuple[float, float]]:
        res = classify_hand_3d_vectors(pts, was_pinched=was_pinched)
        return (str(res["gesture"]), tuple(res["cursor_norm"]))  # type: ignore[return-value]


# =============================================================================
# GEMINI 3.5 FLASH ASYNC WORKER WITH RETRY, JSON STRUCTURED OUTPUT & CACHE
# =============================================================================
def _load_ai_cache() -> Dict[str, Any]:
    if not AI_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(AI_CACHE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_ai_cache(cache: Dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if len(cache) > 64:
            keys = sorted(cache.keys(), key=lambda k: str(cache[k].get("ts", "")))
            for k in keys[:-64]:
                cache.pop(k, None)
        AI_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        _ = exc


class GeminiHealthCheckWorker(QThread):
    completed = Signal(object)  # emits GeminiHealthCheckResult

    def __init__(self, api_key: str, model_id: Optional[str] = None) -> None:
        super().__init__()
        self.api_key = (api_key or "").strip()
        self.model_id = model_id

    def run(self) -> None:
        client = ResilientGeminiClient()
        res = client.test_connection(api_key=self.api_key, primary_model=self.model_id)
        self.completed.emit(res)


class GeminiVisionWorker(QThread):
    finished_text = Signal(str)
    status_changed = Signal(object)  # optional ApiConnectionStatus

    def __init__(
        self,
        api_key: str,
        prompt: str,
        png_bytes: Optional[bytes],
        structured_json: bool = False,
        primary_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.prompt = prompt
        self.png_bytes = png_bytes
        self.structured_json = structured_json
        self.primary_model = primary_model

    def _build_cache_key(self, model_chain: Tuple[str, ...]) -> str:
        h = hashlib.sha256()
        h.update("|".join(model_chain).encode("utf-8"))
        h.update(f"|json={self.structured_json}|".encode("utf-8"))
        h.update(self.prompt.encode("utf-8"))
        if self.png_bytes:
            h.update(self.png_bytes)
        return h.hexdigest()

    @staticmethod
    def _format_structured_response(raw_text: str) -> str:
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.lstrip("`").removeprefix("json").strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                parts = []
                if obj.get("summary"):
                    parts.append(f"**Tóm tắt nội dung:**\n{obj['summary']}")
                if obj.get("steps"):
                    steps_txt = "\n".join(f"- {s}" for s in obj["steps"])
                    parts.append(f"**Lời giải / Phân tích từng bước:**\n{steps_txt}")
                if obj.get("quiz"):
                    q_txt = "\n".join(f"- {q}" for q in obj["quiz"])
                    parts.append(f"**Câu hỏi gợi mở:**\n{q_txt}")
                if parts:
                    return "\n\n".join(parts)
        except Exception as exc:
            _ = exc
        return raw_text

    def run(self) -> None:
        chain_list = build_model_chain(primary_model=self.primary_model)
        primary_model = chain_list[0] if chain_list else DEFAULT_GEMINI_MODEL
        model_chain = tuple(chain_list)
        cache_key = self._build_cache_key(model_chain)
        cache = _load_ai_cache()

        if not self.api_key:
            cached = cache.get(cache_key)
            if cached and cached.get("text"):
                self.finished_text.emit(f"*(Offline Cache)*\n\n{cached['text']}")
                return
            self.status_changed.emit(ApiConnectionStatus.UNCONFIGURED)
            self.finished_text.emit(
                "Chế độ Offline / Chưa cấu hình `GEMINI_API_KEY` trong UI Settings hoặc `.env`.\n"
                "• Tất cả tính năng Thị giác máy tính (Hands/Face ID), Bảng trắng không chạm, Nắn hình lai và Đăng nhập cục bộ vẫn hoạt động 100%.\n"
                f"• Để dùng `{primary_model}`, hãy nhập API Key ở phần Cài đặt API Key phía trên."
            )
            return

        prompt_text = self.prompt
        if self.structured_json:
            prompt_text = (
                f"{self.prompt}\n\nHãy trả lời bằng JSON hợp lệ gồm các khóa: "
                "`summary` (string), `steps` (list string), `quiz` (list string)."
            )

        resilient = ResilientGeminiClient()
        res = resilient.generate_multimodal(
            api_key=self.api_key,
            prompt=prompt_text,
            png_bytes=self.png_bytes,
            primary_model=primary_model,
            fallback_models=chain_list[1:],
            structured_json=self.structured_json,
        )

        if res.ok:
            self.status_changed.emit(ApiConnectionStatus.CONNECTED)
            formatted = (
                self._format_structured_response(res.text)
                if self.structured_json
                else res.text
            )
            cache[cache_key] = {
                "model": res.model_used,
                "text": formatted,
                "ts": datetime.now().isoformat(),
            }
            _save_ai_cache(cache)
            self.finished_text.emit(f"*[{res.model_used}]*\n\n{formatted}")
        else:
            self.status_changed.emit(ApiConnectionStatus.ERROR)
            cached = cache.get(cache_key)
            if cached and cached.get("text"):
                self.finished_text.emit(
                    f"*(Fallback Cache · {res.error_message_vi})*\n\n{OutputSanitizer.sanitize_plain_text(cached['text'])}"
                )
            else:
                self.finished_text.emit(
                    f"⚠ **{res.error_message_vi}**\n\n"
                    "Bảng trắng, nhận diện cử chỉ tay và nắn hình vẫn đang hoạt động bình thường ở chế độ Ngoại tuyến."
                )


class LocalDiscoveryWorker(QThread):
    """Background auto-discovery & warmup for Ollama / LM Studio / llama.cpp."""

    discovered = Signal(list)

    def __init__(self, local_mgr: LocalAIManager) -> None:
        super().__init__()
        self.local_mgr = local_mgr

    def run(self) -> None:
        servers = self.local_mgr.discover_local_engines(timeout=0.45)
        if servers:
            self.local_mgr.warmup_async()
        self.discovered.emit(servers)


class HybridAiWorker(QThread):
    """Non-blocking AI worker supporting token-by-token streaming, OutputSanitizer, and Whiteboard actions."""

    token_delta = Signal(str)
    finished_result = Signal(object)  # emits LocalAiRunResult

    def __init__(
        self,
        local_mgr: LocalAIManager,
        prompt: str,
        *,
        history: Sequence[Dict[str, str]] = (),
        png_bytes: Optional[bytes] = None,
        cloud_api_key: str = "",
        cloud_model: Optional[str] = None,
        structured_json: bool = False,
        include_board_tools: bool = True,
    ) -> None:
        super().__init__()
        self.local_mgr = local_mgr
        self.prompt = prompt
        self.history = list(history)
        self.png_bytes = png_bytes
        self.cloud_api_key = cloud_api_key.strip()
        self.cloud_model = cloud_model
        self.structured_json = structured_json
        self.include_board_tools = include_board_tools

    def run(self) -> None:
        eff_prompt = self.prompt
        if self.include_board_tools and not self.structured_json:
            eff_prompt = f"{self.prompt}\n\n{WhiteboardAgent.build_tool_prompt_instruction()}"
        res = self.local_mgr.generate_hybrid(
            eff_prompt,
            history=self.history,
            png_bytes=self.png_bytes,
            cloud_api_key=self.cloud_api_key,
            cloud_model=self.cloud_model,
            structured_json=self.structured_json,
            on_token=lambda tok: self.token_delta.emit(tok),
        )
        self.finished_result.emit(res)


# =============================================================================
# MODULE 1: TWO-TAB SECURITY GATE (SIGN UP FIRST -> SIGN IN AFTER)
# =============================================================================
FACE_ENROLL_ANGLES = [
    "1/5: Nhìn thẳng vào Camera",
    "2/5: Hơi xoay đầu sang Trái (~15°)",
    "3/5: Hơi xoay đầu sang Phải (~15°)",
    "4/5: Hơi ngẩng đầu lên nhẹ",
    "5/5: Hơi cúi đầu xuống nhẹ",
]


class AuthGateWidget(QWidget):
    authenticated = Signal(dict)

    def __init__(self, cam_worker: CameraHandWorker) -> None:
        super().__init__()
        self.cam_worker = cam_worker
        self.users: List[Dict[str, Any]] = load_users_db()
        self.signup_samples: List[np.ndarray] = []
        self.last_raw_bgr: Optional[np.ndarray] = None
        self.last_face_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_face_emb_512: Optional[List[float]] = None
        self.last_lighting: Dict[str, Any] = {}
        self.last_liveness_diff: float = 0.0
        self.last_paused: bool = False
        self._build_ui()
        self._refresh_tabs()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)

        # Brand Header with SVG Logo (AirTeach AI — DNT - Victer)
        hdr_row = QHBoxLayout()
        hdr_row.addStretch(1)
        hdr_row.addWidget(make_brand_logo_widget(48))
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("AIRTEACH AI — CỔNG ĐĂNG KÝ & ĐĂNG NHẬP GIÁO VIÊN")
        title.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {TEXT_LIGHT};")
        sub_lbl = QLabel("Tổ chức DNT - Victer • Face ID 5-điểm (512-D ArcFace) & Bảo mật PBKDF2")
        sub_lbl.setStyleSheet(f"font-size: 12px; color: {SECONDARY_TEAL}; font-weight: 600;")
        title_box.addWidget(title)
        title_box.addWidget(sub_lbl)
        hdr_row.addLayout(title_box)
        hdr_row.addStretch(1)
        root.addLayout(hdr_row)

        body = QHBoxLayout()
        body.setSpacing(18)

        # Left Card: Live 4:3 Webcam + Circular Face Guide + Multi-Angle Capture
        left_card = QFrame()
        left_card.setStyleSheet(
            f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
        )
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 14, 16, 14)

        cam_hdr = QHBoxLayout()
        self.cam_title_lbl = QLabel("Webcam Face ID (4:3 • 112x112 5-Điểm)")
        self.cam_title_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 600; font-size: 13px;")
        cam_hdr.addWidget(self.cam_title_lbl)
        cam_hdr.addStretch(1)

        self.cam_power_btn = QPushButton("Tắt/Mở Cam")
        self.cam_power_btn.setFixedHeight(28)
        self.cam_power_btn.clicked.connect(self._on_toggle_auth_camera)
        cam_hdr.addWidget(self.cam_power_btn)

        for idx in (0, 1):
            btn = QPushButton(f"Cam {idx}")
            btn.setFixedSize(58, 28)
            btn.clicked.connect(lambda _=False, i=idx: self.cam_worker.select_camera(i))
            cam_hdr.addWidget(btn)
        left_layout.addLayout(cam_hdr)

        self.cam_preview_lbl = QLabel()
        self.cam_preview_lbl.setFixedSize(400, 300)
        self.cam_preview_lbl.setAlignment(Qt.AlignCenter)
        self.cam_preview_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 8px; color: {TEXT_MUTED}; font-weight: 600;"
        )
        left_layout.addWidget(self.cam_preview_lbl, alignment=Qt.AlignCenter)

        self.angle_hint_lbl = QLabel(f"Hướng dẫn góc mặt: {FACE_ENROLL_ANGLES[0]}")
        self.angle_hint_lbl.setAlignment(Qt.AlignCenter)
        self.angle_hint_lbl.setStyleSheet(f"color: {BRIGHT_YELLOW}; font-size: 12px; font-weight: 700;")
        left_layout.addWidget(self.angle_hint_lbl)

        self.capture_face_btn = QPushButton("[Chụp Mẫu Khuôn Mặt 5-Góc (0/5 mẫu)]")
        self.capture_face_btn.setFixedHeight(36)
        self.capture_face_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; font-weight: 700; border-radius: 8px;"
        )
        self.capture_face_btn.clicked.connect(self._on_capture_signup_face)
        left_layout.addWidget(self.capture_face_btn)

        self.cosine_bar = QProgressBar()
        self.cosine_bar.setRange(0, 100)
        self.cosine_bar.setValue(0)
        self.cosine_bar.setFormat(f"ArcFace 512-D Cosine: 0.00 / {DEFAULT_ARCFACE_THRESHOLD:.2f}")
        left_layout.addWidget(self.cosine_bar)

        body.addWidget(left_card, stretch=1)

        # Right Card: Two-Tab Widget (TAB 1: ĐĂNG KÝ | TAB 2: ĐĂNG NHẬP)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{ background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px; }}
            QTabBar::tab {{ background: {BG_DARK}; color: {TEXT_LIGHT}; padding: 10px 18px; font-weight: 700; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
            QTabBar::tab:selected {{ background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; }}
            """
        )

        # TAB 1: SIGN UP
        tab_signup = QWidget()
        su_layout = QVBoxLayout(tab_signup)
        su_layout.setContentsMargins(18, 16, 18, 16)
        su_layout.setSpacing(10)

        su_layout.addWidget(QLabel("1. Tên hiển thị:"))
        self.su_fullname_edit = QLineEdit()
        self.su_fullname_edit.setPlaceholderText("Nhập họ tên giáo viên...")
        su_layout.addWidget(self.su_fullname_edit)

        su_layout.addWidget(QLabel("2. Tên đăng nhập (username):"))
        self.su_username_edit = QLineEdit()
        self.su_username_edit.setPlaceholderText("Ví dụ: giaovien_01")
        su_layout.addWidget(self.su_username_edit)

        su_layout.addWidget(QLabel("3. Mã PIN 4-6 chữ số:"))
        self.su_pin_edit = QLineEdit()
        self.su_pin_edit.setEchoMode(QLineEdit.Password)
        self.su_pin_edit.setPlaceholderText("Nhập mã PIN 4-6 số...")
        su_layout.addWidget(self.su_pin_edit)

        su_layout.addWidget(QLabel("4. Xác nhận mã PIN:"))
        self.su_pin_confirm_edit = QLineEdit()
        self.su_pin_confirm_edit.setEchoMode(QLineEdit.Password)
        self.su_pin_confirm_edit.setPlaceholderText("Nhập lại mã PIN 4-6 số...")
        su_layout.addWidget(self.su_pin_confirm_edit)

        self.su_complete_btn = QPushButton("[Hoàn Tất Đăng Ký] -> Lưu Hồ Sơ Cục Bộ")
        self.su_complete_btn.setFixedHeight(40)
        self.su_complete_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; font-weight: 700; border-radius: 8px;"
        )
        self.su_complete_btn.clicked.connect(self._on_complete_signup)
        su_layout.addWidget(self.su_complete_btn)
        su_layout.addStretch(1)

        # TAB 2: SIGN IN
        tab_signin = QWidget()
        si_layout = QVBoxLayout(tab_signin)
        si_layout.setContentsMargins(18, 16, 18, 16)
        si_layout.setSpacing(10)

        si_layout.addWidget(QLabel("Chọn tài khoản Giáo viên đã đăng ký:"))
        self.si_user_combo = QComboBox()
        si_layout.addWidget(self.si_user_combo)

        si_layout.addWidget(
            QLabel(
                f"Nhập mã PIN đã đăng ký (hoặc bật Camera quét Face ID Cosine >= {DEFAULT_ARCFACE_THRESHOLD:.2f}):"
            )
        )
        self.si_pin_edit = QLineEdit()
        self.si_pin_edit.setEchoMode(QLineEdit.Password)
        self.si_pin_edit.setPlaceholderText("Nhập mã PIN 4-6 số...")
        self.si_pin_edit.returnPressed.connect(self._on_signin_with_pin)
        si_layout.addWidget(self.si_pin_edit)

        self.si_login_btn = QPushButton("[ĐĂNG NHẬP BẰNG MÃ PIN] -> Vào Studio Giảng Dạy")
        self.si_login_btn.setFixedHeight(40)
        self.si_login_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; font-weight: 700; border-radius: 8px;"
        )
        self.si_login_btn.clicked.connect(self._on_signin_with_pin)
        si_layout.addWidget(self.si_login_btn)
        si_layout.addStretch(1)

        self.tabs.addTab(tab_signup, "TAB 1: ĐĂNG KÝ TÀI KHOẢN MỚI")
        self.tabs.addTab(tab_signin, "TAB 2: ĐĂNG NHẬP")
        body.addWidget(self.tabs, stretch=1)

        root.addLayout(body)

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {SECONDARY_TEAL};")
        root.addWidget(self.status_lbl)

        # Footer attribution (DNT - Victer)
        footer_lbl = QLabel("AirTeach AI • Phát triển bởi Tổ chức DNT - Victer • Lưu trữ cục bộ an toàn")
        footer_lbl.setAlignment(Qt.AlignCenter)
        footer_lbl.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        root.addWidget(footer_lbl)

    def _on_toggle_auth_camera(self) -> None:
        if self.cam_worker._camera_paused:
            self.cam_worker.resume_camera_hardware()
            self.cam_power_btn.setText("Tắt Cam")
        else:
            self.cam_worker.release_camera_hardware()
            self.cam_power_btn.setText("Bật Cam")

    def _refresh_tabs(self) -> None:
        self.users = load_users_db()
        self.si_user_combo.clear()
        for u in self.users:
            self.si_user_combo.addItem(f"{u.get('full_name')} ({u.get('username')})", u)

        if not self.users:
            self.tabs.setCurrentIndex(0)
            self.status_lbl.setText(
                "Cơ sở dữ liệu trống — Vui lòng tạo tài khoản Giáo viên lần đầu (PIN + 3..5 mẫu khuôn mặt)."
            )
        else:
            self.tabs.setCurrentIndex(1)
            self.status_lbl.setText(
                f"Đã sẵn sàng {len(self.users)} tài khoản cục bộ. Nhập PIN hoặc bật Camera quét Face ID (>= {DEFAULT_ARCFACE_THRESHOLD:.2f})."
            )

    def on_camera_packet(self, pkt: dict) -> None:
        if not self.isVisible():
            return
        self.last_paused = bool(pkt.get("paused", False))
        qimg: Optional[QImage] = pkt.get("qimage")
        self.last_raw_bgr = pkt.get("raw_bgr")
        self.last_face_bbox = pkt.get("face_bbox")
        self.last_face_emb_512 = pkt.get("face_emb_512")
        self.last_lighting = pkt.get("lighting") or {}
        self.last_liveness_diff = float(pkt.get("liveness_diff") or 0.0)

        if self.last_paused:
            self.cam_power_btn.setText("Bật Cam")
            self.cam_preview_lbl.setPixmap(QPixmap())
            self.cam_preview_lbl.setText(
                "Camera đã giải phóng an toàn.\nBấm [Bật Cam] khi cần quét Face ID."
            )
            return

        if qimg is None:
            self.cam_preview_lbl.setPixmap(QPixmap())
            self.cam_preview_lbl.setText("Chưa tìm thấy Webcam.\nHãy chọn Cam 0 / Cam 1.")
            return

        self.cam_power_btn.setText("Tắt Cam")
        canvas = qimg.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_col = QColor(NEON_CYAN) if self.last_face_bbox is not None else QColor(BRIGHT_YELLOW)
        painter.setPen(QPen(pen_col, 2.2))
        painter.drawEllipse(QPointF(canvas.width() * 0.5, canvas.height() * 0.5), 88.0, 88.0)
        painter.end()
        self.cam_preview_lbl.setPixmap(QPixmap.fromImage(canvas))

        # On TAB 2 (SIGN IN), evaluate live 512-D aligned embedding across enrolled multi-angle templates
        if (
            self.tabs.currentIndex() == 1
            and self.users
            and self.last_face_emb_512 is not None
        ):
            idx = max(0, self.si_user_combo.currentIndex())
            user = self.users[idx]
            emb_path = Path(user.get("face_embedding_path", ""))
            if emb_path.exists():
                try:
                    enrolled_arr = np.load(str(emb_path))
                    if enrolled_arr.ndim == 1:
                        templates = [enrolled_arr.tolist()]
                    else:
                        templates = [row.tolist() for row in enrolled_arr]
                    sim = multi_template_cosine_similarity(
                        self.last_face_emb_512, templates, mode="top2"
                    )
                    self.cosine_bar.setValue(int(max(0.0, min(1.0, sim)) * 100))
                    self.cosine_bar.setFormat(
                        f"ArcFace 5-Pt Cosine: {sim:.3f} / {DEFAULT_ARCFACE_THRESHOLD:.2f} (Liveness: {self.last_liveness_diff:.2f})"
                    )
                    if sim >= DEFAULT_ARCFACE_THRESHOLD and self.last_liveness_diff >= 0.03:
                        self.cam_worker.resume_camera_hardware()
                        self.authenticated.emit(user)
                except Exception as exc:
                    _ = exc

    def _on_capture_signup_face(self) -> None:
        if self.cam_worker._camera_paused:
            self.cam_worker.resume_camera_hardware()
            self.status_lbl.setText("Đang khởi động lại Camera... Vui lòng nhìn vào Webcam và bấm chụp lần nữa.")
            return

        if self.last_face_emb_512 is None or self.last_face_bbox is None:
            sharpness = float(self.last_lighting.get("laplacian_var") or 0.0)
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            if self.last_face_bbox is not None and sharpness < 22.0:
                self.status_lbl.setText(
                    f"Ảnh đang mờ (Độ nét Laplacian={sharpness:.1f} < 22.0). Vui lòng giữ yên mặt và tăng sáng!"
                )
            else:
                self.status_lbl.setText(
                    "Chưa phát hiện đủ 5 điểm khuôn mặt sắc nét trong khung tròn! Hãy nhìn thẳng vào Camera."
                )
            return

        if len(self.signup_samples) >= 5:
            self.signup_samples.clear()

        vec = np.asarray(self.last_face_emb_512, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-7:
            vec = vec / norm
        self.signup_samples.append(vec)

        n = len(self.signup_samples)
        next_hint = FACE_ENROLL_ANGLES[min(n, 4)]
        self.angle_hint_lbl.setText(
            "Đã đủ 5/5 góc khuôn mặt!" if n >= 5 else f"Góc tiếp theo: {next_hint}"
        )
        self.capture_face_btn.setText(f"[Chụp Mẫu Khuôn Mặt ({n}/5 mẫu)]")
        self.status_lbl.setStyleSheet("color: #22C55E; font-weight: 700;")
        self.status_lbl.setText(
            f"Đã chụp mẫu {n}/5 (5-Point Affine 112x112 + Quality Gate). {'Có thể Hoàn tất Đăng ký!' if n >= 3 else 'Chụp thêm ít nhất 3/5 góc!'}"
        )

    def _on_complete_signup(self) -> None:
        full_name = self.su_fullname_edit.text().strip()
        raw_user = self.su_username_edit.text().strip()
        pin = self.su_pin_edit.text().strip()
        pin_confirm = self.su_pin_confirm_edit.text().strip()

        if not full_name or not raw_user:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText("Vui lòng nhập đầy đủ Họ và tên Giáo viên và Tên đăng nhập!")
            return
        if not (4 <= len(pin) <= 6 and pin.isdigit()):
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText("Mã PIN phải gồm từ 4 đến 6 chữ số!")
            return
        if pin != pin_confirm:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText("Mã PIN và Xác nhận mã PIN không khớp!")
            return
        if len(self.signup_samples) < 3:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText(
                f"Cần tối thiểu 3/5 mẫu góc khuôn mặt (hiện tại {len(self.signup_samples)}/5). Hãy bấm nút chụp mẫu!"
            )
            return

        username = sanitize_username(raw_user)
        tmpl_matrix = np.stack(self.signup_samples, axis=0).astype(np.float32)

        FACE_EMB_DIR.mkdir(parents=True, exist_ok=True)
        npy_path = FACE_EMB_DIR / f"{username}.npy"
        np.save(str(npy_path), tmpl_matrix)

        user_record = {
            "username": username,
            "full_name": full_name,
            "pin_hash": hash_pin_pbkdf2(pin),
            "face_embedding_path": str(npy_path).replace("\\", "/"),
            "template_count": int(tmpl_matrix.shape[0]),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        users = [u for u in load_users_db() if u.get("username") != username]
        users.append(user_record)
        save_users_db(users)

        self.signup_samples.clear()
        self.capture_face_btn.setText("[Chụp Mẫu Khuôn Mặt 5-Góc (0/5 mẫu)]")
        self.angle_hint_lbl.setText(f"Hướng dẫn góc mặt: {FACE_ENROLL_ANGLES[0]}")
        self.su_pin_edit.clear()
        self.su_pin_confirm_edit.clear()
        self._refresh_tabs()
        self.tabs.setCurrentIndex(1)
        self.status_lbl.setStyleSheet("color: #22C55E; font-weight: 700;")
        self.status_lbl.setText("Đăng ký thành công! Tài khoản đã lưu cục bộ. Chuyển sang Đăng nhập.")

    def _on_signin_with_pin(self) -> None:
        if not self.users:
            self.tabs.setCurrentIndex(0)
            return
        idx = max(0, self.si_user_combo.currentIndex())
        user = self.users[idx]
        pin = self.si_pin_edit.text().strip()
        stored_hash = user.get("pin_hash") or user.get("pin_sha256")
        if verify_pin_hash(pin, stored_hash):
            self.si_pin_edit.clear()
            self.cam_worker.resume_camera_hardware()
            self.authenticated.emit(user)
        else:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText(f"Mã PIN không chính xác cho tài khoản '{user.get('username')}'!")


# =============================================================================
# MODULE 2.3 & MODULE 4: CLEAN CENTER SMARTBOARD CANVAS (ZERO FAKE LABELS)
# =============================================================================
class SmartboardCanvas(QWidget):
    region_cropped_for_ai = Signal(bytes, QPixmap)
    lasso_action_requested = Signal(str, object)
    hud_status_changed = Signal(str, float, str)
    slides_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dark_mode = True
        self.grid_enabled = True
        self.slide_visible = True
        self.smart_shape_enabled = True
        self.hold_to_snap_350ms = False
        self.active_tool = "PEN"  # PEN, ERASER, LASER, SELECT_REGION, LASSO
        self.ink_color = QColor(NEON_CYAN)
        self.pen_thickness = 4

        self.drawing_fsm = ZeroStrayStrokeDrawingFsm()
        self.hud_mode: DrawingHudMode = DrawingHudMode.HOVER
        self.cooldown_remaining_ms: float = 0.0
        self.cancel_hold_progress: float = 0.0

        self.strokes: List[Dict[str, Any]] = []
        self.redo_strokes: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.ai_action_undo_stack: List[Dict[str, Any]] = []
        self.current_stroke: List[QPointF] = []
        self._stroke_start_ms: float = 0.0
        self._last_motion_ms: float = 0.0
        self._last_motion_pt: Optional[QPointF] = None
        self._hold_snap_ready: bool = True

        # Non-destructive Auto-Shape notification pill with clickable revert button
        self.shape_toast: Optional[Dict[str, Any]] = None
        self.toast_revert_rect: Optional[QRectF] = None

        self.cursor_pos: Optional[QPointF] = None
        self.cursor_mode: str = "HOVER"  # HOVER, DRAWING, LOCKED_COOLDOWN, PAN_ERASE, LASER
        self.selection_start: Optional[QPointF] = None
        self.selection_end: Optional[QPointF] = None
        self.lasso_points: List[QPointF] = []
        self.lasso_region: Optional[LassoSelectionRegion] = None

        self._dirty_tracker = DirtyCanvasBackingTracker()
        self._backing_pixmap: Optional[QPixmap] = None
        self._scene_rev: int = 0

        # Neutral clean guide slide only (zero personal sample data)
        self.slides: List[Dict[str, Any]] = [
            {
                "title": "DNT - Victer Guide • Hướng dẫn nhanh AirTeach AI",
                "subtitle": f"Bảng trắng không chạm & Nắn Hình học Lai + {DEFAULT_GEMINI_MODEL} / Local AI",
                "bullets": [
                    "• Chụm ngón (pinch < 0.28 giữ 3 frame): Vẽ • Nhấc bút (> 0.40): Tự cắt đuôi thừa 105ms.",
                    "• Mất dấu tạm < 150ms: Tự động giữ trạng thái & nối liền nét mượt không gián đoạn.",
                    "• Khoanh AI (Lasso / L): Khoanh vùng giải toán, làm đẹp LaTeX & AI tự ghi lên bảng.",
                ],
                "formula": "Ctrl+Z: Hoàn tác (400ms)  |  L: Khoanh vùng AI (Smart Lasso-to-AI)",
                "pixmap": None,
                "is_tutorial": True,
            }
        ]
        self.current_slide_idx = 0

        # Top-Left buttons: Toggle visibility + Remove clean guide card
        self.toggle_slide_btn = QPushButton("[Ẩn nội dung Slide để lấy bảng trống 100%]", self)
        self.toggle_slide_btn.setGeometry(14, 10, 290, 28)
        self.toggle_slide_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-weight: 700; font-size: 12px;"
        )
        self.toggle_slide_btn.clicked.connect(self.toggle_slide_visibility)

        self.clear_tutorial_btn = QPushButton("[Xóa hướng dẫn]", self)
        self.clear_tutorial_btn.setGeometry(312, 10, 128, 28)
        self.clear_tutorial_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-weight: 600; font-size: 11px;"
        )
        self.clear_tutorial_btn.clicked.connect(self.dismiss_tutorial_slides)

        # Smart Lasso Floating Action Bar (hidden until user completes a Lasso/Region select)
        self.lasso_action_bar = QFrame(self)
        self.lasso_action_bar.setStyleSheet(
            f"background: rgba(15, 23, 42, 240); border: 1.5px solid {NEON_CYAN}; border-radius: 10px;"
        )
        lab_layout = QHBoxLayout(self.lasso_action_bar)
        lab_layout.setContentsMargins(6, 5, 6, 5)
        lab_layout.setSpacing(4)
        for act_id, lbl_txt, col in (
            (LassoQuickAction.SOLVE_AI.value, "Giải bằng AI", PRIMARY_INDIGO),
            (LassoQuickAction.WRITE_SOLUTION_TO_BOARD.value, "Viết lời giải lên bảng", "#0D9488"),
            (LassoQuickAction.BEAUTIFY_LATEX.value, "Làm đẹp chữ/LaTeX", "#7C3AED"),
            (LassoQuickAction.EXPLAIN_QUICK.value, "Giải thích nhanh", SURFACE_DARK),
        ):
            btn = QPushButton(lbl_txt)
            btn.setFixedHeight(26)
            btn.setStyleSheet(
                f"background: {col}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )
            btn.clicked.connect(lambda _=False, aid=act_id: self._emit_lasso_action(aid))
            lab_layout.addWidget(btn)
        close_lasso_btn = QPushButton("✕")
        close_lasso_btn.setFixedSize(24, 24)
        close_lasso_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 6px; font-weight: 800; border: none;"
        )
        close_lasso_btn.clicked.connect(self.hide_lasso_bar)
        lab_layout.addWidget(close_lasso_btn)
        self.lasso_action_bar.adjustSize()
        self.lasso_action_bar.setVisible(False)

    def _bump_scene(self) -> None:
        self._scene_rev += 1
        self._dirty_tracker.mark_dirty()

    def set_dark_mode(self, dark: bool) -> None:
        self.dark_mode = bool(dark)
        self._bump_scene()
        self.update()

    def _emit_lasso_action(self, action_id: str) -> None:
        if self.lasso_region is not None:
            self.lasso_action_requested.emit(action_id, self.lasso_region)

    def hide_lasso_bar(self) -> None:
        self.lasso_action_bar.setVisible(False)
        self.lasso_region = None
        self.lasso_points.clear()
        self.selection_start = None
        self.selection_end = None
        self.update()

    def _show_lasso_action_bar(self, region: LassoSelectionRegion) -> None:
        self.lasso_region = region
        self.lasso_action_bar.adjustSize()
        bar_w = self.lasso_action_bar.width()
        bar_h = self.lasso_action_bar.height()
        x = max(10, min(max(10, self.width() - bar_w - 10), int(region.bbox_x)))
        y = int(region.bbox_y - bar_h - 8)
        if y < 44:
            y = min(max(44, self.height() - bar_h - 10), int(region.bbox_y + region.bbox_h + 8))
        self.lasso_action_bar.move(x, y)
        self.lasso_action_bar.setVisible(True)
        self.lasso_action_bar.raise_()

    def _occupied_rects(self) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        if self.slide_visible and self.slides:
            s_w = min(740.0, max(320.0, self.width() - 40.0))
            s_h = float(int(s_w * 9.0 / 16.0))
            rects.append((16.0, 46.0, s_w, s_h))
        for s in self.strokes:
            pts = s.get("points") or []
            if len(pts) >= 2:
                xs = [float(p.x()) for p in pts]
                ys = [float(p.y()) for p in pts]
                rects.append((min(xs), min(ys), max(20.0, max(xs) - min(xs)), max(20.0, max(ys) - min(ys))))
        for ann in self.annotations:
            rects.append((
                float(ann.get("x", 0.0)),
                float(ann.get("y", 0.0)),
                float(ann.get("w", 280.0)),
                float(ann.get("h", 48.0)),
            ))
        return rects

    def _make_annotation_entry(
        self,
        raw_content: str,
        x: float,
        y: float,
        fmt: str = "latex",
        font_size: int = 16,
        color_hex: Optional[str] = None,
    ) -> Dict[str, Any]:
        sanitized = OutputSanitizer.sanitize_plain_text(raw_content)
        fg_hex = color_hex or (NEON_CYAN if self.dark_mode else PRIMARY_INDIGO)
        pm: Optional[QPixmap] = None
        w_est = max(140.0, min(520.0, len(sanitized) * 10.5 + 32.0))
        h_est = max(42.0, sanitized.count("\n") * 24.0 + 40.0)
        if fmt == "latex":
            png_bytes = render_latex_to_png_bytes(raw_content, color_hex=fg_hex, fontsize=max(12, min(20, font_size)), dpi=145)
            if png_bytes:
                tmp_pm = QPixmap()
                if tmp_pm.loadFromData(png_bytes, "PNG") and not tmp_pm.isNull():
                    pm = tmp_pm
                    w_est = max(90.0, float(pm.width()) + 20.0)
                    h_est = max(40.0, float(pm.height()) + 14.0)
        return {
            "raw": raw_content,
            "text": sanitized,
            "format": fmt,
            "x": float(x),
            "y": float(y),
            "w": w_est,
            "h": h_est,
            "color": fg_hex,
            "font_size": font_size,
            "pixmap": pm,
            "reveal_until_ms": self._now_ms() + 420.0,
        }

    def execute_whiteboard_actions(
        self,
        actions: Sequence[WhiteboardAction],
        preferred_anchor: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Execute structured Agentic Whiteboard actions onto empty canvas space with Undo tracking."""
        if not actions:
            return ""

        canvas_sz = (float(max(640, self.width())), float(max(420, self.height())))
        reports: List[str] = []

        for act in actions:
            if act.tool == "clear_board":
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                self.strokes.clear()
                self.annotations.clear()
                self._bump_scene()
                reports.append("Đã xóa sạch bảng.")
            elif act.tool == "undo_last_ai_action":
                if self.undo_last_ai_action():
                    reports.append("Đã hoàn tác thao tác AI gần nhất.")
                else:
                    reports.append("Không có thao tác AI nào để hoàn tác.")
            elif act.tool == "write_on_board":
                content = str(act.args.get("content") or "").strip()
                if not content:
                    continue
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                fmt = str(act.args.get("format") or "latex")
                font_sz = int(act.args.get("fontSize") or 15)
                col_hex = act.args.get("color")
                est_w = min(520.0, max(180.0, len(content) * 11.0 + 36.0))
                pos = act.args.get("position", "auto")
                if isinstance(pos, dict) and "x" in pos and "y" in pos:
                    ax, ay = float(pos["x"]), float(pos["y"])
                else:
                    ax, ay = find_free_board_region(
                        self._occupied_rects(),
                        canvas_sz,
                        needed_size=(est_w, 56.0),
                        preferred_anchor=preferred_anchor,
                    )
                ann = self._make_annotation_entry(
                    content, ax, ay, fmt=fmt, font_size=font_sz, color_hex=str(col_hex) if col_hex else None
                )
                self.annotations.append(ann)
                self._bump_scene()
                reports.append(f"Đã ghi lên bảng tại ({int(ax)}, {int(ay)}): {ann['text']}")
            elif act.tool == "solve_on_board":
                steps = [str(st).strip() for st in (act.args.get("steps") or []) if str(st).strip()]
                if not steps:
                    continue
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                target_reg = act.args.get("targetRegion")
                pref = preferred_anchor
                if isinstance(target_reg, (list, tuple)) and len(target_reg) >= 2:
                    pref = (float(target_reg[0]) + 40.0, float(target_reg[1]))
                block_h = max(64.0, len(steps) * 48.0)
                bx, by = find_free_board_region(
                    self._occupied_rects(),
                    canvas_sz,
                    needed_size=(360.0, min(canvas_sz[1] - 80.0, block_h)),
                    preferred_anchor=pref,
                )
                cur_y = by
                for st in steps:
                    fmt = "latex" if any(c in st for c in ("=", "\\", "^", "_")) else "text"
                    ann = self._make_annotation_entry(st, bx, cur_y, fmt=fmt, font_size=14)
                    self.annotations.append(ann)
                    cur_y += float(ann["h"]) + 6.0
                self._bump_scene()
                reports.append(f"Đã ghi {len(steps)} bước lời giải gọn gàng lên bảng.")
            elif act.tool == "draw_text":
                text_val = str(act.args.get("text") or act.args.get("content") or "").strip()
                if not text_val:
                    continue
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                ax = (float(act.args.get("x", 120.0)) / 1000.0 * cw) if is_norm else float(act.args.get("x", 120.0))
                ay = (float(act.args.get("y", 160.0)) / 1000.0 * ch) if is_norm else float(act.args.get("y", 160.0))
                font_sz = int(max(11, min(36, float(act.args.get("fontSize") or 16))))
                col_hex = str(act.args.get("color") or "#FEE440")
                ann = self._make_annotation_entry(text_val, ax, ay, fmt="latex", font_size=font_sz, color_hex=col_hex)
                self.annotations.append(ann)
                self._bump_scene()
                reports.append(f"Đã viết lên bảng tại ({int(ax)}, {int(ay)}): {ann['text']}")
            elif act.tool == "draw_polyline":
                raw_pts = act.args.get("points") or []
                if len(raw_pts) < 2:
                    continue
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                px_pts = [
                    ((float(p[0]) / 1000.0 * cw) if is_norm else float(p[0]),
                     (float(p[1]) / 1000.0 * ch) if is_norm else float(p[1]))
                    for p in raw_pts
                ]
                if act.args.get("smooth", True) and len(px_pts) >= 3:
                    px_pts = catmull_rom_spline(px_pts, samples_per_segment=4)
                ink = QColor(str(act.args.get("color") or "#00F5D4"))
                thick = float(max(1.5, min(14.0, float(act.args.get("strokeWidth") or 3.2))))
                qpts = [QPointF(px, py) for (px, py) in px_pts]
                stroke_entry = {
                    "raw_points": list(qpts),
                    "points": qpts[:2],
                    "shape_type": "POLYLINE",
                    "shape_label_vi": "AI Đồ thị",
                    "confidence": 1.0,
                    "is_beautified": False,
                    "color": ink,
                    "thickness": thick,
                }
                self.strokes.append(stroke_entry)
                self._animate_stroke_reveal(stroke_entry, qpts)
                self._bump_scene()
                reports.append(f"Đã vẽ đường cong/đồ thị ({len(qpts)} điểm).")
            elif act.tool == "highlight_region":
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                rx = (float(act.args.get("x", 100.0)) / 1000.0 * cw) if is_norm else float(act.args.get("x", 100.0))
                ry = (float(act.args.get("y", 100.0)) / 1000.0 * ch) if is_norm else float(act.args.get("y", 100.0))
                rw = (float(act.args.get("w", 220.0)) / 1000.0 * cw) if is_norm else float(act.args.get("w", 220.0))
                rh = (float(act.args.get("h", 140.0)) / 1000.0 * ch) if is_norm else float(act.args.get("h", 140.0))
                ink = QColor(str(act.args.get("color") or "#FEE440"))
                qpts = [QPointF(rx, ry), QPointF(rx + rw, ry), QPointF(rx + rw, ry + rh), QPointF(rx, ry + rh), QPointF(rx, ry)]
                self.strokes.append({
                    "raw_points": qpts,
                    "points": qpts,
                    "shape_type": "HIGHLIGHT",
                    "shape_label_vi": "AI Highlight",
                    "confidence": 1.0,
                    "is_beautified": False,
                    "color": ink,
                    "thickness": 2.5,
                })
                lbl = str(act.args.get("label") or "").strip()
                if lbl:
                    self.annotations.append(
                        self._make_annotation_entry(lbl, rx + 6.0, max(24.0, ry - 26.0), fmt="text", font_size=13, color_hex=ink.name())
                    )
                self._bump_scene()
                reports.append(f"Đã khoanh sáng vùng ({int(rx)}, {int(ry)}, {int(rw)}x{int(rh)}).")
            elif act.tool == "clear_region":
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                rx = (float(act.args.get("x", 0.0)) / 1000.0 * cw) if is_norm else float(act.args.get("x", 0.0))
                ry = (float(act.args.get("y", 0.0)) / 1000.0 * ch) if is_norm else float(act.args.get("y", 0.0))
                rw = (float(act.args.get("w", 200.0)) / 1000.0 * cw) if is_norm else float(act.args.get("w", 200.0))
                rh = (float(act.args.get("h", 200.0)) / 1000.0 * ch) if is_norm else float(act.args.get("h", 200.0))
                self.strokes = [
                    st for st in self.strokes
                    if not any(rx <= p.x() <= rx + rw and ry <= p.y() <= ry + rh for p in st.get("points", []))
                ]
                self.annotations = [
                    ann for ann in self.annotations
                    if not (rx <= float(ann.get("x", 0.0)) <= rx + rw and ry <= float(ann.get("y", 0.0)) <= ry + rh)
                ]
                self._bump_scene()
                reports.append("Đã xóa vùng chỉ định trên bảng.")
            elif act.tool == "draw_shape":
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                st_type = str(act.args.get("type") or "arrow")
                scaled_args = dict(act.args)
                if is_norm:
                    ox = float(act.args.get("x", 160.0)) / 1000.0 * cw
                    oy = float(act.args.get("y", 180.0)) / 1000.0 * ch
                    bw = max(36.0, float(act.args.get("w", 200.0)) / 1000.0 * cw)
                    bh = max(36.0, float(act.args.get("h", 150.0)) / 1000.0 * ch)
                    scaled_args["x"] = ox
                    scaled_args["y"] = oy
                    scaled_args["w"] = bw
                    scaled_args["h"] = bh
                    if "radius" in act.args:
                        scaled_args["radius"] = max(20.0, float(act.args["radius"]) / 1000.0 * min(cw, ch))
                    if "x2" in act.args and "y2" in act.args:
                        scaled_args["x2"] = float(act.args["x2"]) / 1000.0 * cw
                        scaled_args["y2"] = float(act.args["y2"]) / 1000.0 * ch
                    if isinstance(act.args.get("points"), list):
                        scaled_args["points"] = [
                            [float(p[0]) / 1000.0 * cw, float(p[1]) / 1000.0 * ch]
                            for p in act.args["points"]
                            if isinstance(p, (list, tuple)) and len(p) >= 2
                        ]
                else:
                    bw = float(act.args.get("w") or 180.0)
                    bh = float(act.args.get("h") or 140.0)
                    ox, oy = find_free_board_region(
                        self._occupied_rects(),
                        canvas_sz,
                        needed_size=(bw, bh),
                        preferred_anchor=preferred_anchor,
                    )
                polylines = build_vector_shape_strokes(st_type, scaled_args, (ox, oy))
                ink = QColor(str(act.args.get("color") or (SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO)))
                thick = float(max(1.5, min(14.0, float(act.args.get("strokeWidth") or 3.5))))
                for poly in polylines:
                    qpts = [QPointF(px, py) for (px, py) in poly]
                    stroke_entry = {
                        "raw_points": list(qpts),
                        "points": qpts[:2] if len(qpts) > 3 else qpts,
                        "shape_type": st_type.upper(),
                        "shape_label_vi": f"AI {st_type}",
                        "confidence": 1.0,
                        "is_beautified": False,
                        "color": ink,
                        "thickness": thick,
                    }
                    self.strokes.append(stroke_entry)
                    if len(qpts) > 3:
                        self._animate_stroke_reveal(stroke_entry, qpts)
                lbl = str(act.args.get("label") or "").strip()
                if lbl:
                    self.annotations.append(
                        self._make_annotation_entry(lbl, ox + 6.0, max(24.0, oy - 24.0), fmt="text", font_size=14, color_hex=ink.name())
                    )
                self._bump_scene()
                reports.append(f"Đã vẽ vector `{st_type}` lên bảng ({int(ox)}, {int(oy)}).")

        self.update()
        return "\n".join(reports)

    def _animate_stroke_reveal(self, stroke_entry: Dict[str, Any], full_pts: List[QPointF]) -> None:
        """Progressively reveal AI-drawn vector strokes at ~60 FPS so students can follow."""
        total = len(full_pts)
        if total <= 3:
            stroke_entry["points"] = full_pts
            return
        step = max(2, total // 14)

        def _tick(idx: int = 2) -> None:
            if idx >= total:
                stroke_entry["points"] = full_pts
                self._bump_scene()
                self.update()
                return
            stroke_entry["points"] = full_pts[:idx]
            self._bump_scene()
            self.update()
            QTimer.singleShot(16, lambda: _tick(idx + step))

        QTimer.singleShot(16, lambda: _tick(min(total, 2 + step)))

    def undo_last_ai_action(self) -> bool:
        if not self.ai_action_undo_stack:
            return False
        snap = self.ai_action_undo_stack.pop()
        self.strokes = list(snap["strokes"])
        self.annotations = list(snap["annotations"])
        self._bump_scene()
        self.update()
        return True

    def replace_lasso_with_formula(self, region: LassoSelectionRegion, formula: str) -> None:
        clean_formula = extract_clean_beautified_formula(formula)
        if not clean_formula:
            return
        self.ai_action_undo_stack.append({
            "strokes": list(self.strokes),
            "annotations": list(self.annotations),
        })
        kept_strokes = []
        for s in self.strokes:
            pts = [(float(p.x()), float(p.y())) for p in (s.get("points") or [])]
            if not stroke_inside_lasso(pts, region):
                kept_strokes.append(s)
        self.strokes = kept_strokes
        ann = self._make_annotation_entry(
            clean_formula,
            region.bbox_x + 8.0,
            region.bbox_y + 8.0,
            fmt="latex",
            font_size=17,
        )
        self.annotations.append(ann)
        self.hide_lasso_bar()
        self._bump_scene()
        self.update()

    def toggle_slide_visibility(self) -> None:
        self.slide_visible = not self.slide_visible
        if self.slide_visible:
            self.toggle_slide_btn.setText("[Ẩn nội dung Slide để lấy bảng trống 100%]")
        else:
            self.toggle_slide_btn.setText("[Hiện nội dung Slide bài giảng]")
        self._bump_scene()
        self.update()

    def dismiss_tutorial_slides(self) -> None:
        self.slides = [s for s in self.slides if not s.get("is_tutorial", False)]
        self.current_slide_idx = min(self.current_slide_idx, max(0, len(self.slides) - 1))
        self.clear_tutorial_btn.setVisible(any(s.get("is_tutorial") for s in self.slides))
        self.slides_changed.emit()
        self._bump_scene()
        self.update()

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0

    def apply_cv_fsm_packet(self, pkt: dict) -> DrawingFsmUpdate:
        now_ms = self._now_ms()
        gesture = str(pkt.get("gesture", "IDLE"))
        pose_tag = str(pkt.get("pose_tag") or gesture or "IDLE")
        pinch_ratio = pkt.get("pinch_ratio")
        cursor_norm = pkt.get("cursor_norm")
        left_pan = bool(pkt.get("left_pan_active", False))
        hand_present = cursor_norm is not None and gesture != "IDLE"

        cursor_px: Optional[Tuple[float, float]] = None
        if cursor_norm is not None:
            px = float(cursor_norm[0]) * max(1, self.width())
            py = float(cursor_norm[1]) * max(1, self.height())
            cursor_px = (px, py)
            self.cursor_pos = QPointF(px, py)

        # Track hold-to-snap stillness if active
        if self.hud_mode == DrawingHudMode.DRAWING and cursor_px is not None:
            qpt = QPointF(cursor_px[0], cursor_px[1])
            if (
                self._last_motion_pt is None
                or math.hypot(qpt.x() - self._last_motion_pt.x(), qpt.y() - self._last_motion_pt.y()) > 5.0
            ):
                self._last_motion_pt = qpt
                self._last_motion_ms = now_ms
            self._hold_snap_ready = (not self.hold_to_snap_350ms) or ((now_ms - self._last_motion_ms) >= 350.0)

        upd = self.drawing_fsm.step(
            now_ms,
            hand_present=hand_present,
            cursor_xy=cursor_px,
            pinch_ratio=pinch_ratio,
            pose_tag=pose_tag,
            left_pan_active=left_pan,
        )
        self._apply_fsm_update(upd, now_ms)
        return upd

    def _apply_fsm_update(self, upd: DrawingFsmUpdate, now_ms: float) -> None:
        self.hud_mode = upd.hud_mode
        self.cooldown_remaining_ms = upd.cooldown_remaining_ms
        self.cancel_hold_progress = upd.cancel_hold_progress
        self.cursor_mode = upd.hud_mode.value

        if upd.action in ("BEGIN_STROKE", "APPEND_POINT", "HOLD_GRACE"):
            if upd.action == "BEGIN_STROKE":
                self.redo_strokes.clear()
                self._stroke_start_ms = now_ms
                self._last_motion_ms = now_ms
                self._last_motion_pt = QPointF(upd.cursor_xy[0], upd.cursor_xy[1]) if upd.cursor_xy else None
                self._hold_snap_ready = not self.hold_to_snap_350ms
            elif upd.action == "HOLD_GRACE":
                self.cursor_mode = "DRAWING"
            if upd.active_points:
                self.current_stroke = [QPointF(x, y) for (x, y) in upd.active_points]
        elif upd.action == "COMMIT_STROKE":
            self.current_stroke = [QPointF(x, y) for (x, y) in upd.committed_points]
            self._finalize_current_stroke(now_ms=now_ms)
        elif upd.action == "CANCEL_STROKE":
            self.current_stroke = []
        elif upd.action == "ROLLBACK_LAST_STROKE":
            self.current_stroke = []
            if self.strokes:
                self.strokes.pop()
                self.shape_toast = None
                self.toast_revert_rect = None
                self._bump_scene()
        elif upd.action == "UNDO":
            self.undo()
        elif upd.action in ("ERASE", "PALM_ERASE") and upd.cursor_xy is not None:
            self._erase_at(QPointF(upd.cursor_xy[0], upd.cursor_xy[1]), 45.0)

        self.hud_status_changed.emit(
            upd.hud_mode.value, float(upd.cooldown_remaining_ms), upd.cancel_reason
        )
        self.update()

    def trigger_esc_cancel(self) -> None:
        now_ms = self._now_ms()
        if self.current_stroke:
            # Mouse or CV active stroke cancel + 300ms lock
            self.current_stroke = []
        upd = self.drawing_fsm.step(
            now_ms,
            hand_present=self.cursor_pos is not None,
            cursor_xy=(self.cursor_pos.x(), self.cursor_pos.y()) if self.cursor_pos else None,
            pinch_ratio=None,
            pose_tag="IDLE",
            esc_pressed=True,
        )
        self._apply_fsm_update(upd, now_ms)

    def request_undo(self) -> bool:
        now_ms = self._now_ms()
        upd = self.drawing_fsm.step(
            now_ms,
            hand_present=self.cursor_pos is not None,
            cursor_xy=(self.cursor_pos.x(), self.cursor_pos.y()) if self.cursor_pos else None,
            pinch_ratio=None,
            pose_tag="IDLE",
            undo_requested=True,
        )
        if upd.action == "UNDO":
            self.hud_status_changed.emit(upd.hud_mode.value, float(upd.cooldown_remaining_ms), "UNDO")
            return True
        return False

    def revert_last_beautified_stroke(self) -> bool:
        if not self.strokes:
            return False
        target_idx = (
            self.shape_toast["stroke_idx"]
            if self.shape_toast and 0 <= self.shape_toast["stroke_idx"] < len(self.strokes)
            else len(self.strokes) - 1
        )
        stroke = self.strokes[target_idx]
        raw_pts = stroke.get("raw_points")
        if raw_pts and len(raw_pts) >= 2:
            stroke["points"] = [QPointF(p) for p in raw_pts]
            stroke["is_beautified"] = False
            stroke["shape_type"] = "FREEHAND"
            stroke["shape_label_vi"] = "Nét gốc mượt"
        self.shape_toast = None
        self.toast_revert_rect = None
        self._bump_scene()
        self.update()
        return True

    def mousePressEvent(self, event) -> None:
        pt = QPointF(event.position())
        self.cursor_pos = pt

        # Check clickable "[Hoàn tác về nét gốc]" toast pill
        if (
            event.button() == Qt.LeftButton
            and self.toast_revert_rect is not None
            and self.shape_toast is not None
            and self.toast_revert_rect.contains(pt)
        ):
            self.revert_last_beautified_stroke()
            return

        if self.drawing_fsm.hud_mode == DrawingHudMode.LOCKED_COOLDOWN:
            return

        if event.button() == Qt.RightButton:
            self.cursor_mode = "PAN_ERASE"
            self._erase_at(pt, 45.0)
            self.update()
            return

        if event.button() == Qt.LeftButton:
            if self.active_tool in ("SELECT_REGION", "LASSO"):
                self.lasso_action_bar.setVisible(False)
                self.lasso_region = None
                if self.active_tool == "LASSO":
                    self.lasso_points = [pt]
                    self.selection_start = None
                    self.selection_end = None
                else:
                    self.selection_start = pt
                    self.selection_end = pt
                    self.lasso_points = []
            elif self.active_tool == "ERASER":
                self.cursor_mode = "PAN_ERASE"
                self._erase_at(pt, 45.0)
            elif self.active_tool == "LASER":
                self.cursor_mode = "LASER"
            else:
                self.cursor_mode = "DRAWING"
                self.redo_strokes.clear()
                now_ms = self._now_ms()
                self._stroke_start_ms = now_ms
                self._last_motion_ms = now_ms
                self._last_motion_pt = pt
                self._hold_snap_ready = not self.hold_to_snap_350ms
                self.current_stroke = [pt]
            self.update()

    def mouseMoveEvent(self, event) -> None:
        pt = QPointF(event.position())
        self.cursor_pos = pt
        if self.drawing_fsm.hud_mode == DrawingHudMode.LOCKED_COOLDOWN:
            self.update()
            return

        if event.buttons() & Qt.RightButton:
            self.cursor_mode = "PAN_ERASE"
            self._erase_at(pt, 45.0)
            self.update()
            return

        if event.buttons() & Qt.LeftButton:
            if self.active_tool == "SELECT_REGION":
                self.selection_end = pt
            elif self.active_tool == "LASSO":
                if not self.lasso_points or math.hypot(pt.x() - self.lasso_points[-1].x(), pt.y() - self.lasso_points[-1].y()) >= 4.0:
                    self.lasso_points.append(pt)
            elif self.active_tool == "ERASER":
                self.cursor_mode = "PAN_ERASE"
                self._erase_at(pt, 45.0)
            elif self.active_tool == "PEN":
                self.cursor_mode = "DRAWING"
                now_ms = self._now_ms()
                if (
                    self._last_motion_pt is None
                    or math.hypot(pt.x() - self._last_motion_pt.x(), pt.y() - self._last_motion_pt.y()) > 5.0
                ):
                    self._last_motion_pt = pt
                    self._last_motion_ms = now_ms
                self._hold_snap_ready = (not self.hold_to_snap_350ms) or ((now_ms - self._last_motion_ms) >= 350.0)
                if self.current_stroke:
                    last_qpt = self.current_stroke[-1]
                    seg_dist = math.hypot(pt.x() - last_qpt.x(), pt.y() - last_qpt.y())
                    if seg_dist > 10.0:
                        p_before = (
                            (self.current_stroke[-2].x(), self.current_stroke[-2].y())
                            if len(self.current_stroke) >= 2
                            else None
                        )
                        pts = interpolate_bridge_points(
                            p_before,
                            (last_qpt.x(), last_qpt.y()),
                            (pt.x(), pt.y()),
                            max_step_px=12.0,
                        )
                        for ix, iy in pts:
                            self.current_stroke.append(QPointF(ix, iy))
                    else:
                        self.current_stroke.append(pt)
                else:
                    self.current_stroke.append(pt)
        else:
            self.cursor_mode = "LASER" if self.active_tool == "LASER" else "HOVER"
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self.active_tool == "LASSO" and len(self.lasso_points) >= 3:
                poly = [(float(p.x()), float(p.y())) for p in self.lasso_points]
                region = compute_lasso_bounding_box(poly, (self.width(), self.height()))
                if region is not None:
                    self._crop_and_emit_selection(
                        QPointF(region.bbox_x, region.bbox_y),
                        QPointF(region.bbox_x + region.bbox_w, region.bbox_y + region.bbox_h),
                        open_bar_only=True,
                        region=region,
                    )
                else:
                    self.lasso_points.clear()
            elif self.active_tool == "SELECT_REGION" and self.selection_start and self.selection_end:
                rect = QRectF(self.selection_start, self.selection_end).normalized()
                if rect.width() >= 18 and rect.height() >= 18:
                    pad_rect = rect.adjusted(-18.0, -18.0, 18.0, 18.0).intersected(QRectF(0, 0, self.width(), self.height()))
                    poly = [
                        (pad_rect.left(), pad_rect.top()),
                        (pad_rect.right(), pad_rect.top()),
                        (pad_rect.right(), pad_rect.bottom()),
                        (pad_rect.left(), pad_rect.bottom()),
                    ]
                    region = LassoSelectionRegion(
                        polygon=poly,
                        bbox_x=pad_rect.left(),
                        bbox_y=pad_rect.top(),
                        bbox_w=pad_rect.width(),
                        bbox_h=pad_rect.height(),
                    )
                    self._crop_and_emit_selection(
                        pad_rect.topLeft(),
                        pad_rect.bottomRight(),
                        open_bar_only=True,
                        region=region,
                    )
                self.selection_start = None
                self.selection_end = None
            elif self.current_stroke:
                now_ms = self._now_ms()
                if self.hold_to_snap_350ms:
                    self._hold_snap_ready = (now_ms - self._last_motion_ms) >= 350.0
                self._finalize_current_stroke(now_ms=now_ms)
        self.cursor_mode = (
            "LOCKED_COOLDOWN"
            if self.drawing_fsm.hud_mode == DrawingHudMode.LOCKED_COOLDOWN
            else "HOVER"
        )
        self.update()

    def _finalize_current_stroke(self, now_ms: Optional[float] = None) -> None:
        if len(self.current_stroke) >= 2:
            ts_ms = now_ms if now_ms is not None else self._now_ms()
            raw_xy = [(p.x(), p.y()) for p in self.current_stroke]
            smooth_raw_xy = catmull_rom_spline(raw_xy, subdivisions=4)
            raw_qpts = [QPointF(x, y) for (x, y) in (smooth_raw_xy if len(smooth_raw_xy) >= 2 else raw_xy)]

            should_beautify = self.smart_shape_enabled and (
                not self.hold_to_snap_350ms or self._hold_snap_ready
            )
            if should_beautify and len(raw_xy) >= 3:
                shape_res = beautify_stroke(raw_xy, closed_gap_ratio=0.18, confidence_gate=0.78)
                final_qpts = [QPointF(x, y) for (x, y) in shape_res.beautified_points]
                is_geo = shape_res.is_geometric
                shape_type = shape_res.shape_type
                shape_label_vi = shape_res.shape_label_vi
                confidence = shape_res.confidence
            else:
                final_qpts = raw_qpts
                is_geo = False
                shape_type = "FREEHAND"
                shape_label_vi = "Nét mượt tự do"
                confidence = 0.0

            self.strokes.append({
                "raw_points": raw_qpts,
                "points": final_qpts,
                "shape_type": shape_type,
                "shape_label_vi": shape_label_vi,
                "confidence": confidence,
                "is_beautified": is_geo,
                "color": QColor(self.ink_color),
                "thickness": self.pen_thickness,
            })
            self._bump_scene()

            if is_geo and final_qpts:
                anchor_x = sum(p.x() for p in final_qpts) / len(final_qpts)
                anchor_y = min(p.y() for p in final_qpts) - 18.0
                self.shape_toast = {
                    "stroke_idx": len(self.strokes) - 1,
                    "label": shape_label_vi,
                    "conf": confidence,
                    "anchor": QPointF(anchor_x, anchor_y),
                    "until_ms": ts_ms + 4500.0,
                }
        self.current_stroke = []

    def _erase_at(self, pt: QPointF, radius: float) -> None:
        kept = []
        removed = False
        for s in self.strokes:
            hit = any(
                math.hypot(p.x() - pt.x(), p.y() - pt.y()) < radius
                for p in s["points"]
            )
            if not hit:
                kept.append(s)
            else:
                removed = True
        self.strokes = kept

        kept_ann = []
        for ann in self.annotations:
            cx = float(ann.get("x", 0.0)) + float(ann.get("w", 120.0)) * 0.5
            cy = float(ann.get("y", 0.0)) + float(ann.get("h", 40.0)) * 0.5
            if math.hypot(cx - pt.x(), cy - pt.y()) < radius + 30.0:
                removed = True
            else:
                kept_ann.append(ann)
        self.annotations = kept_ann
        if removed:
            self._bump_scene()

    def undo(self) -> None:
        if not self.strokes:
            self.undo_last_ai_action()
            return
        top = self.strokes[-1]
        raw_pts = top.get("raw_points")
        if top.get("is_beautified") and raw_pts and len(raw_pts) >= 2:
            # Ctrl+Z Step 1: Revert snapped geometric shape back to smooth raw freehand stroke
            snapped_snapshot = dict(top)
            snapped_snapshot["points"] = [QPointF(p) for p in top.get("points", [])]
            snapped_snapshot["raw_points"] = [QPointF(p) for p in raw_pts]
            snapped_snapshot["_replace_last"] = True
            self.redo_strokes.append(snapped_snapshot)

            top["points"] = [QPointF(p) for p in raw_pts]
            top["is_beautified"] = False
            top["shape_type"] = "FREEHAND"
            top["shape_label_vi"] = "Nét gốc mượt (Ctrl+Z)"
            top["confidence"] = 0.0
        else:
            # Ctrl+Z Step 2 (or normal freehand stroke): Delete stroke
            popped = self.strokes.pop()
            popped.pop("_replace_last", None)
            self.redo_strokes.append(popped)
        self.shape_toast = None
        self.toast_revert_rect = None
        self._bump_scene()
        self.update()

    def redo(self) -> None:
        if not self.redo_strokes:
            return
        item = self.redo_strokes.pop()
        if item.pop("_replace_last", False) and self.strokes:
            self.strokes[-1] = item
        else:
            self.strokes.append(item)
        self._bump_scene()
        self.update()

    def clear_board(self) -> None:
        self.strokes.clear()
        self.redo_strokes.clear()
        self.annotations.clear()
        self.current_stroke.clear()
        self.shape_toast = None
        self.toast_revert_rect = None
        self.hide_lasso_bar()
        self._bump_scene()
        self.update()

    def export_board_png(self, rect: Optional[QRectF] = None, enhance_crop: bool = False) -> Tuple[bytes, QPixmap]:
        full_pm = self.grab()
        if rect is not None and rect.width() > 16 and rect.height() > 16:
            cropped_pm = full_pm.copy(rect.toRect())
        else:
            cropped_pm = full_pm
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        cropped_pm.save(buf, "PNG")
        raw_bytes = bytes(ba)
        if enhance_crop:
            compressed_bytes = enhance_lasso_crop_png(raw_bytes, max_dim=1024) or raw_bytes
        else:
            compressed_bytes = compress_snapshot_png(raw_bytes, max_dim=1024) or raw_bytes
        return compressed_bytes, cropped_pm

    def _crop_and_emit_selection(
        self,
        p0: QPointF,
        p1: QPointF,
        open_bar_only: bool = False,
        region: Optional[LassoSelectionRegion] = None,
    ) -> None:
        rect = QRectF(p0, p1).normalized()
        if rect.width() < 15 or rect.height() < 15:
            return
        png_bytes, cropped_pm = self.export_board_png(rect, enhance_crop=True)
        if open_bar_only and region is not None:
            self._show_lasso_action_bar(region)
            self.region_cropped_for_ai.emit(png_bytes, cropped_pm)
        else:
            self.region_cropped_for_ai.emit(png_bytes, cropped_pm)

    def _paint_static_scene(self, painter: QPainter) -> bool:
        """Render background, grid, slide, committed strokes & math annotations cleanly."""
        has_active_reveal = False
        bg_col = QColor(BG_DARK if self.dark_mode else "#F8FAFC")
        painter.fillRect(self.rect(), bg_col)

        # 1. Single Caro Grid
        if self.grid_enabled:
            step = 24
            minor_pen = QPen(QColor(79, 70, 229, 32 if self.dark_mode else 26), 1.0)
            major_pen = QPen(QColor(6, 182, 212, 62 if self.dark_mode else 46), 1.0)
            for idx, x in enumerate(range(0, self.width(), step)):
                painter.setPen(major_pen if idx % 5 == 0 else minor_pen)
                painter.drawLine(x, 0, x, self.height())
            for idx, y in enumerate(range(0, self.height(), step)):
                painter.setPen(major_pen if idx % 5 == 0 else minor_pen)
                painter.drawLine(0, y, self.width(), y)

        # 2. Optional Slide Content Card
        if self.slide_visible and self.slides:
            slide = self.slides[self.current_slide_idx]
            s_w = min(740, max(320, self.width() - 40))
            s_h = int(s_w * 9 / 16)
            s_rect = QRectF(16, 46, s_w, s_h)
            card_bg = QColor(15, 23, 42, 200) if self.dark_mode else QColor(255, 255, 255, 235)
            card_fg = QColor(TEXT_LIGHT if self.dark_mode else "#0F172A")
            painter.setBrush(QBrush(card_bg))
            painter.setPen(QPen(QColor(BORDER_SLATE if self.dark_mode else "#CBD5E1"), 1.2))
            painter.drawRoundedRect(s_rect, 12.0, 12.0)

            pm: Optional[QPixmap] = slide.get("pixmap")
            if pm is not None:
                painter.drawPixmap(s_rect.toRect(), pm)
            else:
                painter.setPen(QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO))
                painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
                painter.drawText(QPointF(s_rect.left() + 22, s_rect.top() + 34), slide["title"])

                painter.setPen(QColor(TEXT_MUTED if self.dark_mode else "#475569"))
                painter.setFont(QFont("Segoe UI", 11))
                painter.drawText(QPointF(s_rect.left() + 22, s_rect.top() + 58), slide["subtitle"])

                painter.setPen(card_fg)
                painter.setFont(QFont("Segoe UI", 11))
                for i, b in enumerate(slide["bullets"]):
                    painter.drawText(
                        QPointF(s_rect.left() + 24, s_rect.top() + 94 + i * 28), OutputSanitizer.sanitize_plain_text(b)
                    )

                f_rect = QRectF(s_rect.left() + 22, s_rect.bottom() - 62, s_rect.width() - 44, 42)
                painter.setBrush(QBrush(QColor(15, 23, 42, 235) if self.dark_mode else QColor("#EEF2FF")))
                painter.setPen(QPen(QColor(PRIMARY_INDIGO), 1.2))
                painter.drawRoundedRect(f_rect, 8.0, 8.0)
                painter.setPen(QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
                painter.drawText(f_rect, Qt.AlignCenter, OutputSanitizer.sanitize_plain_text(slide["formula"]))

        # 3. Committed Vector Ink Strokes
        for s in self.strokes:
            pts = s["points"]
            if len(pts) >= 2:
                pen = QPen(s["color"], s["thickness"], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                path = QPainterPath(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                painter.drawPath(path)

        # 3B. Agentic Whiteboard Rendered Math & Text Annotations
        now_ms = self._now_ms()
        for ann in self.annotations:
            ax = float(ann.get("x", 40.0))
            ay = float(ann.get("y", 60.0))
            aw = float(ann.get("w", 200.0))
            ah = float(ann.get("h", 44.0))
            reveal_until = float(ann.get("reveal_until_ms", 0.0))
            alpha_f = 1.0
            if reveal_until > now_ms:
                has_active_reveal = True
                alpha_f = max(0.15, min(1.0, 1.0 - (reveal_until - now_ms) / 420.0))

            painter.save()
            painter.setOpacity(alpha_f)
            box = QRectF(ax, ay, aw, ah)
            ann_bg = QColor(15, 23, 42, 215) if self.dark_mode else QColor(255, 255, 255, 240)
            ann_border = QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO)
            painter.setBrush(QBrush(ann_bg))
            painter.setPen(QPen(ann_border, 1.2))
            painter.drawRoundedRect(box, 8.0, 8.0)

            ann_pm: Optional[QPixmap] = ann.get("pixmap")
            if ann_pm is not None and not ann_pm.isNull():
                tx = int(ax + (aw - ann_pm.width()) * 0.5)
                ty = int(ay + (ah - ann_pm.height()) * 0.5)
                painter.drawPixmap(tx, ty, ann_pm)
            else:
                painter.setPen(QColor(ann.get("color") or (NEON_CYAN if self.dark_mode else "#0F172A")))
                painter.setFont(QFont("Segoe UI", max(11, int(ann.get("font_size") or 14)), QFont.Bold))
                painter.drawText(box.adjusted(10.0, 4.0, -10.0, -4.0), Qt.AlignVCenter | Qt.AlignLeft, str(ann.get("text") or ""))
            painter.restore()

        return has_active_reveal

    def paintEvent(self, _event) -> None:
        scene_key = (
            self.width(),
            self.height(),
            self.dark_mode,
            self.grid_enabled,
            self.slide_visible,
            self.current_slide_idx,
            self._scene_rev,
        )
        has_active_reveal = any(float(a.get("reveal_until_ms", 0.0)) > self._now_ms() for a in self.annotations)
        if (
            has_active_reveal
            or self._backing_pixmap is None
            or self._dirty_tracker.needs_rebuild(scene_key)
        ):
            pm = QPixmap(max(1, self.width()), max(1, self.height()))
            bg_p = QPainter(pm)
            bg_p.setRenderHint(QPainter.Antialiasing)
            still_anim = self._paint_static_scene(bg_p)
            bg_p.end()
            self._backing_pixmap = pm
            if not still_anim:
                self._dirty_tracker.mark_clean(scene_key)
            else:
                QTimer.singleShot(25, self.update)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._backing_pixmap is not None:
            painter.drawPixmap(0, 0, self._backing_pixmap)

        # Active live stroke (never blocked by backing cache)
        if len(self.current_stroke) >= 2:
            pen = QPen(self.ink_color, self.pen_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            path = QPainterPath(self.current_stroke[0])
            for p in self.current_stroke[1:]:
                path.lineTo(p)
            painter.drawPath(path)

        # Smart Lasso polygon or Region Selection overlay
        if len(self.lasso_points) >= 2:
            lpath = QPainterPath(self.lasso_points[0])
            for p in self.lasso_points[1:]:
                lpath.lineTo(p)
            if self.lasso_region is not None:
                lpath.closeSubpath()
            painter.setBrush(QBrush(QColor(124, 58, 237, 40)))
            painter.setPen(QPen(QColor(NEON_CYAN), 2.0, Qt.DashLine))
            painter.drawPath(lpath)
        elif self.selection_start and self.selection_end:
            sel_rect = QRectF(self.selection_start, self.selection_end).normalized()
            painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
            painter.setPen(QPen(QColor(ALERT_ROSE), 2.0))
            painter.drawRoundedRect(sel_rect, 8.0, 8.0)

        # 5. Floating Auto-Shape Beautify Toast ("Đã chuẩn hóa: ... • [Hoàn tác về nét gốc]")
        self.toast_revert_rect = None
        if self.shape_toast is not None:
            if self._now_ms() <= float(self.shape_toast.get("until_ms", 0.0)):
                label_txt = f"Đã chuẩn hóa: {self.shape_toast['label']} ({int(self.shape_toast['conf'] * 100)}%)"
                btn_txt = "Hoàn tác về nét gốc"
                tx = max(16.0, min(self.width() - 370.0, float(self.shape_toast["anchor"].x()) - 140.0))
                ty = max(46.0, min(self.height() - 56.0, float(self.shape_toast["anchor"].y()) - 34.0))
                toast_rect = QRectF(tx, ty, 356.0, 34.0)
                painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
                painter.setPen(QPen(QColor(NEON_CYAN), 1.4))
                painter.drawRoundedRect(toast_rect, 8.0, 8.0)

                painter.setPen(QColor(TEXT_LIGHT))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(QRectF(tx + 10.0, ty, 200.0, 34.0), Qt.AlignVCenter | Qt.AlignLeft, label_txt)

                rev_rect = QRectF(tx + 212.0, ty + 5.0, 136.0, 24.0)
                painter.setBrush(QBrush(QColor(PRIMARY_INDIGO)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rev_rect, 5.0, 5.0)
                painter.setPen(QColor(TEXT_LIGHT))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(rev_rect, Qt.AlignCenter, btn_txt)
                self.toast_revert_rect = rev_rect
            else:
                self.shape_toast = None

        # 6. Live 4-Mode HUD Cursor Overlay
        if self.cursor_pos is not None:
            cp = self.cursor_pos
            if self.cursor_mode == "LOCKED_COOLDOWN":
                painter.setBrush(QBrush(QColor(244, 63, 94, 55)))
                painter.setPen(QPen(QColor(ALERT_ROSE), 2.4))
                painter.drawEllipse(cp, 16.0, 16.0)
                painter.drawLine(cp + QPointF(-6, -6), cp + QPointF(6, 6))
                painter.drawLine(cp + QPointF(6, -6), cp + QPointF(-6, 6))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(
                    cp + QPointF(20.0, 4.0),
                    f"KHÓA {int(max(0.0, self.cooldown_remaining_ms))}ms",
                )
            elif self.cursor_mode == "PAN_ERASE":
                painter.setBrush(QBrush(QColor(244, 63, 94, 50)))
                painter.setPen(QPen(QColor(ALERT_ROSE), 2.0, Qt.DashLine))
                painter.drawEllipse(cp, 45.0, 45.0)
            elif self.cursor_mode == "DRAWING":
                painter.setBrush(QBrush(QColor(0, 245, 212, 95)))
                painter.setPen(QPen(QColor(NEON_CYAN), 2.2))
                painter.drawEllipse(cp, 8.0, 8.0)
                if self.cancel_hold_progress > 0.05:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(ALERT_ROSE), 3.0))
                    span_deg = int(360.0 * min(1.0, self.cancel_hold_progress) * 16.0)
                    painter.drawArc(QRectF(cp.x() - 18, cp.y() - 18, 36, 36), 90 * 16, -span_deg)
            elif self.cursor_mode == "LASER":
                painter.setBrush(QBrush(QColor(244, 63, 94, 180)))
                painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                painter.drawEllipse(cp, 7.0, 7.0)
            else:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0))
                painter.drawEllipse(cp, 10.0, 10.0)


# =============================================================================
# MODULE 2: MINIMALIST CLEAN STUDIO WINDOW
# =============================================================================
class AboutBrandDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Giới thiệu — AirTeach AI (DNT - Victer)")
        self.setFixedWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(make_brand_logo_widget(64))
        info = QVBoxLayout()
        t = QLabel("AirTeach AI — DNT - Victer")
        t.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 18px; font-weight: 800;")
        s = QLabel("Nền tảng Giảng dạy Không chạm • Thị giác Máy tính & AI")
        s.setStyleSheet(f"color: {SECONDARY_TEAL}; font-size: 12px; font-weight: 600;")
        info.addWidget(t)
        info.addWidget(s)
        top.addLayout(info)
        layout.addLayout(top)

        desc = QLabel(
            "• Nhận diện Tay 3D + ROI 384x384 + One-Euro Smoothing mọi điều kiện sáng.\n"
            "• Face ID 5-điểm Affine 112x112 (ArcFace 512-D) & bảo mật PIN PBKDF2 cục bộ.\n"
            "• Máy trạng thái vẽ 4 tầng (Schmitt <0.28 / >0.40, cắt đuôi 105ms, khóa 300ms).\n"
            "• Chuẩn hóa hình học lai: PCA + FitEllipse + ApproxPolyDP + $1/$P + Catmull-Rom.\n"
            "• Tích hợp Gemini 3.5 Flash có cache cục bộ & hoạt động offline bền vững."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 12px; line-height: 1.45;")
        layout.addWidget(desc)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)


class ApiKeyOnboardingDialog(QDialog):
    """Zero-Friction First-Run Onboarding: Paste Gemini API Key, Test Connection, or Use Offline."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chào mừng đến với AirTeach AI — Cấu hình Gemini API")
        self.setModal(True)
        self.setMinimumWidth(480)
        if APP_LOGO_ICO.exists():
            self.setWindowIcon(QIcon(str(APP_LOGO_ICO)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        hdr = QHBoxLayout()
        hdr.addWidget(make_brand_logo_widget(36))
        title = QLabel("Thiết lập Trợ giảng AI (Gemini API Key)")
        title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {TEXT_LIGHT};")
        hdr.addWidget(title, 1)
        layout.addLayout(hdr)

        desc = QLabel(
            "Dán Gemini API Key để AI có thể giải bài, vẽ hình và viết công thức trực tiếp lên bảng trắng.\n"
            "Key được lưu an toàn tại thư mục dữ liệu người dùng (%APPDATA%\\AirTeachAI\\config.json)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(desc)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Dán Gemini API Key (AIza...) tại đây")
        self.key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.key_input)

        self.status_lbl = QLabel("Trạng thái: Chưa kiểm tra")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("Kiểm tra kết nối (Test API)")
        self.test_btn.clicked.connect(self._on_test)
        self.save_btn = QPushButton("Lưu & Bắt đầu")
        self.save_btn.setStyleSheet(f"background-color: {PRIMARY_INDIGO}; color: white; font-weight: 700;")
        self.save_btn.clicked.connect(self._on_save)
        self.offline_btn = QPushButton("Dùng ngoại tuyến (Bỏ qua)")
        self.offline_btn.clicked.connect(self._on_offline)
        btn_row.addWidget(self.test_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.offline_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self.saved_key: str = ""
        self.skipped_offline: bool = False
        self._health_worker: Optional[GeminiHealthCheckWorker] = None

    def _on_test(self) -> None:
        key = self.key_input.text().strip()
        if not key:
            self.status_lbl.setText("Vui lòng nhập API Key trước khi kiểm tra.")
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-size: 12px;")
            return
        self.test_btn.setEnabled(False)
        self.status_lbl.setText("Đang gọi thử Gemini API...")
        self.status_lbl.setStyleSheet(f"color: {SECONDARY_TEAL}; font-size: 12px;")
        self._health_worker = GeminiHealthCheckWorker(key)
        self._health_worker.completed.connect(self._on_test_done)
        self._health_worker.start()

    def _on_test_done(self, res: Any) -> None:
        self.test_btn.setEnabled(True)
        if getattr(res, "status", None) == ApiConnectionStatus.CONNECTED:
            self.status_lbl.setText(f"✓ Sẵn sàng ({res.model_used}, {res.latency_ms:.0f} ms)")
            self.status_lbl.setStyleSheet(f"color: {NEON_CYAN}; font-size: 12px; font-weight: 600;")
        else:
            self.status_lbl.setText(f"⚠ {getattr(res, 'message_vi', 'Lỗi kết nối')}")
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-size: 12px;")

    def _on_save(self) -> None:
        key = self.key_input.text().strip()
        if not key:
            self._on_offline()
            return
        save_ui_api_settings({"ui_gemini_api_key": key, "onboarding_skipped": False})
        self.saved_key = key
        self.accept()

    def _on_offline(self) -> None:
        cur = load_ui_api_settings()
        cur["onboarding_skipped"] = "true"
        save_ui_api_settings(cur)
        self.skipped_offline = True
        self.reject()


class AirTeachMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirTeach AI — Clean Studio Smartboard (DNT - Victer)")
        if APP_LOGO_ICO.exists():
            self.setWindowIcon(QIcon(str(APP_LOGO_ICO)))
        self.resize(1400, 820)

        self.logged_in_user: Dict[str, Any] = {}
        self.api_key_state = resolve_api_key_layered()
        self.api_key: str = self.api_key_state.api_key
        self.gemini_conn_status: ApiConnectionStatus = ApiConnectionStatus.UNCONFIGURED
        self.perf_pipeline = PerformancePipeline()
        self.local_ai_manager = LocalAIManager()
        self.gemini_worker: Optional[GeminiVisionWorker] = None
        self.hybrid_worker: Optional[HybridAiWorker] = None
        self.discovery_worker: Optional[LocalDiscoveryWorker] = None
        self.health_worker: Optional[GeminiHealthCheckWorker] = None
        self.dark_mode: bool = True
        self.chat_history: List[Dict[str, str]] = []
        self.last_ai_raw_text: str = ""
        self._stream_raw_accum: str = ""
        self._active_lasso_action: Optional[LassoQuickAction] = None
        self._active_lasso_region: Optional[LassoSelectionRegion] = None
        self.is_recording_audio = False
        self.audio_frames: List[np.ndarray] = []
        self.vu_db: float = -60.0

        self.cam_worker = CameraHandWorker(0)
        self.cam_worker.frame_ready.connect(self._on_camera_packet)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Screen 0: Strict Two-Tab Security Gate (Sign-Up -> Sign-In)
        self.auth_gate = AuthGateWidget(self.cam_worker)
        self.auth_gate.authenticated.connect(self._on_user_authenticated)
        self.stack.addWidget(self.auth_gate)

        # Screen 1: Clean Studio Smartboard
        self.studio_page = QWidget()
        self._build_studio_ui(self.studio_page)
        self.stack.addWidget(self.studio_page)

        # Global Shortcuts: Esc -> Cancel + 300ms Lock | Ctrl+Z -> Undo | L -> Smart Lasso | Ctrl+Shift+D -> Vision Debug HUD
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self._on_hotkey_esc)
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_shortcut.activated.connect(self._on_hotkey_undo)
        self.lasso_shortcut = QShortcut(QKeySequence(Qt.Key_L), self)
        self.lasso_shortcut.activated.connect(lambda: self._set_tool("LASSO"))
        self.debug_hud_shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        self.debug_hud_shortcut.activated.connect(self._on_toggle_vision_debug_hud)

        self.stack.setCurrentIndex(0)
        self.cam_worker.start()
        QTimer.singleShot(450, self._on_discover_local_ai)
        QTimer.singleShot(600, self._maybe_show_first_run_onboarding)

        # Optional Real Microphone Stream for VU Meter
        self.audio_stream = None
        if HAS_SOUNDDEVICE:
            try:
                self.audio_stream = sd.InputStream(
                    samplerate=44100, channels=1, callback=self._audio_callback
                )
                self.audio_stream.start()
            except Exception:
                self.audio_stream = None

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_vu_meter)
        self.ui_timer.start(80)

    def _audio_callback(self, indata, _frames, _time_info, _status) -> None:
        rms = float(np.sqrt(np.mean(np.square(indata))))
        self.vu_db = 20.0 * math.log10(max(rms, 1e-5))
        if self.is_recording_audio:
            self.audio_frames.append(indata.copy())

    def _update_vu_meter(self) -> None:
        norm = int(np.clip((self.vu_db + 55.0) / 55.0 * 100.0, 2.0, 100.0))
        self.vu_bar.setValue(norm)
        self.vu_lbl.setText(f"{self.vu_db:.1f} dB")

    def _build_studio_ui(self, parent: QWidget) -> None:
        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # =====================================================================
        # MODULE 2.1: SINGLE COMPACT HEADER BAR (Height: 46px, Brand Logo + Controls)
        # =====================================================================
        header = QFrame()
        header.setFixedHeight(46)
        header.setStyleSheet(
            f"background: {BG_DARK}; border-bottom: 1px solid {BORDER_SLATE};"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 6, 10, 6)
        h_layout.setSpacing(6)

        h_layout.addWidget(make_brand_logo_widget(28))
        brand_lbl = QLabel("AirTeach AI")
        brand_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 800; font-size: 13px;")
        h_layout.addWidget(brand_lbl)

        about_btn = QPushButton("[Giới thiệu]")
        about_btn.clicked.connect(self._on_show_about)
        h_layout.addWidget(about_btn)

        logout_btn = QPushButton("[Đăng xuất]")
        logout_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {ALERT_ROSE}; border: 1px solid {ALERT_ROSE}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
        )
        logout_btn.clicked.connect(self._on_logout)
        h_layout.addWidget(logout_btn)

        self.teacher_name_lbl = QLabel("GV: Giáo viên")
        self.teacher_name_lbl.setStyleSheet(
            f"color: {SECONDARY_TEAL}; font-weight: 700; font-size: 13px;"
        )
        h_layout.addWidget(self.teacher_name_lbl)

        open_slide_btn = QPushButton("[Mở Ảnh/Slide]")
        open_slide_btn.clicked.connect(self._on_open_slide_file)
        h_layout.addWidget(open_slide_btn)

        prev_btn = QPushButton("<")
        prev_btn.setFixedWidth(28)
        prev_btn.clicked.connect(self._on_prev_slide)
        h_layout.addWidget(prev_btn)

        self.slide_counter_lbl = QLabel("Slide 1/1")
        self.slide_counter_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 700;")
        h_layout.addWidget(self.slide_counter_lbl)

        next_btn = QPushButton(">")
        next_btn.setFixedWidth(28)
        next_btn.clicked.connect(self._on_next_slide)
        h_layout.addWidget(next_btn)

        grid_btn = QPushButton("[Lưới Caro]")
        grid_btn.clicked.connect(self._on_toggle_grid)
        h_layout.addWidget(grid_btn)

        self.theme_btn = QPushButton("[Sáng/Tối]")
        self.theme_btn.clicked.connect(self._on_toggle_theme)
        h_layout.addWidget(self.theme_btn)

        self.status_pill_lbl = QLabel(self.perf_pipeline.telemetry.status_pill_text())
        self.status_pill_lbl.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {NEON_CYAN}; border: 1px solid {SECONDARY_TEAL}; border-radius: 10px; padding: 3px 9px; font-size: 11px; font-weight: 700;"
        )
        h_layout.addWidget(self.status_pill_lbl)

        h_layout.addStretch(1)

        self.rec_btn = QPushButton("[Ghi Âm]")
        self.rec_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
        )
        self.rec_btn.clicked.connect(self._on_toggle_recording)
        h_layout.addWidget(self.rec_btn)

        self.toggle_cam_btn = QPushButton("[Tắt/Mở Camera]")
        self.toggle_cam_btn.clicked.connect(self._on_toggle_studio_camera)
        h_layout.addWidget(self.toggle_cam_btn)

        toggle_ai_btn = QPushButton("[Trợ Lý AI]")
        toggle_ai_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
        )
        toggle_ai_btn.clicked.connect(
            lambda: self.right_dock.setVisible(not self.right_dock.isVisible())
        )
        h_layout.addWidget(toggle_ai_btn)

        root.addWidget(header)

        # =====================================================================
        # CENTER WORKSPACE: LEFT DOCK (270px) | SMARTBOARD CANVAS | RIGHT DOCK (330px)
        # =====================================================================
        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(0)

        self.left_dock = QFrame()
        self.left_dock.setFixedWidth(270)
        self.left_dock.setStyleSheet(
            f"background: {SURFACE_DARK}; border-right: 1px solid {BORDER_SLATE};"
        )
        l_layout = QVBoxLayout(self.left_dock)
        l_layout.setContentsMargins(10, 10, 10, 10)
        l_layout.setSpacing(7)

        self.pip_cam_lbl = QLabel()
        self.pip_cam_lbl.setFixedSize(250, 188)
        self.pip_cam_lbl.setAlignment(Qt.AlignCenter)
        self.pip_cam_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 8px; color: {TEXT_MUTED};"
        )
        l_layout.addWidget(self.pip_cam_lbl)

        self.hand_status_lbl = QLabel("Trạng thái: Hover (Di chuyển)")
        self.hand_status_lbl.setStyleSheet(
            f"color: {NEON_CYAN}; font-weight: 700; font-size: 12px;"
        )
        l_layout.addWidget(self.hand_status_lbl)

        self.telemetry_lbl = QLabel("Ánh sáng: -- | Độ nét: -- | FPS: --")
        self.telemetry_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        l_layout.addWidget(self.telemetry_lbl)

        cam_ctrl_row = QHBoxLayout()
        for c_idx in (0, 1):
            b = QPushButton(f"Cam {c_idx}")
            b.setFixedHeight(26)
            b.clicked.connect(lambda _=False, i=c_idx: self.cam_worker.select_camera(i))
            cam_ctrl_row.addWidget(b)

        self.hdr_chk = QCheckBox("Auto-HDR")
        self.hdr_chk.setChecked(True)
        self.hdr_chk.toggled.connect(self._on_toggle_hdr)
        cam_ctrl_row.addWidget(self.hdr_chk)
        l_layout.addLayout(cam_ctrl_row)

        cam_opt_row = QHBoxLayout()
        cam_opt_row.setSpacing(6)
        self.hd_cam_chk = QCheckBox("HD 720p")
        self.hd_cam_chk.setChecked(bool(self.cam_worker.prefer_hd))
        self.hd_cam_chk.toggled.connect(self._on_toggle_hd_camera)
        cam_opt_row.addWidget(self.hd_cam_chk)

        self.debug_hud_btn = QPushButton("[Debug HUD]")
        self.debug_hud_btn.setCheckable(True)
        self.debug_hud_btn.setChecked(bool(self.cam_worker.debug_hud_enabled))
        self.debug_hud_btn.setFixedHeight(24)
        self.debug_hud_btn.setToolTip("Bật/tắt Vision Debug HUD (Ctrl+Shift+D)")
        self.debug_hud_btn.clicked.connect(self._on_toggle_vision_debug_hud)
        cam_opt_row.addWidget(self.debug_hud_btn)
        l_layout.addLayout(cam_opt_row)

        self.vision_debug_lbl = QLabel()
        self.vision_debug_lbl.setWordWrap(True)
        self.vision_debug_lbl.setVisible(bool(self.cam_worker.debug_hud_enabled))
        self.vision_debug_lbl.setStyleSheet(
            f"background: {BG_DARK}; color: {NEON_CYAN}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 5px; font-family: Consolas, monospace; font-size: 10px;"
        )
        l_layout.addWidget(self.vision_debug_lbl)

        # 4 Compact 4-Tier HUD Mode & Gesture Badges (2x2 Grid)
        badge_grid = QGridLayout()
        badge_grid.setSpacing(6)
        self.gesture_badges: Dict[str, QLabel] = {}
        badge_specs = [
            ("DRAWING", "[Chụm <0.28: Vẽ]", 0, 0),
            ("HOVER", "[Mở >0.40: Di]", 0, 1),
            ("LOCKED_COOLDOWN", "[Xòe/Nắm: Hủy]", 1, 0),
            ("PAN_ERASE", "[5 Ngón: Tẩy]", 1, 1),
        ]
        for key, text, r, c in badge_specs:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(28)
            lbl.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 11px; font-weight: 700;"
            )
            badge_grid.addWidget(lbl, r, c)
            self.gesture_badges[key] = lbl
        l_layout.addLayout(badge_grid)

        l_layout.addStretch(1)

        vu_hdr = QHBoxLayout()
        vu_hdr.addWidget(QLabel("Mic VU Meter"))
        vu_hdr.addStretch(1)
        self.vu_lbl = QLabel("-60.0 dB")
        self.vu_lbl.setStyleSheet(f"color: {SECONDARY_TEAL}; font-weight: 700; font-size: 11px;")
        vu_hdr.addWidget(self.vu_lbl)
        l_layout.addLayout(vu_hdr)

        self.vu_bar = QProgressBar()
        self.vu_bar.setFixedHeight(12)
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setTextVisible(False)
        l_layout.addWidget(self.vu_bar)

        workspace.addWidget(self.left_dock)

        # Center Smartboard Canvas
        self.canvas = SmartboardCanvas()
        self.canvas.region_cropped_for_ai.connect(self._on_region_cropped_for_ai)
        self.canvas.lasso_action_requested.connect(self._on_lasso_quick_action)
        self.canvas.slides_changed.connect(self._update_slide_counter)
        workspace.addWidget(self.canvas, stretch=1)

        # Right Dock (Trợ Lý Sư Phạm Gemini 3.5 Flash)
        self.right_dock = QFrame()
        self.right_dock.setFixedWidth(330)
        self.right_dock.setStyleSheet(
            f"background: {SURFACE_DARK}; border-left: 1px solid {BORDER_SLATE};"
        )
        r_layout = QVBoxLayout(self.right_dock)
        r_layout.setContentsMargins(10, 10, 10, 10)
        r_layout.setSpacing(6)

        ai_hdr = QHBoxLayout()
        ai_title = QLabel(f"Trợ Lý AI ({DEFAULT_GEMINI_MODEL})")
        ai_title.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 700; font-size: 13px;")
        ai_hdr.addWidget(ai_title)

        self.api_status_badge = QLabel("Chưa cấu hình")
        self.api_status_badge.setStyleSheet(
            f"background: {BORDER_SLATE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
        )
        ai_hdr.addWidget(self.api_status_badge)
        ai_hdr.addStretch(1)

        self.change_key_btn = QPushButton("[API Key]")
        self.change_key_btn.setFixedHeight(24)
        self.change_key_btn.clicked.connect(
            lambda: self.api_key_frame.setVisible(not self.api_key_frame.isVisible())
        )
        ai_hdr.addWidget(self.change_key_btn)
        r_layout.addLayout(ai_hdr)

        self.api_key_frame = QFrame()
        self.api_key_frame.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 8px;"
        )
        kf_vbox = QVBoxLayout(self.api_key_frame)
        kf_vbox.setContentsMargins(8, 8, 8, 8)
        kf_vbox.setSpacing(6)

        self.api_key_mask_lbl = QLabel()
        self.api_key_mask_lbl.setStyleSheet(
            f"color: {SECONDARY_TEAL}; font-size: 11px; font-weight: 600; border: none;"
        )
        kf_vbox.addWidget(self.api_key_mask_lbl)

        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self.api_key_edit = QLineEdit(self.api_key if self.api_key_state.source == "ui" else "")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Nhập GEMINI_API_KEY (ưu tiên UI > .env)...")
        self.show_key_btn = QPushButton("Hiện")
        self.show_key_btn.setFixedWidth(44)
        self.show_key_btn.clicked.connect(self._on_toggle_show_api_key)
        input_row.addWidget(self.api_key_edit)
        input_row.addWidget(self.show_key_btn)
        kf_vbox.addLayout(input_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        save_key_btn = QPushButton("Lưu")
        save_key_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 5px; padding: 4px 8px; font-weight: 700;"
        )
        save_key_btn.clicked.connect(self._on_save_api_key)

        self.test_api_btn = QPushButton("Test API")
        self.test_api_btn.setStyleSheet(
            f"background: {SECONDARY_TEAL}; color: {BG_DARK}; border-radius: 5px; padding: 4px 8px; font-weight: 700;"
        )
        self.test_api_btn.clicked.connect(self._on_test_api_connection)

        del_key_btn = QPushButton("Xóa Key")
        del_key_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 5px; padding: 4px 8px; font-weight: 700;"
        )
        del_key_btn.clicked.connect(self._on_delete_api_key)

        btn_row.addWidget(save_key_btn)
        btn_row.addWidget(self.test_api_btn)
        btn_row.addWidget(del_key_btn)
        kf_vbox.addLayout(btn_row)

        self.api_status_detail_lbl = QLabel()
        self.api_status_detail_lbl.setWordWrap(True)
        self.api_status_detail_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; border: none;"
        )
        kf_vbox.addWidget(self.api_status_detail_lbl)

        r_layout.addWidget(self.api_key_frame)
        self.api_key_frame.setVisible(not bool(self.api_key))
        self._refresh_api_key_ui(detail_msg="")

        # Smart Hybrid / Local AI Discovery Strip
        local_row = QHBoxLayout()
        local_row.setSpacing(4)
        self.routing_combo = QComboBox()
        self.routing_combo.addItems(["Hybrid", "Local", "Cloud"])
        self.routing_combo.currentTextChanged.connect(self._on_routing_mode_changed)
        self.local_model_combo = QComboBox()
        self.local_model_combo.addItem("Auto (Local chưa bật)")
        self.local_model_combo.currentIndexChanged.connect(self._on_local_model_selected)
        scan_local_btn = QPushButton("Quét Local")
        scan_local_btn.setFixedWidth(74)
        scan_local_btn.clicked.connect(self._on_discover_local_ai)
        local_row.addWidget(self.routing_combo, stretch=1)
        local_row.addWidget(self.local_model_combo, stretch=2)
        local_row.addWidget(scan_local_btn)
        r_layout.addLayout(local_row)

        for label, prompt_prefix, struct_json in [
            (
                "[1. Giải & Chữa Bài Trên Bảng]",
                "Hãy đọc toàn bộ chữ viết tay/công thức trên bảng, kiểm tra lỗi sai nếu có và giải chi tiết từng bước.",
                False,
            ),
            (
                "[2. Tóm Tắt & Trắc Nghiệm (JSON)]",
                "Hãy tóm tắt trọng tâm bài giảng trên bảng và tạo 3 câu hỏi trắc nghiệm 4 lựa chọn có đáp án giải thích.",
                True,
            ),
            (
                "[3. Hỏi Đáp Sư Phạm]",
                "Hãy gợi ý phương pháp giảng dạy trực quan và ví dụ thực tế cho nội dung trên bảng.",
                False,
            ),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {PRIMARY_INDIGO}; border-radius: 6px; font-weight: 700;"
            )
            btn.clicked.connect(
                lambda _=False, p=prompt_prefix, sj=struct_json: self._analyze_full_board(p, sj)
            )
            r_layout.addWidget(btn)

        self.crop_preview_lbl = QLabel()
        self.crop_preview_lbl.setFixedHeight(90)
        self.crop_preview_lbl.setAlignment(Qt.AlignCenter)
        self.crop_preview_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {SECONDARY_TEAL}; border-radius: 6px;"
        )
        self.crop_preview_lbl.setVisible(False)
        r_layout.addWidget(self.crop_preview_lbl)

        self.ai_output = QTextBrowser()
        self.ai_output.setOpenLinks(False)
        self.ai_output.setStyleSheet(
            f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px; font-size: 13px;"
        )
        init_msg = (
            f"Sẵn sàng hỗ trợ giảng dạy.\n"
            f"• Nhập lệnh trực tiếp như: **ghi lên bảng hộ tôi 2x + 1 = 3 đi** hoặc **vẽ trục tọa độ** (<5ms).\n"
            f"• Dùng **[Khoanh AI (Lasso)]** (phím `L`) để khoanh vùng viết tay → Giải / Viết lên bảng / Làm đẹp LaTeX.\n"
            f"• Hỗ trợ Local AI (Ollama / LM Studio / llama.cpp) + {DEFAULT_GEMINI_MODEL} bền bỉ ngoại tuyến."
        )
        self.last_ai_raw_text = init_msg
        self.ai_output.setHtml(OutputSanitizer.to_rich_html(init_msg, dark_mode=self.dark_mode))
        r_layout.addWidget(self.ai_output, stretch=1)

        push_board_btn = QPushButton("[Đưa nội dung AI lên bảng]")
        push_board_btn.setFixedHeight(26)
        push_board_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {SECONDARY_TEAL}; border-radius: 6px; font-weight: 700; font-size: 11px;"
        )
        push_board_btn.clicked.connect(self._on_push_last_ai_to_board)
        r_layout.addWidget(push_board_btn)

        chat_in_row = QHBoxLayout()
        chat_in_row.setSpacing(4)
        self.chat_input_edit = QLineEdit()
        self.chat_input_edit.setPlaceholderText("Hỏi AI hoặc ra lệnh: ghi lên bảng 2x + 1 = 3...")
        self.chat_input_edit.returnPressed.connect(self._on_chat_submit)
        send_chat_btn = QPushButton("Gửi")
        send_chat_btn.setFixedWidth(52)
        send_chat_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 6px; font-weight: 700;"
        )
        send_chat_btn.clicked.connect(self._on_chat_submit)
        chat_in_row.addWidget(self.chat_input_edit)
        chat_in_row.addWidget(send_chat_btn)
        r_layout.addLayout(chat_in_row)

        workspace.addWidget(self.right_dock)
        root.addLayout(workspace, stretch=1)

        # =====================================================================
        # BOTTOM DRAWING DOCK (Height: 54px)
        # =====================================================================
        bottom = QFrame()
        bottom.setFixedHeight(54)
        bottom.setStyleSheet(
            f"background: {BG_DARK}; border-top: 1px solid {BORDER_SLATE};"
        )
        b_layout = QHBoxLayout(bottom)
        b_layout.setContentsMargins(10, 8, 10, 8)
        b_layout.setSpacing(6)

        for tool_id, label in [
            ("LASSO", "[Khoanh AI (Lasso)]"),
            ("SELECT_REGION", "[Phân Tích Vùng]"),
            ("PEN", "[Bút vẽ]"),
            ("ERASER", "[Tẩy]"),
            ("LASER", "[Laser]"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _=False, t=tool_id: self._set_tool(t))
            b_layout.addWidget(btn)

        self.smart_shape_btn = QPushButton("[Nắn hình: BẬT]")
        self.smart_shape_btn.setFixedHeight(32)
        self.smart_shape_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 8px; font-weight: 700;"
        )
        self.smart_shape_btn.clicked.connect(self._on_toggle_smart_shape)
        b_layout.addWidget(self.smart_shape_btn)

        self.hold_snap_btn = QPushButton("[Giữ 350ms: TẮT]")
        self.hold_snap_btn.setFixedHeight(32)
        self.hold_snap_btn.clicked.connect(self._on_toggle_hold_snap)
        b_layout.addWidget(self.hold_snap_btn)

        for hex_col in ("#00F5D4", "#FEE440", "#FFFFFF", "#38B000", "#FF007F", "#FF6D00"):
            c_btn = QPushButton("")
            c_btn.setFixedSize(22, 22)
            c_btn.setStyleSheet(
                f"background: {hex_col}; border: 1px solid {TEXT_LIGHT}; border-radius: 11px;"
            )
            c_btn.clicked.connect(lambda _=False, hc=hex_col: self._set_ink_color(hc))
            b_layout.addWidget(c_btn)

        self.thick_lbl = QLabel("4px")
        self.thick_lbl.setStyleSheet(f"color: {SECONDARY_TEAL}; font-weight: 700;")
        b_layout.addWidget(self.thick_lbl)

        thick_slider = QSlider(Qt.Horizontal)
        thick_slider.setRange(2, 14)
        thick_slider.setValue(4)
        thick_slider.setFixedWidth(84)
        thick_slider.valueChanged.connect(self._on_thickness_changed)
        b_layout.addWidget(thick_slider)

        b_layout.addStretch(1)

        undo_btn = QPushButton("[Hoàn tác (Ctrl+Z)]")
        undo_btn.clicked.connect(self._on_hotkey_undo)
        b_layout.addWidget(undo_btn)

        redo_btn = QPushButton("[Làm lại (Redo)]")
        redo_btn.clicked.connect(self.canvas.redo)
        b_layout.addWidget(redo_btn)

        clear_btn = QPushButton("[Xóa Bảng]")
        clear_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {ALERT_ROSE}; border: 1px solid {ALERT_ROSE}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
        )
        clear_btn.clicked.connect(self.canvas.clear_board)
        b_layout.addWidget(clear_btn)

        root.addWidget(bottom)

    def _on_show_about(self) -> None:
        AboutBrandDialog(self).exec()

    def _on_hotkey_esc(self) -> None:
        if self.stack.currentIndex() == 1:
            self.canvas.trigger_esc_cancel()

    def _on_hotkey_undo(self) -> None:
        if self.stack.currentIndex() == 1 and self.canvas.request_undo():
            self.canvas.undo()

    def _on_user_authenticated(self, user: Dict[str, Any]) -> None:
        self.logged_in_user = user
        self.teacher_name_lbl.setText(f"GV: {user.get('full_name', 'Giáo viên')}")
        self.cam_worker.resume_camera_hardware()
        self.stack.setCurrentIndex(1)

    def _on_logout(self) -> None:
        # Release webcam hardware cleanly when logging out
        self.cam_worker.release_camera_hardware()
        self.canvas.drawing_fsm.reset()
        self.logged_in_user = {}
        self.auth_gate._refresh_tabs()
        self.stack.setCurrentIndex(0)

    def _on_toggle_studio_camera(self) -> None:
        if self.cam_worker._camera_paused:
            self.cam_worker.resume_camera_hardware()
            self.left_dock.setVisible(True)
        elif not self.left_dock.isVisible():
            self.left_dock.setVisible(True)
        else:
            self.cam_worker.release_camera_hardware()

    def _on_toggle_hd_camera(self, checked: bool) -> None:
        self.cam_worker.set_hd_resolution(bool(checked))

    def _on_toggle_vision_debug_hud(self) -> None:
        new_state = not bool(self.cam_worker.debug_hud_enabled)
        self.cam_worker.set_debug_hud(new_state)
        if hasattr(self, "debug_hud_btn"):
            self.debug_hud_btn.blockSignals(True)
            self.debug_hud_btn.setChecked(new_state)
            self.debug_hud_btn.blockSignals(False)
            self.debug_hud_btn.setStyleSheet(
                f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; font-weight: 700;"
                if new_state
                else ""
            )
        if hasattr(self, "vision_debug_lbl"):
            self.vision_debug_lbl.setVisible(new_state)

    def _on_camera_packet(self, pkt: dict) -> None:
        if self.stack.currentIndex() == 0:
            self.auth_gate.on_camera_packet(pkt)
            return

        qimg: Optional[QImage] = pkt.get("qimage")
        paused: bool = bool(pkt.get("paused", False))
        light: Dict[str, Any] = pkt.get("lighting") or {}
        fps: float = float(pkt.get("fps") or 0.0)

        if paused:
            self.pip_cam_lbl.setPixmap(QPixmap())
            self.pip_cam_lbl.setText("Camera đã giải phóng an toàn")
        elif qimg is not None:
            scaled = qimg.scaled(250, 188, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pip_cam_lbl.setPixmap(QPixmap.fromImage(scaled))
        else:
            self.pip_cam_lbl.setPixmap(QPixmap())
            self.pip_cam_lbl.setText("Chưa kết nối Webcam")

        luma = float(light.get("mean_luma") or 0.0)
        sharp = float(light.get("laplacian_var") or 0.0)
        self.telemetry_lbl.setText(f"Sáng: {luma:.0f} | Nét: {sharp:.0f} | {fps:.1f} FPS")
        self.perf_pipeline.update_cv_fps(fps)
        self._refresh_status_pill()

        if hasattr(self, "vision_debug_lbl") and self.cam_worker.debug_hud_enabled:
            vt = pkt.get("vision_telemetry") or {}
            fstates = vt.get("finger_states") or []
            fangs = vt.get("finger_angles_deg") or []
            short_st = "/".join(s[0] for s in fstates[:5]) if len(fstates) == 5 else "-"
            short_ang = "/".join(str(int(round(a))) for a in fangs[:5]) if len(fangs) == 5 else "-"
            pr_val = vt.get("pinch_ratio")
            pr_txt = f"{float(pr_val):.2f}" if pr_val is not None else "--"
            lmode = vt.get("lighting_mode", "NORMAL")
            byp = "BYPASS" if vt.get("clahe_bypassed") else "CLAHE"
            palm_ok = "FRONT" if vt.get("palm_facing_ok", True) else "SIDE"
            roi_tag = " ROI2x" if vt.get("stage2_roi_zoom") else ""
            kal_tag = " KAL" if vt.get("kalman_predicted") else ""
            cam_fps_v = float(vt.get("cam_fps") or fps)
            ai_fps_v = float(vt.get("ai_fps") or fps)
            cv_ms_v = float(vt.get("cv_latency_ms") or 0.0)
            palm_sc_v = float(vt.get("palm_facing_score") or 0.0)
            self.vision_debug_lbl.setText(
                f"Cam {cam_fps_v:.0f}fps | AI {ai_fps_v:.0f}fps ({cv_ms_v:.1f}ms) {vt.get('resolution', '')}\n"
                f"Light: {lmode} [{byp}] | Lock: {vt.get('teacher_mode', '-')} #{vt.get('locked_teacher_id', '-')} (Ign:{vt.get('ignored_count', 0)})\n"
                f"5-Ngón: {short_st} ({short_ang}°) | Pinch: {pr_txt}\n"
                f"Palm: {palm_ok} ({palm_sc_v:.2f}) | FSM: {vt.get('fsm_state', 'IDLE')}{roi_tag}{kal_tag}"
            )

        fsm_upd = self.canvas.apply_cv_fsm_packet(pkt)
        hud_key = fsm_upd.hud_mode.value
        if hud_key == "LOCKED_COOLDOWN":
            self.hand_status_lbl.setText(
                f"Khóa ngòi bút ({int(max(0.0, fsm_upd.cooldown_remaining_ms))}ms) • {fsm_upd.cancel_reason or 'Cooldown'}"
            )
            self.hand_status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700; font-size: 12px;")
        else:
            self.hand_status_lbl.setText(f"Chế độ: {fsm_upd.hud_mode.label_vi}")
            self.hand_status_lbl.setStyleSheet(f"color: {NEON_CYAN}; font-weight: 700; font-size: 12px;")

        for key, lbl in self.gesture_badges.items():
            if key == hud_key:
                accent = ALERT_ROSE if key == "LOCKED_COOLDOWN" else NEON_CYAN
                lbl.setStyleSheet(
                    f"background: {accent}; color: {BG_DARK}; border-radius: 6px; font-size: 11px; font-weight: 700;"
                )
            else:
                lbl.setStyleSheet(
                    f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 11px; font-weight: 700;"
                )

    def _on_toggle_hdr(self, checked: bool) -> None:
        self.cam_worker.auto_hdr_backlight = checked

    def _update_slide_counter(self) -> None:
        total = len(self.canvas.slides)
        idx = self.canvas.current_slide_idx + 1 if total > 0 else 0
        self.slide_counter_lbl.setText(f"Slide {idx}/{total}")

    def _on_prev_slide(self) -> None:
        if self.canvas.current_slide_idx > 0:
            self.canvas.current_slide_idx -= 1
            self._update_slide_counter()
            self.canvas._bump_scene()
            self.canvas.update()

    def _on_next_slide(self) -> None:
        if self.canvas.current_slide_idx + 1 < len(self.canvas.slides):
            self.canvas.current_slide_idx += 1
            self._update_slide_counter()
            self.canvas._bump_scene()
            self.canvas.update()

    def _on_toggle_grid(self) -> None:
        self.canvas.grid_enabled = not self.canvas.grid_enabled
        self.canvas._bump_scene()
        self.canvas.update()

    def _on_open_slide_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở Slide / Ảnh", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            pm = QPixmap(path)
            if not pm.isNull():
                self.canvas.slides.append({
                    "title": Path(path).name,
                    "subtitle": "",
                    "bullets": [],
                    "formula": "",
                    "pixmap": pm,
                    "is_tutorial": False,
                })
                self.canvas.current_slide_idx = len(self.canvas.slides) - 1
                self._update_slide_counter()
                self.canvas._bump_scene()
                self.canvas.update()

    def _on_toggle_recording(self) -> None:
        if not self.is_recording_audio:
            self.audio_frames.clear()
            self.is_recording_audio = True
            self.rec_btn.setText("[Dừng Ghi Âm]")
        else:
            self.is_recording_audio = False
            self.rec_btn.setText("[Ghi Âm]")
            if self.audio_frames:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                out_wav = RECORDINGS_DIR / f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                audio_np = np.concatenate(self.audio_frames, axis=0)
                pcm16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
                with wave.open(str(out_wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    wf.writeframes(pcm16.tobytes())

    def _set_tool(self, tool_id: str) -> None:
        self.canvas.active_tool = tool_id

    def _set_ink_color(self, hex_col: str) -> None:
        self.canvas.ink_color = QColor(hex_col)
        if self.canvas.active_tool == "ERASER":
            self.canvas.active_tool = "PEN"

    def _on_thickness_changed(self, val: int) -> None:
        self.canvas.pen_thickness = val
        self.thick_lbl.setText(f"{val}px")

    def _on_toggle_smart_shape(self) -> None:
        self.canvas.smart_shape_enabled = not self.canvas.smart_shape_enabled
        if self.canvas.smart_shape_enabled:
            self.smart_shape_btn.setText("[Nắn hình: BẬT]")
            self.smart_shape_btn.setStyleSheet(
                f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 8px; font-weight: 700;"
            )
        else:
            self.smart_shape_btn.setText("[Nắn hình: TẮT]")
            self.smart_shape_btn.setStyleSheet("")

    def _on_toggle_hold_snap(self) -> None:
        self.canvas.hold_to_snap_350ms = not self.canvas.hold_to_snap_350ms
        self.hold_snap_btn.setText(
            "[Giữ 350ms: BẬT]" if self.canvas.hold_to_snap_350ms else "[Giữ 350ms: TẮT]"
        )

    def _refresh_api_key_ui(
        self,
        status: Optional[ApiConnectionStatus] = None,
        detail_msg: Optional[str] = None,
    ) -> None:
        self.api_key_state = resolve_api_key_layered(self.api_key_edit.text().strip() if hasattr(self, "api_key_edit") and self.api_key_state.source == "ui" and self.api_key_edit.text().strip() else None)
        self.api_key = self.api_key_state.api_key

        if status is not None:
            self.gemini_conn_status = status
        elif not self.api_key:
            self.gemini_conn_status = ApiConnectionStatus.UNCONFIGURED

        if not self.api_key:
            self.api_key_mask_lbl.setText("Key: Chưa cấu hình (nhập UI hoặc .env)")
            self.api_status_badge.setText("Chưa cấu hình")
            self.api_status_badge.setStyleSheet(
                f"background: {BORDER_SLATE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
            )
        else:
            src_label = "UI Settings" if self.api_key_state.source == "ui" else ".env / ENV"
            self.api_key_mask_lbl.setText(f"Key: {mask_api_key(self.api_key)} • Nguồn: {src_label}")
            if self.gemini_conn_status == ApiConnectionStatus.CONNECTED:
                self.api_status_badge.setText("Đã kết nối")
                self.api_status_badge.setStyleSheet(
                    "background: #10B981; color: #052E16; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 800;"
                )
            elif self.gemini_conn_status == ApiConnectionStatus.ERROR:
                self.api_status_badge.setText("Lỗi Key/Quota")
                self.api_status_badge.setStyleSheet(
                    f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 800;"
                )
            else:
                self.api_status_badge.setText("Chưa xác thực")
                self.api_status_badge.setStyleSheet(
                    f"background: {BORDER_SLATE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
                )

        if detail_msg is not None:
            self.api_status_detail_lbl.setText(detail_msg)

    def _on_toggle_show_api_key(self) -> None:
        if self.api_key_edit.echoMode() == QLineEdit.Password:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Ẩn")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Hiện")

    def _on_save_api_key(self) -> None:
        entered = self.api_key_edit.text().strip()
        if not entered:
            self._refresh_api_key_ui(
                status=ApiConnectionStatus.ERROR,
                detail_msg="Vui lòng nhập API Key trước khi lưu (hoặc nhấn Xóa Key).",
            )
            return
        save_ui_api_settings(api_key=entered)
        self.api_key = entered
        self.gemini_conn_status = ApiConnectionStatus.UNCONFIGURED
        self._refresh_api_key_ui(
            status=ApiConnectionStatus.UNCONFIGURED,
            detail_msg="Đã lưu Key vào cài đặt cục bộ. Nhấn [Test API] để kiểm tra kết nối thật.",
        )

    def _on_delete_api_key(self) -> None:
        clear_ui_api_key()
        self.api_key_edit.clear()
        self.api_key_state = resolve_api_key_layered()
        self.api_key = self.api_key_state.api_key
        msg = (
            "Đã xóa Key trên UI. Đang dùng Key dự phòng từ .env."
            if self.api_key
            else "Đã xóa API Key khỏi cấu hình cục bộ."
        )
        self._refresh_api_key_ui(status=ApiConnectionStatus.UNCONFIGURED, detail_msg=msg)

    def _on_test_api_connection(self) -> None:
        draft = self.api_key_edit.text().strip()
        effective = draft or self.api_key
        if not effective:
            self._refresh_api_key_ui(
                status=ApiConnectionStatus.UNCONFIGURED,
                detail_msg="Chưa có API Key để kiểm tra. Hãy nhập key hoặc cấu hình .env.",
            )
            return
        if self.health_worker is not None and self.health_worker.isRunning():
            return
        self.test_api_btn.setEnabled(False)
        self.test_api_btn.setText("Đang kiểm...")
        self.api_status_detail_lbl.setText("Đang gọi Gemini generateContent kiểm tra kết nối thật...")
        self.health_worker = GeminiHealthCheckWorker(effective)
        self.health_worker.completed.connect(self._on_health_check_completed)
        self.health_worker.start()

    def _on_health_check_completed(self, res: GeminiHealthCheckResult) -> None:
        self.test_api_btn.setEnabled(True)
        self.test_api_btn.setText("Test API")
        if res.ok:
            msg = f"✓ {res.message_vi} ({res.model_used}, {res.latency_ms:.0f}ms)"
        else:
            msg = f"⚠ {res.message_vi}"
        self._refresh_api_key_ui(status=res.status, detail_msg=msg)

    def _on_worker_status_changed(self, status_obj: object) -> None:
        if isinstance(status_obj, ApiConnectionStatus):
            self._refresh_api_key_ui(status=status_obj)

    def _set_sanitized_ai_html(self, raw_text: str) -> None:
        self.last_ai_raw_text = raw_text
        self.ai_output.setHtml(OutputSanitizer.to_rich_html(raw_text, dark_mode=self.dark_mode))

    def _refresh_status_pill(self) -> None:
        mode = self.local_ai_manager.routing_mode.label_vi
        if self.local_ai_manager.active_model and self.local_ai_manager.routing_mode != AiRoutingMode.CLOUD:
            mdl = self.local_ai_manager.active_model
        else:
            mdl = DEFAULT_GEMINI_MODEL
        self.perf_pipeline.update_ai_metrics(
            ai_mode=mode,
            active_model=mdl,
            tokens_per_sec=self.local_ai_manager.last_tokens_per_sec,
        )
        if hasattr(self, "status_pill_lbl"):
            self.status_pill_lbl.setText(self.perf_pipeline.telemetry.status_pill_text())

    def _on_toggle_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self.canvas.set_dark_mode(self.dark_mode)
        bg = BG_DARK if self.dark_mode else "#F8FAFC"
        fg = TEXT_LIGHT if self.dark_mode else "#0F172A"
        bd = BORDER_SLATE if self.dark_mode else "#CBD5E1"
        self.ai_output.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {bd}; border-radius: 8px; padding: 8px; font-size: 13px;"
        )
        if self.last_ai_raw_text:
            self._set_sanitized_ai_html(self.last_ai_raw_text)

    def _on_routing_mode_changed(self, text: str) -> None:
        mapping = {"Hybrid": AiRoutingMode.HYBRID, "Local": AiRoutingMode.LOCAL, "Cloud": AiRoutingMode.CLOUD}
        self.local_ai_manager.routing_mode = mapping.get(text, AiRoutingMode.HYBRID)
        self._refresh_status_pill()

    def _on_discover_local_ai(self) -> None:
        if self.discovery_worker is not None and self.discovery_worker.isRunning():
            return
        self.discovery_worker = LocalDiscoveryWorker(self.local_ai_manager)
        self.discovery_worker.discovered.connect(self._on_local_discovered)
        self.discovery_worker.start()

    def _on_local_discovered(self, servers: List[DiscoveredLocalServer]) -> None:
        if not hasattr(self, "local_model_combo"):
            return
        self.local_model_combo.blockSignals(True)
        self.local_model_combo.clear()
        added = 0
        for srv in servers:
            for mdl in srv.models:
                self.local_model_combo.addItem(f"{srv.provider}: {mdl}", (srv.base_url, mdl, srv.provider))
                added += 1
            if not srv.models:
                self.local_model_combo.addItem(f"{srv.provider} ({srv.base_url})", (srv.base_url, "", srv.provider))
                added += 1
        if added == 0:
            self.local_model_combo.addItem("Chưa phát hiện Ollama/LM Studio", ("", "", ""))
        self.local_model_combo.blockSignals(False)
        self._refresh_status_pill()

    def _on_local_model_selected(self, idx: int) -> None:
        if idx < 0:
            return
        data = self.local_model_combo.itemData(idx)
        if isinstance(data, tuple) and len(data) == 3 and data[0] and data[1]:
            self.local_ai_manager.select_local_model(data[0], data[1], data[2])
            self.local_ai_manager.warmup_async()
            self._refresh_status_pill()

    def _on_chat_submit(self) -> None:
        user_msg = self.chat_input_edit.text().strip()
        if not user_msg:
            return
        self.chat_input_edit.clear()
        self.right_dock.setVisible(True)

        # Tier 1: Fast Local Intent Parser (<5ms, instant board write/draw/clear)
        fast_act = WhiteboardAgent.parse_fast_intent(user_msg)
        if fast_act is not None:
            summary = self.canvas.execute_whiteboard_actions([fast_act])
            self.perf_pipeline.update_ai_metrics(ai_mode="Instant", active_model="Local-Intent", tokens_per_sec=0.0)
            self._refresh_status_pill()
            self._set_sanitized_ai_html(f"**Lệnh tức thì (<5ms):** {user_msg}\n\n✓ {summary}")
            return

        self.chat_history.append({"role": "user", "content": user_msg})
        self._set_sanitized_ai_html(f"Đang xử lý yêu cầu: **{user_msg}**...")
        self._start_hybrid_worker(user_msg, png_bytes=None, structured_json=False)

    def _on_push_last_ai_to_board(self) -> None:
        if not self.last_ai_raw_text.strip():
            return
        lines = [
            ln.strip("•-* ").strip()
            for ln in OutputSanitizer.sanitize_plain_text(self.last_ai_raw_text).splitlines()
            if ln.strip()
        ]
        if not lines:
            return
        if len(lines) == 1:
            act = WhiteboardAction(tool="write_on_board", args={"content": lines[0], "format": "latex"})
        else:
            act = WhiteboardAction(tool="solve_on_board", args={"steps": lines[:6]})
        self.canvas.execute_whiteboard_actions([act])

    def _on_lasso_quick_action(self, action_id: str, region: LassoSelectionRegion) -> None:
        try:
            qact = LassoQuickAction(action_id)
        except Exception:
            qact = LassoQuickAction.SOLVE_AI
        self._active_lasso_action = qact
        self._active_lasso_region = region
        rect = QRectF(region.bbox_x, region.bbox_y, region.bbox_w, region.bbox_h)
        png_bytes, cropped_pm = self.canvas.export_board_png(rect, enhance_crop=True)
        self.right_dock.setVisible(True)
        self.crop_preview_lbl.setVisible(True)
        self.crop_preview_lbl.setPixmap(
            cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        prompt = build_lasso_prompt(qact)
        self._set_sanitized_ai_html(f"Đang xử lý vùng khoanh ({qact.value})...")
        self._start_hybrid_worker(prompt, png_bytes=png_bytes, structured_json=False, include_board_tools=(qact != LassoQuickAction.BEAUTIFY_LATEX))

    def _on_region_cropped_for_ai(self, _png_bytes: bytes, cropped_pm: QPixmap) -> None:
        self.right_dock.setVisible(True)
        self.crop_preview_lbl.setVisible(True)
        self.crop_preview_lbl.setPixmap(
            cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _analyze_full_board(self, prompt: str, structured_json: bool = False) -> None:
        self.right_dock.setVisible(True)
        self._active_lasso_action = None
        png_bytes, cropped_pm = self.canvas.export_board_png(None, enhance_crop=True)
        self.crop_preview_lbl.setVisible(True)
        self.crop_preview_lbl.setPixmap(
            cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._set_sanitized_ai_html(f"Đang phân tích toàn bảng...")
        self._start_hybrid_worker(prompt, png_bytes=png_bytes, structured_json=structured_json)

    def _start_hybrid_worker(
        self,
        prompt: str,
        png_bytes: Optional[bytes],
        structured_json: bool = False,
        include_board_tools: bool = True,
    ) -> None:
        self.api_key_state = resolve_api_key_layered(
            self.api_key_edit.text().strip() if self.api_key_edit.text().strip() else None
        )
        self.api_key = self.api_key_state.api_key
        self._stream_raw_accum = ""
        eff_prompt = prompt
        if include_board_tools and not structured_json:
            canvas_sz = (float(max(640, self.canvas.width())), float(max(420, self.canvas.height())))
            free_regs = find_free_regions_normalized(self.canvas._occupied_rects(), canvas_sz, max_regions=3)
            eff_prompt = f"{prompt}\n\n{WhiteboardAgent.build_tool_prompt_instruction(canvas_size=canvas_sz, free_regions_norm=free_regs)}"
        self.hybrid_worker = HybridAiWorker(
            self.local_ai_manager,
            eff_prompt,
            history=self.chat_history[-10:],
            png_bytes=png_bytes,
            cloud_api_key=self.api_key,
            structured_json=structured_json,
            include_board_tools=False,
        )
        self.hybrid_worker.token_delta.connect(self._on_hybrid_token)
        self.hybrid_worker.finished_result.connect(self._on_hybrid_finished)
        self.hybrid_worker.start()

    def _on_hybrid_token(self, chunk: str) -> None:
        self._stream_raw_accum += chunk
        self._set_sanitized_ai_html(self._stream_raw_accum)

    def _on_hybrid_finished(self, res: LocalAiRunResult) -> None:
        if not res.ok:
            self._set_sanitized_ai_html(
                f"⚠ **{res.error_message_vi}**\n\nBảng trắng, vẽ tay & cử chỉ vẫn hoạt động 100%."
            )
            if res.engine_used == "Cloud":
                self._refresh_api_key_ui(status=ApiConnectionStatus.ERROR)
            return

        if res.engine_used == "Cloud":
            self._refresh_api_key_ui(status=ApiConnectionStatus.CONNECTED)

        self.perf_pipeline.update_ai_metrics(
            ai_mode=res.engine_used,
            active_model=res.model_used,
            tokens_per_sec=res.tokens_per_sec,
            latency_ms=res.total_ms,
        )
        self._refresh_status_pill()

        # Handle Smart Lasso Beauty Replace flow
        if self._active_lasso_action == LassoQuickAction.BEAUTIFY_LATEX and self._active_lasso_region is not None:
            clean_formula = extract_clean_beautified_formula(res.text)
            if clean_formula:
                self.canvas.replace_lasso_with_formula(self._active_lasso_region, clean_formula)
                self._set_sanitized_ai_html(f"✓ Đã làm đẹp công thức trên bảng:\n\n$${clean_formula}$$")
                self._active_lasso_action = None
                self._active_lasso_region = None
                return

        actions, clean_reply = WhiteboardAgent.extract_actions_from_response(res.text)
        pref_anchor = (
            (self._active_lasso_region.bbox_x + self._active_lasso_region.bbox_w + 24.0, self._active_lasso_region.bbox_y)
            if self._active_lasso_region is not None
            else None
        )
        if actions:
            act_report = self.canvas.execute_whiteboard_actions(actions, preferred_anchor=pref_anchor)
            if act_report:
                clean_reply = f"{clean_reply}\n\n---\n{act_report}" if clean_reply else act_report
        elif self._active_lasso_action == LassoQuickAction.WRITE_SOLUTION_TO_BOARD and clean_reply.strip():
            step_lines = [ln.strip() for ln in clean_reply.splitlines() if ln.strip()][:6]
            self.canvas.execute_whiteboard_actions(
                [WhiteboardAction(tool="solve_on_board", args={"steps": step_lines})],
                preferred_anchor=pref_anchor,
            )

        self._active_lasso_action = None
        self._active_lasso_region = None
        self.chat_history.append({"role": "assistant", "content": clean_reply})
        badge_hdr = f"*[{res.engine_used} · {res.model_used} · {res.tokens_per_sec:.1f} tok/s]*"
        self._set_sanitized_ai_html(f"{badge_hdr}\n\n{clean_reply}")

    def _start_gemini_worker(
        self, prompt: str, png_bytes: Optional[bytes], structured_json: bool = False
    ) -> None:
        self.api_key_state = resolve_api_key_layered(self.api_key_edit.text().strip() if self.api_key_edit.text().strip() else None)
        self.api_key = self.api_key_state.api_key
        self.gemini_worker = GeminiVisionWorker(
            self.api_key, prompt, png_bytes, structured_json=structured_json
        )
        self.gemini_worker.finished_text.connect(self._set_sanitized_ai_html)
        self.gemini_worker.status_changed.connect(self._on_worker_status_changed)
        self.gemini_worker.start()

    def _maybe_show_first_run_onboarding(self) -> None:
        if self.api_key:
            return
        cfg = load_ui_api_settings()
        if str(cfg.get("onboarding_skipped", "")).lower() in ("true", "1"):
            return
        dlg = ApiKeyOnboardingDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_key:
            self.api_key_state = resolve_api_key_layered(dlg.saved_key)
            self.api_key = self.api_key_state.api_key
            self.api_key_edit.setText(self.api_key)
            self._refresh_api_key_ui(status=ApiConnectionStatus.CONNECTED)

    def closeEvent(self, event) -> None:
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as exc:
                _ = exc
        self.cam_worker.release_camera_hardware()
        self.cam_worker.stop()
        super().closeEvent(event)


def main() -> None:
    set_windows_app_user_model_id()
    setup_app_logging()
    app = QApplication(sys.argv)
    if APP_LOGO_ICO.exists():
        app.setWindowIcon(QIcon(str(APP_LOGO_ICO)))
    for fam in ("Segoe UI", "Arial", "Tahoma"):
        if fam in QFontDatabase.families():
            app.setFont(QFont(fam, 10))
            break
    app.setStyleSheet(
        f"""
        QWidget {{ background-color: {BG_DARK}; color: {TEXT_LIGHT}; }}
        QPushButton {{ background-color: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 4px 8px; }}
        QPushButton:hover {{ border: 1px solid {PRIMARY_INDIGO}; }}
        QLineEdit, QComboBox {{ background-color: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 5px; }}
        """
    )
    win = AirTeachMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
