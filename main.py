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
from collections import OrderedDict, deque
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
    QIcon,
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
    QIntValidator,
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
    QInputDialog,
    QScrollArea,
    QMenu,
)

import importlib.util

try:
    from PySide6.QtSvgWidgets import QSvgWidget
    from PySide6.QtSvg import QSvgRenderer
    HAS_QTSVG = True
except Exception as exc:
    _LOGGER.debug("QtSvg optional module unavailable: %s", exc) if "_LOGGER" in globals() else None
    HAS_QTSVG = False

HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
HAS_SOUNDDEVICE = importlib.util.find_spec("sounddevice") is not None
sd: Any = None


def _ensure_sounddevice() -> bool:
    global sd, HAS_SOUNDDEVICE
    if sd is not None:
        return True
    if not HAS_SOUNDDEVICE:
        return False
    try:
        import sounddevice as _sd

        sd = _sd
        return True
    except Exception as exc:
        sys.stderr.write(f"[AirTeachAI] sounddevice unavailable: {exc}\n")
        HAS_SOUNDDEVICE = False
        return False


HAS_GENAI_SDK = importlib.util.find_spec("google.genai") is not None

from cv_engine.canvas_scene import (
    CanvasObject,
    HANDLE_NAMES,
    STICKY_PALETTE,
    SlideSceneGraph,
    compute_points_bbox,
    compute_resized_bbox,
    make_shape_object,
    make_shape_points,
    make_sticky_object,
    make_text_object,
    new_object_id,
    rotate_point,
)
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
from cv_engine.file_attachment_parser import (
    ParsedAttachment,
    parse_attachment_file,
)
from cv_engine.gemini_client import (
    AVAILABLE_GEMINI_MODELS,
    ApiConnectionStatus,
    DEFAULT_GEMINI_MODEL,
    GeminiHealthCheckResult,
    ResilientGeminiClient,
    build_model_chain,
    clear_ui_api_key,
    compress_snapshot_png,
    load_ui_api_settings,
    mask_api_key,
    resolve_api_key_layered,
    resolve_api_key_pool,
    save_ui_api_settings,
    split_api_keys,
)
from cv_engine.hand_gesture import (
    DualHandGestureCoordinator,
    HandGestureEngine,
    HandTelemetryPacket,
    Kinematic60HzExtrapolator,
    SlideNavigationGestureDetector,
    classify_hand_3d_vectors,
    map_active_workspace_norm,
)
from cv_engine.preprocessing import (
    AdaptiveExposureController,
    build_gamma_lut,
    preprocess_adaptive_frame,
)
from cv_engine.document_presenter import (
    LectureDocument,
    SUPPORTED_DOC_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    compute_slide_rect_on_canvas,
    export_annotated_lecture_pdf,
    export_annotated_slides_to_pngs,
    get_office_conversion_notice,
    list_folder_images,
    load_image_file_qimage,
    warmup_document_engines,
)
from cv_engine.gguf_model_manager import LocalAiManagerDialog, get_models_storage_dir, list_downloaded_models
from cv_engine.radial_pie_menu import RadialPieMenuOverlay
from cv_engine.rust_bridge import HAS_RUST_CORE, RUST_CORE_VERSION
from cv_engine.spotlight_tour import SpotlightTourOverlay, build_default_airteach_spotlight_steps
from cv_engine.user_guide_dialog import GestureCheatSheetDialog, OnboardingTourDialog, UserGuideDialog
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
    ensure_concurrent_futures_eager,
    get_app_logger,
    get_config_file_path,
    get_user_data_dir,
    install_global_excepthook,
    resource_path,
    set_windows_app_user_model_id,
    setup_app_logging,
)

ensure_concurrent_futures_eager()
from cv_engine.performance_pipeline import (
    AutoProfileController,
    DirtyCanvasBackingTracker,
    PerformancePipeline,
    PerformanceProfile,
    StaggeredInferenceScheduler,
    compute_cv_tracking_size,
)
from cv_engine.shape_recognizer import beautify_stroke, catmull_rom_spline, resample_arc_length
from cv_engine.whiteboard_agent import (
    WhiteboardAction,
    WhiteboardAgent,
    build_vector_shape_strokes,
    find_free_board_region,
    find_free_regions_normalized,
)

_LOGGER = get_app_logger("main")


def _load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    if env_path is not None:
        candidates = [env_path]
    else:
        exe_env = Path(sys.executable).resolve().parent / ".env" if getattr(sys, "frozen", False) else Path(__file__).resolve().parent / ".env"
        candidates = [exe_env, get_user_data_dir() / ".env"]
    for path in candidates:
        if not path.exists():
            continue
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
            _LOGGER.warning("Unable to read .env at %s: %s", path, exc)


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
_ASSETS_HAND_MODEL = resource_path("assets/models/hand_landmarker.task")
HAND_LANDMARKER_MODEL_PATH = (
    _ASSETS_HAND_MODEL
    if _ASSETS_HAND_MODEL.exists() and _ASSETS_HAND_MODEL.stat().st_size > 500_000
    else (MODELS_DIR / "hand_landmarker.task")
)
RECORDINGS_DIR = DATA_DIR / "recordings"
BRAND_LOGO_SVG = resource_path("assets/branding/airteach_logo.svg")
_ICON_PRIMARY = resource_path("assets/icons/airteach.ico")
APP_LOGO_ICO = _ICON_PRIMARY if _ICON_PRIMARY.exists() else resource_path("assets/logo.ico")

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
    except Exception as exc:
        _LOGGER.warning("Model download failed (%s): %s", path, exc)
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
    except Exception as exc:
        _LOGGER.warning("Failed reading users DB %s: %s", USERS_DB_PATH, exc)
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


class CameraCaptureThread(threading.Thread):
    """
    Lock-free low-latency webcam capture thread (`CAP_DSHOW`/`CAP_V4L2`, `CAP_PROP_BUFFERSIZE=1`).
    Continuously drops stale buffered frames so capture-to-inference latency remains < 16ms.
    """

    def __init__(self, worker: "VisionInferenceWorker") -> None:
        super().__init__(name="airteach-py-capture", daemon=True)
        self.worker = worker

    def run(self) -> None:
        self.worker._capture_loop()


class VisionInferenceWorker(QThread):
    frame_ready = Signal(dict)
    calibration_done = Signal(dict)
    telemetry_ready = Signal(object)

    def __init__(self, initial_cam_idx: int = 0) -> None:
        super().__init__()
        self.vision_cfg = load_vision_config()
        self.cam_idx = initial_cam_idx
        self.auto_hdr_backlight = True
        self.debug_hud_enabled = bool(self.vision_cfg.hud.debug_hud_default_enabled)
        self.skeleton_overlay_enabled = True
        self.teacher_lock_enabled = bool(self.vision_cfg.subject_tracker.teacher_lock_enabled)
        self.perf_mode: str = "AUTO"  # "AUTO" | "ULTRA_LIGHT" | "BALANCED" | "HIGH_QUALITY"
        self.primary_hand_pref: str = "auto"  # "auto" | "right" | "left"
        self.prefer_hd = bool(self.vision_cfg.camera.prefer_hd)
        self.auth_mode: bool = True
        self.running = True
        self._camera_paused = False
        self._is_processing = False
        self._switch_requested: Optional[int] = None
        self._reopen_requested: bool = False
        self._calibrate_requested: bool = False
        self._was_pinched = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._frame_event = threading.Event()
        self._capture_thread: Optional[CameraCaptureThread] = None
        self._cam_fps_smoothed: float = 0.0
        self._last_swipe_dir: Optional[str] = None
        self._last_swipe_until: float = 0.0

    def request_auto_calibration(self) -> None:
        self._calibrate_requested = True

    def set_debug_hud(self, enabled: bool) -> None:
        self.debug_hud_enabled = bool(enabled)

    def set_skeleton_overlay(self, enabled: bool) -> None:
        self.skeleton_overlay_enabled = bool(enabled)

    def set_teacher_lock_enabled(self, enabled: bool) -> None:
        self.teacher_lock_enabled = bool(enabled)

    def set_perf_mode(self, mode: str) -> None:
        if mode in ("AUTO", "ULTRA_LIGHT", "BALANCED", "HIGH_QUALITY"):
            self.perf_mode = mode

    def set_primary_hand_pref(self, pref: str) -> None:
        if pref in ("auto", "right", "left"):
            self.primary_hand_pref = pref

    def set_hd_resolution(self, hd: bool) -> None:
        if self.prefer_hd != bool(hd):
            self.prefer_hd = bool(hd)
            self._reopen_requested = True

    def set_auth_mode(self, enabled: bool) -> None:
        """True on Face-ID AuthGate (needs full-res face & liveness); False in Studio (max hand FPS)."""
        self.auth_mode = bool(enabled)

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
        if not self.wait(6000):
            _LOGGER.warning("CameraHandWorker.stop() waited 6000ms for capture/CV teardown.")

    def _open_camera(self, idx: int) -> Optional[cv2.VideoCapture]:
        req_w = self.vision_cfg.camera.hd_width if self.prefer_hd else self.vision_cfg.camera.width
        req_h = self.vision_cfg.camera.hd_height if self.prefer_hd else self.vision_cfg.camera.height
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        for backend in backends:
            if not self.running:
                return None
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if sys.platform == "win32" and backend == cv2.CAP_DSHOW:
                    # Request MJPG before resolution/FPS so USB webcams unlock 30-60 FPS instead of YUY2 15fps
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
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
        face_extractor = ArcFace512Extractor(models_dir=MODELS_DIR, min_detection_confidence=0.45)
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
            enabled=self.teacher_lock_enabled,
        )
        if not self.running:
            pose_estimator.close()
            face_extractor.close()
            return
        # Micro CPU benchmark (~3ms) to select optimal MediaPipe complexity dynamically
        t_bench0 = time.perf_counter()
        _probe = np.full((160, 160, 3), 127, dtype=np.uint8)
        _ = cv2.GaussianBlur(_probe, (5, 5), 1.2)
        bench_ms = (time.perf_counter() - t_bench0) * 1000.0
        dyn_complexity = 0 if bench_ms > 2.5 else 1

        hand_engine = HandGestureEngine(
            model_path=HAND_LANDMARKER_MODEL_PATH,
            num_hands=max(2, int(vcfg.hand_tracking.num_hands)),
            min_detection_confidence=vcfg.hand_tracking.min_detection_confidence,
            min_presence_confidence=vcfg.hand_tracking.min_presence_confidence,
            min_tracking_confidence=vcfg.hand_tracking.min_tracking_confidence,
            roi_area_threshold=vcfg.hand_tracking.roi_area_threshold,
            roi_expand_factor=max(2.2, float(vcfg.hand_tracking.roi_expand_factor)),
            roi_padding_ratio=max(0.45, float(vcfg.hand_tracking.roi_padding_ratio)),
            roi_target_size=vcfg.hand_tracking.roi_target_size,
            pinch_enter_ratio=0.32,
            pinch_exit_ratio=0.42,
            palm_facing_min_z=vcfg.gesture_3d.palm_facing_min_z,
            grace_period_sec=max(0.14, vcfg.gesture_3d.grace_period_ms / 1000.0),
            voting_window_size=vcfg.gesture_3d.voting_window_size,
            voting_min_votes=vcfg.gesture_3d.voting_min_votes,
            one_euro_min_cutoff=0.8,
            one_euro_beta=0.04,
            one_euro_d_cutoff=vcfg.gesture_3d.one_euro_d_cutoff,
            min_hold_sec=0.120,
            model_complexity=dyn_complexity,
        )
        if not self.running:
            pose_estimator.close()
            hand_engine.close()
            face_extractor.close()
            return
        if not hand_engine._has_detector():
            _LOGGER.error("CameraHandWorker: HandGestureEngine failed init: %s", hand_engine.init_error)
            self.frame_ready.emit({
                "ok": False,
                "paused": False,
                "cam_idx": self.cam_idx,
                "fps": 0.0,
                "qimage": None,
                "raw_bgr": None,
                "lighting": {"mean_luma": 0.0, "laplacian_var": 0.0, "condition": "ERR_MODEL"},
                "error_msg": hand_engine.init_error or "Lỗi khởi tạo MediaPipe HandLandmarker",
                "gesture": "IDLE",
                "pose_tag": "IDLE",
                "cursor_norm": None,
                "nav_event": None,
            })

        self._capture_thread = CameraCaptureThread(self)
        self._capture_thread.start()

        prev_gray: Optional[np.ndarray] = None
        last_time = time.perf_counter()
        smoothed_fps = 0.0
        last_processed_seq = -1
        profile_ctrl = AutoProfileController(
            initial_profile=PerformanceProfile.BALANCED,
            auto_adapt=True,
            downgrade_threshold_ms=32.0,
        )
        scheduler = StaggeredInferenceScheduler()
        dual_coord = DualHandGestureCoordinator(primary_pref=self.primary_hand_pref)
        nav_detector = SlideNavigationGestureDetector(
            swipe_cooldown_sec=0.65,
            vertical_cooldown_sec=0.45,
        )
        cached_face_info: Dict[str, Any] = {
            "face": None,
            "face_boxes": [],
            "face_eyes": None,
            "face_5pt": None,
            "face_emb_512": None,
        }
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
                    frame = self._latest_frame
                    last_processed_seq = self._frame_seq

                self._is_processing = True
                try:
                    t_cv_start = time.perf_counter()
                    now_mono = time.monotonic()
                    if self.perf_mode == "AUTO":
                        profile_ctrl.auto_adapt = True
                    else:
                        try:
                            target_prof = PerformanceProfile(self.perf_mode)
                            if profile_ctrl.active_profile != target_prof or profile_ctrl.auto_adapt:
                                profile_ctrl.set_manual_profile(target_prof, auto_adapt=False)
                        except ValueError:
                            profile_ctrl.auto_adapt = True

                    preset = profile_ctrl.preset
                    exposure_ctrl.check_interval_n = preset.exposure_interval_n
                    teacher_tracker.set_enabled(self.teacher_lock_enabled)
                    dual_coord.set_primary_preference(self.primary_hand_pref)

                    dt = max(1e-3, t_cv_start - last_time)
                    last_time = t_cv_start
                    inst_fps = min(120.0, 1.0 / dt)
                    smoothed_fps = inst_fps if smoothed_fps <= 0.1 else (0.8 * smoothed_fps + 0.2 * inst_fps)

                    bgr = cv2.flip(frame, 1)
                    h, w = bgr.shape[:2]
                    calib_now = bool(self._calibrate_requested)
                    if self.auto_hdr_backlight:
                        if preset.enable_full_frame_clahe or (w <= 640 and h <= 480):
                            bgr, light_diag = exposure_ctrl.process(
                                bgr, is_rgb=False, enabled=True, force_check=calib_now
                            )
                        else:
                            # Fast Core i3 path: evaluate luminance & LAB-CLAHE on 640x360 work frame,
                            # then apply zero-alloc Gamma LUT onto HD frame when low-light/backlit
                            cw_s = min(w, 640)
                            ch_s = max(180, int(round(h * (cw_s / float(max(1, w))))))
                            small_bgr = cv2.resize(bgr, (cw_s, ch_s), interpolation=cv2.INTER_AREA)
                            _, light_diag = exposure_ctrl.process(
                                small_bgr, is_rgb=False, enabled=True, force_check=calib_now
                            )
                            if not light_diag.clahe_bypassed and abs(light_diag.gamma_applied - 1.0) > 0.02:
                                bgr = cv2.LUT(bgr, build_gamma_lut(light_diag.gamma_applied))
                        light_metrics = light_diag.to_dict()
                    else:
                        light_diag = None
                        light_metrics = {"lighting_mode": "NORMAL", "mean_luma": 128.0, "laplacian_variance": 50.0, "clahe_bypassed": True}

                    # Strided liveness diff only needed on Face-ID AuthGate (skip in Studio to boost hand FPS)
                    if self.auth_mode:
                        gray_sub = cv2.cvtColor(bgr[::4, ::4], cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None and prev_gray.shape == gray_sub.shape:
                            liveness_diff = float(np.mean(cv2.absdiff(gray_sub, prev_gray))) / 12.0
                            liveness_diff = float(np.clip(liveness_diff, 0.0, 1.0))
                        else:
                            liveness_diff = 0.0
                        prev_gray = gray_sub
                    else:
                        liveness_diff = 0.0
                        prev_gray = None

                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                    has_locked = teacher_tracker.locked_track_id is not None
                    teacher_lost_sec = (now_mono - last_teacher_seen_ts) if has_locked else 999.0
                    run_pose, run_face = scheduler.next_tick(
                        now_mono,
                        preset,
                        has_locked_teacher=has_locked,
                        teacher_lost_sec=teacher_lost_sec,
                    )
                    # In Studio mode with Teacher Lock disabled, skip Face & Pose nets to maximize hand FPS
                    if not self.auth_mode and not self.teacher_lock_enabled:
                        run_face = False
                        run_pose = False

                    if run_face or run_pose:
                        sw, sh = preset.search_size
                        aux_rgb = (
                            np.ascontiguousarray(cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA), dtype=np.uint8)
                            if (w > sw or h > sh)
                            else rgb
                        )
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

                    # Fail-Open 2-Hand Tracking (Adaptive Stage-1 Search + Full-Res Stage-2 Velocity-Predicted ROI)
                    scan_wh = (
                        max(320, int(preset.search_size[0])),
                        max(180, int(preset.search_size[1])),
                    )
                    left_hand, right_hand = hand_engine.detect_hands(
                        rgb,
                        face_boxes,
                        teacher_tracker=teacher_tracker,
                        teacher_snapshot=subj_snap,
                        wrist_roi_hints=wrist_hints,
                        search_size=scan_wh,
                        roi_target_size=preset.roi_size,
                        enable_roi_sharpen=preset.enable_roi_sharpen,
                    )
                    dual_state = dual_coord.update(left_hand, right_hand, t_sec=now_mono)

                    cv_latency_ms = (time.perf_counter() - t_cv_start) * 1000.0
                    active_prof = profile_ctrl.record_inference_ms(cv_latency_ms)

                    # Downscale Studio preview BEFORE drawing overlays & QImage copy (cuts 4x-16x pixel overhead)
                    needs_overlay = bool(self.debug_hud_enabled or self.skeleton_overlay_enabled or face_bbox is not None)
                    if not self.auth_mode and w > 360:
                        disp_w = 360
                        disp_h = max(180, int(round(h * (disp_w / float(max(1, w))))))
                        display_bgr = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    else:
                        disp_w, disp_h = w, h
                        display_bgr = bgr.copy() if needs_overlay else bgr
                    font_scale_f = max(0.34, min(0.48, disp_w / 760.0))

                    if self.debug_hud_enabled:
                        if subj_snap.locked_torso_bbox and not subj_snap.bypass_active:
                            tx, ty, tw, th = subj_snap.locked_torso_bbox
                            cv2.rectangle(
                                display_bgr,
                                (int(tx * disp_w), int(ty * disp_h)),
                                (int((tx + tw) * disp_w), int((ty + th) * disp_h)),
                                (34, 197, 94),
                                2,
                                cv2.LINE_AA,
                            )
                            tid_lbl = f"TEACHER #{subj_snap.locked_teacher_id} ({subj_snap.mode})"
                            cv2.putText(
                                display_bgr,
                                tid_lbl,
                                (int(tx * disp_w) + 4, max(14, int(ty * disp_h) + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale_f,
                                (34, 197, 94),
                                1,
                                cv2.LINE_AA,
                            )
                        for ig_box in subj_snap.ignored_face_bboxes:
                            ix, iy, iw, ih = ig_box
                            cv2.rectangle(
                                display_bgr,
                                (int(ix * disp_w), int(iy * disp_h)),
                                (int((ix + iw) * disp_w), int((iy + ih) * disp_h)),
                                (68, 68, 239),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                display_bgr,
                                "IGNORED",
                                (int(ix * disp_w) + 2, max(12, int(iy * disp_h) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale_f,
                                (68, 68, 239),
                                1,
                                cv2.LINE_AA,
                            )
                        for wx, wy in subj_snap.locked_wrist_anchors:
                            cv2.circle(display_bgr, (int(wx * disp_w), int(wy * disp_h)), 4, (255, 220, 0), 2, cv2.LINE_AA)
                    elif face_bbox is not None:
                        fx, fy, fw, fh = face_bbox
                        cv2.rectangle(
                            display_bgr,
                            (int(fx * disp_w), int(fy * disp_h)),
                            (int((fx + fw) * disp_w), int((fy + fh) * disp_h)),
                            (212, 245, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    # Raw & Accepted 21-point Skeleton Overlay for BOTH hands (proof of MediaPipe detection)
                    if self.skeleton_overlay_enabled or self.debug_hud_enabled:
                        bone_thick = 1 if disp_w <= 380 else 2
                        tip_r = 3 if disp_w <= 380 else 5
                        joint_r = 2 if disp_w <= 380 else 3
                        for raw_h in hand_engine.last_raw_hands:
                            pts = raw_h.get("lm")
                            if not pts:
                                continue
                            accepted = bool(raw_h.get("accepted", True))
                            bone_col = (212, 245, 0) if accepted else (0, 140, 255)
                            joint_col = (64, 228, 254) if accepted else (0, 90, 255)
                            for i, j in HAND_BONE_PAIRS:
                                p1 = (int(pts[i][0] * disp_w), int(pts[i][1] * disp_h))
                                p2 = (int(pts[j][0] * disp_w), int(pts[j][1] * disp_h))
                                cv2.line(display_bgr, p1, p2, bone_col, bone_thick, cv2.LINE_AA)
                            for idx, pt in enumerate(pts):
                                r = tip_r if idx in (4, 8, 12, 16, 20) else joint_r
                                cv2.circle(
                                    display_bgr,
                                    (int(pt[0] * disp_w), int(pt[1] * disp_h)),
                                    r,
                                    joint_col,
                                    -1,
                                    cv2.LINE_AA,
                                )

                        active_pair: List[Tuple[int, str, Optional[Dict[str, object]], Tuple[int, int, int]]] = []
                        hand_idx = 1
                        if dual_state.primary_hand is not None:
                            p_side_name = "Right" if dual_state.primary_side == "right" else "Left"
                            active_pair.append((hand_idx, p_side_name, dual_state.primary_hand, (0, 245, 212)))
                            hand_idx += 1
                        if dual_state.secondary_hand is not None:
                            s_side_name = "Left" if dual_state.primary_side == "right" else "Right"
                            active_pair.append((hand_idx, s_side_name, dual_state.secondary_hand, (255, 220, 0)))

                        for h_num, side_title, h_obj, col in active_pair:
                            if not h_obj or "lm" not in h_obj:
                                continue
                            wpt = h_obj["lm"][0]
                            raw_pose = str(h_obj.get("pose_tag", "HOVER"))
                            if dual_state.two_hand_mode == "DUAL_PINCH_ZOOM":
                                tag_disp = "ZOOM"
                            elif dual_state.two_hand_mode == "DUAL_PALM_PAN":
                                tag_disp = "PAN"
                            elif h_num == 2 and dual_state.modifier_mode != "NONE":
                                tag_disp = dual_state.modifier_mode
                            elif raw_pose == "INDEX":
                                tag_disp = "DRAW"
                            elif raw_pose == "PINCH":
                                tag_disp = "GRAB"
                            elif raw_pose == "TWO_FINGERS":
                                tag_disp = "POINT"
                            elif raw_pose == "OPEN_PALM":
                                tag_disp = "ERASE"
                            else:
                                tag_disp = raw_pose
                            lbl = f"H{h_num} {side_title} {tag_disp}"
                            cv2.putText(
                                display_bgr,
                                lbl,
                                (max(4, int(wpt[0] * disp_w) - 28), max(14, int(wpt[1] * disp_h) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale_f,
                                col,
                                1,
                                cv2.LINE_AA,
                            )

                        # Draw Active Workspace Margin Guide (12%..88% -> 0%..100% screen corners)
                        mx0, my0 = int(0.12 * disp_w), int(0.12 * disp_h)
                        mx1, my1 = int(0.88 * disp_w), int(0.88 * disp_h)
                        cv2.rectangle(display_bgr, (mx0, my0), (mx1, my1), (79, 70, 229), 1, cv2.LINE_AA)

                        n_hands_active = int(left_hand is not None) + int(right_hand is not None)
                        overlay_txt = f"FPS:{smoothed_fps:.0f} | L:{cv_latency_ms:.0f}ms | H:{n_hands_active}/2"
                        cv2.rectangle(display_bgr, (4, 4), (min(disp_w - 4, 240), 22), (15, 23, 42), -1)
                        cv2.putText(
                            display_bgr,
                            overlay_txt,
                            (8, 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale_f,
                            (0, 245, 212),
                            1,
                            cv2.LINE_AA,
                        )

                    gesture = "IDLE"
                    pose_tag = "IDLE"
                    pinch_ratio = None
                    cursor_norm = None
                    hand_span = 0.20
                    primary = dual_state.primary_hand
                    if primary is not None:
                        gesture = str(primary.get("gesture", "HOVER"))
                        pose_tag = str(primary.get("pose_tag", gesture))
                        pinch_ratio = primary.get("pinch_ratio")
                        hand_span = float(primary.get("hand_span") or 0.20)
                        cn = primary.get("cursor_norm")
                        if cn is not None and len(cn) >= 2:
                            cursor_norm = (float(cn[0]), float(cn[1]))
                        self._was_pinched = pose_tag == "PINCH"
                    else:
                        self._was_pinched = False

                    nav_info = nav_detector.update(
                        primary,
                        suppress=dual_state.suppress_primary_drawing,
                        is_drawing=self._was_pinched,
                        t_sec=now_mono,
                    )
                    if nav_info.get("swipe_direction"):
                        self._last_swipe_dir = str(nav_info["swipe_direction"])
                        self._last_swipe_until = now_mono + 0.65

                    if (self.skeleton_overlay_enabled or self.debug_hud_enabled) and now_mono <= self._last_swipe_until and self._last_swipe_dir:
                        cx_a, cy_a = disp_w // 2, disp_h // 2
                        a_sc = max(0.5, disp_w / 640.0)
                        d_vec = {
                            "LEFT": (-int(75 * a_sc), 0, "<<< NEXT"),
                            "RIGHT": (int(75 * a_sc), 0, "PREV >>>"),
                            "UP": (0, -int(65 * a_sc), "SCROLL DOWN"),
                            "DOWN": (0, int(65 * a_sc), "SCROLL UP"),
                        }.get(self._last_swipe_dir)
                        if d_vec is not None:
                            cv2.arrowedLine(
                                display_bgr,
                                (cx_a - d_vec[0] // 2, cy_a - d_vec[1] // 2),
                                (cx_a + d_vec[0], cy_a + d_vec[1]),
                                (0, 245, 212),
                                3,
                                cv2.LINE_AA,
                                tipLength=0.35,
                            )
                            cv2.putText(
                                display_bgr,
                                d_vec[2],
                                (max(8, cx_a - int(56 * a_sc)), max(28, cy_a - int(36 * a_sc))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.44, 0.58 * a_sc),
                                (250, 204, 21),
                                2,
                                cv2.LINE_AA,
                            )

                    if calib_now:
                        self._calibrate_requested = False
                        mean_luma_c = float(light_metrics.get("mean_luma") or 120.0)
                        lap_c = float(light_metrics.get("laplacian_variance") or 50.0)
                        palm_s = 0.0
                        if primary is not None and isinstance(primary.get("lm"), list) and len(primary["lm"]) >= 10:
                            p0 = primary["lm"][0]
                            p9 = primary["lm"][9]
                            palm_s = math.hypot(float(p0[0]) - float(p9[0]), float(p0[1]) - float(p9[1]))
                        if palm_s > 0.015:
                            est_dist_m = round(max(0.4, min(3.5, 0.22 / palm_s)), 2)
                        elif face_bbox is not None and face_bbox[2] > 0.02:
                            est_dist_m = round(max(0.4, min(3.5, 0.24 / float(face_bbox[2]))), 2)
                        else:
                            est_dist_m = 1.80

                        exposure_ctrl.low_light_mean_luma = 84.0 if mean_luma_c < 95.0 else 72.0
                        if est_dist_m >= 2.0:
                            hand_engine.min_detection_confidence = 0.30
                            hand_engine.roi_expand_factor = 2.4
                            hand_engine.roi_unsharp_strength = 0.85
                            new_cutoff = 1.25
                        else:
                            hand_engine.min_detection_confidence = 0.35
                            hand_engine.roi_expand_factor = 2.2
                            hand_engine.roi_unsharp_strength = 0.65
                            new_cutoff = 1.50
                        hand_engine.left_fsm.cursor_filter.fx.min_cutoff = new_cutoff
                        hand_engine.left_fsm.cursor_filter.fy.min_cutoff = new_cutoff
                        hand_engine.right_fsm.cursor_filter.fx.min_cutoff = new_cutoff
                        hand_engine.right_fsm.cursor_filter.fy.min_cutoff = new_cutoff

                        self.calibration_done.emit({
                            "ok": True,
                            "mean_luma": round(mean_luma_c, 1),
                            "laplacian_var": round(lap_c, 1),
                            "est_distance_m": est_dist_m,
                            "lighting_mode": str(light_metrics.get("lighting_mode", "NORMAL")),
                            "min_conf": hand_engine.min_detection_confidence,
                            "roi_expand": hand_engine.roi_expand_factor,
                            "message_vi": (
                                f"Đã cân chỉnh: Ánh sáng {mean_luma_c:.0f} ({light_metrics.get('lighting_mode', 'NORMAL')}) • "
                                f"Cự ly ~{est_dist_m:.1f}m • Conf={hand_engine.min_detection_confidence:.2f}"
                            ),
                        })

                    left_pan_active = dual_state.two_hand_mode == "DUAL_PALM_PAN"

                    disp_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                    qimg = QImage(
                        disp_rgb.data, disp_w, disp_h, disp_rgb.strides[0], QImage.Format_RGB888
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
                        "raw_hands_count": len(hand_engine.last_raw_hands),
                        "active_hands_count": int(left_hand is not None) + int(right_hand is not None),
                        "rejected_by_lock": hand_engine.last_rejected_by_teacher_lock,
                        "roi_mode": hand_engine.last_tracking_mode,
                        "perf_profile": active_prof.value,
                        "finger_states": (primary or {}).get("finger_states", ["CURLED"] * 5),
                        "finger_angles_deg": (primary or {}).get("finger_angles_deg", [0.0] * 5),
                        "pinch_ratio": pinch_ratio,
                        "palm_facing_ok": (primary or {}).get("palm_facing_ok", True),
                        "palm_facing_score": (primary or {}).get("palm_facing_score", 0.0),
                        "stage2_roi_zoom": bool((primary or {}).get("stage2_roi_zoom", False)),
                        "kalman_predicted": bool((primary or {}).get("kalman_predicted", False)),
                        "fsm_state": (primary or {}).get("fsm_state", "IDLE"),
                    }

                    telemetry_pkt = HandTelemetryPacket(
                        timestamp_sec=time.perf_counter(),
                        cursor_norm=cursor_norm,
                        gesture=gesture,
                        pose_tag=pose_tag,
                        pinch_ratio=float(pinch_ratio) if pinch_ratio is not None else None,
                        hand_span=hand_span,
                        two_hand_mode=dual_state.two_hand_mode,
                        modifier_mode=dual_state.modifier_mode,
                        nav_event=nav_info.get("nav_event"),  # type: ignore[arg-type]
                        swipe_direction=nav_info.get("swipe_direction"),  # type: ignore[arg-type]
                        scroll_delta_norm=float(nav_info.get("scroll_delta_norm") or 0.0),
                        extrapolated=False,
                        cam_fps=self._cam_fps_smoothed or smoothed_fps,
                        ai_fps=smoothed_fps,
                    )
                    self.telemetry_ready.emit(telemetry_pkt)

                    self.frame_ready.emit({
                        "ok": True,
                        "paused": False,
                        "cam_idx": self.cam_idx,
                        "fps": smoothed_fps,
                        "cam_fps": self._cam_fps_smoothed or smoothed_fps,
                        "qimage": qimg,
                        "raw_bgr": bgr if self.auth_mode else None,
                        "lighting": light_metrics,
                        "face_bbox": face_bbox,
                        "face_5pt": cached_face_info.get("face_5pt"),
                        "face_emb_512": cached_face_info.get("face_emb_512"),
                        "liveness_diff": liveness_diff,
                        "gesture": gesture,
                        "pose_tag": pose_tag,
                        "pinch_ratio": pinch_ratio,
                        "hand_span": hand_span,
                        "cursor_norm": cursor_norm,
                        "nav_event": nav_info.get("nav_event"),
                        "swipe_direction": nav_info.get("swipe_direction"),
                        "scroll_delta_norm": float(nav_info.get("scroll_delta_norm") or 0.0),
                        "peace_hold_progress": nav_info.get("peace_hold_progress", 0.0),
                        "question_hold_progress": nav_info.get("question_hold_progress", 0.0),
                        "left_hand": left_hand,
                        "right_hand": right_hand,
                        "raw_hands": hand_engine.last_raw_hands,
                        "dual_hand": {
                            "primary_side": dual_state.primary_side,
                            "two_hand_mode": dual_state.two_hand_mode,
                            "modifier_mode": dual_state.modifier_mode,
                            "zoom_factor": dual_state.zoom_factor,
                            "zoom_center_norm": dual_state.zoom_center_norm,
                            "pan_delta_norm": dual_state.pan_delta_norm,
                            "offhand_command": dual_state.offhand_command,
                            "suppress_primary_drawing": dual_state.suppress_primary_drawing,
                            "status_label_vi": dual_state.status_label_vi,
                        },
                        "left_pan_active": left_pan_active,
                        "vision_telemetry": primary_telemetry,
                        "telemetry_packet": telemetry_pkt,
                    })
                except Exception as exc:
                    _LOGGER.exception("CameraHandWorker frame error: %s", exc)
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


class CameraHandWorker(VisionInferenceWorker):
    """Backward-compatible alias for `VisionInferenceWorker` (decoupled capture + inference)."""


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
        _LOGGER.warning("Failed saving AI cache: %s", exc)


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
            _LOGGER.warning("Structured JSON parse fallback to raw text: %s", exc)
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

        self.capture_face_btn = QPushButton("Chụp Mẫu Khuôn Mặt 5-Góc (0/5 mẫu)")
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

        self.su_complete_btn = QPushButton("Hoàn Tất Đăng Ký — Lưu Hồ Sơ Cục Bộ")
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

        si_pin_lbl = QLabel(
            f"Nhập mã PIN đã đăng ký (hoặc bật Camera quét Face ID Cosine >= {DEFAULT_ARCFACE_THRESHOLD:.2f}):"
        )
        si_pin_lbl.setWordWrap(True)
        si_layout.addWidget(si_pin_lbl)
        self.si_pin_edit = QLineEdit()
        self.si_pin_edit.setEchoMode(QLineEdit.Password)
        self.si_pin_edit.setPlaceholderText("Nhập mã PIN 4-6 số...")
        self.si_pin_edit.returnPressed.connect(self._on_signin_with_pin)
        si_layout.addWidget(self.si_pin_edit)

        self.si_login_btn = QPushButton("ĐĂNG NHẬP BẰNG MÃ PIN — Vào Studio Giảng Dạy")
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
        self.status_lbl.setWordWrap(True)
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
                    _LOGGER.warning("Face ID cosine template check failed: %s", exc)

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
        self.capture_face_btn.setText("Chụp Mẫu Khuôn Mặt 5-Góc (0/5 mẫu)")
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
# MODULE 2.3 & MODULE 4: CLEAN CENTER SMARTBOARD CANVAS + PRESENTER + MINI HUD
# =============================================================================
class DocumentImportWorker(QThread):
    """Non-blocking background worker importing PDF / PPTX / PPT into cached high-DPI LectureDocument."""

    finished_doc = Signal(object, str)

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    def run(self) -> None:
        try:
            doc = LectureDocument.open(self.file_path)
            doc.prewarm_around(0, radius=2)
            self.finished_doc.emit(doc, "")
        except Exception as exc:
            _LOGGER.exception("DocumentImportWorker failed (%s): %s", self.file_path, exc)
            self.finished_doc.emit(None, str(exc))


class VoiceQuestionWorker(QThread):
    """Capture short microphone question and transcribe offline/online safely for Slide AI."""

    finished_transcript = Signal(str, str)

    def __init__(self, duration_sec: float = 4.0) -> None:
        super().__init__()
        self.duration_sec = max(1.5, min(8.0, float(duration_sec)))

    def run(self) -> None:
        if not _ensure_sounddevice() or sd is None:
            self.finished_transcript.emit("", "Micro chưa khả dụng trên máy này.")
            return
        try:
            sr = 16000
            frames = int(sr * self.duration_sec)
            rec = sd.rec(frames, samplerate=sr, channels=1, dtype="int16")
            sd.wait()
            raw_pcm = rec.tobytes()
            try:
                import speech_recognition as sr_lib  # type: ignore

                recognizer = sr_lib.Recognizer()
                audio_data = sr_lib.AudioData(raw_pcm, sr, 2)
                text = recognizer.recognize_google(audio_data, language="vi-VN")
                self.finished_transcript.emit(text.strip(), "")
                return
            except Exception as rec_exc:
                self.finished_transcript.emit("", f"Không nhận dạng được giọng nói ({rec_exc}). Hãy gõ câu hỏi hoặc nhấn Giải Slide AI.")
        except Exception as exc:
            self.finished_transcript.emit("", f"Lỗi thu âm micro: {exc}")


class MiniCameraHudOverlay(QFrame):
    """Toggleable top-right floating Mini Camera HUD with live skeleton, gesture badge, FPS & 1-click Auto-Calibrate."""

    calibrate_clicked = Signal()
    gesture_map_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(224)
        self.setStyleSheet(
            f"background: rgba(15, 23, 42, 228); border: 1.5px solid {SECONDARY_TEAL}; border-radius: 10px;"
        )
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(7, 6, 7, 7)
        vbox.setSpacing(5)

        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        self.badge_lbl = QLabel("POINTER")
        self.badge_lbl.setAlignment(Qt.AlignCenter)
        self.badge_lbl.setFixedHeight(20)
        self.badge_lbl.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 5px; padding: 1px 6px; font-size: 10px; font-weight: 800;"
        )
        hdr.addWidget(self.badge_lbl, stretch=1)

        self.fps_lbl = QLabel("0 FPS")
        self.fps_lbl.setStyleSheet(
            f"color: {NEON_CYAN}; font-size: 10px; font-weight: 800; border: none; background: transparent;"
        )
        hdr.addWidget(self.fps_lbl)

        hide_btn = QPushButton("✕")
        hide_btn.setFixedSize(18, 18)
        hide_btn.setToolTip("Ẩn Mini Camera HUD (Phím H)")
        hide_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_MUTED}; border: none; border-radius: 4px; font-size: 10px; font-weight: 700;"
        )
        hide_btn.clicked.connect(self.close_clicked.emit)
        hdr.addWidget(hide_btn)
        vbox.addLayout(hdr)

        self.cam_lbl = QLabel("Chế độ Chuột / Cảm ứng\n(Chưa có Webcam)")
        self.cam_lbl.setFixedSize(208, 118)
        self.cam_lbl.setAlignment(Qt.AlignCenter)
        self.cam_lbl.setWordWrap(True)
        self.cam_lbl.setStyleSheet(
            f"background: {BG_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 10px; font-weight: 600;"
        )
        vbox.addWidget(self.cam_lbl)

        act_row = QHBoxLayout()
        act_row.setSpacing(4)
        self.calib_btn = QPushButton("Cân chỉnh Sáng")
        self.calib_btn.setFixedHeight(24)
        self.calib_btn.setToolTip("Tự động đo độ sáng và cự ly 1.5m–3.5m để tối ưu nhận diện bàn tay")
        self.calib_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {BRIGHT_YELLOW}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 10px; font-weight: 700; padding: 0 5px;"
        )
        self.calib_btn.clicked.connect(self.calibrate_clicked.emit)
        act_row.addWidget(self.calib_btn, stretch=1)
        vbox.addLayout(act_row)

        self.gesture_map_btn = QPushButton("Sơ đồ Cử chỉ")
        self.gesture_map_btn.setFixedHeight(24)
        self.gesture_map_btn.setToolTip("Mở Sơ đồ Cử chỉ trực quan 2 cột (1 Tay & 2 Tay) kèm thanh Test thử")
        self.gesture_map_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: 1px solid {NEON_CYAN}; border-radius: 6px; font-size: 10px; font-weight: 700; padding: 0 5px;"
        )
        self.gesture_map_btn.clicked.connect(self.gesture_map_clicked.emit)
        vbox.addWidget(self.gesture_map_btn)

        self.info_lbl = QLabel("Cử chỉ: Chờ tay • Chuột sẵn sàng")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 9.5px; border: none; background: transparent;"
        )
        vbox.addWidget(self.info_lbl)
        self.adjustSize()

    def update_hud(
        self,
        qimg: Optional[QImage],
        *,
        badge: str,
        fps: float,
        info_text: str = "",
        paused: bool = False,
    ) -> None:
        badge_upper = (badge or "POINTER").upper()
        self.badge_lbl.setText(badge_upper)
        if badge_upper in ("DRAWING", "HIGHLIGHTER"):
            bg_c, fg_c = NEON_CYAN, BG_DARK
        elif badge_upper in ("ERASER", "LOCKED"):
            bg_c, fg_c = ALERT_ROSE, TEXT_LIGHT
        elif badge_upper in ("NEXT SLIDE", "PREV SLIDE", "SCROLL DOWN", "SCROLL UP", "AI SLIDE", "ZOOM / PAN"):
            bg_c, fg_c = BRIGHT_YELLOW, BG_DARK
        else:
            bg_c, fg_c = PRIMARY_INDIGO, TEXT_LIGHT
        self.badge_lbl.setStyleSheet(
            f"background: {bg_c}; color: {fg_c}; border-radius: 5px; padding: 1px 6px; font-size: 10px; font-weight: 800;"
        )
        self.fps_lbl.setText(f"{fps:.0f} FPS" if fps > 0.5 else "Mouse")
        if info_text:
            self.info_lbl.setText(info_text)

        if paused:
            self.cam_lbl.setPixmap(QPixmap())
            self.cam_lbl.setText("Camera đang tạm tắt\n🖱 Chuột / Cảm ứng sẵn sàng")
        elif qimg is not None:
            scaled = qimg.scaled(198, 124, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam_lbl.setPixmap(QPixmap.fromImage(scaled))
        else:
            self.cam_lbl.setPixmap(QPixmap())
            self.cam_lbl.setText("🖱 Chế độ Chuột / Cảm ứng\n(Không có Webcam)")


class SlideThumbnailStrip(QFrame):
    """Collapsible horizontal slide thumbnail navigator with active indicator & ink markers."""

    slide_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(92)
        self.setStyleSheet(
            f"background: {SURFACE_DARK}; border-top: 1px solid {BORDER_SLATE};"
        )
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(6)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("background: transparent; border: none;")

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.hbox = QHBoxLayout(self.container)
        self.hbox.setContentsMargins(2, 2, 2, 2)
        self.hbox.setSpacing(6)
        self.hbox.addStretch(1)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll)

    def rebuild(
        self,
        slides: Sequence[Dict[str, Any]],
        current_idx: int,
        slide_annotations: Dict[int, Dict[str, Any]],
        lecture_doc: Optional[LectureDocument] = None,
    ) -> None:
        while self.hbox.count() > 0:
            item = self.hbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for idx, sl in enumerate(slides):
            btn = QPushButton()
            btn.setFixedSize(116, 76)
            btn.setCursor(Qt.PointingHandCursor)
            is_cur = idx == current_idx
            layer = slide_annotations.get(idx) or {}
            obj_cnt = len(layer.get("strokes") or []) + len(layer.get("annotations") or []) + len(layer.get("images") or [])
            has_ink = obj_cnt > 0

            pm: Optional[QPixmap] = None
            if lecture_doc is not None and not sl.get("is_tutorial") and idx < lecture_doc.page_count:
                qi = lecture_doc.render_thumbnail_qimage(idx, max_w=120)
                if not qi.isNull():
                    pm = QPixmap.fromImage(qi)
            elif isinstance(sl.get("pixmap"), QPixmap) and not sl["pixmap"].isNull():
                pm = sl["pixmap"].scaled(108, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            card_lay = QVBoxLayout(btn)
            card_lay.setContentsMargins(3, 3, 3, 2)
            card_lay.setSpacing(2)

            thumb_lbl = QLabel()
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_lbl.setFixedHeight(50)
            thumb_lbl.setStyleSheet("background: #0F172A; border-radius: 4px; border: none;")
            if pm is not None:
                thumb_lbl.setPixmap(pm.scaled(108, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumb_lbl.setText(str(sl.get("title") or f"Slide {idx + 1}")[:18])
                thumb_lbl.setStyleSheet(f"background: #0F172A; color: {TEXT_MUTED}; font-size: 9px; border-radius: 4px;")
            card_lay.addWidget(thumb_lbl)

            cap_txt = f"Trang {idx + 1}" + (f" ●{obj_cnt}" if has_ink else "")
            cap_lbl = QLabel(cap_txt)
            cap_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl.setStyleSheet(
                f"color: {NEON_CYAN if is_cur else TEXT_LIGHT}; font-size: 10px; font-weight: {'800' if is_cur else '600'}; background: transparent; border: none;"
            )
            card_lay.addWidget(cap_lbl)

            border_col = NEON_CYAN if is_cur else BORDER_SLATE
            bg_col = "#1E293B" if is_cur else BG_DARK
            btn.setStyleSheet(
                f"QPushButton {{ background: {bg_col}; border: 2px solid {border_col}; border-radius: 7px; padding: 0px; }}"
                f"QPushButton:hover {{ border: 2px solid {SECONDARY_TEAL}; }}"
            )
            btn.clicked.connect(lambda _=False, i=idx: self.slide_selected.emit(i))
            self.hbox.addWidget(btn)

        self.hbox.addStretch(1)


class EngineWarmupWorker(QThread):
    """Non-blocking startup warmer loading PyMuPDF, GenAI SDK & Matplotlib after UI paints."""

    warmup_done = Signal(dict)

    def run(self) -> None:
        info: Dict[str, bool] = {}
        if self.isInterruptionRequested():
            return
        try:
            info.update(warmup_document_engines())
        except Exception:
            info["pymupdf"] = False
        if self.isInterruptionRequested():
            return
        try:
            from cv_engine.gemini_client import _ensure_genai_sdk

            info["genai"] = bool(_ensure_genai_sdk())
        except Exception:
            info["genai"] = False
        if self.isInterruptionRequested():
            return
        try:
            from cv_engine.output_sanitizer import _ensure_matplotlib

            info["matplotlib"] = bool(_ensure_matplotlib())
        except Exception:
            info["matplotlib"] = False
        if not self.isInterruptionRequested():
            self.warmup_done.emit(info)


class BoardNavMiniHud(QFrame):
    """Compact corner Mini-HUD showing Zoom %, Board Coords (X,Y), Page X/Y, Fit/Reset & Pan Lock."""

    fit_width_clicked = Signal()
    fit_page_clicked = Signal()
    fit_content_clicked = Signal()
    reset_view_clicked = Signal()
    pan_lock_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dark_mode = True
        self._apply_hud_style(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(5)

        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setStyleSheet(f"color: {NEON_CYAN}; font-weight: 800; font-size: 11px;")
        self.coord_lbl = QLabel("X:0 Y:0")
        self.coord_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600; font-size: 10.5px;")
        self.page_lbl = QLabel("Trang 1/1")
        self.page_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700; font-size: 11px;")

        btn_c = QPushButton("Vừa Nội Dung")
        btn_c.setToolTip("Thu phóng & căn giữa toàn bộ nội dung trên bảng (Fit to Content)")
        btn_c.clicked.connect(self.fit_content_clicked.emit)

        btn_p = QPushButton("Vừa Trang")
        btn_p.setToolTip("Thu phóng vừa khít toàn trang hiện tại")
        btn_p.clicked.connect(self.fit_page_clicked.emit)

        btn_r = QPushButton("100%")
        btn_r.setToolTip("Đặt lại thu phóng 100% và căn giữa bảng (Ctrl+0)")
        btn_r.clicked.connect(self.reset_view_clicked.emit)

        self.pan_lock_btn = QPushButton("🔓 Pan")
        self.pan_lock_btn.setCheckable(True)
        self.pan_lock_btn.setChecked(False)
        self.pan_lock_btn.setToolTip("Khóa / mở khóa di chuyển bảng (chống trượt nhầm khi đang viết)")
        self.pan_lock_btn.toggled.connect(self._on_pan_lock_toggled)

        lay.addWidget(self.zoom_lbl)
        lay.addWidget(self.coord_lbl)
        lay.addWidget(self.page_lbl)
        lay.addWidget(btn_c)
        lay.addWidget(btn_p)
        lay.addWidget(btn_r)
        lay.addWidget(self.pan_lock_btn)
        for btn_w in self.findChildren(QPushButton):
            btn_w.setFocusPolicy(Qt.NoFocus)
        self.adjustSize()

    def _apply_hud_style(self, dark: bool) -> None:
        self._dark_mode = bool(dark)
        if dark:
            self.setStyleSheet(
                f"QFrame {{ background: rgba(15, 23, 42, 232); border: 1px solid {BORDER_SLATE}; border-radius: 10px; }}"
                f"QLabel {{ color: {TEXT_LIGHT}; background: transparent; border: none; font-size: 11px; font-weight: 700; }}"
                f"QPushButton {{ background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {BORDER_SLATE}; "
                f"border-radius: 6px; padding: 3px 7px; font-size: 10.5px; font-weight: 700; }}"
                f"QPushButton:hover {{ border-color: {NEON_CYAN}; color: {TEXT_LIGHT}; }}"
                f"QPushButton:checked {{ background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-color: {ALERT_ROSE}; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame {{ background: rgba(255, 255, 255, 240); border: 1px solid #CBD5E1; border-radius: 10px; }}"
                f"QLabel {{ color: #0F172A; background: transparent; border: none; font-size: 11px; font-weight: 700; }}"
                f"QPushButton {{ background: #F1F5F9; color: {PRIMARY_INDIGO}; border: 1px solid #CBD5E1; "
                f"border-radius: 6px; padding: 3px 7px; font-size: 10.5px; font-weight: 700; }}"
                f"QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; color: #0F172A; }}"
                f"QPushButton:checked {{ background: {ALERT_ROSE}; color: #FFFFFF; border-color: {ALERT_ROSE}; }}"
            )

    def set_dark_mode(self, dark: bool) -> None:
        self._apply_hud_style(dark)
        self.zoom_lbl.setStyleSheet(
            f"color: {NEON_CYAN if dark else PRIMARY_INDIGO}; font-weight: 800; font-size: 11px;"
        )
        muted = TEXT_MUTED if dark else "#475569"
        self.coord_lbl.setStyleSheet(f"color: {muted}; font-weight: 600; font-size: 10.5px;")
        self.page_lbl.setStyleSheet(f"color: {muted}; font-weight: 700; font-size: 11px;")

    def _on_pan_lock_toggled(self, locked: bool) -> None:
        self.pan_lock_btn.setText("🔒 Khóa" if locked else "🔓 Pan")
        self.pan_lock_toggled.emit(bool(locked))
        self.adjustSize()

    def update_state(
        self,
        zoom_scale: float,
        page_idx: int,
        page_count: int,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        pan_locked: bool = False,
    ) -> None:
        pct = int(round(float(zoom_scale) * 100.0))
        self.zoom_lbl.setText(f"{pct}%")
        z = max(0.20, float(zoom_scale))
        bx = int(round(-float(pan_x) / z))
        by = int(round(-float(pan_y) / z))
        self.coord_lbl.setText(f"X:{bx} Y:{by}")
        total = max(1, int(page_count))
        cur = max(1, min(total, int(page_idx) + 1))
        self.page_lbl.setText(f"Trang {cur}/{total}")
        if self.pan_lock_btn.isChecked() != bool(pan_locked):
            self.pan_lock_btn.blockSignals(True)
            self.pan_lock_btn.setChecked(bool(pan_locked))
            self.pan_lock_btn.setText("🔒 Khóa" if pan_locked else "🔓 Pan")
            self.pan_lock_btn.blockSignals(False)
        self.adjustSize()


class QuickTipBanner(QFrame):
    """Compact dismissible first-launch quick tips pill on canvas."""

    guide_requested = Signal()
    dismissed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: rgba(15, 23, 42, 232); border: 1px solid {PRIMARY_INDIGO}; border-radius: 10px; }}"
            f"QLabel {{ color: {TEXT_LIGHT}; background: transparent; border: none; font-size: 11px; }}"
            f"QPushButton {{ background: {SURFACE_DARK}; color: {NEON_CYAN}; border: 1px solid {BORDER_SLATE}; "
            f"border-radius: 6px; padding: 3px 8px; font-size: 10.5px; font-weight: 700; }}"
            f"QPushButton:hover {{ border-color: {NEON_CYAN}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 8, 6)
        lay.setSpacing(8)
        lbl = QLabel(
            "<b>Mẹo nhanh:</b> Giơ <b>1 ngón trỏ</b> để vẽ · Chụm <b>Cái+Trỏ</b> kéo Object · Giữ <b>Chuột phải</b> mở Bảng Tròn · Cuộn chuột Zoom"
        )
        guide_btn = QPushButton("? Chi tiết")
        guide_btn.clicked.connect(self.guide_requested.emit)
        close_btn = QPushButton("Đã hiểu ✕")
        close_btn.clicked.connect(self._on_dismiss)
        lay.addWidget(lbl)
        lay.addWidget(guide_btn)
        lay.addWidget(close_btn)
        self.adjustSize()

    def _on_dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class SmartboardCanvas(QWidget):
    region_cropped_for_ai = Signal(bytes, QPixmap)
    lasso_action_requested = Signal(str, object)
    hud_status_changed = Signal(str, float, str)
    slides_changed = Signal()
    file_dropped = Signal(str)
    pie_tool_selected = Signal(str)
    guide_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dark_mode = True
        self.grid_enabled = True
        self.slide_visible = True
        self.smart_shape_enabled = True
        self.hold_to_snap_350ms = False
        self.active_tool = "PEN"  # PEN, HIGHLIGHTER, ERASER, SELECT, TEXT, STICKY, SHAPE, LASER, PAN, SELECT_REGION, LASSO
        self.active_shape_kind: str = "rect"
        self.sticky_color_idx: int = 0
        self.ink_color = QColor(NEON_CYAN)
        self.highlighter_color = QColor(250, 204, 21, 96)
        self.pen_thickness = 4

        self.drawing_fsm = ZeroStrayStrokeDrawingFsm(
            pinch_enter=0.32,
            pinch_exit=0.42,
            tail_trim_ms=100.0,
            allow_index_point_draw=True,
        )
        self.hud_mode: DrawingHudMode = DrawingHudMode.HOVER
        self.live_gesture_hud_text: str = ""
        self.live_gesture_hud_accent: str = NEON_CYAN
        self.cooldown_remaining_ms: float = 0.0
        self.cancel_hold_progress: float = 0.0
        self.dynamic_eraser_radius: float = 45.0

        # Pan, Zoom (25%-400%) & Continuous Vertical Document Scroll State
        self.zoom_scale: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.pan_locked: bool = False
        self.continuous_scroll_mode: bool = False
        self.doc_scroll_y: float = 0.0
        self.active_modifier_mode: str = "NONE"
        self.two_hand_status_text: str = ""
        self._lasso_dwell_ready: bool = False
        self._space_panning: bool = False
        self._mouse_pan_last: Optional[QPointF] = None
        self._img_drag_state: Optional[Dict[str, Any]] = None
        self._selected_board_image_idx: Optional[int] = None
        self._obj_interaction: Optional[Dict[str, Any]] = None
        self._marquee_rect: Optional[QRectF] = None
        self._shape_preview_rect: Optional[QRectF] = None
        self._cv_grab_obj_id: Optional[str] = None
        self._cv_grab_last_pt: Optional[Tuple[float, float]] = None

        self.lecture_doc: Optional[LectureDocument] = None
        self.slide_annotations: Dict[int, Dict[str, List[Any]]] = {}
        self.current_stroke: List[QPointF] = []
        self.laser_trail: deque = deque(maxlen=48)
        self._stroke_start_ms: float = 0.0
        self._last_motion_ms: float = 0.0
        self._last_motion_pt: Optional[QPointF] = None
        self._hold_snap_ready: bool = True

        # Non-destructive Auto-Shape notification & Gesture action toast pills
        self.shape_toast: Optional[Dict[str, Any]] = None
        self.toast_revert_rect: Optional[QRectF] = None
        self.gesture_toast: Optional[Dict[str, Any]] = None

        self.cursor_pos: Optional[QPointF] = None
        self._prev_cursor_pos: Optional[QPointF] = None
        self.cursor_mode: str = "HOVER"  # HOVER, DRAWING, LOCKED_COOLDOWN, PAN_ERASE, LASER
        self.selection_start: Optional[QPointF] = None
        self.selection_end: Optional[QPointF] = None
        self.lasso_points: List[QPointF] = []
        self.lasso_region: Optional[LassoSelectionRegion] = None

        self._dirty_tracker = DirtyCanvasBackingTracker()
        self._backing_pixmap: Optional[QPixmap] = None
        self._grid_tile_cache: Dict[Tuple[bool, float], QPixmap] = {}
        self._scaled_page_pixmap_cache: OrderedDict[Tuple[int, int, int, int], QPixmap] = OrderedDict()
        self._scene_rev: int = 0

        # Neutral clean guide slide only (zero personal sample data)
        self.slides: List[Dict[str, Any]] = [
            {
                "title": "DNT - Victer Guide • Hướng dẫn nhanh AirTeach AI",
                "subtitle": f"Bảng trắng không chạm & Trình chiếu Word / PDF / PPTX / Ảnh + {DEFAULT_GEMINI_MODEL}",
                "bullets": [
                    "• 1 tay: Giơ 1 ngón trỏ để vẽ · 2 ngón di con trỏ · Chụm Cái+Trỏ kéo Object · Xòe 5 ngón tẩy · Vẫy trái/phải đổi Slide.",
                    "• 2 tay: Chụm 2 tay phóng to/thu nhỏ 25–400% · Xòe 2 tay (hoặc nắm 2 tay) kéo bảng / cuộn tài liệu.",
                    "• Chuột phải (Nhấn giữ → kéo → thả): Mở Bảng Tròn 8 công cụ · Cuộn chuột để Zoom · Ctrl+0: Về Giữa.",
                ],
                "formula": "Ctrl+Z: Hoàn tác  |  F1: Hướng dẫn chi tiết  |  Space+Kéo / Chuột giữa: Di chuyển bảng",
                "pixmap": None,
                "is_tutorial": True,
            }
        ]
        self.current_slide_idx = 0
        self._ensure_slide_layer(0)

        # Top-Left buttons: Toggle visibility + Remove clean guide card
        self.toggle_slide_btn = QPushButton("Ẩn Slide -> Bảng trống", self)
        self.toggle_slide_btn.setGeometry(14, 10, 168, 28)
        self.toggle_slide_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {BORDER_SLATE}; border-radius: 7px; font-weight: 700; font-size: 11px;"
        )
        self.toggle_slide_btn.clicked.connect(self.toggle_slide_visibility)

        self.clear_tutorial_btn = QPushButton("Xóa hướng dẫn", self)
        self.clear_tutorial_btn.setGeometry(188, 10, 114, 28)
        self.clear_tutorial_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; border-radius: 7px; font-weight: 600; font-size: 11px;"
        )
        self.clear_tutorial_btn.clicked.connect(self.dismiss_tutorial_slides)

        # Toggleable Top-Right Corner Mini Camera HUD & Auto-Calibration
        self.mini_hud = MiniCameraHudOverlay(self)
        self.mini_hud.setVisible(True)

        # Bottom-Right Board Navigation Mini-HUD (Zoom %, Coords, Page X/Y, Fit Content, Fit Page, 100%, Pan Lock)
        self.nav_hud = BoardNavMiniHud(self)
        self.nav_hud.fit_width_clicked.connect(self.fit_to_width)
        self.nav_hud.fit_page_clicked.connect(self.fit_to_page)
        self.nav_hud.fit_content_clicked.connect(self.fit_to_content)
        self.nav_hud.reset_view_clicked.connect(self.reset_viewport)
        self.nav_hud.pan_lock_toggled.connect(self.set_pan_locked)
        self.nav_hud.update_state(self.zoom_scale, self.current_slide_idx, len(self.slides), self.pan_x, self.pan_y, self.pan_locked)

        # Bottom-Left Quick Tip Banner (dismissible on first launch)
        self._quick_tip_user_dismissed: bool = False
        self.quick_tip_banner = QuickTipBanner(self)
        self.quick_tip_banner.guide_requested.connect(self.guide_requested.emit)
        try:
            _cfg_tip = load_ui_api_settings()
            if str(_cfg_tip.get("quick_tip_dismissed", "")).lower() in ("true", "1"):
                self._quick_tip_user_dismissed = True
                self.quick_tip_banner.setVisible(False)
        except Exception as exc:
            _LOGGER.debug("Cannot read quick_tip_dismissed preference: %s", exc)
        self.quick_tip_banner.dismissed.connect(self._persist_quick_tip_dismissed)

        # Right-Click Blender/Maya-style Radial Pie Menu (8 ergonomic directional slices)
        self.pie_menu = RadialPieMenuOverlay(self)
        self.pie_menu.action_triggered.connect(self._on_pie_menu_action)

        self._reposition_overlays()

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

        # Floating Object Action Bar (shown when 1+ CanvasObjects are selected)
        self.obj_action_bar = QFrame(self)
        self.obj_action_bar.setStyleSheet(
            f"QFrame {{ background: rgba(15, 23, 42, 240); border: 1.5px solid {NEON_CYAN}; border-radius: 9px; }}"
            f"QPushButton {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; "
            f"border-radius: 6px; padding: 3px 8px; font-weight: 700; font-size: 10.5px; }}"
            f"QPushButton:hover {{ border-color: {NEON_CYAN}; }}"
        )
        oab_lay = QHBoxLayout(self.obj_action_bar)
        oab_lay.setContentsMargins(6, 4, 6, 4)
        oab_lay.setSpacing(4)
        self.obj_edit_text_btn = QPushButton("✎ Sửa chữ")
        self.obj_edit_text_btn.clicked.connect(self._on_edit_selected_text_dialog)
        oab_lay.addWidget(self.obj_edit_text_btn)

        self._obj_recolor_buttons: Dict[str, QPushButton] = {}
        for rc_hex, rc_name in (
            ("#00F5D4", "Xanh Neon"),
            ("#FEE440", "Vàng Sáng"),
            ("#F43F5E", "Đỏ Hồng"),
            ("#A855F7", "Tím"),
        ):
            rc_btn = QPushButton("")
            rc_btn.setFixedSize(18, 18)
            rc_btn.setCursor(Qt.PointingHandCursor)
            rc_btn.setToolTip(f"Đổi màu đối tượng đang chọn sang {rc_name} ({rc_hex})")
            rc_btn.setStyleSheet(
                f"QPushButton {{ background: {rc_hex}; border: 1.5px solid {TEXT_LIGHT}; "
                f"border-radius: 9px; padding: 0px; }}"
                f"QPushButton:hover {{ border: 2px solid #FFFFFF; }}"
            )
            rc_btn.clicked.connect(lambda _=False, hc=rc_hex: self.recolor_selected_objects(hc))
            oab_lay.addWidget(rc_btn)
            self._obj_recolor_buttons[rc_hex] = rc_btn

        btn_dup = QPushButton("⧉ Nhân bản")
        btn_dup.setToolTip("Nhân bản đối tượng đang chọn (Ctrl+D)")
        btn_dup.clicked.connect(self.duplicate_selected_objects)
        oab_lay.addWidget(btn_dup)

        btn_front = QPushButton("▲ Lên trên")
        btn_front.setToolTip("Đưa đối tượng lên lớp trên cùng")
        btn_front.clicked.connect(self.bring_selected_to_front)
        oab_lay.addWidget(btn_front)

        btn_back = QPushButton("▼ Xuống dưới")
        btn_back.setToolTip("Đưa đối tượng xuống lớp dưới cùng")
        btn_back.clicked.connect(self.send_selected_to_back)
        oab_lay.addWidget(btn_back)

        btn_del = QPushButton("🗑 Xóa")
        btn_del.setToolTip("Xóa đối tượng đang chọn (Delete)")
        btn_del.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border: none; border-radius: 6px; padding: 3px 8px; font-weight: 800; font-size: 10.5px;"
        )
        btn_del.clicked.connect(self.delete_selected_objects)
        oab_lay.addWidget(btn_del)
        for btn_w in self.obj_action_bar.findChildren(QPushButton):
            btn_w.setFocusPolicy(Qt.NoFocus)
        self.obj_action_bar.adjustSize()
        self.obj_action_bar.setVisible(False)

    def _ensure_slide_layer(self, slide_idx: int) -> Dict[str, Any]:
        idx = max(0, int(slide_idx))
        layer = self.slide_annotations.get(idx)
        if layer is None:
            layer = {
                "strokes": [],
                "redo_strokes": [],
                "annotations": [],
                "images": [],
                "ai_action_undo_stack": [],
                "scene": SlideSceneGraph(layer=idx),
            }
            self.slide_annotations[idx] = layer
        else:
            if "images" not in layer:
                layer["images"] = []
            if "scene" not in layer or not isinstance(layer["scene"], SlideSceneGraph):
                layer["scene"] = SlideSceneGraph(layer=idx)
        return layer

    @property
    def scene(self) -> SlideSceneGraph:
        return self._ensure_slide_layer(self.current_slide_idx)["scene"]

    @staticmethod
    def _stroke_dict_to_object(s: Dict[str, Any], layer_idx: int, z_index: int) -> CanvasObject:
        oid = str(s.get("_obj_id") or new_object_id("str"))
        s["_obj_id"] = oid
        pts_xy = [(float(p.x()), float(p.y())) if isinstance(p, QPointF) else (float(p[0]), float(p[1])) for p in (s.get("points") or [])]
        raw_xy = [(float(p.x()), float(p.y())) if isinstance(p, QPointF) else (float(p[0]), float(p[1])) for p in (s.get("raw_points") or pts_xy)]
        bx, by, bw, bh = compute_points_bbox(pts_xy, min_size=18.0)
        col = s.get("color", NEON_CYAN)
        col_hex = col.name(QColor.HexArgb) if isinstance(col, QColor) else str(col)
        is_beaut = bool(s.get("is_beautified", False))
        st = str(s.get("shape_type") or "FREEHAND")
        otype = "shape" if (is_beaut and st not in ("FREEHAND", "HIGHLIGHTER")) else "stroke"
        return CanvasObject(
            id=oid,
            obj_type=otype,
            x=float(s.get("x", bx)),
            y=float(s.get("y", by)),
            w=max(16.0, float(s.get("w", bw))),
            h=max(16.0, float(s.get("h", bh))),
            rotation=float(s.get("rotation", 0.0)),
            z_index=int(s.get("z_index") or z_index),
            layer=layer_idx,
            points=pts_xy,
            raw_points=raw_xy,
            color=col_hex,
            thickness=float(s.get("thickness", 4.0)),
            is_highlighter=bool(s.get("is_highlighter", False)),
            is_beautified=is_beaut,
            shape_type=st,
            shape_label_vi=str(s.get("shape_label_vi") or ""),
            confidence=float(s.get("confidence", 0.0)),
            is_ai=bool(s.get("is_ai", False)),
        )

    @staticmethod
    def _ann_dict_to_object(ann: Dict[str, Any], layer_idx: int, z_index: int) -> CanvasObject:
        otype = "sticky" if ann.get("obj_type") == "sticky" else "text"
        oid = str(ann.get("_obj_id") or new_object_id("stk" if otype == "sticky" else "txt"))
        ann["_obj_id"] = oid
        return CanvasObject(
            id=oid,
            obj_type=otype,
            x=float(ann.get("x", 40.0)),
            y=float(ann.get("y", 60.0)),
            w=max(40.0, float(ann.get("w", 200.0))),
            h=max(32.0, float(ann.get("h", 48.0))),
            rotation=float(ann.get("rotation", 0.0)),
            z_index=int(ann.get("z_index") or z_index),
            layer=layer_idx,
            text=str(ann.get("text") or ""),
            raw_text=str(ann.get("raw") or ann.get("text") or ""),
            font_size=int(ann.get("font_size") or 14),
            color=str(ann.get("color") or NEON_CYAN),
            bg_color=str(ann.get("bg_color") or ""),
            border_color=str(ann.get("border_color") or ""),
            reveal_until_ms=float(ann.get("reveal_until_ms", 0.0)),
            pixmap=ann.get("pixmap"),
            is_ai=bool(ann.get("is_ai", False)),
        )

    @staticmethod
    def _img_dict_to_object(img: Dict[str, Any], layer_idx: int, z_index: int) -> CanvasObject:
        oid = str(img.get("_obj_id") or new_object_id("img"))
        img["_obj_id"] = oid
        return CanvasObject(
            id=oid,
            obj_type="image",
            x=float(img.get("x", 80.0)),
            y=float(img.get("y", 80.0)),
            w=max(48.0, float(img.get("w", 240.0))),
            h=max(48.0, float(img.get("h", 180.0))),
            rotation=float(img.get("rotation", 0.0)),
            z_index=int(img.get("z_index") or z_index),
            layer=layer_idx,
            pixmap=img.get("pixmap"),
            name=str(img.get("name") or ""),
            source_path=str(img.get("source_path") or ""),
        )

    def _sync_legacy_into_scene(self, slide_idx: Optional[int] = None) -> SlideSceneGraph:
        """Ensure any strokes/annotations/images appended via legacy lists exist as CanvasObjects."""
        idx = self.current_slide_idx if slide_idx is None else max(0, int(slide_idx))
        layer = self._ensure_slide_layer(idx)
        sc: SlideSceneGraph = layer["scene"]
        existing = {o.id: o for o in sc.objects}

        for img in layer.get("images") or []:
            oid = img.get("_obj_id")
            if not oid or oid not in existing:
                obj = self._img_dict_to_object(img, idx, sc.next_z_index())
                sc.add_object(obj, record_undo=False, select=False)
                existing[obj.id] = obj
            else:
                # Sync in-place dict mutations (e.g., move_board_image)
                obj = existing[oid]
                obj.x = float(img.get("x", obj.x))
                obj.y = float(img.get("y", obj.y))
                obj.w = max(48.0, float(img.get("w", obj.w)))
                obj.h = max(48.0, float(img.get("h", obj.h)))
                obj.rotation = float(img.get("rotation", obj.rotation))

        for s in layer.get("strokes") or []:
            oid = s.get("_obj_id")
            if not oid or oid not in existing:
                obj = self._stroke_dict_to_object(s, idx, sc.next_z_index())
                sc.add_object(obj, record_undo=False, select=False)
                existing[obj.id] = obj
            else:
                # Keep progressive AI stroke reveal synced
                obj = existing[oid]
                pts = s.get("points") or []
                if len(pts) != len(obj.points):
                    obj.points = [(float(p.x()), float(p.y())) if isinstance(p, QPointF) else (float(p[0]), float(p[1])) for p in pts]
                    bx, by, bw, bh = compute_points_bbox(obj.points, min_size=18.0)
                    obj.x, obj.y, obj.w, obj.h = bx, by, bw, bh

        for ann in layer.get("annotations") or []:
            oid = ann.get("_obj_id")
            if not oid or oid not in existing:
                obj = self._ann_dict_to_object(ann, idx, sc.next_z_index())
                sc.add_object(obj, record_undo=False, select=False)
                existing[obj.id] = obj

        return sc

    def _sync_scene_to_legacy(self, slide_idx: Optional[int] = None) -> None:
        """Rebuild legacy lists in-place from SlideSceneGraph so all callers & tests stay consistent."""
        idx = self.current_slide_idx if slide_idx is None else max(0, int(slide_idx))
        layer = self._ensure_slide_layer(idx)
        sc: SlideSceneGraph = layer["scene"]
        new_strokes: List[Dict[str, Any]] = []
        new_anns: List[Dict[str, Any]] = []
        new_imgs: List[Dict[str, Any]] = []

        for obj in sc.ordered_objects():
            if obj.obj_type in ("stroke", "shape"):
                col = QColor(obj.color)
                if not col.isValid():
                    col = QColor(NEON_CYAN)
                if obj.is_highlighter and col.alpha() > 120:
                    col.setAlpha(96)
                new_strokes.append({
                    "_obj_id": obj.id,
                    "points": [QPointF(px, py) for (px, py) in obj.points],
                    "raw_points": [QPointF(px, py) for (px, py) in (obj.raw_points or obj.points)],
                    "shape_type": obj.shape_type,
                    "shape_label_vi": obj.shape_label_vi,
                    "confidence": obj.confidence,
                    "is_beautified": obj.is_beautified,
                    "is_highlighter": obj.is_highlighter,
                    "is_ai": obj.is_ai,
                    "color": col,
                    "thickness": obj.thickness,
                    "x": obj.x,
                    "y": obj.y,
                    "w": obj.w,
                    "h": obj.h,
                    "rotation": obj.rotation,
                    "z_index": obj.z_index,
                })
            elif obj.obj_type in ("text", "sticky"):
                new_anns.append({
                    "_obj_id": obj.id,
                    "obj_type": obj.obj_type,
                    "raw": obj.raw_text or obj.text,
                    "text": obj.text,
                    "format": "latex" if (obj.pixmap is not None) else "text",
                    "x": obj.x,
                    "y": obj.y,
                    "w": obj.w,
                    "h": obj.h,
                    "rotation": obj.rotation,
                    "color": obj.color,
                    "bg_color": obj.bg_color,
                    "border_color": obj.border_color,
                    "font_size": obj.font_size,
                    "pixmap": obj.pixmap,
                    "reveal_until_ms": obj.reveal_until_ms,
                    "is_ai": obj.is_ai,
                    "z_index": obj.z_index,
                })
            elif obj.obj_type == "image":
                new_imgs.append({
                    "_obj_id": obj.id,
                    "x": obj.x,
                    "y": obj.y,
                    "w": obj.w,
                    "h": obj.h,
                    "rotation": obj.rotation,
                    "pixmap": obj.pixmap,
                    "name": obj.name,
                    "source_path": obj.source_path,
                    "z_index": obj.z_index,
                })

        layer["strokes"][:] = new_strokes
        layer["annotations"][:] = new_anns
        layer["images"][:] = new_imgs

    @property
    def strokes(self) -> List[Dict[str, Any]]:
        return self._ensure_slide_layer(self.current_slide_idx)["strokes"]

    @strokes.setter
    def strokes(self, value: List[Dict[str, Any]]) -> None:
        self._ensure_slide_layer(self.current_slide_idx)["strokes"] = list(value)

    @property
    def redo_strokes(self) -> List[Dict[str, Any]]:
        return self._ensure_slide_layer(self.current_slide_idx)["redo_strokes"]

    @redo_strokes.setter
    def redo_strokes(self, value: List[Dict[str, Any]]) -> None:
        self._ensure_slide_layer(self.current_slide_idx)["redo_strokes"] = list(value)

    @property
    def annotations(self) -> List[Dict[str, Any]]:
        return self._ensure_slide_layer(self.current_slide_idx)["annotations"]

    @annotations.setter
    def annotations(self, value: List[Dict[str, Any]]) -> None:
        self._ensure_slide_layer(self.current_slide_idx)["annotations"] = list(value)

    @property
    def board_images(self) -> List[Dict[str, Any]]:
        return self._ensure_slide_layer(self.current_slide_idx)["images"]

    @board_images.setter
    def board_images(self, value: List[Dict[str, Any]]) -> None:
        self._ensure_slide_layer(self.current_slide_idx)["images"] = list(value)

    @property
    def ai_action_undo_stack(self) -> List[Dict[str, Any]]:
        return self._ensure_slide_layer(self.current_slide_idx)["ai_action_undo_stack"]

    @ai_action_undo_stack.setter
    def ai_action_undo_stack(self, value: List[Dict[str, Any]]) -> None:
        self._ensure_slide_layer(self.current_slide_idx)["ai_action_undo_stack"] = list(value)

    def _persist_quick_tip_dismissed(self) -> None:
        self._quick_tip_user_dismissed = True
        try:
            save_ui_api_settings(quick_tip_dismissed=True)
        except Exception as exc:
            _LOGGER.warning("Cannot persist quick_tip_dismissed preference: %s", exc)

    def _refresh_nav_hud(self) -> None:
        if hasattr(self, "nav_hud") and self.nav_hud is not None:
            self.nav_hud.update_state(
                self.zoom_scale,
                self.current_slide_idx,
                len(self.slides),
                self.pan_x,
                self.pan_y,
                self.pan_locked,
            )
            self.nav_hud.adjustSize()
            nx = max(12, self.width() - self.nav_hud.width() - 14)
            ny = max(50, self.height() - self.nav_hud.height() - 12)
            self.nav_hud.move(nx, ny)
            self.nav_hud.raise_()
        self._refresh_obj_action_bar()

    def _refresh_obj_action_bar(self) -> None:
        if not hasattr(self, "obj_action_bar") or self.obj_action_bar is None:
            return
        sc = self.scene
        sel = sc.selected_objects()
        if not sel or self.continuous_scroll_mode or self._obj_interaction is not None:
            self.obj_action_bar.setVisible(False)
            return
        bounds = sc.selection_world_bounds()
        if bounds is None:
            self.obj_action_bar.setVisible(False)
            return
        bx, by, bw, bh = bounds
        is_single_text = len(sel) == 1 and sel[0].obj_type in ("text", "sticky")
        self.obj_edit_text_btn.setVisible(is_single_text)
        self.obj_action_bar.adjustSize()
        bar_w = self.obj_action_bar.width()
        bar_h = self.obj_action_bar.height()
        top_center_scr = self._board_to_screen(QPointF(bx + bw * 0.5, by))
        bot_center_scr = self._board_to_screen(QPointF(bx + bw * 0.5, by + bh))
        ox = int(max(10.0, min(max(10.0, self.width() - bar_w - 10.0), top_center_scr.x() - bar_w * 0.5)))
        oy = int(top_center_scr.y() - bar_h - 38.0)
        if oy < 44:
            oy = int(min(max(44.0, self.height() - bar_h - 56.0), bot_center_scr.y() + 14.0))
        self.obj_action_bar.move(ox, oy)
        self.obj_action_bar.setVisible(True)
        self.obj_action_bar.raise_()

    def _reposition_overlays(self) -> None:
        if hasattr(self, "mini_hud") and self.mini_hud is not None:
            self.mini_hud.adjustSize()
            mx = max(12, self.width() - self.mini_hud.width() - 14)
            self.mini_hud.move(mx, 10)
            self.mini_hud.raise_()
        self._refresh_nav_hud()
        if hasattr(self, "quick_tip_banner") and self.quick_tip_banner is not None:
            self.quick_tip_banner.adjustSize()
            nav_w = self.nav_hud.width() if (hasattr(self, "nav_hud") and self.nav_hud is not None) else 0
            fits_without_overlap = (self.quick_tip_banner.width() + nav_w + 48) <= self.width()
            if self._quick_tip_user_dismissed or not fits_without_overlap:
                self.quick_tip_banner.setVisible(False)
            else:
                self.quick_tip_banner.setVisible(True)
                qy = max(50, self.height() - self.quick_tip_banner.height() - 12)
                self.quick_tip_banner.move(14, qy)
                self.quick_tip_banner.raise_()
        if hasattr(self, "pie_menu") and self.pie_menu is not None and self.pie_menu.isVisible():
            self.pie_menu.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlays()
        self._bump_scene()

    def focusOutEvent(self, event) -> None:
        if self._space_panning:
            self._space_panning = False
            self._mouse_pan_last = None
            self.unsetCursor()
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_panning = True
            self.setCursor(Qt.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_panning = False
            self._mouse_pan_last = None
            self.unsetCursor()
        super().keyReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    lp = Path(url.toLocalFile())
                    if lp.is_dir() and list_folder_images(lp):
                        event.acceptProposedAction()
                        return
                    ext = lp.suffix.lower()
                    if ext in SUPPORTED_DOC_EXTENSIONS or ext in SUPPORTED_IMAGE_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    lp = Path(url.toLocalFile())
                    if lp.is_dir() and list_folder_images(lp):
                        self.load_image_folder(lp)
                        event.acceptProposedAction()
                        return
                    ext = lp.suffix.lower()
                    if ext in SUPPORTED_DOC_EXTENSIONS or ext in SUPPORTED_IMAGE_EXTENSIONS:
                        self.file_dropped.emit(str(lp))
                        event.acceptProposedAction()
                        return
        event.ignore()

    def _compute_active_slide_rect(self) -> QRectF:
        if not self.slides or not (0 <= self.current_slide_idx < len(self.slides)):
            return compute_slide_rect_on_canvas((self.width(), self.height()), 16.0 / 9.0, is_tutorial=False)
        sl = self.slides[self.current_slide_idx]
        is_tut = bool(sl.get("is_tutorial", False))
        ar = float(sl.get("aspect_ratio") or (16.0 / 9.0))
        if self.lecture_doc is not None and not is_tut and self.current_slide_idx < self.lecture_doc.page_count:
            ar = self.lecture_doc.get_page_aspect_ratio(self.current_slide_idx)
        elif isinstance(sl.get("pixmap"), QPixmap) and not sl["pixmap"].isNull():
            pm = sl["pixmap"]
            ar = float(pm.width()) / max(1.0, float(pm.height()))
        return compute_slide_rect_on_canvas((self.width(), self.height()), ar, is_tutorial=is_tut)

    def attach_lecture_document(self, doc: LectureDocument) -> None:
        if self.lecture_doc is not None and self.lecture_doc is not doc:
            self.lecture_doc.close()
        self.lecture_doc = doc
        self._scaled_page_pixmap_cache.clear()
        self.slide_annotations.clear()
        self.current_stroke.clear()
        self.shape_toast = None
        self.toast_revert_rect = None
        self.hide_lasso_bar()
        self.slides = []
        for idx in range(doc.page_count):
            self.slides.append({
                "title": f"{doc.source_path.name} — Trang {idx + 1}/{doc.page_count}",
                "subtitle": f"Backend: {doc.backend_used}",
                "bullets": [],
                "formula": "",
                "pixmap": None,
                "aspect_ratio": doc.get_page_aspect_ratio(idx),
                "is_tutorial": False,
            })
            self._ensure_slide_layer(idx)
        self.current_slide_idx = 0
        self.doc_scroll_y = 0.0
        self.continuous_scroll_mode = (getattr(doc, "doc_type", "PDF") == "WORD")
        self.slide_visible = True
        self.toggle_slide_btn.setText("Ẩn Slide -> Bảng trống")
        self.clear_tutorial_btn.setVisible(False)
        doc.prewarm_around(0, radius=2, scale=2.25)
        self.reset_viewport()
        self.slides_changed.emit()
        self._bump_scene()
        self.update()

    def load_document_file(self, file_path: Path | str) -> bool:
        p = Path(file_path)
        if p.is_dir():
            return self.load_image_folder(p)
        ext = p.suffix.lower()
        if ext in SUPPORTED_DOC_EXTENSIONS:
            doc = LectureDocument.open(p)
            self.attach_lecture_document(doc)
            return True
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            qimg = load_image_file_qimage(p, target_max_dim=2048)
            pm = QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap(str(p))
            if not pm.isNull():
                if self.lecture_doc is not None:
                    self.lecture_doc.close()
                    self.lecture_doc = None
                    self.slides = []
                    self.slide_annotations.clear()
                self.slides = [s for s in self.slides if not s.get("is_tutorial", False)]
                self.slides.append({
                    "title": p.name,
                    "subtitle": "",
                    "bullets": [],
                    "formula": "",
                    "pixmap": pm,
                    "aspect_ratio": float(pm.width()) / max(1.0, float(pm.height())),
                    "is_tutorial": False,
                })
                self.current_slide_idx = len(self.slides) - 1
                self._ensure_slide_layer(self.current_slide_idx)
                self.clear_tutorial_btn.setVisible(False)
                self.slide_visible = True
                self.slides_changed.emit()
                self._refresh_nav_hud()
                self._bump_scene()
                self.update()
                return True
        return False

    def load_image_folder(self, folder_path: Path | str) -> bool:
        """Open a folder of images (`.png`, `.jpg`, `.webp`, `.bmp`, `.svg`) as a slide deck."""
        imgs = list_folder_images(folder_path)
        if not imgs:
            return False
        new_slides: List[Dict[str, Any]] = []
        for img_p in imgs:
            qimg = load_image_file_qimage(img_p, target_max_dim=1920)
            pm = QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap(str(img_p))
            if pm.isNull():
                continue
            new_slides.append({
                "title": img_p.name,
                "subtitle": "",
                "bullets": [],
                "formula": "",
                "pixmap": pm,
                "aspect_ratio": float(pm.width()) / max(1.0, float(pm.height())),
                "is_tutorial": False,
            })
        if not new_slides:
            return False
        if self.lecture_doc is not None:
            self.lecture_doc.close()
            self.lecture_doc = None
        self.slide_annotations.clear()
        self.slides = new_slides
        self.current_slide_idx = 0
        for idx in range(len(self.slides)):
            self._ensure_slide_layer(idx)
        self.clear_tutorial_btn.setVisible(False)
        self.slide_visible = True
        self.reset_viewport()
        self.slides_changed.emit()
        self.show_gesture_toast(f"🖼 Đã mở thư mục {len(new_slides)} ảnh thành Slide", duration_ms=1400.0)
        return True

    def insert_image_on_board(self, file_path: Path | str) -> bool:
        """Insert an independent movable/resizable/rotatable Image CanvasObject onto the active slide."""
        p = Path(file_path)
        if not p.exists():
            return False
        qimg = load_image_file_qimage(p, target_max_dim=1600)
        pm = QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap(str(p))
        if pm.isNull():
            return False
        cw = max(640.0, float(self.width() or 1000.0))
        ch = max(420.0, float(self.height() or 640.0))
        ar = float(pm.width()) / max(1.0, float(pm.height()))
        target_w = min(460.0, max(200.0, cw * 0.36))
        target_h = target_w / max(0.25, ar)
        if target_h > ch * 0.52:
            target_h = ch * 0.52
            target_w = target_h * max(0.25, ar)
        sc = self._sync_legacy_into_scene()
        n_existing = len(self.board_images)
        offset = float((n_existing % 5) * 28)
        center_b = self._screen_to_board(QPointF(cw * 0.5, ch * 0.5))
        obj = CanvasObject(
            id=new_object_id("img"),
            obj_type="image",
            x=center_b.x() - target_w * 0.5 + offset,
            y=center_b.y() - target_h * 0.5 + offset,
            w=target_w,
            h=target_h,
            rotation=0.0,
            z_index=sc.next_z_index(),
            layer=self.current_slide_idx,
            pixmap=pm,
            name=p.name,
            source_path=str(p),
        )
        sc.add_object(obj, record_undo=True, select=True)
        self._sync_scene_to_legacy()
        self._selected_board_image_idx = len(self.board_images) - 1
        self._bump_scene()
        self.show_gesture_toast(f"🖼 Đã chèn '{p.name}' (Chọn để di chuyển, co giãn 8 góc hoặc xoay)", duration_ms=1800.0)
        self.update()
        return True

    def move_board_image(self, idx: int, x: float, y: float) -> bool:
        self._sync_legacy_into_scene()
        if 0 <= idx < len(self.board_images):
            oid = self.board_images[idx].get("_obj_id")
            obj = self.scene.get_by_id(oid) if oid else None
            if obj is not None:
                self.scene.push_undo_snapshot()
                obj.x = float(x)
                obj.y = float(y)
                self._sync_scene_to_legacy()
            else:
                self.board_images[idx]["x"] = float(x)
                self.board_images[idx]["y"] = float(y)
            self._bump_scene()
            self.update()
            return True
        return False

    def resize_board_image(self, idx: int, w: float, h: float) -> bool:
        self._sync_legacy_into_scene()
        if 0 <= idx < len(self.board_images):
            oid = self.board_images[idx].get("_obj_id")
            obj = self.scene.get_by_id(oid) if oid else None
            if obj is not None:
                self.scene.push_undo_snapshot()
                obj.w = max(48.0, float(w))
                obj.h = max(48.0, float(h))
                self._sync_scene_to_legacy()
            else:
                self.board_images[idx]["w"] = max(48.0, float(w))
                self.board_images[idx]["h"] = max(48.0, float(h))
            self._bump_scene()
            self.update()
            return True
        return False

    def add_text_object(
        self,
        text: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        *,
        font_size: int = 16,
        color: Optional[str] = None,
        is_ai: bool = False,
        select: bool = True,
    ) -> CanvasObject:
        sc = self._sync_legacy_into_scene()
        if x is None or y is None:
            c_b = self._screen_to_board(QPointF(max(320.0, self.width() * 0.5), max(210.0, self.height() * 0.45)))
            x = c_b.x() - 180.0
            y = c_b.y() - 60.0
        fg = color or (TEXT_LIGHT if self.dark_mode else "#0F172A")
        obj = make_text_object(
            text,
            float(x),
            float(y),
            color=fg,
            border_color=NEON_CYAN if self.dark_mode else PRIMARY_INDIGO,
            font_size=font_size,
            z_index=sc.next_z_index(),
            layer=self.current_slide_idx,
            is_ai=is_ai,
        )
        sc.add_object(obj, record_undo=True, select=select)
        self._sync_scene_to_legacy()
        self.slides_changed.emit()
        self._bump_scene()
        self.update()
        return obj

    def add_sticky_object(
        self,
        text: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        *,
        palette_idx: Optional[int] = None,
        select: bool = True,
    ) -> CanvasObject:
        sc = self._sync_legacy_into_scene()
        if x is None or y is None:
            c_b = self._screen_to_board(QPointF(max(320.0, self.width() * 0.5), max(210.0, self.height() * 0.5)))
            x = c_b.x() - 110.0
            y = c_b.y() - 75.0
        p_idx = self.sticky_color_idx if palette_idx is None else int(palette_idx)
        self.sticky_color_idx = (p_idx + 1) % len(STICKY_PALETTE)
        obj = make_sticky_object(
            text,
            float(x),
            float(y),
            palette_idx=p_idx,
            z_index=sc.next_z_index(),
            layer=self.current_slide_idx,
        )
        sc.add_object(obj, record_undo=True, select=select)
        self._sync_scene_to_legacy()
        self.slides_changed.emit()
        self._bump_scene()
        self.update()
        return obj

    def add_shape_object(
        self,
        shape_kind: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        color: Optional[str] = None,
        thickness: Optional[float] = None,
        select: bool = True,
    ) -> CanvasObject:
        sc = self._sync_legacy_into_scene()
        obj = make_shape_object(
            shape_kind,
            x,
            y,
            w,
            h,
            color=color or self.ink_color.name(),
            thickness=float(self.pen_thickness if thickness is None else thickness),
            z_index=sc.next_z_index(),
            layer=self.current_slide_idx,
        )
        sc.add_object(obj, record_undo=True, select=select)
        self._sync_scene_to_legacy()
        self.slides_changed.emit()
        self._bump_scene()
        self.update()
        return obj

    def edit_object_text(self, obj_id: str, new_text: str) -> bool:
        sc = self._sync_legacy_into_scene()
        obj = sc.get_by_id(obj_id)
        if obj is None or obj.obj_type not in ("text", "sticky"):
            return False
        clean = str(new_text or "").strip()
        if not clean:
            return False
        sc.push_undo_snapshot()
        obj.text = clean
        obj.raw_text = clean
        obj.pixmap = None
        lines = clean.splitlines() or [clean]
        if obj.obj_type == "text":
            max_chars = max(len(ln) for ln in lines)
            obj.w = max(obj.w, min(680.0, max_chars * (obj.font_size * 0.62) + 32.0))
            chars_per_line = max(14, int((obj.w - 24.0) / max(6.0, obj.font_size * 0.58)))
            total_lines = sum(max(1, math.ceil(len(ln) / float(chars_per_line))) for ln in lines)
            obj.h = max(48.0, min(600.0, total_lines * (obj.font_size * 1.48) + 22.0))
        self._sync_scene_to_legacy()
        self._bump_scene()
        self.update()
        return True

    def _on_edit_selected_text_dialog(self) -> None:
        sel = self.scene.selected_objects()
        if len(sel) != 1 or sel[0].obj_type not in ("text", "sticky"):
            return
        obj = sel[0]
        new_txt, ok = QInputDialog.getMultiLineText(
            self,
            "Chỉnh sửa nội dung" if obj.obj_type == "text" else "Chỉnh sửa Ghi chú (Sticky Note)",
            "Nội dung:",
            obj.text,
        )
        if ok and new_txt.strip():
            self.edit_object_text(obj.id, new_txt)

    def select_object_at(self, board_x: float, board_y: float, *, additive: bool = False) -> Optional[CanvasObject]:
        sc = self._sync_legacy_into_scene()
        hit = sc.hit_test(board_x, board_y, tolerance=max(10.0, 12.0 / max(0.25, self.zoom_scale)))
        if hit is None:
            if not additive:
                sc.clear_selection()
        elif additive:
            sc.toggle_select(hit.id)
        else:
            sc.select_only(hit.id)
        self._bump_scene()
        self.update()
        return hit

    def duplicate_selected_objects(self) -> List[CanvasObject]:
        sc = self._sync_legacy_into_scene()
        clones = sc.duplicate_selected(offset_x=26.0, offset_y=26.0)
        if clones:
            self._sync_scene_to_legacy()
            self.slides_changed.emit()
            self.show_gesture_toast(f"⧉ Đã nhân bản {len(clones)} đối tượng (Ctrl+D)", duration_ms=900.0)
            self._bump_scene()
            self.update()
        return clones

    def delete_selected_objects(self) -> int:
        sc = self._sync_legacy_into_scene()
        count = sc.delete_selected()
        if count > 0:
            self._selected_board_image_idx = None
            self._sync_scene_to_legacy()
            self.slides_changed.emit()
            self.show_gesture_toast(f"🗑 Đã xóa {count} đối tượng", duration_ms=850.0)
            self._bump_scene()
            self.update()
        return count

    def recolor_selected_objects(self, color_hex: str) -> int:
        sc = self._sync_legacy_into_scene()
        count = sc.recolor_selected(color_hex)
        if count > 0:
            self._sync_scene_to_legacy()
            self.slides_changed.emit()
            self.show_gesture_toast(f"🎨 Đã đổi màu {count} đối tượng ({color_hex})", duration_ms=900.0)
            self._bump_scene()
            self.update()
        return count

    def bring_selected_to_front(self) -> bool:
        sc = self._sync_legacy_into_scene()
        if sc.bring_selected_to_front():
            self._sync_scene_to_legacy()
            self.show_gesture_toast("▲ Đưa lên lớp trên cùng", duration_ms=800.0)
            self._bump_scene()
            self.update()
            return True
        return False

    def send_selected_to_back(self) -> bool:
        sc = self._sync_legacy_into_scene()
        if sc.send_selected_to_back():
            self._sync_scene_to_legacy()
            self.show_gesture_toast("▼ Đưa xuống lớp dưới cùng", duration_ms=800.0)
            self._bump_scene()
            self.update()
            return True
        return False

    def add_blank_page(self) -> int:
        """Append a fresh blank whiteboard page and switch directly to it."""
        new_idx = len(self.slides)
        self.slides.append({
            "title": f"Trang Trắng {new_idx + 1}",
            "subtitle": "",
            "bullets": [],
            "formula": "",
            "pixmap": None,
            "aspect_ratio": 16.0 / 9.0,
            "is_tutorial": False,
            "is_blank": True,
        })
        self._ensure_slide_layer(new_idx)
        self.set_current_slide(new_idx)
        self.show_gesture_toast(f"📄 Đã thêm Trang mới ({new_idx + 1}/{len(self.slides)})", duration_ms=1100.0)
        return new_idx

    def set_pan_locked(self, locked: bool) -> None:
        self.pan_locked = bool(locked)
        self._refresh_nav_hud()
        self.show_gesture_toast(
            "🔒 Đã khóa di chuyển bảng" if self.pan_locked else "🔓 Đã mở khóa di chuyển bảng",
            duration_ms=950.0,
        )

    def set_current_slide(self, idx: int) -> bool:
        if not self.slides:
            return False
        clamped = max(0, min(len(self.slides) - 1, int(idx)))
        if clamped == self.current_slide_idx:
            return False
        self.current_stroke.clear()
        self.shape_toast = None
        self.toast_revert_rect = None
        self._selected_board_image_idx = None
        self._obj_interaction = None
        self._marquee_rect = None
        self._shape_preview_rect = None
        self.scene.clear_selection()
        self.hide_lasso_bar()
        self.current_slide_idx = clamped
        self._ensure_slide_layer(clamped)
        if self.continuous_scroll_mode and self.lecture_doc is not None:
            for p_i, cont_r, _canon_r, _sc in self._compute_continuous_page_rects():
                if p_i == clamped:
                    self.doc_scroll_y = max(0.0, cont_r.top() - 48.0)
                    break
        if self.lecture_doc is not None and clamped < self.lecture_doc.page_count:
            self.lecture_doc.prewarm_around(clamped, radius=2, scale=2.25)
        self.slides_changed.emit()
        self._refresh_nav_hud()
        self._bump_scene()
        self.update()
        return True

    def show_gesture_toast(self, text_vi: str, duration_ms: float = 1150.0) -> None:
        self.gesture_toast = {
            "text": str(text_vi),
            "until_ms": self._now_ms() + max(250.0, float(duration_ms)),
        }
        self.update()

    def _get_page_qpixmap(self, page_idx: int, target_w: int, target_h: int) -> Optional[QPixmap]:
        if self.lecture_doc is None or not (0 <= page_idx < self.lecture_doc.page_count):
            return None
        tw = max(64, int(round(target_w)))
        th = max(64, int(round(target_h)))
        # Discrete LOD bucket so panning NEVER triggers page re-rasterization
        max_dim = max(tw, th)
        lod_dim = max(768, min(3840, ((max_dim + 511) // 512) * 512))
        ar_bucket = int(round((tw / max(1, th)) * 100.0))
        doc_rev = id(self.lecture_doc)
        key = (doc_rev, page_idx, lod_dim, ar_bucket)
        cached = self._scaled_page_pixmap_cache.get(key)
        if cached is not None and not cached.isNull():
            self._scaled_page_pixmap_cache.move_to_end(key)
            return cached
        render_scale = max(2.25, min(4.0, float(lod_dim) / 640.0))
        qimg = self.lecture_doc.render_page_qimage(page_idx, scale=render_scale)
        if qimg.isNull():
            return None
        lod_w = lod_dim if tw >= th else max(64, int(round(lod_dim * (tw / max(1, th)))))
        lod_h = lod_dim if th > tw else max(64, int(round(lod_dim * (th / max(1, tw)))))
        scaled_img = qimg.scaled(lod_w, lod_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm = QPixmap.fromImage(scaled_img)
        self._scaled_page_pixmap_cache[key] = pm
        self._scaled_page_pixmap_cache.move_to_end(key)
        while len(self._scaled_page_pixmap_cache) > 36:
            self._scaled_page_pixmap_cache.popitem(last=False)
        return pm

    def _compute_continuous_page_rects(self) -> List[Tuple[int, QRectF, QRectF, float]]:
        """Compute vertical stack of document pages `(page_idx, cont_rect, canon_rect, scale)`."""
        cw = max(320.0, float(self.width()))
        ch = max(240.0, float(self.height()))
        n_pages = self.lecture_doc.page_count if self.lecture_doc is not None else len(self.slides)
        if n_pages <= 0:
            return []
        avail_w = max(280.0, min(980.0, cw - 64.0))
        y_cursor = 48.0
        gap = 24.0
        out: List[Tuple[int, QRectF, QRectF, float]] = []
        for i in range(n_pages):
            ar = (
                self.lecture_doc.get_page_aspect_ratio(i)
                if self.lecture_doc is not None
                else float(self.slides[i].get("aspect_ratio") or (16.0 / 9.0))
            )
            pw = avail_w
            ph = pw / max(0.3, min(3.5, ar))
            px = 0.5 * (cw - pw)
            cont_r = QRectF(px, y_cursor, pw, ph)
            canon_r = compute_slide_rect_on_canvas((cw, ch), ar, is_tutorial=False)
            ratio = pw / max(1.0, canon_r.width())
            out.append((i, cont_r, canon_r, ratio))
            y_cursor += ph + gap
        return out

    def set_continuous_scroll_mode(self, enabled: bool) -> None:
        self.continuous_scroll_mode = bool(enabled)
        if self.continuous_scroll_mode and self.lecture_doc is not None:
            for p_i, cont_r, _canon_r, _sc in self._compute_continuous_page_rects():
                if p_i == self.current_slide_idx:
                    self.doc_scroll_y = max(0.0, cont_r.top() - 48.0)
                    break
        else:
            self.doc_scroll_y = 0.0
        self._bump_scene()
        self.update()

    def scroll_document(self, delta_px: float) -> float:
        """Scroll multi-page document vertically and keep `current_slide_idx` synchronized."""
        if self.lecture_doc is None:
            return 0.0
        if not self.continuous_scroll_mode:
            if delta_px > 40.0:
                self.next_slide()
            elif delta_px < -40.0:
                self.prev_slide()
            return self.doc_scroll_y

        layout = self._compute_continuous_page_rects()
        if not layout:
            return 0.0
        content_bottom = layout[-1][1].bottom() + 48.0
        view_h = max(240.0, float(self.height()) / max(0.5, float(self.zoom_scale)))
        max_scroll = max(0.0, content_bottom - view_h)
        new_y = max(0.0, min(max_scroll, self.doc_scroll_y + float(delta_px)))
        if abs(new_y - self.doc_scroll_y) < 0.5:
            return self.doc_scroll_y
        self.doc_scroll_y = new_y

        mid_y = self.doc_scroll_y + view_h * 0.45
        best_idx = self.current_slide_idx
        best_dist = 1e9
        for p_i, cont_r, _, _ in layout:
            dist = abs(cont_r.center().y() - mid_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = p_i
        if best_idx != self.current_slide_idx:
            self.current_slide_idx = best_idx
            self._ensure_slide_layer(best_idx)
            self.lecture_doc.prewarm_around(best_idx, radius=2, scale=2.25)
            self.slides_changed.emit()

        self._bump_scene()
        self.update()
        return self.doc_scroll_y

    def _map_board_to_page_canon(
        self,
        board_pt: QPointF,
        *,
        pick_page: bool = False,
    ) -> QPointF:
        """Map board coordinates between Continuous Vertical Scroll layout and active page canonical coords."""
        if not self.continuous_scroll_mode or self.lecture_doc is None:
            return board_pt
        layout = self._compute_continuous_page_rects()
        if not layout:
            return board_pt
        vy = board_pt.y() + self.doc_scroll_y
        target = None
        if pick_page:
            best_d = 1e9
            for item in layout:
                cont_r = item[1]
                if cont_r.top() <= vy <= cont_r.bottom():
                    target = item
                    break
                d = min(abs(vy - cont_r.top()), abs(vy - cont_r.bottom()))
                if d < best_d:
                    best_d = d
                    target = item
            if target is not None and target[0] != self.current_slide_idx:
                self.current_slide_idx = target[0]
                self._ensure_slide_layer(target[0])
                self.slides_changed.emit()
        else:
            for item in layout:
                if item[0] == self.current_slide_idx:
                    target = item
                    break
        if target is None:
            return board_pt
        _, cont_r, canon_r, ratio = target
        cx = canon_r.left() + (board_pt.x() - cont_r.left()) / max(0.1, ratio)
        cy = canon_r.top() + (vy - cont_r.top()) / max(0.1, ratio)
        return QPointF(cx, cy)

    def _cursor_dirty_rect(self, pt1: Optional[QPointF], pt2: Optional[QPointF] = None, margin: float = 115.0) -> Optional[QRectF]:
        pts = [p for p in (pt1, pt2) if p is not None]
        if not pts:
            return None
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        pad = max(48.0, float(margin))
        return QRectF(
            max(0.0, min(xs) - pad),
            max(0.0, min(ys) - pad),
            (max(xs) - min(xs)) + 2.0 * pad,
            (max(ys) - min(ys)) + 2.0 * pad,
        ).intersected(QRectF(0.0, 0.0, max(1.0, float(self.width())), max(1.0, float(self.height()))))

    def apply_60hz_extrapolated_cursor(self, cursor_norm: Tuple[float, float]) -> None:
        """Ultra-low-latency 60Hz cursor/ink-tip updater driven between CV inference ticks."""
        if cursor_norm is None or len(cursor_norm) < 2:
            return
        sx = float(cursor_norm[0]) * max(1, self.width())
        sy = float(cursor_norm[1]) * max(1, self.height())
        new_pos = QPointF(sx, sy)
        old_pos = self.cursor_pos
        if old_pos is not None and math.hypot(new_pos.x() - old_pos.x(), new_pos.y() - old_pos.y()) < 0.6:
            return
        self.cursor_pos = new_pos
        if self.hud_mode == DrawingHudMode.DRAWING:
            bpt = self._map_board_to_page_canon(self._screen_to_board(new_pos), pick_page=False)
            if not self.current_stroke or math.hypot(
                bpt.x() - self.current_stroke[-1].x(), bpt.y() - self.current_stroke[-1].y()
            ) >= 1.5:
                self.current_stroke.append(bpt)
        dirty = self._cursor_dirty_rect(old_pos, new_pos, margin=max(90.0, self.dynamic_eraser_radius + 36.0))
        if dirty is not None and self.gesture_toast is None and self.shape_toast is None:
            self.update(dirty.toRect())
        else:
            self.update()

    def next_slide(self) -> bool:
        return self.set_current_slide(self.current_slide_idx + 1)

    def prev_slide(self) -> bool:
        return self.set_current_slide(self.current_slide_idx - 1)

    def clear_slide_ink(self) -> None:
        """Clear objects/ink ONLY on the active slide (preserving other slides + Ctrl+Z undo)."""
        sc = self._sync_legacy_into_scene()
        if self.strokes or self.annotations or sc.objects:
            self.ai_action_undo_stack.append({
                "strokes": list(self.strokes),
                "annotations": list(self.annotations),
            })
            sc.clear_all(record_undo=True)
        self._sync_scene_to_legacy()
        self.redo_strokes.clear()
        self._selected_board_image_idx = None
        self._obj_interaction = None
        self.current_stroke.clear()
        self.shape_toast = None
        self.toast_revert_rect = None
        self.hide_lasso_bar()
        self.slides_changed.emit()
        self._bump_scene()
        self.update()

    def capture_active_slide_context(self) -> Dict[str, Any]:
        """Capture active slide image (slide + teacher ink/annotations) + extracted slide text for Slide AI."""
        if self.width() >= 64 and self.height() >= 64:
            png_bytes, pm = self.export_board_png(None, enhance_crop=False)
        else:
            pm = QPixmap(1280, 720)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            orig_w, orig_h = self.width(), self.height()
            self.resize(1280, 720)
            self._paint_static_scene(p)
            p.end()
            self.resize(max(1, orig_w), max(1, orig_h))
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.WriteOnly)
            pm.save(buf, "PNG")
            buf.close()
            raw = bytes(ba)
            png_bytes = compress_snapshot_png(raw, max_dim=1024) or raw

        slide_text = ""
        slide_title = "Bảng trắng AirTeach AI"
        if self.slides and 0 <= self.current_slide_idx < len(self.slides):
            sl = self.slides[self.current_slide_idx]
            slide_title = str(sl.get("title") or slide_title)
            if self.lecture_doc is not None and not sl.get("is_tutorial") and self.current_slide_idx < self.lecture_doc.page_count:
                slide_text = self.lecture_doc.get_page_text(self.current_slide_idx)
            else:
                parts = [slide_title, str(sl.get("subtitle") or "")]
                parts.extend(str(b) for b in (sl.get("bullets") or []))
                if sl.get("formula"):
                    parts.append(str(sl["formula"]))
                slide_text = "\n".join(x for x in parts if x.strip())

        return {
            "png_bytes": png_bytes,
            "pixmap": pm,
            "slide_text": slide_text.strip(),
            "slide_title": slide_title,
            "slide_index": self.current_slide_idx,
            "slide_count": len(self.slides),
        }

    def _bump_scene(self) -> None:
        self._scene_rev += 1
        self._dirty_tracker.mark_dirty()
        self._refresh_nav_hud()

    def _screen_to_board(self, pt: QPointF) -> QPointF:
        z = max(0.20, float(self.zoom_scale))
        return QPointF((pt.x() - self.pan_x) / z, (pt.y() - self.pan_y) / z)

    def _board_to_screen(self, pt: QPointF) -> QPointF:
        return QPointF(pt.x() * self.zoom_scale + self.pan_x, pt.y() * self.zoom_scale + self.pan_y)

    def apply_zoom_factor(self, factor: float, center_norm: Optional[Tuple[float, float]] = None) -> None:
        if abs(factor - 1.0) < 0.002:
            return
        new_z = max(0.25, min(4.0, self.zoom_scale * float(factor)))
        if abs(new_z - self.zoom_scale) < 1e-4:
            return
        cx = (center_norm[0] * max(1, self.width())) if center_norm else (self.width() * 0.5)
        cy = (center_norm[1] * max(1, self.height())) if center_norm else (self.height() * 0.5)
        bx = (cx - self.pan_x) / self.zoom_scale
        by = (cy - self.pan_y) / self.zoom_scale
        self.zoom_scale = new_z
        self.pan_x = cx - bx * new_z
        self.pan_y = cy - by * new_z
        self.gesture_toast = {
            "text": f"🔍 Thu Phóng {int(round(new_z * 100))}%",
            "until_ms": self._now_ms() + 950.0,
        }
        self._bump_scene()
        self.update()

    def apply_pan_delta_norm(self, dx_norm: float, dy_norm: float) -> None:
        if self.pan_locked:
            return
        if abs(dx_norm) < 1e-4 and abs(dy_norm) < 1e-4:
            return
        self.pan_x += float(dx_norm) * max(1, self.width())
        self.pan_y += float(dy_norm) * max(1, self.height())
        self._bump_scene()
        self.update()

    def reset_viewport(self) -> None:
        self.zoom_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._bump_scene()
        self.update()

    def fit_to_content(self) -> None:
        """Zoom and center viewport to fit all CanvasObjects (and active slide if visible)."""
        sc = self._sync_legacy_into_scene()
        bounds = sc.content_bounds()
        rects: List[Tuple[float, float, float, float]] = []
        if bounds is not None:
            rects.append((bounds[0], bounds[1], bounds[0] + bounds[2], bounds[1] + bounds[3]))
        if self.slide_visible and self.slides and not bool(self.slides[self.current_slide_idx].get("is_blank", False)):
            sr = self._compute_active_slide_rect()
            if sr.width() >= 20.0 and sr.height() >= 20.0:
                rects.append((sr.left(), sr.top(), sr.right(), sr.bottom()))
        if not rects:
            self.reset_viewport()
            return
        x0 = min(r[0] for r in rects)
        y0 = min(r[1] for r in rects)
        x1 = max(r[2] for r in rects)
        y1 = max(r[3] for r in rects)
        cw = max(40.0, x1 - x0)
        ch = max(40.0, y1 - y0)
        avail_w = max(200.0, float(self.width()) - 64.0)
        avail_h = max(180.0, float(self.height()) - 96.0)
        new_z = max(0.25, min(4.0, min(avail_w / cw, avail_h / ch)))
        self.zoom_scale = new_z
        self.pan_x = (self.width() - cw * new_z) * 0.5 - x0 * new_z
        self.pan_y = (self.height() - ch * new_z) * 0.5 - y0 * new_z
        self.show_gesture_toast(f"⊡ Vừa Nội Dung ({int(round(new_z * 100))}%)", duration_ms=950.0)
        self._bump_scene()
        self.update()

    def fit_to_width(self) -> None:
        sr = self._compute_active_slide_rect()
        if sr.width() < 20.0:
            self.reset_viewport()
            return
        avail_w = max(200.0, float(self.width()) - 40.0)
        new_z = max(0.25, min(4.0, avail_w / sr.width()))
        self.zoom_scale = new_z
        self.pan_x = (self.width() - sr.width() * new_z) * 0.5 - sr.left() * new_z
        self.pan_y = 20.0 - sr.top() * new_z
        self.show_gesture_toast(f"↔ Vừa Ngang ({int(round(new_z * 100))}%)", duration_ms=900.0)
        self._bump_scene()
        self.update()

    def fit_to_page(self) -> None:
        sr = self._compute_active_slide_rect()
        if sr.width() < 20.0 or sr.height() < 20.0:
            self.reset_viewport()
            return
        avail_w = max(200.0, float(self.width()) - 48.0)
        avail_h = max(180.0, float(self.height()) - 84.0)
        new_z = max(0.25, min(4.0, min(avail_w / sr.width(), avail_h / sr.height())))
        self.zoom_scale = new_z
        self.pan_x = (self.width() - sr.width() * new_z) * 0.5 - sr.left() * new_z
        self.pan_y = (self.height() - sr.height() * new_z) * 0.5 - sr.top() * new_z
        self.show_gesture_toast(f"⛶ Vừa Trang ({int(round(new_z * 100))}%)", duration_ms=900.0)
        self._bump_scene()
        self.update()

    def _on_pie_menu_action(self, action_id: str) -> None:
        if action_id in ("PEN", "HIGHLIGHTER", "PAN", "ERASER", "LASER", "SELECT", "TEXT", "STICKY"):
            self.active_tool = action_id
            names = {
                "PEN": "✎ Bút Vẽ (Pen)",
                "HIGHLIGHTER": "🖍 Bút Dạ Quang (Highlighter)",
                "SELECT": "⬉ Chọn / Di Chuyển Object",
                "PAN": "✥ Di Chuyển Bảng (Pan)",
                "ERASER": "⌫ Tẩy Nét (Eraser)",
                "TEXT": "T Nhấn lên bảng để thêm Văn bản",
                "STICKY": "🗒 Nhấn lên bảng để thêm Ghi chú",
                "LASER": "◉ Con Trỏ Laser",
            }
            self.show_gesture_toast(names.get(action_id, action_id), duration_ms=1000.0)
            self.pie_tool_selected.emit(action_id)
        elif action_id == "SHAPES":
            self.active_tool = "SHAPE"
            self.show_gesture_toast(f"◇ Vẽ Hình Khối ({self.active_shape_kind}) — Kéo chuột để tạo", duration_ms=1100.0)
            self.pie_tool_selected.emit("SHAPE")
        elif action_id.startswith("SHAPE:"):
            kind = action_id.split(":", 1)[1].strip().lower() or "rect"
            self.active_shape_kind = kind
            self.active_tool = "SHAPE"
            self.show_gesture_toast(f"◇ Hình khối: {kind.upper()} — Kéo chuột để tạo", duration_ms=1100.0)
            self.pie_tool_selected.emit(action_id)
        elif action_id.startswith("COLOR:"):
            hex_col = action_id.split(":", 1)[1].strip()
            self.ink_color = QColor(hex_col)
            self.active_tool = "PEN"
            self.show_gesture_toast(f"● Đổi màu mực: {hex_col}", duration_ms=950.0)
            self.pie_tool_selected.emit(action_id)
        elif action_id == "ASK_AI_REGION":
            self.active_tool = "SELECT_REGION"
            self.show_gesture_toast("✦ Kéo chuột khoanh vùng để Hỏi Trợ Lý AI", duration_ms=1300.0)
            self.pie_tool_selected.emit("SELECT_REGION")
        elif action_id == "UNDO":
            if self.request_undo():
                self.undo()
            self.show_gesture_toast("↶ Hoàn Tác (Undo)", duration_ms=850.0)
            self.pie_tool_selected.emit("UNDO")
        elif action_id == "CLEAR_OR_NEW":
            # Slice 8 ("Cử Chỉ / AI"): opens the 2-column Gesture Cheat-Sheet safely without wiping ink!
            # Destructive CLEAR_BOARD & NEW_PAGE remain explicit choices in the outer sub-ring.
            self.show_gesture_toast("🖐 Mở Sơ Đồ Cử Chỉ & Trợ Lý AI", duration_ms=1100.0)
            self.pie_tool_selected.emit("OPEN_GESTURE_MAP")
        elif action_id == "NEW_PAGE":
            self.add_blank_page()
            self.pie_tool_selected.emit("NEW_PAGE")
        elif action_id == "CLEAR_BOARD":
            self.clear_board()
            self.show_gesture_toast("🗑 Đã xóa sạch bảng (Ctrl+Z để hoàn tác)", duration_ms=1050.0)
            self.pie_tool_selected.emit("CLEAR_BOARD")
        elif action_id == "TOGGLE_SMART_SHAPE":
            self.smart_shape_enabled = not self.smart_shape_enabled
            state_txt = "BẬT" if self.smart_shape_enabled else "TẮT"
            self.show_gesture_toast(f"◇ Nắn Hình Thông Minh: {state_txt}", duration_ms=1000.0)
            self.pie_tool_selected.emit("TOGGLE_SMART_SHAPE")
        elif action_id == "OPEN_AI_CHAT":
            self.show_gesture_toast("🤖 Đã mở Trợ Lý AI", duration_ms=1100.0)
            self.pie_tool_selected.emit("OPEN_AI_CHAT")
        elif action_id == "OPEN_GESTURE_MAP":
            self.show_gesture_toast("🖐 Mở Sơ Đồ Cử Chỉ 1 & 2 Tay", duration_ms=1100.0)
            self.pie_tool_selected.emit("OPEN_GESTURE_MAP")

    def wheelEvent(self, event) -> None:
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        if (event.modifiers() & Qt.ShiftModifier) and (delta_y != 0 or delta_x != 0):
            if not self.pan_locked:
                horiz = delta_x if delta_x != 0 else delta_y
                self.pan_x += float(horiz) * 0.65
                self._bump_scene()
                self.update()
            event.accept()
            return
        if delta_x != 0 and delta_y == 0:
            if not self.pan_locked:
                self.pan_x += float(delta_x) * 0.65
                self._bump_scene()
                self.update()
            event.accept()
            return
        if delta_y != 0:
            if self.continuous_scroll_mode and not (event.modifiers() & Qt.ControlModifier):
                self.scroll_document(-float(delta_y) * 0.75)
                event.accept()
                return
            factor = 1.08 if delta_y > 0 else (1.0 / 1.08)
            pos = event.position()
            c_norm = (pos.x() / max(1, self.width()), pos.y() / max(1, self.height()))
            self.apply_zoom_factor(factor, c_norm)
            event.accept()

    def set_dark_mode(self, dark: bool) -> None:
        self.dark_mode = bool(dark)
        if hasattr(self, "nav_hud") and self.nav_hud is not None:
            self.nav_hud.set_dark_mode(self.dark_mode)
        if hasattr(self, "pie_menu") and self.pie_menu is not None:
            self.pie_menu.set_dark_mode(self.dark_mode)
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
        tl_scr = self._board_to_screen(QPointF(region.bbox_x, region.bbox_y))
        br_scr = self._board_to_screen(QPointF(region.bbox_x + region.bbox_w, region.bbox_y + region.bbox_h))
        x = max(10, min(max(10, self.width() - bar_w - 10), int(tl_scr.x())))
        y = int(tl_scr.y() - bar_h - 8)
        if y < 44:
            y = min(max(44, self.height() - bar_h - 10), int(br_scr.y() + 8))
        self.lasso_action_bar.move(x, y)
        self.lasso_action_bar.setVisible(True)
        self.lasso_action_bar.raise_()

    def _occupied_rects(self) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        if self.slide_visible and self.slides:
            sr = self._compute_active_slide_rect()
            rects.append((sr.left(), sr.top(), sr.width(), sr.height()))
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
        snapshot_pushed = False

        def _push_ai_undo_once() -> None:
            nonlocal snapshot_pushed
            if not snapshot_pushed:
                self.ai_action_undo_stack.append({
                    "strokes": list(self.strokes),
                    "annotations": list(self.annotations),
                })
                snapshot_pushed = True

        for act in actions:
            if act.tool == "clear_board":
                _push_ai_undo_once()
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
                _push_ai_undo_once()
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
                _push_ai_undo_once()
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
                _push_ai_undo_once()
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
                _push_ai_undo_once()
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
                    "is_ai": True,
                    "color": ink,
                    "thickness": thick,
                }
                self.strokes.append(stroke_entry)
                self._animate_stroke_reveal(stroke_entry, qpts)
                self._bump_scene()
                reports.append(f"Đã vẽ đường cong/đồ thị ({len(qpts)} điểm).")
            elif act.tool == "highlight_region":
                _push_ai_undo_once()
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
                    "is_ai": True,
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
                _push_ai_undo_once()
                cw, ch = canvas_sz
                is_norm = bool(act.args.get("normalized", False))
                rx = (float(act.args.get("x", 0.0)) / 1000.0 * cw) if is_norm else float(act.args.get("x", 0.0))
                ry = (float(act.args.get("y", 0.0)) / 1000.0 * ch) if is_norm else float(act.args.get("y", 0.0))
                rw = (float(act.args.get("w", 200.0)) / 1000.0 * cw) if is_norm else float(act.args.get("w", 200.0))
                rh = (float(act.args.get("h", 200.0)) / 1000.0 * ch) if is_norm else float(act.args.get("h", 200.0))
                reg_rect = QRectF(rx, ry, rw, rh)
                self.strokes = [
                    st for st in self.strokes
                    if not any(reg_rect.contains(p) for p in st.get("points", []))
                ]
                kept_ann = []
                for ann in self.annotations:
                    ax0 = float(ann.get("x", 0.0))
                    ay0 = float(ann.get("y", 0.0))
                    aw0 = float(ann.get("w", 120.0))
                    ah0 = float(ann.get("h", 40.0))
                    if not reg_rect.intersects(QRectF(ax0, ay0, aw0, ah0)):
                        kept_ann.append(ann)
                self.annotations = kept_ann
                self._bump_scene()
                reports.append("Đã xóa vùng chỉ định trên bảng.")
            elif act.tool == "draw_shape":
                _push_ai_undo_once()
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
                        "is_ai": True,
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
        sc = self.scene
        sc.objects.clear()
        sc.selected_ids.clear()
        self._sync_legacy_into_scene()
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
            self.toggle_slide_btn.setText("Ẩn Slide -> Bảng trống")
        else:
            self.toggle_slide_btn.setText("Hiện nội dung Slide")
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
    def _hand_pose_glyph(hand_obj: Any) -> str:
        if not isinstance(hand_obj, dict):
            return "—"
        tag = str(hand_obj.get("pose_tag") or hand_obj.get("gesture") or "HOVER").upper()
        glyph_map = {
            "PINCH": "🤏",
            "DRAW": "☝️",
            "INDEX": "☝️",
            "TWO_FINGERS": "✌️",
            "THREE_FINGERS": "🤟",
            "OPEN_PALM": "🖐️",
            "ERASE": "🖐️",
            "FIST": "✊",
            "HOVER": "🖐",
            "IDLE": "—",
        }
        return glyph_map.get(tag, "🖐")

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
        dual_info = pkt.get("dual_hand") or {}
        two_mode = str(dual_info.get("two_hand_mode") or "NONE")
        self._two_hand_nav_active = two_mode in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN")
        suppress_nav = bool(pkt.get("suppress_fsm_for_nav", False))
        suppress_draw = bool(dual_info.get("suppress_primary_drawing", False)) or suppress_nav
        self.active_modifier_mode = str(dual_info.get("modifier_mode") or "NONE")
        self.two_hand_status_text = str(dual_info.get("status_label_vi") or "")
        hand_span = float(pkt.get("hand_span") or 0.20)
        base_dim = float(min(max(360, self.width()), max(360, self.height())))
        self.dynamic_eraser_radius = max(26.0, min(105.0, hand_span * base_dim * 0.42)) / max(0.25, self.zoom_scale)
        hand_present = cursor_norm is not None and gesture != "IDLE"

        old_cursor = self.cursor_pos
        cursor_px: Optional[Tuple[float, float]] = None
        if cursor_norm is not None:
            sx = float(cursor_norm[0]) * max(1, self.width())
            sy = float(cursor_norm[1]) * max(1, self.height())
            self.cursor_pos = QPointF(sx, sy)
            bpt = self._map_board_to_page_canon(
                self._screen_to_board(self.cursor_pos),
                pick_page=(self.hud_mode != DrawingHudMode.DRAWING),
            )
            cursor_px = (bpt.x(), bpt.y())
            if self.active_tool == "LASER" or (
                self.hud_mode == DrawingHudMode.HOVER and pose_tag in ("INDEX", "HOVER")
            ):
                self.laser_trail.append((QPointF(bpt.x(), bpt.y()), now_ms))
        self._prev_cursor_pos = old_cursor

        # 1-Hand Pinch Object Grab & Drag (when pinching over an existing CanvasObject)
        is_pinch_active = (
            hand_present
            and cursor_px is not None
            and not suppress_draw
            and (pose_tag == "PINCH" or (pinch_ratio is not None and float(pinch_ratio) < 0.30))
        )
        pinch_released = (
            not hand_present
            or suppress_draw
            or pose_tag in ("OPEN_PALM", "FIST", "TWO_FINGERS", "INDEX")
            or (pinch_ratio is not None and float(pinch_ratio) > 0.40)
        )
        grab_active = False
        if self._cv_grab_obj_id is not None:
            if pinch_released or cursor_px is None:
                self._cv_grab_obj_id = None
                self._cv_grab_last_pt = None
            else:
                sc = self._sync_legacy_into_scene()
                obj = sc.get_by_id(self._cv_grab_obj_id)
                if obj is not None and not obj.locked and self._cv_grab_last_pt is not None:
                    dx = cursor_px[0] - self._cv_grab_last_pt[0]
                    dy = cursor_px[1] - self._cv_grab_last_pt[1]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        obj.translate(dx, dy)
                        self._sync_scene_to_legacy()
                        self._bump_scene()
                    self._cv_grab_last_pt = cursor_px
                    grab_active = True
                else:
                    self._cv_grab_obj_id = None
                    self._cv_grab_last_pt = None
        elif is_pinch_active and self.hud_mode != DrawingHudMode.DRAWING and cursor_px is not None:
            sc = self._sync_legacy_into_scene()
            hit_obj = sc.hit_test(cursor_px[0], cursor_px[1], tolerance=max(18.0, 22.0 / max(0.25, self.zoom_scale)))
            if hit_obj is not None and not hit_obj.locked:
                sc.push_undo_snapshot()
                sc.select_only(hit_obj.id)
                self._cv_grab_obj_id = hit_obj.id
                self._cv_grab_last_pt = cursor_px
                self._bump_scene()
                grab_active = True

        # Track hold-to-snap & closed-loop Lasso dwell stillness while drawing
        if self.hud_mode == DrawingHudMode.DRAWING and cursor_px is not None:
            qpt = QPointF(cursor_px[0], cursor_px[1])
            if (
                self._last_motion_pt is None
                or math.hypot(qpt.x() - self._last_motion_pt.x(), qpt.y() - self._last_motion_pt.y()) > 6.0
            ):
                self._last_motion_pt = qpt
                self._last_motion_ms = now_ms
            still_ms = now_ms - self._last_motion_ms
            self._hold_snap_ready = (not self.hold_to_snap_350ms) or (still_ms >= 350.0)
            self._lasso_dwell_ready = still_ms >= 450.0

        select_mode_block_draw = self.active_tool == "SELECT"
        upd = self.drawing_fsm.step(
            now_ms,
            hand_present=hand_present,
            cursor_xy=cursor_px,
            pinch_ratio=pinch_ratio,
            pose_tag=pose_tag,
            left_pan_active=left_pan,
            suppress_drawing=(suppress_draw or grab_active or select_mode_block_draw),
        )
        self._apply_fsm_update(upd, now_ms)

        # Live Gesture Recognition HUD status badge (clear 1-hand & 2-hand Vietnamese feedback)
        if not hand_present and two_mode not in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN"):
            self.live_gesture_hud_text = ""
        elif two_mode == "DUAL_PINCH_ZOOM":
            self.live_gesture_hud_text = f"✋ 2 Tay: Đang Zoom {int(round(self.zoom_scale * 100))}%"
            self.live_gesture_hud_accent = BRIGHT_YELLOW
        elif two_mode == "DUAL_PALM_PAN":
            action_vi = "Đang Cuộn Tài Liệu" if self.continuous_scroll_mode else "Đang Di Chuyển Bảng"
            self.live_gesture_hud_text = f"✋ 2 Tay: {action_vi}"
            self.live_gesture_hud_accent = SECONDARY_TEAL
        elif grab_active:
            self.live_gesture_hud_text = "🤏 1 Tay: Đang Cầm & Kéo Object"
            self.live_gesture_hud_accent = BRIGHT_YELLOW
            self.cursor_mode = "TWO_HAND_NAV"
        elif select_mode_block_draw:
            self.live_gesture_hud_text = "⬉ Chế độ Chọn: Chụm lên Object để kéo"
            self.live_gesture_hud_accent = SECONDARY_TEAL
        elif upd.hud_mode == DrawingHudMode.DRAWING:
            tool_vi = "Bút Dạ Quang" if self.active_tool == "HIGHLIGHTER" else "Đang Vẽ"
            self.live_gesture_hud_text = f"☝️ 1 Tay: {tool_vi}"
            self.live_gesture_hud_accent = NEON_CYAN
        elif upd.hud_mode == DrawingHudMode.PAN_ERASE or pose_tag == "OPEN_PALM":
            self.live_gesture_hud_text = "🖐️ 1 Tay: Đang Tẩy (Xòe Bàn Tay)"
            self.live_gesture_hud_accent = ALERT_ROSE
        elif upd.hud_mode == DrawingHudMode.LOCKED_COOLDOWN:
            self.live_gesture_hud_text = f"🔒 Đã Hủy • Khóa Tạm {int(max(0.0, upd.cooldown_remaining_ms))}ms"
            self.live_gesture_hud_accent = ALERT_ROSE
        elif pose_tag == "TWO_FINGERS":
            self.live_gesture_hud_text = "✌️ 1 Tay: Con Trỏ / Hover"
            self.live_gesture_hud_accent = SECONDARY_TEAL
        else:
            self.live_gesture_hud_text = "🖐 1 Tay: Con Trỏ Chờ (Hover)"
            self.live_gesture_hud_accent = PRIMARY_INDIGO

        if self.live_gesture_hud_text:
            has_lr = ("left_hand" in pkt or "right_hand" in pkt) and (
                pkt.get("left_hand") is not None or pkt.get("right_hand") is not None
            )
            fps_raw = pkt.get("fps")
            if fps_raw is None:
                vt_pkt = pkt.get("vision_telemetry")
                if isinstance(vt_pkt, dict):
                    fps_raw = vt_pkt.get("ai_fps") or vt_pkt.get("cam_fps")
            fps_val = 0.0
            if fps_raw is not None:
                try:
                    fps_val = float(fps_raw)
                except (TypeError, ValueError):
                    fps_val = 0.0
            telemetry_parts: List[str] = []
            if has_lr:
                l_g = self._hand_pose_glyph(pkt.get("left_hand"))
                r_g = self._hand_pose_glyph(pkt.get("right_hand"))
                telemetry_parts.append(f"L:{l_g} R:{r_g}")
            if fps_val > 0.5:
                telemetry_parts.append(f"{int(round(fps_val))} FPS")
            if telemetry_parts:
                self.live_gesture_hud_text = (
                    f"{self.live_gesture_hud_text} | {' | '.join(telemetry_parts)}"
                )
        return upd

    def _apply_fsm_update(self, upd: DrawingFsmUpdate, now_ms: float) -> None:
        self.hud_mode = upd.hud_mode
        self.cooldown_remaining_ms = upd.cooldown_remaining_ms
        self.cancel_hold_progress = upd.cancel_hold_progress
        if getattr(self, "_two_hand_nav_active", False):
            self.cursor_mode = "TWO_HAND_NAV"
        else:
            self.cursor_mode = upd.hud_mode.value

        if upd.action in ("BEGIN_STROKE", "APPEND_POINT", "HOLD_GRACE"):
            if upd.action == "BEGIN_STROKE":
                self.redo_strokes.clear()
                self._stroke_start_ms = now_ms
                self._last_motion_ms = now_ms
                self._last_motion_pt = QPointF(upd.cursor_xy[0], upd.cursor_xy[1]) if upd.cursor_xy else None
                self._hold_snap_ready = not self.hold_to_snap_350ms
                self._lasso_dwell_ready = False
            elif upd.action == "HOLD_GRACE":
                self.cursor_mode = "DRAWING"
            if upd.active_points:
                self.current_stroke = [QPointF(x, y) for (x, y) in upd.active_points]
        elif upd.action == "COMMIT_STROKE":
            self.current_stroke = [QPointF(x, y) for (x, y) in upd.committed_points]
            self._finalize_current_stroke(now_ms=now_ms, via_cv=True)
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
            self._erase_at(QPointF(upd.cursor_xy[0], upd.cursor_xy[1]), self.dynamic_eraser_radius)

        self.hud_status_changed.emit(
            upd.hud_mode.value, float(upd.cooldown_remaining_ms), upd.cancel_reason
        )
        if upd.action in ("APPEND_POINT", "HOLD_GRACE", "NONE") and self.shape_toast is None and self.gesture_toast is None:
            dirty = self._cursor_dirty_rect(
                self._prev_cursor_pos,
                self.cursor_pos,
                margin=max(110.0, self.dynamic_eraser_radius + 45.0),
            )
            if dirty is not None:
                self.update(dirty.toRect())
                return
        self.update()

    def trigger_esc_cancel(self) -> None:
        if hasattr(self, "pie_menu") and self.pie_menu.isVisible():
            self.pie_menu.cancel()
            self.update()
            return
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
        sc = self._sync_legacy_into_scene()
        if not self.strokes:
            return False
        target_idx = (
            self.shape_toast["stroke_idx"]
            if self.shape_toast and 0 <= self.shape_toast["stroke_idx"] < len(self.strokes)
            else len(self.strokes) - 1
        )
        stroke = self.strokes[target_idx]
        oid = stroke.get("_obj_id")
        obj = sc.get_by_id(oid) if oid else None
        if obj is not None and obj.raw_points and len(obj.raw_points) >= 2:
            sc.push_undo_snapshot()
            obj.points = list(obj.raw_points)
            bx, by, bw, bh = compute_points_bbox(obj.points, min_size=18.0)
            obj.x, obj.y, obj.w, obj.h = bx, by, bw, bh
            obj.obj_type = "stroke"
            obj.is_beautified = False
            obj.shape_type = "FREEHAND"
            obj.shape_label_vi = "Nét gốc mượt"
            obj.confidence = 0.0
            self._sync_scene_to_legacy()
        else:
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

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            pt_screen = QPointF(event.position())
            pt = self._map_board_to_page_canon(self._screen_to_board(pt_screen), pick_page=True)
            sc = self._sync_legacy_into_scene()
            hit = sc.hit_test(pt.x(), pt.y(), tolerance=max(10.0, 12.0 / max(0.25, self.zoom_scale)))
            if hit is not None and hit.obj_type in ("text", "sticky"):
                sc.select_only(hit.id)
                self._on_edit_selected_text_dialog()
                return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        pt_screen = QPointF(event.position())
        self.cursor_pos = pt_screen
        pt_raw = self._screen_to_board(pt_screen)
        pt = self._map_board_to_page_canon(pt_raw, pick_page=True)

        # Check clickable "[Hoàn tác về nét gốc]" toast pill (screen space)
        if (
            event.button() == Qt.LeftButton
            and self.toast_revert_rect is not None
            and self.shape_toast is not None
            and self.toast_revert_rect.contains(pt_screen)
        ):
            self.revert_last_beautified_stroke()
            return

        if self.drawing_fsm.hud_mode == DrawingHudMode.LOCKED_COOLDOWN:
            return

        if event.button() == Qt.RightButton:
            self.pie_menu.open_at(pt_screen, smart_shape_enabled=self.smart_shape_enabled)
            self.update()
            return

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and (self._space_panning or self.active_tool == "PAN")
        ):
            if self.pan_locked:
                self.show_gesture_toast("🔒 Bảng đang khóa Pan (nhấn nút Khóa góc phải dưới để mở)", duration_ms=1100.0)
                return
            self._mouse_pan_last = pt_screen
            self.cursor_mode = "TWO_HAND_NAV"
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            sc = self._sync_legacy_into_scene()
            # 1. SELECT tool: full Object manipulation (on-box delete handle, 8-handle resize, rotate, drag-move, marquee multi-select)
            if self.active_tool == "SELECT":
                sel_now = sc.selected_objects()
                if len(sel_now) == 1 and not sel_now[0].locked:
                    del_pt = self._selected_delete_handle_board_pos(sel_now[0])
                    del_hit_r = 13.0 / max(0.25, float(self.zoom_scale))
                    if math.hypot(pt.x() - del_pt.x(), pt.y() - del_pt.y()) <= del_hit_r:
                        self.delete_selected_objects()
                        return

                handle_hit = sc.hit_test_transform_handle(pt.x(), pt.y(), zoom_scale=self.zoom_scale)
                if handle_hit is not None:
                    handle_name, obj = handle_hit
                    sc.push_undo_snapshot()
                    cx, cy = obj.center
                    start_ang = math.degrees(math.atan2(pt.y() - cy, pt.x() - cx))
                    self._obj_interaction = {
                        "mode": "rotate" if handle_name == "rotate" else "resize",
                        "handle": handle_name,
                        "obj_id": obj.id,
                        "orig_obj": obj.clone(new_id=False),
                        "start_pt": (pt.x(), pt.y()),
                        "start_angle": start_ang,
                        "orig_rotation": obj.rotation,
                    }
                    self.obj_action_bar.setVisible(False)
                    self.update()
                    return

                hit = sc.hit_test(pt.x(), pt.y(), tolerance=max(10.0, 12.0 / max(0.25, self.zoom_scale)))
                additive = bool(event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))
                if hit is not None:
                    if additive:
                        sc.toggle_select(hit.id)
                    elif hit.id not in sc.selected_ids:
                        sc.select_only(hit.id)
                    sel_objs = sc.selected_objects()
                    if sel_objs and not all(o.locked for o in sel_objs):
                        sc.push_undo_snapshot()
                        self._obj_interaction = {
                            "mode": "move",
                            "start_pt": (pt.x(), pt.y()),
                            "orig_map": {o.id: o.clone(new_id=False) for o in sel_objs},
                        }
                        self.obj_action_bar.setVisible(False)
                    self._bump_scene()
                    self.update()
                    return
                else:
                    if not additive:
                        sc.clear_selection()
                    self._marquee_rect = QRectF(pt, pt)
                    self._obj_interaction = {
                        "mode": "marquee",
                        "start_pt": QPointF(pt),
                        "additive": additive,
                    }
                    self._bump_scene()
                    self.update()
                    return

            # 2. TEXT / STICKY creation tools
            if self.active_tool == "TEXT":
                txt, ok = QInputDialog.getMultiLineText(self, "Thêm Văn Bản Lên Bảng", "Nhập nội dung văn bản:", "")
                if ok and txt.strip():
                    self.add_text_object(txt.strip(), pt.x(), pt.y(), select=True)
                    self.active_tool = "SELECT"
                    self.pie_tool_selected.emit("SELECT")
                return

            if self.active_tool == "STICKY":
                txt, ok = QInputDialog.getMultiLineText(self, "Thêm Ghi Chú (Sticky Note)", "Nội dung ghi chú:", "")
                if ok and txt.strip():
                    self.add_sticky_object(txt.strip(), pt.x(), pt.y(), select=True)
                    self.active_tool = "SELECT"
                    self.pie_tool_selected.emit("SELECT")
                return

            # 3. SHAPE interactive drag-to-create tool
            if self.active_tool == "SHAPE":
                self._shape_preview_rect = QRectF(pt, pt)
                self._obj_interaction = {
                    "mode": "create_shape",
                    "start_pt": QPointF(pt),
                    "shape_kind": self.active_shape_kind,
                }
                self.update()
                return

            # Non-SELECT tools: allow direct transform handle drag if an object was selected, or board image grip
            if self._try_begin_board_image_drag(pt):
                self.update()
                return
            if sc.selected_ids:
                sc.clear_selection()
                self._selected_board_image_idx = None
                self._bump_scene()
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
                self._erase_at(pt, self.dynamic_eraser_radius)
            elif self.active_tool == "LASER":
                self.cursor_mode = "LASER"
                self.laser_trail.append((pt, self._now_ms()))
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

    def _try_begin_board_image_drag(self, pt: QPointF) -> bool:
        imgs = self.board_images
        for idx in range(len(imgs) - 1, -1, -1):
            item = imgs[idx]
            ix = float(item.get("x", 0.0))
            iy = float(item.get("y", 0.0))
            iw = max(48.0, float(item.get("w", 240.0)))
            ih = max(48.0, float(item.get("h", 180.0)))
            resize_grip = QRectF(ix + iw - 20.0, iy + ih - 20.0, 24.0, 24.0)
            move_header = QRectF(ix, iy, iw, min(28.0, ih))
            if resize_grip.contains(pt):
                self._selected_board_image_idx = idx
                self._img_drag_state = {
                    "mode": "resize",
                    "idx": idx,
                    "start_pt": QPointF(pt),
                    "orig": (ix, iy, iw, ih),
                }
                self._bump_scene()
                return True
            if move_header.contains(pt):
                self._selected_board_image_idx = idx
                self._img_drag_state = {
                    "mode": "move",
                    "idx": idx,
                    "start_pt": QPointF(pt),
                    "orig": (ix, iy, iw, ih),
                }
                self._bump_scene()
                return True
        return False

    def mouseMoveEvent(self, event) -> None:
        pt_screen = QPointF(event.position())
        prev_scr = self.cursor_pos
        self.cursor_pos = pt_screen
        pt_raw = self._screen_to_board(pt_screen)
        pt = self._map_board_to_page_canon(pt_raw, pick_page=False)

        if not self.pie_menu.isHidden() and (event.buttons() & Qt.RightButton):
            self.pie_menu.update_pointer(pt_screen)
            return

        # Active Scene Graph Object Interaction (Move / Resize / Rotate / Marquee / Create Shape)
        if self._obj_interaction is not None and (event.buttons() & Qt.LeftButton):
            mode = self._obj_interaction["mode"]
            sc = self.scene
            if mode == "move":
                sx, sy = self._obj_interaction["start_pt"]
                dx, dy = pt.x() - sx, pt.y() - sy
                for oid, orig in self._obj_interaction["orig_map"].items():
                    obj = sc.get_by_id(oid)
                    if obj is not None and not obj.locked:
                        obj.x = orig.x + dx
                        obj.y = orig.y + dy
                        if orig.points:
                            obj.points = [(px + dx, py + dy) for (px, py) in orig.points]
                        if orig.raw_points:
                            obj.raw_points = [(px + dx, py + dy) for (px, py) in orig.raw_points]
                self._sync_scene_to_legacy()
                self._bump_scene()
                self.update()
                return
            elif mode == "resize":
                orig: CanvasObject = self._obj_interaction["orig_obj"]
                obj = sc.get_by_id(self._obj_interaction["obj_id"])
                if obj is not None:
                    nx, ny, nw, nh = compute_resized_bbox(
                        orig,
                        self._obj_interaction["handle"],
                        (pt.x(), pt.y()),
                        min_size=24.0,
                    )
                    obj.apply_bbox_transform(orig, nx, ny, nw, nh)
                    self._sync_scene_to_legacy()
                    self._bump_scene()
                    self.update()
                return
            elif mode == "rotate":
                orig = self._obj_interaction["orig_obj"]
                obj = sc.get_by_id(self._obj_interaction["obj_id"])
                if obj is not None:
                    cx, cy = orig.center
                    cur_ang = math.degrees(math.atan2(pt.y() - cy, pt.x() - cx))
                    rot = self._obj_interaction["orig_rotation"] + (cur_ang - self._obj_interaction["start_angle"])
                    if event.modifiers() & Qt.ShiftModifier:
                        rot = round(rot / 15.0) * 15.0
                    obj.rotation = rot % 360.0
                    self._sync_scene_to_legacy()
                    self._bump_scene()
                    self.update()
                return
            elif mode == "marquee":
                self._marquee_rect = QRectF(self._obj_interaction["start_pt"], pt).normalized()
                self.update()
                return
            elif mode == "create_shape":
                self._shape_preview_rect = QRectF(self._obj_interaction["start_pt"], pt).normalized()
                self.update()
                return

        if self._img_drag_state is not None and (event.buttons() & Qt.LeftButton):
            idx = int(self._img_drag_state["idx"])
            if 0 <= idx < len(self.board_images):
                ox, oy, ow, oh = self._img_drag_state["orig"]
                sp = self._img_drag_state["start_pt"]
                dx = pt.x() - sp.x()
                dy = pt.y() - sp.y()
                if self._img_drag_state["mode"] == "resize":
                    self.board_images[idx]["w"] = max(48.0, ow + dx)
                    self.board_images[idx]["h"] = max(48.0, oh + dy)
                else:
                    self.board_images[idx]["x"] = ox + dx
                    self.board_images[idx]["y"] = oy + dy
                self._sync_legacy_into_scene()
                self._bump_scene()
                self.update()
            return

        if self._mouse_pan_last is not None and (
            (event.buttons() & Qt.MiddleButton)
            or ((event.buttons() & Qt.LeftButton) and (self._space_panning or self.active_tool == "PAN"))
        ):
            if not self.pan_locked:
                delta = pt_screen - self._mouse_pan_last
                self._mouse_pan_last = pt_screen
                self.pan_x += delta.x()
                self.pan_y += delta.y()
                self._bump_scene()
                self.update()
            return

        if self.drawing_fsm.hud_mode == DrawingHudMode.LOCKED_COOLDOWN:
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
                self._erase_at(pt, self.dynamic_eraser_radius)
            elif self.active_tool == "LASER":
                self.cursor_mode = "LASER"
                self.laser_trail.append((pt, self._now_ms()))
            elif self.active_tool in ("PEN", "HIGHLIGHTER"):
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
            if self.active_tool == "LASER":
                self.laser_trail.append((pt, self._now_ms()))
            elif self.active_tool == "SELECT" and not self._space_panning:
                sc = self.scene
                hh = sc.hit_test_transform_handle(pt.x(), pt.y(), zoom_scale=self.zoom_scale)
                if hh is not None:
                    hname = hh[0]
                    cur_map = {
                        "nw": Qt.SizeFDiagCursor,
                        "se": Qt.SizeFDiagCursor,
                        "ne": Qt.SizeBDiagCursor,
                        "sw": Qt.SizeBDiagCursor,
                        "n": Qt.SizeVerCursor,
                        "s": Qt.SizeVerCursor,
                        "e": Qt.SizeHorCursor,
                        "w": Qt.SizeHorCursor,
                        "rotate": Qt.PointingHandCursor,
                    }
                    self.setCursor(cur_map.get(hname, Qt.ArrowCursor))
                else:
                    self.unsetCursor()
        if not (event.buttons() & Qt.LeftButton) or self.active_tool in ("PEN", "HIGHLIGHTER", "LASER"):
            if self.shape_toast is None and self.gesture_toast is None and self.active_tool not in ("LASSO", "SELECT_REGION", "SELECT", "SHAPE"):
                dirty = self._cursor_dirty_rect(
                    prev_scr,
                    self.cursor_pos,
                    margin=max(96.0, self.dynamic_eraser_radius + 40.0),
                )
                if dirty is not None:
                    self.update(dirty.toRect())
                    return
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton and not self.pie_menu.isHidden():
            self.pie_menu.release_and_dismiss(QPointF(event.position()))
            self.update()
            return
        if self._obj_interaction is not None:
            inter = self._obj_interaction
            self._obj_interaction = None
            mode = inter["mode"]
            if mode == "marquee":
                if self._marquee_rect is not None and (self._marquee_rect.width() >= 6.0 or self._marquee_rect.height() >= 6.0):
                    self.scene.marquee_select(
                        self._marquee_rect.left(),
                        self._marquee_rect.top(),
                        self._marquee_rect.width(),
                        self._marquee_rect.height(),
                        additive=bool(inter.get("additive", False)),
                    )
                self._marquee_rect = None
            elif mode == "create_shape":
                rect = self._shape_preview_rect
                self._shape_preview_rect = None
                if rect is not None and (rect.width() >= 14.0 or rect.height() >= 14.0):
                    self.add_shape_object(
                        inter.get("shape_kind", self.active_shape_kind),
                        rect.left(),
                        rect.top(),
                        max(24.0, rect.width()),
                        max(24.0, rect.height()),
                        select=True,
                    )
            self._bump_scene()
            self.update()
            return
        if self._img_drag_state is not None:
            self._img_drag_state = None
            self.update()
            return
        if self._mouse_pan_last is not None:
            self._mouse_pan_last = None
            if self._space_panning:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            self.cursor_mode = "HOVER"
            self.update()
            return
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

    def _try_trigger_closed_loop_lasso(self, raw_xy: Sequence[Tuple[float, float]]) -> bool:
        if len(raw_xy) < 10:
            return False
        xs = [p[0] for p in raw_xy]
        ys = [p[1] for p in raw_xy]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        gap = math.hypot(raw_xy[0][0] - raw_xy[-1][0], raw_xy[0][1] - raw_xy[-1][1])
        if diag < 40.0 or gap > max(36.0, 0.25 * diag):
            return False
        region = compute_lasso_bounding_box(raw_xy, (max(1280.0, self.width()), max(720.0, self.height())))
        if region is None:
            return False
        has_content_inside = any(
            stroke_inside_lasso([(float(p.x()), float(p.y())) for p in (s.get("points") or [])], region)
            for s in self.strokes
        )
        if not has_content_inside and self.active_tool != "LASSO":
            return False
        self.lasso_points = [QPointF(x, y) for (x, y) in raw_xy]
        self._crop_and_emit_selection(
            QPointF(region.bbox_x, region.bbox_y),
            QPointF(region.bbox_x + region.bbox_w, region.bbox_y + region.bbox_h),
            open_bar_only=True,
            region=region,
        )
        return True

    def _finalize_current_stroke(self, now_ms: Optional[float] = None, via_cv: bool = False) -> None:
        if len(self.current_stroke) >= 2:
            ts_ms = now_ms if now_ms is not None else self._now_ms()
            raw_xy = [(p.x(), p.y()) for p in self.current_stroke]
            if self.active_tool == "LASSO" or (via_cv and self._lasso_dwell_ready and self._try_trigger_closed_loop_lasso(raw_xy)):
                if self.active_tool == "LASSO" and len(raw_xy) >= 4:
                    self._try_trigger_closed_loop_lasso(raw_xy)
                self.current_stroke = []
                self._lasso_dwell_ready = False
                return

            smooth_raw_xy = catmull_rom_spline(raw_xy, subdivisions=4)
            raw_qpts = [QPointF(x, y) for (x, y) in (smooth_raw_xy if len(smooth_raw_xy) >= 2 else raw_xy)]

            is_hi = self.active_tool == "HIGHLIGHTER"
            if via_cv and self.active_modifier_mode == "LINE_LOCK" and not is_hi and len(raw_xy) >= 2:
                line_pts = resample_arc_length([raw_xy[0], raw_xy[-1]], step_px=12.0)
                final_qpts = [QPointF(x, y) for (x, y) in (line_pts if len(line_pts) >= 2 else [raw_xy[0], raw_xy[-1]])]
                is_geo = True
                shape_type = "LINE"
                shape_label_vi = "Đoạn thẳng (Khóa tay trái)"
                confidence = 0.99
            else:
                conf_gate = 0.62 if (via_cv and self.active_modifier_mode == "SHAPE_LOCK") else 0.78
                should_beautify = (not is_hi) and (self.smart_shape_enabled or (via_cv and self.active_modifier_mode == "SHAPE_LOCK")) and (
                    not self.hold_to_snap_350ms or self._hold_snap_ready or (via_cv and self.active_modifier_mode == "SHAPE_LOCK")
                )
                if should_beautify and len(raw_xy) >= 3:
                    shape_res = beautify_stroke(raw_xy, closed_gap_ratio=0.18, confidence_gate=conf_gate)
                    final_qpts = [QPointF(x, y) for (x, y) in shape_res.beautified_points]
                    is_geo = shape_res.is_geometric
                    shape_type = shape_res.shape_type
                    shape_label_vi = shape_res.shape_label_vi
                    confidence = shape_res.confidence
                else:
                    final_qpts = raw_qpts
                    is_geo = False
                    shape_type = "HIGHLIGHTER" if is_hi else "FREEHAND"
                    shape_label_vi = "Bút dạ quang" if is_hi else "Nét mượt tự do"
                    confidence = 0.0

            stroke_col = QColor(self.highlighter_color if is_hi else self.ink_color)
            if is_hi and stroke_col.alpha() > 120:
                stroke_col.setAlpha(96)
            stroke_thick = max(18, int(self.pen_thickness * 4)) if is_hi else self.pen_thickness

            sc = self._sync_legacy_into_scene()
            raw_xy_smooth = [(p.x(), p.y()) for p in raw_qpts]
            final_xy = [(p.x(), p.y()) for p in final_qpts]
            rx, ry, rw, rh = compute_points_bbox(raw_xy_smooth, min_size=18.0)

            # Record pre-stroke snapshot for Ctrl+Z
            sc.push_undo_snapshot()
            new_obj = CanvasObject(
                id=new_object_id("shp" if is_geo else "str"),
                obj_type="stroke",
                x=rx,
                y=ry,
                w=rw,
                h=rh,
                rotation=0.0,
                z_index=sc.next_z_index(),
                layer=self.current_slide_idx,
                points=raw_xy_smooth,
                raw_points=list(raw_xy_smooth),
                color=stroke_col.name(QColor.HexArgb),
                thickness=float(stroke_thick),
                is_highlighter=is_hi,
                is_beautified=False,
                shape_type="HIGHLIGHTER" if is_hi else "FREEHAND",
                shape_label_vi="Bút dạ quang" if is_hi else "Nét gốc mượt (Ctrl+Z)",
                confidence=0.0,
            )
            sc.add_object(new_obj, record_undo=False, select=False)
            if is_geo:
                # Push intermediate FREEHAND snapshot so 1st Ctrl+Z reverts snapped shape -> smooth raw stroke
                sc.push_undo_snapshot()
                fx, fy, fw, fh = compute_points_bbox(final_xy, min_size=18.0)
                new_obj.obj_type = "shape"
                new_obj.x, new_obj.y, new_obj.w, new_obj.h = fx, fy, fw, fh
                new_obj.points = final_xy
                new_obj.is_beautified = True
                new_obj.shape_type = shape_type
                new_obj.shape_label_vi = shape_label_vi
                new_obj.confidence = confidence
            self._sync_scene_to_legacy()
            self.slides_changed.emit()
            self._bump_scene()

            if is_geo and final_qpts:
                anchor_x = sum(p.x() for p in final_qpts) / len(final_qpts)
                anchor_y = min(p.y() for p in final_qpts) - 18.0
                scr_anchor = self._board_to_screen(QPointF(anchor_x, anchor_y))
                self.shape_toast = {
                    "stroke_idx": len(self.strokes) - 1,
                    "label": shape_label_vi,
                    "conf": confidence,
                    "anchor": scr_anchor,
                    "until_ms": ts_ms + 4500.0,
                }
        self.current_stroke = []
        self._lasso_dwell_ready = False

    def _erase_at(self, pt: QPointF, radius: float) -> None:
        sc = self._sync_legacy_into_scene()
        removed_ids: List[str] = []
        for obj in sc.objects:
            if obj.locked:
                continue
            if obj.obj_type in ("stroke", "shape"):
                if any(math.hypot(px - pt.x(), py - pt.y()) < radius for (px, py) in obj.points):
                    removed_ids.append(obj.id)
            elif obj.obj_type in ("text", "sticky"):
                cx, cy = obj.center
                if math.hypot(cx - pt.x(), cy - pt.y()) < radius + 30.0:
                    removed_ids.append(obj.id)
        if removed_ids:
            sc.push_undo_snapshot()
            rem = set(removed_ids)
            sc.objects = [o for o in sc.objects if o.id not in rem]
            sc.selected_ids = [oid for oid in sc.selected_ids if oid not in rem]
            self._sync_scene_to_legacy()
            self._bump_scene()

    def undo(self) -> None:
        sc = self._sync_legacy_into_scene()
        if sc.can_undo():
            sc.undo()
            self._sync_scene_to_legacy()
            self.shape_toast = None
            self.toast_revert_rect = None
            self._bump_scene()
            self.update()
            return
        if not self.strokes or (self.strokes[-1].get("is_ai") and self.ai_action_undo_stack):
            self.undo_last_ai_action()
            return
        top = self.strokes[-1]
        raw_pts = top.get("raw_points")
        if top.get("is_beautified") and raw_pts and len(raw_pts) >= 2:
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
            popped = self.strokes.pop()
            popped.pop("_replace_last", None)
            self.redo_strokes.append(popped)
            oid = popped.get("_obj_id")
            if oid:
                sc.objects = [o for o in sc.objects if o.id != oid]
        self.shape_toast = None
        self.toast_revert_rect = None
        self._bump_scene()
        self.update()

    def redo(self) -> None:
        sc = self._sync_legacy_into_scene()
        if sc.can_redo():
            sc.redo()
            self._sync_scene_to_legacy()
            self._bump_scene()
            self.update()
            return
        if not self.redo_strokes:
            return
        item = self.redo_strokes.pop()
        if item.pop("_replace_last", False) and self.strokes:
            self.strokes[-1] = item
        else:
            self.strokes.append(item)
        self._sync_legacy_into_scene()
        self._bump_scene()
        self.update()

    def clear_board(self) -> None:
        sc = self._sync_legacy_into_scene()
        sc.clear_all(record_undo=True)
        self._sync_scene_to_legacy()
        self.redo_strokes.clear()
        self._selected_board_image_idx = None
        self._obj_interaction = None
        self.current_stroke.clear()
        self.shape_toast = None
        self.toast_revert_rect = None
        self.hide_lasso_bar()
        self._bump_scene()
        self.update()

    def export_board_png(self, rect: Optional[QRectF] = None, enhance_crop: bool = False) -> Tuple[bytes, QPixmap]:
        # Render clean offscreen scene + live stroke without floating HUD/button overlays obscuring slide content
        w_out = max(640, int(self.width()))
        h_out = max(420, int(self.height()))
        full_pm = QPixmap(w_out, h_out)
        p_out = QPainter(full_pm)
        p_out.setRenderHint(QPainter.Antialiasing)
        self._paint_static_scene(p_out)
        if len(self.current_stroke) >= 2:
            z = max(0.20, float(self.zoom_scale))
            p_out.save()
            p_out.translate(self.pan_x, self.pan_y)
            p_out.scale(z, z)
            is_hi_cur = self.active_tool == "HIGHLIGHTER"
            live_col = QColor(self.highlighter_color if is_hi_cur else self.ink_color)
            if is_hi_cur and live_col.alpha() > 120:
                live_col.setAlpha(96)
            live_thick = max(18, int(self.pen_thickness * 4)) if is_hi_cur else self.pen_thickness
            p_out.setPen(
                QPen(
                    live_col,
                    live_thick,
                    Qt.SolidLine,
                    Qt.FlatCap if is_hi_cur else Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            lpath = QPainterPath(self.current_stroke[0])
            for pt_s in self.current_stroke[1:]:
                lpath.lineTo(pt_s)
            p_out.drawPath(lpath)
            p_out.restore()
        p_out.end()

        if rect is not None and rect.width() > 16 and rect.height() > 16:
            tl = self._board_to_screen(rect.topLeft())
            br = self._board_to_screen(rect.bottomRight())
            scr_rect = QRectF(tl, br).normalized().toRect().intersected(full_pm.rect())
            cropped_pm = full_pm.copy(scr_rect) if scr_rect.width() > 16 and scr_rect.height() > 16 else full_pm
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

    def _get_grid_tile_pixmap(self, zoom: float) -> QPixmap:
        """Cached 120x120 tileable grid QPixmap so panning never redraws hundreds of lines."""
        step = 24
        tile_sz = step * 5
        key = (bool(self.dark_mode), round(float(zoom), 2))
        cached = self._grid_tile_cache.get(key)
        if cached is not None and not cached.isNull():
            return cached
        tile = QPixmap(tile_sz, tile_sz)
        tile.fill(Qt.transparent)
        tp = QPainter(tile)
        minor_w = max(0.5, 1.0 / max(0.20, float(zoom)))
        minor_pen = QPen(QColor(79, 70, 229, 32 if self.dark_mode else 26), minor_w)
        major_pen = QPen(QColor(6, 182, 212, 62 if self.dark_mode else 46), minor_w)
        for i in range(5):
            tp.setPen(major_pen if i == 0 else minor_pen)
            pos = i * step
            tp.drawLine(pos, 0, pos, tile_sz)
            tp.drawLine(0, pos, tile_sz, pos)
        tp.end()
        if len(self._grid_tile_cache) > 24:
            self._grid_tile_cache.clear()
        self._grid_tile_cache[key] = tile
        return tile

    @staticmethod
    def _fit_pixmap_rect(container: QRectF, pm: QPixmap) -> QRectF:
        """Compute exact centered QRectF preserving the pixmap's native aspect ratio (Qt.KeepAspectRatio)."""
        pw = float(max(1, pm.width()))
        ph = float(max(1, pm.height()))
        cw = float(max(1.0, container.width()))
        ch = float(max(1.0, container.height()))
        scale = min(cw / pw, ch / ph)
        dw = pw * scale
        dh = ph * scale
        return QRectF(
            container.left() + 0.5 * (cw - dw),
            container.top() + 0.5 * (ch - dh),
            dw,
            dh,
        )

    def _selected_delete_handle_board_pos(self, sobj: CanvasObject) -> QPointF:
        """Return world board coordinates of the red ✕ delete badge at the selected box's top-right corner."""
        inv_z = 1.0 / max(0.25, float(self.zoom_scale))
        off = 14.0 * inv_z
        lx = sobj.x + sobj.w + off
        ly = sobj.y - off
        if abs(sobj.rotation) > 1e-3:
            cx, cy = sobj.center
            rx, ry = rotate_point(lx, ly, cx, cy, sobj.rotation)
            return QPointF(rx, ry)
        return QPointF(lx, ly)

    def _paint_static_scene(self, painter: QPainter) -> bool:
        """Render background, grid, slide, committed strokes & math annotations cleanly."""
        has_active_reveal = False
        bg_col = QColor(BG_DARK if self.dark_mode else "#F8FAFC")
        painter.fillRect(self.rect(), bg_col)

        z = max(0.20, float(self.zoom_scale))
        painter.save()
        painter.translate(self.pan_x, self.pan_y)
        painter.scale(z, z)

        # 1. Single Cached Tileable Caro Grid (mapped into Pan/Zoom board viewport)
        if self.grid_enabled:
            step = 24
            tile_sz = step * 5
            bx0 = int(math.floor((-self.pan_x / z) / tile_sz)) * tile_sz - tile_sz
            bx1 = int(math.ceil((self.width() - self.pan_x) / z)) + tile_sz * 2
            by0 = int(math.floor((-self.pan_y / z) / tile_sz)) * tile_sz - tile_sz
            by1 = int(math.ceil((self.height() - self.pan_y) / z)) + tile_sz * 2
            tile_pm = self._get_grid_tile_pixmap(z)
            painter.drawTiledPixmap(QRectF(bx0, by0, bx1 - bx0, by1 - by0), tile_pm)

        # 2 & 3. Slide / Document Rendering & Committed Strokes (Continuous Scroll vs Single-Page)
        now_ms = self._now_ms()
        rendered_continuous = False
        if (
            self.continuous_scroll_mode
            and self.slide_visible
            and self.lecture_doc is not None
            and self.lecture_doc.page_count > 0
            and not bool((self.slides[0] if self.slides else {}).get("is_tutorial", False))
        ):
            rendered_continuous = True
            layout = self._compute_continuous_page_rects()
            vis_top = (-self.pan_y / z) - 180.0
            vis_bot = ((self.height() - self.pan_y) / z) + 180.0
            for p_i, cont_r, canon_r, scale_r in layout:
                draw_r = cont_r.translated(0.0, -self.doc_scroll_y)
                if draw_r.bottom() < vis_top or draw_r.top() > vis_bot:
                    continue
                page_pm = self._get_page_qpixmap(
                    p_i,
                    int(round(draw_r.width() * max(1.0, z))),
                    int(round(draw_r.height() * max(1.0, z))),
                )
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                border_col = QColor(NEON_CYAN if p_i == self.current_slide_idx else SECONDARY_TEAL)
                painter.setPen(QPen(border_col, 2.0 if p_i == self.current_slide_idx else 1.1))
                painter.drawRoundedRect(draw_r.adjusted(-1, -1, 1, 1), 6.0, 6.0)
                if page_pm is not None and not page_pm.isNull():
                    fit_r = self._fit_pixmap_rect(draw_r, page_pm)
                    painter.drawPixmap(fit_r.toRect(), page_pm)

                # Draw per-page vector strokes & AI annotations mapped cleanly onto this page rect
                layer = self._ensure_slide_layer(p_i)
                for s in layer.get("strokes") or []:
                    pts = s.get("points") or []
                    if len(pts) < 2:
                        continue
                    col = QColor(s["color"])
                    is_hi = bool(s.get("is_highlighter", False))
                    if is_hi and col.alpha() > 120:
                        col.setAlpha(96)
                    painter.setPen(
                        QPen(
                            col,
                            float(s["thickness"]) * scale_r,
                            Qt.SolidLine,
                            Qt.FlatCap if is_hi else Qt.RoundCap,
                            Qt.RoundJoin,
                        )
                    )
                    tpts = [
                        QPointF(
                            draw_r.left() + (p.x() - canon_r.left()) * scale_r,
                            draw_r.top() + (p.y() - canon_r.top()) * scale_r,
                        )
                        for p in pts
                    ]
                    path = QPainterPath(tpts[0])
                    for p in tpts[1:]:
                        path.lineTo(p)
                    painter.drawPath(path)

                for ann in layer.get("annotations") or []:
                    ax = draw_r.left() + (float(ann.get("x", 40.0)) - canon_r.left()) * scale_r
                    ay = draw_r.top() + (float(ann.get("y", 60.0)) - canon_r.top()) * scale_r
                    aw = float(ann.get("w", 200.0)) * scale_r
                    ah = float(ann.get("h", 44.0)) * scale_r
                    box = QRectF(ax, ay, aw, ah)
                    ann_bg = QColor(15, 23, 42, 215) if self.dark_mode else QColor(255, 255, 255, 240)
                    painter.setBrush(QBrush(ann_bg))
                    painter.setPen(QPen(QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO), 1.2))
                    painter.drawRoundedRect(box, 8.0, 8.0)
                    ann_pm: Optional[QPixmap] = ann.get("pixmap")
                    if ann_pm is not None and not ann_pm.isNull():
                        painter.drawPixmap(
                            QRectF(ax + 6.0, ay + 4.0, max(12.0, aw - 12.0), max(12.0, ah - 8.0)).toRect(),
                            ann_pm,
                        )
                    else:
                        painter.setPen(QColor(ann.get("color") or (NEON_CYAN if self.dark_mode else "#0F172A")))
                        painter.setFont(QFont("Segoe UI", max(10, int(ann.get("font_size") or 13)), QFont.Bold))
                        painter.drawText(box.adjusted(8.0, 4.0, -8.0, -4.0), Qt.AlignVCenter | Qt.AlignLeft, str(ann.get("text") or ""))

        if not rendered_continuous and self.slide_visible and self.slides and 0 <= self.current_slide_idx < len(self.slides):
            slide = self.slides[self.current_slide_idx]
            is_tut = bool(slide.get("is_tutorial", False))
            s_rect = self._compute_active_slide_rect()
            card_bg = QColor(15, 23, 42, 200) if self.dark_mode else QColor(255, 255, 255, 240)
            card_fg = QColor(TEXT_LIGHT if self.dark_mode else "#0F172A")

            doc_pm: Optional[QPixmap] = None
            if (
                self.lecture_doc is not None
                and not is_tut
                and self.current_slide_idx < self.lecture_doc.page_count
            ):
                doc_pm = self._get_page_qpixmap(
                    self.current_slide_idx,
                    int(round(s_rect.width() * max(1.0, z))),
                    int(round(s_rect.height() * max(1.0, z))),
                )

            pm: Optional[QPixmap] = slide.get("pixmap")
            if doc_pm is not None and not doc_pm.isNull():
                fit_r = self._fit_pixmap_rect(s_rect, doc_pm)
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                painter.setPen(QPen(QColor(SECONDARY_TEAL), 1.4))
                painter.drawRoundedRect(fit_r.adjusted(-1, -1, 1, 1), 6.0, 6.0)
                painter.drawPixmap(fit_r.toRect(), doc_pm)
            elif pm is not None and not pm.isNull():
                fit_r = self._fit_pixmap_rect(s_rect, pm)
                painter.setBrush(QBrush(card_bg))
                painter.setPen(QPen(QColor(BORDER_SLATE if self.dark_mode else "#CBD5E1"), 1.2))
                painter.drawRoundedRect(fit_r.adjusted(-1, -1, 1, 1), 8.0, 8.0)
                painter.drawPixmap(fit_r.toRect(), pm)
            elif not bool(slide.get("is_blank", False)):
                painter.setBrush(QBrush(card_bg))
                painter.setPen(QPen(QColor(BORDER_SLATE if self.dark_mode else "#CBD5E1"), 1.2))
                painter.drawRoundedRect(s_rect, 12.0, 12.0)

                painter.setPen(QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO))
                painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
                painter.drawText(
                    QRectF(s_rect.left() + 20, s_rect.top() + 14, max(80.0, s_rect.width() - 40), 28),
                    Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap,
                    slide["title"],
                )

                painter.setPen(QColor(TEXT_MUTED if self.dark_mode else "#475569"))
                painter.setFont(QFont("Segoe UI", 10.5))
                painter.drawText(
                    QRectF(s_rect.left() + 20, s_rect.top() + 44, max(80.0, s_rect.width() - 40), 24),
                    Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap,
                    slide["subtitle"],
                )

                painter.setPen(card_fg)
                painter.setFont(QFont("Segoe UI", 10.5))
                cur_y = s_rect.top() + 78.0
                max_y = s_rect.bottom() - 72.0
                b_w = max(80.0, s_rect.width() - 44.0)
                for b in slide["bullets"]:
                    if cur_y >= max_y:
                        break
                    b_rect = QRectF(s_rect.left() + 22.0, cur_y, b_w, max(24.0, max_y - cur_y))
                    used = painter.drawText(
                        b_rect,
                        Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap,
                        OutputSanitizer.sanitize_plain_text(b),
                    )
                    cur_y += max(26.0, float(used.height()) + 8.0)

                f_rect = QRectF(s_rect.left() + 20, s_rect.bottom() - 56, max(80.0, s_rect.width() - 40), 40)
                painter.setBrush(QBrush(QColor(15, 23, 42, 235) if self.dark_mode else QColor("#EEF2FF")))
                painter.setPen(QPen(QColor(PRIMARY_INDIGO), 1.2))
                painter.drawRoundedRect(f_rect, 8.0, 8.0)
                painter.setPen(QColor(SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO))
                painter.setFont(QFont("Segoe UI", 10.5, QFont.Bold))
                painter.drawText(f_rect.adjusted(8, 2, -8, -2), Qt.AlignCenter | Qt.TextWordWrap, OutputSanitizer.sanitize_plain_text(slide["formula"]))

        # 2B & 3. Unified Object-Based Canvas Rendering (sorted by z_index, with full rotation & Sticky Notes)
        if not rendered_continuous:
            sc = self._sync_legacy_into_scene()
            if len(sc.objects) > 64:
                margin_b = 120.0 / z
                vx = (-self.pan_x / z) - margin_b
                vy = (-self.pan_y / z) - margin_b
                vw = (max(1.0, float(self.width())) / z) + 2.0 * margin_b
                vh = (max(1.0, float(self.height())) / z) + 2.0 * margin_b
                render_objs = sc.visible_objects_in_rect(vx, vy, vw, vh)
            else:
                render_objs = sc.ordered_objects()
            for obj in render_objs:
                painter.save()
                cx, cy = obj.center
                if abs(obj.rotation) > 1e-3:
                    painter.translate(cx, cy)
                    painter.rotate(obj.rotation)
                    painter.translate(-cx, -cy)

                if obj.obj_type == "image":
                    img_pm: Optional[QPixmap] = obj.pixmap
                    if img_pm is not None and not img_pm.isNull():
                        img_rect = QRectF(obj.x, obj.y, obj.w, obj.h)
                        painter.drawPixmap(img_rect.toRect(), img_pm)
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor(BORDER_SLATE), 1.0))
                        painter.drawRoundedRect(img_rect, 4.0, 4.0)
                elif obj.obj_type in ("stroke", "shape"):
                    if len(obj.points) >= 2:
                        col = QColor(obj.color)
                        if not col.isValid():
                            col = QColor(NEON_CYAN)
                        if obj.is_highlighter and col.alpha() > 120:
                            col.setAlpha(96)
                        pen = QPen(
                            col,
                            obj.thickness,
                            Qt.SolidLine,
                            Qt.FlatCap if obj.is_highlighter else Qt.RoundCap,
                            Qt.RoundJoin,
                        )
                        painter.setPen(pen)
                        if obj.fill_color:
                            painter.setBrush(QBrush(QColor(obj.fill_color)))
                        else:
                            painter.setBrush(Qt.NoBrush)
                        path = QPainterPath(QPointF(obj.points[0][0], obj.points[0][1]))
                        for px, py in obj.points[1:]:
                            path.lineTo(QPointF(px, py))
                        painter.drawPath(path)
                elif obj.obj_type == "sticky":
                    box = QRectF(obj.x, obj.y, obj.w, obj.h)
                    bg = QColor(obj.bg_color or "#FEF08A")
                    bd = QColor(obj.border_color or "#CA8A04")
                    fg = QColor(obj.color or "#1E293B")
                    # Soft drop shadow
                    painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(box.translated(3.0, 4.0), 8.0, 8.0)
                    painter.setBrush(QBrush(bg))
                    painter.setPen(QPen(bd, 1.8))
                    painter.drawRoundedRect(box, 8.0, 8.0)
                    # Top sticky accent band
                    painter.fillRect(QRectF(box.left() + 1.0, box.top() + 1.0, box.width() - 2.0, 10.0), bd)
                    painter.setPen(fg)
                    painter.setFont(QFont("Segoe UI", max(10, int(obj.font_size)), QFont.DemiBold))
                    painter.drawText(
                        box.adjusted(10.0, 16.0, -10.0, -8.0),
                        Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap,
                        obj.text,
                    )
                elif obj.obj_type == "text":
                    box = QRectF(obj.x, obj.y, obj.w, obj.h)
                    reveal_until = float(obj.reveal_until_ms or 0.0)
                    if reveal_until > now_ms:
                        has_active_reveal = True
                        alpha_f = max(0.15, min(1.0, 1.0 - (reveal_until - now_ms) / 420.0))
                        painter.setOpacity(alpha_f)
                    if obj.bg_color:
                        ann_bg = QColor(obj.bg_color)
                    else:
                        ann_bg = QColor(15, 23, 42, 222) if self.dark_mode else QColor(255, 255, 255, 242)
                    ann_border = QColor(obj.border_color or (SECONDARY_TEAL if self.dark_mode else PRIMARY_INDIGO))
                    painter.setBrush(QBrush(ann_bg))
                    painter.setPen(QPen(ann_border, 1.3))
                    painter.drawRoundedRect(box, 8.0, 8.0)

                    ann_pm: Optional[QPixmap] = obj.pixmap
                    if ann_pm is not None and not ann_pm.isNull():
                        target_r = QRectF(obj.x + 6.0, obj.y + 4.0, max(12.0, obj.w - 12.0), max(12.0, obj.h - 8.0))
                        painter.drawPixmap(target_r.toRect(), ann_pm)
                    else:
                        painter.setPen(QColor(obj.color or (NEON_CYAN if self.dark_mode else "#0F172A")))
                        painter.setFont(QFont("Segoe UI", max(10, int(obj.font_size or 14)), QFont.DemiBold))
                        painter.drawText(
                            box.adjusted(10.0, 6.0, -10.0, -6.0),
                            Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap,
                            obj.text,
                        )
                painter.restore()

            # 3C. Selection Bounding Boxes, 8 Resize Handles & Top Rotate Handle
            sel_objs = sc.selected_objects()
            if sel_objs:
                inv_z = 1.0 / max(0.25, z)
                for sobj in sel_objs:
                    painter.save()
                    cx, cy = sobj.center
                    if abs(sobj.rotation) > 1e-3:
                        painter.translate(cx, cy)
                        painter.rotate(sobj.rotation)
                        painter.translate(-cx, -cy)
                    sbox = QRectF(sobj.x, sobj.y, sobj.w, sobj.h)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(NEON_CYAN), 1.8 * inv_z, Qt.DashLine))
                    painter.drawRect(sbox)

                    if len(sel_objs) == 1 and not sobj.locked:
                        rot_off = 28.0 * inv_z
                        top_mid = QPointF(cx, sobj.y)
                        rot_pt = QPointF(cx, sobj.y - rot_off)
                        painter.setPen(QPen(QColor(NEON_CYAN), 1.6 * inv_z))
                        painter.drawLine(top_mid, rot_pt)
                        # Rotate handle circle
                        r_rad = 6.5 * inv_z
                        painter.setBrush(QBrush(QColor(PRIMARY_INDIGO)))
                        painter.setPen(QPen(QColor(NEON_CYAN), 1.6 * inv_z))
                        painter.drawEllipse(rot_pt, r_rad, r_rad)
                        # 8 Resize square handles
                        h_half = 4.8 * inv_z
                        painter.setBrush(QBrush(QColor("#FFFFFF")))
                        painter.setPen(QPen(QColor(PRIMARY_INDIGO), 1.5 * inv_z))
                        local_pts = [
                            (sobj.x, sobj.y),
                            (cx, sobj.y),
                            (sobj.x + sobj.w, sobj.y),
                            (sobj.x + sobj.w, cy),
                            (sobj.x + sobj.w, sobj.y + sobj.h),
                            (cx, sobj.y + sobj.h),
                            (sobj.x, sobj.y + sobj.h),
                            (sobj.x, cy),
                        ]
                        for hx, hy in local_pts:
                            painter.drawRect(QRectF(hx - h_half, hy - h_half, 2.0 * h_half, 2.0 * h_half))
                        # On-bounding-box red Delete handle (✕) at outer top-right corner
                        del_off = 14.0 * inv_z
                        dx_btn = sobj.x + sobj.w + del_off
                        dy_btn = sobj.y - del_off
                        d_rad = 8.5 * inv_z
                        painter.setBrush(QBrush(QColor(ALERT_ROSE)))
                        painter.setPen(QPen(QColor("#FFFFFF"), 1.5 * inv_z))
                        painter.drawEllipse(QPointF(dx_btn, dy_btn), d_rad, d_rad)
                        x_half = 3.6 * inv_z
                        painter.setPen(QPen(QColor("#FFFFFF"), 1.8 * inv_z, Qt.SolidLine, Qt.RoundCap))
                        painter.drawLine(QPointF(dx_btn - x_half, dy_btn - x_half), QPointF(dx_btn + x_half, dy_btn + x_half))
                        painter.drawLine(QPointF(dx_btn + x_half, dy_btn - x_half), QPointF(dx_btn - x_half, dy_btn + x_half))
                    painter.restore()

        painter.restore()
        return has_active_reveal

    def paintEvent(self, _event) -> None:
        z = max(0.20, float(self.zoom_scale))
        scene_key = (
            self.width(),
            self.height(),
            self.dark_mode,
            self.grid_enabled,
            self.slide_visible,
            self.current_slide_idx,
            self._scene_rev,
            round(z, 3),
            round(self.pan_x, 1),
            round(self.pan_y, 1),
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

        # Active live stroke & Lasso (transformed into board viewport coordinates)
        painter.save()
        painter.translate(self.pan_x, self.pan_y)
        painter.scale(z, z)
        if len(self.current_stroke) >= 2:
            is_hi_cur = self.active_tool == "HIGHLIGHTER"
            live_col = QColor(self.highlighter_color if is_hi_cur else self.ink_color)
            if is_hi_cur and live_col.alpha() > 120:
                live_col.setAlpha(96)
            live_thick = max(18, int(self.pen_thickness * 4)) if is_hi_cur else self.pen_thickness
            pen = QPen(
                live_col,
                live_thick,
                Qt.SolidLine,
                Qt.FlatCap if is_hi_cur else Qt.RoundCap,
                Qt.RoundJoin,
            )
            painter.setPen(pen)
            preview_pts = self.current_stroke
            if self.continuous_scroll_mode and self.lecture_doc is not None:
                for p_i, cont_r, canon_r, scale_r in self._compute_continuous_page_rects():
                    if p_i == self.current_slide_idx:
                        draw_r = cont_r.translated(0.0, -self.doc_scroll_y)
                        preview_pts = [
                            QPointF(
                                draw_r.left() + (p.x() - canon_r.left()) * scale_r,
                                draw_r.top() + (p.y() - canon_r.top()) * scale_r,
                            )
                            for p in self.current_stroke
                        ]
                        break
            if self.active_modifier_mode == "LINE_LOCK" and not is_hi_cur and len(preview_pts) >= 2:
                painter.drawLine(preview_pts[0], preview_pts[-1])
            else:
                path = QPainterPath(preview_pts[0])
                for p in preview_pts[1:]:
                    path.lineTo(p)
                painter.drawPath(path)

        # Fading red laser pointer trail
        now_ms = self._now_ms()
        while self.laser_trail and (now_ms - self.laser_trail[0][1]) > 450.0:
            self.laser_trail.popleft()
        if len(self.laser_trail) >= 2:
            pts_t = list(self.laser_trail)
            for idx in range(len(pts_t) - 1):
                p0, _ = pts_t[idx]
                p1, t1 = pts_t[idx + 1]
                fade = max(0.05, min(1.0, 1.0 - (now_ms - t1) / 450.0))
                painter.setPen(QPen(QColor(244, 63, 94, int(210 * fade)), max(2.0, 6.0 * fade) / z, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(p0, p1)
            QTimer.singleShot(25, self.update)

        # Smart Lasso polygon, Marquee Multi-Select, Shape Creation Preview, or Region Selection overlay
        if self._marquee_rect is not None:
            painter.setBrush(QBrush(QColor(6, 182, 212, 36)))
            painter.setPen(QPen(QColor(NEON_CYAN), 1.6 / z, Qt.DashLine))
            painter.drawRect(self._marquee_rect)
        elif self._shape_preview_rect is not None:
            pr = self._shape_preview_rect
            pts_prev = make_shape_points(self.active_shape_kind, pr.left(), pr.top(), pr.width(), pr.height())
            if len(pts_prev) >= 2:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(self.ink_color, max(2.0, float(self.pen_thickness)), Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
                spath = QPainterPath(QPointF(pts_prev[0][0], pts_prev[0][1]))
                for px, py in pts_prev[1:]:
                    spath.lineTo(QPointF(px, py))
                painter.drawPath(spath)
        elif len(self.lasso_points) >= 2:
            lpath = QPainterPath(self.lasso_points[0])
            for p in self.lasso_points[1:]:
                lpath.lineTo(p)
            if self.lasso_region is not None:
                lpath.closeSubpath()
            painter.setBrush(QBrush(QColor(124, 58, 237, 40)))
            painter.setPen(QPen(QColor(NEON_CYAN), 2.0 / z, Qt.DashLine))
            painter.drawPath(lpath)
        elif self.selection_start and self.selection_end:
            sel_rect = QRectF(self.selection_start, self.selection_end).normalized()
            painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
            painter.setPen(QPen(QColor(ALERT_ROSE), 2.0 / z))
            painter.drawRoundedRect(sel_rect, 8.0, 8.0)
        painter.restore()

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
            elif self.cursor_mode == "TWO_HAND_NAV":
                painter.setBrush(QBrush(QColor(6, 182, 212, 52)))
                painter.setPen(QPen(QColor(NEON_CYAN), 2.2))
                painter.drawEllipse(cp, 20.0, 20.0)
                painter.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0))
                painter.drawLine(cp + QPointF(-10.0, 0.0), cp + QPointF(10.0, 0.0))
                painter.drawLine(cp + QPointF(0.0, -10.0), cp + QPointF(0.0, 10.0))
            elif self.cursor_mode == "PAN_ERASE":
                er_r = max(24.0, min(110.0, self.dynamic_eraser_radius * z))
                painter.setBrush(QBrush(QColor(244, 63, 94, 50)))
                painter.setPen(QPen(QColor(ALERT_ROSE), 2.0, Qt.DashLine))
                painter.drawEllipse(cp, er_r, er_r)
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
                painter.setBrush(QBrush(QColor(244, 63, 94, 195)))
                painter.setPen(QPen(QColor("#FFFFFF"), 1.6))
                painter.drawEllipse(cp, 7.5, 7.5)
            else:
                # Dual-ring glowing hover cursor for crisp 60Hz visibility from 3m
                painter.setBrush(QBrush(QColor(6, 182, 212, 42)))
                painter.setPen(QPen(QColor(NEON_CYAN), 1.8))
                painter.drawEllipse(cp, 14.0, 14.0)
                painter.setBrush(QBrush(QColor(244, 63, 94, 190)))
                painter.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0))
                painter.drawEllipse(cp, 5.5, 5.5)

        # 7. Non-Intrusive Top-Center Floating Gesture Toast & Live Gesture Recognition HUD Pill
        toast_active = False
        if self.gesture_toast is not None:
            if self._now_ms() <= float(self.gesture_toast.get("until_ms", 0.0)):
                toast_active = True
                msg_g = str(self.gesture_toast.get("text") or "")
                tw_g = min(max(210.0, len(msg_g) * 9.5 + 36.0), max(220.0, self.width() - 80.0))
                gx = 0.5 * (self.width() - tw_g)
                gy = 12.0
                g_rect = QRectF(gx, gy, tw_g, 32.0)
                painter.setBrush(QBrush(QColor(15, 23, 42, 232)))
                painter.setPen(QPen(QColor(NEON_CYAN), 1.6))
                painter.drawRoundedRect(g_rect, 16.0, 16.0)
                painter.setPen(QColor(TEXT_LIGHT))
                painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                painter.drawText(g_rect, Qt.AlignCenter, msg_g)
                QTimer.singleShot(40, self.update)
            else:
                self.gesture_toast = None

        if self.live_gesture_hud_text:
            g_txt = self.live_gesture_hud_text
            gw = min(max(210.0, len(g_txt) * 8.6 + 34.0), max(220.0, self.width() - 80.0))
            gx = 0.5 * (self.width() - gw)
            gy = 50.0 if toast_active else 12.0
            hud_rect = QRectF(gx, gy, gw, 30.0)
            hud_bg = QColor(15, 23, 42, 228) if self.dark_mode else QColor(255, 255, 255, 238)
            hud_fg = QColor(TEXT_LIGHT if self.dark_mode else "#0F172A")
            painter.setBrush(QBrush(hud_bg))
            painter.setPen(QPen(QColor(self.live_gesture_hud_accent), 1.8))
            painter.drawRoundedRect(hud_rect, 15.0, 15.0)
            painter.setPen(hud_fg)
            painter.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
            painter.drawText(hud_rect, Qt.AlignCenter, g_txt)


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
            "• Nhận diện 2 Bàn Tay 3D Fail-Open + ROI 256x256 (0.3m–3m) + One-Euro Smoothing.\n"
            "• Phối hợp 2 tay (Dual-Pinch Zoom, Dual-Palm Pan, Tay phụ đổi công cụ/Undo).\n"
            "• Face ID 5-điểm Affine 112x112 (ArcFace 512-D) & bảo mật PIN PBKDF2 cục bộ.\n"
            "• Máy trạng thái vẽ 4 tầng (Schmitt <0.28 / >0.40, vẽ 1 ngón, cắt đuôi 105ms).\n"
            "• Chuẩn hóa hình học lai: PCA + FitEllipse + ApproxPolyDP + $1/$P + Catmull-Rom.\n"
            "• Agent Bảng trắng JSON [0..1000] & Gemini API bền bỉ ngoại tuyến."
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
        save_ui_api_settings(ui_api_key=key, onboarding_skipped=False)
        self.saved_key = key
        self.accept()

    def _on_offline(self) -> None:
        save_ui_api_settings(onboarding_skipped=True)
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
        self.api_key: str = self.api_key_state.key
        self.gemini_conn_status: ApiConnectionStatus = ApiConnectionStatus.UNCONFIGURED
        self.perf_pipeline = PerformancePipeline()
        self.local_ai_manager = LocalAIManager()
        self.gemini_worker: Optional[GeminiVisionWorker] = None
        self.hybrid_worker: Optional[HybridAiWorker] = None
        self.discovery_worker: Optional[LocalDiscoveryWorker] = None
        self.health_worker: Optional[GeminiHealthCheckWorker] = None
        self.doc_import_worker: Optional[DocumentImportWorker] = None
        self.voice_worker: Optional[VoiceQuestionWorker] = None
        self.warmup_worker: Optional[EngineWarmupWorker] = None
        self.active_gguf_path: str = ""
        self.preferred_cloud_model: str = ""
        self._pending_attachment: Optional[ParsedAttachment] = None
        self.spotlight_overlay: Optional[SpotlightTourOverlay] = None
        self._stream_last_render_t: float = 0.0
        self._load_persisted_local_ai_settings()
        self.dark_mode: bool = True
        self.chat_history: List[Dict[str, str]] = []
        self.last_ai_raw_text: str = ""
        self._stream_raw_accum: str = ""
        self._active_lasso_action: Optional[LassoQuickAction] = None
        self._active_lasso_region: Optional[LassoSelectionRegion] = None
        self._ink_palette: List[str] = ["#00F5D4", "#FEE440", "#FFFFFF", "#38B000", "#FF007F", "#FF6D00"]
        self._ink_palette_idx: int = 0
        self._last_nav_badge: str = ""
        self._last_nav_badge_until: float = 0.0
        self.is_recording_audio = False
        self.audio_frames: List[np.ndarray] = []
        self.vu_db: float = -60.0
        self.audio_stream: Optional[Any] = None
        self._shutting_down: bool = False

        self.cam_worker = CameraHandWorker(0)
        self.cam_worker.frame_ready.connect(self._on_camera_packet)
        self.cam_worker.calibration_done.connect(self._on_calibration_done)

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
        self.mini_hud_shortcut = QShortcut(QKeySequence(Qt.Key_H), self)
        self.mini_hud_shortcut.activated.connect(self._on_toggle_mini_hud)
        self.prev_slide_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_slide_shortcut.activated.connect(self._on_prev_slide)
        self.next_slide_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_slide_shortcut.activated.connect(self._on_next_slide)
        self.pgup_shortcut = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        self.pgup_shortcut.activated.connect(self._on_prev_slide)
        self.pgdn_shortcut = QShortcut(QKeySequence(Qt.Key_PageDown), self)
        self.pgdn_shortcut.activated.connect(self._on_next_slide)
        self.reset_zoom_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self.reset_zoom_shortcut.activated.connect(self._on_reset_zoom)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self._on_hotkey_redo)
        self.redo_alt_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.redo_alt_shortcut.activated.connect(self._on_hotkey_redo)
        self.dup_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self.dup_shortcut.activated.connect(self._on_hotkey_duplicate)
        self.del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.del_shortcut.activated.connect(self._on_hotkey_delete)
        self.select_tool_shortcut = QShortcut(QKeySequence("V"), self)
        self.select_tool_shortcut.activated.connect(lambda: self._on_hotkey_tool("SELECT"))
        self.text_tool_shortcut = QShortcut(QKeySequence("T"), self)
        self.text_tool_shortcut.activated.connect(lambda: self._on_hotkey_tool("TEXT"))
        self.sticky_tool_shortcut = QShortcut(QKeySequence("N"), self)
        self.sticky_tool_shortcut.activated.connect(lambda: self._on_hotkey_tool("STICKY"))
        self.shape_tool_shortcut = QShortcut(QKeySequence("S"), self)
        self.shape_tool_shortcut.activated.connect(lambda: self._on_hotkey_tool("SHAPE"))
        self.guide_shortcut = QShortcut(QKeySequence(Qt.Key_F1), self)
        self.guide_shortcut.activated.connect(self._on_show_user_guide)

        self.stack.currentChanged.connect(lambda idx: self.cam_worker.set_auth_mode(idx == 0))
        self.stack.setCurrentIndex(0)
        self._cam_start_timer = QTimer(self)
        self._cam_start_timer.setSingleShot(True)
        self._cam_start_timer.timeout.connect(self._start_camera_deferred)
        self._cam_start_timer.start(40)

        self._warmup_timer = QTimer(self)
        self._warmup_timer.setSingleShot(True)
        self._warmup_timer.timeout.connect(self._start_engine_warmup)
        self._warmup_timer.start(100)

        self._discovery_timer = QTimer(self)
        self._discovery_timer.setSingleShot(True)
        self._discovery_timer.timeout.connect(self._on_discover_local_ai)
        self._discovery_timer.start(450)

        self._onboarding_timer = QTimer(self)
        self._onboarding_timer.setSingleShot(True)
        self._onboarding_timer.timeout.connect(self._maybe_show_first_run_onboarding)
        self._onboarding_timer.start(600)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_vu_meter)
        self.ui_timer.start(80)

        self.extrapolator = Kinematic60HzExtrapolator(min_cutoff=0.8, beta=0.04, d_cutoff=1.0)
        self.cursor_60hz_timer = QTimer(self)
        self.cursor_60hz_timer.timeout.connect(self._on_60hz_cursor_tick)
        self.cursor_60hz_timer.start(16)

        self._app_event_filter_installed = False
        qapp_inst = QApplication.instance()
        if qapp_inst is not None:
            qapp_inst.installEventFilter(self)
            self._app_event_filter_installed = True

    def _load_persisted_local_ai_settings(self) -> None:
        cfg = load_ui_api_settings()
        rm = str(cfg.get("ai_routing_mode") or "").strip().upper()
        if rm in ("LOCAL", "CLOUD", "HYBRID"):
            self.local_ai_manager.routing_mode = AiRoutingMode(rm)
        gp = str(cfg.get("active_gguf_path") or "").strip()
        if gp and Path(gp).exists():
            self.active_gguf_path = gp
            self.local_ai_manager.set_gguf_model(gp)
        self.preferred_cloud_model = str(cfg.get("gemini_model_id") or "").strip()

    def eventFilter(self, watched, event) -> bool:
        from PySide6.QtCore import QEvent

        if (
            not self._shutting_down
            and self.stack.currentIndex() == 1
            and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
        ):
            fw = QApplication.focusWidget()
            if not isinstance(fw, (QLineEdit, QTextBrowser, QComboBox)):
                if event.type() == QEvent.Type.KeyPress:
                    self.canvas._space_panning = True
                    self.canvas.setCursor(Qt.OpenHandCursor)
                else:
                    self.canvas._space_panning = False
                    self.canvas._mouse_pan_last = None
                    self.canvas.unsetCursor()
                return True
        return super().eventFilter(watched, event)

    def _start_camera_deferred(self) -> None:
        if not self._shutting_down and not self.cam_worker.isRunning():
            self.cam_worker.start()

    def _start_audio_stream(self) -> None:
        if self._shutting_down or self.audio_stream is not None:
            return
        if not _ensure_sounddevice() or sd is None:
            return
        try:
            self.audio_stream = sd.InputStream(
                samplerate=44100, channels=1, callback=self._audio_callback
            )
            self.audio_stream.start()
        except Exception as exc:
            _LOGGER.warning("Audio input stream unavailable: %s", exc)
            self.audio_stream = None

    def _stop_audio_stream(self) -> None:
        if self.audio_stream is None:
            return
        try:
            self.audio_stream.stop()
            self.audio_stream.close()
        except Exception as exc:
            _LOGGER.warning("audio_stream close error: %s", exc)
        self.audio_stream = None

    def _audio_callback(self, indata, _frames, _time_info, _status) -> None:
        if self._shutting_down:
            return
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
        # MODULE 2.1: CLEAN STUDIO HEADER BAR (Height: 42px, Zero-Overflow 1366px+)
        # =====================================================================
        self.header_bar = QFrame()
        header = self.header_bar
        header.setFixedHeight(42)
        header.setStyleSheet(
            f"QFrame {{ background: {BG_DARK}; border-bottom: 1px solid {BORDER_SLATE}; }}"
            f"QPushButton {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; "
            f"border-radius: 6px; padding: 2px 5px; min-width: 0px; font-size: 11px; font-weight: 700; }}"
            f"QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; }}"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(6, 4, 6, 4)
        h_layout.setSpacing(4)

        h_layout.addWidget(make_brand_logo_widget(24))
        self.brand_lbl = QLabel("AirTeach AI")
        self.brand_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 800; font-size: 12.5px; border: none;")
        h_layout.addWidget(self.brand_lbl)

        self.teacher_name_lbl = QLabel("GV: Giáo viên")
        self.teacher_name_lbl.setVisible(False)  # Shown in tooltip/profile so header stays ultra-clean on 1366px

        doc_menu_btn = QPushButton("📂 Tài Liệu ▾")
        self.doc_menu_btn = doc_menu_btn
        doc_menu_btn.setFixedHeight(28)
        doc_menu_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        doc_menu_btn.setToolTip("Mở PDF / PPTX / DOCX / Ảnh / Thư mục ảnh, chèn Object Ảnh hoặc xuất PDF/PNG")
        doc_menu = QMenu(doc_menu_btn)
        doc_menu.setStyleSheet(
            f"QMenu {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}"
            f"QMenu::item:selected {{ background: {PRIMARY_INDIGO}; }}"
        )
        doc_menu.addAction("📄 Mở Tài Liệu / Ảnh Nền Slide (PDF / PPTX / DOCX / Ảnh)", self._on_open_slide_file)
        doc_menu.addAction("🖼 Mở Thư Mục Ảnh Thành Slide", self._on_open_image_folder)
        doc_menu.addAction("📌 Chèn Ảnh Thành Object Độc Lập (Di chuyển / Resize / Xoay)", self._on_insert_board_image)
        doc_menu.addAction("➕ Thêm Trang Trắng Mới", lambda: self.canvas.add_blank_page())
        doc_menu.addSeparator()
        doc_menu.addAction("💾 Xuất Bài Giảng Đã Chú Thích (.PDF)", self._on_export_annotated_pdf)
        doc_menu.addAction("🖼 Xuất Toàn Bộ Trang Kèm Chú Thích Ra Ảnh (.PNG)", self._on_export_annotated_pngs)
        doc_menu_btn.setMenu(doc_menu)
        h_layout.addWidget(doc_menu_btn)

        prev_btn = QPushButton("◀")
        prev_btn.setFixedSize(24, 28)
        prev_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0px; min-width: 24px; max-width: 24px; font-size: 11px; font-weight: 700;"
        )
        prev_btn.setToolTip("Trang trước (← / PageUp)")
        prev_btn.clicked.connect(self._on_prev_slide)
        h_layout.addWidget(prev_btn)

        self.page_jump_edit = QLineEdit("1")
        self.page_jump_edit.setFixedSize(34, 26)
        self.page_jump_edit.setAlignment(Qt.AlignCenter)
        self.page_jump_edit.setValidator(QIntValidator(1, 9999, self.page_jump_edit))
        self.page_jump_edit.setToolTip("Nhập số trang rồi nhấn Enter để nhảy nhanh")
        self.page_jump_edit.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {NEON_CYAN}; border: 1px solid {BORDER_SLATE}; border-radius: 5px; padding: 1px 2px; font-size: 11px; font-weight: 800;"
        )
        self.page_jump_edit.returnPressed.connect(self._on_page_jump_submitted)
        h_layout.addWidget(self.page_jump_edit)

        self.slide_counter_lbl = QLabel("/ 1")
        self.slide_counter_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 700; font-size: 11px; border: none; padding: 0 2px;")
        h_layout.addWidget(self.slide_counter_lbl)

        next_btn = QPushButton("▶")
        next_btn.setFixedSize(24, 28)
        next_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0px; min-width: 24px; max-width: 24px; font-size: 11px; font-weight: 700;"
        )
        next_btn.setToolTip("Trang tiếp theo (→ / PageDown)")
        next_btn.clicked.connect(self._on_next_slide)
        h_layout.addWidget(next_btn)

        self.cont_scroll_btn = QPushButton("Cuộn")
        self.cont_scroll_btn.setFixedHeight(28)
        self.cont_scroll_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.cont_scroll_btn.setCheckable(True)
        self.cont_scroll_btn.setChecked(False)
        self.cont_scroll_btn.setToolTip("Bật/tắt chế độ cuộn dọc liên tục cho Word/PDF nhiều trang")
        self.cont_scroll_btn.clicked.connect(self._on_toggle_continuous_scroll)
        h_layout.addWidget(self.cont_scroll_btn)

        self.thumb_toggle_btn = QPushButton("Slides")
        self.thumb_toggle_btn.setFixedHeight(28)
        self.thumb_toggle_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.thumb_toggle_btn.setCheckable(True)
        self.thumb_toggle_btn.setChecked(False)
        self.thumb_toggle_btn.setToolTip("Ẩn/hiện thanh thumbnail xem trước các trang")
        self.thumb_toggle_btn.clicked.connect(self._on_toggle_thumb_strip)
        h_layout.addWidget(self.thumb_toggle_btn)

        explain_slide_hdr_btn = QPushButton("✨ Giải")
        explain_slide_hdr_btn.setFixedHeight(28)
        explain_slide_hdr_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        explain_slide_hdr_btn.setToolTip("Nhờ Trợ lý AI phân tích & giải chi tiết Slide đang trình chiếu")
        explain_slide_hdr_btn.setStyleSheet(
            f"background: #0D9488; color: {TEXT_LIGHT}; border-radius: 7px; padding: 2px 7px; min-width: 0px; font-weight: 700; border: none;"
        )
        explain_slide_hdr_btn.clicked.connect(lambda: self._on_explain_active_slide())
        h_layout.addWidget(explain_slide_hdr_btn)

        self.ai_mode_quick_btn = QPushButton("⚡ Hybrid")
        self.ai_mode_quick_btn.setFixedHeight(28)
        self.ai_mode_quick_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.ai_mode_quick_btn.setToolTip("Chuyển nhanh 1-click giữa Cloud Gemini ↔ Local AI Offline ↔ Hybrid")
        self.ai_mode_quick_btn.clicked.connect(self._on_quick_toggle_ai_mode)
        h_layout.addWidget(self.ai_mode_quick_btn)

        self.local_mgr_btn = QPushButton("🤖 Hub")
        self.local_mgr_btn.setFixedHeight(28)
        self.local_mgr_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.local_mgr_btn.setToolTip("Mở Kho Model AI Local Offline (GGUF từ HuggingFace / GitHub)")
        self.local_mgr_btn.clicked.connect(self._on_open_local_ai_manager)
        h_layout.addWidget(self.local_mgr_btn)

        self.gesture_map_btn = QPushButton("Sơ đồ Cử chỉ")
        self.gesture_map_btn.setFixedHeight(28)
        self.gesture_map_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.gesture_map_btn.setToolTip("Mở Sơ đồ Cử chỉ trực quan 2 cột (1 Tay & 2 Tay) kèm thanh test trực tiếp")
        self.gesture_map_btn.clicked.connect(self._on_show_gesture_cheat_sheet)
        h_layout.addWidget(self.gesture_map_btn)

        self.status_pill_lbl = QLabel(self.perf_pipeline.telemetry.status_pill_text())
        self.status_pill_lbl.setMinimumWidth(54)
        self.status_pill_lbl.setMaximumWidth(165)
        self.status_pill_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.status_pill_lbl.setFixedHeight(26)
        self.status_pill_lbl.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {NEON_CYAN}; border: 1px solid {SECONDARY_TEAL}; border-radius: 10px; padding: 2px 7px; font-size: 10px; font-weight: 700;"
        )
        h_layout.addWidget(self.status_pill_lbl, stretch=1)

        self.rec_btn = QPushButton("🎙")
        self.rec_btn.setFixedSize(28, 28)
        self.rec_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.rec_btn.setToolTip("Ghi âm bài giảng cục bộ (.WAV)")
        self.rec_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 7px; font-weight: 700; border: none; padding: 0px; min-width: 28px; max-width: 28px;"
        )
        self.rec_btn.clicked.connect(self._on_toggle_recording)
        h_layout.addWidget(self.rec_btn)

        self.toggle_cam_btn = QPushButton("Cam")
        self.toggle_cam_btn.setFixedHeight(28)
        self.toggle_cam_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.toggle_cam_btn.setCheckable(True)
        self.toggle_cam_btn.setChecked(False)
        self.toggle_cam_btn.setToolTip("Ẩn/hiện bảng điều khiển Camera & CV")
        self.toggle_cam_btn.clicked.connect(self._on_toggle_studio_camera)
        h_layout.addWidget(self.toggle_cam_btn)

        self.mini_hud_btn = QPushButton("HUD")
        self.mini_hud_btn.setFixedHeight(28)
        self.mini_hud_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.mini_hud_btn.setCheckable(True)
        self.mini_hud_btn.setChecked(True)
        self.mini_hud_btn.setToolTip("Bật/tắt Mini Camera HUD góc phải (Phím H)")
        self.mini_hud_btn.clicked.connect(self._on_toggle_mini_hud)
        h_layout.addWidget(self.mini_hud_btn)

        self.toggle_ai_btn = QPushButton("Trợ Lý AI")
        self.toggle_ai_btn.setFixedHeight(28)
        self.toggle_ai_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.toggle_ai_btn.setCheckable(True)
        self.toggle_ai_btn.setChecked(False)
        self.toggle_ai_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 7px; padding: 2px 7px; min-width: 0px; font-weight: 700; border: none;"
        )
        self.toggle_ai_btn.clicked.connect(self._on_toggle_ai_drawer)
        h_layout.addWidget(self.toggle_ai_btn)

        guide_btn = QPushButton("? Hướng dẫn")
        self.guide_btn = guide_btn
        guide_btn.setFixedHeight(28)
        guide_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        guide_btn.setToolTip("Chạy Hướng Dẫn Tương Tác Spotlight 6 Bước (Phím F1: Mở Sổ Tay Đầy Đủ & Sandbox)")
        guide_btn.clicked.connect(self._on_show_onboarding_tour)
        h_layout.addWidget(guide_btn)

        self.theme_toggle_btn = QPushButton("☀")
        self.theme_toggle_btn.setFixedSize(28, 28)
        self.theme_toggle_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.theme_toggle_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0px; min-width: 28px; max-width: 28px; font-size: 12px; font-weight: 700;"
        )
        self.theme_toggle_btn.setToolTip("Chuyển nhanh Dark Mode ↔ Light Mode toàn ứng dụng")
        self.theme_toggle_btn.clicked.connect(self._on_toggle_theme)
        h_layout.addWidget(self.theme_toggle_btn)

        gear_btn = QPushButton("⚙")
        self.gear_btn = gear_btn
        gear_btn.setFixedSize(28, 28)
        gear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        gear_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 0px; min-width: 28px; max-width: 28px; font-size: 12px; font-weight: 700;"
        )
        gear_btn.setToolTip("Cài đặt lưới, giao diện, thử cử chỉ & tài khoản")
        gear_menu = QMenu(gear_btn)
        self.gear_menu = gear_menu
        gear_menu.setStyleSheet(
            f"QMenu {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}"
            f"QMenu::item:selected {{ background: {PRIMARY_INDIGO}; }}"
        )
        gear_menu.addAction("🧪 Phòng Thử Cử Chỉ Trực Tiếp (Gesture Sandbox)", lambda: self._on_show_user_guide(4))
        gear_menu.addAction("🔦 Hướng Dẫn Tương Tác Spotlight (6 Bước)", self._on_show_onboarding_tour)
        gear_menu.addSeparator()
        gear_menu.addAction("Bật/Tắt Lưới Caro", self._on_toggle_grid)
        gear_menu.addAction("Đổi Giao Diện Sáng / Tối", self._on_toggle_theme)
        gear_menu.addAction("Giới Thiệu AirTeach AI", self._on_show_about)
        gear_menu.addSeparator()
        gear_menu.addAction("Đăng Xuất", self._on_logout)
        gear_btn.setMenu(gear_menu)
        h_layout.addWidget(gear_btn)

        root.addWidget(header)

        # =====================================================================
        # CENTER WORKSPACE: LEFT DOCK (268px) | SMARTBOARD CANVAS | RIGHT DOCK (360px)
        # =====================================================================
        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(0)

        self.left_dock = QFrame()
        self.left_dock.setObjectName("LeftDock")
        self.left_dock.setFixedWidth(268)
        self.left_dock.setStyleSheet(
            f"QFrame#LeftDock {{ background: {SURFACE_DARK}; border-right: 1px solid {BORDER_SLATE}; }}"
            f"QFrame#LeftDock QLabel {{ color: {TEXT_LIGHT}; background: transparent; border: none; }}"
            f"QFrame#LeftDock QCheckBox {{ color: {TEXT_LIGHT}; background: transparent; font-size: 11px; spacing: 5px; }}"
            f"QFrame#LeftDock QPushButton {{ background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 3px 6px; font-size: 11px; font-weight: 700; }}"
            f"QFrame#LeftDock QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; }}"
            f"QFrame#LeftDock QComboBox {{ background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 3px 6px; font-size: 11px; }}"
            f"QFrame#LeftDock QProgressBar {{ background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 5px; }}"
            f"QFrame#LeftDock QProgressBar::chunk {{ background: {SECONDARY_TEAL}; border-radius: 4px; }}"
        )
        l_layout = QVBoxLayout(self.left_dock)
        l_layout.setContentsMargins(9, 8, 9, 8)
        l_layout.setSpacing(5)

        self.pip_cam_lbl = QLabel()
        self.pip_cam_lbl.setFixedSize(248, 158)
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
        cam_ctrl_row.setSpacing(5)
        for c_idx in (0, 1):
            b = QPushButton(f"Cam {c_idx}")
            b.setFixedHeight(26)
            b.clicked.connect(lambda _=False, i=c_idx: self.cam_worker.select_camera(i))
            cam_ctrl_row.addWidget(b, stretch=1)

        self.cam_hw_btn = QPushButton("Tắt Cam")
        self.cam_hw_btn.setFixedHeight(26)
        self.cam_hw_btn.setToolTip("Tạm giải phóng hoặc kết nối lại phần cứng Webcam")
        self.cam_hw_btn.clicked.connect(self._on_toggle_camera_hardware)
        cam_ctrl_row.addWidget(self.cam_hw_btn, stretch=1)
        l_layout.addLayout(cam_ctrl_row)

        cam_opt_row = QHBoxLayout()
        cam_opt_row.setSpacing(5)
        self.hdr_chk = QCheckBox("Auto-HDR")
        self.hdr_chk.setChecked(True)
        self.hdr_chk.toggled.connect(self._on_toggle_hdr)
        cam_opt_row.addWidget(self.hdr_chk)

        self.hd_cam_chk = QCheckBox("HD 720p")
        self.hd_cam_chk.setChecked(bool(self.cam_worker.prefer_hd))
        self.hd_cam_chk.toggled.connect(self._on_toggle_hd_camera)
        cam_opt_row.addWidget(self.hd_cam_chk)

        self.debug_hud_btn = QPushButton("Debug")
        self.debug_hud_btn.setCheckable(True)
        self.debug_hud_btn.setChecked(bool(self.cam_worker.debug_hud_enabled))
        self.debug_hud_btn.setFixedHeight(24)
        self.debug_hud_btn.setToolTip("Bật/tắt Vision Debug HUD (Ctrl+Shift+D)")
        self.debug_hud_btn.clicked.connect(self._on_toggle_vision_debug_hud)
        cam_opt_row.addWidget(self.debug_hud_btn)
        l_layout.addLayout(cam_opt_row)

        cv_failopen_row = QHBoxLayout()
        cv_failopen_row.setSpacing(6)
        self.teacher_lock_chk = QCheckBox("Teacher Lock")
        self.teacher_lock_chk.setChecked(bool(self.cam_worker.teacher_lock_enabled))
        self.teacher_lock_chk.setToolTip(
            "Chỉ lọc tay người khác khi có >=2 người; tự động Bypass khi 1 người hoặc Pose chưa nhận."
        )
        self.teacher_lock_chk.toggled.connect(self._on_toggle_teacher_lock)
        cv_failopen_row.addWidget(self.teacher_lock_chk)

        self.skeleton_chk = QCheckBox("Xương 21đ")
        self.skeleton_chk.setChecked(bool(self.cam_worker.skeleton_overlay_enabled))
        self.skeleton_chk.setToolTip("Vẽ trực tiếp 21 điểm khớp xương 2 bàn tay lên preview camera")
        self.skeleton_chk.toggled.connect(self.cam_worker.set_skeleton_overlay)
        cv_failopen_row.addWidget(self.skeleton_chk)
        l_layout.addLayout(cv_failopen_row)

        perf_row = QHBoxLayout()
        perf_row.setSpacing(6)
        perf_lbl = QLabel("Hiệu năng:")
        perf_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        perf_row.addWidget(perf_lbl)
        self.perf_combo = QComboBox()
        self.perf_combo.addItem("Tự động (i3)", "AUTO")
        self.perf_combo.addItem("Siêu nhẹ (i3 cũ)", "ULTRA_LIGHT")
        self.perf_combo.addItem("Cân bằng", "BALANCED")
        self.perf_combo.addItem("Chất lượng cao", "HIGH_QUALITY")
        self.perf_combo.setCurrentIndex(0)
        self.perf_combo.currentIndexChanged.connect(self._on_perf_profile_changed)
        perf_row.addWidget(self.perf_combo, stretch=1)
        l_layout.addLayout(perf_row)

        self.vision_debug_lbl = QLabel()
        self.vision_debug_lbl.setWordWrap(True)
        self.vision_debug_lbl.setVisible(bool(self.cam_worker.debug_hud_enabled))
        self.vision_debug_lbl.setStyleSheet(
            f"background: {BG_DARK}; color: {NEON_CYAN}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 5px; font-family: Consolas, monospace; font-size: 10px;"
        )
        l_layout.addWidget(self.vision_debug_lbl)

        # 4 Compact 4-Tier HUD Mode & Gesture Badges (2x2 Grid — synced with GestureCheatSheet)
        badge_grid = QGridLayout()
        badge_grid.setSpacing(5)
        self.gesture_badges: Dict[str, QLabel] = {}
        badge_specs = [
            ("DRAWING", "1 Ngón: Vẽ", 0, 0),
            ("HOVER", "2 Ngón: Trỏ", 0, 1),
            ("LOCKED_COOLDOWN", "Chụm: Kéo Obj", 1, 0),
            ("PAN_ERASE", "5 Ngón: Tẩy", 1, 1),
        ]
        for key, text, r, c in badge_specs:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(26)
            lbl.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 11px; font-weight: 700;"
            )
            badge_grid.addWidget(lbl, r, c)
            self.gesture_badges[key] = lbl
        l_layout.addLayout(badge_grid)

        self.cam_gesture_map_btn = QPushButton("Sơ đồ Cử chỉ (1 && 2 Tay)")
        self.cam_gesture_map_btn.setFixedHeight(30)
        self.cam_gesture_map_btn.setToolTip("Mở bảng tra cứu Cử chỉ 1 & 2 Tay kèm thanh Test trực tiếp")
        self.cam_gesture_map_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: 1px solid {NEON_CYAN}; border-radius: 7px; font-size: 11.5px; font-weight: 700;"
        )
        self.cam_gesture_map_btn.clicked.connect(self._on_show_gesture_cheat_sheet)
        l_layout.addWidget(self.cam_gesture_map_btn)

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
        self.left_dock.setVisible(False)

        # Center Smartboard Canvas + Collapsible Slide Thumbnail Strip
        center_panel = QWidget()
        center_vbox = QVBoxLayout(center_panel)
        center_vbox.setContentsMargins(0, 0, 0, 0)
        center_vbox.setSpacing(0)

        self.canvas = SmartboardCanvas()
        self.canvas.region_cropped_for_ai.connect(self._on_region_cropped_for_ai)
        self.canvas.lasso_action_requested.connect(self._on_lasso_quick_action)
        self.canvas.slides_changed.connect(self._update_slide_counter)
        self.canvas.file_dropped.connect(self._import_document_async)
        self.canvas.pie_tool_selected.connect(self._on_canvas_pie_tool_selected)
        self.canvas.guide_requested.connect(self._on_show_user_guide)
        self.canvas.mini_hud.calibrate_clicked.connect(self._on_auto_calibrate)
        self.canvas.mini_hud.gesture_map_clicked.connect(self._on_show_gesture_cheat_sheet)
        self.canvas.mini_hud.close_clicked.connect(lambda: self._set_mini_hud_visible(False))
        center_vbox.addWidget(self.canvas, stretch=1)

        self.thumb_strip = SlideThumbnailStrip()
        self.thumb_strip.slide_selected.connect(self._on_jump_to_slide)
        center_vbox.addWidget(self.thumb_strip)
        self.thumb_strip.rebuild(
            self.canvas.slides,
            self.canvas.current_slide_idx,
            self.canvas.slide_annotations,
            self.canvas.lecture_doc,
        )
        self.thumb_strip.setVisible(False)

        workspace.addWidget(center_panel, stretch=1)

        # Right Dock (Glassmorphic AI Assistant Drawer)
        self.right_dock = QFrame()
        self.right_dock.setObjectName("RightDock")
        self.right_dock.setFixedWidth(360)
        self.right_dock.setStyleSheet(
            f"QFrame#RightDock {{ background: rgba(15, 23, 42, 244); border-left: 1.5px solid {SECONDARY_TEAL}; }}"
            f"QFrame#RightDock QLabel {{ color: {TEXT_LIGHT}; background: transparent; border: none; }}"
            f"QFrame#RightDock QCheckBox {{ color: {TEXT_LIGHT}; background: transparent; font-size: 11px; spacing: 5px; }}"
            f"QFrame#RightDock QPushButton {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 3px 7px; font-size: 11px; font-weight: 700; }}"
            f"QFrame#RightDock QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; }}"
            f"QFrame#RightDock QComboBox, QFrame#RightDock QLineEdit {{ background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; padding: 4px 7px; font-size: 11.5px; }}"
        )
        r_layout = QVBoxLayout(self.right_dock)
        r_layout.setContentsMargins(10, 10, 10, 10)
        r_layout.setSpacing(6)

        ai_hdr = QHBoxLayout()
        ai_hdr.setSpacing(6)
        ai_title = QLabel("Trợ Lý Sư Phạm AI")
        ai_title.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 800; font-size: 13.5px; border: none; background: transparent;")
        ai_hdr.addWidget(ai_title)
        ai_hdr.addStretch(1)

        self.change_key_btn = QPushButton("API Key")
        self.change_key_btn.setFixedHeight(24)
        self.change_key_btn.clicked.connect(
            lambda: self.api_key_frame.setVisible(not self.api_key_frame.isVisible())
        )
        ai_hdr.addWidget(self.change_key_btn)
        r_layout.addLayout(ai_hdr)

        ai_status_row = QHBoxLayout()
        ai_status_row.setSpacing(6)
        self.api_status_badge = QLabel("Chưa cấu hình")
        self.api_status_badge.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.api_status_badge.setStyleSheet(
            f"background: {BORDER_SLATE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
        )
        ai_status_row.addWidget(self.api_status_badge)
        ai_status_row.addStretch(1)
        r_layout.addLayout(ai_status_row)

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
        self.api_key_edit.setPlaceholderText("1 hoặc nhiều GEMINI_API_KEY (cách nhau dấu phẩy)...")
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

        # Smart Hybrid / Local AI Discovery Strip (clean 2-row layout — zero text clipping)
        local_row = QHBoxLayout()
        local_row.setSpacing(5)
        self.routing_combo = QComboBox()
        self.routing_combo.addItems(["Hybrid", "Local", "Cloud"])
        self.routing_combo.setFixedWidth(84)
        self.routing_combo.setCurrentText(self.local_ai_manager.routing_mode.label_vi)
        self.routing_combo.currentTextChanged.connect(self._on_routing_mode_changed)
        self.local_model_combo = QComboBox()
        self.local_model_combo.addItem("Auto (Local chưa bật)")
        self.local_model_combo.currentIndexChanged.connect(self._on_local_model_selected)
        local_row.addWidget(self.routing_combo)
        local_row.addWidget(self.local_model_combo, stretch=1)
        r_layout.addLayout(local_row)

        local_btn_row = QHBoxLayout()
        local_btn_row.setSpacing(5)
        scan_local_btn = QPushButton("Quét Local")
        scan_local_btn.setFixedHeight(26)
        scan_local_btn.clicked.connect(self._on_discover_local_ai)
        local_mgr_dock_btn = QPushButton("Kho Model Local (GGUF Hub)")
        local_mgr_dock_btn.setFixedHeight(26)
        local_mgr_dock_btn.setToolTip("Tải model GGUF từ HuggingFace / GitHub & chuyển chế độ AI")
        local_mgr_dock_btn.clicked.connect(self._on_open_local_ai_manager)
        local_btn_row.addWidget(scan_local_btn)
        local_btn_row.addWidget(local_mgr_dock_btn, stretch=1)
        r_layout.addLayout(local_btn_row)

        # Full Gemini Cloud Model Selector (persisted via gemini_model_id; zero impact on Header Bar width)
        cloud_row = QHBoxLayout()
        cloud_row.setSpacing(4)
        cloud_lbl = QLabel("Gemini:")
        cloud_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; border: none; background: transparent;"
        )
        cloud_row.addWidget(cloud_lbl)
        self.cloud_model_combo = QComboBox()
        self.cloud_model_combo.setToolTip("Chọn model Gemini Cloud (hoặc Auto để tự động fallback chuỗi tối ưu)")
        for model_id, display_name in AVAILABLE_GEMINI_MODELS:
            self.cloud_model_combo.addItem(display_name, model_id)
        persisted_mid = str(load_ui_api_settings().get("gemini_model_id") or self.preferred_cloud_model or "").strip()
        if persisted_mid:
            found_idx = self.cloud_model_combo.findData(persisted_mid)
            if found_idx >= 0:
                self.cloud_model_combo.setCurrentIndex(found_idx)
            else:
                self.cloud_model_combo.addItem(f"{persisted_mid} (Tùy chỉnh)", persisted_mid)
                self.cloud_model_combo.setCurrentIndex(self.cloud_model_combo.count() - 1)
            self.preferred_cloud_model = persisted_mid
        self.cloud_model_combo.currentIndexChanged.connect(self._on_cloud_model_changed)
        cloud_row.addWidget(self.cloud_model_combo, stretch=1)
        r_layout.addLayout(cloud_row)

        slide_ai_row = QHBoxLayout()
        slide_ai_row.setSpacing(5)
        explain_btn = QPushButton("✨ Giải thích / Giải Slide này")
        explain_btn.setFixedHeight(32)
        explain_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: 1px solid {NEON_CYAN}; border-radius: 7px; font-weight: 800;"
        )
        explain_btn.clicked.connect(lambda: self._on_explain_active_slide())
        slide_ai_row.addWidget(explain_btn, stretch=1)

        self.voice_ask_btn = QPushButton("🎙 Hỏi Giọng")
        self.voice_ask_btn.setFixedHeight(32)
        self.voice_ask_btn.setToolTip("Hỏi AI bằng giọng nói về nội dung Slide đang hiển thị")
        self.voice_ask_btn.setStyleSheet(
            f"background: #0D9488; color: {TEXT_LIGHT}; border-radius: 7px; padding: 0 8px; font-weight: 700;"
        )
        self.voice_ask_btn.clicked.connect(self._on_voice_ask_slide)
        slide_ai_row.addWidget(self.voice_ask_btn)

        self.expand_ai_btn = QPushButton("⇄")
        self.expand_ai_btn.setFixedSize(28, 32)
        self.expand_ai_btn.setToolTip("Mở rộng / thu gọn ngăn Trợ Lý AI")
        self.expand_ai_btn.clicked.connect(self._on_toggle_ai_width)
        slide_ai_row.addWidget(self.expand_ai_btn)
        r_layout.addLayout(slide_ai_row)

        self._quick_ai_actions: List[Tuple[QPushButton, bool]] = []
        for label, prompt_prefix, struct_json in [
            (
                "1. Giải && Chữa Bài Trên Bảng",
                "Hãy phân tích nội dung tài liệu/slide đang trình chiếu cùng chữ viết tay/công thức của giáo viên, kiểm tra lỗi sai nếu có, giải chi tiết từng bước bằng Markdown + LaTeX rõ ràng và trả về hành động viết/vẽ lời giải lên bảng khi phù hợp.",
                False,
            ),
            (
                "2. Tóm Tắt && Tạo Trắc Nghiệm",
                "Hãy tóm tắt trọng tâm trang tài liệu/slide đang hiển thị bằng Markdown + LaTeX rõ ràng, sau đó tạo 3 câu hỏi trắc nghiệm 4 lựa chọn (A/B/C/D) kèm đáp án và giải thích chi tiết từng câu.",
                False,
            ),
            (
                "3. Gợi Ý Sư Phạm && Tương Tác",
                "Hãy gợi ý phương pháp giảng dạy trực quan, câu hỏi thảo luận và ví dụ thực tế sinh động cho nội dung trên trang tài liệu đang trình chiếu.",
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
            self._quick_ai_actions.append((btn, struct_json))

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
            f"• Dùng **Khoanh AI (Lasso)** (phím `L`) để khoanh vùng viết tay → Giải / Viết lên bảng / Làm đẹp LaTeX.\n"
            f"• Hỗ trợ Local AI (GGUF / Ollama / LM Studio) + {DEFAULT_GEMINI_MODEL} bền bỉ ngoại tuyến."
        )
        self.last_ai_raw_text = init_msg
        self.ai_output.setHtml(OutputSanitizer.to_rich_html(init_msg, dark_mode=self.dark_mode))
        r_layout.addWidget(self.ai_output, stretch=1)

        ai_act_row = QHBoxLayout()
        ai_act_row.setSpacing(6)
        self.copy_ai_btn = QPushButton("Sao chép")
        self.copy_ai_btn.setFixedHeight(28)
        self.copy_ai_btn.setToolTip("Sao chép toàn bộ câu trả lời AI vào Clipboard")
        self.copy_ai_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-weight: 700; font-size: 11px;"
        )
        self.copy_ai_btn.clicked.connect(self._on_copy_last_ai_reply)
        ai_act_row.addWidget(self.copy_ai_btn)

        push_board_btn = QPushButton("Đưa nội dung lên bảng")
        push_board_btn.setFixedHeight(28)
        push_board_btn.setToolTip("Tạo ngay một Text Object đầy đủ nội dung trên Bảng (kéo/resize tự do)")
        push_board_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {SECONDARY_TEAL}; border-radius: 6px; font-weight: 700; font-size: 11px;"
        )
        push_board_btn.clicked.connect(self._on_push_last_ai_to_board)
        ai_act_row.addWidget(push_board_btn, stretch=1)
        r_layout.addLayout(ai_act_row)

        self.chat_attach_board_chk = QCheckBox("Đính kèm ngữ cảnh Slide && Bảng vẽ hiện tại")
        self.chat_attach_board_chk.setChecked(True)
        self.chat_attach_board_chk.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        r_layout.addWidget(self.chat_attach_board_chk)

        # Hideable File Attachment Preview Chip (UTF-8 PDF/DOCX/PPTX/Image/TXT)
        self.attached_file_frame = QFrame()
        self.attached_file_frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE_DARK}; border: 1px solid {SECONDARY_TEAL}; border-radius: 7px; }}"
        )
        af_row = QHBoxLayout(self.attached_file_frame)
        af_row.setContentsMargins(7, 4, 5, 4)
        af_row.setSpacing(6)
        self.attached_file_lbl = QLabel("")
        self.attached_file_lbl.setWordWrap(True)
        self.attached_file_lbl.setStyleSheet(
            f"color: {NEON_CYAN}; font-size: 11px; font-weight: 700; border: none; background: transparent;"
        )
        af_row.addWidget(self.attached_file_lbl, stretch=1)
        self.clear_attachment_btn = QPushButton("✕")
        self.clear_attachment_btn.setFixedSize(20, 20)
        self.clear_attachment_btn.setToolTip("Gỡ tệp đính kèm")
        self.clear_attachment_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border: none; border-radius: 5px; font-size: 10px; font-weight: 800; padding: 0px;"
        )
        self.clear_attachment_btn.clicked.connect(self.clear_chat_attachment)
        af_row.addWidget(self.clear_attachment_btn)
        self.attached_file_frame.setVisible(False)
        r_layout.addWidget(self.attached_file_frame)

        chat_in_row = QHBoxLayout()
        chat_in_row.setSpacing(4)
        self.attach_file_btn = QPushButton("📎")
        self.attach_file_btn.setFixedSize(32, 30)
        self.attach_file_btn.setToolTip("Đính kèm tệp PDF / DOCX / PPTX / Ảnh / TXT gửi AI (chuẩn UTF-8)")
        self.attach_file_btn.clicked.connect(self._on_pick_chat_attachment)
        chat_in_row.addWidget(self.attach_file_btn)
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
        self.right_dock.setVisible(False)
        root.addLayout(workspace, stretch=1)

        # =====================================================================
        # SLEEK ROUNDED FLOATING BOTTOM TOOL DOCK (Height: 48px, Zero-Overflow)
        # =====================================================================
        self.bottom_bar = QFrame()
        bottom = self.bottom_bar
        bottom.setFixedHeight(48)
        bottom.setStyleSheet(f"background: {BG_DARK}; border-top: 1px solid {BORDER_SLATE};")
        outer_b = QHBoxLayout(bottom)
        outer_b.setContentsMargins(8, 4, 8, 4)
        outer_b.setSpacing(0)
        outer_b.addStretch(1)

        self.floating_dock = QFrame()
        self.floating_dock.setFixedHeight(38)
        self.floating_dock.setStyleSheet(
            f"QFrame {{ background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 14px; }}"
        )
        b_layout = QHBoxLayout(self.floating_dock)
        b_layout.setContentsMargins(8, 3, 8, 3)
        b_layout.setSpacing(4)

        self._tool_buttons: Dict[str, QPushButton] = {}
        for tool_id, label, tip in [
            ("SELECT", "⬉ Chọn", "Chọn, di chuyển, co giãn 8 góc & xoay Object (Phím V)"),
            ("PEN", "✎ Bút", "Bút vẽ mượt + Nắn hình thông minh"),
            ("HIGHLIGHTER", "🖍 Nhớ", "Bút dạ quang bán trong suốt"),
            ("ERASER", "⌫ Tẩy", "Tẩy nét vẽ & văn bản"),
            ("SHAPE", "◇ Hình", "Vẽ hình khối độc lập (Phím S)"),
            ("TEXT", "T Chữ", "Chèn hộp văn bản độc lập (Phím T)"),
            ("STICKY", "🗒 Note", "Chèn giấy ghi chú Sticky Note (Phím N)"),
            ("PAN", "✥ Kéo", "Di chuyển bảng vô tận (Space + Kéo hoặc Chuột giữa)"),
            ("LASER", "◉ Laser", "Con trỏ Laser thuyết trình"),
            ("LASSO", "✦ AI", "Khoanh vùng gọi Trợ lý AI giải / làm đẹp LaTeX"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setToolTip(tip)
            if tool_id == "SHAPE":
                shape_menu = QMenu(btn)
                shape_menu.setStyleSheet(
                    f"QMenu {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; padding: 4px; }}"
                    f"QMenu::item {{ padding: 5px 12px; border-radius: 5px; }}"
                    f"QMenu::item:selected {{ background: {PRIMARY_INDIGO}; }}"
                )
                for sk, slbl in (
                    ("rect", "▭ Hình chữ nhật"),
                    ("circle", "◯ Hình tròn"),
                    ("triangle", "△ Hình tam giác"),
                    ("arrow", "➜ Mũi tên"),
                    ("line", "╱ Đoạn thẳng"),
                ):
                    shape_menu.addAction(slbl, lambda k=sk: self._select_shape_kind(k))
                btn.setMenu(shape_menu)
            else:
                btn.clicked.connect(lambda _=False, t=tool_id: self._set_tool(t))
            b_layout.addWidget(btn)
            self._tool_buttons[tool_id] = btn
        self._sync_tool_buttons_ui()

        for hex_col in self._ink_palette[:5]:
            c_btn = QPushButton("")
            c_btn.setFixedSize(18, 18)
            c_btn.setToolTip(f"Đổi màu mực {hex_col}")
            c_btn.setStyleSheet(
                f"background: {hex_col}; border: 1.5px solid {TEXT_LIGHT}; border-radius: 9px; padding: 0px;"
            )
            c_btn.clicked.connect(lambda _=False, hc=hex_col: self._set_ink_color(hc))
            b_layout.addWidget(c_btn)

        self.thick_lbl = QLabel("4px")
        self.thick_lbl.setStyleSheet(f"color: {SECONDARY_TEAL}; font-weight: 700; font-size: 10.5px; border: none; background: transparent;")
        b_layout.addWidget(self.thick_lbl)

        thick_slider = QSlider(Qt.Horizontal)
        thick_slider.setRange(2, 14)
        thick_slider.setValue(4)
        thick_slider.setFixedWidth(56)
        thick_slider.setToolTip("Độ dày nét bút (2px – 14px)")
        thick_slider.valueChanged.connect(self._on_thickness_changed)
        b_layout.addWidget(thick_slider)

        undo_btn = QPushButton("↶")
        undo_btn.setFixedSize(30, 30)
        undo_btn.setToolTip("Hoàn tác (Ctrl+Z)")
        undo_btn.clicked.connect(self._on_hotkey_undo)
        b_layout.addWidget(undo_btn)

        redo_btn = QPushButton("↷")
        redo_btn.setFixedSize(30, 30)
        redo_btn.setToolTip("Làm lại (Ctrl+Y)")
        redo_btn.clicked.connect(self.canvas.redo)
        b_layout.addWidget(redo_btn)

        clear_btn = QPushButton("🗑")
        clear_btn.setFixedSize(30, 30)
        clear_btn.setToolTip("Xóa sạch bảng trang hiện tại (Ctrl+Z để hoàn tác)")
        clear_btn.setStyleSheet(
            f"background: {BG_DARK}; color: {ALERT_ROSE}; border: 1px solid {ALERT_ROSE}; border-radius: 7px; font-weight: 700; padding: 0px;"
        )
        clear_btn.clicked.connect(self.canvas.clear_board)
        b_layout.addWidget(clear_btn)

        # Secondary drawing preferences neatly grouped inside compact [⋯] menu so dock never overflows 1366px
        self.smart_shape_btn = QPushButton("◇ Nắn Hình: BẬT")
        self.smart_shape_btn.setVisible(False)
        self.hold_snap_btn = QPushButton("Giữ 350ms: TẮT")
        self.hold_snap_btn.setVisible(False)
        self.draw_mode_btn = QPushButton("Vẽ: 1 Ngón + Chụm")
        self.draw_mode_btn.setVisible(False)

        more_btn = QPushButton("⋯")
        more_btn.setFixedSize(30, 30)
        more_btn.setToolTip("Tùy chọn vẽ: Nắn hình, Giữ 350ms, Chế độ vẽ 1 ngón, Thêm trang mới, Xóa mực")
        more_menu = QMenu(more_btn)
        more_menu.setStyleSheet(
            f"QMenu {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}"
            f"QMenu::item:selected {{ background: {PRIMARY_INDIGO}; }}"
        )
        self._act_smart_shape = more_menu.addAction("◇ Nắn Hình Thông Minh (BẬT)", self._on_toggle_smart_shape)
        self._act_hold_snap = more_menu.addAction("⏱ Giữ 350ms mới nắn hình (TẮT)", self._on_toggle_hold_snap)
        self._act_draw_mode = more_menu.addAction("☝️ Vẽ 1 Ngón Trỏ + Chụm (BẬT)", self._on_toggle_draw_mode)
        more_menu.addSeparator()
        more_menu.addAction("📄 Thêm Trang Trắng Mới", lambda: self.canvas.add_blank_page())
        more_menu.addAction("⊡ Vừa Nội Dung (Fit to Content)", self.canvas.fit_to_content)
        more_menu.addAction("🎯 Về 100% (Ctrl+0)", self._on_reset_zoom)
        more_menu.addAction("🧹 Chỉ Xóa Nét Mực Trang Này", self.canvas.clear_slide_ink)
        more_btn.setMenu(more_menu)
        b_layout.addWidget(more_btn)

        outer_b.addWidget(self.floating_dock)
        outer_b.addStretch(1)
        root.addWidget(bottom)

        # Keep Space+Drag and Canvas shortcuts active after clicking any Header/Bottom toolbar button
        for bar in (header, bottom):
            for btn_w in bar.findChildren(QPushButton):
                btn_w.setFocusPolicy(Qt.NoFocus)

    def _on_show_about(self) -> None:
        AboutBrandDialog(self).exec()

    def _on_hotkey_esc(self) -> None:
        if self.spotlight_overlay is not None and self.spotlight_overlay.isVisible():
            self.spotlight_overlay.skip_tour()
            return
        if self.stack.currentIndex() == 1:
            self.canvas.trigger_esc_cancel()

    def _on_hotkey_undo(self) -> None:
        if self.stack.currentIndex() == 1 and self.canvas.request_undo():
            self.canvas.undo()

    def _on_hotkey_redo(self) -> None:
        if self.stack.currentIndex() == 1:
            self.canvas.redo()

    def _on_hotkey_duplicate(self) -> None:
        if self.stack.currentIndex() == 1 and not isinstance(QApplication.focusWidget(), (QLineEdit, QTextBrowser)):
            self.canvas.duplicate_selected_objects()

    def _on_hotkey_delete(self) -> None:
        if self.stack.currentIndex() == 1 and not isinstance(QApplication.focusWidget(), (QLineEdit, QTextBrowser)):
            self.canvas.delete_selected_objects()

    def _on_hotkey_tool(self, tool_id: str) -> None:
        if self.stack.currentIndex() == 1 and not isinstance(QApplication.focusWidget(), (QLineEdit, QTextBrowser, QComboBox)):
            self._set_tool(tool_id)

    def _on_user_authenticated(self, user: Dict[str, Any]) -> None:
        self.logged_in_user = user
        self.teacher_name_lbl.setText(f"GV: {user.get('full_name', 'Giáo viên')}")
        self.cam_worker.resume_camera_hardware()
        self._start_audio_stream()
        self.stack.setCurrentIndex(1)

    def _on_logout(self) -> None:
        # Release webcam and microphone hardware cleanly when logging out
        self._stop_audio_stream()
        self.cam_worker.release_camera_hardware()
        self.canvas.drawing_fsm.reset()
        self.logged_in_user = {}
        self.auth_gate._refresh_tabs()
        self.stack.setCurrentIndex(0)

    def _on_toggle_studio_camera(self) -> None:
        vis = not self.left_dock.isVisible()
        self.left_dock.setVisible(vis)
        if hasattr(self, "toggle_cam_btn"):
            self.toggle_cam_btn.blockSignals(True)
            self.toggle_cam_btn.setChecked(vis)
            self.toggle_cam_btn.blockSignals(False)
        if vis:
            # Avoid duplicate camera previews covering the canvas when left_dock opens
            self._set_mini_hud_visible(False)

    def _on_toggle_camera_hardware(self) -> None:
        if self.cam_worker._camera_paused:
            self.cam_worker.resume_camera_hardware()
            if hasattr(self, "cam_hw_btn"):
                self.cam_hw_btn.setText("Tắt Cam")
        else:
            self.cam_worker.release_camera_hardware()
            if hasattr(self, "cam_hw_btn"):
                self.cam_hw_btn.setText("Bật Cam")

    def _on_toggle_hd_camera(self, checked: bool) -> None:
        self.cam_worker.set_hd_resolution(bool(checked))

    def _on_toggle_teacher_lock(self, checked: bool) -> None:
        self.cam_worker.set_teacher_lock_enabled(bool(checked))

    def _on_perf_profile_changed(self, _idx: int) -> None:
        if not hasattr(self, "perf_combo"):
            return
        mode = str(self.perf_combo.currentData() or "AUTO")
        self.cam_worker.set_perf_mode(mode)

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
        if self._shutting_down:
            return
        if self.stack.currentIndex() == 0:
            self.auth_gate.on_camera_packet(pkt)
            return

        err_msg = pkt.get("error_msg")
        if err_msg:
            self.pip_cam_lbl.setPixmap(QPixmap())
            self.pip_cam_lbl.setText(str(err_msg))
            self.hand_status_lbl.setText(f"Lỗi Camera/CV: {err_msg}")
            self.hand_status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700; font-size: 11px;")
            return

        qimg: Optional[QImage] = pkt.get("qimage")
        paused: bool = bool(pkt.get("paused", False))
        light: Dict[str, Any] = pkt.get("lighting") or {}
        fps: float = float(pkt.get("fps") or 0.0)

        if paused:
            self.pip_cam_lbl.setPixmap(QPixmap())
            self.pip_cam_lbl.setText("Camera đã giải phóng an toàn")
        elif qimg is not None:
            scaled = qimg.scaled(248, 158, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pip_cam_lbl.setPixmap(QPixmap.fromImage(scaled))
        else:
            self.pip_cam_lbl.setPixmap(QPixmap())
            self.pip_cam_lbl.setText("Chưa kết nối Webcam")

        luma = float(light.get("mean_luma") or 0.0)
        sharp = float(light.get("laplacian_var") or 0.0)
        raw_hands = pkt.get("raw_hands") or []
        accepted_hands = (1 if pkt.get("left_hand") else 0) + (1 if pkt.get("right_hand") else 0)
        vt = pkt.get("vision_telemetry") or {}
        prof_name = str(vt.get("perf_profile") or "AUTO")
        self.telemetry_lbl.setText(
            f"Sáng: {luma:.0f} | Tay: {accepted_hands}/{len(raw_hands)} | {fps:.1f} FPS ({prof_name})"
        )
        self.perf_pipeline.update_cv_fps(fps)
        self._refresh_status_pill()

        # Dual-hand cooperative gestures (Zoom, Pan, Off-Hand Quick Switch)
        dual_hand = pkt.get("dual_hand") or {}
        two_mode = str(dual_hand.get("two_hand_mode") or "SINGLE")
        if two_mode == "DUAL_PINCH_ZOOM":
            zf = float(dual_hand.get("zoom_factor") or 1.0)
            zc = dual_hand.get("zoom_center_norm")
            if abs(zf - 1.0) > 1e-3:
                self.canvas.apply_zoom_factor(zf, zc)
        elif two_mode == "DUAL_PALM_PAN":
            pdelta = dual_hand.get("pan_delta_norm") or (0.0, 0.0)
            if isinstance(pdelta, (list, tuple)) and len(pdelta) >= 2:
                dx_n, dy_n = float(pdelta[0]), float(pdelta[1])
                if self.canvas.continuous_scroll_mode and abs(dy_n) > abs(dx_n) * 1.6 and abs(dy_n) > 0.012:
                    self.canvas.scroll_document(-dy_n * max(420, self.canvas.height()) * 1.45)
                else:
                    self.canvas.apply_pan_delta_norm(dx_n, dy_n)

        off_cmd = dual_hand.get("offhand_command")
        if off_cmd == "TOGGLE_PEN_ERASER":
            next_tool = "PEN" if self.canvas.active_tool == "ERASER" else "ERASER"
            self._set_tool(next_tool)
        elif off_cmd == "CYCLE_COLOR" and self._ink_palette:
            self._ink_palette_idx = (self._ink_palette_idx + 1) % len(self._ink_palette)
            self._set_ink_color(self._ink_palette[self._ink_palette_idx])
        elif off_cmd == "UNDO":
            self._on_hotkey_undo()
        elif off_cmd == "SET_PEN":
            self._set_tool("PEN")

        if hasattr(self, "vision_debug_lbl") and self.cam_worker.debug_hud_enabled:
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
                f"Cam {cam_fps_v:.0f}fps | AI {ai_fps_v:.0f}fps ({cv_ms_v:.1f}ms) {vt.get('resolution', '')} [{prof_name}]\n"
                f"Hands: Raw={len(raw_hands)} Accepted={accepted_hands} | {two_mode} | Lock: {vt.get('teacher_mode', '-')} #{vt.get('locked_teacher_id', '-')} (Ign:{vt.get('ignored_count', 0)})\n"
                f"Light: {lmode} [{byp}] | 5-Ngón: {short_st} ({short_ang}°) | Pinch: {pr_txt}\n"
                f"Palm: {palm_ok} ({palm_sc_v:.2f}) | FSM: {vt.get('fsm_state', 'IDLE')}{roi_tag}{kal_tag}"
            )

        # Slide & Document Air-Navigation (evaluated BEFORE FSM so air-swipes NEVER erase ink!)
        nav_ev = pkt.get("nav_event")
        scroll_delta = float(pkt.get("scroll_delta_norm") or 0.0)
        is_nav_frame = bool(
            nav_ev in ("SLIDE_NEXT", "SLIDE_PREV", "SCROLL_DOWN", "SCROLL_UP", "ASK_SLIDE_AI")
            or (abs(scroll_delta) > 1e-3 and self.canvas.continuous_scroll_mode)
        )
        fsm_pkt = dict(pkt, suppress_fsm_for_nav=True) if is_nav_frame else pkt
        fsm_upd = self.canvas.apply_cv_fsm_packet(fsm_pkt)
        hud_key = fsm_upd.hud_mode.value
        self.extrapolator.ingest_vision(pkt.get("cursor_norm"), t_sec=time.perf_counter())

        now_mono = time.monotonic()
        if nav_ev == "SLIDE_NEXT":
            self._on_next_slide()
            self.canvas.show_gesture_toast("⏭ Slide Tiếp Theo")
            self._last_nav_badge = "NEXT SLIDE"
            self._last_nav_badge_until = now_mono + 1.2
        elif nav_ev == "SLIDE_PREV":
            self._on_prev_slide()
            self.canvas.show_gesture_toast("⏮ Slide Trước")
            self._last_nav_badge = "PREV SLIDE"
            self._last_nav_badge_until = now_mono + 1.2
        elif nav_ev == "SCROLL_DOWN":
            if self.canvas.continuous_scroll_mode:
                self.canvas.scroll_document(max(180.0, abs(scroll_delta) * max(400, self.canvas.height())))
            else:
                self._on_next_slide()
            self.canvas.show_gesture_toast("↕ Cuộn Tài Liệu ↓")
            self._last_nav_badge = "SCROLL DOWN"
            self._last_nav_badge_until = now_mono + 1.0
        elif nav_ev == "SCROLL_UP":
            if self.canvas.continuous_scroll_mode:
                self.canvas.scroll_document(-max(180.0, abs(scroll_delta) * max(400, self.canvas.height())))
            else:
                self._on_prev_slide()
            self.canvas.show_gesture_toast("↕ Cuộn Tài Liệu ↑")
            self._last_nav_badge = "SCROLL UP"
            self._last_nav_badge_until = now_mono + 1.0
        elif abs(scroll_delta) > 1e-3 and self.canvas.continuous_scroll_mode:
            self.canvas.scroll_document(scroll_delta * max(360, self.canvas.height()))
            self.canvas.show_gesture_toast("↕ Cuộn Tài Liệu", duration_ms=650.0)
        elif nav_ev == "ASK_SLIDE_AI":
            self.canvas.show_gesture_toast("✨ AI Đang Giải Slide...", duration_ms=1400.0)
            self._last_nav_badge = "AI SLIDE"
            self._last_nav_badge_until = now_mono + 1.6
            self._on_explain_active_slide()

        if now_mono <= self._last_nav_badge_until and self._last_nav_badge:
            mini_badge = self._last_nav_badge
        elif two_mode in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN"):
            mini_badge = "ZOOM / PAN"
        elif hud_key == "DRAWING":
            mini_badge = "HIGHLIGHTER" if self.canvas.active_tool == "HIGHLIGHTER" else "DRAWING"
        elif hud_key == "PAN_ERASE":
            mini_badge = "ERASER"
        elif hud_key == "LOCKED_COOLDOWN":
            mini_badge = "LOCKED"
        else:
            mini_badge = "POINTER"

        if hasattr(self.canvas, "mini_hud"):
            self.canvas.mini_hud.update_hud(
                qimg,
                badge=mini_badge,
                fps=fps,
                info_text=f"Sáng {luma:.0f} ({light.get('lighting_mode', 'NORMAL')}) • Tay {accepted_hands}/{len(raw_hands)}",
                paused=paused,
            )

        dual_status = str(dual_hand.get("status_label_vi") or "").strip()
        if two_mode in ("DUAL_PINCH_ZOOM", "DUAL_PALM_PAN") and dual_status:
            self.hand_status_lbl.setText(f"2 Tay: {dual_status} (Zoom {int(round(self.canvas.zoom_scale * 100))}%)")
            self.hand_status_lbl.setStyleSheet(f"color: {SECONDARY_TEAL}; font-weight: 700; font-size: 12px;")
        elif hud_key == "LOCKED_COOLDOWN":
            self.hand_status_lbl.setText(
                f"Khóa ngòi bút ({int(max(0.0, fsm_upd.cooldown_remaining_ms))}ms) • {fsm_upd.cancel_reason or 'Cooldown'}"
            )
            self.hand_status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700; font-size: 12px;")
        else:
            extra = f" • {dual_status}" if dual_status else ""
            self.hand_status_lbl.setText(f"Chế độ: {fsm_upd.hud_mode.label_vi}{extra}")
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

    def _on_auto_calibrate(self) -> None:
        self.cam_worker.request_auto_calibration()
        if hasattr(self.canvas, "mini_hud"):
            self.canvas.mini_hud.info_lbl.setText("Đang đo ánh sáng & cự ly 1.5m–3.5m...")

    def _on_calibration_done(self, info: dict) -> None:
        msg = str(info.get("message_vi") or "Đã hoàn tất cân chỉnh.")
        if hasattr(self.canvas, "mini_hud"):
            self.canvas.mini_hud.info_lbl.setText(msg)
        self.hand_status_lbl.setText(msg)

    def _set_mini_hud_visible(self, visible: bool) -> None:
        if hasattr(self.canvas, "mini_hud"):
            self.canvas.mini_hud.setVisible(bool(visible))
            self.canvas._reposition_overlays()
        if hasattr(self, "mini_hud_btn"):
            self.mini_hud_btn.blockSignals(True)
            self.mini_hud_btn.setChecked(bool(visible))
            self.mini_hud_btn.blockSignals(False)

    def _on_toggle_mini_hud(self) -> None:
        if hasattr(self.canvas, "mini_hud"):
            self._set_mini_hud_visible(not self.canvas.mini_hud.isVisible())

    def _on_toggle_thumb_strip(self) -> None:
        if hasattr(self, "thumb_strip"):
            vis = not self.thumb_strip.isVisible()
            self.thumb_strip.setVisible(vis)
            self.thumb_toggle_btn.setChecked(vis)

    def _on_toggle_ai_width(self) -> None:
        w = 460 if self.right_dock.width() < 410 else 360
        self.right_dock.setFixedWidth(w)

    def _on_60hz_cursor_tick(self) -> None:
        if self._shutting_down or self.stack.currentIndex() != 1:
            return
        pt, extrap = self.extrapolator.sample_60hz(time.perf_counter())
        if pt is not None and extrap:
            self.canvas.apply_60hz_extrapolated_cursor(pt)

    def _on_toggle_continuous_scroll(self) -> None:
        enabled = not bool(self.canvas.continuous_scroll_mode)
        self.canvas.set_continuous_scroll_mode(enabled)
        if hasattr(self, "cont_scroll_btn"):
            self.cont_scroll_btn.setChecked(enabled)
        self.canvas.show_gesture_toast("📄 Cuộn Dọc: BẬT" if enabled else "📑 Từng Trang: BẬT", duration_ms=950.0)

    def _update_slide_counter(self) -> None:
        total = max(1, len(self.canvas.slides))
        idx = self.canvas.current_slide_idx + 1 if len(self.canvas.slides) > 0 else 1
        if hasattr(self, "page_jump_edit") and self.page_jump_edit is not None:
            self.page_jump_edit.blockSignals(True)
            self.page_jump_edit.setText(str(idx))
            self.page_jump_edit.setValidator(QIntValidator(1, total, self.page_jump_edit))
            self.page_jump_edit.blockSignals(False)
        self.slide_counter_lbl.setText(f"/ {total}")
        if hasattr(self, "cont_scroll_btn"):
            self.cont_scroll_btn.setChecked(bool(self.canvas.continuous_scroll_mode))
        if hasattr(self, "thumb_strip"):
            self.thumb_strip.rebuild(
                self.canvas.slides,
                self.canvas.current_slide_idx,
                self.canvas.slide_annotations,
                self.canvas.lecture_doc,
            )

    def _on_page_jump_submitted(self) -> None:
        if not hasattr(self, "page_jump_edit") or self.page_jump_edit is None:
            return
        raw = self.page_jump_edit.text().strip()
        total = max(1, len(self.canvas.slides))
        try:
            page_num = int(raw)
        except ValueError:
            self._update_slide_counter()
            return
        clamped = max(1, min(total, page_num))
        self.canvas.set_current_slide(clamped - 1)
        self._update_slide_counter()
        self.canvas.setFocus()

    def _on_jump_to_slide(self, idx: int) -> None:
        self.canvas.set_current_slide(idx)

    def _on_prev_slide(self) -> None:
        self.canvas.prev_slide()

    def _on_next_slide(self) -> None:
        self.canvas.next_slide()

    def _on_toggle_grid(self) -> None:
        self.canvas.grid_enabled = not self.canvas.grid_enabled
        self.canvas._bump_scene()
        self.canvas.update()

    def _import_document_async(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        ext = p.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            self.canvas.load_document_file(p)
            return
        if ext not in SUPPORTED_DOC_EXTENSIONS:
            return
        if self.doc_import_worker is not None and self.doc_import_worker.isRunning():
            return
        self.slide_counter_lbl.setText("Đang nạp Slide...")
        self.doc_import_worker = DocumentImportWorker(str(p))
        self.doc_import_worker.finished_doc.connect(self._on_doc_import_finished)
        self.doc_import_worker.start()

    def _on_doc_import_finished(self, doc: Any, err_msg: str) -> None:
        if doc is None or err_msg:
            self._update_slide_counter()
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"⚠ Không thể mở tài liệu: {err_msg or 'Lỗi định dạng'}")
            return
        self.canvas.attach_lecture_document(doc)
        notice = get_office_conversion_notice(getattr(doc, "doc_type", "PDF"), getattr(doc, "backend_used", ""))
        notice_line = f"\n• **Trạng thái chuyển đổi:** {notice}" if notice else ""
        self.right_dock.setVisible(True)
        self._set_sanitized_ai_html(
            f"✓ Đã nhập bài giảng **{doc.source_path.name}** ({doc.page_count} trang · Backend `{doc.backend_used}`).{notice_line}\n\n"
            "• Mỗi trang có lớp Object chú thích độc lập (lật trang không dính nét).\n"
            "• Dùng cử chỉ **Vẫy tay trái/phải**, nút `◀`/`▶`, thanh **Slides** hoặc phím `←`/`→` để chuyển trang."
        )

    def _on_open_slide_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Mở Tài Liệu Giảng Dạy (Word / PDF / PowerPoint / Ảnh / SVG)",
            "",
            "Tất cả Tài liệu (*.docx *.doc *.pdf *.pptx *.ppt *.ppsx *.potx *.png *.jpg *.jpeg *.webp *.bmp *.svg);;"
            "Word (*.docx *.doc);;PDF (*.pdf);;PowerPoint (*.pptx *.ppt *.ppsx *.potx);;Images (*.png *.jpg *.jpeg *.webp *.bmp *.svg)",
        )
        if path:
            self._import_document_async(path)

    def _on_open_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục Ảnh Để Trình Chiếu Slide")
        if folder:
            ok = self.canvas.load_image_folder(folder)
            if not ok:
                self.right_dock.setVisible(True)
                self._set_sanitized_ai_html("⚠ Thư mục đã chọn không chứa file ảnh hợp lệ (`.png`, `.jpg`, `.webp`, `.bmp`, `.svg`).")

    def _on_insert_board_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chèn Hình Ảnh Lên Trang Bảng Hiện Tại",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.svg)",
        )
        if path:
            self.canvas.insert_image_on_board(path)

    def _on_show_gesture_cheat_sheet(self) -> None:
        dlg = GestureCheatSheetDialog(self)
        next_action = [""]

        def _req_spot() -> None:
            next_action[0] = "spotlight"

        def _req_full() -> None:
            next_action[0] = "guide"

        dlg.start_spotlight_requested.connect(_req_spot)
        dlg.open_full_guide_requested.connect(_req_full)
        connected = False
        try:
            self.cam_worker.frame_ready.connect(dlg.ingest_camera_packet)
            connected = True
        except Exception:
            connected = False
        try:
            dlg.exec()
        finally:
            if connected:
                try:
                    self.cam_worker.frame_ready.disconnect(dlg.ingest_camera_packet)
                except Exception:
                    pass
        if next_action[0] == "spotlight":
            self.start_spotlight_tour(0)
        elif next_action[0] == "guide":
            self._on_show_user_guide(0)

    def _on_show_user_guide(self, initial_tab: int = 0) -> None:
        dlg = UserGuideDialog(self, initial_tab=int(initial_tab))
        launch_spotlight = [False]

        def _req_spotlight() -> None:
            launch_spotlight[0] = True
            dlg.accept()

        dlg.start_spotlight_requested.connect(_req_spotlight)
        # Wire live webcam gesture stream directly into Tab 5 Gesture Sandbox while dialog is open
        connected = False
        try:
            self.cam_worker.frame_ready.connect(dlg.sandbox.ingest_camera_packet)
            connected = True
        except Exception:
            connected = False
        try:
            dlg.exec()
        finally:
            if connected:
                try:
                    self.cam_worker.frame_ready.disconnect(dlg.sandbox.ingest_camera_packet)
                except Exception:
                    pass
        if launch_spotlight[0]:
            self.start_spotlight_tour(0)

    def _on_spotlight_tour_finished(self, dont_show_again: bool) -> None:
        if dont_show_again:
            try:
                save_ui_api_settings(tour_completed=True)
            except Exception as exc:
                _LOGGER.debug("Cannot persist tour_completed: %s", exc)

    def start_spotlight_tour(self, step_idx: int = 0) -> SpotlightTourOverlay:
        if self.stack.currentIndex() != 1:
            self.stack.setCurrentIndex(1)
        if self.spotlight_overlay is None:
            steps = build_default_airteach_spotlight_steps(self)
            self.spotlight_overlay = SpotlightTourOverlay(self, steps)
            self.spotlight_overlay.tour_finished.connect(self._on_spotlight_tour_finished)
            self.spotlight_overlay.open_sandbox_requested.connect(
                lambda: self._on_show_user_guide(initial_tab=4)
            )
        self.spotlight_overlay.start(step_idx)
        return self.spotlight_overlay

    def _on_show_onboarding_tour(self) -> None:
        self.start_spotlight_tour(0)

    def _select_shape_kind(self, shape_kind: str) -> None:
        self.canvas.active_shape_kind = str(shape_kind or "rect").lower()
        self._set_tool("SHAPE")
        self.canvas.show_gesture_toast(
            f"◇ Hình khối: {self.canvas.active_shape_kind.upper()} — Kéo chuột trên bảng để tạo",
            duration_ms=1050.0,
        )

    def _on_toggle_ai_drawer(self) -> None:
        vis = not self.right_dock.isVisible()
        self.right_dock.setVisible(vis)
        if hasattr(self, "toggle_ai_btn"):
            self.toggle_ai_btn.setChecked(vis)

    def _on_open_local_ai_manager(self) -> None:
        mode_map = {
            AiRoutingMode.LOCAL: "local_only",
            AiRoutingMode.CLOUD: "cloud_only",
            AiRoutingMode.HYBRID: "hybrid",
        }
        dlg = LocalAiManagerDialog(
            current_routing_mode=mode_map.get(self.local_ai_manager.routing_mode, "hybrid"),
            active_gguf_path=self.active_gguf_path,
            local_provider=self.local_ai_manager.active_provider or "Ollama",
            local_model=self.local_ai_manager.active_model or "",
            parent=self,
        )
        dlg.settings_applied.connect(self._on_local_ai_settings_applied)
        dlg.exec()

    def _on_local_ai_settings_applied(self, data: Dict[str, Any]) -> None:
        rm = str(data.get("routing_mode") or "hybrid")
        if rm == "local_only":
            self.local_ai_manager.routing_mode = AiRoutingMode.LOCAL
            if hasattr(self, "routing_combo"):
                self.routing_combo.setCurrentText("Local")
        elif rm == "cloud_only":
            self.local_ai_manager.routing_mode = AiRoutingMode.CLOUD
            if hasattr(self, "routing_combo"):
                self.routing_combo.setCurrentText("Cloud")
        else:
            self.local_ai_manager.routing_mode = AiRoutingMode.HYBRID
            if hasattr(self, "routing_combo"):
                self.routing_combo.setCurrentText("Hybrid")

        gguf_path = str(data.get("active_gguf_path") or "").strip()
        self.active_gguf_path = gguf_path
        self.local_ai_manager.set_gguf_model(gguf_path)
        try:
            save_ui_api_settings(
                ai_routing_mode=self.local_ai_manager.routing_mode.value,
                active_gguf_path=gguf_path,
            )
        except Exception as exc:
            _LOGGER.warning("Cannot persist Local AI settings: %s", exc)
        self._refresh_status_pill()

    def _start_engine_warmup(self) -> None:
        if self._shutting_down:
            return
        if self.warmup_worker is not None and self.warmup_worker.isRunning():
            return
        if hasattr(self, "status_pill_lbl"):
            core_tag = "⚡ Rust Core" if HAS_RUST_CORE else "Python Core"
            self.status_pill_lbl.setText(f"{core_tag} · Đang nạp phần cứng, tài liệu & AI...")
        self.perf_pipeline.probe_hardware_async()
        self.warmup_worker = EngineWarmupWorker(self)
        self.warmup_worker.warmup_done.connect(self._on_engine_warmup_done)
        self.warmup_worker.start()

    def _on_engine_warmup_done(self, _info: Dict[str, bool]) -> None:
        self._refresh_status_pill()

    def _on_export_annotated_pdf(self) -> None:
        default_name = "AirTeach_Annotated_Lecture.pdf"
        if self.canvas.lecture_doc is not None:
            default_name = f"{self.canvas.lecture_doc.source_path.stem}_annotated.pdf"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất Bài Giảng Kèm Chú Thích Ra PDF",
            default_name,
            "PDF (*.pdf)",
        )
        if not out_path:
            return
        try:
            saved = export_annotated_lecture_pdf(
                out_path,
                self.canvas.slides,
                self.canvas.slide_annotations,
                (float(max(640, self.canvas.width())), float(max(420, self.canvas.height()))),
                lecture_doc=self.canvas.lecture_doc,
            )
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"✓ Đã xuất bài giảng PDF kèm toàn bộ nét vẽ lên `{saved}`.")
        except Exception as exc:
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"⚠ Lỗi xuất PDF chú thích: {exc}")

    def _on_export_annotated_pngs(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Chọn Thư Mục Lưu Các Trang Bài Giảng Kèm Chú Thích (.PNG)",
        )
        if not out_dir:
            return
        try:
            saved_list = export_annotated_slides_to_pngs(
                out_dir,
                self.canvas.slides,
                self.canvas.slide_annotations,
                (float(max(640, self.canvas.width())), float(max(420, self.canvas.height()))),
                lecture_doc=self.canvas.lecture_doc,
                scale=2.0,
            )
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(
                f"✓ Đã xuất **{len(saved_list)} trang ảnh PNG sắc nét** kèm toàn bộ chú thích vào `{out_dir}`."
            )
        except Exception as exc:
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"⚠ Lỗi xuất ảnh PNG chú thích: {exc}")

    def _on_explain_active_slide(self, extra_question: str = "") -> None:
        ctx = self.canvas.capture_active_slide_context()
        self.right_dock.setVisible(True)
        self._active_lasso_action = None
        pm = ctx.get("pixmap")
        if isinstance(pm, QPixmap) and not pm.isNull():
            self.crop_preview_lbl.setVisible(True)
            self.crop_preview_lbl.setPixmap(
                pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        slide_txt = str(ctx.get("slide_text") or "")[:2800]
        header_info = f"Slide {int(ctx.get('slide_index', 0)) + 1}/{int(ctx.get('slide_count', 1))}: {ctx.get('slide_title', '')}"
        q_part = f"\nCâu hỏi bổ sung của giáo viên: {extra_question}" if extra_question.strip() else ""
        prompt = (
            f"Hãy phân tích chuyên sâu và toàn diện Slide đang trình chiếu ({header_info}) kết hợp hình ảnh chụp bảng + các nét viết/khoanh chú thích tay của giáo viên."
            f"{q_part}\n"
            + (f"\nVăn bản trích xuất từ Slide:\n\"\"\"\n{slide_txt}\n\"\"\"\n" if slide_txt else "")
            + "\nHãy trình bày bài giảng chi tiết, có cấu trúc Markdown rõ ràng và công thức LaTeX chuẩn ($...$ / $$...$$) bằng tiếng Việt gồm:\n"
            "## 1. Giải thích trọng tâm & bản chất kiến thức\n"
            "Phân tích đầy đủ nội dung Slide và vùng giáo viên đánh dấu, nêu ví dụ minh họa hoặc bảng so sánh nếu phù hợp.\n"
            "## 2. Lời giải chi tiết từng bước\n"
            "Trình bày từng bước giải tường minh, kiểm tra lỗi sai thường gặp (nếu có bài tập, công thức Toán/Lý/Hóa).\n"
            "## 3. Câu hỏi trắc nghiệm kiểm tra nhanh\n"
            "Tạo 2 câu hỏi trắc nghiệm 4 lựa chọn (A/B/C/D) kèm đáp án và giải thích chi tiết."
        )
        self._set_sanitized_ai_html(f"✨ Đang phân tích **{header_info}** bằng AI...")
        self._start_hybrid_worker(prompt, png_bytes=ctx.get("png_bytes"), structured_json=False)

    def _on_voice_ask_slide(self) -> None:
        if self.voice_worker is not None and self.voice_worker.isRunning():
            return
        self.voice_ask_btn.setEnabled(False)
        self.voice_ask_btn.setText("🎙 Đang nghe...")
        self.voice_worker = VoiceQuestionWorker(duration_sec=4.0)
        self.voice_worker.finished_transcript.connect(self._on_voice_transcript_ready)
        self.voice_worker.start()

    def _on_voice_transcript_ready(self, transcript: str, err_msg: str) -> None:
        self.voice_ask_btn.setEnabled(True)
        self.voice_ask_btn.setText("🎙 Hỏi Giọng")
        if transcript:
            self._on_explain_active_slide(extra_question=transcript)
        else:
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"⚠ {err_msg or 'Không nghe rõ giọng nói.'}")

    def _on_toggle_recording(self) -> None:
        if not self.is_recording_audio:
            self.audio_frames.clear()
            self.is_recording_audio = True
            self.rec_btn.setText("⏹")
            self.rec_btn.setToolTip("Đang ghi âm... Nhấn để dừng và lưu (.WAV)")
            self.rec_btn.setStyleSheet(
                f"background: #DC2626; color: {TEXT_LIGHT}; border: 2px solid {BRIGHT_YELLOW}; border-radius: 7px; font-weight: 800; padding: 0px; min-width: 28px; max-width: 28px;"
            )
        else:
            self.is_recording_audio = False
            self.rec_btn.setText("🎙")
            self.rec_btn.setToolTip("Ghi âm bài giảng cục bộ (.WAV)")
            self.rec_btn.setStyleSheet(
                f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 7px; font-weight: 700; border: none; padding: 0px; min-width: 28px; max-width: 28px;"
            )
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

    def _sync_tool_buttons_ui(self) -> None:
        if not hasattr(self, "_tool_buttons"):
            return
        cur = self.canvas.active_tool
        surf = SURFACE_DARK if self.dark_mode else "#F1F5F9"
        fg = TEXT_LIGHT if self.dark_mode else "#0F172A"
        bd = BORDER_SLATE if self.dark_mode else "#CBD5E1"
        for tid, btn in self._tool_buttons.items():
            if tid == cur:
                btn.setStyleSheet(
                    f"background: {PRIMARY_INDIGO}; color: #FFFFFF; border: 1.5px solid {NEON_CYAN}; border-radius: 8px; padding: 3px 8px; font-weight: 800; font-size: 11px;"
                )
            else:
                btn.setStyleSheet(
                    f"background: {surf}; color: {fg}; border: 1px solid {bd}; border-radius: 8px; padding: 3px 7px; font-weight: 600; font-size: 11px;"
                )

    def _on_canvas_pie_tool_selected(self, action_id: str) -> None:
        if action_id == "OPEN_AI_CHAT":
            self.right_dock.setVisible(True)
            if hasattr(self, "toggle_ai_btn"):
                self.toggle_ai_btn.blockSignals(True)
                self.toggle_ai_btn.setChecked(True)
                self.toggle_ai_btn.blockSignals(False)
            if hasattr(self, "chat_input_edit"):
                self.chat_input_edit.setFocus()
        elif action_id == "OPEN_GESTURE_MAP":
            self._on_show_gesture_cheat_sheet()
        elif action_id == "TOGGLE_SMART_SHAPE":
            if self.canvas.smart_shape_enabled:
                self.smart_shape_btn.setText("◇ Nắn Hình: BẬT")
                self.smart_shape_btn.setStyleSheet(
                    f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 8px; font-weight: 700;"
                )
            else:
                self.smart_shape_btn.setText("◇ Nắn Hình: TẮT")
                self.smart_shape_btn.setStyleSheet("")
        self._sync_tool_buttons_ui()

    def _set_tool(self, tool_id: str) -> None:
        self.canvas.active_tool = tool_id
        self._sync_tool_buttons_ui()

    def _set_ink_color(self, hex_col: str) -> None:
        self.canvas.ink_color = QColor(hex_col)
        if self.canvas.active_tool == "SELECT" and self.canvas.scene.selected_ids:
            self.canvas.recolor_selected_objects(hex_col)
        elif self.canvas.active_tool == "HIGHLIGHTER":
            hi = QColor(hex_col)
            hi.setAlpha(96)
            self.canvas.highlighter_color = hi
        elif self.canvas.active_tool == "ERASER":
            self.canvas.active_tool = "PEN"
        self._sync_tool_buttons_ui()

    def _on_thickness_changed(self, val: int) -> None:
        self.canvas.pen_thickness = val
        self.thick_lbl.setText(f"{val}px")

    def _on_toggle_smart_shape(self) -> None:
        self.canvas.smart_shape_enabled = not self.canvas.smart_shape_enabled
        if self.canvas.smart_shape_enabled:
            self.smart_shape_btn.setText("◇ Nắn Hình: BẬT")
            if hasattr(self, "_act_smart_shape"):
                self._act_smart_shape.setText("◇ Nắn Hình Thông Minh (BẬT)")
        else:
            self.smart_shape_btn.setText("◇ Nắn Hình: TẮT")
            if hasattr(self, "_act_smart_shape"):
                self._act_smart_shape.setText("◇ Nắn Hình Thông Minh (TẮT)")

    def _on_toggle_hold_snap(self) -> None:
        self.canvas.hold_to_snap_350ms = not self.canvas.hold_to_snap_350ms
        st = "BẬT" if self.canvas.hold_to_snap_350ms else "TẮT"
        self.hold_snap_btn.setText(f"Giữ 350ms: {st}")
        if hasattr(self, "_act_hold_snap"):
            self._act_hold_snap.setText(f"⏱ Giữ 350ms mới nắn hình ({st})")

    def _on_toggle_draw_mode(self) -> None:
        cur = bool(self.canvas.drawing_fsm.allow_index_point_draw)
        nxt = not cur
        self.canvas.drawing_fsm.allow_index_point_draw = nxt
        if hasattr(self, "draw_mode_btn"):
            self.draw_mode_btn.setText("Vẽ: 1 Ngón + Chụm" if nxt else "Vẽ: Chỉ Chụm")
        if hasattr(self, "_act_draw_mode"):
            self._act_draw_mode.setText("☝️ Vẽ 1 Ngón Trỏ + Chụm (BẬT)" if nxt else "☝️ Vẽ 1 Ngón Trỏ + Chụm (TẮT)")

    def _on_reset_zoom(self) -> None:
        self.canvas.reset_viewport()

    def _refresh_api_key_ui(
        self,
        status: Optional[ApiConnectionStatus] = None,
        detail_msg: Optional[str] = None,
    ) -> None:
        self.api_key_state = resolve_api_key_layered(
            self.api_key_edit.text().strip()
            if hasattr(self, "api_key_edit") and self.api_key_state.source == "ui" and self.api_key_edit.text().strip()
            else None
        )
        self.api_key = self.api_key_state.key

        if status is not None:
            self.gemini_conn_status = status
        elif not self.api_key:
            self.gemini_conn_status = ApiConnectionStatus.UNCONFIGURED

        if not self.api_key:
            self.api_key_mask_lbl.setText("Key: Chưa cấu hình (Offline/Local AI vẫn hoạt động)")
            self.api_status_badge.setText("Ngoại tuyến")
            self.api_status_badge.setStyleSheet(
                f"background: {BORDER_SLATE}; color: {TEXT_LIGHT}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
            )
        else:
            src_label = "UI Settings" if self.api_key_state.source == "ui" else ".env / ENV"
            key_pool = split_api_keys(self.api_key)
            key_valid_syntax = any(len(k) >= 30 and k.startswith("AIza") for k in key_pool)
            self.api_key_mask_lbl.setText(
                f"Key ({len(key_pool)}): {mask_api_key(self.api_key)} • {src_label}"
            )
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
            elif key_valid_syntax:
                self.api_status_badge.setText("Sẵn sàng (Auto-Key)")
                self.api_status_badge.setStyleSheet(
                    f"background: {SECONDARY_TEAL}; color: {BG_DARK}; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 800;"
                )
            else:
                self.api_status_badge.setText("Chờ xác thực")
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
        save_ui_api_settings(ui_api_key=entered, onboarding_skipped=False)
        self.api_key = entered
        self.gemini_conn_status = ApiConnectionStatus.UNCONFIGURED
        self._refresh_api_key_ui(
            status=ApiConnectionStatus.UNCONFIGURED,
            detail_msg="Đã lưu Key vào %APPDATA%\\AirTeachAI\\config.json. Nhấn Test API để kiểm tra kết nối thật.",
        )

    def _on_delete_api_key(self) -> None:
        clear_ui_api_key()
        self.api_key_edit.clear()
        self.api_key_state = resolve_api_key_layered()
        self.api_key = self.api_key_state.key
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
        self.health_worker = GeminiHealthCheckWorker(effective, model_id=self._selected_cloud_model())
        self.health_worker.completed.connect(self._on_health_check_completed)
        self.health_worker.start()

    def _on_health_check_completed(self, res: GeminiHealthCheckResult) -> None:
        self.test_api_btn.setEnabled(True)
        self.test_api_btn.setText("Test API")
        if res.status == ApiConnectionStatus.CONNECTED:
            msg = f"✓ {res.message_vi} ({res.model_used}, {res.latency_ms:.0f}ms)"
        else:
            msg = f"⚠ {res.message_vi}"
        self._refresh_api_key_ui(status=res.status, detail_msg=msg)

    def _on_worker_status_changed(self, status_obj: object) -> None:
        if isinstance(status_obj, ApiConnectionStatus):
            self._refresh_api_key_ui(status=status_obj)

    def _set_sanitized_ai_html(self, raw_text: str, render_math_images: bool = True) -> None:
        self.last_ai_raw_text = raw_text
        self.ai_output.setHtml(
            OutputSanitizer.to_rich_html(
                raw_text,
                dark_mode=self.dark_mode,
                render_math_images=render_math_images,
            )
        )

    def _selected_cloud_model(self) -> Optional[str]:
        if not hasattr(self, "cloud_model_combo") or self.cloud_model_combo is None:
            return None
        mid = str(self.cloud_model_combo.currentData() or "").strip()
        return mid if mid else None

    def _on_cloud_model_changed(self, _idx: int) -> None:
        selected = self._selected_cloud_model() or ""
        self.preferred_cloud_model = selected
        try:
            save_ui_api_settings(preferred_model=selected)
        except Exception as exc:
            _LOGGER.debug("Cannot persist preferred Gemini model: %s", exc)
        self._refresh_status_pill()

    def _refresh_status_pill(self) -> None:
        mode = self.local_ai_manager.routing_mode.label_vi
        if self.local_ai_manager.active_model and self.local_ai_manager.routing_mode != AiRoutingMode.CLOUD:
            mdl = self.local_ai_manager.active_model
        else:
            mdl = self._selected_cloud_model() or DEFAULT_GEMINI_MODEL
        self.perf_pipeline.update_ai_metrics(
            ai_mode=mode,
            active_model=mdl,
            tokens_per_sec=self.local_ai_manager.last_tokens_per_sec,
        )
        if hasattr(self, "ai_mode_quick_btn"):
            rm = self.local_ai_manager.routing_mode
            if rm == AiRoutingMode.LOCAL:
                self.ai_mode_quick_btn.setText("💻 Local Offline")
            elif rm == AiRoutingMode.CLOUD:
                self.ai_mode_quick_btn.setText("☁ Cloud Gemini")
            else:
                self.ai_mode_quick_btn.setText("⚡ Hybrid AI")
        if hasattr(self, "status_pill_lbl"):
            base_txt = self.perf_pipeline.telemetry.status_pill_text()
            prefix = "⚡ Rust · " if HAS_RUST_CORE else ""
            self.status_pill_lbl.setText(f"{prefix}{base_txt}")

    def _on_quick_toggle_ai_mode(self) -> None:
        """1-click cycle: Cloud Gemini <-> Local AI Offline <-> Hybrid."""
        order = (AiRoutingMode.CLOUD, AiRoutingMode.LOCAL, AiRoutingMode.HYBRID)
        cur = self.local_ai_manager.routing_mode
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else AiRoutingMode.LOCAL
        self.local_ai_manager.routing_mode = nxt
        if hasattr(self, "routing_combo"):
            self.routing_combo.blockSignals(True)
            self.routing_combo.setCurrentText(nxt.label_vi)
            self.routing_combo.blockSignals(False)
        try:
            save_ui_api_settings(ai_routing_mode=nxt.value)
        except Exception as exc:
            _LOGGER.debug("Cannot persist quick AI mode: %s", exc)
        self._refresh_status_pill()
        self.canvas.show_gesture_toast(f"🤖 Chế độ AI: {nxt.label_vi}", duration_ms=1050.0)

    def _apply_theme_styles(self) -> None:
        dark = bool(self.dark_mode)
        bg = BG_DARK if dark else "#F8FAFC"
        surf = SURFACE_DARK if dark else "#FFFFFF"
        surf_alt = SURFACE_DARK if dark else "#F1F5F9"
        fg = TEXT_LIGHT if dark else "#0F172A"
        bd = BORDER_SLATE if dark else "#CBD5E1"
        accent_txt = NEON_CYAN if dark else PRIMARY_INDIGO

        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.setStyleSheet(
                f"QWidget {{ background-color: {bg}; color: {fg}; }}"
                f"QPushButton {{ background-color: {surf_alt}; color: {fg}; border: 1px solid {bd}; border-radius: 7px; padding: 3px 7px; font-weight: 600; }}"
                f"QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; }}"
                f"QLineEdit, QComboBox {{ background-color: {bg}; color: {fg}; border: 1px solid {bd}; border-radius: 6px; padding: 5px; }}"
            )
        if hasattr(self, "header_bar"):
            self.header_bar.setStyleSheet(
                f"QFrame {{ background: {bg}; border-bottom: 1px solid {bd}; }}"
                f"QPushButton {{ background: {surf_alt}; color: {fg}; border: 1px solid {bd}; "
                f"border-radius: 6px; padding: 2px 5px; min-width: 0px; font-size: 11px; font-weight: 700; }}"
                f"QPushButton:hover {{ border-color: {PRIMARY_INDIGO}; }}"
            )
        if hasattr(self, "brand_lbl"):
            self.brand_lbl.setStyleSheet(f"color: {fg}; font-weight: 800; font-size: 12.5px; border: none; background: transparent;")
        if hasattr(self, "page_jump_edit"):
            self.page_jump_edit.setStyleSheet(
                f"background: {surf_alt}; color: {accent_txt}; border: 1px solid {bd}; border-radius: 5px; padding: 1px 2px; font-size: 11px; font-weight: 800;"
            )
        if hasattr(self, "slide_counter_lbl"):
            self.slide_counter_lbl.setStyleSheet(f"color: {fg}; font-weight: 700; font-size: 11px; border: none; background: transparent; padding: 0 2px;")
        if hasattr(self, "status_pill_lbl"):
            self.status_pill_lbl.setStyleSheet(
                f"background: {surf_alt}; color: {accent_txt}; border: 1px solid {SECONDARY_TEAL if dark else PRIMARY_INDIGO}; border-radius: 10px; padding: 2px 7px; font-size: 10px; font-weight: 700;"
            )
        if hasattr(self, "bottom_bar"):
            self.bottom_bar.setStyleSheet(f"background: {bg}; border-top: 1px solid {bd};")
        if hasattr(self, "floating_dock"):
            self.floating_dock.setStyleSheet(
                f"QFrame {{ background: {surf}; border: 1px solid {bd}; border-radius: 14px; }}"
            )
        if hasattr(self, "left_dock"):
            self.left_dock.setStyleSheet(
                f"QFrame#LeftDock {{ background: {surf}; border-right: 1px solid {bd}; }}"
                f"QFrame#LeftDock QLabel {{ color: {fg}; background: transparent; border: none; }}"
                f"QFrame#LeftDock QCheckBox {{ color: {fg}; background: transparent; font-size: 11px; spacing: 5px; }}"
                f"QFrame#LeftDock QPushButton {{ background: {surf_alt}; color: {fg}; border: 1px solid {bd}; border-radius: 6px; padding: 3px 6px; font-size: 11px; font-weight: 700; }}"
                f"QFrame#LeftDock QComboBox {{ background: {bg}; color: {fg}; border: 1px solid {bd}; border-radius: 6px; padding: 3px 6px; font-size: 11px; }}"
            )
        if hasattr(self, "right_dock"):
            rd_bg = "rgba(15, 23, 42, 244)" if dark else "rgba(255, 255, 255, 248)"
            self.right_dock.setStyleSheet(
                f"QFrame#RightDock {{ background: {rd_bg}; border-left: 1.5px solid {SECONDARY_TEAL if dark else PRIMARY_INDIGO}; }}"
                f"QFrame#RightDock QLabel {{ color: {fg}; background: transparent; border: none; }}"
                f"QFrame#RightDock QCheckBox {{ color: {fg}; background: transparent; font-size: 11px; spacing: 5px; }}"
                f"QFrame#RightDock QPushButton {{ background: {surf_alt}; color: {fg}; border: 1px solid {bd}; border-radius: 6px; padding: 3px 7px; font-size: 11px; font-weight: 700; }}"
                f"QFrame#RightDock QComboBox, QFrame#RightDock QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {bd}; border-radius: 6px; padding: 4px 7px; font-size: 11.5px; }}"
            )
        if hasattr(self, "attached_file_frame"):
            self.attached_file_frame.setStyleSheet(
                f"QFrame {{ background: {surf_alt}; border: 1px solid {SECONDARY_TEAL if dark else PRIMARY_INDIGO}; border-radius: 7px; }}"
            )
        if hasattr(self, "attached_file_lbl"):
            self.attached_file_lbl.setStyleSheet(
                f"color: {accent_txt}; font-size: 11px; font-weight: 700; border: none; background: transparent;"
            )
        if hasattr(self, "thumb_strip"):
            self.thumb_strip.setStyleSheet(f"background: {surf}; border-top: 1px solid {bd};")
        if hasattr(self, "theme_toggle_btn"):
            self.theme_toggle_btn.setText("☀" if dark else "🌙")
        self.ai_output.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {bd}; border-radius: 8px; padding: 8px; font-size: 13px;"
        )
        self._sync_tool_buttons_ui()
        if self.last_ai_raw_text:
            self._set_sanitized_ai_html(self.last_ai_raw_text)

    def _on_toggle_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self.canvas.set_dark_mode(self.dark_mode)
        self._apply_theme_styles()

    def _on_routing_mode_changed(self, text: str) -> None:
        mapping = {"Hybrid": AiRoutingMode.HYBRID, "Local": AiRoutingMode.LOCAL, "Cloud": AiRoutingMode.CLOUD}
        self.local_ai_manager.routing_mode = mapping.get(text, AiRoutingMode.HYBRID)
        try:
            save_ui_api_settings(ai_routing_mode=self.local_ai_manager.routing_mode.value)
        except Exception as exc:
            _LOGGER.debug("Cannot persist routing mode: %s", exc)
        self._refresh_status_pill()

    def _on_discover_local_ai(self) -> None:
        if self._shutting_down:
            return
        if self.discovery_worker is not None and self.discovery_worker.isRunning():
            return
        self.discovery_worker = LocalDiscoveryWorker(self.local_ai_manager)
        self.discovery_worker.discovered.connect(self._on_local_discovered)
        self.discovery_worker.start()

    def _on_local_discovered(self, servers: List[DiscoveredLocalServer]) -> None:
        if self._shutting_down or not hasattr(self, "local_model_combo"):
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

    @staticmethod
    def _prompt_needs_board_tools(user_msg: str) -> bool:
        low = (user_msg or "").strip().lower()
        board_keywords = (
            "lên bảng",
            "trên bảng",
            "vẽ hình",
            "vẽ đồ thị",
            "vẽ trục",
            "vẽ sơ đồ",
            "vẽ tam giác",
            "vẽ đường tròn",
            "vẽ mũi tên",
            "khoanh vùng",
            "xóa bảng",
            "ghi hộ",
            "viết hộ",
            "draw_",
            "json hành động",
        )
        return any(kw in low for kw in board_keywords)

    def clear_chat_attachment(self) -> None:
        self._pending_attachment = None
        if hasattr(self, "attached_file_lbl"):
            self.attached_file_lbl.setText("")
        if hasattr(self, "attached_file_frame"):
            self.attached_file_frame.setVisible(False)

    def attach_file_to_chat(self, path: Path | str) -> bool:
        try:
            att = parse_attachment_file(path)
        except Exception as exc:
            _LOGGER.warning("Cannot parse chat attachment %s: %s", path, exc)
            self.right_dock.setVisible(True)
            self._set_sanitized_ai_html(f"⚠ Không thể đọc tệp đính kèm: {exc}")
            return False

        self._pending_attachment = att
        meta_bits: List[str] = []
        if att.kind == "image":
            meta_bits.append("Ảnh PNG")
        elif att.kind == "document":
            meta_bits.append(f"{max(1, int(att.page_count))} trang")
            if att.extracted_text:
                meta_bits.append(f"{len(att.extracted_text):,} ký tự")
        else:
            meta_bits.append(f"{len(att.extracted_text):,} ký tự")
        if att.warning_vi:
            meta_bits.append("⚠")
        summary_chip = f"📎 {att.filename} ({' · '.join(meta_bits)})"
        if hasattr(self, "attached_file_lbl"):
            self.attached_file_lbl.setText(summary_chip)
            self.attached_file_lbl.setToolTip(att.warning_vi or f"{att.filename} ({att.mime_type})")
        if hasattr(self, "attached_file_frame"):
            self.attached_file_frame.setVisible(True)
        self.right_dock.setVisible(True)
        if hasattr(self, "toggle_ai_btn"):
            self.toggle_ai_btn.blockSignals(True)
            self.toggle_ai_btn.setChecked(True)
            self.toggle_ai_btn.blockSignals(False)
        if att.image_bytes and hasattr(self, "crop_preview_lbl"):
            pm = QPixmap()
            if pm.loadFromData(att.image_bytes, "PNG") and not pm.isNull():
                self.crop_preview_lbl.setVisible(True)
                self.crop_preview_lbl.setPixmap(
                    pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        return True

    def _on_pick_chat_attachment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Đính Kèm Tệp Gửi Trợ Lý AI (PDF / DOCX / PPTX / Ảnh / TXT)",
            "",
            "Tất cả tệp hỗ trợ (*.pdf *.docx *.doc *.pptx *.ppt *.png *.jpg *.jpeg *.webp *.bmp *.svg *.txt *.md *.csv *.json);;"
            "Tài liệu (*.pdf *.docx *.doc *.pptx *.ppt);;"
            "Hình ảnh (*.png *.jpg *.jpeg *.webp *.bmp *.svg);;"
            "Văn bản UTF-8 (*.txt *.md *.csv *.json)",
        )
        if path:
            if self.attach_file_to_chat(path):
                self.right_dock.setVisible(True)
                if hasattr(self, "chat_input_edit"):
                    self.chat_input_edit.setFocus()

    def _on_chat_submit(self) -> None:
        user_msg = self.chat_input_edit.text().strip()
        att = self._pending_attachment
        if not user_msg and att is None:
            return
        if not user_msg and att is not None:
            user_msg = "Hãy phân tích và tóm tắt chi tiết tệp đính kèm này."
        self.chat_input_edit.clear()
        self.right_dock.setVisible(True)

        # Tier 1: Fast Local Intent Parser (<5ms, instant board write/draw/clear) - only when no file attached
        if att is None:
            fast_act = WhiteboardAgent.parse_fast_intent(user_msg)
            if fast_act is not None:
                summary = self.canvas.execute_whiteboard_actions([fast_act])
                self.perf_pipeline.update_ai_metrics(ai_mode="Instant", active_model="Local-Intent", tokens_per_sec=0.0)
                self._refresh_status_pill()
                self._set_sanitized_ai_html(f"**Lệnh tức thì (<5ms):** {user_msg}\n\n✓ {summary}")
                return

        display_user_msg = f"{user_msg} [📎 {att.filename}]" if att is not None else user_msg
        self.chat_history.append({"role": "user", "content": display_user_msg})
        self._set_sanitized_ai_html(f"Đang xử lý yêu cầu: **{display_user_msg}**...", render_math_images=False)

        ctx_png: Optional[bytes] = None
        worker_prompt = user_msg

        if att is not None:
            att_header = (
                f"Tệp đính kèm: {att.filename} "
                f"(MIME: {att.mime_type}, số trang: {max(1, int(att.page_count))})"
            )
            if att.warning_vi:
                att_header += f"\nLưu ý trích xuất: {att.warning_vi}"
            if att.extracted_text:
                worker_prompt = (
                    f"{worker_prompt}\n\n{att_header}\n"
                    f"Nội dung tệp đính kèm (UTF-8 NFC):\n\"\"\"\n{att.extracted_text}\n\"\"\""
                )
            else:
                worker_prompt = f"{worker_prompt}\n\n{att_header}"
            if att.image_bytes:
                ctx_png = att.image_bytes
            self.clear_chat_attachment()

        if hasattr(self, "chat_attach_board_chk") and self.chat_attach_board_chk.isChecked():
            has_doc = self.canvas.lecture_doc is not None
            has_ink = bool(self.canvas.strokes or self.canvas.annotations or self.canvas.board_images)
            cur_sl = self.canvas.slides[self.canvas.current_slide_idx] if self.canvas.slides else {}
            if has_doc or has_ink or not bool(cur_sl.get("is_tutorial", False)):
                ctx = self.canvas.capture_active_slide_context()
                if ctx_png is None:
                    ctx_png = ctx.get("png_bytes")
                sl_txt = str(ctx.get("slide_text") or "").strip()[:3600]
                if sl_txt:
                    worker_prompt = (
                        f"{worker_prompt}\n\nNgữ cảnh Slide/Bảng hiện tại "
                        f"(Trang {int(ctx.get('slide_index', 0)) + 1}/{int(ctx.get('slide_count', 1))}):\n"
                        f"\"\"\"\n{sl_txt}\n\"\"\""
                    )
        wants_tools = self._prompt_needs_board_tools(user_msg)
        self._start_hybrid_worker(
            worker_prompt,
            png_bytes=ctx_png,
            structured_json=False,
            include_board_tools=wants_tools,
            cloud_model=self._selected_cloud_model(),
        )

    def _on_copy_last_ai_reply(self) -> None:
        txt = OutputSanitizer.sanitize_plain_text(self.last_ai_raw_text or "").strip()
        if not txt:
            return
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(txt)
        self.canvas.show_gesture_toast("📋 Đã sao chép nội dung AI vào Clipboard", duration_ms=1000.0)

    def _on_push_last_ai_to_board(self) -> None:
        raw = (self.last_ai_raw_text or "").strip()
        if not raw:
            return
        cleaned = OutputSanitizer.sanitize_plain_text(raw).strip()
        # Strip leading engine metadata line like *[Cloud · gemini-2.5-flash · 42 tok/s]*
        lines = [ln for ln in cleaned.splitlines() if not (ln.strip().startswith("*[") and ln.strip().endswith("]*"))]
        full_body = "\n".join(lines).strip()
        if not full_body:
            return
        cw = float(max(640, self.canvas.width()))
        ch = float(max(420, self.canvas.height()))
        ax, ay = find_free_board_region(
            self.canvas._occupied_rects(),
            (cw, ch),
            needed_size=(min(560.0, cw * 0.48), min(380.0, ch * 0.55)),
        )
        obj = self.canvas.add_text_object(
            full_body,
            x=ax,
            y=ay,
            font_size=14,
            is_ai=True,
            select=True,
        )
        self.canvas.active_tool = "SELECT"
        self._sync_tool_buttons_ui()
        self.canvas.show_gesture_toast(
            f"📌 Đã tạo Text Object AI lên bảng ({int(obj.w)}×{int(obj.h)}px — kéo/resize tự do)",
            duration_ms=1350.0,
        )

    def _on_lasso_quick_action(self, action_id: str, region: LassoSelectionRegion) -> None:
        try:
            qact = LassoQuickAction(action_id)
        except Exception as exc:
            _LOGGER.warning("Unknown lasso quick action '%s', fallback to SOLVE_AI: %s", action_id, exc)
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
        canvas_sz = (float(max(640, self.canvas.width())), float(max(420, self.canvas.height())))
        free_regs = find_free_regions_normalized(self.canvas._occupied_rects(), canvas_sz, max_regions=3)
        prompt = build_lasso_prompt(qact, free_regions_norm=free_regs)
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
        ctx = self.canvas.capture_active_slide_context()
        png_bytes = ctx.get("png_bytes")
        cropped_pm = ctx.get("pixmap")
        if isinstance(cropped_pm, QPixmap) and not cropped_pm.isNull():
            self.crop_preview_lbl.setVisible(True)
            self.crop_preview_lbl.setPixmap(
                cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        slide_txt = str(ctx.get("slide_text") or "")[:2800]
        header_info = (
            f"Trang {int(ctx.get('slide_index', 0)) + 1}/{int(ctx.get('slide_count', 1))}: "
            f"{ctx.get('slide_title', '')}"
        )
        full_prompt = prompt
        if slide_txt:
            full_prompt = (
                f"{prompt}\n\nNgữ cảnh tài liệu đang mở ({header_info}):\n\"\"\"\n{slide_txt}\n\"\"\""
            )
        self._set_sanitized_ai_html(f"✨ Đang phân tích **{header_info}** bằng AI...")
        self._start_hybrid_worker(full_prompt, png_bytes=png_bytes, structured_json=structured_json)

    def _start_hybrid_worker(
        self,
        prompt: str,
        png_bytes: Optional[bytes],
        structured_json: bool = False,
        include_board_tools: bool = True,
        cloud_model: Optional[str] = None,
    ) -> None:
        pool = resolve_api_key_pool(
            self.api_key_edit.text().strip() if self.api_key_edit.text().strip() else None
        )
        self.api_key_state = resolve_api_key_layered(
            self.api_key_edit.text().strip() if self.api_key_edit.text().strip() else None
        )
        self.api_key = self.api_key_state.key
        if self.hybrid_worker is not None and self.hybrid_worker.isRunning():
            try:
                self.hybrid_worker.token_delta.disconnect(self._on_hybrid_token)
                self.hybrid_worker.finished_result.disconnect(self._on_hybrid_finished)
            except Exception as exc:
                _LOGGER.debug("Prior hybrid_worker disconnect note: %s", exc)
            self.hybrid_worker.requestInterruption()
        self._stream_raw_accum = ""
        self._stream_last_render_t = 0.0
        eff_prompt = prompt
        if include_board_tools and not structured_json:
            canvas_sz = (float(max(640, self.canvas.width())), float(max(420, self.canvas.height())))
            free_regs = find_free_regions_normalized(self.canvas._occupied_rects(), canvas_sz, max_regions=3)
            eff_prompt = f"{prompt}\n\n{WhiteboardAgent.build_tool_prompt_instruction(canvas_size=canvas_sz, free_regions_norm=free_regs)}"
        eff_cloud_model = cloud_model if cloud_model is not None else self._selected_cloud_model()
        self.hybrid_worker = HybridAiWorker(
            self.local_ai_manager,
            eff_prompt,
            history=self.chat_history[-10:],
            png_bytes=png_bytes,
            cloud_api_key=",".join(pool) if pool else self.api_key,
            cloud_model=eff_cloud_model,
            structured_json=structured_json,
            include_board_tools=False,
        )
        self.hybrid_worker.token_delta.connect(self._on_hybrid_token)
        self.hybrid_worker.finished_result.connect(self._on_hybrid_finished)
        self.hybrid_worker.start()

    def _on_hybrid_token(self, chunk: str) -> None:
        if not chunk:
            return
        self._stream_raw_accum += chunk
        now_t = time.perf_counter()
        if (now_t - self._stream_last_render_t) >= 0.06 or "\n" in chunk:
            self._stream_last_render_t = now_t
            self._set_sanitized_ai_html(self._stream_raw_accum, render_math_images=False)
            sb = self.ai_output.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

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
        badge_hdr = f"*{res.engine_used} · {res.model_used} · {res.tokens_per_sec:.1f} tok/s*"
        self._set_sanitized_ai_html(f"{badge_hdr}\n\n{clean_reply}")

    def _start_gemini_worker(
        self, prompt: str, png_bytes: Optional[bytes], structured_json: bool = False
    ) -> None:
        self.api_key_state = resolve_api_key_layered(self.api_key_edit.text().strip() if self.api_key_edit.text().strip() else None)
        self.api_key = self.api_key_state.key
        self.gemini_worker = GeminiVisionWorker(
            self.api_key,
            prompt,
            png_bytes,
            structured_json=structured_json,
            primary_model=self._selected_cloud_model(),
        )
        self.gemini_worker.finished_text.connect(self._set_sanitized_ai_html)
        self.gemini_worker.status_changed.connect(self._on_worker_status_changed)
        self.gemini_worker.start()

    def _maybe_show_first_run_onboarding(self) -> None:
        if (
            self._shutting_down
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or os.environ.get("AIRTEACH_NO_MODAL")
        ):
            return
        cfg = load_ui_api_settings()
        needs_tour = str(cfg.get("tour_completed", "")).lower() not in ("true", "1")
        if not self.api_key and str(cfg.get("onboarding_skipped", "")).lower() not in ("true", "1"):
            dlg = ApiKeyOnboardingDialog(self)
            if dlg.exec() == QDialog.Accepted and dlg.saved_key:
                self.api_key_state = resolve_api_key_layered(dlg.saved_key)
                self.api_key = self.api_key_state.key
                self.api_key_edit.setText(self.api_key)
                self._refresh_api_key_ui(status=ApiConnectionStatus.CONNECTED)
        if needs_tour:
            self.start_spotlight_tour(0)

    def shutdown_resources(self) -> None:
        """Idempotent clean shutdown of Webcam, Audio, Timers, and Worker QThreads."""
        if self._shutting_down:
            return
        self._shutting_down = True
        if self.spotlight_overlay is not None:
            try:
                self.spotlight_overlay.hide()
            except Exception:
                pass
        for timer_attr in ("ui_timer", "cursor_60hz_timer", "_cam_start_timer", "_warmup_timer", "_discovery_timer", "_onboarding_timer"):
            tmr = getattr(self, timer_attr, None)
            if tmr is not None:
                try:
                    tmr.stop()
                except Exception as exc:
                    _LOGGER.warning("%s stop error: %s", timer_attr, exc)
        self._stop_audio_stream()
        try:
            self.cam_worker.release_camera_hardware()
            self.cam_worker.stop()
        except Exception as exc:
            _LOGGER.warning("cam_worker stop error: %s", exc)
        for attr in ("warmup_worker", "gemini_worker", "hybrid_worker", "discovery_worker", "health_worker", "doc_import_worker", "voice_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                try:
                    if hasattr(worker, "isRunning") and worker.isRunning():
                        worker.requestInterruption()
                        if attr == "warmup_worker":
                            # Never TerminateThread a worker performing Python imports (would leak _imp lock)
                            worker.wait(5000)
                        elif not worker.wait(1500):
                            worker.terminate()
                            worker.wait(200)
                except Exception as exc:
                    _LOGGER.warning("worker %s stop error: %s", attr, exc)
        if getattr(self, "_app_event_filter_installed", False):
            qapp_inst = QApplication.instance()
            if qapp_inst is not None:
                qapp_inst.removeEventFilter(self)
            self._app_event_filter_installed = False
        if hasattr(self, "canvas") and getattr(self.canvas, "lecture_doc", None) is not None:
            try:
                self.canvas.lecture_doc.close()
            except Exception as exc:
                _LOGGER.debug("lecture_doc close note during shutdown: %s", exc)

    def closeEvent(self, event) -> None:
        self.shutdown_resources()
        super().closeEvent(event)


def main() -> None:
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    set_windows_app_user_model_id()
    setup_app_logging()
    install_global_excepthook()
    app = QApplication(sys.argv)
    if APP_LOGO_ICO.exists():
        app.setWindowIcon(QIcon(str(APP_LOGO_ICO)))
    fonts_dir = resource_path("assets/fonts")
    for font_file in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"):
        fp = fonts_dir / font_file
        if fp.exists():
            QFontDatabase.addApplicationFont(str(fp))
    QFont.insertSubstitutions("Segoe UI", ["Noto Sans", "Arial", "Tahoma"])
    available_families = set(QFontDatabase.families())
    for fam in ("Segoe UI", "Noto Sans", "Arial", "Tahoma"):
        if fam in available_families:
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
    app.aboutToQuit.connect(win.shutdown_resources)
    if smoke_test:
        win.stack.setCurrentIndex(1)
        win.show()

        def _finish_smoke() -> None:
            assert win.canvas is not None
            assert win.canvas.mini_hud is not None
            assert win.thumb_strip is not None
            win.close()
            print("SMOKE_TEST_OK", flush=True)
            app.quit()

        QTimer.singleShot(250, _finish_smoke)
        sys.exit(app.exec())
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
