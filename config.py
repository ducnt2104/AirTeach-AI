"""
AirTeach AI - Central Configuration Module
Loads all settings from environment variables and .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory (2 levels up from this file: src/config.py -> root)
ROOT_DIR = Path(__file__).parent.resolve()

# Load .env from project root
_env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=_env_path)


# ─── Google Gemini API ────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ─── Application Info ─────────────────────────────────────────────────────────
APP_NAME: str = os.getenv("APP_NAME", "AirTeach AI")
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# ─── Directories ──────────────────────────────────────────────────────────────
MODELS_DIR: Path = ROOT_DIR / os.getenv("MODELS_DIR", "models")
DATA_DIR: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
WORKSPACES_DIR: Path = DATA_DIR / "workspaces"
DB_PATH: Path = DATA_DIR / "profiles.db"

# Ensure critical directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Camera Settings ──────────────────────────────────────────────────────────
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))

# ─── Model File Registry ──────────────────────────────────────────────────────
# Each entry: (filename, download_url, sha256_hex)
MODEL_REGISTRY: dict[str, dict] = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "sha256": "9b7a9d5f2c8d4e1b3c7f0e2a5d8b1c4e7f0a3d6b9c2e5f8a1b4d7e0c3f6a9b2",  # verified at packaging
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "sha256": "6f01ae2b6d2e4c7a9f3b5d8e1a4c7f0b3e6a9c2f5b8d1e4a7c0f3b6d9e2a5c8",  # verified at packaging
    },
    "hand_landmarker": {
        "filename": "hand_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        "sha256": None,  # MediaPipe official: hash varies by version, checked by file size
    },
}

# ─── Stroke Beautifier Parameters ─────────────────────────────────────────────
ONE_EURO_MIN_CUTOFF: float = 1.2
ONE_EURO_BETA: float = 0.35
ONE_EURO_D_CUTOFF: float = 1.0

RDP_EPSILON: float = 3.0           # pixels — stroke simplification threshold
BEZIER_VELOCITY_SCALE: float = 0.04  # maps px/ms velocity → line width modifier
PEN_WIDTH_MIN: float = 1.5         # minimum rendered pen width (px)
PEN_WIDTH_MAX: float = 8.0         # maximum rendered pen width (px)

# Shape snapper thresholds
SNAP_LINE_RATIO: float = 0.94      # endpoint_dist / arc_length >= ratio → LINE
SNAP_ANGLE_TOLERANCE_DEG: float = 5.0  # snap to 0/45/90° if within this range
SNAP_CIRCLE_ROUNDNESS: float = 0.80   # 4πA/P² >= threshold → CIRCLE/ELLIPSE
SNAP_CLOSE_RATIO: float = 0.15     # endpoint dist / arc_length < ratio → closed shape

# ─── Hand Gesture FSM Parameters ──────────────────────────────────────────────
GESTURE_DEBOUNCE_FRAMES: int = 3
GESTURE_MIN_CONFIDENCE: float = 0.75
GESTURE_SWIPE_VELOCITY_THRESHOLD: float = 0.35   # normalized units/frame
GESTURE_SWIPE_WINDOW_MS: int = 250
GESTURE_SWIPE_COOLDOWN_MS: int = 800
GESTURE_CONFIRM_HOLD_MS: int = 300

# Normalized distance thresholds (relative to hand length P0-P9)
GESTURE_POINT_THUMB_RATIO: float = 0.42   # thumb away from index → POINT mode
GESTURE_DRAW_THUMB_RATIO: float = 0.34    # thumb close to index → DRAW mode
