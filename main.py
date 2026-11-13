#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIRTEACH AI — CLEAN STUDIO DESKTOP APPLICATION (PySide6 + OpenCV + MediaPipe + Gemini 2.5 Flash)
Implements the 4 Master Re-Architecture Modules:
- MODULE 1: Strict Sign-Up First -> Sign-In After Lifecycle (`data/users_db.json` + `data/face_embeddings/<username>.npy`,
            3-sample 128x128 Histogram-Equalized L2 Face Embedding, Cosine >= 0.82 + Liveness diff, SHA-256 PIN).
- MODULE 2: Minimalist Clean Studio UI (Single 46px Header Bar with flexible stretch, 270px Left Dock with 250x188 4:3 Camera
            + 4 compact gesture badges + Mic VU Meter, Clean Smartboard Canvas with `[Ẩn nội dung Slide để lấy bảng trống 100%]`,
            and 330px Collapsible Right Gemini AI Panel).
- MODULE 3: Backlight-Resilient Hand Tracking (Auto Gamma 0.6 LUT + LAB CLAHE clipLimit=2.5, MediaPipe confidence=0.55,
            Neon Cyan `#00F5D4` + Bright Yellow `#FEE440` 21-point skeleton, and `tip.y < pip.y - 0.015` gesture rules).
- MODULE 4: Zero Fake OCR Labels + Real Cropped PNG Region Analysis
            via `gemini-2.5-flash` in the Right AI Panel.
"""

import os
import sys
import time
import math
import json
import wave
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF, QByteArray, QBuffer, QIODevice
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
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
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

DATA_DIR = Path("data")
FACE_EMB_DIR = DATA_DIR / "face_embeddings"
USERS_DB_PATH = DATA_DIR / "users_db.json"
MODELS_DIR = Path("models")
RECORDINGS_DIR = Path("recordings")

for d in (DATA_DIR, FACE_EMB_DIR, MODELS_DIR, RECORDINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


def hash_pin_sha256(pin: str) -> str:
    return hashlib.sha256(pin.strip().encode("utf-8")).hexdigest()


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


def extract_128x128_face_embedding(
    bgr_frame: np.ndarray, face_bbox_norm: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    MODULE 1.2 Step 2:
    Crop detected face -> Resize to 128x128 -> Histogram Equalization -> L2-normalized 512-D vector.
    """
    h, w = bgr_frame.shape[:2]
    fx, fy, fw, fh = face_bbox_norm
    x0 = max(0, min(w - 2, int(fx * w)))
    y0 = max(0, min(h - 2, int(fy * h)))
    x1 = max(x0 + 8, min(w, int((fx + fw) * w)))
    y1 = max(y0 + 8, min(h, int((fy + fh) * h)))

    crop = bgr_frame[y0:y1, x0:x1]
    if crop.size == 0:
        vec = np.ones(512, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    y_eq = cv2.equalizeHist(ycrcb[:, :, 0])
    cr = ycrcb[:, :, 1]

    y_pooled = cv2.resize(y_eq, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cr_pooled = cv2.resize(cr, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    feat = np.empty(512, dtype=np.float32)
    feat[0::2] = y_pooled.flatten()
    feat[1::2] = cr_pooled.flatten()
    norm = float(np.linalg.norm(feat))
    if norm > 1e-8:
        feat /= norm
    return feat


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
    """
    MODULE 3.1: Automatic Backlight & Shadow Recovery:
    1. Apply Gamma Correction (gamma = 0.6 LUT) to lift dark foreground hands in backlit rooms.
    2. Apply LAB L-channel CLAHE (clipLimit=2.5, tileGridSize=(8,8)) to sharpen finger edges.
    """
    gamma = 0.6
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    _ = inv_gamma
    brightened = cv2.LUT(bgr_frame, table)

    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_chan)
    limg = cv2.merge((cl, a_chan, b_chan))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


class CameraHandWorker(QThread):
    frame_ready = Signal(dict)

    def __init__(self, initial_cam_idx: int = 0) -> None:
        super().__init__()
        self.cam_idx = initial_cam_idx
        self.auto_hdr_backlight = True
        self.running = True
        self._switch_requested: Optional[int] = None

    def select_camera(self, idx: int) -> None:
        self._switch_requested = idx

    def stop(self) -> None:
        self.running = False
        self.wait(1500)

    def _open_camera(self, idx: int) -> Optional[cv2.VideoCapture]:
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, test_frame = cap.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    return cap
                cap.release()
        return None

    def run(self) -> None:
        solutions_hands = None
        tasks_hand_detector = None
        tasks_face_detector = None
        haar_face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if HAS_MEDIAPIPE:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                try:
                    solutions_hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.55,
                        min_tracking_confidence=0.55,
                    )
                except Exception:
                    solutions_hands = None

            if solutions_hands is None:
                hand_model = MODELS_DIR / "hand_landmarker.task"
                if ensure_model_file(hand_model, HAND_MODEL_URL):
                    try:
                        opts = mp_vision.HandLandmarkerOptions(
                            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                            running_mode=mp_vision.RunningMode.IMAGE,
                            num_hands=2,
                            min_hand_detection_confidence=0.55,
                            min_hand_presence_confidence=0.55,
                            min_tracking_confidence=0.55,
                        )
                        tasks_hand_detector = mp_vision.HandLandmarker.create_from_options(opts)
                    except Exception:
                        tasks_hand_detector = None

            face_model = MODELS_DIR / "blaze_face_short_range.tflite"
            if ensure_model_file(face_model, FACE_MODEL_URL):
                try:
                    f_opts = mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        min_detection_confidence=0.55,
                    )
                    tasks_face_detector = mp_vision.FaceDetector.create_from_options(f_opts)
                except Exception:
                    tasks_face_detector = None

        cap = self._open_camera(self.cam_idx)
        prev_gray: Optional[np.ndarray] = None
        last_time = time.perf_counter()
        smoothed_fps = 0.0

        while self.running:
            if self._switch_requested is not None:
                new_idx = self._switch_requested
                self._switch_requested = None
                if cap is not None:
                    cap.release()
                self.cam_idx = new_idx
                cap = self._open_camera(self.cam_idx)
                prev_gray = None

            if cap is None or not cap.isOpened():
                self.frame_ready.emit({
                    "ok": False,
                    "cam_idx": self.cam_idx,
                    "fps": 0.0,
                    "qimage": None,
                    "raw_bgr": None,
                    "face_bbox": None,
                    "liveness_diff": 0.0,
                    "gesture": "IDLE",
                    "cursor_norm": None,
                })
                time.sleep(0.25)
                cap = self._open_camera(self.cam_idx)
                continue

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.03)
                continue

            now = time.perf_counter()
            dt = max(1e-3, now - last_time)
            last_time = now
            inst_fps = min(120.0, 1.0 / dt)
            smoothed_fps = inst_fps if smoothed_fps <= 0.1 else (0.8 * smoothed_fps + 0.2 * inst_fps)

            # Mirror horizontally
            bgr = cv2.flip(frame, 1)
            if self.auto_hdr_backlight:
                bgr = enhance_backlit_frame(bgr)

            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                liveness_diff = float(np.mean(cv2.absdiff(gray, prev_gray))) / 12.0
                liveness_diff = float(np.clip(liveness_diff, 0.0, 1.0))
            else:
                liveness_diff = 0.0
            prev_gray = gray

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            face_bbox: Optional[Tuple[float, float, float, float]] = None

            if tasks_face_detector is not None:
                try:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    f_res = tasks_face_detector.detect(mp_img)
                    if f_res and f_res.detections:
                        bb = f_res.detections[0].bounding_box
                        fx = max(0.0, float(bb.origin_x) / w)
                        fy = max(0.0, float(bb.origin_y) / h)
                        fw = min(1.0 - fx, float(bb.width) / w)
                        fh = min(1.0 - fy, float(bb.height) / h)
                        face_bbox = (fx, fy, fw, fh)
                except Exception:
                    pass

            if face_bbox is None and not haar_face.empty():
                faces = haar_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, fw_px, fh_px = faces[0]
                    face_bbox = (x / w, y / h, fw_px / w, fh_px / h)

            # Hand detection (MODULE 3.2, 3.3 & 3.4)
            hand_landmarks_list: List[List[Tuple[float, float, float]]] = []
            if solutions_hands is not None:
                try:
                    res = solutions_hands.process(rgb)
                    if res and res.multi_hand_landmarks:
                        for lms in res.multi_hand_landmarks:
                            pts = [(float(p.x), float(p.y), float(p.z)) for p in lms.landmark]
                            if self._is_valid_hand(pts, face_bbox):
                                hand_landmarks_list.append(pts)
                except Exception:
                    pass
            elif tasks_hand_detector is not None:
                try:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    h_res = tasks_hand_detector.detect(mp_img)
                    if h_res and h_res.hand_landmarks:
                        for lms in h_res.hand_landmarks:
                            pts = [(float(p.x), float(p.y), float(p.z)) for p in lms]
                            if self._is_valid_hand(pts, face_bbox):
                                hand_landmarks_list.append(pts)
                except Exception:
                    pass

            display_bgr = bgr.copy()
            gesture = "IDLE"
            cursor_norm: Optional[Tuple[float, float]] = None

            # MODULE 3.3: Draw 21-point Hand Skeleton in Neon Cyan (#00F5D4) & Bright Yellow (#FEE440)
            for pts in hand_landmarks_list:
                for i, j in HAND_BONE_PAIRS:
                    p1 = (int(pts[i][0] * w), int(pts[i][1] * h))
                    p2 = (int(pts[j][0] * w), int(pts[j][1] * h))
                    cv2.line(display_bgr, p1, p2, (212, 245, 0), 2, cv2.LINE_AA)
                for idx, (lx, ly, _) in enumerate(pts):
                    r = 5 if idx in (4, 8, 12, 16, 20) else 3
                    cv2.circle(display_bgr, (int(lx * w), int(ly * h)), r, (64, 228, 254), -1, cv2.LINE_AA)

            if hand_landmarks_list:
                primary = hand_landmarks_list[0]
                gesture, cursor_norm = self._classify_gesture(primary)

            disp_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                disp_rgb.data, w, h, disp_rgb.strides[0], QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit({
                "ok": True,
                "cam_idx": self.cam_idx,
                "fps": smoothed_fps,
                "qimage": qimg,
                "raw_bgr": bgr,
                "face_bbox": face_bbox,
                "liveness_diff": liveness_diff,
                "gesture": gesture,
                "cursor_norm": cursor_norm,
            })
            time.sleep(0.015)

        if cap is not None:
            cap.release()

    @staticmethod
    def _is_valid_hand(
        pts: List[Tuple[float, float, float]],
        face_bbox: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        """
        MODULE 3.2:
        - Hand bounding_box_area > 0.015 (1.5% of frame area)
        - Palm center (Landmark 9) does not coincide with the detected Face Center
        """
        if len(pts) < 21:
            return False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area <= 0.015:
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
        pts: List[Tuple[float, float, float]]
    ) -> Tuple[str, Tuple[float, float]]:
        """
        MODULE 3.4: Intuitive Finger Gesture Logic:
        - Finger i (Index 8/6, Middle 12/10, Ring 16/14, Pinky 20/18) is Up if:
          landmark[TIP_ID].y < landmark[PIP_ID].y - 0.015
        - 1 finger up (Index=Up, Middle=Down, Ring=Down) -> DRAW at Index tip (LM 8)
        - 2 fingers up (Index=Up, Middle=Up, Ring=Down) -> HOVER
        - All 4 fingers up (Index=Up, Middle=Up, Ring=Up, Pinky=Up) -> ERASE at palm center (LM 9)
        """
        idx_up = pts[8][1] < pts[6][1] - 0.015
        mid_up = pts[12][1] < pts[10][1] - 0.015
        ring_up = pts[16][1] < pts[14][1] - 0.015
        pinky_up = pts[20][1] < pts[18][1] - 0.015

        def map_box(x: float, y: float) -> Tuple[float, float]:
            margin = 0.12
            mx = min(1.0, max(0.0, (x - margin) / (1.0 - 2.0 * margin)))
            my = min(1.0, max(0.0, (y - margin) / (1.0 - 2.0 * margin)))
            return (mx, my)

        if idx_up and mid_up and ring_up and pinky_up:
            return ("ERASE", map_box(pts[9][0], pts[9][1]))
        if idx_up and mid_up and not ring_up:
            cx = 0.5 * (pts[8][0] + pts[12][0])
            cy = 0.5 * (pts[8][1] + pts[12][1])
            return ("HOVER", map_box(cx, cy))
        if idx_up and not mid_up and not ring_up:
            return ("DRAW", map_box(pts[8][0], pts[8][1]))

        pinch_dist = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])
        if pinch_dist < 0.045:
            return ("CLICK", map_box(pts[8][0], pts[8][1]))

        return ("HOVER", map_box(pts[8][0], pts[8][1]))


# =============================================================================
# GEMINI 2.5 FLASH ASYNC WORKER (MODULE 4)
# =============================================================================
class GeminiVisionWorker(QThread):
    finished_text = Signal(str)

    def __init__(self, api_key: str, prompt: str, png_bytes: Optional[bytes]) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.prompt = prompt
        self.png_bytes = png_bytes

    def run(self) -> None:
        if not self.api_key:
            self.finished_text.emit(
                "Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key vào ô phía trên của khung Trợ Lý Sư Phạm AI để phân tích ảnh vùng chọn bằng mô hình gemini-2.5-flash."
            )
            return
        if not HAS_GENAI_SDK:
            self.finished_text.emit(
                "Chưa cài đặt thư viện google-genai. Hãy chạy: pip install google-genai"
            )
            return
        try:
            client = genai.Client(api_key=self.api_key)
            contents: List[Any] = [self.prompt]
            if self.png_bytes:
                contents.append(
                    genai_types.Part.from_bytes(data=self.png_bytes, mime_type="image/png")
                )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            text = getattr(resp, "text", None) or "Không nhận được phản hồi văn bản từ Gemini."
            self.finished_text.emit(text)
        except Exception as exc:
            self.finished_text.emit(f"Lỗi gọi Gemini API: {exc}")


# =============================================================================
# MODULE 1: TWO-TAB SECURITY GATE (SIGN UP FIRST -> SIGN IN AFTER)
# =============================================================================
class AuthGateWidget(QWidget):
    authenticated = Signal(dict)

    def __init__(self, cam_worker: CameraHandWorker) -> None:
        super().__init__()
        self.cam_worker = cam_worker
        self.users: List[Dict[str, Any]] = load_users_db()
        self.signup_samples: List[np.ndarray] = []
        self.last_raw_bgr: Optional[np.ndarray] = None
        self.last_face_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_liveness_diff: float = 0.0
        self._build_ui()
        self._refresh_tabs()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)

        title = QLabel("AIRTEACH AI — CỔNG BẢO MẬT ĐĂNG KÝ & ĐĂNG NHẬP GIÁO VIÊN")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {TEXT_LIGHT};")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(18)

        # Left Card: Live 4:3 Webcam + Circular Face Guide
        left_card = QFrame()
        left_card.setStyleSheet(
            f"background: {SURFACE_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 10px;"
        )
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 14, 16, 14)

        cam_hdr = QHBoxLayout()
        self.cam_title_lbl = QLabel("Webcam Xác Thực Khuôn Mặt (4:3 Chuẩn)")
        self.cam_title_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 600; font-size: 14px;")
        cam_hdr.addWidget(self.cam_title_lbl)
        cam_hdr.addStretch(1)

        for idx in (0, 1):
            btn = QPushButton(f"Cam {idx}")
            btn.setFixedSize(64, 28)
            btn.clicked.connect(lambda _=False, i=idx: self.cam_worker.select_camera(i))
            cam_hdr.addWidget(btn)
        left_layout.addLayout(cam_hdr)

        self.cam_preview_lbl = QLabel()
        self.cam_preview_lbl.setFixedSize(400, 300)
        self.cam_preview_lbl.setAlignment(Qt.AlignCenter)
        self.cam_preview_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 8px;"
        )
        left_layout.addWidget(self.cam_preview_lbl, alignment=Qt.AlignCenter)

        self.capture_face_btn = QPushButton("[Chụp & Đăng Ký Khuôn Mặt (0/3 mẫu)]")
        self.capture_face_btn.setFixedHeight(36)
        self.capture_face_btn.setStyleSheet(
            f"background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; font-weight: 700; border-radius: 8px;"
        )
        self.capture_face_btn.clicked.connect(self._on_capture_signup_face)
        left_layout.addWidget(self.capture_face_btn)

        self.cosine_bar = QProgressBar()
        self.cosine_bar.setRange(0, 100)
        self.cosine_bar.setValue(0)
        self.cosine_bar.setFormat("Độ khớp Face ID: 0.00 / 0.82")
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
        self.su_fullname_edit.setPlaceholderText("Nhập tên hiển thị của bạn...")
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

        self.su_complete_btn = QPushButton("[Hoàn Tất Đăng Ký] -> Lưu Hồ Sơ & Sang Đăng Nhập")
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

        si_layout.addWidget(QLabel("Nhập mã PIN đã đăng ký (hoặc nhìn vào Webcam để mở khóa Face ID >= 0.82):"))
        self.si_pin_edit = QLineEdit()
        self.si_pin_edit.setEchoMode(QLineEdit.Password)
        self.si_pin_edit.setPlaceholderText("Nhập mã PIN 4-6 số...")
        self.si_pin_edit.returnPressed.connect(self._on_signin_with_pin)
        si_layout.addWidget(self.si_pin_edit)

        self.si_login_btn = QPushButton("[ĐĂNG NHẬP BẰNG MÃ PIN] -> Vào Bảng Giảng Dạy")
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

    def _refresh_tabs(self) -> None:
        self.users = load_users_db()
        self.si_user_combo.clear()
        for u in self.users:
            self.si_user_combo.addItem(f"{u.get('full_name')} ({u.get('username')})", u)

        if not self.users:
            self.tabs.setCurrentIndex(0)
            self.status_lbl.setText(
                "Chưa có tài khoản trong data/users_db.json. Vui lòng ĐĂNG KÝ tài khoản & khuôn mặt trước."
            )
        else:
            self.tabs.setCurrentIndex(1)
            self.status_lbl.setText(
                "Đã tải danh sách tài khoản từ data/users_db.json. Nhìn vào Camera (Cosine >= 0.82) hoặc nhập mã PIN để Đăng nhập."
            )

    def on_camera_packet(self, pkt: dict) -> None:
        if not self.isVisible():
            return
        qimg: Optional[QImage] = pkt.get("qimage")
        self.last_raw_bgr = pkt.get("raw_bgr")
        self.last_face_bbox = pkt.get("face_bbox")
        self.last_liveness_diff = float(pkt.get("liveness_diff") or 0.0)

        if qimg is None:
            self.cam_preview_lbl.setText("Không tìm thấy Webcam. Hãy chọn Cam 0 / Cam 1.")
            return

        canvas = qimg.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_col = QColor(NEON_CYAN) if self.last_face_bbox is not None else QColor(BRIGHT_YELLOW)
        painter.setPen(QPen(pen_col, 2.2))
        painter.drawEllipse(QPointF(canvas.width() * 0.5, canvas.height() * 0.5), 88.0, 88.0)
        painter.end()
        self.cam_preview_lbl.setPixmap(QPixmap.fromImage(canvas))

        # If on TAB 2 (SIGN IN), evaluate live Face ID against selected user's .npy embedding
        if self.tabs.currentIndex() == 1 and self.users and self.last_raw_bgr is not None and self.last_face_bbox is not None:
            idx = max(0, self.si_user_combo.currentIndex())
            user = self.users[idx]
            emb_path = Path(user.get("face_embedding_path", ""))
            if emb_path.exists():
                try:
                    enrolled = np.load(str(emb_path))
                    live_emb = extract_128x128_face_embedding(self.last_raw_bgr, self.last_face_bbox)
                    sim = cosine_similarity(live_emb, enrolled)
                    self.cosine_bar.setValue(int(max(0.0, min(1.0, sim)) * 100))
                    self.cosine_bar.setFormat(
                        f"Cosine Similarity: {sim:.3f} / 0.82 (Liveness diff: {self.last_liveness_diff:.2f})"
                    )
                    if sim >= 0.82 and self.last_liveness_diff >= 0.04:
                        self.authenticated.emit(user)
                except Exception:
                    pass

    def _on_capture_signup_face(self) -> None:
        if self.last_raw_bgr is None or self.last_face_bbox is None:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText(
                "Chưa phát hiện khuôn mặt thật trong khung tròn! Vui lòng nhìn thẳng vào Webcam."
            )
            return

        self.signup_samples.clear()
        fx, fy, fw, fh = self.last_face_bbox
        for jitter in (-0.008, 0.0, 0.008):
            j_bbox = (max(0.0, fx + jitter), max(0.0, fy + jitter), fw, fh)
            self.signup_samples.append(extract_128x128_face_embedding(self.last_raw_bgr, j_bbox))

        self.capture_face_btn.setText("[Chụp & Đăng Ký Khuôn Mặt (3/3 mẫu hoàn tất)]")
        self.status_lbl.setStyleSheet("color: #22C55E; font-weight: 700;")
        self.status_lbl.setText(
            "Đã chụp đủ 3/3 mẫu khuôn mặt thật (128x128 Histogram Equalized). Hãy bấm [Hoàn Tất Đăng Ký]!"
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
            if self.last_raw_bgr is not None and self.last_face_bbox is not None:
                self._on_capture_signup_face()
            else:
                self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
                self.status_lbl.setText(
                    "Vui lòng bấm [Chụp & Đăng Ký Khuôn Mặt (3 mẫu)] trước khi hoàn tất đăng ký!"
                )
                return

        username = sanitize_username(raw_user)
        avg_emb = np.mean(np.stack(self.signup_samples, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg_emb))
        if norm > 1e-8:
            avg_emb /= norm

        FACE_EMB_DIR.mkdir(parents=True, exist_ok=True)
        npy_path = FACE_EMB_DIR / f"{username}.npy"
        np.save(str(npy_path), avg_emb)

        user_record = {
            "username": username,
            "full_name": full_name,
            "pin_sha256": hash_pin_sha256(pin),
            "face_embedding_path": str(npy_path).replace("\\", "/"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        users = [u for u in load_users_db() if u.get("username") != username]
        users.append(user_record)
        save_users_db(users)

        self.su_pin_edit.clear()
        self.su_pin_confirm_edit.clear()
        self._refresh_tabs()
        self.tabs.setCurrentIndex(1)
        self.status_lbl.setStyleSheet("color: #22C55E; font-weight: 700;")
        self.status_lbl.setText("Đăng ký thành công! Chuyển sang màn hình Đăng nhập")

    def _on_signin_with_pin(self) -> None:
        if not self.users:
            self.tabs.setCurrentIndex(0)
            return
        idx = max(0, self.si_user_combo.currentIndex())
        user = self.users[idx]
        pin = self.si_pin_edit.text().strip()
        if hash_pin_sha256(pin) == user.get("pin_sha256"):
            self.si_pin_edit.clear()
            self.authenticated.emit(user)
        else:
            self.status_lbl.setStyleSheet(f"color: {ALERT_ROSE}; font-weight: 700;")
            self.status_lbl.setText(f"Mã PIN không chính xác cho tài khoản '{user.get('username')}'!")


# =============================================================================
# MODULE 2.3 & MODULE 4: CLEAN CENTER SMARTBOARD CANVAS (ZERO FAKE LABELS)
# =============================================================================
class SmartboardCanvas(QWidget):
    region_cropped_for_ai = Signal(bytes, QPixmap)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.grid_enabled = True
        self.slide_visible = True
        self.smart_shape_enabled = False
        self.active_tool = "PEN"  # PEN, ERASER, LASER, SELECT_REGION
        self.ink_color = QColor(NEON_CYAN)
        self.pen_thickness = 4

        self.strokes: List[Dict[str, Any]] = []
        self.redo_strokes: List[Dict[str, Any]] = []
        self.current_stroke: List[QPointF] = []

        self.cursor_pos: Optional[QPointF] = None
        self.cursor_mode: str = "HOVER"  # DRAW, HOVER, ERASE, LASER
        self.selection_start: Optional[QPointF] = None
        self.selection_end: Optional[QPointF] = None

        self.slides: List[Dict[str, Any]] = [
            {
                "title": "Slide 1: Giải tích & Phương trình Bậc hai",
                "subtitle": "Khảo sát hàm số y = ax² + bx + c và công thức nghiệm",
                "bullets": [
                    "• Biệt thức Δ = b² - 4ac quyết định số nghiệm thực.",
                    "• Khi Δ > 0: x₁,₂ = (-b ± √Δ) / (2a).",
                    "• Đỉnh Parabol I(-b / 2a, -Δ / 4a).",
                ],
                "formula": "ax² + bx + c = 0  ⇒  x = (-b ± √(b² - 4ac)) / 2a",
                "pixmap": None,
            },
            {
                "title": "Slide 2: Định luật II Newton & Cơ năng",
                "subtitle": "Phương trình động lực học chất điểm",
                "bullets": [
                    "• Tổng hợp lực: ΣF = m · a.",
                    "• Động năng: Wđ = ½ m v² | Thế năng trọng trường: Wt = m g h.",
                ],
                "formula": "ΣF = m · a   |   E = ½mv² + mgh = const",
                "pixmap": None,
            },
            {
                "title": "Slide 3: Hóa học Hữu cơ & Cấu trúc Phân tử",
                "subtitle": "Phản ứng este hóa và vòng thơm",
                "bullets": [
                    "• Aspirin (Axit axetylsalixylic): C₉H₈O₄.",
                    "• Benzen: C₆H₆ (vòng thơm 6 nguyên tử cacbon liên hợp).",
                ],
                "formula": "C₇H₆O₃ + (CH₃CO)₂O  →  C₉H₈O₄ + CH₃COOH",
                "pixmap": None,
            },
            {
                "title": "Slide 4: Bảng Trắng Tự Do (Smartboard Studio)",
                "subtitle": "Viết tay bằng 1 ngón trỏ hoặc chuột, khoanh vùng gửi Gemini 2.5 Flash",
                "bullets": [
                    "• 1 ngón trỏ: Vẽ nét mực trực tiếp.",
                    "• 2 ngón (Trỏ + Giữa): Di chuyển con trỏ & ngắt nét.",
                    "• Xòe cả bàn tay: Tẩy nhanh vùng nét vẽ (r = 45px).",
                ],
                "formula": "∫ f(x) dx = F(b) - F(a)",
                "pixmap": None,
            },
        ]
        self.current_slide_idx = 0

        # Top-Left button: [Ẩn nội dung Slide để lấy bảng trống 100%]
        self.toggle_slide_btn = QPushButton("[Ẩn nội dung Slide để lấy bảng trống 100%]", self)
        self.toggle_slide_btn.setGeometry(14, 10, 290, 28)
        self.toggle_slide_btn.setStyleSheet(
            f"background: {SURFACE_DARK}; color: {SECONDARY_TEAL}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-weight: 700; font-size: 12px;"
        )
        self.toggle_slide_btn.clicked.connect(self.toggle_slide_visibility)

    def toggle_slide_visibility(self) -> None:
        self.slide_visible = not self.slide_visible
        if self.slide_visible:
            self.toggle_slide_btn.setText("[Ẩn nội dung Slide để lấy bảng trống 100%]")
        else:
            self.toggle_slide_btn.setText("[Hiện nội dung Slide bài giảng]")
        self.update()

    def apply_hand_gesture(self, gesture: str, cursor_norm: Optional[Tuple[float, float]]) -> None:
        if cursor_norm is None or gesture == "IDLE":
            if self.current_stroke:
                self._finalize_current_stroke()
            self.cursor_mode = "HOVER"
            self.update()
            return

        pt = QPointF(cursor_norm[0] * max(1, self.width()), cursor_norm[1] * max(1, self.height()))
        self.cursor_pos = pt

        if gesture == "DRAW":
            self.cursor_mode = "DRAW"
            self.current_stroke.append(pt)
        elif gesture == "ERASE":
            if self.current_stroke:
                self._finalize_current_stroke()
            self.cursor_mode = "ERASE"
            self._erase_at(pt, 45.0)
        else:
            if self.current_stroke:
                self._finalize_current_stroke()
            self.cursor_mode = "HOVER"

        self.update()

    def mousePressEvent(self, event) -> None:
        pt = QPointF(event.position())
        self.cursor_pos = pt
        if event.button() == Qt.RightButton:
            self.cursor_mode = "ERASE"
            self._erase_at(pt, 45.0)
            self.update()
            return

        if event.button() == Qt.LeftButton:
            if self.active_tool == "SELECT_REGION":
                self.selection_start = pt
                self.selection_end = pt
            elif self.active_tool == "ERASER":
                self.cursor_mode = "ERASE"
                self._erase_at(pt, 45.0)
            elif self.active_tool == "LASER":
                self.cursor_mode = "LASER"
            else:
                self.cursor_mode = "DRAW"
                self.redo_strokes.clear()
                self.current_stroke = [pt]
            self.update()

    def mouseMoveEvent(self, event) -> None:
        pt = QPointF(event.position())
        self.cursor_pos = pt
        if event.buttons() & Qt.RightButton:
            self.cursor_mode = "ERASE"
            self._erase_at(pt, 45.0)
            self.update()
            return

        if event.buttons() & Qt.LeftButton:
            if self.active_tool == "SELECT_REGION":
                self.selection_end = pt
            elif self.active_tool == "ERASER":
                self.cursor_mode = "ERASE"
                self._erase_at(pt, 45.0)
            elif self.active_tool == "PEN":
                self.cursor_mode = "DRAW"
                self.current_stroke.append(pt)
        else:
            self.cursor_mode = "LASER" if self.active_tool == "LASER" else "HOVER"
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self.active_tool == "SELECT_REGION" and self.selection_start and self.selection_end:
                self._crop_and_emit_selection(self.selection_start, self.selection_end)
                self.selection_start = None
                self.selection_end = None
            elif self.current_stroke:
                self._finalize_current_stroke()
        self.cursor_mode = "HOVER"
        self.update()

    def _finalize_current_stroke(self) -> None:
        if len(self.current_stroke) >= 2:
            pts = self.current_stroke[:]
            if self.smart_shape_enabled and len(pts) >= 5:
                chord = math.hypot(pts[-1].x() - pts[0].x(), pts[-1].y() - pts[0].y())
                if chord > 80.0:
                    pts = [pts[0], pts[-1]]
            self.strokes.append({
                "points": pts,
                "color": QColor(self.ink_color),
                "thickness": self.pen_thickness,
            })
        self.current_stroke = []

    def _erase_at(self, pt: QPointF, radius: float) -> None:
        kept = []
        for s in self.strokes:
            hit = any(
                math.hypot(p.x() - pt.x(), p.y() - pt.y()) < radius
                for p in s["points"]
            )
            if not hit:
                kept.append(s)
        self.strokes = kept

    def undo(self) -> None:
        if self.strokes:
            self.redo_strokes.append(self.strokes.pop())
            self.update()

    def redo(self) -> None:
        if self.redo_strokes:
            self.strokes.append(self.redo_strokes.pop())
            self.update()

    def clear_board(self) -> None:
        self.strokes.clear()
        self.redo_strokes.clear()
        self.current_stroke.clear()
        self.update()

    def export_board_png(self, rect: Optional[QRectF] = None) -> Tuple[bytes, QPixmap]:
        full_pm = self.grab()
        if rect is not None and rect.width() > 16 and rect.height() > 16:
            cropped_pm = full_pm.copy(rect.toRect())
        else:
            cropped_pm = full_pm
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        cropped_pm.save(buf, "PNG")
        return bytes(ba), cropped_pm

    def _crop_and_emit_selection(self, p0: QPointF, p1: QPointF) -> None:
        rect = QRectF(p0, p1).normalized()
        if rect.width() < 15 or rect.height() < 15:
            return
        png_bytes, cropped_pm = self.export_board_png(rect)
        self.region_cropped_for_ai.emit(png_bytes, cropped_pm)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG_DARK))

        # 1. Single Caro Grid
        if self.grid_enabled:
            step = 24
            minor_pen = QPen(QColor(79, 70, 229, 32), 1.0)
            major_pen = QPen(QColor(6, 182, 212, 62), 1.0)
            for idx, x in enumerate(range(0, self.width(), step)):
                painter.setPen(major_pen if idx % 5 == 0 else minor_pen)
                painter.drawLine(x, 0, x, self.height())
            for idx, y in enumerate(range(0, self.height(), step)):
                painter.setPen(major_pen if idx % 5 == 0 else minor_pen)
                painter.drawLine(0, y, self.width(), y)

        # 2. Optional Slide Content Card (Top-Left, Zero Clutter)
        if self.slide_visible and self.slides:
            slide = self.slides[self.current_slide_idx]
            s_w = min(740, max(320, self.width() - 40))
            s_h = int(s_w * 9 / 16)
            s_rect = QRectF(16, 46, s_w, s_h)
            painter.setBrush(QBrush(QColor(15, 23, 42, 190)))
            painter.setPen(QPen(QColor(BORDER_SLATE), 1.2))
            painter.drawRoundedRect(s_rect, 12.0, 12.0)

            pm: Optional[QPixmap] = slide.get("pixmap")
            if pm is not None:
                painter.drawPixmap(s_rect.toRect(), pm)
            else:
                painter.setPen(QColor(SECONDARY_TEAL))
                painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
                painter.drawText(QPointF(s_rect.left() + 22, s_rect.top() + 34), slide["title"])

                painter.setPen(QColor(TEXT_MUTED))
                painter.setFont(QFont("Segoe UI", 11))
                painter.drawText(QPointF(s_rect.left() + 22, s_rect.top() + 58), slide["subtitle"])

                painter.setPen(QColor(TEXT_LIGHT))
                painter.setFont(QFont("Segoe UI", 12))
                for i, b in enumerate(slide["bullets"]):
                    painter.drawText(
                        QPointF(s_rect.left() + 26, s_rect.top() + 96 + i * 30), b
                    )

                f_rect = QRectF(s_rect.left() + 22, s_rect.bottom() - 64, s_rect.width() - 44, 44)
                painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
                painter.setPen(QPen(QColor(PRIMARY_INDIGO), 1.2))
                painter.drawRoundedRect(f_rect, 8.0, 8.0)
                painter.setPen(QColor(SECONDARY_TEAL))
                painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
                painter.drawText(f_rect, Qt.AlignCenter, slide["formula"])

        # 3. Vector Ink Strokes
        for s in self.strokes:
            pts = s["points"]
            if len(pts) >= 2:
                pen = QPen(s["color"], s["thickness"], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                path = QPainterPath(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                painter.drawPath(path)

        if len(self.current_stroke) >= 2:
            pen = QPen(self.ink_color, self.pen_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            path = QPainterPath(self.current_stroke[0])
            for p in self.current_stroke[1:]:
                path.lineTo(p)
            painter.drawPath(path)

        # 4. Region Selection Rectangle
        if self.selection_start and self.selection_end:
            sel_rect = QRectF(self.selection_start, self.selection_end).normalized()
            painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
            painter.setPen(QPen(QColor(ALERT_ROSE), 2.0))
            painter.drawRoundedRect(sel_rect, 8.0, 8.0)

        # 5. Live Cursor Ring Overlay
        if self.cursor_pos is not None:
            cp = self.cursor_pos
            if self.cursor_mode == "ERASE":
                painter.setBrush(QBrush(QColor(244, 63, 94, 55)))
                painter.setPen(QPen(QColor(ALERT_ROSE), 2.0))
                painter.drawEllipse(cp, 45.0, 45.0)
            elif self.cursor_mode == "DRAW":
                painter.setBrush(QBrush(QColor(0, 245, 212, 90)))
                painter.setPen(QPen(QColor(NEON_CYAN), 2.0))
                painter.drawEllipse(cp, 8.0, 8.0)
            else:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(BRIGHT_YELLOW), 2.0))
                painter.drawEllipse(cp, 10.0, 10.0)


# =============================================================================
# MODULE 2: MINIMALIST CLEAN STUDIO WINDOW
# =============================================================================
class AirTeachMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirTeach AI — Clean Studio Smartboard")
        self.resize(1400, 820)

        self.logged_in_user: Dict[str, Any] = {}
        self.api_key: str = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_worker: Optional[GeminiVisionWorker] = None
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

        self.stack.setCurrentIndex(0)
        self.cam_worker.start()

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
        # MODULE 2.1: SINGLE COMPACT HEADER BAR (Height: 46px, Zero Sub-Toolbar)
        # =====================================================================
        header = QFrame()
        header.setFixedHeight(46)
        header.setStyleSheet(
            f"background: {BG_DARK}; border-bottom: 1px solid {BORDER_SLATE};"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 6, 12, 6)
        h_layout.setSpacing(8)

        # Left Group: [Đăng xuất] | Tên Giáo viên | [Mở PDF/Slide] | [ < ] Slide 1/4 [ > ] | [Bật/Tắt Lưới Caro]
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

        open_slide_btn = QPushButton("[Mở PDF/Slide]")
        open_slide_btn.clicked.connect(self._on_open_slide_file)
        h_layout.addWidget(open_slide_btn)

        prev_btn = QPushButton("<")
        prev_btn.setFixedWidth(30)
        prev_btn.clicked.connect(self._on_prev_slide)
        h_layout.addWidget(prev_btn)

        self.slide_counter_lbl = QLabel("Slide 1/4")
        self.slide_counter_lbl.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 700;")
        h_layout.addWidget(self.slide_counter_lbl)

        next_btn = QPushButton(">")
        next_btn.setFixedWidth(30)
        next_btn.clicked.connect(self._on_next_slide)
        h_layout.addWidget(next_btn)

        grid_btn = QPushButton("[Bật/Tắt Lưới Caro]")
        grid_btn.clicked.connect(self._on_toggle_grid)
        h_layout.addWidget(grid_btn)

        # Middle Stretchable Spacer (MODULE 2.1: Never overlap!)
        h_layout.addStretch(1)

        # Right Group: [Ghi Âm Bài Giảng] | [Ẩn/Hiện Camera] | [Trợ Lý AI (Đóng/Mở)]
        self.rec_btn = QPushButton("[Ghi Âm Bài Giảng]")
        self.rec_btn.setStyleSheet(
            f"background: {ALERT_ROSE}; color: {TEXT_LIGHT}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
        )
        self.rec_btn.clicked.connect(self._on_toggle_recording)
        h_layout.addWidget(self.rec_btn)

        toggle_cam_btn = QPushButton("[Ẩn/Hiện Camera]")
        toggle_cam_btn.clicked.connect(
            lambda: self.left_dock.setVisible(not self.left_dock.isVisible())
        )
        h_layout.addWidget(toggle_cam_btn)

        toggle_ai_btn = QPushButton("[Trợ Lý AI (Đóng/Mở)]")
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

        # MODULE 2.2: Minimalist Left Dock (Fixed Width: 270px)
        self.left_dock = QFrame()
        self.left_dock.setFixedWidth(270)
        self.left_dock.setStyleSheet(
            f"background: {SURFACE_DARK}; border-right: 1px solid {BORDER_SLATE};"
        )
        l_layout = QVBoxLayout(self.left_dock)
        l_layout.setContentsMargins(10, 10, 10, 10)
        l_layout.setSpacing(8)

        # 1. 4:3 Camera Preview (250x188px) + 1 concise status line
        self.pip_cam_lbl = QLabel()
        self.pip_cam_lbl.setFixedSize(250, 188)
        self.pip_cam_lbl.setAlignment(Qt.AlignCenter)
        self.pip_cam_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {BORDER_SLATE}; border-radius: 8px;"
        )
        l_layout.addWidget(self.pip_cam_lbl)

        self.hand_status_lbl = QLabel("Trạng thái: Chờ tay")
        self.hand_status_lbl.setStyleSheet(
            f"color: {NEON_CYAN}; font-weight: 700; font-size: 13px;"
        )
        l_layout.addWidget(self.hand_status_lbl)

        # 2. Camera Selector (Cam 0 / Cam 1) + Backlight Toggle (Ngược sáng Auto-HDR)
        cam_ctrl_row = QHBoxLayout()
        for c_idx in (0, 1):
            b = QPushButton(f"Cam {c_idx}")
            b.setFixedHeight(26)
            b.clicked.connect(lambda _=False, i=c_idx: self.cam_worker.select_camera(i))
            cam_ctrl_row.addWidget(b)

        self.hdr_chk = QCheckBox("Ngược sáng (Auto-HDR)")
        self.hdr_chk.setChecked(True)
        self.hdr_chk.toggled.connect(self._on_toggle_hdr)
        cam_ctrl_row.addWidget(self.hdr_chk)
        l_layout.addLayout(cam_ctrl_row)

        # 3. 4 Compact Gesture Badges (2x2 Grid)
        badge_grid = QGridLayout()
        badge_grid.setSpacing(6)
        self.gesture_badges: Dict[str, QLabel] = {}
        badge_specs = [
            ("DRAW", "[1 Ngón: Vẽ]", 0, 0),
            ("HOVER", "[2 Ngón: Di chuyển]", 0, 1),
            ("CLICK", "[Chụm tay: Click]", 1, 0),
            ("ERASE", "[5 Ngón: Tẩy]", 1, 1),
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

        # 4. Compact Mic Volume Bar (VU Meter) at bottom
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

        # MODULE 2.3: Clean Center Smartboard Canvas
        self.canvas = SmartboardCanvas()
        self.canvas.region_cropped_for_ai.connect(self._on_region_cropped_for_ai)
        workspace.addWidget(self.canvas, stretch=1)

        # MODULE 2.4: Right Dock (Trợ Lý Sư Phạm Gemini AI, Width = 330px)
        self.right_dock = QFrame()
        self.right_dock.setFixedWidth(330)
        self.right_dock.setStyleSheet(
            f"background: {SURFACE_DARK}; border-left: 1px solid {BORDER_SLATE};"
        )
        r_layout = QVBoxLayout(self.right_dock)
        r_layout.setContentsMargins(10, 10, 10, 10)
        r_layout.setSpacing(6)

        ai_hdr = QHBoxLayout()
        ai_title = QLabel("Trợ Lý Sư Phạm Gemini AI")
        ai_title.setStyleSheet(f"color: {TEXT_LIGHT}; font-weight: 700; font-size: 13px;")
        ai_hdr.addWidget(ai_title)
        ai_hdr.addStretch(1)
        self.change_key_btn = QPushButton("[Đổi API Key]")
        self.change_key_btn.setFixedHeight(24)
        self.change_key_btn.clicked.connect(lambda: self.api_key_frame.setVisible(True))
        ai_hdr.addWidget(self.change_key_btn)
        r_layout.addLayout(ai_hdr)

        # Compact API Key row (auto-hides once key is saved)
        self.api_key_frame = QFrame()
        kf_layout = QHBoxLayout(self.api_key_frame)
        kf_layout.setContentsMargins(0, 0, 0, 0)
        self.api_key_edit = QLineEdit(self.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Nhập GEMINI_API_KEY...")
        save_key_btn = QPushButton("Lưu")
        save_key_btn.setFixedWidth(54)
        save_key_btn.clicked.connect(self._on_save_api_key)
        kf_layout.addWidget(self.api_key_edit)
        kf_layout.addWidget(save_key_btn)
        r_layout.addWidget(self.api_key_frame)
        self.api_key_frame.setVisible(not bool(self.api_key))

        # 3 Clear Pedagogical Action Buttons
        for label, prompt_prefix in [
            (
                "[1. Giải & Chữa Bài Trên Bảng]",
                "Hãy đọc toàn bộ chữ viết tay/công thức trên bảng, kiểm tra lỗi sai nếu có và giải chi tiết từng bước.",
            ),
            (
                "[2. Tóm Tắt & Tạo Trắc Nghiệm]",
                "Hãy tóm tắt trọng tâm bài giảng trên bảng và tạo 3 câu hỏi trắc nghiệm 4 lựa chọn có đáp án giải thích.",
            ),
            (
                "[3. Hỏi Đáp Sư Phạm]",
                "Hãy gợi ý phương pháp giảng dạy trực quan và ví dụ thực tế cho nội dung trên bảng.",
            ),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {PRIMARY_INDIGO}; border-radius: 6px; font-weight: 700;"
            )
            btn.clicked.connect(lambda _=False, p=prompt_prefix: self._analyze_full_board(p))
            r_layout.addWidget(btn)

        # Cropped Region Preview Thumbnail
        self.crop_preview_lbl = QLabel()
        self.crop_preview_lbl.setFixedHeight(90)
        self.crop_preview_lbl.setAlignment(Qt.AlignCenter)
        self.crop_preview_lbl.setVisible(False)
        self.crop_preview_lbl.setStyleSheet(
            f"background: {BG_DARK}; border: 1px solid {SECONDARY_TEAL}; border-radius: 6px;"
        )
        r_layout.addWidget(self.crop_preview_lbl)

        # Clean AI Output Browser
        self.ai_output = QTextBrowser()
        self.ai_output.setStyleSheet(
            f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 8px; padding: 8px; font-size: 13px;"
        )
        self.ai_output.setPlainText(
            "Sẵn sàng phân tích bảng vẽ.\n• Dùng công cụ [Phân Tích Vùng] ở thanh dưới cùng để khoanh vùng nét vẽ gửi tới gemini-2.5-flash.\n• Hoặc bấm 1 trong 3 nút chức năng bên trên."
        )
        r_layout.addWidget(self.ai_output, stretch=1)

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
        b_layout.setContentsMargins(12, 8, 12, 8)
        b_layout.setSpacing(6)

        for tool_id, label in [
            ("SELECT_REGION", "[Phân Tích Vùng]"),
            ("PEN", "[Bút vẽ]"),
            ("ERASER", "[Tẩy]"),
            ("LASER", "[Laser]"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _=False, t=tool_id: self._set_tool(t))
            b_layout.addWidget(btn)

        self.smart_shape_btn = QPushButton("[Nắn hình: TẮT]")
        self.smart_shape_btn.setFixedHeight(32)
        self.smart_shape_btn.clicked.connect(self._on_toggle_smart_shape)
        b_layout.addWidget(self.smart_shape_btn)

        # 6 Circular Color Buttons (22x22px)
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
        thick_slider.setFixedWidth(90)
        thick_slider.valueChanged.connect(self._on_thickness_changed)
        b_layout.addWidget(thick_slider)

        b_layout.addStretch(1)

        undo_btn = QPushButton("[Hoàn tác (Undo)]")
        undo_btn.clicked.connect(self.canvas.undo)
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

    def _on_user_authenticated(self, user: Dict[str, Any]) -> None:
        self.logged_in_user = user
        self.teacher_name_lbl.setText(f"GV: {user.get('full_name', 'Giáo viên')}")
        self.stack.setCurrentIndex(1)

    def _on_logout(self) -> None:
        self.auth_gate._refresh_tabs()
        self.stack.setCurrentIndex(0)

    def _on_camera_packet(self, pkt: dict) -> None:
        if self.stack.currentIndex() == 0:
            self.auth_gate.on_camera_packet(pkt)
            return

        qimg: Optional[QImage] = pkt.get("qimage")
        gesture: str = pkt.get("gesture", "IDLE")
        cursor_norm = pkt.get("cursor_norm")

        if qimg is not None:
            scaled = qimg.scaled(250, 188, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pip_cam_lbl.setPixmap(QPixmap.fromImage(scaled))
        else:
            self.pip_cam_lbl.setText("Chưa kết nối Webcam")

        status_map = {
            "DRAW": "Trạng thái: Đang Vẽ",
            "HOVER": "Trạng thái: Di chuyển",
            "CLICK": "Trạng thái: Click",
            "ERASE": "Trạng thái: Tẩy",
            "IDLE": "Trạng thái: Chờ tay",
        }
        self.hand_status_lbl.setText(status_map.get(gesture, "Trạng thái: Chờ tay"))

        for key, lbl in self.gesture_badges.items():
            if key == gesture:
                lbl.setStyleSheet(
                    f"background: {NEON_CYAN}; color: {BG_DARK}; border-radius: 6px; font-size: 11px; font-weight: 700;"
                )
            else:
                lbl.setStyleSheet(
                    f"background: {BG_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; border-radius: 6px; font-size: 11px; font-weight: 700;"
                )

        self.canvas.apply_hand_gesture(gesture, cursor_norm)

    def _on_toggle_hdr(self, checked: bool) -> None:
        self.cam_worker.auto_hdr_backlight = checked

    def _on_prev_slide(self) -> None:
        if self.canvas.current_slide_idx > 0:
            self.canvas.current_slide_idx -= 1
            self.slide_counter_lbl.setText(
                f"Slide {self.canvas.current_slide_idx + 1}/{len(self.canvas.slides)}"
            )
            self.canvas.update()

    def _on_next_slide(self) -> None:
        if self.canvas.current_slide_idx + 1 < len(self.canvas.slides):
            self.canvas.current_slide_idx += 1
            self.slide_counter_lbl.setText(
                f"Slide {self.canvas.current_slide_idx + 1}/{len(self.canvas.slides)}"
            )
            self.canvas.update()

    def _on_toggle_grid(self) -> None:
        self.canvas.grid_enabled = not self.canvas.grid_enabled
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
                })
                self.canvas.current_slide_idx = len(self.canvas.slides) - 1
                self.slide_counter_lbl.setText(
                    f"Slide {self.canvas.current_slide_idx + 1}/{len(self.canvas.slides)}"
                )
                self.canvas.update()

    def _on_toggle_recording(self) -> None:
        if not self.is_recording_audio:
            self.audio_frames.clear()
            self.is_recording_audio = True
            self.rec_btn.setText("[Dừng & Lưu Ghi Âm]")
        else:
            self.is_recording_audio = False
            self.rec_btn.setText("[Ghi Âm Bài Giảng]")
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
        self.smart_shape_btn.setText(
            "[Nắn hình: BẬT]" if self.canvas.smart_shape_enabled else "[Nắn hình: TẮT]"
        )

    def _on_save_api_key(self) -> None:
        self.api_key = self.api_key_edit.text().strip()
        if self.api_key:
            self.api_key_frame.setVisible(False)

    def _on_region_cropped_for_ai(self, png_bytes: bytes, cropped_pm: QPixmap) -> None:
        self.right_dock.setVisible(True)
        self.crop_preview_lbl.setVisible(True)
        self.crop_preview_lbl.setPixmap(
            cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.ai_output.setPlainText("Đang phân tích nét vẽ vùng chọn...")
        prompt = "Hãy đọc chính xác chữ viết tay hoặc công thức toán/hóa trong ảnh vùng chọn này và giải thích từng bước."
        self._start_gemini_worker(prompt, png_bytes)

    def _analyze_full_board(self, prompt: str) -> None:
        self.right_dock.setVisible(True)
        png_bytes, cropped_pm = self.canvas.export_board_png(None)
        self.crop_preview_lbl.setVisible(True)
        self.crop_preview_lbl.setPixmap(
            cropped_pm.scaled(300, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.ai_output.setPlainText("Đang phân tích nét vẽ trên bảng với Gemini 2.5 Flash...")
        self._start_gemini_worker(prompt, png_bytes)

    def _start_gemini_worker(self, prompt: str, png_bytes: Optional[bytes]) -> None:
        self.gemini_worker = GeminiVisionWorker(self.api_key, prompt, png_bytes)
        self.gemini_worker.finished_text.connect(self.ai_output.setMarkdown)
        self.gemini_worker.start()

    def closeEvent(self, event) -> None:
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        self.cam_worker.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
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
