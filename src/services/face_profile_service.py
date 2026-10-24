"""
AirTeach AI - Teacher Face Profile Service
==========================================
Handles teacher enrollment (face registration) and authentication
using OpenCV YuNet (face detection) + SFace (face recognition) ONNX models.

Security:
 - Face embeddings encrypted with Fernet symmetric encryption.
 - PIN hashed with PBKDF2-HMAC-SHA256 (200,000 iterations).
 - Liveness check: tracks relative eye/nose displacement across 15 frames.

Storage: SQLite at data/profiles.db
Workspaces: data/workspaces/<teacher_uuid>/
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.363          # SFace recommended threshold
ENROLL_FRAMES = 5                  # number of face captures during enrollment
LIVENESS_FRAMES = 15               # frames to check for liveness
LIVENESS_MOTION_THRESHOLD = 0.003  # minimum relative landmark displacement
PBKDF2_ITERATIONS = 200_000
KEYSTORE_FILENAME = ".keystore"    # stored alongside profiles.db


# ─── Database Setup ──────────────────────────────────────────────────────────

def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite profiles database with schema migrations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id                TEXT PRIMARY KEY,
            teacher_code      TEXT UNIQUE,
            full_name         TEXT NOT NULL,
            subject           TEXT NOT NULL DEFAULT '',
            encrypted_embedding BLOB NOT NULL,
            pin_salt          BLOB NOT NULL,
            pin_hash          BLOB NOT NULL,
            workspace_dir     TEXT NOT NULL,
            created_at        TEXT NOT NULL
        )
    """)
    conn.commit()

    # Migration: add teacher_code column if table existed without it
    cur = conn.execute("PRAGMA table_info(teachers)")
    existing_cols = {row["name"] for row in cur.fetchall()}
    if "teacher_code" not in existing_cols:
        conn.execute("ALTER TABLE teachers ADD COLUMN teacher_code TEXT")
        conn.execute(
            "UPDATE teachers SET teacher_code = 'GV-' || UPPER(SUBSTR(id, 1, 6)) WHERE teacher_code IS NULL"
        )
        conn.commit()

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_code ON teachers(teacher_code)")
    conn.commit()
    return conn


# ─── Key Management ───────────────────────────────────────────────────────────

class _KeyStore:
    """Manages a per-application Fernet key stored on disk."""

    def __init__(self, keystore_path: Path) -> None:
        self._path = keystore_path
        self._key: Optional[bytes] = None

    def get_key(self) -> bytes:
        if self._key is not None:
            return self._key
        if self._path.exists():
            self._key = self._path.read_bytes()
        else:
            self._key = Fernet.generate_key()
            self._path.write_bytes(self._key)
            # Restrict permissions on POSIX
            try:
                os.chmod(self._path, 0o600)
            except OSError as _e:
                logger.debug("chmod %s skipped (Windows or permission denied): %s", self._path, _e)
        return self._key

    def get_fernet(self) -> Fernet:
        return Fernet(self.get_key())


# ─── Face Model Loader ────────────────────────────────────────────────────────

class FaceModels:
    """
    Lazy-loads YuNet (face detector) and SFace (face recognizer) ONNX models.
    Falls back to Haar cascade detection if ONNX weights are unavailable.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._use_fallback_detector = False

    def _load_detector(self) -> None:
        yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        if yunet_path.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(yunet_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5,
                )
                logger.info("YuNet detector loaded from %s", yunet_path)
                return
            except cv2.error as e:
                logger.warning("YuNet load failed: %s — falling back to Haar.", e)
        # Fallback: OpenCV built-in Haar cascade
        self._use_fallback_detector = True
        logger.warning("Using Haar cascade fallback for face detection.")

    def _load_recognizer(self) -> bool:
        sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        if sface_path.exists():
            try:
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(sface_path), config=""
                )
                logger.info("SFace recognizer loaded from %s", sface_path)
                return True
            except cv2.error as e:
                logger.warning("SFace load failed: %s", e)
        return False

    def ensure_loaded(self) -> None:
        if self._detector is None and not self._use_fallback_detector:
            self._load_detector()
        if self._recognizer is None:
            self._load_recognizer()

    def detect_faces(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Detect faces in *frame_bgr*. Returns (N, 15) detections or None.
        Each row: [x, y, w, h, eye_rx, eye_ry, eye_lx, eye_ly, nose_x, nose_y,
                   mouth_rx, mouth_ry, mouth_lx, mouth_ly, score]
        """
        self.ensure_loaded()
        h, w = frame_bgr.shape[:2]

        if not self._use_fallback_detector and self._detector is not None:
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(frame_bgr)
            return faces  # None if no faces

        # Haar fallback — return a synthetic (N, 15) row with approximate landmarks
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        import os as _os
        cascade_available = _os.path.isfile(cascade_path)
        if cascade_available:
            cascade = cv2.CascadeClassifier(cascade_path)
            cascade_ok = not cascade.empty()
        else:
            cascade_ok = False

        if cascade_ok:
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
        else:
            # OpenCV 5+ does not ship Haar XML files; synthesize a full-frame face
            # so that enrollment can proceed and store a placeholder embedding.
            # Real face auth will use YuNet ONNX once models are downloaded.
            logger.warning(
                "Haar cascade XML not found at '%s'. "
                "Returning synthetic full-frame detection (PIN-only auth mode).",
                cascade_path,
            )
            fh_full, fw_full = frame_bgr.shape[:2]
            cx, cy = fw_full // 2, fh_full // 2
            synthetic_row = np.array([[
                0, 0, fw_full, fh_full,
                cx - fw_full // 5, cy - fh_full // 8,
                cx + fw_full // 5, cy - fh_full // 8,
                cx, cy,
                cx - fw_full // 5, cy + fh_full // 5,
                cx + fw_full // 5, cy + fh_full // 5,
                0.50,   # low confidence score → signals placeholder
            ]], dtype=np.float32)
            return synthetic_row

        if len(rects) == 0:
            return None
        faces_out = []
        for (x, y, fw, fh) in rects:
            score = 0.90
            cx, cy = x + fw // 2, y + fh // 2
            row = [x, y, fw, fh,
                   cx - fw // 5, cy - fh // 8,   # right eye approx
                   cx + fw // 5, cy - fh // 8,   # left eye approx
                   cx, cy,                         # nose
                   cx - fw // 5, cy + fh // 5,   # mouth right
                   cx + fw // 5, cy + fh // 5,   # mouth left
                   score]
            faces_out.append(row)
        return np.array(faces_out, dtype=np.float32)

    def extract_feature(self, frame_bgr: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
        """
        Align-crop the face region and extract a 128-D L2-normalized feature vector.
        Returns None if the recognizer model is unavailable.
        """
        self.ensure_loaded()
        if self._recognizer is None:
            return None
        face_box = face_row[np.newaxis, :]  # shape (1, 15)
        try:
            aligned = self._recognizer.alignCrop(frame_bgr, face_box[0])
            feat = self._recognizer.feature(aligned)   # shape (1, 128)
            feat = feat.flatten().astype(np.float64)
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat /= norm
            return feat
        except cv2.error as e:
            logger.warning("SFace feature extraction failed: %s", e)
            return None

    def cosine_similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """Cosine similarity ∈ [0, 1] between two L2-normalized feature vectors."""
        return float(np.dot(feat_a, feat_b))


# ─── Liveness Checker ─────────────────────────────────────────────────────────

class LivenessChecker:
    """
    Anti-spoofing: tracks relative landmark displacement across consecutive frames.
    If nose tip doesn't move relative to face bounding box over LIVENESS_FRAMES,
    the input is likely a static photo → reject.
    """

    def __init__(self, required_frames: int = LIVENESS_FRAMES) -> None:
        self._required = required_frames
        self._nose_history: list[tuple[float, float]] = []
        self._face_size_history: list[float] = []

    def reset(self) -> None:
        self._nose_history.clear()
        self._face_size_history.clear()

    def update(self, face_row: np.ndarray) -> bool:
        """
        Feed one face detection row. Returns True when liveness is confirmed.
        Liveness = relative nose position variance > LIVENESS_MOTION_THRESHOLD.
        """
        x, y, w, h = face_row[:4]
        nose_x, nose_y = face_row[8], face_row[9]
        # Normalize nose by face bbox
        rel_x = (nose_x - x) / max(w, 1)
        rel_y = (nose_y - y) / max(h, 1)
        self._nose_history.append((rel_x, rel_y))
        self._face_size_history.append(float(w * h))
        if len(self._nose_history) > self._required * 2:
            self._nose_history.pop(0)
            self._face_size_history.pop(0)

        if len(self._nose_history) < self._required:
            return False

        xs = [p[0] for p in self._nose_history]
        ys = [p[1] for p in self._nose_history]
        motion = float(np.std(xs) + np.std(ys))
        return motion > LIVENESS_MOTION_THRESHOLD

    def check(self) -> bool:
        """Evaluate if recent face history exhibits authentic physiological motion."""
        if len(self._nose_history) < self._required:
            return False
        xs = [p[0] for p in self._nose_history]
        ys = [p[1] for p in self._nose_history]
        motion = float(np.std(xs) + np.std(ys))
        return motion > LIVENESS_MOTION_THRESHOLD

    def is_ready(self) -> bool:
        return len(self._nose_history) >= self._required


# ─── Teacher Profile Service (Main Class) ─────────────────────────────────────

class TeacherProfileService:
    """
    Manages teacher registration, authentication, and workspace lifecycle.

    Args:
        db_path:    Path to SQLite database file.
        models_dir: Directory containing YuNet and SFace ONNX weights.
    """

    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_SECONDS: int = 30

    def __init__(self, db_path: Path, models_dir: Path) -> None:
        self.db_path = db_path
        self.models_dir = models_dir
        self._conn = _get_connection(db_path)
        self._keystore = _KeyStore(db_path.parent / KEYSTORE_FILENAME)
        self._models = FaceModels(models_dir)
        self.last_match_score: float = -1.0
        self.last_match_candidate: str = ""
        self._failed_attempts: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def normalize_teacher_code(code: Optional[str]) -> str:
        """Strip and normalize teacher code to uppercase with clean dashes."""
        if not code:
            return ""
        cleaned = code.strip().upper()
        return "-".join(cleaned.split())

    @staticmethod
    def teacher_row_to_dict(row: Any) -> dict:
        """Convert a sqlite3.Row or dict to a uniform dictionary."""
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        keys = row.keys() if hasattr(row, "keys") else []
        return {
            "id": row["id"],
            "teacher_code": row["teacher_code"] if "teacher_code" in keys and row["teacher_code"] else row["id"][:8].upper(),
            "full_name": row["full_name"],
            "subject": row["subject"] if "subject" in keys else "",
            "workspace_dir": row["workspace_dir"] if "workspace_dir" in keys else "",
            "created_at": row["created_at"] if "created_at" in keys else datetime.now(timezone.utc).isoformat(),
        }

    def is_locked(self, teacher_key: str) -> tuple[bool, int]:
        """Check if an account/key is locked after too many failed attempts."""
        key = self.normalize_teacher_code(teacher_key)
        lock_until = self._locked_until.get(key, 0.0)
        remaining = int(lock_until - time.time())
        if remaining > 0:
            return True, remaining
        if key in self._locked_until:
            del self._locked_until[key]
        return False, 0

    def record_failed_attempt(self, teacher_key: str) -> int:
        """Record a failed PIN attempt and lock if threshold reached."""
        key = self.normalize_teacher_code(teacher_key)
        attempts = self._failed_attempts.get(key, 0) + 1
        self._failed_attempts[key] = attempts
        if attempts >= self.MAX_FAILED_ATTEMPTS:
            self._locked_until[key] = time.time() + self.LOCKOUT_SECONDS
            self._failed_attempts[key] = 0
            logger.warning("Teacher '%s' locked for %d seconds due to %d failed attempts.", key, self.LOCKOUT_SECONDS, attempts)
        return attempts

    def reset_failed_attempts(self, teacher_key: str) -> None:
        """Reset failed attempt counters upon successful login."""
        key = self.normalize_teacher_code(teacher_key)
        self._failed_attempts.pop(key, None)
        self._locked_until.pop(key, None)

    def _encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        fernet = self._keystore.get_fernet()
        raw = embedding.astype(np.float64).tobytes()
        return fernet.encrypt(raw)

    def _decrypt_embedding(self, blob: bytes) -> np.ndarray:
        fernet = self._keystore.get_fernet()
        raw = fernet.decrypt(blob)
        return np.frombuffer(raw, dtype=np.float64).copy()

    @staticmethod
    def _hash_pin(pin: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256",
            pin.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
        )

    def _verify_pin(self, teacher_row: sqlite3.Row, pin: str) -> bool:
        salt = bytes(teacher_row["pin_salt"])
        expected = bytes(teacher_row["pin_hash"])
        actual = self._hash_pin(pin, salt)
        return secrets.compare_digest(actual, expected)

    def _get_all_teachers(self) -> list[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM teachers")
        return cur.fetchall()

    def get_teacher_by_code(self, teacher_code: str) -> Optional[sqlite3.Row]:
        """Lookup teacher by unique teacher code (case-insensitive) or UUID prefix."""
        code = self.normalize_teacher_code(teacher_code)
        if not code:
            return None
        cur = self._conn.execute("SELECT * FROM teachers WHERE teacher_code = ?", (code,))
        row = cur.fetchone()
        if row is None:
            cur = self._conn.execute(
                "SELECT * FROM teachers WHERE id = ? OR UPPER(id) LIKE ?",
                (teacher_code, f"{code}%"),
            )
            row = cur.fetchone()
        return row

    def get_teacher_by_id(self, teacher_id: str) -> Optional[sqlite3.Row]:
        """Lookup teacher by exact UUID id."""
        cur = self._conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        return cur.fetchone()

    # ── Enrollment ────────────────────────────────────────────────────

    def enroll_from_frames(
        self,
        frames: list[np.ndarray],
        full_name: str,
        subject: str,
        pin: str,
        teacher_code: Optional[str] = None,
    ) -> str:
        """
        Register a new teacher from a list of face frames (at least ENROLL_FRAMES).

        Args:
            frames:       List of BGR frames captured from webcam.
            full_name:    Teacher's display name.
            subject:      Teaching subject.
            pin:          4–6 digit backup PIN.
            teacher_code: Optional custom teacher ID/code (e.g. 'GV01'). Auto-generated if omitted.

        Returns:
            The new teacher UUID string.

        Raises:
            ValueError: If invalid fields, duplicate code, or insufficient faces.
            RuntimeError: If database persistence verification fails.
        """
        if not (4 <= len(pin) <= 6 and pin.isdigit()):
            raise ValueError("Mã PIN phải gồm từ 4 đến 6 chữ số.")
        name = full_name.strip()
        if not name:
            raise ValueError("Họ và tên giáo viên không được để trống.")

        code = self.normalize_teacher_code(teacher_code)
        if code:
            existing = self.get_teacher_by_code(code)
            if existing is not None:
                raise ValueError(f"Mã giáo viên '{code}' đã tồn tại trong hệ thống. Vui lòng chọn mã khác.")
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM teachers")
            count = cur.fetchone()[0] + 1
            candidate = f"GV{count:02d}"
            while self.get_teacher_by_code(candidate) is not None:
                count += 1
                candidate = f"GV{count:02d}"
            code = candidate

        embeddings: list[np.ndarray] = []
        for frame in frames:
            faces = self._models.detect_faces(frame)
            if faces is None or len(faces) == 0:
                continue
            best = faces[np.argmax(faces[:, 14])]  # highest confidence
            feat = self._models.extract_feature(frame, best)
            if feat is not None:
                embeddings.append(feat)

        if len(embeddings) < max(1, ENROLL_FRAMES // 2):
            if self._models._recognizer is None:
                logger.warning("SFace not available; storing placeholder embedding for PIN-only auth.")
                embeddings = [np.zeros(128, dtype=np.float64)]
            else:
                raise ValueError(
                    f"Only {len(embeddings)} valid face detections from {len(frames)} frames. "
                    f"Need at least {ENROLL_FRAMES // 2}."
                )

        # Average embedding, L2-normalize
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-6:
            avg_embedding /= norm

        teacher_id = str(uuid.uuid4())
        workspace_dir = str(self.db_path.parent / "workspaces" / teacher_id)
        ws_path = Path(workspace_dir)
        ws_path.mkdir(parents=True, exist_ok=True)
        # Write marker file to verify filesystem access
        init_marker = ws_path / ".workspace_init"
        init_marker.write_text(
            json.dumps({"teacher_id": teacher_id, "teacher_code": code, "name": name}),
            encoding="utf-8",
        )

        pin_salt = secrets.token_bytes(32)
        pin_hash = self._hash_pin(pin, pin_salt)
        encrypted_blob = self._encrypt_embedding(avg_embedding)
        created_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO teachers
                (id, teacher_code, full_name, subject, encrypted_embedding, pin_salt, pin_hash, workspace_dir, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (teacher_id, code, name, subject.strip(), encrypted_blob, pin_salt, pin_hash,
             workspace_dir, created_at),
        )
        self._conn.commit()

        # Strict read-back verification from storage
        cur = self._conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        saved_row = cur.fetchone()
        if saved_row is None:
            raise RuntimeError("Lỗi xác thực lưu trữ: Không tìm thấy hồ sơ giáo viên vừa tạo trong cơ sở dữ liệu.")

        logger.info("Enrolled teacher '%s' (code=%s, id=%s)", name, code, teacher_id)
        return teacher_id

    def enroll_teacher_pin_only(
        self,
        full_name: str,
        subject: str,
        pin: str,
        teacher_code: Optional[str] = None,
    ) -> str:
        """
        Register a new teacher profile using PIN-only authentication (no webcam required).
        """
        if not (4 <= len(pin) <= 6 and pin.isdigit()):
            raise ValueError("Mã PIN phải gồm từ 4 đến 6 chữ số.")
        name = full_name.strip()
        if not name:
            raise ValueError("Họ và tên giáo viên không được để trống.")

        code = self.normalize_teacher_code(teacher_code)
        if code:
            existing = self.get_teacher_by_code(code)
            if existing is not None:
                raise ValueError(f"Mã giáo viên '{code}' đã tồn tại trong hệ thống. Vui lòng chọn mã khác.")
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM teachers")
            count = cur.fetchone()[0] + 1
            candidate = f"GV{count:02d}"
            while self.get_teacher_by_code(candidate) is not None:
                count += 1
                candidate = f"GV{count:02d}"
            code = candidate

        placeholder_embedding = np.zeros(128, dtype=np.float64)
        teacher_id = str(uuid.uuid4())
        workspace_dir = str(self.db_path.parent / "workspaces" / teacher_id)
        ws_path = Path(workspace_dir)
        ws_path.mkdir(parents=True, exist_ok=True)
        init_marker = ws_path / ".workspace_init"
        init_marker.write_text(
            json.dumps({"teacher_id": teacher_id, "teacher_code": code, "name": name}),
            encoding="utf-8",
        )

        pin_salt = secrets.token_bytes(32)
        pin_hash = self._hash_pin(pin, pin_salt)
        encrypted_blob = self._encrypt_embedding(placeholder_embedding)
        created_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO teachers
                (id, teacher_code, full_name, subject, encrypted_embedding, pin_salt, pin_hash, workspace_dir, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (teacher_id, code, name, subject.strip(), encrypted_blob, pin_salt, pin_hash,
             workspace_dir, created_at),
        )
        self._conn.commit()

        # Strict read-back verification from storage
        cur = self._conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        saved_row = cur.fetchone()
        if saved_row is None:
            raise RuntimeError("Lỗi xác thực lưu trữ: Không tìm thấy hồ sơ giáo viên vừa tạo trong cơ sở dữ liệu.")

        logger.info("Enrolled teacher (PIN-only) '%s' (code=%s, id=%s)", name, code, teacher_id)
        return teacher_id

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate_by_face(
        self,
        frame: np.ndarray,
        liveness_checker: LivenessChecker,
    ) -> Optional[sqlite3.Row]:
        """
        Try to authenticate the person in *frame* by face recognition.

        Args:
            frame:            BGR webcam frame.
            liveness_checker: LivenessChecker that has been updated with
                              recent frames to verify anti-spoofing.

        Returns:
            Teacher DB row if authenticated, else None.
        """
        faces = self._models.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return None

        best_face = faces[np.argmax(faces[:, 14])]
        is_lively = liveness_checker.update(best_face)

        if not liveness_checker.is_ready():
            logger.debug("Liveness check not yet ready.")
            return None

        if not is_lively and not liveness_checker.check():
            logger.debug("Liveness anti-spoofing check failed (insufficient micro-motion).")
            return None

        probe_feat = self._models.extract_feature(frame, best_face)
        if probe_feat is None:
            logger.debug("Feature extraction failed (model unavailable).")
            return None

        teachers = self._get_all_teachers()
        if not teachers:
            self.last_match_score = -1.0
            self.last_match_candidate = ""
            return None

        best_match: Optional[sqlite3.Row] = None
        best_score = -1.0

        for row in teachers:
            try:
                gallery_feat = self._decrypt_embedding(bytes(row["encrypted_embedding"]))
                score = self._models.cosine_similarity(probe_feat, gallery_feat)
                if score > best_score:
                    best_score = score
                    best_match = row
            except Exception as exc:
                logger.warning("Embedding decrypt/compare failed for %s: %s", row["id"], exc)

        self.last_match_score = best_score
        self.last_match_candidate = best_match["full_name"] if best_match else ""

        if best_score >= COSINE_THRESHOLD and best_match is not None:
            logger.info(
                "Face auth SUCCESS: %s (score=%.3f)", best_match["full_name"], best_score
            )
            return best_match

        logger.debug("Face auth failed. Best score=%.3f (threshold=%.3f)", best_score, COSINE_THRESHOLD)
        return None

    def authenticate_by_teacher_code_and_pin(
        self,
        teacher_code: str,
        pin: str,
    ) -> Optional[sqlite3.Row]:
        """
        Authenticate by teacher code (e.g. 'GV01') and PIN with rate-limiting.
        """
        code = self.normalize_teacher_code(teacher_code)
        if not code:
            return None

        locked, remaining = self.is_locked(code)
        if locked:
            logger.warning("Teacher '%s' is locked (%d s remaining).", code, remaining)
            return None

        row = self.get_teacher_by_code(code)
        if row is None:
            logger.warning("No teacher found with code '%s'.", code)
            return None

        if self._verify_pin(row, pin):
            self.reset_failed_attempts(code)
            logger.info("Code+PIN auth SUCCESS for teacher code=%s", code)
            return row

        self.record_failed_attempt(code)
        return None

    def authenticate_by_pin(
        self,
        teacher_id: str,
        pin: str,
    ) -> Optional[sqlite3.Row]:
        """
        Fallback PIN authentication by teacher id or code with rate-limiting.

        Returns:
            Teacher DB row if PIN matches, else None.
        """
        locked, remaining = self.is_locked(teacher_id)
        if locked:
            logger.warning("Teacher id '%s' is locked (%d s remaining).", teacher_id, remaining)
            return None

        cur = self._conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        row = cur.fetchone()
        if row is None:
            # Fallback to teacher_code search
            row = self.get_teacher_by_code(teacher_id)
            if row is None:
                return None

        if self._verify_pin(row, pin):
            self.reset_failed_attempts(teacher_id)
            logger.info("PIN auth SUCCESS for teacher id=%s", teacher_id)
            return row

        self.record_failed_attempt(teacher_id)
        return None

    def authenticate_by_name_and_pin(
        self,
        full_name: str,
        pin: str,
    ) -> Optional[sqlite3.Row]:
        """
        PIN-only authentication by full name with rate-limiting.
        """
        name_clean = full_name.strip()
        locked, remaining = self.is_locked(name_clean)
        if locked:
            logger.warning("Teacher name '%s' is locked (%d s remaining).", name_clean, remaining)
            return None

        cur = self._conn.execute(
            "SELECT * FROM teachers WHERE LOWER(full_name) = LOWER(?)", (name_clean,)
        )
        rows = cur.fetchall()
        for row in rows:
            if self._verify_pin(row, pin):
                self.reset_failed_attempts(name_clean)
                logger.info("PIN-by-name auth SUCCESS for '%s'", name_clean)
                return row

        self.record_failed_attempt(name_clean)
        return None

    # ── Profile Management ────────────────────────────────────────────────────

    def list_teachers(self) -> list[dict]:
        """Return lightweight dicts of all enrolled teachers including teacher_code."""
        teachers = self._get_all_teachers()
        res = []
        for row in teachers:
            keys = row.keys() if hasattr(row, "keys") else []
            t_code = row["teacher_code"] if "teacher_code" in keys and row["teacher_code"] else row["id"][:8].upper()
            res.append({
                "id": row["id"],
                "teacher_code": t_code,
                "full_name": row["full_name"],
                "subject": row["subject"] if "subject" in keys else "",
                "workspace_dir": row["workspace_dir"] if "workspace_dir" in keys else "",
                "created_at": row["created_at"] if "created_at" in keys else "",
            })
        return res

    def delete_teacher(self, teacher_id: str) -> bool:
        """Remove a teacher profile. Does NOT delete workspace files."""
        cur = self._conn.execute(
            "DELETE FROM teachers WHERE id = ?", (teacher_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted teacher id=%s", teacher_id)
        return deleted

    def get_workspace_path(self, teacher_id: str) -> Optional[Path]:
        """Return the workspace Path for a teacher, or None if not found."""
        cur = self._conn.execute(
            "SELECT workspace_dir FROM teachers WHERE id = ?", (teacher_id,)
        )
        row = cur.fetchone()
        if row:
            return Path(row["workspace_dir"])
        return None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
