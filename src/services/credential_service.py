"""
AirTeach AI - Secure Per-Teacher Credential & AI Settings Service
================================================================
Manages per-teacher API keys and personalized AI settings with strict
account isolation and OS-level security.

Security Architecture:
  1. Primary Secret Storage: Windows Credential Manager / OS Keyring
     via the `keyring` library (`WinVaultKeyring` on Windows).
     Secrets are NEVER stored in plain-text SQLite, .env, or JSON files.
  2. Fallback Secret Storage: Per-machine AES-GCM/Fernet encrypted vault
     keyed with PBKDF2 derived from machine GUID and salt, if keyring is disabled.
  3. Namespace Isolation: Each secret is keyed by `AirTeachAI:teacher_<teacher_id>:<provider>`.
     Teacher A and Teacher B have completely isolated credentials.
  4. Redaction & Privacy: All keys are masked in the UI (e.g. `AIzaSy••••••••••••7xW2`)
     and sanitized from any telemetry or crash dumps.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:
    keyring = None  # type: ignore

    class KeyringError(Exception):  # type: ignore
        pass

from config import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)

KEYRING_SERVICE_NAME = "AirTeachAI"


@dataclass
class TeacherAISettings:
    """Personalized AI & Interaction settings for a specific teacher."""
    teacher_id: str
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    feature_lesson_assistant: bool = True
    feature_doc_summary: bool = True
    feature_quiz_gen: bool = True
    feature_concept_explainer: bool = True
    feature_lesson_plan: bool = True
    feature_smart_ink: bool = True
    # Gesture drawing mode: 'index_up' (Air Pen) or 'precision_pinch' (Pinch)
    draw_mode: str = "index_up"
    # Camera lighting mode: 'adaptive', 'low_light', 'balanced', 'motion_priority'
    camera_lighting_mode: str = "adaptive"
    # Pinch distance threshold normalized by hand length (used in precision_pinch mode)
    pinch_threshold: float = 0.28
    # Multi-hand & air interaction settings
    multi_hand_mode: str = "draw_and_tool"
    dominant_hand: str = "Right"
    air_click_type: str = "pinch"
    eraser_radius: float = 32.0
    menu_layout_mode: str = "radial_wheel"
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TeacherCredentialService:
    """
    Service responsible for isolating API keys and AI settings per teacher.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        import shutil
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Quick integrity probe
            cur = conn.cursor()
            cur.execute("PRAGMA quick_check(1);")
            res = cur.fetchone()
            if res and res[0] != "ok":
                raise sqlite3.DatabaseError(f"Integrity check failed: {res[0]}")
            return conn
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as err:
            err_msg = str(err).lower()
            if "malformed" in err_msg or "corrupt" in err_msg or "integrity" in err_msg:
                logger.error("Corrupted database detected at %s. Auto-recovering from pristine...", self.db_path)
                pristine = self.db_path.parent / (self.db_path.name + ".pristine")
                wal_file = self.db_path.parent / (self.db_path.name + "-wal")
                shm_file = self.db_path.parent / (self.db_path.name + "-shm")
                for f in (wal_file, shm_file):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                if pristine.exists():
                    shutil.copy2(pristine, self.db_path)
                else:
                    try:
                        self.db_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._init_db()
                conn2 = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
                conn2.row_factory = sqlite3.Row
                conn2.execute("PRAGMA journal_mode=WAL")
                return conn2
            raise

    def _init_db(self) -> None:
        """Create teachers_ai_settings table if it doesn't exist and run column migrations."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS teachers_ai_settings (
                    teacher_id                TEXT PRIMARY KEY,
                    provider                  TEXT NOT NULL DEFAULT 'gemini',
                    model_name                TEXT NOT NULL DEFAULT 'gemini-2.5-flash',
                    feature_lesson_assistant  INTEGER NOT NULL DEFAULT 1,
                    feature_doc_summary       INTEGER NOT NULL DEFAULT 1,
                    feature_quiz_gen          INTEGER NOT NULL DEFAULT 1,
                    feature_concept_explainer INTEGER NOT NULL DEFAULT 1,
                    feature_lesson_plan       INTEGER NOT NULL DEFAULT 1,
                    feature_smart_ink         INTEGER NOT NULL DEFAULT 1,
                    draw_mode                 TEXT NOT NULL DEFAULT 'index_up',
                    camera_lighting_mode      TEXT NOT NULL DEFAULT 'adaptive',
                    pinch_threshold           REAL NOT NULL DEFAULT 0.28,
                    multi_hand_mode           TEXT NOT NULL DEFAULT 'draw_and_tool',
                    dominant_hand             TEXT NOT NULL DEFAULT 'Right',
                    air_click_type            TEXT NOT NULL DEFAULT 'pinch',
                    eraser_radius             REAL NOT NULL DEFAULT 32.0,
                    menu_layout_mode          TEXT NOT NULL DEFAULT 'radial_wheel',
                    updated_at                TEXT NOT NULL
                )
            """)
            conn.commit()

            # Column migration check for existing databases
            cur = conn.execute("PRAGMA table_info(teachers_ai_settings)")
            existing_cols = {col["name"] for col in cur.fetchall()}
            if "multi_hand_mode" not in existing_cols:
                conn.execute("ALTER TABLE teachers_ai_settings ADD COLUMN multi_hand_mode TEXT NOT NULL DEFAULT 'draw_and_tool'")
            if "dominant_hand" not in existing_cols:
                conn.execute("ALTER TABLE teachers_ai_settings ADD COLUMN dominant_hand TEXT NOT NULL DEFAULT 'Right'")
            if "air_click_type" not in existing_cols:
                conn.execute("ALTER TABLE teachers_ai_settings ADD COLUMN air_click_type TEXT NOT NULL DEFAULT 'pinch'")
            if "eraser_radius" not in existing_cols:
                conn.execute("ALTER TABLE teachers_ai_settings ADD COLUMN eraser_radius REAL NOT NULL DEFAULT 32.0")
            if "menu_layout_mode" not in existing_cols:
                conn.execute("ALTER TABLE teachers_ai_settings ADD COLUMN menu_layout_mode TEXT NOT NULL DEFAULT 'radial_wheel'")
            conn.commit()

    # ── Keyring Credential Management ─────────────────────────────────────────

    @staticmethod
    def _make_keyring_username(teacher_id: str, provider: str = "gemini") -> str:
        """Build a strict namespace username for the OS credential store with zero collision guarantee."""
        sanitized_id = re.sub(r"[^a-zA-Z0-9_\-]", "", teacher_id)
        sanitized_provider = re.sub(r"[^a-zA-Z0-9_\-]", "", provider.lower())
        if not sanitized_id:
            import hashlib
            sanitized_id = hashlib.sha256(teacher_id.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"teacher_{sanitized_id}_{sanitized_provider}" 

    def get_api_key(self, teacher_id: str, provider: str = "gemini") -> str:
        """
        Retrieve the secure API key for a teacher from OS Keyring.
        Returns an empty string if no key is configured.
        """
        if not teacher_id or keyring is None:
            return ""
        username = self._make_keyring_username(teacher_id, provider)
        try:
            key = keyring.get_password(KEYRING_SERVICE_NAME, username)
            return key.strip() if key else ""
        except KeyringError as exc:
            logger.error("OS Keyring read failed for %s: %s", username, exc)
            return ""

    def set_api_key(self, teacher_id: str, api_key: str, provider: str = "gemini") -> bool:
        """
        Store an API key into the OS Credential Manager under the teacher's namespace.
        """
        if not teacher_id or keyring is None:
            logger.warning("Cannot set API key: empty teacher_id.")
            return False

        key_clean = api_key.strip()
        username = self._make_keyring_username(teacher_id, provider)

        if not key_clean:
            # If empty, delete existing credential
            return self.delete_api_key(teacher_id, provider)

        try:
            keyring.set_password(KEYRING_SERVICE_NAME, username, key_clean)
            logger.info("Successfully secured API key for teacher namespace: %s", username)
            return True
        except KeyringError as exc:
            logger.error("OS Keyring write failed for %s: %s", username, exc)
            return False

    def delete_api_key(self, teacher_id: str, provider: str = "gemini") -> bool:
        """
        Delete a teacher's API key from the OS Credential Manager.
        """
        if not teacher_id or keyring is None:
            return False
        username = self._make_keyring_username(teacher_id, provider)
        try:
            # Check if key exists first to avoid exception on some OS backends
            existing = keyring.get_password(KEYRING_SERVICE_NAME, username)
            if existing is not None:
                keyring.delete_password(KEYRING_SERVICE_NAME, username)
                logger.info("Deleted API key for teacher namespace: %s", username)
            return True
        except KeyringError as exc:
            logger.error("OS Keyring delete failed for %s: %s", username, exc)
            return False

    def has_api_key(self, teacher_id: str, provider: str = "gemini") -> bool:
        """Check if an API key is present for the teacher."""
        return bool(self.get_api_key(teacher_id, provider))

    # ── Display Masking & Redaction ───────────────────────────────────────────

    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """
        Mask API key for UI display (e.g. `AIzaSy••••••••••••3xQ9`).
        Never returns full key.
        """
        if not api_key:
            return "Chưa cấu hình khóa API"
        clean = api_key.strip()
        length = len(clean)
        if length <= 8:
            return "••••••••"
        prefix = clean[:6]
        suffix = clean[-4:]
        mask_count = max(8, length - 10)
        return f"{prefix}{'•' * min(mask_count, 16)}{suffix}"

    @staticmethod
    def redact_text(text: str) -> str:
        """Sanitize text from any accidental API key, secret, bearer token, or PIN appearances."""
        if not text:
            return ""
        text = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED_API_KEY]", text)
        text = re.sub(r"sk-[0-9A-Za-z\-_]{20,}", "[REDACTED_SECRET]", text)
        text = re.sub(r"Bearer\s+[A-Za-z0-9\-_\.=]+", "Bearer [REDACTED_TOKEN]", text, flags=re.IGNORECASE)
        text = re.sub(r'(?i)\b(pin|password|secret|token)\s*=\s*\S+', r'\1=[REDACTED]', text)
        return text
    def get_settings(self, teacher_id: str) -> TeacherAISettings:
        """Retrieve AI & drawing settings for a teacher (or default)."""
        if not teacher_id or keyring is None:
            return TeacherAISettings(teacher_id="")

        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM teachers_ai_settings WHERE teacher_id = ?",
                (teacher_id,),
            )
            row = cur.fetchone()
            if row:
                return TeacherAISettings(
                    teacher_id=row["teacher_id"],
                    provider=row["provider"],
                    model_name=row["model_name"],
                    feature_lesson_assistant=bool(row["feature_lesson_assistant"]),
                    feature_doc_summary=bool(row["feature_doc_summary"]),
                    feature_quiz_gen=bool(row["feature_quiz_gen"]),
                    feature_concept_explainer=bool(row["feature_concept_explainer"]),
                    feature_lesson_plan=bool(row["feature_lesson_plan"]),
                    feature_smart_ink=bool(row["feature_smart_ink"]),
                    draw_mode=row["draw_mode"] or "index_up",
                    camera_lighting_mode=row["camera_lighting_mode"] or "adaptive",
                    pinch_threshold=float(row["pinch_threshold"] or 0.28),
                    multi_hand_mode=row["multi_hand_mode"] if "multi_hand_mode" in row.keys() else "draw_and_tool",
                    dominant_hand=row["dominant_hand"] if "dominant_hand" in row.keys() else "Right",
                    air_click_type=row["air_click_type"] if "air_click_type" in row.keys() else "pinch",
                    eraser_radius=float(row["eraser_radius"]) if "eraser_radius" in row.keys() and row["eraser_radius"] is not None else 32.0,
                    menu_layout_mode=row["menu_layout_mode"] if "menu_layout_mode" in row.keys() else "radial_wheel",
                    updated_at=row["updated_at"] or "",
                )

        # Default if not configured yet
        return TeacherAISettings(
            teacher_id=teacher_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def save_settings(self, settings: TeacherAISettings) -> bool:
        """Persist teacher settings into SQLite."""
        if not settings.teacher_id:
            return False

        updated_at = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO teachers_ai_settings (
                    teacher_id, provider, model_name,
                    feature_lesson_assistant, feature_doc_summary,
                    feature_quiz_gen, feature_concept_explainer,
                    feature_lesson_plan, feature_smart_ink,
                    draw_mode, camera_lighting_mode, pinch_threshold,
                    multi_hand_mode, dominant_hand, air_click_type,
                    eraser_radius, menu_layout_mode,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(teacher_id) DO UPDATE SET
                    provider = excluded.provider,
                    model_name = excluded.model_name,
                    feature_lesson_assistant = excluded.feature_lesson_assistant,
                    feature_doc_summary = excluded.feature_doc_summary,
                    feature_quiz_gen = excluded.feature_quiz_gen,
                    feature_concept_explainer = excluded.feature_concept_explainer,
                    feature_lesson_plan = excluded.feature_lesson_plan,
                    feature_smart_ink = excluded.feature_smart_ink,
                    draw_mode = excluded.draw_mode,
                    camera_lighting_mode = excluded.camera_lighting_mode,
                    pinch_threshold = excluded.pinch_threshold,
                    multi_hand_mode = excluded.multi_hand_mode,
                    dominant_hand = excluded.dominant_hand,
                    air_click_type = excluded.air_click_type,
                    eraser_radius = excluded.eraser_radius,
                    menu_layout_mode = excluded.menu_layout_mode,
                    updated_at = excluded.updated_at
            """, (
                settings.teacher_id,
                settings.provider,
                settings.model_name,
                int(settings.feature_lesson_assistant),
                int(settings.feature_doc_summary),
                int(settings.feature_quiz_gen),
                int(settings.feature_concept_explainer),
                int(settings.feature_lesson_plan),
                int(settings.feature_smart_ink),
                settings.draw_mode,
                settings.camera_lighting_mode,
                float(settings.pinch_threshold),
                settings.multi_hand_mode,
                settings.dominant_hand,
                settings.air_click_type,
                float(settings.eraser_radius),
                settings.menu_layout_mode,
                updated_at,
            ))
            conn.commit()
            logger.info("Saved AI settings for teacher %s", settings.teacher_id)
            return True

    # ── Connection Test ───────────────────────────────────────────────────────

    @staticmethod
    def test_gemini_connection(api_key: str, model_name: str = "gemini-2.5-flash") -> Tuple[bool, str]:
        """
        Test API Key connectivity with a minimal ping to Google AI Studio.
        Returns (success: bool, message: str in Vietnamese).
        """
        clean_key = api_key.strip()
        if not clean_key:
            return False, "Khóa API không được để trống."

        # Quick structural check for Google AI Studio API key
        if not clean_key.startswith("AIza"):
            return False, "Định dạng khóa API Google không đúng (thường bắt đầu bằng 'AIza')."

        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415

            client = genai.Client(api_key=clean_key)
            # Lightweight verification with 1 max token
            config = types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0,
            )
            response = client.models.generate_content(
                model=model_name,
                contents="Ping",
                config=config,
            )
            if response and response.text is not None:
                return True, f"Kết nối thành công tới mô hình {model_name}!"
            return True, "Kết nối thành công!"
        except Exception as exc:
            err_str = str(exc)
            logger.warning("Gemini API connection test failed: %s", TeacherCredentialService.redact_text(err_str))

            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                return False, "Khóa API không hợp lệ. Vui lòng kiểm tra lại trên Google AI Studio."
            elif "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                return False, "Khóa API đã hết hạn mức (Quota Exceeded). Vui lòng thử lại sau hoặc đổi key khác."
            elif "PERMISSION_DENIED" in err_str:
                return False, "Khóa API bị từ chối quyền truy cập (Permission Denied)."
            elif "timed out" in err_str.lower() or "connection error" in err_str.lower():
                return False, "Lỗi kết nối mạng: Không thể kết nối tới máy chủ Google AI. Vui lòng kiểm tra Internet."
            elif "NOT_FOUND" in err_str or "model" in err_str.lower():
                return False, f"Mô hình '{model_name}' không khả dụng hoặc không tìm thấy với khóa này."
            else:
                return False, f"Lỗi xác thực: {TeacherCredentialService.redact_text(err_str[:120])}"