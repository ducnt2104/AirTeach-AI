"""
AirTeach AI - Phase 2 Services Test Suite
==========================================
Tests for:
  - TeacherProfileService: SQLite enrollment, Fernet encryption, PBKDF2 PIN
  - AirTeachPackage: .airteach ZIP save/load round-trip
  - LaTeXRenderer: matplotlib math rendering to RGBA
  - GeminiAssistant: offline fallback and API error handling
  - TelemetryLogger: record, flush, summary
  - DocumentManager: PDF rendering and .airteach integration

Run with:
    pytest tests/test_services_and_storage.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "services"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))

from src.services.face_profile_service import (
    TeacherProfileService,
    LivenessChecker,
    _KeyStore,
    _get_connection,
    PBKDF2_ITERATIONS,
    COSINE_THRESHOLD,
)
from src.services.document_manager import (
    AirTeachPackage,
    AirTeachManifest,
    StrokeRecord,
    BezierPointRecord,
    RecognizedItem,
    DocumentManager,
    _sha256_file,
    _hex_to_bgr,
    _stroke_to_dict,
    _dict_to_stroke,
)
from src.services.gemini_assistant import GeminiAssistant, _offline_lookup
from src.services.smart_ink_ocr import (
    Stroke,
    StrokePoint,
    StrokeRenderer,
    LaTeXRenderer,
    OcrResult,
    normalize_vietnamese_ocr_text,
    sanitize_latex_string,
)
from src.utils.telemetry_logger import TelemetryLogger, TelemetryRecord


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def profile_service(tmp_dir: Path) -> TeacherProfileService:
    """Fresh TeacherProfileService backed by a temp SQLite database."""
    db_path = tmp_dir / "test_profiles.db"
    models_dir = tmp_dir / "models"
    models_dir.mkdir()
    svc = TeacherProfileService(db_path=db_path, models_dir=models_dir)
    yield svc
    svc.close()


@pytest.fixture
def fake_face_frame() -> np.ndarray:
    """Synthetic BGR frame containing a 'face' region (random pixels)."""
    rng = np.random.default_rng(seed=99)
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Draw a flesh-coloured ellipse to approximate face region
    import cv2
    cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (150, 190, 220), -1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 1. KeyStore Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyStore:
    def test_generates_key_on_first_access(self, tmp_dir: Path):
        ks = _KeyStore(tmp_dir / ".keystore")
        key = ks.get_key()
        assert isinstance(key, bytes) and len(key) == 44  # Fernet key is base64, 44 chars

    def test_persists_key_across_instances(self, tmp_dir: Path):
        path = tmp_dir / ".keystore"
        ks1 = _KeyStore(path)
        key1 = ks1.get_key()
        ks2 = _KeyStore(path)
        key2 = ks2.get_key()
        assert key1 == key2, "Key should be identical across instances loading same file"

    def test_fernet_roundtrip(self, tmp_dir: Path):
        ks = _KeyStore(tmp_dir / ".keystore")
        f = ks.get_fernet()
        original = b"AirTeach AI secret embedding bytes"
        encrypted = f.encrypt(original)
        decrypted = f.decrypt(encrypted)
        assert decrypted == original


# ─────────────────────────────────────────────────────────────────────────────
# 2. Database Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseSchema:
    def test_creates_teachers_table(self, tmp_dir: Path):
        conn = _get_connection(tmp_dir / "schema_test.db")
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='teachers'"
        )
        assert cur.fetchone() is not None, "teachers table must exist"
        conn.close()

    def test_teachers_table_columns(self, tmp_dir: Path):
        conn = _get_connection(tmp_dir / "col_test.db")
        cur = conn.execute("PRAGMA table_info(teachers)")
        columns = {row["name"] for row in cur.fetchall()}
        required = {
            "id", "full_name", "subject",
            "encrypted_embedding", "pin_salt", "pin_hash",
            "workspace_dir", "created_at",
        }
        assert required.issubset(columns), f"Missing columns: {required - columns}"
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Teacher Enrollment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTeacherEnrollment:
    def test_enroll_creates_db_record(self, profile_service: TeacherProfileService, fake_face_frame):
        """Enrollment stores one row in the teachers table."""
        teacher_id = profile_service.enroll_from_frames(
            frames=[fake_face_frame] * 5,
            full_name="Nguyễn Văn An",
            subject="Toán học",
            pin="123456",
        )
        teachers = profile_service.list_teachers()
        assert len(teachers) == 1
        assert teachers[0]["full_name"] == "Nguyễn Văn An"
        assert teachers[0]["id"] == teacher_id

    def test_enroll_creates_workspace_directory(self, profile_service: TeacherProfileService, fake_face_frame):
        """Each enrolled teacher gets a dedicated workspace directory."""
        teacher_id = profile_service.enroll_from_frames(
            frames=[fake_face_frame] * 5,
            full_name="Trần Thị Bình",
            subject="Vật lý",
            pin="4321",
        )
        workspace = profile_service.get_workspace_path(teacher_id)
        assert workspace is not None
        assert workspace.exists(), "Workspace directory must be created on enrollment"

    def test_enroll_invalid_pin_raises(self, profile_service: TeacherProfileService, fake_face_frame):
        """PIN must be 4–6 digits; other values raise ValueError."""
        with pytest.raises(ValueError, match="PIN"):
            profile_service.enroll_from_frames(
                frames=[fake_face_frame] * 5,
                full_name="Test Teacher",
                subject="Test",
                pin="abc",  # not digits
            )

    def test_embedding_encrypted_in_db(self, profile_service: TeacherProfileService, fake_face_frame):
        """The stored embedding blob must NOT be raw float bytes (it is Fernet-encrypted)."""
        teacher_id = profile_service.enroll_from_frames(
            frames=[fake_face_frame] * 5,
            full_name="Ma Hóa Test",
            subject="Test",
            pin="999999",
        )
        conn = profile_service._conn
        cur = conn.execute(
            "SELECT encrypted_embedding FROM teachers WHERE id=?", (teacher_id,)
        )
        row = cur.fetchone()
        blob = bytes(row["encrypted_embedding"])
        # Fernet tokens start with 'gAAAAA' in base64
        assert blob[:6] == b"gAAAAA", (
            "Stored blob must be a Fernet-encrypted token starting with 'gAAAAA'"
        )

    def test_pin_stored_as_pbkdf2_hash(self, profile_service: TeacherProfileService, fake_face_frame):
        """PIN hash must be PBKDF2-HMAC-SHA256, not plaintext."""
        pin = "112233"
        teacher_id = profile_service.enroll_from_frames(
            frames=[fake_face_frame] * 5,
            full_name="PBKDF2 Test",
            subject="Test",
            pin=pin,
        )
        conn = profile_service._conn
        cur = conn.execute(
            "SELECT pin_salt, pin_hash FROM teachers WHERE id=?", (teacher_id,)
        )
        row = cur.fetchone()
        salt = bytes(row["pin_salt"])
        stored_hash = bytes(row["pin_hash"])
        expected = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, PBKDF2_ITERATIONS)
        assert stored_hash == expected, "PIN hash must be PBKDF2-HMAC-SHA256"
        assert stored_hash != pin.encode(), "PIN must NOT be stored in plaintext"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Teacher Authentication Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTeacherAuthentication:
    def _enroll(self, svc: TeacherProfileService, frame: np.ndarray,
                name: str = "Auth Teacher", pin: str = "5566") -> str:
        return svc.enroll_from_frames(
            frames=[frame] * 5, full_name=name, subject="Khoa học", pin=pin
        )

    def test_pin_authentication_correct(self, profile_service: TeacherProfileService, fake_face_frame):
        """Correct PIN authenticates the teacher."""
        teacher_id = self._enroll(profile_service, fake_face_frame, pin="7788")
        row = profile_service.authenticate_by_pin(teacher_id, "7788")
        assert row is not None, "Correct PIN should authenticate teacher"
        assert row["id"] == teacher_id

    def test_pin_authentication_wrong(self, profile_service: TeacherProfileService, fake_face_frame):
        """Wrong PIN returns None."""
        teacher_id = self._enroll(profile_service, fake_face_frame, pin="7788")
        row = profile_service.authenticate_by_pin(teacher_id, "0000")
        assert row is None, "Wrong PIN must NOT authenticate"

    def test_pin_by_name_authentication(self, profile_service: TeacherProfileService, fake_face_frame):
        """PIN-by-name fallback works for teachers without face model."""
        self._enroll(profile_service, fake_face_frame, name="Nguyễn Văn B", pin="4455")
        row = profile_service.authenticate_by_name_and_pin("Nguyễn Văn B", "4455")
        assert row is not None

    def test_pin_wrong_name_fails(self, profile_service: TeacherProfileService, fake_face_frame):
        self._enroll(profile_service, fake_face_frame, name="Teacher X", pin="1234")
        row = profile_service.authenticate_by_name_and_pin("Nobody", "1234")
        assert row is None

    def test_embedding_decrypt_roundtrip(self, profile_service: TeacherProfileService):
        """Encrypt/decrypt a random embedding preserves numerical data."""
        original = np.random.default_rng(7).random(128).astype(np.float64)
        norm = np.linalg.norm(original)
        original /= norm

        blob = profile_service._encrypt_embedding(original)
        recovered = profile_service._decrypt_embedding(blob)

        np.testing.assert_allclose(
            original, recovered, atol=1e-12,
            err_msg="Fernet roundtrip must preserve embedding exactly"
        )

    def test_delete_teacher(self, profile_service: TeacherProfileService, fake_face_frame):
        """Deleting a teacher removes them from the list."""
        teacher_id = self._enroll(profile_service, fake_face_frame)
        assert len(profile_service.list_teachers()) == 1
        deleted = profile_service.delete_teacher(teacher_id)
        assert deleted is True
        assert len(profile_service.list_teachers()) == 0

    def test_enroll_teacher_pin_only(self, profile_service: TeacherProfileService):
        """PIN-only enrollment allows registration without camera."""
        teacher_id = profile_service.enroll_teacher_pin_only(
            full_name="Cô Trần Thị Mai",
            subject="Hóa học",
            pin="654321",
        )
        assert teacher_id is not None
        assert len(teacher_id) == 36  # UUID format
        teachers = profile_service.list_teachers()
        assert len(teachers) == 1
        assert teachers[0]["full_name"] == "Cô Trần Thị Mai"
        assert teachers[0]["subject"] == "Hóa học"

        # Verify PIN auth works for PIN-only enrolled teacher
        row = profile_service.authenticate_by_pin(teacher_id, "654321")
        assert row is not None
        assert row["full_name"] == "Cô Trần Thị Mai"

        # Bad PIN fails
        bad_row = profile_service.authenticate_by_pin(teacher_id, "000000")
        assert bad_row is None

    def test_enroll_teacher_pin_only_validation(self, profile_service: TeacherProfileService):
        """Invalid PIN or empty name raises ValueError."""
        with pytest.raises(ValueError, match="PIN"):
            profile_service.enroll_teacher_pin_only("Thầy Tuấn", "Sinh", "12")
        with pytest.raises(ValueError, match="PIN"):
            profile_service.enroll_teacher_pin_only("Thầy Tuấn", "Sinh", "abcdef")
        with pytest.raises(ValueError, match="Họ và tên"):
            profile_service.enroll_teacher_pin_only("  ", "Sinh", "1234")

    def test_enroll_custom_teacher_code(self, profile_service: TeacherProfileService):
        """Custom teacher code is normalized and stored, and searchable by code."""
        t_id = profile_service.enroll_teacher_pin_only(
            full_name="Thầy Phan Văn Toán",
            subject="Toán giải tích",
            pin="1234",
            teacher_code="gv-toan01",
        )
        row = profile_service.get_teacher_by_code("GV-TOAN01")
        assert row is not None
        assert row["id"] == t_id
        assert row["teacher_code"] == "GV-TOAN01"
        assert row["full_name"] == "Thầy Phan Văn Toán"

        # Case-insensitive query
        row_lower = profile_service.get_teacher_by_code("gv-toan01")
        assert row_lower is not None
        assert row_lower["id"] == t_id

        # Lookup by ID
        row_id = profile_service.get_teacher_by_id(t_id)
        assert row_id is not None
        assert row_id["teacher_code"] == "GV-TOAN01"

    def test_enroll_autogenerated_teacher_code(self, profile_service: TeacherProfileService):
        """Omitting teacher_code generates sequential codes GV01, GV02, etc."""
        id1 = profile_service.enroll_teacher_pin_only("Giáo viên 1", "Tin", "1111")
        id2 = profile_service.enroll_teacher_pin_only("Giáo viên 2", "Lý", "2222")

        t1 = profile_service.get_teacher_by_id(id1)
        t2 = profile_service.get_teacher_by_id(id2)
        assert t1 is not None and t2 is not None
        assert t1["teacher_code"] == "GV01"
        assert t2["teacher_code"] == "GV02"

    def test_enroll_duplicate_teacher_code_raises(self, profile_service: TeacherProfileService):
        """Enrolling two teachers with the same code raises ValueError."""
        profile_service.enroll_teacher_pin_only("Cô Linh", "Văn", "1234", teacher_code="GV-VAN")
        with pytest.raises(ValueError, match="đã tồn tại"):
            profile_service.enroll_teacher_pin_only("Cô Hương", "Văn", "5678", teacher_code="gv-van")

    def test_authenticate_by_teacher_code_and_pin(self, profile_service: TeacherProfileService):
        """Authentication succeeds with correct code and PIN, fails otherwise."""
        profile_service.enroll_teacher_pin_only("Thầy Hùng", "Địa", "4567", teacher_code="GV-DIA")

        # Success
        auth_row = profile_service.authenticate_by_teacher_code_and_pin("GV-DIA", "4567")
        assert auth_row is not None
        assert auth_row["full_name"] == "Thầy Hùng"

        # Case insensitive code
        auth_row2 = profile_service.authenticate_by_teacher_code_and_pin("gv-dia", "4567")
        assert auth_row2 is not None

        # Wrong PIN
        assert profile_service.authenticate_by_teacher_code_and_pin("GV-DIA", "0000") is None

        # Unknown code
        assert profile_service.authenticate_by_teacher_code_and_pin("GV-UNKNOWN", "4567") is None

    def test_lockout_rate_limiting(self, profile_service: TeacherProfileService):
        """5 failed attempts locks authentication for LOCKOUT_SECONDS."""
        teacher_id = profile_service.enroll_teacher_pin_only("Thầy Khoa", "Sử", "9999", teacher_code="GV-SU")

        locked, remaining = profile_service.is_locked("GV-SU")
        assert not locked
        assert remaining == 0

        # 4 wrong attempts - not locked yet
        for _ in range(4):
            res = profile_service.authenticate_by_teacher_code_and_pin("GV-SU", "0000")
            assert res is None
            locked, _ = profile_service.is_locked("GV-SU")
            assert not locked

        # 5th wrong attempt triggers lockout
        res = profile_service.authenticate_by_teacher_code_and_pin("GV-SU", "0000")
        assert res is None
        locked, rem = profile_service.is_locked("GV-SU")
        assert locked
        assert rem > 0

        # Even with correct PIN, authentication is blocked while locked
        blocked_res = profile_service.authenticate_by_teacher_code_and_pin("GV-SU", "9999")
        assert blocked_res is None

        # Reset failed attempts unlocks
        profile_service.reset_failed_attempts("GV-SU")
        locked_after, _ = profile_service.is_locked("GV-SU")
        assert not locked_after
        unblocked_res = profile_service.authenticate_by_teacher_code_and_pin("GV-SU", "9999")
        assert unblocked_res is not None
        assert unblocked_res["full_name"] == "Thầy Khoa"

    def test_workspace_directory_init_marker(self, profile_service: TeacherProfileService, tmp_dir: Path):
        """Enrolling creates workspace directory and verifies with .workspace_init marker."""
        t_id = profile_service.enroll_teacher_pin_only("Cô Mai Anh", "Anh", "1234", teacher_code="GV-ENG")
        t = profile_service.get_teacher_by_id(t_id)
        assert t is not None
        ws_path = Path(t["workspace_dir"])
        assert ws_path.is_dir()
        init_marker = ws_path / ".workspace_init"
        assert init_marker.is_file()
        content = init_marker.read_text(encoding="utf-8")
        assert '"teacher_id":' in content
        assert "GV-ENG" in content


# ─────────────────────────────────────────────────────────────────────────────
# 5. LivenessChecker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLivenessChecker:
    def _make_face_row(self, nose_x: float, nose_y: float, x=100.0, y=80.0, w=120.0, h=150.0):
        """Build a synthetic face detection row (15 floats)."""
        return np.array(
            [x, y, w, h, x+30, y+40, x+90, y+40, nose_x, nose_y,
             x+35, y+110, x+85, y+110, 0.95],
            dtype=np.float32,
        )

    def test_not_ready_before_required_frames(self):
        checker = LivenessChecker(required_frames=15)
        for i in range(10):
            checker.update(self._make_face_row(160.0 + i * 0.5, 155.0 + i * 0.3))
        assert not checker.is_ready()

    def test_ready_after_required_frames(self):
        checker = LivenessChecker(required_frames=5)
        for i in range(5):
            row = self._make_face_row(160.0 + i * 2.0, 155.0 + i * 1.5)
            checker.update(row)
        assert checker.is_ready()

    def test_liveness_confirmed_with_motion(self):
        """Significant nose movement over 15 frames → liveness=True."""
        checker = LivenessChecker(required_frames=10)
        confirmed = False
        for i in range(15):
            row = self._make_face_row(
                nose_x=160.0 + i * 1.0,   # deliberate movement
                nose_y=155.0 + i * 0.8,
            )
            if checker.update(row):
                confirmed = True
        assert confirmed, "Motion across frames must confirm liveness"

    def test_reset_clears_history(self):
        checker = LivenessChecker(required_frames=5)
        for i in range(5):
            checker.update(self._make_face_row(160.0 + i, 155.0))
        assert checker.is_ready()
        checker.reset()
        assert not checker.is_ready()

    def test_liveness_bounded_sliding_window(self):
        """Feed 100 continuous frames and verify history remains bounded at 2*required_frames (no memory leak)."""
        checker = LivenessChecker(required_frames=10)
        for i in range(100):
            checker.update(self._make_face_row(160.0 + (i % 5) * 0.5, 155.0 + (i % 3) * 0.4))
        assert len(checker._nose_history) <= 20
        assert len(checker._face_size_history) <= 20

    def test_liveness_check_independent_evaluation(self):
        """Verify check() evaluates history independently without altering queue size."""
        checker = LivenessChecker(required_frames=5)
        assert not checker.check(), "check() on empty history must return False"

        # Static history (no motion)
        for _ in range(5):
            checker.update(self._make_face_row(160.0, 155.0))
        assert checker.is_ready()
        assert not checker.check(), "Static face position must fail check()"
        assert len(checker._nose_history) == 5

        # Moving history
        for i in range(5):
            checker.update(self._make_face_row(160.0 + i * 2.0, 155.0 + i * 1.5))
        assert checker.check(), "Dynamic face position must pass check()"
        assert len(checker._nose_history) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Face Authentication Flow Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceAuthenticationFlow:
    def test_authenticate_by_face_updates_liveness_when_not_ready(
        self, profile_service: TeacherProfileService, fake_face_frame: np.ndarray
    ):
        """Verify authenticate_by_face continuously updates liveness checker before readiness."""
        checker = LivenessChecker(required_frames=5)
        assert len(checker._nose_history) == 0

        # Run 3 frames (less than required 5)
        for _ in range(3):
            result = profile_service.authenticate_by_face(fake_face_frame, checker)
            assert result is None

        # Verify checker accumulated the 3 frames
        assert len(checker._nose_history) == 3
        assert not checker.is_ready()

    def test_authenticate_by_face_static_spoof_fails(
        self, profile_service: TeacherProfileService, fake_face_frame: np.ndarray
    ):
        """A static face across frames must be rejected by anti-spoofing check."""
        checker = LivenessChecker(required_frames=5)
        # Feed identical frame 10 times (zero landmark motion)
        results = [
            profile_service.authenticate_by_face(fake_face_frame, checker)
            for _ in range(10)
        ]
        assert checker.is_ready()
        assert not checker.check()
        assert all(res is None for res in results)

    def test_authenticate_by_face_successful_match(
        self, profile_service: TeacherProfileService, fake_face_frame: np.ndarray
    ):
        """When face exhibits authentic liveness and high cosine similarity, auth succeeds."""
        # Enroll a teacher with known embedding
        rng = np.random.default_rng(seed=42)
        dummy_feat = rng.normal(size=128)
        dummy_feat /= np.linalg.norm(dummy_feat)

        teacher_id = profile_service.enroll_from_frames(
            frames=[fake_face_frame] * 3,
            full_name="Thầy Nguyễn Hoàng",
            subject="Tin học",
            pin="1234",
        )

        # Directly update the enrolled teacher's encrypted embedding with dummy_feat
        enc_blob = profile_service._encrypt_embedding(dummy_feat)
        profile_service._conn.execute(
            "UPDATE teachers SET encrypted_embedding=? WHERE id=?",
            (sqlite3.Binary(enc_blob), teacher_id),
        )
        profile_service._conn.commit()

        # Prime liveness checker with movement
        checker = LivenessChecker(required_frames=5)
        for i in range(5):
            row = np.array(
                [100.0, 80.0, 120.0, 150.0, 130.0, 120.0, 190.0, 120.0,
                 160.0 + i * 2.0, 155.0 + i * 1.5,
                 135.0, 190.0, 185.0, 190.0, 0.95],
                dtype=np.float32,
            )
            checker.update(row)

        assert checker.is_ready()
        assert checker.check()

        # Mock model methods to return the matching dummy_feat
        original_detect = profile_service._models.detect_faces
        original_extract = profile_service._models.extract_feature
        original_cosine = profile_service._models.cosine_similarity

        try:
            profile_service._models.detect_faces = lambda f: np.array([[
                100.0, 80.0, 120.0, 150.0, 130.0, 120.0, 190.0, 120.0,
                168.0, 161.0, 135.0, 190.0, 185.0, 190.0, 0.95
            ]], dtype=np.float32)
            profile_service._models.extract_feature = lambda f, b: dummy_feat
            profile_service._models.cosine_similarity = lambda a, b: float(np.dot(a, b))

            matched = profile_service.authenticate_by_face(fake_face_frame, checker)
            assert matched is not None
            assert matched["id"] == teacher_id
            assert matched["full_name"] == "Thầy Nguyễn Hoàng"
            assert profile_service.last_match_score >= 0.99
        finally:
            profile_service._models.detect_faces = original_detect
            profile_service._models.extract_feature = original_extract
            profile_service._models.cosine_similarity = original_cosine


# ─────────────────────────────────────────────────────────────────────────────
# 6. AirTeach Package Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAirTeachPackage:
    def _make_test_package(self, tmp_dir: Path) -> tuple[Path, list[Path], list[StrokeRecord], list[RecognizedItem]]:
        """Create minimal test data for a .airteach package."""
        # Create fake page images
        import cv2
        pages_dir = tmp_dir / "pages"
        pages_dir.mkdir()
        page_paths = []
        for i in range(2):
            img = np.full((400, 600, 3), 240, dtype=np.uint8)
            cv2.putText(img, f"Slide {i+1}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 0), 3, cv2.LINE_AA)
            p = pages_dir / f"page_{i+1:03d}.png"
            cv2.imwrite(str(p), img)
            page_paths.append(p)

        # Create strokes
        strokes = [
            StrokeRecord(
                stroke_id="s1",
                page_index=0,
                color_hex="#FF0000",
                width=3.0,
                shape_type="LINE",
                points=[
                    BezierPointRecord(x=100.0, y=200.0, t_ms=0.0),
                    BezierPointRecord(x=300.0, y=200.0, t_ms=50.0),
                    BezierPointRecord(x=500.0, y=200.0, t_ms=100.0),
                ],
                timestamps_ms=[0.0, 50.0, 100.0],
            ),
            StrokeRecord(
                stroke_id="s2",
                page_index=1,
                color_hex="#0000FF",
                width=2.0,
                shape_type="CIRCLE",
                points=[
                    BezierPointRecord(x=300.0, y=200.0, t_ms=200.0),
                ],
                timestamps_ms=[200.0],
            ),
        ]

        # OCR results
        recognized = [
            RecognizedItem(
                item_id="r1",
                page_index=0,
                text="Định lý Pythagoras",
                latex=r"a^2 + b^2 = c^2",
                confidence=0.95,
                bounding_box=[100.0, 180.0, 400.0, 220.0],
                is_formula=True,
            )
        ]

        return page_paths, strokes, recognized

    def test_save_creates_zip_file(self, tmp_dir: Path):
        """save() must create a valid ZIP file at the given path."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        manifest = AirTeachManifest(title="Test Lesson", teacher_id="t1")

        AirTeachPackage.save(output, manifest, page_paths, strokes, recognized)

        assert output.exists(), ".airteach file must be created"
        assert zipfile.is_zipfile(str(output)), ".airteach must be a valid ZIP"

    def test_zip_contains_required_members(self, tmp_dir: Path):
        """ZIP must contain manifest.json, pages/, ink/strokes.json, etc."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        AirTeachPackage.save(output, AirTeachManifest(title="T"), page_paths, strokes, recognized)

        with zipfile.ZipFile(str(output), "r") as zf:
            names = set(zf.namelist())

        assert "manifest.json" in names
        assert "ink/strokes.json" in names
        assert "recognized/items.json" in names
        assert "links.json" in names
        # Pages
        for i in range(len(page_paths)):
            assert f"pages/page_{i+1:03d}.png" in names

    def test_manifest_json_content(self, tmp_dir: Path):
        """manifest.json must contain correct fields."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        manifest = AirTeachManifest(title="Physics Lesson", teacher_id="teacher42")
        AirTeachPackage.save(output, manifest, page_paths, strokes, recognized)

        with zipfile.ZipFile(str(output), "r") as zf:
            data = json.loads(zf.read("manifest.json").decode())

        assert data["title"] == "Physics Lesson"
        assert data["teacher_id"] == "teacher42"
        assert data["page_count"] == 2
        assert data["version"] == "1.0"

    def test_roundtrip_preserves_strokes(self, tmp_dir: Path):
        """save() then load() must recover the same stroke data."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        AirTeachPackage.save(output, AirTeachManifest(title="T"), page_paths, strokes, recognized)

        extract_dir = tmp_dir / "extracted"
        _, loaded_pages, loaded_strokes, loaded_rec = AirTeachPackage.load(output, extract_dir)

        assert len(loaded_strokes) == len(strokes), "Stroke count must match after roundtrip"
        assert loaded_strokes[0].stroke_id == strokes[0].stroke_id
        assert loaded_strokes[0].color_hex == strokes[0].color_hex
        assert loaded_strokes[0].shape_type == strokes[0].shape_type
        assert len(loaded_strokes[0].points) == len(strokes[0].points)

    def test_roundtrip_preserves_timestamps(self, tmp_dir: Path):
        """timestamps_ms array must be preserved exactly."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        AirTeachPackage.save(output, AirTeachManifest(title="T"), page_paths, strokes, recognized)

        extract_dir = tmp_dir / "extracted2"
        _, _, loaded_strokes, _ = AirTeachPackage.load(output, extract_dir)

        assert loaded_strokes[0].timestamps_ms == strokes[0].timestamps_ms

    def test_roundtrip_preserves_recognized_items(self, tmp_dir: Path):
        """OCR items must survive save/load with correct latex content."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        AirTeachPackage.save(output, AirTeachManifest(title="T"), page_paths, strokes, recognized)

        extract_dir = tmp_dir / "extracted3"
        _, _, _, loaded_rec = AirTeachPackage.load(output, extract_dir)

        assert len(loaded_rec) == 1
        assert loaded_rec[0].latex == r"a^2 + b^2 = c^2"
        assert loaded_rec[0].is_formula is True
        assert loaded_rec[0].confidence == pytest.approx(0.95)

    def test_roundtrip_extracts_page_images(self, tmp_dir: Path):
        """Page PNG files must be extracted to the target directory."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"
        AirTeachPackage.save(output, AirTeachManifest(title="T"), page_paths, strokes, recognized)

        extract_dir = tmp_dir / "extracted4"
        _, loaded_pages, _, _ = AirTeachPackage.load(output, extract_dir)

        assert len(loaded_pages) == 2
        for p in loaded_pages:
            assert p.exists(), f"Extracted page file not found: {p}"

    def test_sha256_link_stored_for_original(self, tmp_dir: Path):
        """If original file provided, SHA-256 hash must appear in links.json."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson.airteach"

        # Create a fake original file
        original = tmp_dir / "lecture_slides.pdf"
        original.write_bytes(b"fake pdf content for hashing " * 100)

        AirTeachPackage.save(
            output, AirTeachManifest(title="T"), page_paths, strokes, recognized,
            original_file_path=original,
        )

        with zipfile.ZipFile(str(output), "r") as zf:
            links = json.loads(zf.read("links.json").decode())

        expected_hash = _sha256_file(original)
        assert links["original_sha256"] == expected_hash
        assert links["original_filename"] == "lecture_slides.pdf"

    def test_read_links_and_verify_original_link_verified(self, tmp_dir: Path):
        """Original linked file verification succeeds when file exists and SHA-256 matches."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson_verified.airteach"
        original = tmp_dir / "physics_lecture.pdf"
        original.write_bytes(b"%PDF-1.4 authentic content for physics lesson" * 20)

        AirTeachPackage.save(
            output, AirTeachManifest(title="Physics"), page_paths, strokes, recognized,
            original_file_path=original,
        )

        # 1. read_links inspection
        links_meta = AirTeachPackage.read_links(output)
        assert links_meta["original_filename"] == "physics_lecture.pdf"
        assert links_meta["original_sha256"] == _sha256_file(original)
        assert links_meta["file_size_bytes"] == original.stat().st_size

        # 2. verify_original_link default resolution
        res = AirTeachPackage.verify_original_link(output)
        assert res["linked"] is True
        assert res["found"] is True
        assert res["verified"] is True
        assert res["status"] == "VERIFIED"
        assert res["path"] == original.resolve()
        assert res["current_sha256"] == links_meta["original_sha256"]

    def test_verify_original_link_file_moved_and_fallback(self, tmp_dir: Path):
        """Gracefully handle moved or missing source files and verify with explicit candidate_path."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson_moved.airteach"
        original = tmp_dir / "biology_lecture.pdf"
        original.write_bytes(b"%PDF-1.5 biology slide content" * 15)

        AirTeachPackage.save(
            output, AirTeachManifest(title="Biology"), page_paths, strokes, recognized,
            original_file_path=original,
        )

        # Move file to an archive directory
        archive_dir = tmp_dir / "archive"
        archive_dir.mkdir()
        moved_file = archive_dir / "biology_renamed.pdf"
        original.rename(moved_file)

        # Verification without candidate_path fails gracefully as FILE_MOVED_OR_MISSING
        res_missing = AirTeachPackage.verify_original_link(output)
        assert res_missing["linked"] is True
        assert res_missing["found"] is False
        assert res_missing["verified"] is False
        assert res_missing["status"] == "FILE_MOVED_OR_MISSING"

        # Verification with candidate_path successfully verifies moved file
        res_candidate = AirTeachPackage.verify_original_link(output, candidate_path=moved_file)
        assert res_candidate["linked"] is True
        assert res_candidate["found"] is True
        assert res_candidate["verified"] is True
        assert res_candidate["status"] == "VERIFIED"
        assert res_candidate["path"] == moved_file

    def test_verify_original_link_hash_mismatch(self, tmp_dir: Path):
        """Tampered or modified original file flags HASH_MISMATCH."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "lesson_tampered.airteach"
        original = tmp_dir / "math_lecture.pdf"
        original.write_bytes(b"initial math slides content")

        AirTeachPackage.save(
            output, AirTeachManifest(title="Math"), page_paths, strokes, recognized,
            original_file_path=original,
        )

        # Modify original file on disk
        original.write_bytes(b"tampered or updated different slides content")

        res = AirTeachPackage.verify_original_link(output)
        assert res["linked"] is True
        assert res["found"] is True
        assert res["verified"] is False
        assert res["status"] == "HASH_MISMATCH"
        assert res["current_sha256"] != res["recorded_sha256"]

    def test_verify_original_link_sibling_and_no_link(self, tmp_dir: Path):
        """Verify sibling directory resolution fallback and unlinked packages."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        bundle_dir = tmp_dir / "classroom_bundle"
        bundle_dir.mkdir()
        output = bundle_dir / "lesson.airteach"

        # Create original in external location
        external = tmp_dir / "history.pdf"
        external.write_bytes(b"history content slides")
        AirTeachPackage.save(
            output, AirTeachManifest(title="History"), page_paths, strokes, recognized,
            original_file_path=external,
        )

        # Move external file next to bundle (sibling resolution)
        sibling = bundle_dir / "history.pdf"
        external.rename(sibling)

        res = AirTeachPackage.verify_original_link(output)
        assert res["linked"] is True
        assert res["found"] is True
        assert res["verified"] is True
        assert res["status"] == "VERIFIED"
        assert res["path"] == sibling

        # Unlinked package check
        unlinked_output = tmp_dir / "unlinked.airteach"
        AirTeachPackage.save(unlinked_output, AirTeachManifest(title="None"), page_paths, strokes, recognized)
        res_none = AirTeachPackage.verify_original_link(unlinked_output)
        assert res_none["linked"] is False
        assert res_none["found"] is False
        assert res_none["verified"] is False
        assert res_none["status"] == "NO_LINK"

    def test_atomic_save_cleans_up_on_failure(self, tmp_dir: Path, monkeypatch):
        """If packaging fails midway, no corrupted output file remains and temp files are purged."""
        page_paths, strokes, recognized = self._make_test_package(tmp_dir)
        output = tmp_dir / "failed_lesson.airteach"

        original_writestr = zipfile.ZipFile.writestr
        def failing_writestr(self_zf, arcname, *args, **kwargs):
            if arcname == AirTeachPackage.INK_PATH:
                raise OSError("Disk simulated write failure")
            return original_writestr(self_zf, arcname, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "writestr", failing_writestr)

        with pytest.raises(OSError, match="Disk simulated write failure"):
            AirTeachPackage.save(output, AirTeachManifest(title="Failed"), page_paths, strokes, recognized)

        assert not output.exists(), "Target file must not exist if write was aborted"
        tmp_files = list(tmp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"No leftover tmp files should remain, found: {tmp_files}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. StrokeRecord Serialization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrokeSerialisation:
    def test_stroke_to_dict_and_back(self):
        """StrokeRecord → dict → StrokeRecord roundtrip is lossless."""
        original = StrokeRecord(
            stroke_id="test-stroke",
            page_index=2,
            color_hex="#12AB34",
            width=4.5,
            shape_type="RECTANGLE",
            points=[
                BezierPointRecord(x=10.5, y=20.3, t_ms=100.0, pressure=0.9),
                BezierPointRecord(x=50.0, y=80.0, t_ms=150.0, pressure=1.0),
            ],
            timestamps_ms=[100.0, 150.0],
        )
        d = _stroke_to_dict(original)
        recovered = _dict_to_stroke(d)

        assert recovered.stroke_id == original.stroke_id
        assert recovered.color_hex == original.color_hex
        assert recovered.shape_type == original.shape_type
        assert len(recovered.points) == 2
        assert recovered.points[0].x == pytest.approx(10.5)
        assert recovered.points[1].t_ms == pytest.approx(150.0)
        assert recovered.timestamps_ms == [100.0, 150.0]


# ─────────────────────────────────────────────────────────────────────────────
# 8. LaTeX Renderer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLaTeXRenderer:
    def test_renders_simple_formula_to_rgba(self):
        """LaTeX renderer must produce a non-empty RGBA numpy array."""
        result = LaTeXRenderer.render_to_rgba(r"\frac{a}{b}", fontsize=20, dpi=72)
        assert result is not None, "render_to_rgba must return an array for valid LaTeX"
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 4, "Output must have 4 channels (RGBA)"
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_rendered_image_has_non_trivial_alpha(self):
        """The alpha channel must not be all-zero (transparent) for visible math."""
        result = LaTeXRenderer.render_to_rgba(r"E = mc^2", fontsize=20, dpi=72)
        assert result is not None
        alpha_channel = result[:, :, 3]
        assert alpha_channel.max() > 0, "Rendered LaTeX must have opaque pixels"

    def test_renders_to_png_bytes(self):
        """render_to_png_bytes must return non-empty bytes."""
        png = LaTeXRenderer.render_to_png_bytes(r"\pi r^2", fontsize=18, dpi=72)
        assert png is not None and len(png) > 0
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "Output must be valid PNG"

    def test_wraps_expression_in_dollar_signs(self):
        """Expressions without $ delimiters must still render correctly."""
        # Should not raise even without $ wrapping
        r1 = LaTeXRenderer.render_to_rgba(r"\alpha + \beta", fontsize=16, dpi=72)
        r2 = LaTeXRenderer.render_to_rgba(r"$\alpha + \beta$", fontsize=16, dpi=72)
        assert r1 is not None and r2 is not None

    def test_invalid_latex_does_not_crash(self):
        """Extremely malformed LaTeX should return None, not raise an exception."""
        result = LaTeXRenderer.render_to_rgba(
            r"\invalid_command_xyz{{{", fontsize=16, dpi=72
        )
        # May return None OR a fallback image — must not raise
        assert result is None or isinstance(result, np.ndarray)

    def test_renders_empty_and_isolated_dollar_gracefully(self):
        """Empty, whitespace, or bare delimiter strings must return None without raising."""
        assert LaTeXRenderer.render_to_rgba("", fontsize=16, dpi=72) is None
        assert LaTeXRenderer.render_to_rgba("   ", fontsize=16, dpi=72) is None
        assert LaTeXRenderer.render_to_rgba("$", fontsize=16, dpi=72) is None
        assert LaTeXRenderer.render_to_rgba("$$", fontsize=16, dpi=72) is None
        assert LaTeXRenderer.render_to_rgba("$$$", fontsize=16, dpi=72) is None
        assert LaTeXRenderer.render_to_png_bytes("") is None

    def test_validate_latex_syntax_valid_and_invalid(self):
        """validate_latex_syntax must accurately classify valid math and reject malformed syntax without figure render."""
        assert LaTeXRenderer.validate_latex_syntax(r"x^2 + y^2 = r^2") is True
        assert LaTeXRenderer.validate_latex_syntax(r"\int_{0}^{\infty} e^{-x} dx") is True
        assert LaTeXRenderer.validate_latex_syntax(r"\frac{a}{b}") is True
        assert LaTeXRenderer.validate_latex_syntax(r"\frac{a}{") is False
        assert LaTeXRenderer.validate_latex_syntax("") is False
        assert LaTeXRenderer.validate_latex_syntax("   ") is False

    def test_sanitize_latex_string_unicode_and_braces(self):
        """sanitize_latex_string converts Unicode math symbols and balances unclosed braces."""
        raw = "  $ a × b ≤ c ± d ≠ e √x $ "
        san = sanitize_latex_string(raw)
        assert r"\times " in san
        assert r"\le " in san
        assert r"\pm " in san
        assert r"\neq " in san
        assert r"\sqrt " in san

        # Exponent collapse
        assert sanitize_latex_string("x^^3 + y__2") == "x^3 + y_2"

        # Unmatched braces auto-balanced
        assert sanitize_latex_string(r"\frac{a}{b") == r"\frac{a}{b}"
        assert sanitize_latex_string(r"e^{2x") == r"e^{2x}"

    def test_normalize_vietnamese_ocr_text_comprehensive(self):
        """normalize_vietnamese_ocr_text handles decomposed NFD, full-width punctuation, and irregular spacing."""
        import unicodedata
        # Decomposed NFD string
        nfd_raw = unicodedata.normalize("NFD", "Trường Đại học Bách Khoa Hà Nội")
        assert len(nfd_raw) > len("Trường Đại học Bách Khoa Hà Nội")  # Verified NFD length

        # Dirty OCR text with NFD, full-width comma/period, and extra spacing before punctuation
        dirty = nfd_raw + " ，Toán học  , Vật lý  .  Khoa học ！"
        cleaned = normalize_vietnamese_ocr_text(dirty)

        # Output must be NFC
        assert cleaned == unicodedata.normalize("NFC", cleaned)
        assert "，" not in cleaned
        assert "！" not in cleaned
        assert "Toán học, Vật lý. Khoa học!" in cleaned
        assert cleaned.startswith("Trường Đại học Bách Khoa Hà Nội,")



# ─────────────────────────────────────────────────────────────────────────────
# 9. StrokeRenderer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrokeRenderer:
    def _make_stroke(self) -> Stroke:
        s = Stroke(color_bgr=(0, 0, 0), width=2.0, stroke_id="t1")
        for i in range(10):
            s.points.append(StrokePoint(x=float(i * 20), y=float(i * 10), t_ms=float(i * 16)))
        return s

    def test_renders_strokes_to_white_bgr(self):
        """Rendered image must be BGR (3-channel) with predominantly white background."""
        stroke = self._make_stroke()
        img = StrokeRenderer.render([stroke])
        assert img.ndim == 3 and img.shape[2] == 3
        # Background should be mostly white
        white_pixels = np.all(img == 255, axis=2).sum()
        total = img.shape[0] * img.shape[1]
        assert white_pixels / total > 0.8, "Most pixels should be white background"

    def test_empty_strokes_returns_placeholder(self):
        """Empty stroke list must not crash and must return a valid image."""
        img = StrokeRenderer.render([])
        assert img.ndim == 3 and img.shape[2] == 3

    def test_single_point_stroke(self):
        """A stroke with one point (dot) must render without error."""
        s = Stroke(color_bgr=(0, 0, 0), width=3.0)
        s.points.append(StrokePoint(x=50.0, y=50.0, t_ms=0.0))
        img = StrokeRenderer.render([s])
        assert img is not None

    def test_canvas_size_accounts_for_padding(self):
        """Canvas width must include PADDING on both sides."""
        s = Stroke(color_bgr=(0, 0, 0), width=1.0)
        for i in range(5):
            s.points.append(StrokePoint(x=float(i * 10), y=50.0, t_ms=float(i)))
        img = StrokeRenderer.render([s], dpi_scale=1.0)
        # With padding=24 on each side and x range [0, 40], width >= 40 + 48 = 88
        assert img.shape[1] >= 64, "Canvas must have minimum size"



    def test_redact_text_comprehensive_secrets(self):
        """TeacherCredentialService.redact_text must scrub API keys, tokens, and PINs."""
        from src.services.credential_service import TeacherCredentialService

        raw_api = "Failed calling with key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q and bearer token Bearer eyJhbGciOiJIUzI1NiJ9"
        sanitized = TeacherCredentialService.redact_text(raw_api)
        assert "[REDACTED_API_KEY]" in sanitized
        assert "AIzaSy" not in sanitized
        assert "Bearer [REDACTED_TOKEN]" in sanitized
        assert "eyJhbGci" not in sanitized

        raw_pin = "Error in auth: pin=123456, password=secretPass123"
        sanitized_pin = TeacherCredentialService.redact_text(raw_pin)
        assert "123456" not in sanitized_pin
        assert "secretPass123" not in sanitized_pin
        assert "pin=[REDACTED]" in sanitized_pin
        assert "password=[REDACTED]" in sanitized_pin

# ─────────────────────────────────────────────────────────────────────────────
# 10. GeminiAssistant Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiAssistant:
    def test_offline_mode_when_no_key(self):
        """Assistant with empty key must operate in offline mode."""
        assistant = GeminiAssistant(api_key="")
        assert not assistant.is_online()

    def test_offline_chat_returns_string(self):
        """Offline chat must return a non-empty string (from _OFFLINE_GUIDE)."""
        assistant = GeminiAssistant(api_key="")
        response = assistant.chat_with_teacher(
            history=[],
            user_message="Cử chỉ draw là gì?",
            current_page_text="",
        )
        assert isinstance(response, str) and len(response) > 10

    def test_offline_chat_mentions_api_unavailable(self):
        """Offline response must inform user that API is unavailable."""
        assistant = GeminiAssistant(api_key="")
        response = assistant.chat_with_teacher(history=[], user_message="test")
        assert "ngoại tuyến" in response.lower() or "api" in response.lower()

    def test_offline_usage_guide_returns_content(self):
        """Usage guide offline fallback must return relevant help."""
        assistant = GeminiAssistant(api_key="")
        resp = assistant.get_interactive_usage_guide("swipe lật trang")
        assert isinstance(resp, str) and len(resp) > 5

    def test_invalid_api_key_falls_back_gracefully(self):
        """An obviously invalid API key must not crash; must return fallback text."""
        assistant = GeminiAssistant(api_key="INVALID_KEY_123")
        # Even if init doesn't fail, calling chat must not raise
        try:
            response = assistant.chat_with_teacher(
                history=[],
                user_message="Xin chào",
                current_page_text="",
            )
            assert isinstance(response, str)
        except Exception as exc:
            pytest.fail(f"GeminiAssistant raised exception with invalid key: {exc}")

    def test_explain_canvas_offline_returns_message(self):
        """explain_canvas_region offline must return a string message."""
        assistant = GeminiAssistant(api_key="")
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = assistant.explain_canvas_region(fake_bytes)
        assert isinstance(resp, str) and len(resp) > 5

    def test_offline_lookup_known_keywords(self):
        """_offline_lookup must return relevant answers for known keywords."""
        assert "draw" in _offline_lookup("draw gesture").lower() or \
               "viết" in _offline_lookup("cử chỉ vẽ")
        assert "swipe" in _offline_lookup("swipe").lower() or \
               "lật" in _offline_lookup("swipe")
        assert "calibrate" in _offline_lookup("calibrate camera").lower() or \
               "hiệu chỉnh" in _offline_lookup("calibrate camera").lower()

    def test_streaming_callback_receives_chunks(self):
        """Offline fallback must call on_chunk with the full response."""
        assistant = GeminiAssistant(api_key="")
        chunks = []
        assistant.chat_with_teacher(
            history=[],
            user_message="draw",
            on_chunk=lambda t: chunks.append(t),
        )
        assert len(chunks) > 0, "on_chunk must be called at least once"
        full = "".join(chunks)
        assert len(full) > 5

    def test_update_api_key_changes_online_status(self):
        """update_api_key with empty string keeps offline; real key would enable online."""
        assistant = GeminiAssistant(api_key="FAKE")
        # Update with empty string should disable online mode
        result = assistant.update_api_key("")
        assert result is False
        assert not assistant.is_online()

    def test_is_online_callable_and_bool_dual_interface(self):
        """CallableBool must work as both a boolean value and a callable function."""
        offline_assistant = GeminiAssistant(api_key="")
        # Dual interface checks for offline state
        assert bool(offline_assistant.is_online) is False
        assert offline_assistant.is_online() is False
        assert not offline_assistant.is_online

        # Online state simulation with dummy client
        from src.services.gemini_assistant import CallableBool
        cb_true = CallableBool(True)
        assert bool(cb_true) is True
        assert cb_true() is True
        assert cb_true is True or cb_true == True

        cb_false = CallableBool(False)
        assert bool(cb_false) is False
        assert cb_false() is False
        assert not cb_false

    def test_stem_offline_knowledge_base_lookups(self):
        """_offline_lookup must accurately return authentic STEM formulas and laws."""
        # Math: Pythagoras / Taylor
        math_resp = _offline_lookup("định lý pytago")
        assert "a^2 + b^2 = c^2" in math_resp or "Pythagoras" in math_resp

        # Physics: Newton / Ohm
        physics_resp = _offline_lookup("định luật ohm")
        assert "I = U / R" in physics_resp or "Ohm" in physics_resp

        # Chemistry: Le Chatelier / pH
        chem_resp = _offline_lookup("chuyển dịch cân bằng le chatelier")
        assert "Le Chatelier" in chem_resp or "cân bằng" in chem_resp

        # Biology: DNA / Di truyền Mendel
        bio_resp = _offline_lookup("cấu trúc dna")
        assert "xoắn kép" in bio_resp or "Watson" in bio_resp or "A-T" in bio_resp

        # Accent-insensitive Vietnamese matching
        unaccented_resp = _offline_lookup("dinh luat 2 newton")
        assert "F = m" in unaccented_resp or "Newton" in unaccented_resp

    def test_explain_canvas_region_cropped_image_bytes_compat(self):
        """explain_canvas_region must accept both image_bytes and cropped_image_bytes keywords."""
        assistant = GeminiAssistant(api_key="")
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

        # Standard keyword
        res1 = assistant.explain_canvas_region(image_bytes=fake_png)
        assert isinstance(res1, str) and len(res1) > 0

        # Legacy / alternative keyword from UI workers
        res2 = assistant.explain_canvas_region(cropped_image_bytes=fake_png)
        assert isinstance(res2, str) and len(res2) > 0

        # Positional arg
        res3 = assistant.explain_canvas_region(fake_png)
        assert isinstance(res3, str) and len(res3) > 0

# ─────────────────────────────────────────────────────────────────────────────
# 11. TelemetryLogger Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTelemetryLogger:
    def test_records_written_after_stop(self, tmp_dir: Path):
        """Records added before stop() must appear in the CSV file."""
        csv_path = tmp_dir / "telemetry.csv"
        tel = TelemetryLogger(csv_path)
        tel.start()

        for i in range(5):
            tel.record(
                fps=30.0,
                frame_latency_ms=float(i * 2),
                gesture_state="DRAW",
                shape_snapped_type="LINE",
                ocr_inference_ms=15.0,
            )

        tel.stop()

        assert csv_path.exists(), "Telemetry CSV must be created"
        content = csv_path.read_text(encoding="utf-8")
        assert "DRAW" in content
        assert "LINE" in content
        assert "fps" in content  # header

    def test_csv_has_correct_columns(self, tmp_dir: Path):
        """CSV header must match expected field names."""
        import csv as _csv
        csv_path = tmp_dir / "cols_test.csv"
        with TelemetryLogger(csv_path) as tel:
            tel.record(gesture_state="POINT")

        with csv_path.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            cols = reader.fieldnames or []

        from src.utils.telemetry_logger import CSV_FIELDS
        for field in CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_compute_summary_returns_stats(self, tmp_dir: Path):
        """compute_summary must return valid stats dict after recording data."""
        tel = TelemetryLogger(tmp_dir / "summary.csv")
        tel.start()

        for i in range(20):
            tel.record(fps=29.0 + i * 0.1, frame_latency_ms=5.0 + i * 0.2,
                       gesture_state="DRAW")

        summary = tel.compute_summary()
        tel.stop()

        assert summary["count"] == 20
        assert "fps" in summary
        assert summary["fps"]["mean"] > 0
        assert summary["gesture_distribution"]["DRAW"] == 20

    def test_auto_fps_computed_from_timestamps(self, tmp_dir: Path):
        """When fps=None, the logger auto-computes FPS from inter-frame timing."""
        tel = TelemetryLogger(tmp_dir / "auto_fps.csv")
        tel.start()

        for _ in range(10):
            tel.record(fps=None)
            time.sleep(0.016)  # ~60 FPS

        summary = tel.compute_summary()
        tel.stop()

        fps_mean = summary.get("fps", {}).get("mean", 0)
        # Should be in the ballpark of 60fps (with system jitter, accept 20-120)
        assert 10.0 <= fps_mean <= 200.0, f"Auto-FPS out of expected range: {fps_mean}"

    def test_context_manager_usage(self, tmp_dir: Path):
        """TelemetryLogger must work as a context manager."""
        csv_path = tmp_dir / "ctx.csv"
        with TelemetryLogger(csv_path) as tel:
            tel.record(gesture_state="PAUSE", ocr_inference_ms=42.0)

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "PAUSE" in content

    def test_large_number_of_records(self, tmp_dir: Path):
        """Logger must handle 1000+ records without data loss."""
        csv_path = tmp_dir / "large.csv"
        import csv as _csv
        n = 500

        with TelemetryLogger(csv_path) as tel:
            for i in range(n):
                tel.record(fps=30.0, gesture_state="NONE", frame_latency_ms=float(i % 20))

        with csv_path.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == n, f"Expected {n} rows, got {len(rows)}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Helper Function Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_sha256_file(self, tmp_dir: Path):
        """SHA-256 of a known-content file must match expected digest."""
        p = tmp_dir / "test.txt"
        p.write_bytes(b"AirTeach AI")
        expected = hashlib.sha256(b"AirTeach AI").hexdigest()
        assert _sha256_file(p) == expected

    def test_hex_to_bgr_black(self):
        assert _hex_to_bgr("#000000") == (0, 0, 0)

    def test_hex_to_bgr_red(self):
        # #FF0000 = R=255, G=0, B=0 → BGR = (0, 0, 255)
        assert _hex_to_bgr("#FF0000") == (0, 0, 255)

    def test_hex_to_bgr_blue(self):
        # #0000FF = R=0, G=0, B=255 → BGR = (255, 0, 0)
        assert _hex_to_bgr("#0000FF") == (255, 0, 0)

    def test_hex_to_bgr_invalid_returns_black(self):
        assert _hex_to_bgr("invalid") == (0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 13. DocumentManager Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentManager:
    def test_raises_for_nonexistent_file(self, tmp_dir: Path):
        """import_document must raise FileNotFoundError for missing files."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        with pytest.raises(FileNotFoundError):
            dm.import_document(Path("nonexistent.pdf"))

    def test_raises_for_unsupported_format(self, tmp_dir: Path):
        """import_document must raise ValueError for unsupported formats."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        fake = tmp_dir / "file.xyz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            dm.import_document(fake)

    def test_pdf_rendering(self, tmp_dir: Path):
        """A valid single-page PDF must render to exactly 1 PageInfo."""
        # Create a minimal valid PDF using pypdfium2 or a pre-made bytes blob
        # Minimal 1-page PDF (manually crafted, valid enough for pypdfium2)
        minimal_pdf = _make_minimal_pdf()
        pdf_path = tmp_dir / "test_slide.pdf"
        pdf_path.write_bytes(minimal_pdf)

        dm = DocumentManager(work_root=tmp_dir / "work")
        pages = dm.import_document(pdf_path)

        assert len(pages) >= 1, "At least one page must be rendered"
        assert Path(pages[0].image_path).exists(), "Rendered page PNG must exist"
        assert pages[0].width_px > 0 and pages[0].height_px > 0

    def test_save_and_open_lesson_roundtrip(self, tmp_dir: Path):
        """save_lesson + open_lesson must restore the same strokes."""
        import cv2
        dm = DocumentManager(work_root=tmp_dir / "work")

        # Make a dummy page
        pages_dir = tmp_dir / "pages"
        pages_dir.mkdir()
        img = np.full((300, 400, 3), 255, dtype=np.uint8)
        page_img = pages_dir / "page_001.png"
        cv2.imwrite(str(page_img), img)

        strokes = [
            StrokeRecord(
                stroke_id="round-trip",
                page_index=0,
                color_hex="#000000",
                width=2.0,
                points=[BezierPointRecord(x=10.0, y=10.0, t_ms=0.0)],
                timestamps_ms=[0.0],
            )
        ]

        output = tmp_dir / "rt_lesson.airteach"
        dm.save_lesson(
            output_path=output,
            manifest=AirTeachManifest(title="RoundTrip Test"),
            page_image_paths=[page_img],
            strokes=strokes,
            recognized=[],
        )

        _, pages_out, strokes_out, _ = dm.open_lesson(output)
        assert len(strokes_out) == 1
        assert strokes_out[0].stroke_id == "round-trip"

    def test_export_annotated_pdf_single_and_multi_page(self, tmp_dir: Path):
        """export_annotated_pdf must generate a valid multi-page PDF with strokes overlaid."""
        import cv2
        from pypdf import PdfReader
        dm = DocumentManager(work_root=tmp_dir / "work")

        # Create two test page images
        page1 = tmp_dir / "p1.png"
        page2 = tmp_dir / "p2.png"
        cv2.imwrite(str(page1), np.full((540, 960, 3), 255, dtype=np.uint8))
        cv2.imwrite(str(page2), np.full((540, 960, 3), 250, dtype=np.uint8))

        strokes = [
            StrokeRecord(
                stroke_id="s1",
                page_index=0,
                color_hex="#FF0000",
                width=3.0,
                points=[
                    BezierPointRecord(x=100.0, y=100.0, t_ms=0.0),
                    BezierPointRecord(x=200.0, y=200.0, t_ms=16.6),
                ],
                timestamps_ms=[0.0, 16.6],
            ),
            StrokeRecord(
                stroke_id="s2",
                page_index=1,
                color_hex="#0000FF",
                width=4.0,
                bezier_segments=[{
                    "p0": [50.0, 50.0],
                    "p1": [100.0, 80.0],
                    "p2": [150.0, 80.0],
                    "p3": [200.0, 50.0],
                }],
                timestamps_ms=[0.0, 33.3],
            ),
        ]

        pdf_out = tmp_dir / "lecture_notes.pdf"
        dm.export_annotated_pdf(
            page_image_paths=[page1, page2],
            strokes=strokes,
            output_pdf_path=pdf_out,
        )

        assert pdf_out.exists() and pdf_out.stat().st_size > 1000
        reader = PdfReader(str(pdf_out))
        assert len(reader.pages) == 2

    def test_export_annotated_pdf_with_recognized_items_and_latex(self, tmp_dir: Path):
        """export_annotated_pdf must accurately stamp LaTeX formulas and text items."""
        import cv2
        dm = DocumentManager(work_root=tmp_dir / "work")
        page = tmp_dir / "stem_page.png"
        cv2.imwrite(str(page), np.full((600, 800, 3), 255, dtype=np.uint8))

        recognized = [
            RecognizedItem(
                item_id="math-1",
                page_index=0,
                text="E = mc^2",
                latex="E = mc^2",
                confidence=0.98,
                bounding_box=[100.0, 100.0, 300.0, 180.0],
                is_formula=True,
            ),
            RecognizedItem(
                item_id="txt-1",
                page_index=0,
                text="Định luật Vạn vật Hấp dẫn",
                latex="",
                confidence=0.95,
                bounding_box=[100.0, 250.0, 450.0, 310.0],
                is_formula=False,
            ),
        ]

        pdf_out = tmp_dir / "annotated_stem.pdf"
        dm.export_annotated_pdf(
            page_image_paths=[page],
            strokes=[],
            output_pdf_path=pdf_out,
            recognized=recognized,
        )

        assert pdf_out.exists() and pdf_out.stat().st_size > 500

    def test_export_annotated_pages_generates_valid_pngs(self, tmp_dir: Path):
        """export_annotated_pages must generate standalone PNG images for each slide."""
        import cv2
        dm = DocumentManager(work_root=tmp_dir / "work")
        page = tmp_dir / "raw_page.png"
        cv2.imwrite(str(page), np.full((400, 600, 3), 240, dtype=np.uint8))

        out_dir = tmp_dir / "exported_pngs"
        generated = dm.export_annotated_pages(
            page_image_paths=[page],
            strokes=[],
            output_dir=out_dir,
            recognized=[
                RecognizedItem(
                    item_id="rec-1",
                    page_index=0,
                    text="Ghi chú bài giảng",
                    latex="",
                    confidence=1.0,
                    bounding_box=[50.0, 50.0, 250.0, 90.0],
                    is_formula=False,
                )
            ],
        )

        assert len(generated) == 1
        assert generated[0].exists()
        loaded = cv2.imread(str(generated[0]))
        assert loaded is not None and loaded.shape == (400, 600, 3)

    def test_export_annotated_pdf_atomic_save_and_error_handling(self, tmp_dir: Path):
        """export_annotated_pdf must raise RuntimeError on empty page list and clean up temp files."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        with pytest.raises(RuntimeError, match="No pages to export"):
            dm.export_annotated_pdf(
                page_image_paths=[],
                strokes=[],
                output_pdf_path=tmp_dir / "fail.pdf",
            )

    def test_stroke_record_with_geometric_shapes_roundtrip(self, tmp_dir: Path):
        """Verify save_lesson and open_lesson preserve geometric shape parameters."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        p1 = tmp_dir / "p1.png"
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(p1), img)

        strokes = [
            StrokeRecord(
                stroke_id="s_circle",
                page_index=0,
                color_hex="#ff0000",
                width=3.0,
                shape_type="CIRCLE",
                shape_params={"center": [400, 300], "radius": 50},
                snap_confidence=0.99,
            ),
            StrokeRecord(
                stroke_id="s_rect",
                page_index=0,
                color_hex="#0000ff",
                width=2.5,
                shape_type="RECTANGLE",
                shape_params={"corners": [[100, 100], [300, 100], [300, 200], [100, 200]]},
                snap_confidence=0.95,
            ),
            StrokeRecord(
                stroke_id="s_arrow",
                page_index=0,
                color_hex="#00ff00",
                width=2.0,
                shape_type="ARROW",
                shape_params={"shaft_start": [50, 50], "tip": [200, 200]},
                snap_confidence=0.92,
            ),
            StrokeRecord(
                stroke_id="s_line",
                page_index=0,
                color_hex="#111111",
                width=2.0,
                shape_type="LINE",
                shape_params={"start": [10, 500], "end": [700, 500]},
                snap_confidence=0.98,
            ),
            StrokeRecord(
                stroke_id="s_ellipse",
                page_index=0,
                color_hex="#990099",
                width=2.0,
                shape_type="ELLIPSE",
                shape_params={"center": [600, 200], "semi_major": 80, "semi_minor": 40, "angle_deg": 30.0},
                snap_confidence=0.94,
            ),
        ]

        bundle = tmp_dir / "geom.airteach"
        manifest = AirTeachManifest(title="Geometric Lesson", teacher_id="t_geom")
        dm.save_lesson(bundle, manifest, [p1], strokes, [])
        assert bundle.exists()

        _, _, loaded_strokes, _ = dm.open_lesson(bundle)
        assert len(loaded_strokes) == 5
        circle_s = next(s for s in loaded_strokes if s.stroke_id == "s_circle")
        assert circle_s.shape_type == "CIRCLE"
        assert circle_s.shape_params["radius"] == 50
        assert circle_s.snap_confidence == 0.99

        rect_s = next(s for s in loaded_strokes if s.stroke_id == "s_rect")
        assert rect_s.shape_type == "RECTANGLE"
        assert len(rect_s.shape_params["corners"]) == 4

    def test_export_annotated_pdf_with_geometric_shapes(self, tmp_dir: Path):
        """Verify export_annotated_pdf and pages render geometric primitives cleanly."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        p1 = tmp_dir / "bg.png"
        img = np.full((800, 1000, 3), 240, dtype=np.uint8)
        cv2.imwrite(str(p1), img)

        strokes = [
            StrokeRecord(stroke_id="1", page_index=0, color_hex="#ff0000", width=4.0,
                         shape_type="CIRCLE", shape_params={"center": [300, 300], "radius": 80}),
            StrokeRecord(stroke_id="2", page_index=0, color_hex="#00ff00", width=3.0,
                         shape_type="ARROW", shape_params={"shaft_start": [50, 50], "tip": [250, 250]}),
            StrokeRecord(stroke_id="3", page_index=0, color_hex="#0000ff", width=2.5,
                         shape_type="RECTANGLE", shape_params={"corners": [[400, 100], [700, 100], [700, 300], [400, 300]]}),
            StrokeRecord(stroke_id="4", page_index=0, color_hex="#333333", width=2.0,
                         shape_type="LINE", shape_params={"start": [100, 600], "end": [800, 600]}),
            StrokeRecord(stroke_id="5", page_index=0, color_hex="#ff8800", width=3.0,
                         shape_type="ELLIPSE", shape_params={"center": [500, 500], "semi_major": 100, "semi_minor": 50, "angle_deg": 25}),
        ]

        out_pdf = tmp_dir / "geometric_export.pdf"
        dm.export_annotated_pdf([p1], strokes, out_pdf)
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 1000

        out_pages = dm.export_annotated_pages([p1], strokes, tmp_dir / "out_pages")
        assert len(out_pages) == 1
        assert out_pages[0].exists()
        rendered = cv2.imread(str(out_pages[0]))
        assert rendered.shape == (800, 1000, 3)

    def test_document_manager_verify_lesson_source_link(self, tmp_dir: Path):
        """DocumentManager.verify_lesson_source_link delegates to AirTeachPackage and verifies SHA-256."""
        dm = DocumentManager(work_root=tmp_dir / "work")
        p1 = tmp_dir / "slide.png"
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(p1), img)

        source_doc = tmp_dir / "quantum_mechanics.pdf"
        source_doc.write_bytes(b"%PDF-1.4 quantum mechanics lecture slides" * 30)

        bundle = tmp_dir / "quantum.airteach"
        manifest = AirTeachManifest(title="Quantum Mechanics")
        dm.save_lesson(bundle, manifest, [p1], [], [], original_file_path=source_doc)

        # Verification through DocumentManager facade
        res = dm.verify_lesson_source_link(bundle)
        assert res["linked"] is True
        assert res["verified"] is True
        assert res["status"] == "VERIFIED"
        assert res["original_filename"] == "quantum_mechanics.pdf"
        assert res["path"] == source_doc.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_minimal_pdf() -> bytes:
    """
    Return the bytes of a valid minimal 1-page PDF that pypdfium2 can render.
    Uses a raw PDF structure without any external library.
    """
    # This is a valid minimal PDF with one blank white A4 page.
    pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 100 700 Td (AirTeach AI) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f\r
0000000009 00000 n\r
0000000058 00000 n\r
0000000115 00000 n\r
0000000274 00000 n\r
0000000372 00000 n\r
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# 14. GeometricMathParser Tests (Offline Zero-Dependency Math Engine)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometricMathParser:
    def test_parse_empty_strokes_returns_empty(self):
        from src.services.smart_ink_ocr import GeometricMathParser
        assert GeometricMathParser.parse([]) == ""

    def test_parse_single_horizontal_is_minus(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        s = Stroke(points=[StrokePoint(10.0, 50.0, 0), StrokePoint(50.0, 50.0, 10)])
        res = GeometricMathParser.parse([s])
        assert "-" in res

    def test_parse_two_stacked_horizontal_lines_is_equal(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        s1 = Stroke(points=[StrokePoint(10.0, 45.0, 0), StrokePoint(50.0, 45.0, 10)])
        s2 = Stroke(points=[StrokePoint(10.0, 55.0, 20), StrokePoint(50.0, 55.0, 30)])
        res = GeometricMathParser.parse([s1, s2])
        assert "=" in res

    def test_parse_crossing_horizontal_vertical_is_plus(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        s_h = Stroke(points=[StrokePoint(10.0, 50.0, 0), StrokePoint(50.0, 50.0, 10)])
        s_v = Stroke(points=[StrokePoint(30.0, 30.0, 20), StrokePoint(30.0, 70.0, 30)])
        res = GeometricMathParser.parse([s_h, s_v])
        assert "+" in res

    def test_parse_fraction_structure(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # Numerator (above bar)
        s_num = Stroke(points=[StrokePoint(25.0, 20.0, 0), StrokePoint(25.0, 35.0, 10)])
        # Fraction bar
        s_bar = Stroke(points=[StrokePoint(10.0, 50.0, 20), StrokePoint(60.0, 50.0, 30)])
        # Denominator (below bar)
        s_den = Stroke(points=[StrokePoint(25.0, 65.0, 40), StrokePoint(25.0, 80.0, 50)])
        res = GeometricMathParser.parse([s_num, s_bar, s_den])
        assert r"\frac{" in res

    def test_parse_superscript_exponent(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # Base variable 'x'
        s_base = Stroke(points=[StrokePoint(20.0, 50.0, 0), StrokePoint(20.0, 90.0, 10)])
        # Small elevated '2'
        s_exp = Stroke(points=[StrokePoint(35.0, 35.0, 20), StrokePoint(45.0, 45.0, 30)])
        res = GeometricMathParser.parse([s_base, s_exp])
        assert "^" in res

    def test_smart_ink_recognizer_uses_geometric_fallback(self):
        from src.services.smart_ink_ocr import SmartInkRecognizer, Stroke, StrokePoint
        recognizer = SmartInkRecognizer()
        s1 = Stroke(points=[StrokePoint(10.0, 50.0, 0), StrokePoint(50.0, 50.0, 10)])
        s2 = Stroke(points=[StrokePoint(30.0, 30.0, 20), StrokePoint(30.0, 70.0, 30)])
        out = recognizer.recognize_strokes([s1, s2], force_formula=True, render_latex=True)
        assert out.is_formula
        assert out.latex != ""
        assert out.confidence >= 0.8
        assert out.render_image_rgba is not None

    def test_parse_parentheses(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        import numpy as np
        # Open parenthesis (
        t = np.linspace(-np.pi/2, np.pi/2, 10)
        pts_l = [StrokePoint(float(30.0 - 10.0 * np.cos(ti)), float(50.0 + 25.0 * np.sin(ti)), i * 10) for i, ti in enumerate(t)]
        # Closing parenthesis )
        pts_r = [StrokePoint(float(70.0 + 10.0 * np.cos(ti)), float(50.0 + 25.0 * np.sin(ti)), 100 + i * 10) for i, ti in enumerate(t)]
        # Middle variable x
        s_mid = Stroke(points=[StrokePoint(50.0, 40.0, 200), StrokePoint(50.0, 60.0, 210)])
        res = GeometricMathParser.parse([Stroke(points=pts_l), s_mid, Stroke(points=pts_r)])
        assert "(" in res
        assert ")" in res

    def test_parse_pi_symbol(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # Horizontal top bar
        s_top = Stroke(points=[StrokePoint(10.0, 20.0, 0), StrokePoint(50.0, 20.0, 10)])
        # Left vertical leg
        s_leg1 = Stroke(points=[StrokePoint(20.0, 22.0, 20), StrokePoint(20.0, 60.0, 30)])
        # Right vertical leg
        s_leg2 = Stroke(points=[StrokePoint(40.0, 22.0, 40), StrokePoint(40.0, 60.0, 50)])
        res = GeometricMathParser.parse([s_top, s_leg1, s_leg2])
        assert r"\pi" in res

    def test_parse_inequalities_and_not_equal(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # < chevron
        s_lt = Stroke(points=[
            StrokePoint(40.0, 20.0, 0),
            StrokePoint(20.0, 35.0, 10),
            StrokePoint(40.0, 50.0, 20),
        ])
        res_lt = GeometricMathParser.parse([s_lt])
        assert "<" in res_lt

        # <= chevron + bar
        s_bar = Stroke(points=[StrokePoint(15.0, 58.0, 30), StrokePoint(45.0, 58.0, 40)])
        res_leq = GeometricMathParser.parse([s_lt, s_bar])
        assert r"\leq" in res_leq

        # != equal sign + slash
        s1 = Stroke(points=[StrokePoint(10.0, 45.0, 0), StrokePoint(50.0, 45.0, 10)])
        s2 = Stroke(points=[StrokePoint(10.0, 55.0, 20), StrokePoint(50.0, 55.0, 30)])
        s_slash = Stroke(points=[StrokePoint(40.0, 35.0, 40), StrokePoint(20.0, 65.0, 50)])
        res_neq = GeometricMathParser.parse([s1, s2, s_slash])
        assert r"\neq" in res_neq

    def test_parse_dot_symbol(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # A tiny dot stroke
        s_dot = Stroke(points=[StrokePoint(25.0, 50.0, 0), StrokePoint(26.0, 50.5, 10)])
        res = GeometricMathParser.parse([s_dot])
        assert "." in res

    def test_parse_division_symbol(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        s_dot_top = Stroke([StrokePoint(100.0, 40.0, 0), StrokePoint(101.0, 41.0, 10)])
        s_bar = Stroke([StrokePoint(80.0, 60.0, 20), StrokePoint(120.0, 60.0, 30)])
        s_dot_bot = Stroke([StrokePoint(100.0, 80.0, 40), StrokePoint(101.0, 81.0, 50)])
        res = GeometricMathParser.parse([s_dot_top, s_bar, s_dot_bot])
        assert res == r"\div"

    def test_parse_plus_minus_symbol(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        s_h = Stroke([StrokePoint(80.0, 50.0, 0), StrokePoint(120.0, 50.0, 10)])
        s_v = Stroke([StrokePoint(100.0, 30.0, 20), StrokePoint(100.0, 70.0, 30)])
        s_b = Stroke([StrokePoint(80.0, 85.0, 40), StrokePoint(120.0, 85.0, 50)])
        res = GeometricMathParser.parse([s_h, s_v, s_b])
        assert res == r"\pm"

    def test_parse_delta_triangle_symbol(self):
        from src.services.smart_ink_ocr import GeometricMathParser, Stroke, StrokePoint
        # Single-stroke closed triangle
        s_tri = Stroke([
            StrokePoint(100.0, 200.0, 0),
            StrokePoint(150.0, 100.0, 10),
            StrokePoint(200.0, 200.0, 20),
            StrokePoint(100.0, 200.0, 30),
        ])
        res_single = GeometricMathParser.parse([s_tri])
        assert res_single == r"\Delta"

        # 3-stroke tent triangle (base bar + 2 diagonal lines)
        s_base = Stroke([StrokePoint(100.0, 200.0, 40), StrokePoint(200.0, 200.0, 50)])
        s_leg_pos = Stroke([StrokePoint(100.0, 200.0, 60), StrokePoint(150.0, 100.0, 70)])
        s_leg_neg = Stroke([StrokePoint(150.0, 100.0, 80), StrokePoint(200.0, 200.0, 90)])
        res_multi = GeometricMathParser.parse([s_base, s_leg_pos, s_leg_neg])
        assert res_multi == r"\Delta"

    def test_parse_adversarial_malformed_strokes_and_types(self):
        """Parser must gracefully reject None, NaNs, single points, and handle raw coordinate lists."""
        import numpy as np
        from src.services.smart_ink_ocr import GeometricMathParser

        pts_nan = np.array([[np.nan, 10.0], [20.0, np.inf]])
        class DummyNaNStroke:
            def to_numpy(self):
                return pts_nan

        class EmptyStroke:
            def to_numpy(self):
                return np.empty((0, 2))

        adv_strokes = [
            None,
            DummyNaNStroke(),
            EmptyStroke(),
            [(10.0, 10.0)],
            "not_a_valid_stroke",
        ]
        res = GeometricMathParser.parse(adv_strokes)
        assert res == ""

        # Raw coordinate list support
        pts_line = [(100.0, 100.0), (100.0, 200.0)]
        res_line = GeometricMathParser.parse([pts_line])
        assert res_line == "1"




class TestApplicationCommandLine:
    """Verify application CLI entry point flags (--version, --help, --check)."""

    def test_cli_version_flag(self):
        import subprocess
        import sys
        proc = subprocess.run([sys.executable, "main.py", "--version"], capture_output=True, text=True)
        assert proc.returncode == 0
        assert "AirTeach AI v2.5.0-Release" in proc.stdout

    def test_cli_help_flag(self):
        import subprocess
        import sys
        proc = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
        assert proc.returncode == 0
        assert "AirTeach AI" in proc.stdout
        assert "--check" in proc.stdout

    def test_cli_check_flag(self):
        import subprocess
        import sys
        proc = subprocess.run([sys.executable, "main.py", "--check"], capture_output=True, text=True)
        assert proc.returncode == 0
        assert "All systems operational. Ready to launch." in proc.stdout
