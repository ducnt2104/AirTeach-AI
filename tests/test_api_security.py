"""
Tests for AirTeach AI Secure Credential & Per-Teacher AI Settings
=================================================================
Verifies:
  - Account namespace isolation (Teacher A vs Teacher B)
  - Key masking and redaction
  - Save, load, update, delete credentials
  - Per-teacher AI settings persistence
  - Robust handling of missing / empty keys
"""
import os
import tempfile
from pathlib import Path

import pytest

from src.services.credential_service import TeacherAISettings, TeacherCredentialService


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_profiles.db"
    return db_file


@pytest.fixture
def cred_service(temp_db: Path):
    return TeacherCredentialService(db_path=temp_db)


def test_per_teacher_isolation(cred_service: TeacherCredentialService):
    teacher_a = "teacher_uuid_alpha_111"
    teacher_b = "teacher_uuid_beta_222"

    key_a = "AIzaSyFakeKeyForTeacherAlpha123456789"
    key_b = "AIzaSyFakeKeyForTeacherBeta987654321"

    # Set Key for Teacher A
    assert cred_service.set_api_key(teacher_a, key_a)
    assert cred_service.has_api_key(teacher_a)

    # Teacher B must have NO key yet
    assert not cred_service.has_api_key(teacher_b)
    assert cred_service.get_api_key(teacher_b) == ""

    # Set Key for Teacher B
    assert cred_service.set_api_key(teacher_b, key_b)
    assert cred_service.has_api_key(teacher_b)

    # Read back and verify strict isolation
    assert cred_service.get_api_key(teacher_a) == key_a
    assert cred_service.get_api_key(teacher_b) == key_b

    # Modify Teacher A's key — Teacher B must NOT change
    new_key_a = "AIzaSyUpdatedKeyAlpha111222333444555"
    assert cred_service.set_api_key(teacher_a, new_key_a)
    assert cred_service.get_api_key(teacher_a) == new_key_a
    assert cred_service.get_api_key(teacher_b) == key_b

    # Delete Teacher A's key — Teacher B must still have their key
    assert cred_service.delete_api_key(teacher_a)
    assert not cred_service.has_api_key(teacher_a)
    assert cred_service.get_api_key(teacher_a) == ""
    assert cred_service.get_api_key(teacher_b) == key_b

    # Clean up Teacher B
    cred_service.delete_api_key(teacher_b)


def test_key_masking():
    # Long key
    key = "AIzaSyD849f8a37fbc82e9120ac92834b7f94xQ9"
    masked = TeacherCredentialService.mask_api_key(key)
    assert masked.startswith("AIzaSy")
    assert masked.endswith("4xQ9")
    assert "•" in masked
    assert key not in masked  # The full plaintext key should never equal masked

    # Empty key
    assert "Chưa cấu hình" in TeacherCredentialService.mask_api_key("")

    # Short key
    assert TeacherCredentialService.mask_api_key("12345") == "••••••••"


def test_key_redaction():
    text = "Error calling endpoint with key AIzaSyTestKey1234567890ABCDEFGHIJKLM failed"
    redacted = TeacherCredentialService.redact_text(text)
    assert "AIzaSyTestKey1234567890ABCDEFGHIJKLM" not in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_per_teacher_ai_settings_persistence(cred_service: TeacherCredentialService):
    teacher_id = "teacher_settings_test_001"

    # Default settings before save
    default_settings = cred_service.get_settings(teacher_id)
    assert default_settings.teacher_id == teacher_id
    assert default_settings.model_name == "gemini-2.5-flash"
    assert default_settings.draw_mode == "index_up"
    assert default_settings.feature_lesson_assistant is True

    # Custom settings
    custom = TeacherAISettings(
        teacher_id=teacher_id,
        provider="gemini",
        model_name="gemini-1.5-pro",
        feature_lesson_assistant=True,
        feature_doc_summary=False,
        feature_quiz_gen=True,
        feature_concept_explainer=False,
        feature_lesson_plan=True,
        feature_smart_ink=True,
        draw_mode="precision_pinch",
        camera_lighting_mode="low_light",
        pinch_threshold=0.25,
    )

    assert cred_service.save_settings(custom)

    # Retrieve and verify
    loaded = cred_service.get_settings(teacher_id)
    assert loaded.model_name == "gemini-1.5-pro"
    assert loaded.feature_doc_summary is False
    assert loaded.feature_concept_explainer is False
    assert loaded.draw_mode == "precision_pinch"
    assert loaded.camera_lighting_mode == "low_light"
    assert abs(loaded.pinch_threshold - 0.25) < 1e-4


def test_keyring_username_sanitization_and_collision_resistance():
    # Normal ID
    u1 = TeacherCredentialService._make_keyring_username("teacher_01", "gemini")
    assert u1 == "teacher_teacher_01_gemini"

    # Path traversal attempt
    u2 = TeacherCredentialService._make_keyring_username("../../admin", "gemini")
    assert "../" not in u2
    assert u2 == "teacher_admin_gemini"

    # Unicode Vietnamese name
    u3 = TeacherCredentialService._make_keyring_username("Nguyễn_Văn_A", "gemini")
    assert "gemini" in u3
    assert len(u3) > 10

    # Two distinct symbol strings must NOT collide
    u_sym1 = TeacherCredentialService._make_keyring_username("###", "gemini")
    u_sym2 = TeacherCredentialService._make_keyring_username("$$$", "gemini")
    assert u_sym1 != u_sym2


def test_advanced_secret_redaction():
    text1 = "Google key: AIzaSyD1234567890ABCDEFGHIJKLM"
    text2 = "OpenAI key: sk-proj-1234567890abcdefghijklmnopqr"
    combined = f"{text1} and {text2}"

    redacted = TeacherCredentialService.redact_text(combined)
    assert "AIzaSyD1234567890ABCDEFGHIJKLM" not in redacted
    assert "sk-proj-1234567890abcdefghijklmnopqr" not in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "[REDACTED_SECRET]" in redacted
