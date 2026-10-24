"""
AirTeach AI - Auditory Feedback & Synthetic Acoustic Engine Tests
=================================================================
Verifies in-memory synthetic PCM WAV waveform generation, volume control,
singleton caching, event trigger bindings, and SettingsDialog persistence.
"""
import io
import wave
import pytest
from PySide6.QtWidgets import QApplication

from src.utils.audio_feedback import (
    AudioFeedbackManager,
    SoundEffect,
    _synthesize_pcm_wav,
)
from src.ui.settings_dialog import SettingsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_synthesize_pcm_wav_header_and_structure():
    """Verify generated WAV bytes adhere strictly to standard 16-bit PCM WAV spec."""
    raw_wav = _synthesize_pcm_wav(
        frequencies=[440.0, 880.0],
        duration_sec=0.05,
        sample_rate=22050,
        volume=0.8,
        decay_rate=4.0,
        attack_rate=40.0,
        pitch_bend_rate=-500.0,
    )
    assert len(raw_wav) > 44  # WAV header is 44 bytes

    # Parse with Python standard wave library
    buf = io.BytesIO(raw_wav)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 22050
        num_frames = wf.getnframes()
        assert num_frames == int(22050 * 0.05)


def test_audio_feedback_manager_singleton_and_controls():
    """Verify singleton lifecycle, enabled toggle, and volume clamping."""
    mgr1 = AudioFeedbackManager.instance()
    mgr2 = AudioFeedbackManager.instance()
    assert mgr1 is mgr2

    # Volume clamping
    mgr1.volume = 1.5
    assert mgr1.volume == 1.0

    mgr1.volume = -0.5
    assert mgr1.volume == 0.0

    mgr1.volume = 0.7
    assert abs(mgr1.volume - 0.7) < 1e-4

    # Enabled state
    mgr1.enabled = False
    assert not mgr1.enabled
    mgr1.play(SoundEffect.CLICK)  # Should return immediately without playing

    mgr1.enabled = True
    assert mgr1.enabled


def test_audio_feedback_manager_play_methods_safe_execution():
    """Verify all sound effect trigger methods run safely without throwing exceptions."""
    mgr = AudioFeedbackManager.instance()
    mgr.enabled = True
    mgr.volume = 0.8

    # All public playback methods must be non-blocking & exception-safe
    mgr.play_click()
    mgr.play_snap()
    mgr.play_page_turn()
    mgr.play_erase()
    mgr.play_timer_chime()
    mgr.play_undo()


def test_settings_dialog_audio_feedback_integration(qapp):
    """Verify SettingsDialog exposes audio controls and synchronizes with manager."""
    dlg = SettingsDialog(teacher_id="test_teacher", teacher_name="Teacher Audio Test")
    mgr = AudioFeedbackManager.instance()

    assert hasattr(dlg, "audio_feedback_check")
    assert hasattr(dlg, "audio_volume_spin")
    assert hasattr(dlg, "btn_test_audio")

    # Change settings in UI
    dlg.audio_feedback_check.setChecked(True)
    dlg.audio_volume_spin.setValue(85)
    assert dlg.audio_feedback_enabled is True
    assert abs(dlg.audio_volume - 0.85) < 1e-3

    # Click test audio button
    dlg.btn_test_audio.click()

    # Save all settings
    dlg._on_save_all()
    assert mgr.enabled is True
    assert abs(mgr.volume - 0.85) < 1e-3
