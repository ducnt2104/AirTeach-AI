"""
Unit tests for InteractiveFFTItem and InformaticsTools.create_fft_diagram.
Verifies Cooley-Tukey Radix-2 FFT algorithm O(N log N), twiddle factor calculations,
Parseval's energy conservation theorem, Shannon-Nyquist aliasing fold-over effect,
Fourier series harmonic deconvolution, DTMF dual-tone detection, touchless air gestures,
and lesson board integration.
"""

import sys
import math
import cmath
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_fft import (
    InteractiveFFTItem,
    FFT_PRESETS,
)
from src.ui.stem_tools import InformaticsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_fft_initialization(qapp):
    item = InteractiveFFTItem()
    assert item.active_preset.code == "AUDIO_A440_CONCERT_PITCH"
    assert math.isclose(item.f0_hz, 125.0, rel_tol=1e-4)
    assert math.isclose(item.sampling_rate_hz, 1000.0, rel_tol=1e-4)
    assert math.isclose(item.nyquist_frequency_hz, 500.0, rel_tol=1e-4)
    assert math.isclose(item.frequency_resolution_hz, 1000.0 / 64.0, rel_tol=1e-4)
    assert item.boundingRect().width() == InteractiveFFTItem.CARD_W
    assert item.boundingRect().height() == InteractiveFFTItem.CARD_H


def test_cooley_tukey_radix2_exact_math(qapp):
    """Verifies Cooley-Tukey Radix-2 FFT output against analytical DFT definitions."""
    # 1. Delta impulse delta[n]: FFT should be all 1.0 + 0.0j
    impulse = [complex(1.0 if i == 0 else 0.0, 0.0) for i in range(8)]
    fft_impulse = InteractiveFFTItem._cooley_tukey_fft(impulse)
    for v in fft_impulse:
        assert math.isclose(abs(v), 1.0, rel_tol=1e-5)

    # 2. DC constant signal x[n] = 1.0: X[0] = N, all other X[k] = 0.0
    dc = [complex(1.0, 0.0) for _ in range(8)]
    fft_dc = InteractiveFFTItem._cooley_tukey_fft(dc)
    assert math.isclose(abs(fft_dc[0]), 8.0, rel_tol=1e-5)
    for k in range(1, 8):
        assert abs(fft_dc[k]) < 1e-5


def test_parseval_theorem_energy_conservation(qapp):
    """
    Parseval's Energy Conservation Theorem Invariant:
    sum(|x[n]|^2) == (1/N) * sum(|X[k]|^2)
    Must hold identically across all presets and arbitrary signals.
    """
    item = InteractiveFFTItem()
    for preset_code in FFT_PRESETS.keys():
        item.load_preset(preset_code)
        time_energy = item._parseval_time_energy
        freq_energy = item._parseval_freq_energy
        assert math.isclose(time_energy, freq_energy, rel_tol=1e-4), (
            f"Parseval theorem violated for preset {preset_code}: {time_energy} vs {freq_energy}"
        )


def test_frequency_resolution_and_peak_detection(qapp):
    item = InteractiveFFTItem()
    item.load_preset("AUDIO_A440_CONCERT_PITCH")
    # Peak frequency should match fundamental f0 (125 Hz)
    assert math.isclose(item.peak_frequency_hz, 125.0, rel_tol=0.1)


def test_square_wave_odd_harmonics(qapp):
    item = InteractiveFFTItem()
    item.load_preset("SQUARE_WAVE_FOURIER_SERIES")

    # In square wave with f0 = 62.5 Hz (bin 4), 3f0 = 187.5 Hz (bin 12), 5f0 = 312.5 Hz (bin 20)
    # Bin index = freq / (1000 / 64) = freq / 15.625
    # Bin 4: 62.5 Hz
    # Bin 8: 125 Hz (even harmonic -> amplitude should be nearly zero)
    # Bin 12: 187.5 Hz (3rd harmonic -> amplitude ~ 1/3)
    bin_f0 = int(round(62.5 / item.frequency_resolution_hz))
    bin_2f0 = int(round(125.0 / item.frequency_resolution_hz))
    bin_3f0 = int(round(187.5 / item.frequency_resolution_hz))

    mag_f0 = item._fft_magnitudes[bin_f0]
    mag_2f0 = item._fft_magnitudes[bin_2f0]
    mag_3f0 = item._fft_magnitudes[bin_3f0]

    # Fundamental should be strong
    assert mag_f0 > 0.5
    # Even harmonic should be negligible
    assert mag_2f0 < 0.15
    # 3rd harmonic should be present and approximately 1/3 of fundamental
    assert mag_3f0 > 0.15


def test_nyquist_aliasing_foldover(qapp):
    item = InteractiveFFTItem()
    item.load_preset("NYQUIST_ALIASING_DEMONSTRATION")

    # f_sample = 500 Hz -> Nyquist = 250 Hz
    # Input f0 = 375 Hz > 250 Hz
    # Aliased peak appears at |375 - 500| = 125 Hz
    assert math.isclose(item.peak_frequency_hz, 125.0, rel_tol=0.1)


def test_dtmf_dual_tone_peaks(qapp):
    item = InteractiveFFTItem()
    item.load_preset("DUAL_TONE_MULTI_FREQUENCY_DTMF")

    # Both tones (125 Hz and 281.25 Hz) should produce peaks
    bin1 = int(round(125.0 / item.frequency_resolution_hz))
    bin2 = int(round(281.25 / item.frequency_resolution_hz))

    assert item._fft_magnitudes[bin1] > 0.4
    assert item._fft_magnitudes[bin2] > 0.4


def test_noisy_signal_detection(qapp):
    item = InteractiveFFTItem()
    item.load_preset("NOISY_SIGNAL_SPECTRAL_FILTER")
    # Target tone is at 187.5 Hz
    bin_target = int(round(187.5 / item.frequency_resolution_hz))
    assert item._fft_magnitudes[bin_target] > 0.25


def test_presets_loading_and_steppers(qapp):
    item = InteractiveFFTItem()
    item.step_frequency(31.25)
    assert math.isclose(item.f0_hz, 156.25, rel_tol=1e-4)

    item.step_sampling_rate(250.0)
    assert math.isclose(item.sampling_rate_hz, 1250.0, rel_tol=1e-4)

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_air_pinch_and_drag(qapp):
    item = InteractiveFFTItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(40.0, 30.0))
    assert item.pos() == start_pos + QPointF(40.0, 30.0)

    # Air pinch on btn_f_up
    rect_f_up = item._buttons["btn_f_up"]
    scene_pt = item.mapToScene(rect_f_up.center())
    prev_f0 = item.f0_hz
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item.f0_hz > prev_f0


def test_stamp_diagram(qapp):
    item = InteractiveFFTItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "peak_freq_hz" in stamped[0]
    assert "time_energy" in stamped[0]
    assert "freq_energy" in stamped[0]


def test_stem_tools_fft_diagram(qapp):
    items = InformaticsTools.create_fft_diagram(
        center=(400.0, 300.0),
        width=700.0,
        height=400.0,
        preset_code="AUDIO_A440_CONCERT_PITCH",
        f0_hz=125.0,
        sampling_rate_hz=1000.0,
        peak_freq_hz=125.0,
        resolution_hz=15.625,
        nyquist_hz=500.0,
    )
    assert len(items) >= 20
    assert any("COOLEY-TUKEY" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_fft_integration(qapp):
    board = LessonBoard()
    assert board.is_fft_visible() is False
    assert board.get_fft() is None

    # Show
    item = board.show_fft(True)
    assert item is not None
    assert board.is_fft_visible() is True
    assert board.get_fft() is item
    assert board.get_fft_item() is item

    # Toggle to hide
    visible = board.toggle_fft()
    assert visible is False
    assert board.is_fft_visible() is False
    assert board.get_fft() is None
    assert board.get_fft_item() is not None

    # Toggle to show
    visible = board.toggle_fft()
    assert visible is True
    assert board.is_fft_visible() is True
