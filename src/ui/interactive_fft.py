"""
AirTeach AI — Interactive Computer Science & Signal Processing: Cooley-Tukey Fast Fourier Transform (FFT).
Simulates Discrete Fourier Transform (DFT), Radix-2 Decimation-in-Time Cooley-Tukey algorithm O(N log N),
complex twiddle factors W_N^k = e^(-i 2πk/N), butterfly flow diagrams, Shannon-Nyquist sampling theorem,
aliasing fold-over effects, harmonic spectrum deconvolution, and Parseval's energy conservation theorem.
"""

from __future__ import annotations
import math
import cmath
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QLinearGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class FFTPreset:
    code: str
    title_vi: str
    badge_label: str
    signal_type: str         # "SINE_HARMONICS", "SQUARE_SERIES", "ALIASING_DEMO", "DTMF_KEYPAD", "NOISY_SIGNAL"
    f0_hz: float             # Fundamental frequency Hz
    f_sample_hz: float       # Sampling rate Hz (e.g. 1000 Hz, Nyquist = 500 Hz)
    noise_amplitude: float   # 0.0 to 1.0
    description: str


FFT_PRESETS: dict[str, FFTPreset] = {
    "AUDIO_A440_CONCERT_PITCH": FFTPreset(
        code="AUDIO_A440_CONCERT_PITCH",
        title_vi="Nốt La Chuẩn A4 (440 Hz) & Bội âm Họa tần (Harmonics)",
        badge_label="CONCERT A440",
        signal_type="SINE_HARMONICS",
        f0_hz=125.0,  # Scaled for 64-point / 1000 Hz grid
        f_sample_hz=1000.0,
        noise_amplitude=0.0,
        description="Sóng âm thuần khiết kết hợp hài âm bậc 2 (2f₀) và bậc 3 (3f₀). Biến đổi Fourier phân rã tín hiệu miền thời gian thành các vạch phổ Dirac riêng biệt tương ứng với từng tần số.",
    ),
    "SQUARE_WAVE_FOURIER_SERIES": FFTPreset(
        code="SQUARE_WAVE_FOURIER_SERIES",
        title_vi="Khai triển Chuỗi Fourier Sóng vuông — Hài lẻ (1/k)",
        badge_label="SQUARE FOURIER",
        signal_type="SQUARE_SERIES",
        f0_hz=62.5,
        f_sample_hz=1000.0,
        noise_amplitude=0.0,
        description="Sóng vuông cấu thành từ tổng vô hạn các hài bậc lẻ: x(t) = sin(ωt) + (1/3)sin(3ωt) + (1/5)sin(5ωt) + ... Phổ biên độ FFT thể hiện rõ sự suy giảm tỷ lệ nghịch 1/k và hiện tượng gợn sóng Gibbs.",
    ),
    "NYQUIST_ALIASING_DEMONSTRATION": FFTPreset(
        code="NYQUIST_ALIASING_DEMONSTRATION",
        title_vi="Hiện tượng Chồng phổ Nyquist (Aliasing) — Tần số Ảo vặn ngược",
        badge_label="NYQUIST ALIAS",
        signal_type="ALIASING_DEMO",
        f0_hz=375.0,  # > f_Nyquist (500 Hz) -> aliases to |375 - 1000| = 625 -> 125 Hz
        f_sample_hz=500.0,  # f_Nyquist = 250 Hz, f0 = 375 Hz -> alias at 125 Hz!
        noise_amplitude=0.0,
        description="Định lý Shannon-Nyquist: f_s phải ≥ 2 f_max. Khi tín hiệu 375 Hz bị lấy mẫu ở tốc độ 500 Hz (Nyquist=250 Hz), đỉnh phổ thật biến mất và bị 'gập gương' thành đỉnh ảo giả ở |375 - 500| = 125 Hz.",
    ),
    "DUAL_TONE_MULTI_FREQUENCY_DTMF": FFTPreset(
        code="DUAL_TONE_MULTI_FREQUENCY_DTMF",
        title_vi="Tín hiệu Bàn phím Điện thoại DTMF — Giải mã Phổ Kép",
        badge_label="DTMF DUAL TONE",
        signal_type="DTMF_KEYPAD",
        f0_hz=125.0,  # Row frequency
        f_sample_hz=1000.0,
        noise_amplitude=0.0,
        description="Chuẩn DTMF kết hợp đồng thời 2 tần số âm thanh (tần số hàng 125 Hz + tần số cột 281 Hz) để mã hóa phím bấm. Bộ lọc FFT tách biệt hoàn hảo 2 đỉnh phổ để máy chủ viễn thông giải mã phím.",
    ),
    "NOISY_SIGNAL_SPECTRAL_FILTER": FFTPreset(
        code="NOISY_SIGNAL_SPECTRAL_FILTER",
        title_vi="Phát hiện Tín hiệu Chìm trong Nhiễu Trắng Gaussian (SNR < 0 dB)",
        badge_label="NOISE EXTRACTION",
        signal_type="NOISY_SIGNAL",
        f0_hz=187.5,
        f_sample_hz=1000.0,
        noise_amplitude=0.85,
        description="Trong miền thời gian, dạng sóng bị nhiễu che lấp hoàn toàn không thể nhận diện mắt thường. Trong miền tần số, FFT tập trung năng lượng tín hiệu điều hòa thành một đỉnh nhọn vượt trội khỏi sàn nhiễu.",
    ),
}


class InteractiveFFTItem(QGraphicsObject):
    """
    Interactive Cooley-Tukey Radix-2 FFT & Spectrum Analyzer.
    Provides live sampling of waveforms, Radix-2 Cooley-Tukey butterfly FFT computation,
    magnitude and phase spectrum display, Nyquist aliasing visualization,
    and Parseval's energy theorem verification.
    """

    preset_changed = Signal(str)
    fft_computed = Signal(float, float)  # peak_freq_hz, total_energy
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0
    N_POINTS = 64  # Power of 2 Radix-2 FFT sample count

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset
        self._active_preset_code: str = "AUDIO_A440_CONCERT_PITCH"
        self._preset: FFTPreset = FFT_PRESETS[self._active_preset_code]

        # Wave & Sampling Parameters
        self._f0_hz: float = self._preset.f0_hz
        self._f_sample_hz: float = self._preset.f_sample_hz
        self._noise_amp: float = self._preset.noise_amplitude

        # Discrete Time Signal & Spectrum
        self._sim_time_s: float = 0.0
        self._is_running: bool = True
        self._time_samples: list[float] = [0.0] * self.N_POINTS
        self._fft_magnitudes: list[float] = [0.0] * (self.N_POINTS // 2)
        self._fft_phases: list[float] = [0.0] * (self.N_POINTS // 2)
        self._peak_freq_hz: float = 0.0
        self._parseval_time_energy: float = 0.0
        self._parseval_freq_energy: float = 0.0

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Update initial buffer
        self._update_samples_and_fft()

        # 30 fps animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> FFTPreset:
        return self._preset

    @property
    def f0_hz(self) -> float:
        return self._f0_hz

    @property
    def sampling_rate_hz(self) -> float:
        return self._f_sample_hz

    @property
    def nyquist_frequency_hz(self) -> float:
        return self._f_sample_hz / 2.0

    @property
    def frequency_resolution_hz(self) -> float:
        """Delta f = f_s / N (Hz per bin)."""
        return self._f_sample_hz / float(self.N_POINTS)

    @property
    def peak_frequency_hz(self) -> float:
        return self._peak_freq_hz

    # -------------------------------------------------------------------------
    # Cooley-Tukey Radix-2 FFT & Signal Generation
    # -------------------------------------------------------------------------
    def _generate_signal_value(self, t: float) -> float:
        """Generates continuous signal x(t) based on active preset mode."""
        st = self._preset.signal_type
        f0 = self._f0_hz
        val = 0.0

        if st == "SINE_HARMONICS":
            # Fundamental + 2nd + 3rd harmonic
            val = (
                1.0 * math.sin(2.0 * math.pi * f0 * t)
                + 0.5 * math.sin(2.0 * math.pi * (2.0 * f0) * t)
                + 0.25 * math.sin(2.0 * math.pi * (3.0 * f0) * t)
            )
        elif st == "SQUARE_SERIES":
            # Sum of odd harmonics: sin(ωt) + 1/3 sin(3ωt) + 1/5 sin(5ωt) + 1/7 sin(7ωt)
            val = (
                math.sin(2.0 * math.pi * f0 * t)
                + (1.0 / 3.0) * math.sin(2.0 * math.pi * 3.0 * f0 * t)
                + (1.0 / 5.0) * math.sin(2.0 * math.pi * 5.0 * f0 * t)
                + (1.0 / 7.0) * math.sin(2.0 * math.pi * 7.0 * f0 * t)
            )
        elif st == "ALIASING_DEMO":
            # Pure high frequency sine violating Nyquist
            val = 1.0 * math.sin(2.0 * math.pi * f0 * t)
        elif st == "DTMF_KEYPAD":
            # Dual tones: f0 (e.g. 125 Hz) + secondary column tone (e.g. 281 Hz)
            val = 0.8 * math.sin(2.0 * math.pi * f0 * t) + 0.8 * math.sin(2.0 * math.pi * 281.25 * t)
        else:
            # SINE submerged in Gaussian noise
            val = 0.5 * math.sin(2.0 * math.pi * f0 * t)

        # Add noise if configured
        if self._noise_amp > 0.0:
            val += (random.random() - 0.5) * 2.0 * self._noise_amp

        return val

    @staticmethod
    def _cooley_tukey_fft(x: list[complex]) -> list[complex]:
        """
        Recursive Cooley-Tukey Radix-2 Decimation-in-Time (DIT) FFT algorithm.
        O(N log2 N) complex operations.
        """
        n = len(x)
        if n <= 1:
            return x

        even = InteractiveFFTItem._cooley_tukey_fft(x[0::2])
        odd = InteractiveFFTItem._cooley_tukey_fft(x[1::2])

        # Twiddle factor W_n^k = exp(-2j * pi * k / n)
        factor = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
        left = [even[k] + factor[k] for k in range(n // 2)]
        right = [even[k] - factor[k] for k in range(n // 2)]
        return left + right

    def _update_samples_and_fft(self) -> None:
        """Samples time waveform and computes Radix-2 FFT spectrum and Parseval metrics."""
        dt = 1.0 / self._f_sample_hz
        t_start = self._sim_time_s

        # Sample N points
        self._time_samples = [self._generate_signal_value(t_start + i * dt) for i in range(self.N_POINTS)]

        # Time-domain energy (sum of x[n]^2)
        self._parseval_time_energy = sum(s**2 for s in self._time_samples)

        # Run Cooley-Tukey Radix-2 FFT
        complex_input = [complex(s, 0.0) for s in self._time_samples]
        fft_result = self._cooley_tukey_fft(complex_input)

        # Frequency-domain energy: (1/N) * sum(|X[k]|^2)
        self._parseval_freq_energy = (1.0 / self.N_POINTS) * sum(abs(v)**2 for v in fft_result)

        # Extract single-sided magnitude spectrum (first N/2 bins)
        half_n = self.N_POINTS // 2
        mags: list[float] = []
        phases: list[float] = []

        for k in range(half_n):
            val = fft_result[k]
            # Normalized magnitude: 2/N for AC, 1/N for DC
            norm_factor = 2.0 / self.N_POINTS if k > 0 else 1.0 / self.N_POINTS
            mag = abs(val) * norm_factor
            phase = math.atan2(val.imag, val.real)
            mags.append(mag)
            phases.append(phase)

        self._fft_magnitudes = mags
        self._fft_phases = phases

        # Identify peak frequency
        if mags:
            max_idx = max(range(len(mags)), key=lambda i: mags[i])
            self._peak_freq_hz = max_idx * self.frequency_resolution_hz
        else:
            self._peak_freq_hz = 0.0

        self.fft_computed.emit(self._peak_freq_hz, self._parseval_freq_energy)

    def _on_sim_tick(self) -> None:
        if self._is_running:
            # Advance simulated time
            self._sim_time_s += 0.002
            self._update_samples_and_fft()
        self.update()

    # -------------------------------------------------------------------------
    # Presets & Interactive Steppers
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a digital signal processing preset."""
        if preset_code not in FFT_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = FFT_PRESETS[preset_code]
        self._f0_hz = self._preset.f0_hz
        self._f_sample_hz = self._preset.f_sample_hz
        self._noise_amp = self._preset.noise_amplitude
        self._update_samples_and_fft()
        self.preset_changed.emit(preset_code)
        self.update()

    def step_frequency(self, delta_hz: float) -> None:
        """Steps fundamental frequency f0."""
        self._f0_hz = max(10.0, min(self._f_sample_hz, self._f0_hz + delta_hz))
        self._update_samples_and_fft()
        self.update()

    def step_sampling_rate(self, delta_hz: float) -> None:
        """Steps sampling rate f_s."""
        self._f_sample_hz = max(200.0, min(4000.0, self._f_sample_hz + delta_hz))
        self._update_samples_and_fft()
        self.update()

    def toggle_animation(self) -> None:
        """Toggles real-time streaming animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_f_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_f_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_fs_down"] = QRectF(290.0, bar_y, 50.0, 34.0)
        self._buttons["btn_fs_up"] = QRectF(345.0, bar_y, 50.0, 34.0)
        self._buttons["btn_p_square"] = QRectF(402.0, bar_y, 90.0, 34.0)
        self._buttons["btn_p_nyquist"] = QRectF(497.0, bar_y, 95.0, 34.0)
        self._buttons["btn_p_dtmf"] = QRectF(597.0, bar_y, 80.0, 34.0)
        self._buttons["btn_p_noise"] = QRectF(682.0, bar_y, 85.0, 34.0)
        self._buttons["btn_play"] = QRectF(772.0, bar_y, 65.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(842.0, bar_y, 88.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(FFT_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_f_down":
            self.step_frequency(-15.625)
        elif key == "btn_f_up":
            self.step_frequency(15.625)
        elif key == "btn_fs_down":
            self.step_sampling_rate(-250.0)
        elif key == "btn_fs_up":
            self.step_sampling_rate(250.0)
        elif key == "btn_p_square":
            self.load_preset("SQUARE_WAVE_FOURIER_SERIES")
        elif key == "btn_p_nyquist":
            self.load_preset("NYQUIST_ALIASING_DEMONSTRATION")
        elif key == "btn_p_dtmf":
            self.load_preset("DUAL_TONE_MULTI_FREQUENCY_DTMF")
        elif key == "btn_p_noise":
            self.load_preset("NOISY_SIGNAL_SPECTRAL_FILTER")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "f0_hz": self._f0_hz,
            "sampling_rate_hz": self._f_sample_hz,
            "peak_freq_hz": self._peak_freq_hz,
            "resolution_hz": self.frequency_resolution_hz,
            "nyquist_hz": self.nyquist_frequency_hz,
            "time_energy": self._parseval_time_energy,
            "freq_energy": self._parseval_freq_energy,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_oscilloscope_panel(painter)
        self._paint_fft_spectrum_panel(painter)
        self._paint_butterfly_diagram_panel(painter)
        self._paint_analytical_metrics(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(9, 14, 25, 252))
        grad.setColorAt(1.0, QColor(5, 8, 16, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"XỬ LÝ TÍN HIỆU SỐ & THUẬT TOÁN COOLEY-TUKEY FFT: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(22, 48, f"Biến đổi Fourier N = {self.N_POINTS} điểm | Độ phức tạp O(N log₂ N) = {self.N_POINTS * int(math.log2(self.N_POINTS))} phép tính (vs N² = {self.N_POINTS**2}) | f_s = {self._f_sample_hz:.0f} Hz | Nyquist = {self.nyquist_frequency_hz:.0f} Hz")

    def _paint_oscilloscope_panel(self, painter: QPainter) -> None:
        """Renders time-domain discrete oscilloscope display x[n]."""
        ox = 20.0
        oy = 60.0
        ow = 440.0
        oh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(ox, oy, ow, oh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ox + 12.0, oy + 20.0, "MIỀN THỜI GIAN: DAO ĐỘNG KÝ x[n] (N = 64 điểm):")

        # Inner oscilloscope grid
        gx = ox + 40.0
        gy = oy + oh * 0.55
        gw = ow - 55.0
        gh = oh - 60.0

        painter.setPen(QPen(QColor(51, 65, 85), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(gx, gy, gx + gw, gy)
        painter.drawLine(gx, gy - gh * 0.4, gx + gw, gy - gh * 0.4)
        painter.drawLine(gx, gy + gh * 0.4, gx + gw, gy + gh * 0.4)

        # Draw continuous path
        wave_path = QPainterPath()
        step_x = gw / float(self.N_POINTS - 1)
        amp_scale = (gh * 0.38) / 1.5

        for i, val in enumerate(self._time_samples):
            px = gx + i * step_x
            py = gy - val * amp_scale
            if i == 0:
                wave_path.moveTo(px, py)
            else:
                wave_path.lineTo(px, py)

        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawPath(wave_path)

        # Sample points dots
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(0, self.N_POINTS, 4):
            px = gx + i * step_x
            py = gy - self._time_samples[i] * amp_scale
            painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

        # Time axis label
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(gx, gy + gh * 0.45 + 12.0, "t = 0")
        painter.drawText(gx + gw - 35.0, gy + gh * 0.45 + 12.0, f"Δt = {self.N_POINTS / self._f_sample_hz * 1000:.1f}ms")

    def _paint_fft_spectrum_panel(self, painter: QPainter) -> None:
        """Renders frequency-domain discrete magnitude spectrum |X[k]|."""
        fx = 20.0
        fy = 315.0
        fw = 440.0
        fh = 245.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(fx, fy, fw, fh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.drawText(fx + 12.0, fy + 20.0, "MIỀN TẦN SỐ: PHỔ BIÊN ĐỘ COOLEY-TUKEY FFT |X[k]|:")

        # Inner plot area
        px0 = fx + 40.0
        py0 = fy + fh - 35.0
        pw = fw - 55.0
        ph = fh - 65.0

        # Axes
        painter.setPen(QPen(QColor(100, 116, 139), 1.0))
        painter.drawLine(px0, py0, px0 + pw, py0)
        painter.drawLine(px0, py0, px0, py0 - ph)

        # Draw magnitude bars
        half_n = len(self._fft_magnitudes)
        bar_w = (pw / float(half_n)) * 0.75
        bar_gap = (pw / float(half_n))

        max_mag = max(1.0, max(self._fft_magnitudes) if self._fft_magnitudes else 1.0)
        scale_y = (ph * 0.85) / max_mag

        for k, mag in enumerate(self._fft_magnitudes):
            bx = px0 + k * bar_gap + 2.0
            bh = mag * scale_y
            by = py0 - bh

            # Highlight peak bar in red/orange, others in amber
            is_peak = math.isclose(k * self.frequency_resolution_hz, self._peak_freq_hz, abs_tol=1e-3)
            bar_col = QColor(239, 68, 68) if is_peak else QColor(245, 158, 11)

            painter.setBrush(QBrush(bar_col))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(bx, by, bar_w, bh), 2.0, 2.0)

        # Frequency axis ticks
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px0 - 5.0, py0 + 14.0, "0 Hz")
        painter.drawText(px0 + pw / 2.0 - 15.0, py0 + 14.0, f"{self.nyquist_frequency_hz / 2.0:.0f} Hz")
        painter.drawText(px0 + pw - 30.0, py0 + 14.0, f"{self.nyquist_frequency_hz:.0f} Hz")

        # Peak indicator text
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.drawText(px0 + 15.0, py0 - ph + 15.0, f"★ Đỉnh phổ cực đại: {self._peak_freq_hz:.1f} Hz")

    def _paint_butterfly_diagram_panel(self, painter: QPainter) -> None:
        """Renders 8-point Cooley-Tukey butterfly signal flow graph (DIT)."""
        bx = 475.0
        by = 60.0
        bw = 455.0
        bh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bx, by, bw, bh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(16, 185, 129)))
        painter.drawText(bx + 12.0, by + 20.0, "SƠ ĐỒ BƯỚM (BUTTERFLY FLOW GRAPH) COOLEY-TUKEY 8 ĐIỂM:")

        # 8 inputs with bit-reversal order: [0, 4, 2, 6, 1, 5, 3, 7]
        bit_rev_order = [0, 4, 2, 6, 1, 5, 3, 7]
        start_x = bx + 45.0
        stage_w = 90.0
        row_h = 22.0
        start_y = by + 40.0

        painter.setFont(QFont("Consolas", 7))
        # Draw input nodes
        for r, bit_idx in enumerate(bit_rev_order):
            ry = start_y + r * row_h
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(start_x - 35.0, ry + 4.0, f"x[{bit_idx}]")

            # Horizontal line through 3 stages (log2(8) = 3)
            painter.setPen(QPen(QColor(51, 65, 85), 1.0))
            painter.drawLine(start_x, ry, start_x + stage_w * 3.5, ry)

        # Stage 1: 2-point butterflies between adjacent pairs (stride 1)
        painter.setPen(QPen(QColor(16, 185, 129), 1.4))
        for p in range(0, 8, 2):
            y1 = start_y + p * row_h
            y2 = start_y + (p + 1) * row_h
            x_b = start_x + stage_w * 0.8
            painter.drawLine(start_x, y1, x_b, y2)
            painter.drawLine(start_x, y2, x_b, y1)

        # Stage 2: 4-point butterflies (stride 2)
        painter.setPen(QPen(QColor(56, 189, 248), 1.4))
        for block in [0, 4]:
            for offset in [0, 1]:
                y1 = start_y + (block + offset) * row_h
                y2 = start_y + (block + offset + 2) * row_h
                x_b1 = start_x + stage_w * 1.0
                x_b2 = start_x + stage_w * 1.9
                painter.drawLine(x_b1, y1, x_b2, y2)
                painter.drawLine(x_b1, y2, x_b2, y1)

        # Stage 3: 8-point butterflies (stride 4)
        painter.setPen(QPen(QColor(168, 85, 247), 1.4))
        for offset in range(4):
            y1 = start_y + offset * row_h
            y2 = start_y + (offset + 4) * row_h
            x_b1 = start_x + stage_w * 2.1
            x_b2 = start_x + stage_w * 3.2
            painter.drawLine(x_b1, y1, x_b2, y2)
            painter.drawLine(x_b1, y2, x_b2, y1)

        # Output labels X[0] to X[7]
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(245, 158, 11)))
        for r in range(8):
            ry = start_y + r * row_h
            painter.drawText(start_x + stage_w * 3.5 + 6.0, ry + 4.0, f"X[{r}]")

        # Twiddle factor badge
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(168, 85, 247)))
        painter.drawText(bx + 12.0, by + bh - 8.0, "Thừa số quay: W_N^k = e^(-i 2πk/N) | Đảo bit đầu vào: Decimation-in-Time (DIT)")

    def _paint_analytical_metrics(self, painter: QPainter) -> None:
        """Renders mathematical metrics, Parseval's theorem energy, and Nyquist status."""
        mx = 475.0
        my = 315.0
        mw = 455.0
        mh = 245.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(mx + 12.0, my + 22.0, "THÔNG SỐ TOÁN HỌC & ĐỊNH LÝ PARSEVAL:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(226, 232, 240)))

        is_nyquist_safe = self._f0_hz <= self.nyquist_frequency_hz
        nyq_status = "THỎA MÃN ĐỊNH LÝ SHANNON (f₀ ≤ f_Nyquist)" if is_nyquist_safe else "VI PHẠM SHANNON → XUẤT HIỆN TẦN SỐ ẢO (ALIASING)"
        nyq_col = QColor(16, 185, 129) if is_nyquist_safe else QColor(239, 68, 68)

        energy_diff = abs(self._parseval_time_energy - self._parseval_freq_energy)

        lines = [
            f"• Tần số cơ bản f₀: {self._f0_hz:.1f} Hz | Chu kỳ T₀: {1000.0 / self._f0_hz:.2f} ms",
            f"• Tốc độ lấy mẫu f_s: {self._f_sample_hz:.0f} Hz (Khoảng lấy mẫu Δt = {1000.0 / self._f_sample_hz:.3f} ms)",
            f"• Tần số giới hạn Nyquist: f_Nyquist = f_s / 2 = {self.nyquist_frequency_hz:.1f} Hz",
            f"• Độ phân giải tần số: Δf = f_s / N = {self.frequency_resolution_hz:.2f} Hz / bin",
            f"• Đỉnh phổ lớn nhất phát hiện: {self._peak_freq_hz:.1f} Hz",
            f"• Định lý Bảo toàn Năng lượng Parseval:",
            f"   - Năng lượng miền thời gian Σ |x[n]|² = {self._parseval_time_energy:.4f}",
            f"   - Năng lượng miền tần số (1/N) Σ |X[k]|² = {self._parseval_freq_energy:.4f} (Sai số = {energy_diff:.2e})",
        ]
        curr_y = my + 42.0
        for line in lines:
            painter.drawText(mx + 12.0, curr_y, line)
            curr_y += 17.0

        # Nyquist status badge
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(nyq_col))
        painter.drawText(mx + 12.0, curr_y + 4.0, nyq_status)

        # Description note
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        desc_rect = QRectF(mx + 12.0, curr_y + 18.0, mw - 24.0, 36.0)
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        self._layout_buttons()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Preset Selector
        self._draw_btn(painter, self._buttons["btn_preset"], "btn_preset", f"Preset: {self._preset.badge_label}")

        # 2. Fundamental Frequency Steppers
        self._draw_btn(painter, self._buttons["btn_f_down"], "btn_f_down", "-f₀")
        self._draw_btn(painter, self._buttons["btn_f_up"], "btn_f_up", "+f₀")

        # 3. Sampling Rate Steppers
        self._draw_btn(painter, self._buttons["btn_fs_down"], "btn_fs_down", "-f_s")
        self._draw_btn(painter, self._buttons["btn_fs_up"], "btn_fs_up", "+f_s")

        # 4. Quick Presets
        self._draw_btn(painter, self._buttons["btn_p_square"], "btn_p_square", "Sóng Vuông")
        self._draw_btn(painter, self._buttons["btn_p_nyquist"], "btn_p_nyquist", "Chồng Phổ")
        self._draw_btn(painter, self._buttons["btn_p_dtmf"], "btn_p_dtmf", "DTMF 2-Tone")
        self._draw_btn(painter, self._buttons["btn_p_noise"], "btn_p_noise", "Lọc Nhiễu")

        # 5. Play / Pause
        play_label = "⏸ Dừng" if self._is_running else "▶ Tiếp"
        self._draw_btn(painter, self._buttons["btn_play"], "btn_play", play_label)

        # 6. Stamp Diagram
        self._draw_btn(painter, self._buttons["btn_stamp"], "btn_stamp", "📌 Chèn Vector")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, text: str) -> None:
        is_hovered = self._hovered_btn == key
        bg_col = QColor(30, 41, 59, 220) if not is_hovered else QColor(51, 65, 85, 240)
        border_col = QColor(56, 189, 248) if is_hovered else QColor(71, 85, 105)

        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.2))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor(248, 250, 252) if is_hovered else QColor(203, 213, 225)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # Mouse fallbacks
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        new_hover = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                new_hover = key
                break
        if new_hover != self._hovered_btn:
            self._hovered_btn = new_hover
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._handle_button_click(key)
                event.accept()
                return
        super().mousePressEvent(event)
