"""
AirTeach AI — Interactive Electrodynamics & AC Circuit Analysis:
RLC Resonant Circuit, Phasor Diagram, Quality Factor Q & Frequency Response Simulator.
Milestone 151 (Task I.138).

Provides real-time interactive simulation of:
1. Series RLC AC circuit impedance:
   Z = R + j * (omega * L - 1 / (omega * C)) = |Z| * exp(j * phi)
   |Z| = sqrt(R^2 + (X_L - X_C)^2), phi = arctan((X_L - X_C) / R)
2. Resonant frequency: f_0 = 1 / (2 * pi * sqrt(L * C)), where X_L = X_C and |Z| = R (minimum).
3. Quality factor: Q = (omega_0 * L) / R = (1 / R) * sqrt(L / C).
4. Half-power (-3dB) bandwidth: Delta f = f_0 / Q = R / (2 * pi * L).
5. 2D rotating phasor diagram in the complex plane: V_R (0°), V_L (+90°), V_C (-90°), and V_0.
6. Sinusoidal dual-trace AC oscilloscope: v(t) = V_0 * cos(omega * t) vs i(t) = I_0 * cos(omega * t - phi).
7. 5 Authentic Curriculum Presets: AM radio tuner, Hi-Fi audio crossover, Tesla coil tank,
   Mains 50Hz power factor correction, and Critically damped pulse circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsObject,
    QGraphicsItem,
)


@dataclass(frozen=True)
class RLCPreset:
    """Represents a curriculum AC circuit scenario."""
    code: str
    title_vi: str
    circuit_name: str
    r_ohms: float                 # Resistance R in Ohms
    l_henries: float              # Inductance L in Henries
    c_farads: float               # Capacitance C in Farads
    v0_volts: float               # Source voltage amplitude V_0 in Volts
    accent_color: QColor          # UI theme color
    description: str


RLC_PRESETS: dict[str, RLCPreset] = {
    "AM_RADIO_TUNER_540KHZ_HIGH_Q": RLCPreset(
        code="AM_RADIO_TUNER_540KHZ_HIGH_Q",
        title_vi="Bộ Thu Radio AM 540 kHz: Hệ Số Q Cao Chọn Lọc",
        circuit_name="AM Receiver Antenna Tuner (540 kHz)",
        r_ohms=10.0,
        l_henries=220.0e-6,        # 220 uH
        c_farads=395.7e-12,        # 395.7 pF
        v0_volts=1.0,              # 1 V RF signal
        accent_color=QColor(0, 210, 255), # Cyan
        description="Mạch cộng hưởng chọn sóng radio AM; Q cao (~74.6) giúp chọn lọc chính xác tần số đài phát thanh và lọc bỏ nhiễu lân cận.",
    ),
    "AUDIO_CROSSOVER_LOWPASS_SUBWOOFER": RLCPreset(
        code="AUDIO_CROSSOVER_LOWPASS_SUBWOOFER",
        title_vi="Bộ Phân Tần Âm Thanh Loa Siêu Trầm Subwoofer (120 Hz)",
        circuit_name="Hi-Fi 2nd-Order Audio Crossover (120 Hz)",
        r_ohms=4.0,                # 4 Ohm loudspeaker impedance
        l_henries=10.6e-3,         # 10.6 mH
        c_farads=165.8e-6,         # 165.8 uF
        v0_volts=10.0,             # 10 V audio signal
        accent_color=QColor(255, 140, 0), # Deep Orange
        description="Bộ lọc phân tần âm thanh cho loa trầm; đáp tuyến tần số Butterworth phẳng với Q ~ 2.0 chống méo tiếng.",
    ),
    "TESLA_COIL_PRIMARY_TANK_250KHZ": RLCPreset(
        code="TESLA_COIL_PRIMARY_TANK_250KHZ",
        title_vi="Mạch Sơ Cấp Cuộn Dây Tesla (250 kHz Khuyếch Đại Điện Áp)",
        circuit_name="Tesla Coil Resonant Tank (250 kHz)",
        r_ohms=1.5,
        l_henries=50.0e-6,         # 50 uH
        c_farads=8.1e-9,           # 8.1 nF
        v0_volts=100.0,            # 100 V pulsed AC
        accent_color=QColor(180, 80, 255), # Tesla Violet
        description="Cộng hưởng điện thế khổng lồ; tại f0 điện áp trên tụ điện và cuộn cảm phóng đại gấp Q lần (V_C = Q * V0) sinh tia lửa điện cao tần.",
    ),
    "MAINS_POWER_FACTOR_CORRECTION_50HZ": RLCPreset(
        code="MAINS_POWER_FACTOR_CORRECTION_50HZ",
        title_vi="Bù Hệ Số Công Suất Lưới Điện 50 Hz (Power Factor cosφ = 1)",
        circuit_name="Mains 50Hz Industrial Grid Filter",
        r_ohms=50.0,
        l_henries=500.0e-3,        # 500 mH
        c_farads=20.26e-6,         # 20.26 uF
        v0_volts=220.0,            # 220 V RMS / 311 V peak
        accent_color=QColor(0, 230, 160), # Grid Teal
        description="Bù công suất phản kháng trong công nghiệp; tại cộng hưởng dòng điện cùng pha với điện áp (cos phi = 1) tối ưu hóa hiệu suất truyền tải.",
    ),
    "CRITICALLY_DAMPED_PULSE_CIRCUIT": RLCPreset(
        code="CRITICALLY_DAMPED_PULSE_CIRCUIT",
        title_vi="Mạch Tắt Dần Tới Hạn (Critically Damped Circuit ζ = 1)",
        circuit_name="Critically Damped Pulse Shaper",
        r_ohms=157.0,              # R = 2 * sqrt(L / C) = 157 Ohm
        l_henries=10.0e-3,         # 10 mH
        c_farads=1.62e-6,          # 1.62 uF
        v0_volts=12.0,             # 12 V
        accent_color=QColor(239, 71, 111), # Coral Red
        description="Mạch tắt dần nhanh nhất mà không có dao động sóng dội ngược (R = 2*sqrt(L/C)), ứng dụng tạo xung chuẩn trong viễn thông.",
    ),
}


class InteractiveRLCResonanceItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating AC RLC Resonant Circuits,
    complex impedance, phasor diagrams, Bode frequency response, and oscilloscope waveforms.
    """

    # Signals
    rlc_computed = Signal(dict)
    stamp_requested = Signal(dict)

    CARD_W: float = 680.0
    CARD_H: float = 410.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "AM_RADIO_TUNER_540KHZ_HIGH_Q"
        self._preset: RLCPreset = RLC_PRESETS[self._active_preset_code]

        self._r_ohms: float = self._preset.r_ohms
        self._l_henries: float = self._preset.l_henries
        self._c_farads: float = self._preset.c_farads
        self._v0_volts: float = self._preset.v0_volts

        # Current operating frequency f (Hz). Default to resonant frequency f0.
        self._freq_hz: float = self.resonant_frequency_hz

        # Scope animation time
        self._time_sec: float = 0.0

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None

        # Timer for 60fps real-time AC waveform animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    # ------------------------------------------------------------------
    # Physics Properties & Analytical AC Circuit Laws
    # ------------------------------------------------------------------

    @property
    def resonant_angular_freq_rad_s(self) -> float:
        """Resonant angular frequency omega_0 = 1 / sqrt(L * C)."""
        lc = max(1e-25, self._l_henries * self._c_farads)
        return 1.0 / math.sqrt(lc)

    @property
    def resonant_frequency_hz(self) -> float:
        """Resonant frequency f_0 = 1 / (2 * pi * sqrt(L * C))."""
        return self.resonant_angular_freq_rad_s / (2.0 * math.pi)

    @property
    def operating_angular_freq_rad_s(self) -> float:
        """Current operating angular frequency omega = 2 * pi * f."""
        return 2.0 * math.pi * max(1e-3, self._freq_hz)

    @property
    def inductive_reactance_ohms(self) -> float:
        """Inductive reactance X_L = omega * L."""
        return self.operating_angular_freq_rad_s * self._l_henries

    @property
    def capacitive_reactance_ohms(self) -> float:
        """Capacitive reactance X_C = 1 / (omega * C)."""
        denom = self.operating_angular_freq_rad_s * self._c_farads
        return 1.0 / denom if denom > 0.0 else 1e9

    @property
    def net_reactance_ohms(self) -> float:
        """Net reactance X = X_L - X_C."""
        return self.inductive_reactance_ohms - self.capacitive_reactance_ohms

    @property
    def impedance_magnitude_ohms(self) -> float:
        """Total impedance magnitude |Z| = sqrt(R^2 + (X_L - X_C)^2)."""
        r = max(1e-4, self._r_ohms)
        x = self.net_reactance_ohms
        return math.sqrt(r * r + x * x)

    @property
    def phase_angle_rad(self) -> float:
        """Phase angle phi = arctan((X_L - X_C) / R)."""
        r = max(1e-4, self._r_ohms)
        return math.atan2(self.net_reactance_ohms, r)

    @property
    def phase_angle_deg(self) -> float:
        """Phase angle phi in degrees."""
        return math.degrees(self.phase_angle_rad)

    @property
    def current_amplitude_amps(self) -> float:
        """Current amplitude I_0 = V_0 / |Z|."""
        z = self.impedance_magnitude_ohms
        return self._v0_volts / z if z > 0.0 else 0.0

    @property
    def quality_factor_Q(self) -> float:
        """Quality factor Q = (omega_0 * L) / R."""
        r = max(1e-4, self._r_ohms)
        return (self.resonant_angular_freq_rad_s * self._l_henries) / r

    @property
    def bandwidth_hz(self) -> float:
        """Half-power (-3dB) bandwidth Delta f = f_0 / Q = R / (2 * pi * L)."""
        q = max(1e-4, self.quality_factor_Q)
        return self.resonant_frequency_hz / q

    @property
    def vr_amplitude_volts(self) -> float:
        """Voltage amplitude across resistor V_R = I_0 * R."""
        return self.current_amplitude_amps * self._r_ohms

    @property
    def vl_amplitude_volts(self) -> float:
        """Voltage amplitude across inductor V_L = I_0 * X_L."""
        return self.current_amplitude_amps * self.inductive_reactance_ohms

    @property
    def vc_amplitude_volts(self) -> float:
        """Voltage amplitude across capacitor V_C = I_0 * X_C."""
        return self.current_amplitude_amps * self.capacitive_reactance_ohms

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches curriculum scenario."""
        if preset_code in RLC_PRESETS:
            self._active_preset_code = preset_code
            self._preset = RLC_PRESETS[preset_code]
            self._r_ohms = self._preset.r_ohms
            self._l_henries = self._preset.l_henries
            self._c_farads = self._preset.c_farads
            self._v0_volts = self._preset.v0_volts
            self._freq_hz = self.resonant_frequency_hz
            self._emit_computed_values()
            self.update()

    def set_frequency_hz(self, freq: float) -> None:
        """Sets operating frequency f in Hz."""
        self._freq_hz = max(1.0, min(100.0e6, freq))
        self._emit_computed_values()
        self.update()

    def tune_to_resonance(self) -> None:
        """Directly sets operating frequency to exact resonance f_0."""
        self.set_frequency_hz(self.resonant_frequency_hz)

    def set_resistance_ohms(self, r_val: float) -> None:
        """Sets circuit resistance R in Ohms."""
        self._r_ohms = max(0.1, min(10000.0, r_val))
        self._emit_computed_values()
        self.update()

    def reset_defaults(self) -> None:
        """Restores default values of current preset."""
        self.set_preset(self._active_preset_code)

    def _emit_computed_values(self) -> None:
        """Emits real-time computed AC telemetry."""
        data = {
            "preset_code": self._active_preset_code,
            "circuit_name": self._preset.circuit_name,
            "f0_hz": self.resonant_frequency_hz,
            "freq_hz": self._freq_hz,
            "r_ohms": self._r_ohms,
            "l_henries": self._l_henries,
            "c_farads": self._c_farads,
            "z_ohms": self.impedance_magnitude_ohms,
            "i0_amps": self.current_amplitude_amps,
            "phi_deg": self.phase_angle_deg,
            "quality_factor_Q": self.quality_factor_Q,
            "bandwidth_hz": self.bandwidth_hz,
        }
        self.rlc_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Animation tick for AC oscilloscope waveforms and rotating phasor."""
        self._time_sec += 0.03
        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        """Arranges interactive control buttons in bottom panel."""
        ctrl_x = 385.0
        row1_y = 248.0
        btn_w_half = 72.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["tune_res"] = QRectF(ctrl_x + 104.0, row1_y, 172.0, btn_h)

        row2_y = 278.0
        self._buttons["f_dec"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["f_inc"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["f_dec_fine"] = QRectF(ctrl_x + 158.0, row2_y, 56.0, btn_h)
        self._buttons["f_inc_fine"] = QRectF(ctrl_x + 220.0, row2_y, 56.0, btn_h)

        row3_y = 308.0
        self._buttons["r_dec"] = QRectF(ctrl_x, row3_y, btn_w_half, btn_h)
        self._buttons["r_inc"] = QRectF(ctrl_x + 78.0, row3_y, btn_w_half, btn_h)
        self._buttons["q_high"] = QRectF(ctrl_x + 158.0, row3_y, 118.0, btn_h)

        row4_y = 342.0
        self._buttons["reset"] = QRectF(ctrl_x, row4_y, 100.0, 26.0)
        self._buttons["stamp"] = QRectF(ctrl_x + 108.0, row4_y, 168.0, 26.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Rendering Pipeline
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option=None, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        self._paint_card_background(painter)

        # 2. Header and Title Banner
        self._paint_header(painter)

        # 3. Left Side: Phasor Diagram & Dual-Trace Oscilloscope
        self._paint_phasor_and_scope(painter)

        # 4. Right Side: Bode Resonance Curve & AC Telemetry
        self._paint_resonance_curve_and_telemetry(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "⚡ CỘNG HƯỞNG RLC & GIẢN ĐỒ VÉC-TƠ FREZNEL (PHASOR)")

        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        phi = self.phase_angle_deg
        state_txt = "Cộng hưởng thuần trở (φ = 0°)" if abs(phi) < 0.5 else ("Tính cảm kháng (u sớm pha hơn i)" if phi > 0 else "Tính dung kháng (i sớm pha hơn u)")
        painter.drawText(20, 44, f"Mạch: {self._preset.circuit_name} | Trạng thái: {state_txt}")

        # Badge (Top Right)
        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        accent = self._preset.accent_color
        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        f0 = self.resonant_frequency_hz
        f0_str = f"{f0*1e-3:.1f} kHz" if f0 >= 1000.0 else f"{f0:.1f} Hz"
        badge_txt = f"f₀ = {f0_str} • Q = {self.quality_factor_Q:.1f}"
        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(accent)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_phasor_and_scope(self, painter: QPainter) -> None:
        """
        Renders:
        - 2D Phasor diagram in complex plane (VR, VL, VC, V0)
        - Dual-trace oscilloscope showing v(t) and i(t) with phase shift phi
        """
        box = QRectF(20.0, 56.0, 345.0, 180.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.2))
        painter.setBrush(QBrush(QColor(8, 12, 20, 220)))
        painter.drawRoundedRect(box, 8.0, 8.0)

        # 1. Phasor Diagram (Left half of box: cx = 95, cy = 145)
        cx = 100.0
        cy = 145.0

        # Axes
        painter.setPen(QPen(QColor(50, 68, 95), 1.0))
        painter.drawLine(cx - 65.0, cy, cx + 65.0, cy) # Real axis (I reference)
        painter.drawLine(cx, cy - 65.0, cx, cy + 65.0) # Imaginary axis

        # Phasor lengths normalized
        vr_val = self.vr_amplitude_volts
        vl_val = self.vl_amplitude_volts
        vc_val = self.vc_amplitude_volts
        v0_val = max(1e-3, self._v0_volts)

        scale = 45.0 / v0_val
        vr_px = min(60.0, vr_val * scale)
        vl_px = min(60.0, vl_val * scale)
        vc_px = min(60.0, vc_val * scale)

        # VR phasor (Green, along +X real axis)
        painter.setPen(QPen(QColor("#4ade80"), 2.2))
        painter.drawLine(cx, cy, cx + vr_px, cy)

        # VL phasor (Purple, along +Y imag axis upwards)
        painter.setPen(QPen(QColor("#c084fc"), 2.0))
        painter.drawLine(cx, cy, cx, cy - vl_px)

        # VC phasor (Cyan, along -Y imag axis downwards)
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawLine(cx, cy, cx, cy + vc_px)

        # V0 Resultant phasor (Gold, at angle -phi from horizontal)
        phi_rad = self.phase_angle_rad
        v0_px = min(60.0, v0_val * scale)
        end_x = cx + v0_px * math.cos(phi_rad)
        end_y = cy - v0_px * math.sin(phi_rad)
        painter.setPen(QPen(QColor("#fbbf24"), 2.5))
        painter.drawLine(cx, cy, end_x, end_y)

        # Label phasor elements
        font_p = QFont("Segoe UI", 6, QFont.Weight.Bold)
        painter.setFont(font_p)
        painter.setPen(QColor("#4ade80"))
        painter.drawText(int(cx + vr_px + 2.0), int(cy - 2.0), "V_R")

        painter.setPen(QColor("#c084fc"))
        painter.drawText(int(cx + 4.0), int(cy - vl_px + 8.0), "V_L")

        painter.setPen(QColor("#38bdf8"))
        painter.drawText(int(cx + 4.0), int(cy + vc_px - 2.0), "V_C")

        painter.setPen(QColor("#fbbf24"))
        painter.drawText(int(end_x + 4.0), int(end_y - 2.0), f"V₀ (φ={self.phase_angle_deg:+.1f}°)")

        # 2. Dual-trace AC Oscilloscope (Right half of box: x = 180 to 350)
        sc_x0 = 190.0
        sc_y0 = cy
        sc_w = 160.0
        sc_h = 45.0

        # Scope zero line
        painter.setPen(QPen(QColor(50, 68, 95), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(sc_x0, sc_y0, sc_x0 + sc_w, sc_y0)

        # Draw v(t) (Gold trace) and i(t) (Cyan trace)
        path_v = QPainterPath()
        path_i = QPainterPath()

        omega_anim = 6.0
        phi_scope = -self.phase_angle_rad

        for i in range(int(sc_w)):
            frac = i / sc_w
            angle = frac * 4.0 * math.pi + self._time_sec * omega_anim

            y_v = sc_y0 - math.cos(angle) * (sc_h * 0.8)
            y_i = sc_y0 - math.cos(angle - phi_scope) * (sc_h * 0.75 * (self.current_amplitude_amps / max(1e-4, (self._v0_volts / self._r_ohms))))

            pt_v = QPointF(sc_x0 + i, y_v)
            pt_i = QPointF(sc_x0 + i, y_i)

            if i == 0:
                path_v.moveTo(pt_v)
                path_i.moveTo(pt_i)
            else:
                path_v.lineTo(pt_v)
                path_i.lineTo(pt_i)

        painter.setPen(QPen(QColor("#fbbf24"), 1.6))
        painter.drawPath(path_v)

        painter.setPen(QPen(QColor("#38bdf8"), 1.6))
        painter.drawPath(path_i)

        # Oscilloscope labels
        painter.setFont(font_p)
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(int(sc_x0), int(box.top() + 18.0), "v(t) — Điện áp nguồn (Gold)")
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(int(sc_x0), int(box.top() + 30.0), "i(t) — Dòng điện mạch (Cyan)")

        # Telemetry panel below box (y = 245 to 395)
        self._paint_telemetry_table(painter)

    def _paint_telemetry_table(self, painter: QPainter) -> None:
        """Renders exact electrical parameters table."""
        t_box = QRectF(20.0, 245.0, 345.0, 150.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(10, 15, 25, 220)))
        painter.drawRoundedRect(t_box, 8.0, 8.0)

        font_h = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_h)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(t_box.left() + 12.0), int(t_box.top() + 18.0), "⚡ THÔNG SỐ ĐIỆN XOAY CHIỀU & CỘNG HƯỞNG")

        font_k = QFont("Segoe UI", 7, QFont.Weight.Normal)
        font_v = QFont("Consolas", 8, QFont.Weight.Bold)

        f0 = self.resonant_frequency_hz
        f_cur = self._freq_hz
        z_mag = self.impedance_magnitude_ohms
        i_amp = self.current_amplitude_amps

        f0_str = f"{f0*1e-3:.2f} kHz" if f0 >= 1000.0 else f"{f0:.2f} Hz"
        f_str = f"{f_cur*1e-3:.2f} kHz" if f_cur >= 1000.0 else f"{f_cur:.2f} Hz"
        i_str = f"{i_amp*1e3:.2f} mA" if i_amp < 1.0 else f"{i_amp:.3f} A"

        lines = [
            ("Tần số cộng hưởng f₀:", f0_str, QColor("#fbbf24")),
            ("Tần số hoạt động f:", f_str, QColor("#38bdf8")),
            ("Tổng trở |Z|:", f"{z_mag:.2f} Ω (R = {self._r_ohms:.1f} Ω)", QColor("#4ade80")),
            ("Cường độ dòng cực đại I₀:", i_str, QColor("#22d3ee")),
            ("Độ lệch pha φ (u so với i):", f"{self.phase_angle_deg:+.2f}° ({self.phase_angle_rad:+.3f} rad)", QColor("#f43f5e") if abs(self.phase_angle_deg) > 1.0 else QColor("#4ade80")),
        ]

        curr_y = t_box.top() + 38.0
        for lbl, val_str, val_col in lines:
            painter.setFont(font_k)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(t_box.left() + 12.0), int(curr_y), lbl)

            painter.setFont(font_v)
            painter.setPen(val_col)
            painter.drawText(int(t_box.left() + 185.0), int(curr_y), val_str)
            curr_y += 22.0

    def _paint_resonance_curve_and_telemetry(self, painter: QPainter) -> None:
        """Renders Bode resonance curve of current amplitude I(f) vs frequency f."""
        plot_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(plot_rect, 8.0, 8.0)

        # Plot axes
        ax_x0 = plot_rect.left() + 35.0
        ax_y0 = plot_rect.bottom() - 25.0
        ax_w = plot_rect.width() - 50.0
        ax_h = plot_rect.height() - 50.0

        painter.setPen(QPen(QColor(50, 68, 95), 1.0))
        painter.drawLine(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0) # f axis
        painter.drawLine(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h) # I axis

        font_ax = QFont("Segoe UI", 6, QFont.Weight.Normal)
        painter.setFont(font_ax)
        painter.setPen(QColor(130, 150, 180))
        painter.drawText(int(ax_x0 + ax_w - 40.0), int(ax_y0 + 15.0), "f (Hz) →")
        painter.drawText(int(ax_x0 - 25.0), int(ax_y0 - ax_h + 8.0), "I(f)")

        # Peak current at resonance
        f0 = self.resonant_frequency_hz
        i_max = self._v0_volts / self._r_ohms

        # Plot domain: f from 0.2*f0 to 1.8*f0
        f_min = max(1.0, f0 * 0.2)
        f_max = f0 * 1.8

        curve_path = QPainterPath()
        points: list[QPointF] = []
        n_steps = 70

        for i in range(n_steps + 1):
            f_eval = f_min + (i / n_steps) * (f_max - f_min)
            w_eval = 2.0 * math.pi * f_eval
            xl_eval = w_eval * self._l_henries
            xc_eval = 1.0 / (w_eval * self._c_farads) if w_eval * self._c_farads > 0 else 1e9
            z_eval = math.sqrt(self._r_ohms ** 2 + (xl_eval - xc_eval) ** 2)
            i_eval = self._v0_volts / z_eval if z_eval > 0 else 0.0

            px = ax_x0 + ((f_eval - f_min) / (f_max - f_min)) * ax_w
            py = ax_y0 - (i_eval / (i_max * 1.15)) * ax_h
            points.append(QPointF(px, py))

        if points:
            curve_path.moveTo(points[0])
            for pt in points[1:]:
                curve_path.lineTo(pt)

            # Area under curve
            area_path = QPainterPath(curve_path)
            area_path.lineTo(points[-1].x(), ax_y0)
            area_path.lineTo(points[0].x(), ax_y0)
            area_path.closeSubpath()

            grad = QLinearGradient(0.0, ax_y0 - ax_h, 0.0, ax_y0)
            accent = self._preset.accent_color
            grad.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 120))
            grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 20))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawPath(area_path)

            painter.setPen(QPen(accent, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(curve_path)

        # Mark resonant frequency line f0 (Gold dashed)
        if f_min <= f0 <= f_max:
            f0_x = ax_x0 + ((f0 - f_min) / (f_max - f_min)) * ax_w
            painter.setPen(QPen(QColor("#fbbf24"), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(f0_x, ax_y0, f0_x, ax_y0 - ax_h * 0.95)

            font_m = QFont("Segoe UI", 6, QFont.Weight.Bold)
            painter.setFont(font_m)
            painter.setPen(QColor("#fbbf24"))
            painter.drawText(int(f0_x - 6.0), int(ax_y0 - ax_h * 0.98), "f₀")

        # Mark current operating frequency cursor (Cyan dot)
        if f_min <= self._freq_hz <= f_max:
            cur_x = ax_x0 + ((self._freq_hz - f_min) / (f_max - f_min)) * ax_w
            cur_y = ax_y0 - (self.current_amplitude_amps / (i_max * 1.15)) * ax_h
            painter.setPen(QPen(QColor(255, 255, 255), 1.2))
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.drawEllipse(QPointF(cur_x, cur_y), 3.5, 3.5)

    def _paint_buttons(self, painter: QPainter) -> None:
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "tune_res": "🎯 Chỉnh f = f₀ (Cộng hưởng)",
            "f_dec": "f: −10%",
            "f_inc": "f: +10%",
            "f_dec_fine": "f: −1%",
            "f_inc_fine": "f: +1%",
            "r_dec": "R: −20%",
            "r_inc": "R: +20%",
            "q_high": "Tăng độ chọn Q",
            "reset": "🔄 Mặc định",
            "stamp": "📌 Đóng dấu bài học",
        }

        font_btn = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_btn)

        for btn_key, rect in self._buttons.items():
            is_hover = (self._hovered_btn == btn_key)
            if btn_key == "stamp":
                border_col = QColor(0, 230, 180) if is_hover else QColor(0, 180, 140)
                fill_col = QColor(0, 140, 100, 180) if is_hover else QColor(0, 100, 75, 150)
                txt_col = QColor(255, 255, 255)
            elif btn_key == "reset":
                border_col = QColor(220, 100, 100) if is_hover else QColor(160, 70, 70)
                fill_col = QColor(140, 50, 50, 160) if is_hover else QColor(90, 35, 35, 140)
                txt_col = QColor(255, 230, 230)
            elif btn_key == "tune_res":
                border_col = QColor(240, 180, 50) if is_hover else QColor(190, 140, 30)
                fill_col = QColor(160, 110, 20, 160) if is_hover else QColor(100, 70, 15, 140)
                txt_col = QColor(255, 250, 220)
            else:
                border_col = QColor(80, 140, 220) if is_hover else QColor(45, 75, 120)
                fill_col = QColor(35, 60, 95, 180) if is_hover else QColor(22, 38, 62, 160)
                txt_col = QColor(225, 238, 255)

            painter.setPen(QPen(border_col, 1.0))
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(txt_col)
            label = button_labels.get(btn_key, btn_key)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_hover = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_hover != self._hovered_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _handle_button_click(self, btn_key: str) -> None:
        preset_keys = list(RLC_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "tune_res":
            self.tune_to_resonance()
        elif btn_key == "f_dec":
            self.set_frequency_hz(self._freq_hz * 0.90)
        elif btn_key == "f_inc":
            self.set_frequency_hz(self._freq_hz * 1.10)
        elif btn_key == "f_dec_fine":
            self.set_frequency_hz(self._freq_hz * 0.99)
        elif btn_key == "f_inc_fine":
            self.set_frequency_hz(self._freq_hz * 1.01)
        elif btn_key == "r_dec":
            self.set_resistance_ohms(self._r_ohms * 0.80)
        elif btn_key == "r_inc":
            self.set_resistance_ohms(self._r_ohms * 1.20)
        elif btn_key == "q_high":
            # Halve R to double Q
            self.set_resistance_ohms(self._r_ohms * 0.50)
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "circuit_name": self._preset.circuit_name,
            "f0_hz": self.resonant_frequency_hz,
            "freq_hz": self._freq_hz,
            "r_ohms": self._r_ohms,
            "l_henries": self._l_henries,
            "c_farads": self._c_farads,
            "z_ohms": self.impedance_magnitude_ohms,
            "i0_amps": self.current_amplitude_amps,
            "phi_deg": self.phase_angle_deg,
            "quality_factor_Q": self.quality_factor_Q,
            "bandwidth_hz": self.bandwidth_hz,
        }
        self.stamp_requested.emit(data)
