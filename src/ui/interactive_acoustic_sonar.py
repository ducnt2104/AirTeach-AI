"""
AirTeach AI - Interactive Acoustic Wave Physics: Doppler Spectrogram, Ultrasonic Echolocation & Sonar Beamforming
===================================================================================================================
Authentic acoustics and wave physics simulation based on classical wave mechanics:
- Sound Propagation in Fluid Medium:
    c = 343.0 m/s (Standard air at 20°C, 1 atm, adiabatic index γ = 1.4, R = 287 J/(kg·K))
- Classical Doppler Effect for Sound Waves:
    f_o = f_s * (c - v_o · u) / (c - v_s · u)
    where:
      f_s: Source emission frequency (Hz)
      f_o: Observed frequency (Hz)
      v_s: Source velocity vector (m/s)
      v_o: Observer velocity vector (m/s, default stationary)
      u: Unit direction vector from source to observer
    - Approach: v_s · u > 0 => f_o > f_s (Higher perceived frequency / blue shift)
    - Recede:   v_s · u < 0 => f_o < f_s (Lower perceived frequency / red shift)
    - Closest Approach: v_s · u = 0 => f_o = f_s with maximum frequency drop rate df_o/dt
- Supersonic Flow & Mach Cone Formation:
    Mach number M = v_s / c
    When M > 1.0, wavefronts cannot propagate ahead of source.
    Mach cone half-angle:
      μ = arcsin(1 / M) = arcsin(c / v_s)
    Envelope forms conical shockwave discontinuity (sonic boom / tiếng nổ siêu thanh).
- Ultrasonic Echolocation (Time-of-Flight / TOF):
    Pulse round-trip propagation time:
      Δt = 2 * d / c  =>  d = c * Δt / 2
    Obstacle reflection wave propagation and echo pulse reception.
- Phased Array Sonar / Ultrasound Beamforming:
    Linear array of N elements spaced by pitch d = λ / 2.
    Steering angle θ_0 ∈ [-60°, +60°].
    Array Directivity Factor AF(θ):
      AF(θ) = | sin(N * π * (d/λ) * (sin θ - sin θ_0)) / (N * sin(π * (d/λ) * (sin θ - sin θ_0))) |
    Direct acoustic energy radiation along steerable main beam lobe.
- Real-Time Time-Frequency Waterfall Spectrogram:
    Scrolling spectro-temporal heatmap recording historical f_o(t) signature (Doppler S-curve).
- 5 Curriculum Presets:
    1. AMBULANCE_DOPPLER: Classical emergency siren approach, pass-by, and recession.
    2. SUPERSONIC_MACH_CONE: Jet fighter breaking sound barrier (M = 1.40, Mach shock cone).
    3. BAT_ECHOLOCATION: Biosonar ultrasonic ping, obstacle reflection, and TOF distance calculation.
    4. SONAR_BEAMFORMING: Phased array acoustic steering and radiation directivity pattern.
    5. DOPPLER_SPECTROGRAM: Multi-harmonic engine flyby with waterfall spectrogram S-curve.
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Speed of sound in standard dry air at 20°C (m/s)
SPEED_OF_SOUND_AIR: float = 343.0


@dataclass(frozen=True)
class AcousticPreset:
    code: str
    name_vi: str
    source_velocity_m_s: float
    source_freq_hz: float
    mode: str  # "doppler", "mach", "echolocation", "beamforming", "spectrogram"
    beam_angle_deg: float
    obstacle_distance_m: float
    description: str


PRESETS: dict[str, AcousticPreset] = {
    "AMBULANCE_DOPPLER": AcousticPreset(
        code="AMBULANCE_DOPPLER",
        name_vi="1. Còi Cấp cứu & Hiệu ứng Doppler",
        source_velocity_m_s=34.3,  # 0.10 Mach (~123.5 km/h)
        source_freq_hz=700.0,
        mode="doppler",
        beam_angle_deg=0.0,
        obstacle_distance_m=10.0,
        description="Nguồn âm di chuyển vận tốc v = 34.3 m/s. Sóng âm bị dồn ép phía trước (tần số cao) và dãn ra phía sau (tần số thấp).",
    ),
    "SUPERSONIC_MACH_CONE": AcousticPreset(
        code="SUPERSONIC_MACH_CONE",
        name_vi="2. Máy bay Siêu thanh & Nón Sóng Mach",
        source_velocity_m_s=480.2,  # 1.40 Mach (~1729 km/h)
        source_freq_hz=400.0,
        mode="mach",
        beam_angle_deg=0.0,
        obstacle_distance_m=10.0,
        description="Máy bay bay nhanh hơn âm thanh (Mach M = 1.40). Sóng âm tạo thành mặt nón sóng xung kích với góc Mach μ = arcsin(1/M) = 45.6°.",
    ),
    "BAT_ECHOLOCATION": AcousticPreset(
        code="BAT_ECHOLOCATION",
        name_vi="3. Định vị Tiếng vang Siêu âm (Echolocation)",
        source_velocity_m_s=0.0,
        source_freq_hz=40000.0,  # 40 kHz ultrasound
        mode="echolocation",
        beam_angle_deg=0.0,
        obstacle_distance_m=6.86,  # ~20 ms round-trip TOF
        description="Loài dơi phát xung siêu âm 40 kHz. Đo thời gian hồi tiếp Δt để tính chính xác cự ly vật cản: d = c · Δt / 2.",
    ),
    "SONAR_BEAMFORMING": AcousticPreset(
        code="SONAR_BEAMFORMING",
        name_vi="4. Sonar Phased Array & Quét Chùm tia",
        source_velocity_m_s=0.0,
        source_freq_hz=20000.0,  # 20 kHz sonar
        mode="beamforming",
        beam_angle_deg=30.0,  # Steered 30 degrees
        obstacle_distance_m=10.0,
        description="Mảng 8 đầu dò siêu âm tạo trễ pha k·d·sin(θ_0) để uốn chùm tia định hướng quét mục tiêu dưới nước hoặc trong không khí.",
    ),
    "DOPPLER_SPECTROGRAM": AcousticPreset(
        code="DOPPLER_SPECTROGRAM",
        name_vi="5. Phổ Waterfall Thời gian - Tần số",
        source_velocity_m_s=51.45,  # 0.15 Mach (~185 km/h)
        source_freq_hz=600.0,
        mode="spectrogram",
        beam_angle_deg=0.0,
        obstacle_distance_m=10.0,
        description="Biểu đồ phổ thác nước (Waterfall Spectrogram) ghi lại đường cong Doppler chữ S đặc trưng khi nguồn âm tiến gần và rời xa.",
    ),
}


class InteractiveAcousticSonarItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for Acoustic Wave Physics, Doppler Spectrogram,
    Ultrasonic Echolocation, and Phased Array Sonar Beamforming.
    """

    # Signals
    preset_changed = Signal(str)
    velocity_changed = Signal(float)
    frequency_changed = Signal(float)
    beam_angle_changed = Signal(float)
    closed = Signal()

    def __init__(
        self,
        board: Optional["LessonBoard"] = None,
        pos: tuple[float, float] | QPointF = (80.0, 80.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board

        # Visual dimensions
        self._width: float = 940.0
        self._height: float = 650.0

        # Authentic physics state
        self._c: float = SPEED_OF_SOUND_AIR  # 343 m/s
        self._source_velocity: float = 34.3  # m/s
        self._source_freq: float = 700.0  # Hz
        self._current_preset: str = "AMBULANCE_DOPPLER"
        self._mode: str = "doppler"  # "doppler", "mach", "echolocation", "beamforming", "spectrogram"
        self._beam_angle_deg: float = 0.0  # steering angle in degrees
        self._obstacle_distance_m: float = 10.0  # obstacle range for echolocation
        self._num_array_elements: int = 8  # for phased array sonar

        # Coordinate positions in simulation frame
        # Simulation domain: [-15m, +15m] horizontally, [-10m, +10m] vertically
        self._source_x_m: float = -12.0
        self._source_y_m: float = 0.0
        self._observer_x_m: float = 0.0
        self._observer_y_m: float = 5.0

        # Animation & wave propagation history
        # Each wave pulse: (x_emit, y_emit, t_emit_sec)
        self._sim_time_sec: float = 0.0
        self._is_playing: bool = True
        self._wavefronts: list[dict[str, float]] = []
        self._echo_wavefronts: list[dict[str, float]] = []
        self._emission_interval_sec: float = 0.04  # new wavefront every 40 ms
        self._last_emission_sec: float = 0.0
        self._last_pulse_emit_time: float = 0.0

        # Waterfall Spectrogram buffer: list of (time_sec, observed_freq_hz)
        self._spectrogram_history: list[tuple[float, float]] = []
        self._max_spectrogram_points: int = 140

        # UI interaction state
        self._active_slider: Optional[str] = None  # "velocity", "freq", "beam_angle", "obstacle"
        self._dragging_source: bool = False
        self._dragging_observer: bool = False
        self._hovered_button: Optional[str] = None

        # Air gestures support
        self._air_tap_cooldown: int = 0

        # Flags & setup
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        if isinstance(pos, QPointF):
            self.setPos(pos.x(), pos.y())
        else:
            self.setPos(pos[0], pos[1])

        # Initialize simulation particles & wavefronts
        self._reset_simulation()

    # -------------------------------------------------------------------------
    # Physics Calculations
    # -------------------------------------------------------------------------

    @property
    def mach_number(self) -> float:
        """Calculate Mach number M = v_s / c."""
        return abs(self._source_velocity) / self._c

    @property
    def mach_angle_deg(self) -> Optional[float]:
        """Calculate Mach half-angle μ = arcsin(1/M) in degrees if M >= 1.0."""
        m = self.mach_number
        if m >= 1.0:
            return math.degrees(math.asin(1.0 / m))
        return None

    def calculate_observed_frequency(self) -> float:
        """
        Calculate instantaneous observed frequency f_o at observer position:
        f_o = f_s * (c / (c - v_s · u)) where u is unit vector from source to observer.
        """
        dx = self._observer_x_m - self._source_x_m
        dy = self._observer_y_m - self._source_y_m
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return self._source_freq

        # Unit vector from source towards observer
        ux = dx / dist
        uy = dy / dist

        # Source velocity is along horizontal x-axis: v = (v_s, 0)
        v_parallel = self._source_velocity * ux

        # Relativistic or acoustic singularity guard
        denom = self._c - v_parallel
        if abs(denom) < 1.0:
            denom = 1.0 if denom >= 0 else -1.0

        f_o = self._source_freq * (self._c / denom)
        return max(20.0, min(100000.0, f_o))

    def calculate_array_factor(self, theta_rad: float) -> float:
        """
        Calculate directivity Array Factor AF(θ) for N-element phased array with d = λ/2:
        AF(θ) = | sin(N * π/2 * (sin θ - sin θ_0)) / (N * sin(π/2 * (sin θ - sin θ_0))) |
        """
        n = self._num_array_elements
        theta_0_rad = math.radians(self._beam_angle_deg)
        psi = math.pi * (math.sin(theta_rad) - math.sin(theta_0_rad)) / 2.0

        if abs(psi) < 1e-6:
            return 1.0

        sin_n_psi = math.sin(n * psi)
        sin_psi = math.sin(psi)
        if abs(sin_psi) < 1e-6:
            return 1.0

        val = abs(sin_n_psi / (n * sin_psi))
        return max(0.0, min(1.0, val))

    def calculate_echolocation_tof_ms(self) -> float:
        """Calculate round-trip time of flight Δt = 2*d / c in milliseconds."""
        return (2.0 * self._obstacle_distance_m / self._c) * 1000.0

    # -------------------------------------------------------------------------
    # Simulation Step & Control
    # -------------------------------------------------------------------------

    def _reset_simulation(self) -> None:
        """Reset source position, time, and wavefront buffers."""
        self._sim_time_sec = 0.0
        self._source_x_m = -12.0
        self._source_y_m = 0.0
        self._wavefronts.clear()
        self._echo_wavefronts.clear()
        self._last_emission_sec = 0.0
        self._last_pulse_emit_time = 0.0
        self._spectrogram_history.clear()

    def step_simulation(self, dt_sec: float = 0.033) -> None:
        """Advance acoustic wave simulation by dt_sec."""
        if not self._is_playing:
            return

        self._sim_time_sec += dt_sec

        # Advance moving source if in motion
        if abs(self._source_velocity) > 0.1:
            self._source_x_m += self._source_velocity * dt_sec
            # Loop source when leaving simulation domain (+14m back to -14m)
            if self._source_x_m > 14.0 and self._source_velocity > 0:
                self._source_x_m = -14.0
                self._wavefronts.clear()
            elif self._source_x_m < -14.0 and self._source_velocity < 0:
                self._source_x_m = 14.0
                self._wavefronts.clear()

        # Emit new wavefronts periodically
        if self._mode in ("doppler", "mach", "spectrogram"):
            if self._sim_time_sec - self._last_emission_sec >= self._emission_interval_sec:
                self._wavefronts.append({
                    "x": self._source_x_m,
                    "y": self._source_y_m,
                    "t": self._sim_time_sec,
                })
                self._last_emission_sec = self._sim_time_sec

        elif self._mode == "echolocation":
            # Emit echolocation ultrasound pulse every 0.8 seconds
            if self._sim_time_sec - self._last_pulse_emit_time >= 0.8:
                self._wavefronts.append({
                    "x": self._source_x_m,
                    "y": self._source_y_m,
                    "t": self._sim_time_sec,
                    "reflected": False,
                })
                self._last_pulse_emit_time = self._sim_time_sec

        # Prune old wavefronts that expanded beyond 30 meters
        pruned_wavefronts = []
        for wf in self._wavefronts:
            radius = self._c * (self._sim_time_sec - wf["t"])
            if self._mode == "echolocation" and not wf.get("reflected", False):
                # Check if pulse reaches obstacle at x = obstacle_distance_m
                dist_to_obs = self._obstacle_distance_m - wf["x"]
                if radius >= dist_to_obs and dist_to_obs > 0:
                    wf["reflected"] = True
                    # Spawn reflected echo wavefront originating from obstacle
                    self._echo_wavefronts.append({
                        "x": self._obstacle_distance_m,
                        "y": 0.0,
                        "t": self._sim_time_sec,
                    })
            if radius < 28.0:
                pruned_wavefronts.append(wf)
        self._wavefronts = pruned_wavefronts

        # Prune old echo wavefronts
        pruned_echoes = []
        for ew in self._echo_wavefronts:
            radius = self._c * (self._sim_time_sec - ew["t"])
            if radius < 28.0:
                pruned_echoes.append(ew)
        self._echo_wavefronts = pruned_echoes

        # Update Waterfall Spectrogram history
        f_obs = self.calculate_observed_frequency()
        self._spectrogram_history.append((self._sim_time_sec, f_obs))
        if len(self._spectrogram_history) > self._max_spectrogram_points:
            self._spectrogram_history.pop(0)

        # Trigger redraw
        self.update()

    def set_preset(self, preset_code: str) -> None:
        """Apply one of the 5 curriculum presets."""
        if preset_code not in PRESETS:
            logger.warning(f"Unknown acoustic preset: {preset_code}")
            return

        p = PRESETS[preset_code]
        self._current_preset = preset_code
        self._source_velocity = p.source_velocity_m_s
        self._source_freq = p.source_freq_hz
        self._mode = p.mode
        self._beam_angle_deg = p.beam_angle_deg
        self._obstacle_distance_m = p.obstacle_distance_m

        if self._mode == "echolocation":
            self._source_x_m = -4.0
            self._observer_x_m = -4.0
            self._observer_y_m = 0.0
        elif self._mode == "beamforming":
            self._source_x_m = 0.0
            self._source_y_m = -6.0
            self._observer_x_m = 0.0
            self._observer_y_m = 5.0
        else:
            self._source_x_m = -12.0
            self._source_y_m = 0.0
            self._observer_x_m = 0.0
            self._observer_y_m = 4.5

        self._reset_simulation()
        self.preset_changed.emit(preset_code)
        self.velocity_changed.emit(self._source_velocity)
        self.frequency_changed.emit(self._source_freq)
        self.beam_angle_changed.emit(self._beam_angle_deg)
        self.update()

    def set_velocity(self, v_m_s: float) -> None:
        """Set source velocity in m/s (can be negative or positive)."""
        self._source_velocity = max(-700.0, min(700.0, v_m_s))
        self.velocity_changed.emit(self._source_velocity)
        self.update()

    def set_frequency(self, freq_hz: float) -> None:
        """Set source base frequency in Hz."""
        self._source_freq = max(100.0, min(50000.0, freq_hz))
        self.frequency_changed.emit(self._source_freq)
        self.update()

    def set_beam_angle(self, angle_deg: float) -> None:
        """Set sonar beam steering angle in degrees."""
        self._beam_angle_deg = max(-60.0, min(60.0, angle_deg))
        self.beam_angle_changed.emit(self._beam_angle_deg)
        self.update()

    def toggle_play_pause(self) -> None:
        """Toggle simulation pause state."""
        self._is_playing = not self._is_playing
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1. Main outer card background
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor(14, 20, 32, 245))
        grad.setColorAt(1.0, QColor(9, 13, 22, 250))
        painter.setPen(QPen(QColor(42, 70, 110, 200), 1.5))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header bar with title and close button
        self._paint_header(painter)

        # 3. Preset selector bar (5 buttons)
        self._paint_preset_bar(painter)

        # 4. Main simulation canvas (acoustic field & wavefronts)
        self._paint_simulation_canvas(painter)

        # 5. Right-side dashboard: Sliders, Metrics & Spectrogram
        self._paint_control_dashboard(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Draw top banner with title, badges, and close button."""
        title_font = QFont("Segoe UI", 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(240, 246, 255)))
        painter.drawText(
            QRectF(20, 14, 600, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            "SÓNG ÂM: HIỆU ỨNG DOPPLER, ĐỊNH VỊ SIÊU ÂM & SONAR BEAMFORMING",
        )

        # Subtitle / Model badge
        sub_font = QFont("Segoe UI", 8, QFont.Normal)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(130, 175, 230)))
        painter.drawText(
            QRectF(20, 36, 600, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Acoustic Wave Mechanics · Mach Shock Cone · Time-of-Flight Sonar · Phased Array",
        )

        # Close button [X]
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        is_hover = self._hovered_button == "close"
        painter.setPen(QPen(QColor(230, 80, 80) if is_hover else QColor(160, 60, 60), 1.2))
        painter.setBrush(QBrush(QColor(220, 60, 60, 80 if is_hover else 30)))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(close_rect, Qt.AlignCenter, "×")

    def _paint_preset_bar(self, painter: QPainter) -> None:
        """Draw the 5 preset tabs."""
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0

        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            p = PRESETS[key]
            x = start_x + i * (btn_w + 8.0)
            rect = QRectF(x, y, btn_w, btn_h)

            is_active = (self._current_preset == key)
            is_hover = (self._hovered_button == f"preset_{key}")

            if is_active:
                painter.setPen(QPen(QColor(56, 189, 248), 1.6))
                painter.setBrush(QBrush(QColor(24, 76, 120, 220)))
            elif is_hover:
                painter.setPen(QPen(QColor(70, 130, 180), 1.2))
                painter.setBrush(QBrush(QColor(30, 50, 75, 180)))
            else:
                painter.setPen(QPen(QColor(40, 60, 90), 1.0))
                painter.setBrush(QBrush(QColor(20, 32, 50, 140)))

            painter.drawRoundedRect(rect, 6.0, 6.0)

            font = QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 205, 230)))
            painter.drawText(rect, Qt.AlignCenter, p.name_vi)

    def _paint_simulation_canvas(self, painter: QPainter) -> None:
        """Draw interactive wave field, wavefronts, Mach cone, and markers."""
        canvas_rect = QRectF(20, 96, 560, 470)

        # Background grid
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(6, 11, 20, 240)))
        painter.drawRoundedRect(canvas_rect, 10.0, 10.0)

        painter.save()
        painter.setClipRect(canvas_rect)

        # Coordinate conversion:
        # Simulation Domain: X ∈ [-15m, +15m] (30m total), Y ∈ [-10m, +10m] (20m total)
        cx = canvas_rect.center().x()
        cy = canvas_rect.center().y()
        scale_x = canvas_rect.width() / 30.0  # px per meter (~18.66 px/m)
        scale_y = canvas_rect.height() / 20.0  # px per meter (~23.5 px/m)

        def to_screen(xm: float, ym: float) -> QPointF:
            return QPointF(cx + xm * scale_x, cy - ym * scale_y)

        # Draw subtle grid lines
        painter.setPen(QPen(QColor(25, 40, 65, 80), 0.8, Qt.DashLine))
        for gx in range(-15, 16, 5):
            p1 = to_screen(gx, -10)
            p2 = to_screen(gx, 10)
            painter.drawLine(p1, p2)
        for gy in range(-10, 11, 5):
            p1 = to_screen(-15, gy)
            p2 = to_screen(15, gy)
            painter.drawLine(p1, p2)

        # Draw axis labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(90, 125, 165)))
        painter.drawText(to_screen(13.5, 0.8), "+15m")
        painter.drawText(to_screen(-14.5, 0.8), "-15m")
        painter.drawText(to_screen(0.5, 9.0), "+10m")

        # ---------------------------------------------------------------------
        # Mode Specific Wave Rendering
        # ---------------------------------------------------------------------

        if self._mode in ("doppler", "mach", "spectrogram"):
            # Draw expanding circular wavefronts
            for wf in self._wavefronts:
                radius_m = self._c * (self._sim_time_sec - wf["t"])
                if radius_m <= 0:
                    continue

                sp = to_screen(wf["x"], wf["y"])
                rx = radius_m * scale_x
                ry = radius_m * scale_y

                # Color fades as wave propagates outward
                alpha = max(20, min(180, int(220 * (1.0 - radius_m / 28.0))))
                if self.mach_number >= 1.0:
                    pen_color = QColor(255, 120, 60, alpha)  # Shockwave orange
                else:
                    pen_color = QColor(56, 189, 248, alpha)  # Doppler sky blue

                painter.setPen(QPen(pen_color, 1.2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(sp, rx, ry)

            # Draw Mach Cone tangent envelope if M >= 1.0
            if self.mach_number >= 1.0 and abs(self._source_velocity) > 1.0:
                mu_deg = self.mach_angle_deg
                if mu_deg is not None:
                    mu_rad = math.radians(mu_deg)
                    sp = to_screen(self._source_x_m, self._source_y_m)
                    sign = 1.0 if self._source_velocity > 0 else -1.0

                    cone_len = 35.0  # meters
                    dx_back = -sign * cone_len * math.cos(mu_rad)
                    dy_cone = cone_len * math.sin(mu_rad)

                    p_top = to_screen(self._source_x_m + dx_back, self._source_y_m + dy_cone)
                    p_bot = to_screen(self._source_x_m + dx_back, self._source_y_m - dy_cone)

                    # Mach wedge filled polygon
                    cone_path = QPainterPath()
                    cone_path.moveTo(sp)
                    cone_path.lineTo(p_top)
                    cone_path.lineTo(p_bot)
                    cone_path.closeSubpath()

                    painter.setPen(QPen(QColor(255, 80, 40, 220), 2.2))
                    painter.setBrush(QBrush(QColor(255, 60, 20, 35)))
                    painter.drawPath(cone_path)

                    # Draw Mach angle label
                    painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                    painter.setPen(QPen(QColor(255, 180, 80)))
                    label_pos = to_screen(self._source_x_m - sign * 4.0, self._source_y_m + 2.0)
                    painter.drawText(label_pos, f"Mach Cone μ = {mu_deg:.1f}°")

        elif self._mode == "echolocation":
            # Draw bat / emitter at source_x_m
            # Draw obstacle at obstacle_distance_m
            obs_pt = to_screen(self._obstacle_distance_m, 0.0)
            painter.setPen(QPen(QColor(240, 180, 60), 2.0))
            painter.setBrush(QBrush(QColor(180, 120, 30, 180)))
            painter.drawRect(QRectF(obs_pt.x() - 10, obs_pt.y() - 100, 20, 200))

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QPen(QColor(255, 210, 100)))
            painter.drawText(QRectF(obs_pt.x() - 40, obs_pt.y() + 110, 80, 20), Qt.AlignCenter, "VẬT CẢN (Obstacle)")

            # Draw outgoing ultrasonic pulses
            for wf in self._wavefronts:
                radius_m = self._c * (self._sim_time_sec - wf["t"])
                sp = to_screen(wf["x"], wf["y"])
                rx = radius_m * scale_x
                ry = radius_m * scale_y

                alpha = max(20, min(220, int(240 * (1.0 - radius_m / 24.0))))
                painter.setPen(QPen(QColor(56, 189, 248, alpha), 1.8))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(sp, rx, ry)

            # Draw reflected echo pulses
            for ew in self._echo_wavefronts:
                radius_m = self._c * (self._sim_time_sec - ew["t"])
                sp = to_screen(ew["x"], ew["y"])
                rx = radius_m * scale_x
                ry = radius_m * scale_y

                alpha = max(20, min(220, int(240 * (1.0 - radius_m / 24.0))))
                painter.setPen(QPen(QColor(250, 100, 180, alpha), 1.8, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(sp, rx, ry)

        elif self._mode == "beamforming":
            # Draw phased array transducers along bottom (y = -6m)
            array_center = to_screen(0.0, -6.0)
            element_pitch_px = 12.0
            total_elements = self._num_array_elements
            start_elem_x = array_center.x() - ((total_elements - 1) * element_pitch_px) / 2.0

            # Draw transducer bar
            bar_rect = QRectF(start_elem_x - 14, array_center.y() - 8, total_elements * element_pitch_px + 16, 16)
            painter.setPen(QPen(QColor(60, 90, 130), 1.2))
            painter.setBrush(QBrush(QColor(25, 42, 68, 220)))
            painter.drawRoundedRect(bar_rect, 4.0, 4.0)

            for e in range(total_elements):
                ex = start_elem_x + e * element_pitch_px
                elem_pt = QPointF(ex, array_center.y())
                painter.setPen(QPen(QColor(56, 189, 248), 1.5))
                painter.setBrush(QBrush(QColor(100, 220, 255)))
                painter.drawEllipse(elem_pt, 3.5, 3.5)

            # Draw directional acoustic beam polar intensity fan
            steer_rad = math.radians(self._beam_angle_deg)
            beam_path = QPainterPath()
            beam_path.moveTo(array_center)

            fan_steps = 90
            for s in range(fan_steps + 1):
                ang = math.radians(-85.0 + s * (170.0 / fan_steps))
                af = self.calculate_array_factor(ang)
                # Polar beam distance in meters
                r_beam = af * 14.0
                bx = r_beam * math.sin(ang)
                by = -6.0 + r_beam * math.cos(ang)
                pt = to_screen(bx, by)
                beam_path.lineTo(pt)
            beam_path.closeSubpath()

            grad_beam = QRadialGradient(array_center, 300.0)
            grad_beam.setColorAt(0.0, QColor(56, 189, 248, 160))
            grad_beam.setColorAt(0.5, QColor(34, 197, 94, 90))
            grad_beam.setColorAt(1.0, QColor(14, 165, 233, 0))

            painter.setPen(QPen(QColor(56, 189, 248, 200), 1.5))
            painter.setBrush(QBrush(grad_beam))
            painter.drawPath(beam_path)

            # Steering vector centerline
            steer_target = to_screen(12.0 * math.sin(steer_rad), -6.0 + 12.0 * math.cos(steer_rad))
            painter.setPen(QPen(QColor(250, 204, 21), 1.8, Qt.DashLine))
            painter.drawLine(array_center, steer_target)

            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(QPen(QColor(250, 204, 21)))
            painter.drawText(steer_target.x() + 8, steer_target.y(), f"θ₀ = {self._beam_angle_deg:+.1f}°")

        # ---------------------------------------------------------------------
        # Draw Source Emitter Marker
        # ---------------------------------------------------------------------
        src_pt = to_screen(self._source_x_m, self._source_y_m)
        painter.setPen(QPen(QColor(240, 246, 255), 1.5))
        painter.setBrush(QBrush(QColor(239, 68, 68) if self.mach_number >= 1.0 else QColor(56, 189, 248)))
        painter.drawEllipse(src_pt, 8.0, 8.0)

        # Velocity arrow
        if abs(self._source_velocity) > 0.1:
            arrow_len = min(60.0, max(15.0, abs(self._source_velocity) * 0.1))
            sign = 1.0 if self._source_velocity > 0 else -1.0
            p_arrow_end = QPointF(src_pt.x() + sign * arrow_len, src_pt.y())
            painter.setPen(QPen(QColor(250, 204, 21), 2.0))
            painter.drawLine(src_pt, p_arrow_end)
            # Arrow head
            ah1 = QPointF(p_arrow_end.x() - sign * 6, p_arrow_end.y() - 4)
            ah2 = QPointF(p_arrow_end.x() - sign * 6, p_arrow_end.y() + 4)
            painter.drawLine(p_arrow_end, ah1)
            painter.drawLine(p_arrow_end, ah2)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(src_pt.x() - 30, src_pt.y() - 12, "NGUỒN ÂM (S)")

        # ---------------------------------------------------------------------
        # Draw Observer Receiver Marker
        # ---------------------------------------------------------------------
        if self._mode != "beamforming":
            obs_pt = to_screen(self._observer_x_m, self._observer_y_m)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(QBrush(QColor(34, 197, 94)))
            painter.drawEllipse(obs_pt, 8.0, 8.0)

            # Dashed line from source to observer
            painter.setPen(QPen(QColor(100, 140, 190, 140), 1.0, Qt.DotLine))
            painter.drawLine(src_pt, obs_pt)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QPen(QColor(74, 222, 128)))
            f_obs_now = self.calculate_observed_frequency()
            painter.drawText(obs_pt.x() + 12, obs_pt.y() + 4, f"NGƯỜI QUAN SÁT (O): {f_obs_now:.1f} Hz")

        painter.restore()

        # Canvas bottom bar: Play / Pause, Reset, Status summary
        self._paint_canvas_bottom_bar(painter, canvas_rect)

    def _paint_canvas_bottom_bar(self, painter: QPainter, canvas_rect: QRectF) -> None:
        """Draw interactive playback controls under canvas."""
        y_bar = canvas_rect.bottom() + 10
        btn_h = 28.0

        # Play / Pause button
        play_rect = QRectF(canvas_rect.left(), y_bar, 100, btn_h)
        is_hover_play = self._hovered_button == "toggle_play"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_play else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_play else 140)))
        painter.drawRoundedRect(play_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(play_rect, Qt.AlignCenter, "⏸ TẠM DỪNG" if self._is_playing else "▶ TIẾP TỤC")

        # Reset button
        reset_rect = QRectF(play_rect.right() + 10, y_bar, 100, btn_h)
        is_hover_reset = self._hovered_button == "reset_sim"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_reset else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_reset else 140)))
        painter.drawRoundedRect(reset_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(reset_rect, Qt.AlignCenter, "↺ ĐẶT LẠI")

        # Live info status text
        info_rect = QRectF(reset_rect.right() + 14, y_bar, canvas_rect.right() - reset_rect.right() - 14, btn_h)
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(160, 195, 235)))
        mach_str = f"M = {self.mach_number:.2f} ({'Siêu thanh' if self.mach_number >= 1.0 else 'Dưới thanh'})"
        info_str = f"v_s = {self._source_velocity:+.1f} m/s ({self._source_velocity * 3.6:+.0f} km/h)  ·  {mach_str}  ·  c = {self._c:.0f} m/s"
        painter.drawText(info_rect, Qt.AlignVCenter | Qt.AlignLeft, info_str)

    def _paint_control_dashboard(self, painter: QPainter) -> None:
        """Draw sliders, live physics readouts, and Waterfall Spectrogram."""
        dash_rect = QRectF(596, 96, 324, 534)
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(10, 16, 28, 230)))
        painter.drawRoundedRect(dash_rect, 10.0, 10.0)

        y = dash_rect.top() + 14

        # Section 1: Dynamic Sliders
        header_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(header_font)
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_rect.left() + 16, y, 290, 20), Qt.AlignLeft, "THÔNG SỐ VẬT LÝ ÂM HỌC")
        y += 26

        # Slider 1: Source Velocity v_s
        self._paint_slider(
            painter,
            slider_id="velocity",
            label="Vận tốc nguồn âm v_s",
            val_text=f"{self._source_velocity:+.1f} m/s (Mach {self.mach_number:.2f})",
            min_val=-550.0,
            max_val=550.0,
            current_val=self._source_velocity,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 54

        # Slider 2: Source Frequency f_s
        self._paint_slider(
            painter,
            slider_id="freq",
            label="Tần số phát sóng f_s",
            val_text=f"{self._source_freq:.0f} Hz" if self._source_freq < 10000 else f"{self._source_freq/1000.0:.1f} kHz",
            min_val=100.0,
            max_val=45000.0,
            current_val=self._source_freq,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 54

        # Slider 3: Context-sensitive slider (Beam angle or Obstacle distance)
        if self._mode == "beamforming":
            self._paint_slider(
                painter,
                slider_id="beam_angle",
                label="Góc bẻ chùm Sonar θ₀",
                val_text=f"{self._beam_angle_deg:+.1f}°",
                min_val=-60.0,
                max_val=60.0,
                current_val=self._beam_angle_deg,
                y_pos=y,
                dash_left=dash_rect.left(),
            )
        elif self._mode == "echolocation":
            self._paint_slider(
                painter,
                slider_id="obstacle",
                label="Cự ly vật cản d",
                val_text=f"{self._obstacle_distance_m:.2f} m (TOF: {self.calculate_echolocation_tof_ms():.1f} ms)",
                min_val=1.0,
                max_val=14.0,
                current_val=self._obstacle_distance_m,
                y_pos=y,
                dash_left=dash_rect.left(),
            )
        y += 56

        # Section 2: Real-time Physics Readouts Box
        readout_rect = QRectF(dash_rect.left() + 14, y, 296, 120)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(16, 25, 42, 220)))
        painter.drawRoundedRect(readout_rect, 8.0, 8.0)

        f_o = self.calculate_observed_frequency()
        freq_ratio = f_o / self._source_freq if self._source_freq > 0 else 1.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(120, 180, 240)))
        painter.drawText(QRectF(readout_rect.left() + 10, readout_rect.top() + 8, 276, 18), "KẾT QUẢ ĐO TỨC THỜI:")

        metrics = [
            ("Tần số quan sát f_o:", f"{f_o:.1f} Hz (x{freq_ratio:.2f})"),
            ("Độ lệch tần số Δf:", f"{f_o - self._source_freq:+.1f} Hz"),
            ("Số Mach (M = v/c):", f"{self.mach_number:.3f}"),
            ("Góc nón Mach (μ):", f"{self.mach_angle_deg:.1f}°" if self.mach_angle_deg is not None else "Không hình thành (M < 1)"),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
        for m_idx, (m_lbl, m_val) in enumerate(metrics):
            my = readout_rect.top() + 30 + m_idx * 21
            painter.setPen(QPen(QColor(180, 205, 230)))
            painter.drawText(QRectF(readout_rect.left() + 10, my, 140, 18), Qt.AlignLeft, m_lbl)
            painter.setPen(QPen(QColor(250, 204, 21) if "Mach" in m_lbl else QColor(74, 222, 128)))
            painter.drawText(QRectF(readout_rect.right() - 146, my, 136, 18), Qt.AlignRight, m_val)

        y += 134

        # Section 3: Waterfall Spectrogram Display
        self._paint_waterfall_spectrogram(painter, dash_rect.left() + 14, y, 296, 160)

    def _paint_slider(
        self,
        painter: QPainter,
        slider_id: str,
        label: str,
        val_text: str,
        min_val: float,
        max_val: float,
        current_val: float,
        y_pos: float,
        dash_left: float,
    ) -> None:
        """Render a styled draggable slider track with knob."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(180, 205, 230)))
        painter.drawText(QRectF(dash_left + 16, y_pos, 160, 16), Qt.AlignLeft, label)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_left + 170, y_pos, 136, 16), Qt.AlignRight, val_text)

        # Track
        track_rect = QRectF(dash_left + 16, y_pos + 20, 292, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(25, 40, 62)))
        painter.drawRoundedRect(track_rect, 4.0, 4.0)

        # Filled portion
        fraction = max(0.0, min(1.0, (current_val - min_val) / (max_val - min_val)))
        fill_w = max(6.0, fraction * track_rect.width())
        fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_w, track_rect.height())
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(fill_rect, 4.0, 4.0)

        # Knob
        knob_x = track_rect.left() + fraction * track_rect.width()
        knob_pt = QPointF(knob_x, track_rect.center().y())
        is_active = (self._active_slider == slider_id)
        painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 220, 255), 1.5))
        painter.setBrush(QBrush(QColor(56, 189, 248) if is_active else QColor(30, 80, 140)))
        painter.drawEllipse(knob_pt, 7.0, 7.0)

    def _paint_waterfall_spectrogram(
        self,
        painter: QPainter,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        """Render scrolling Waterfall Spectrogram showing Doppler S-curve."""
        spec_rect = QRectF(x, y, w, h)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(8, 14, 24, 240)))
        painter.drawRoundedRect(spec_rect, 8.0, 8.0)

        painter.save()
        painter.setClipRect(spec_rect)

        # Spectrogram Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(x + 10, y + 6, w - 20, 16), "WATERFALL SPECTROGRAM (f vs t)")

        # Frequency range in spectrogram: [0.2 * f_s, 2.5 * f_s]
        base_f = max(100.0, self._source_freq)
        f_min = 0.2 * base_f
        f_max = 2.5 * base_f

        graph_top = y + 26
        graph_h = h - 34
        graph_w = w - 20
        graph_left = x + 10

        # Center reference line (f_s)
        ref_fraction = (base_f - f_min) / (f_max - f_min)
        ref_y = graph_top + (1.0 - ref_fraction) * graph_h
        painter.setPen(QPen(QColor(100, 130, 170, 120), 0.8, Qt.DashLine))
        painter.drawLine(QPointF(graph_left, ref_y), QPointF(graph_left + graph_w, ref_y))

        painter.setFont(QFont("Segoe UI", 6, QFont.Normal))
        painter.setPen(QPen(QColor(120, 150, 190)))
        painter.drawText(graph_left + 4, ref_y - 2, f"f_s ({base_f:.0f} Hz)")

        # Draw time-frequency curve
        if len(self._spectrogram_history) >= 2:
            curve_path = QPainterPath()
            pts = self._spectrogram_history
            num_pts = len(pts)

            for idx, (t_val, f_val) in enumerate(pts):
                px = graph_left + (idx / (self._max_spectrogram_points - 1)) * graph_w
                frac = (f_val - f_min) / (f_max - f_min)
                frac = max(0.0, min(1.0, frac))
                py = graph_top + (1.0 - frac) * graph_h

                if idx == 0:
                    curve_path.moveTo(px, py)
                else:
                    curve_path.lineTo(px, py)

            painter.setPen(QPen(QColor(56, 189, 248), 1.8))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(curve_path)

        painter.restore()

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self.closed.emit()
            if self.board:
                self.board.toggle_acoustic_sonar()
            event.accept()
            return

        # 2. Preset tabs
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self.set_preset(key)
                event.accept()
                return

        # 3. Canvas bottom bar controls (Play/Pause, Reset)
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        play_rect = QRectF(canvas_rect.left(), y_bar, 100, 28)
        if play_rect.contains(pos):
            self.toggle_play_pause()
            event.accept()
            return

        reset_rect = QRectF(play_rect.right() + 10, y_bar, 100, 28)
        if reset_rect.contains(pos):
            self._reset_simulation()
            self.update()
            event.accept()
            return

        # 4. Sliders
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        # Velocity slider track
        vel_track = QRectF(dash_left + 16, dash_rect.top() + 60, slider_w, 20)
        if vel_track.contains(pos):
            self._active_slider = "velocity"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -550.0, 550.0, self.set_velocity)
            event.accept()
            return

        # Frequency slider track
        freq_track = QRectF(dash_left + 16, dash_rect.top() + 114, slider_w, 20)
        if freq_track.contains(pos):
            self._active_slider = "freq"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 100.0, 45000.0, self.set_frequency)
            event.accept()
            return

        # Context-sensitive slider
        if self._mode == "beamforming":
            beam_track = QRectF(dash_left + 16, dash_rect.top() + 170, slider_w, 20)
            if beam_track.contains(pos):
                self._active_slider = "beam_angle"
                self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -60.0, 60.0, self.set_beam_angle)
                event.accept()
                return
        elif self._mode == "echolocation":
            obs_track = QRectF(dash_left + 16, dash_rect.top() + 170, slider_w, 20)
            if obs_track.contains(pos):
                self._active_slider = "obstacle"
                def set_obs(val: float) -> None:
                    self._obstacle_distance_m = val
                    self.update()
                self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 1.0, 14.0, set_obs)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        if self._active_slider == "velocity":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -550.0, 550.0, self.set_velocity)
            event.accept()
            return
        elif self._active_slider == "freq":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 100.0, 45000.0, self.set_frequency)
            event.accept()
            return
        elif self._active_slider == "beam_angle":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -60.0, 60.0, self.set_beam_angle)
            event.accept()
            return
        elif self._active_slider == "obstacle":
            def set_obs(val: float) -> None:
                self._obstacle_distance_m = val
                self.update()
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 1.0, 14.0, set_obs)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._active_slider = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: Any) -> None:
        pos = event.pos()
        old_hover = self._hovered_button
        self._hovered_button = None

        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self._hovered_button = "close"

        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self._hovered_button = f"preset_{key}"

        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 100, 28).contains(pos):
            self._hovered_button = "toggle_play"
        elif QRectF(canvas_rect.left() + 110, y_bar, 100, 28).contains(pos):
            self._hovered_button = "reset_sim"

        if old_hover != self._hovered_button:
            self.update()

        super().hoverMoveEvent(event)

    def _handle_slider_move(
        self,
        mouse_x: float,
        track_left: float,
        track_width: float,
        min_val: float,
        max_val: float,
        setter_func: Any,
    ) -> None:
        fraction = max(0.0, min(1.0, (mouse_x - track_left) / track_width))
        val = min_val + fraction * (max_val - min_val)
        setter_func(val)

    # -------------------------------------------------------------------------
    # Touchless Webcam Air Gestures
    # -------------------------------------------------------------------------

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process air tap gesture on buttons or sliders."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # Close button
        if QRectF(self._width - 36, 12, 24, 24).contains(pos):
            self.closed.emit()
            if self.board:
                self.board.toggle_acoustic_sonar()
            return True

        # Presets
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            if QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h).contains(pos):
                self.set_preset(key)
                return True

        # Play / Pause
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 100, 28).contains(pos):
            self.toggle_play_pause()
            return True

        # Reset
        if QRectF(canvas_rect.left() + 110, y_bar, 100, 28).contains(pos):
            self._reset_simulation()
            return True

        return True

    # -------------------------------------------------------------------------
    # Serialization & Canvas Stamping
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach packaging."""
        return {
            "type": "InteractiveAcousticSonarItem",
            "pos": [self.x(), self.y()],
            "preset": self._current_preset,
            "source_velocity": self._source_velocity,
            "source_freq": self._source_freq,
            "mode": self._mode,
            "beam_angle_deg": self._beam_angle_deg,
            "obstacle_distance_m": self._obstacle_distance_m,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        board: Optional["LessonBoard"] = None,
    ) -> "InteractiveAcousticSonarItem":
        """Reconstruct from dictionary."""
        pos = data.get("pos", [80.0, 80.0])
        item = cls(board=board, pos=QPointF(pos[0], pos[1]))
        preset = data.get("preset", "AMBULANCE_DOPPLER")
        item.set_preset(preset)
        if "source_velocity" in data:
            item.set_velocity(data["source_velocity"])
        if "source_freq" in data:
            item.set_frequency(data["source_freq"])
        if "beam_angle_deg" in data:
            item.set_beam_angle(data["beam_angle_deg"])
        if "obstacle_distance_m" in data:
            item._obstacle_distance_m = data["obstacle_distance_m"]
        return item

    def stamp_to_canvas(self) -> "InteractiveAcousticSonarItem":
        """Duplicate snapshot item onto lesson canvas."""
        clone = InteractiveAcousticSonarItem(
            board=self.board,
            pos=(self.x() + 30.0, self.y() + 30.0),
        )
        clone.set_preset(self._current_preset)
        clone.set_velocity(self._source_velocity)
        clone.set_frequency(self._source_freq)
        clone.set_beam_angle(self._beam_angle_deg)
        clone._obstacle_distance_m = self._obstacle_distance_m
        if self.scene():
            self.scene().addItem(clone)
        return clone
