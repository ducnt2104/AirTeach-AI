"""
AirTeach AI - Interactive Physics Simple Harmonic Motion & Damped / Forced Resonance Simulator
==============================================================================================
Provides an authentic physics simulation of mechanical oscillations, damping, and resonance:
  - Theoretical Foundations (Zero Mock Data):
      1. Simple Harmonic Motion (SHM - Dao Động Điều Hòa):
           Equation of motion: m·x''(t) + k·x(t) = 0  =>  x''(t) + ω₀²·x(t) = 0
           Natural angular frequency: ω₀ = sqrt(k / m)  (rad/s)
           Period: T₀ = 2π / ω₀ = 2π · sqrt(m / k)  (s)
           Kinematic coordinates:
             x(t) = A · cos(ω₀·t + φ₀)
             v(t) = -ω₀ · A · sin(ω₀·t + φ₀)
             a(t) = -ω₀² · x(t)
           Mechanical Energy Conservation:
             W_t = 0.5 · k · x²  (Thế năng đàn hồi)
             W_d = 0.5 · m · v²  (Động năng)
             W = W_t + W_d = 0.5 · k · A² = const
      2. Damped Oscillations (Dao Động Tắt Dần):
           Drag force: F_drag = -b · v
           Differential equation: x''(t) + 2γ·x'(t) + ω₀²·x(t) = 0, where γ = b / (2m), ζ = b / (2·sqrt(km))
           Three Damping Regimes:
             - Underdamped (ζ < 1): x(t) = A₀ · exp(-γ·t) · cos(ω_d·t + φ),  ω_d = ω₀·sqrt(1 - ζ²)
             - Critically damped (ζ = 1): x(t) = (C₁ + C₂·t) · exp(-ω₀·t)  (Returns to 0 fastest without overshoot)
             - Overdamped (ζ > 1): x(t) = C₁·exp(r₁·t) + C₂·exp(r₂·t),  r₁,₂ = -γ ± sqrt(γ² - ω₀²)
      3. Driven Oscillation & Resonance (Dao Động Cưỡng Bức & Hiện Tượng Cộng Hưởng):
           Driving force: F(t) = F₀ · cos(ω·t)
           Steady-state amplitude response:
             A(ω) = (F₀ / m) / sqrt((ω₀² - ω²)² + (2γ·ω)²)
           Resonance frequency: ω_res = sqrt(ω₀² - 2γ²)
           Quality factor: Q = ω₀ / (2γ) = 1 / (2ζ)
           Phase lag: tan(δ) = (2γ·ω) / (ω₀² - ω²)
  - 5 Authentic Curriculum Presets:
      - IDEAL_SPRING_CONSERVATIVE: Con lắc lò xo lý tưởng không ma sát bảo toàn cơ năng tuyệt đối
      - UNDERDAMPED_AIR_FLUID: Tắt dần trong không khí biên độ giảm theo hàm mũ exp(-γt)
      - CRITICAL_AUTOMOTIVE_SHOCK: Giảm xóc xe hơi tới hạn (ζ = 1.0) dập tắt rung lắc tức thì
      - OVERDAMPED_VISCOUS_HONEY: Chuyển động trong chất lỏng độ nhớt cực cao (dầu đặc / mật ong)
      - DRIVEN_ACOUSTIC_RESONANCE: Dao động cưỡng bức biên độ đạt cực đại khi tần số quét qua f₀
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
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


@dataclass(frozen=True)
class OscillatorPreset:
    """Mechanical and kinematic parameter preset for harmonic oscillator simulator."""

    code: str
    title_vi: str
    regime: str  # "SHM_IDEAL", "UNDERDAMPED", "CRITICAL", "OVERDAMPED", "FORCED_RESONANCE"
    mass_kg: float  # Mass of suspended block
    spring_k_n_m: float  # Spring stiffness constant k (N/m)
    damping_b_ns_m: float  # Viscous damping coefficient b (N·s/m)
    driving_f0_n: float  # Amplitude of periodic driving force F0 (N)
    driving_freq_rad_s: float  # Driving angular frequency ω (rad/s)
    initial_amplitude_m: float  # Initial displacement x0 (m)
    description_vi: str
    formula_vi: str


PRESETS: dict[str, OscillatorPreset] = {
    "IDEAL_SPRING_CONSERVATIVE": OscillatorPreset(
        code="IDEAL_SPRING_CONSERVATIVE",
        title_vi="Con Lắc Lò Xo Lý Tưởng — Bảo Toàn Cơ Năng Tuyệt Đối (SHM)",
        regime="SHM_IDEAL",
        mass_kg=0.5,
        spring_k_n_m=50.0,
        damping_b_ns_m=0.0,
        driving_f0_n=0.0,
        driving_freq_rad_s=0.0,
        initial_amplitude_m=0.15,
        description_vi="Dao động điều hòa không ma sát. Động năng và thế năng liên tục chuyển hóa lẫn nhau, cơ năng toàn phần bảo toàn vĩnh cửu.",
        formula_vi="x(t) = A·cos(ω₀t)  |  ω₀ = √(k/m) = 10.0 rad/s  |  W = ½kA² = const",
    ),
    "UNDERDAMPED_AIR_FLUID": OscillatorPreset(
        code="UNDERDAMPED_AIR_FLUID",
        title_vi="Dao Động Tắt Dần Trong Không Khí & Nước Loãng (Underdamped)",
        regime="UNDERDAMPED",
        mass_kg=0.5,
        spring_k_n_m=50.0,
        damping_b_ns_m=0.8,
        driving_f0_n=0.0,
        driving_freq_rad_s=0.0,
        initial_amplitude_m=0.18,
        description_vi="Lực cản nhớt tỉ lệ với vận tốc F = -bv. Biên độ dao động suy giảm tiệm cận theo đường bao hàm mũ exp(-γt).",
        formula_vi="x(t) = A₀·e^(-γt)·cos(ω_d·t)  |  γ = b/(2m) = 0.80 s⁻¹  |  ω_d = √(ω₀²-γ²)",
    ),
    "CRITICAL_AUTOMOTIVE_SHOCK": OscillatorPreset(
        code="CRITICAL_AUTOMOTIVE_SHOCK",
        title_vi="Giảm Xóc Ô Tô Tới Hạn — Dập Tắt Rung Lắc Tức Thì (Critical Damping)",
        regime="CRITICAL",
        mass_kg=1.0,
        spring_k_n_m=100.0,
        damping_b_ns_m=20.0,  # b_crit = 2*sqrt(k*m) = 2*sqrt(100*1) = 20.0
        driving_f0_n=0.0,
        driving_freq_rad_s=0.0,
        initial_amplitude_m=0.20,
        description_vi="Hệ số cản tới hạn ζ = 1.0. Pittông giảm chấn triệt tiêu dao động nhanh nhất mà không bị dội ngược qua vị trí cân bằng.",
        formula_vi="x(t) = (C₁ + C₂·t)·e^(-ω₀t)  |  b_crit = 2√(k·m) = 20.0 N·s/m  (ζ = 1.0)",
    ),
    "OVERDAMPED_VISCOUS_HONEY": OscillatorPreset(
        code="OVERDAMPED_VISCOUS_HONEY",
        title_vi="Quá Tắt Dần Trong Dầu Nhờn & Mật Ong Đậm Đặc (Overdamped)",
        regime="OVERDAMPED",
        mass_kg=0.5,
        spring_k_n_m=30.0,
        damping_b_ns_m=22.0,  # b_crit = 2*sqrt(15) = 7.75 << 22.0
        driving_f0_n=0.0,
        driving_freq_rad_s=0.0,
        initial_amplitude_m=0.20,
        description_vi="Độ nhớt quá lớn (ζ = 2.84 >> 1). Vật dịch chuyển chậm chạp về vị trí cân bằng mà không hề có dao động tuần hoàn.",
        formula_vi="x(t) = C₁·e^(r₁t) + C₂·e^(r₂t)  |  r₁,₂ = -γ ± √(γ² - ω₀²)  |  ζ = 2.84",
    ),
    "DRIVEN_ACOUSTIC_RESONANCE": OscillatorPreset(
        code="DRIVEN_ACOUSTIC_RESONANCE",
        title_vi="Dao Động Cưỡng Bức & Hiện Tượng Cộng Hưởng Cơ Học (Resonance)",
        regime="FORCED_RESONANCE",
        mass_kg=0.5,
        spring_k_n_m=50.0,
        damping_b_ns_m=0.4,
        driving_f0_n=4.0,
        driving_freq_rad_s=9.9,  # Close to ω0 = 10.0 rad/s
        initial_amplitude_m=0.05,
        description_vi="Ngoại lực tuần hoàn F₀·cos(ωt) truyền năng lượng cho hệ. Khi tần số kích thích tiệm cận tần số riêng ω₀, biên độ vọt lên cực đại.",
        formula_vi="A(ω) = (F₀/m) / √((ω₀² - ω²)² + (2γω)²)  |  Đỉnh cộng hưởng tại ω ≈ ω₀",
    ),
}


class InteractiveHarmonicOscillatorItem(QGraphicsObject):
    """
    Interactive PySide6 QGraphicsItem simulating Simple Harmonic Motion,
    Damped Oscillations (underdamped, critical, overdamped), and Forced Resonance.
    """

    WIDTH = 740.0
    HEIGHT = 500.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: Union[tuple[float, float], QPointF] = (200.0, 90.0)) -> None:
        super().__init__()
        if isinstance(pos, QPointF):
            self.setPos(pos)
        elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
            self.setPos(float(pos[0]), float(pos[1]))
        else:
            self.setPos(200.0, 90.0)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active Preset and Parameters
        self._active_preset_code: str = "IDEAL_SPRING_CONSERVATIVE"
        self._preset: OscillatorPreset = PRESETS[self._active_preset_code]

        self._mass_kg: float = self._preset.mass_kg
        self._spring_k: float = self._preset.spring_k_n_m
        self._damping_b: float = self._preset.damping_b_ns_m
        self._driving_f0: float = self._preset.driving_f0_n
        self._driving_omega: float = self._preset.driving_freq_rad_s
        self._initial_a: float = self._preset.initial_amplitude_m

        # Dynamic State Variables
        self._sim_time_s: float = 0.0
        self._pos_x_m: float = self._initial_a
        self._vel_v_ms: float = 0.0
        self._acc_a_ms2: float = 0.0
        self._is_animating: bool = True

        # Oscilloscope Waveform History: list of (t_rel, x_m)
        self._waveform_history: list[tuple[float, float]] = []

        # Animation timer (~30 FPS, dt = 0.033s)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Initialize click and air-pinch bounding boxes."""
        w, h = self.WIDTH, self.HEIGHT
        self._btn_rects.clear()

        # Header close button
        self._btn_rects["close"] = QRectF(w - 36.0, 10.0, 26.0, 26.0)

        # 5 Preset Buttons (top toolbar)
        p_keys = list(PRESETS.keys())
        p_w = (w - 28.0) / len(p_keys)
        for i, k in enumerate(p_keys):
            self._btn_rects[f"preset_{k}"] = QRectF(14.0 + i * p_w, 44.0, p_w - 4.0, 24.0)

        # Bottom toolbar buttons
        y_bot = h - 42.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 95.0, 30.0)
        self._btn_rects["dec_k"] = QRectF(118.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_k"] = QRectF(152.0, y_bot, 32.0, 30.0)
        self._btn_rects["dec_m"] = QRectF(258.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_m"] = QRectF(292.0, y_bot, 32.0, 30.0)
        self._btn_rects["dec_b"] = QRectF(398.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_b"] = QRectF(432.0, y_bot, 32.0, 30.0)
        self._btn_rects["reset_sim"] = QRectF(482.0, y_bot, 85.0, 30.0)
        self._btn_rects["stamp"] = QRectF(w - 128.0, y_bot, 112.0, 30.0)

    # -------------------------------------------------------------------------
    # Physical Properties & Calculations
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> OscillatorPreset:
        return self._preset

    @property
    def mass_kg(self) -> float:
        return self._mass_kg

    @property
    def spring_k(self) -> float:
        return self._spring_k

    @property
    def damping_b(self) -> float:
        return self._damping_b

    @property
    def natural_frequency_rad_s(self) -> float:
        """Undamped natural angular frequency ω₀ = sqrt(k / m)."""
        return math.sqrt(self._spring_k / max(0.01, self._mass_kg))

    @property
    def natural_frequency_hz(self) -> float:
        """Natural cyclic frequency f₀ = ω₀ / (2π)."""
        return self.natural_frequency_rad_s / (2.0 * math.pi)

    @property
    def damping_ratio_zeta(self) -> float:
        """Damping ratio ζ = b / (2·sqrt(k·m))."""
        crit_b = 2.0 * math.sqrt(self._spring_k * max(0.01, self._mass_kg))
        return self._damping_b / max(1e-4, crit_b)

    @property
    def quality_factor_q(self) -> float:
        """Resonant quality factor Q = 1 / (2ζ)."""
        zeta = self.damping_ratio_zeta
        return (1.0 / (2.0 * zeta)) if zeta > 1e-4 else float("inf")

    @property
    def resonant_frequency_rad_s(self) -> float:
        """Peak amplitude resonance frequency ω_res = sqrt(max(0, ω₀² - 2γ²))."""
        omega0 = self.natural_frequency_rad_s
        gamma = self._damping_b / (2.0 * max(0.01, self._mass_kg))
        val = omega0 ** 2 - 2.0 * (gamma ** 2)
        return math.sqrt(max(0.0, val))

    @property
    def potential_energy_j(self) -> float:
        """Elastic potential energy W_t = 0.5 · k · x²."""
        return 0.5 * self._spring_k * (self._pos_x_m ** 2)

    @property
    def kinetic_energy_j(self) -> float:
        """Kinetic energy W_d = 0.5 · m · v²."""
        return 0.5 * self._mass_kg * (self._vel_v_ms ** 2)

    @property
    def total_energy_j(self) -> float:
        """Total mechanical energy W = W_t + W_d."""
        return self.potential_energy_j + self.kinetic_energy_j

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    # -------------------------------------------------------------------------
    # Public Controller Actions
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic harmonic oscillator curriculum preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._mass_kg = self._preset.mass_kg
        self._spring_k = self._preset.spring_k_n_m
        self._damping_b = self._preset.damping_b_ns_m
        self._driving_f0 = self._preset.driving_f0_n
        self._driving_omega = self._preset.driving_freq_rad_s
        self._initial_a = self._preset.initial_amplitude_m

        self.reset_simulation()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_spring_constant(self, k_n_m: float) -> None:
        """Set spring stiffness (clamped between 5.0 and 200.0 N/m)."""
        self._spring_k = max(5.0, min(200.0, float(k_n_m)))
        self.update()

    def set_mass(self, m_kg: float) -> None:
        """Set oscillator mass (clamped between 0.1 and 5.0 kg)."""
        self._mass_kg = max(0.1, min(5.0, float(m_kg)))
        self.update()

    def set_damping(self, b_ns_m: float) -> None:
        """Set viscous damping coefficient (clamped between 0.0 and 30.0 N·s/m)."""
        self._damping_b = max(0.0, min(30.0, float(b_ns_m)))
        self.update()

    def reset_simulation(self) -> None:
        """Reset state variables to initial amplitude."""
        self._sim_time_s = 0.0
        self._pos_x_m = self._initial_a
        self._vel_v_ms = 0.0
        self._acc_a_ms2 = 0.0
        self._waveform_history.clear()
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle live oscillator motion."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def stamp_to_board(self) -> None:
        """Stamps active Harmonic Oscillator diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_harmonic_oscillator_diagram(
            center=center,
            mass_kg=self._mass_kg,
            spring_k=self._spring_k,
            damping_b=self._damping_b,
            omega_drive=self._driving_omega,
            mode=self._preset.regime,
        )
        for itm in items:
            scene.addItem(itm)

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Control buttons
        if self._btn_rects["close"].contains(local_pos):
            self.closed.emit()
            self.setVisible(False)
            return True

        if self._btn_rects["play_pause"].contains(local_pos):
            self.toggle_animation()
            return True

        if self._btn_rects["dec_k"].contains(local_pos):
            self.set_spring_constant(self._spring_k - 5.0)
            return True
        if self._btn_rects["inc_k"].contains(local_pos):
            self.set_spring_constant(self._spring_k + 5.0)
            return True

        if self._btn_rects["dec_m"].contains(local_pos):
            self.set_mass(self._mass_kg - 0.1)
            return True
        if self._btn_rects["inc_m"].contains(local_pos):
            self.set_mass(self._mass_kg + 0.1)
            return True

        if self._btn_rects["dec_b"].contains(local_pos):
            self.set_damping(self._damping_b - 0.5)
            return True
        if self._btn_rects["inc_b"].contains(local_pos):
            self.set_damping(self._damping_b + 0.5)
            return True

        if self._btn_rects["reset_sim"].contains(local_pos):
            self.reset_simulation()
            return True

        if self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Numerical Integration (Velocity Verlet / Runge-Kutta 4)
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Integrates equations of motion over dt = 0.016s (2 sub-steps of 0.033s)."""
        if not self._is_animating:
            return

        dt = 0.0165
        for _ in range(2):
            self._sim_time_s += dt

            # Net forces: F_net = -k·x - b·v + F_drive
            f_spring = -self._spring_k * self._pos_x_m
            f_damping = -self._damping_b * self._vel_v_ms
            f_drive = self._driving_f0 * math.cos(self._driving_omega * self._sim_time_s) if self._preset.regime == "FORCED_RESONANCE" else 0.0

            f_net = f_spring + f_damping + f_drive
            self._acc_a_ms2 = f_net / max(0.01, self._mass_kg)

            # Symplectic Euler integration
            self._vel_v_ms += self._acc_a_ms2 * dt
            self._pos_x_m += self._vel_v_ms * dt

        # Record to oscilloscope history
        self._waveform_history.append((self._sim_time_s, self._pos_x_m))
        if len(self._waveform_history) > 120:
            self._waveform_history.pop(0)

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#080c16")))
        pen_border = QPen(QColor("#0284c7"), 2.0) if self.isSelected() else QPen(QColor("#1e293b"), 1.6)
        painter.setPen(pen_border)
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Preset Selector Buttons
        self._paint_preset_toolbar(painter)

        # 4. Left Viewport: Spring-Mass-Dashpot Physical Apparatus
        self._paint_apparatus_viewport(painter)

        # 5. Right Viewport: Oscilloscope Waveform & Resonance Curve
        self._paint_oscilloscope_viewport(painter)

        # 6. Bottom Inspector Deck & Controls
        self._paint_bottom_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Header with badge and close icon."""
        badge_rect = QRectF(14.0, 12.0, 24.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "🌀")

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(46.0, 29.0, "DAO ĐỘNG CƠ HỌC: ĐIỀU HÒA, TẮT DẦN & CỘNG HƯỞNG (SHM / RESONANCE)")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(480.0, 29.0, f"Chế độ: {self._preset.regime} | ζ = {self.damping_ratio_zeta:.2f}")

        # Close button
        r_cl = self._btn_rects["close"]
        hov = (self._hovered_button == "close")
        painter.setBrush(QBrush(QColor("#dc2626") if hov else QColor("#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r_cl, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_cl, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_preset_toolbar(self, painter: QPainter) -> None:
        """5 Preset buttons with active indicators."""
        labels = {
            "IDEAL_SPRING_CONSERVATIVE": "⚖️ Điều Hòa Lý Tưởng",
            "UNDERDAMPED_AIR_FLUID": "💧 Tắt Dần Dưới Hạn",
            "CRITICAL_AUTOMOTIVE_SHOCK": "🚗 Tắt Dần Tới Hạn",
            "OVERDAMPED_VISCOUS_HONEY": "🍯 Quá Tắt Dần (Nhớt)",
            "DRIVEN_ACOUSTIC_RESONANCE": "🔔 Cộng Hưởng Cưỡng Bức",
        }
        for code, r in self._btn_rects.items():
            if not code.startswith("preset_"):
                continue
            pk = code.replace("preset_", "")
            is_act = (pk == self._active_preset_code)
            is_hov = (self._hovered_button == code)

            col_bg = QColor("#0284c7") if is_act else (QColor("#1e293b") if is_hov else QColor("#0f172a"))
            col_border = QColor("#38bdf8") if is_act else QColor("#334155")
            painter.setBrush(QBrush(col_bg))
            painter.setPen(QPen(col_border, 1.2 if is_act else 0.8))
            painter.drawRoundedRect(r, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_act else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_act else QColor("#cbd5e1"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, labels.get(pk, pk))

    def _paint_apparatus_viewport(self, painter: QPainter) -> None:
        """Left viewport: Fixed wall, Helical spring, Mass block, Dashpot, and Live Energy gauges."""
        vp_rect = QRectF(14.0, 76.0, 340.0, 290.0)
        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(24.0, 94.0, "MÔ HÌNH VẬT LÝ CON LẮC LÒ XO & PITTÔNG DẬP TẮT")

        # Fixed Ceiling / Top Anchor
        ceil_y = 115.0
        painter.setPen(QPen(QColor("#64748b"), 3.0))
        painter.drawLine(30, int(ceil_y), 190, int(ceil_y))

        # Equilibrium position line (x = 0)
        eq_y = 220.0
        painter.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(25, int(eq_y), 325, int(eq_y))
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#64748b"))
        painter.drawText(275, int(eq_y - 4), "VTCB (x = 0)")

        # Current block position (scale: 1m = 400px)
        scale_pos = 380.0
        block_cy = eq_y + self._pos_x_m * scale_pos
        spring_x = 75.0

        # Draw Helical Spring (Zigzag path)
        spring_path = QPainterPath()
        spring_path.moveTo(spring_x, ceil_y)
        coils = 14
        h_total = max(20.0, block_cy - 20.0 - ceil_y)
        dy = h_total / coils
        for i in range(1, coils):
            sx = spring_x + (10.0 if (i % 2 == 1) else -10.0)
            sy = ceil_y + i * dy
            spring_path.lineTo(sx, sy)
        spring_path.lineTo(spring_x, block_cy - 20.0)

        painter.setPen(QPen(QColor("#38bdf8"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(spring_path)

        # Draw Suspended Mass Block (m)
        bw, bh = 46.0, 40.0
        block_rect = QRectF(spring_x - bw / 2.0, block_cy - bh / 2.0, bw, bh)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawRoundedRect(block_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(block_rect, Qt.AlignmentFlag.AlignCenter, f"{self._mass_kg:.1f}kg")

        # Dashpot Damper (piston in fluid cylinder on the right of spring)
        dash_x = 150.0
        # Cylinder
        cyl_h = 70.0
        cyl_rect = QRectF(dash_x - 12.0, eq_y - cyl_h / 2.0 + 20.0, 24.0, cyl_h)
        col_fluid = QColor(245, 158, 11, 40) if self._damping_b > 5.0 else QColor(56, 189, 248, 30)
        painter.setBrush(QBrush(col_fluid))
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawRect(cyl_rect)

        # Piston rod to block
        piston_y = block_cy
        painter.setPen(QPen(QColor("#94a3b8"), 2.0))
        painter.drawLine(int(dash_x), int(ceil_y), int(dash_x), int(piston_y))
        # Piston head
        painter.drawLine(int(dash_x - 10), int(piston_y), int(dash_x + 10), int(piston_y))

        # Live Energy Bar Gauges (Right edge of left viewport)
        eg_x = 225.0
        eg_w = 95.0
        eg_y = 115.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))

        # Kinetic Energy (Cyan)
        max_e = max(0.01, 0.5 * self._spring_k * (self._initial_a ** 2))
        w_d_pct = min(1.0, self.kinetic_energy_j / max_e)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(int(eg_x), int(eg_y), f"Động năng W_d: {self.kinetic_energy_j*1000:.0f}mJ")
        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(eg_x, eg_y + 4.0, eg_w * w_d_pct, 7.0), 3.0, 3.0)

        # Potential Energy (Amber)
        w_t_pct = min(1.0, self.potential_energy_j / max_e)
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(int(eg_x), int(eg_y + 26.0), f"Thế năng W_t: {self.potential_energy_j*1000:.0f}mJ")
        painter.setBrush(QBrush(QColor("#d97706")))
        painter.drawRoundedRect(QRectF(eg_x, eg_y + 30.0, eg_w * w_t_pct, 7.0), 3.0, 3.0)

        # Total Mechanical Energy (Emerald)
        w_tot_pct = min(1.0, self.total_energy_j / max_e)
        painter.setPen(QColor("#10b981"))
        painter.drawText(int(eg_x), int(eg_y + 52.0), f"Cơ năng W: {self.total_energy_j*1000:.0f}mJ")
        painter.setBrush(QBrush(QColor("#059669")))
        painter.drawRoundedRect(QRectF(eg_x, eg_y + 56.0, eg_w * w_tot_pct, 7.0), 3.0, 3.0)

    def _paint_oscilloscope_viewport(self, painter: QPainter) -> None:
        """Right viewport: Real-time waveform x(t) and Resonance amplitude curve A(ω)."""
        vp_rect = QRectF(366.0, 76.0, 360.0, 290.0)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        # 1. Upper Half: Real-Time Oscilloscope Waveform x(t)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(376.0, 94.0, "DAO ĐỘNG KÝ X(t) — ĐƯỜNG CONG LI ĐỘ THỜI GIAN")

        osc_x0 = 390.0
        osc_y0 = 155.0
        osc_w = 320.0
        osc_h = 45.0

        # Baseline
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(osc_x0), int(osc_y0), int(osc_x0 + osc_w), int(osc_y0))

        # Plot scrolling waveform
        if len(self._waveform_history) >= 2:
            wave_path = QPainterPath()
            t_min = self._waveform_history[0][0]
            t_max = self._waveform_history[-1][0]
            dt_span = max(0.5, t_max - t_min)

            for i, (t_pt, x_val) in enumerate(self._waveform_history):
                px = osc_x0 + ((t_pt - t_min) / dt_span) * osc_w
                py = osc_y0 - (x_val / max(0.05, self._initial_a)) * osc_h
                if i == 0:
                    wave_path.moveTo(px, py)
                else:
                    wave_path.lineTo(px, py)

            painter.setPen(QPen(QColor("#10b981"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(wave_path)

        # 2. Lower Half: Resonance Amplitude Response Curve A(ω)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(376.0, 225.0, "ĐƯỜNG CONG BIÊN ĐỘ CỘNG HƯỞNG A(ω) LORENTZ")

        res_x0 = 390.0
        res_y0 = 345.0
        res_w = 320.0
        res_h = 90.0

        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(res_x0), int(res_y0), int(res_x0 + res_w), int(res_y0))
        painter.drawLine(int(res_x0), int(res_y0), int(res_x0), int(res_y0 - res_h))

        # Resonance curve equation: A(w) = (F0/m) / sqrt((w0^2 - w^2)^2 + (2*gamma*w)^2)
        omega0 = self.natural_frequency_rad_s
        gamma = self._damping_b / (2.0 * max(0.01, self._mass_kg))
        max_omega = omega0 * 2.2
        scale_w = res_w / max_omega

        curve_path = QPainterPath()
        f0_val = max(1.0, self._driving_f0 if self._driving_f0 > 0 else 5.0)

        for step in range(101):
            w_i = (step / 100.0) * max_omega
            denom = math.sqrt((omega0 ** 2 - w_i ** 2) ** 2 + (2.0 * gamma * w_i) ** 2)
            amp = (f0_val / self._mass_kg) / max(0.1, denom)

            rx = res_x0 + w_i * scale_w
            ry = res_y0 - min(res_h - 10.0, amp * 12.0)
            if step == 0:
                curve_path.moveTo(rx, ry)
            else:
                curve_path.lineTo(rx, ry)

        painter.setPen(QPen(QColor("#f59e0b"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(curve_path)

        # Resonance peak marker (ω0)
        peak_x = res_x0 + omega0 * scale_w
        painter.setPen(QPen(QColor("#ef4444"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(int(peak_x), int(res_y0 - res_h + 10), int(peak_x), int(res_y0))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(int(peak_x - 18), int(res_y0 + 14), f"ω₀={omega0:.1f}")

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Bottom HUD and parameter stepper buttons."""
        y_hud = self.HEIGHT - 125.0
        hud_rect = QRectF(14.0, y_hud, self.WIDTH - 28.0, 75.0)
        painter.setBrush(QBrush(QColor("#050b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(hud_rect, 8.0, 8.0)

        omega0 = self.natural_frequency_rad_s
        f0 = self.natural_frequency_hz
        t0 = 1.0 / f0
        zeta = self.damping_ratio_zeta

        # Metrics Grid
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(24.0, y_hud + 18.0, f"Tần số góc riêng ω₀: {omega0:.2f} rad/s  |  f₀ = {f0:.2f} Hz  |  Chu kỳ T₀ = {t0:.2f} s")

        painter.setPen(QColor("#f59e0b"))
        regime_vi = "Tắt dần nhẹ" if zeta < 1.0 else ("Tới hạn" if abs(zeta - 1.0) < 0.05 else "Quá tắt dần")
        painter.drawText(24.0, y_hud + 36.0, f"Hệ số giảm chấn ζ = {zeta:.3f} ({regime_vi})  |  Hệ số phẩm chất Q = {self.quality_factor_q:.1f}")

        painter.setPen(QColor("#10b981"))
        painter.drawText(24.0, y_hud + 54.0, f"Li độ tức thời x: {self._pos_x_m*100:+.2f} cm  |  Vận tốc v: {self._vel_v_ms:+.2f} m/s  |  Gia tốc a: {self._acc_a_ms2:+.1f} m/s²")

        painter.setPen(QColor("#94a3b8"))
        painter.drawText(390.0, y_hud + 18.0, f"Độ cứng k = {self._spring_k:.1f} N/m  |  Khối lượng m = {self._mass_kg:.2f} kg  |  Cản b = {self._damping_b:.2f} Ns/m")
        painter.drawText(390.0, y_hud + 36.0, f"Công thức: {self._preset.formula_vi}")
        painter.drawText(390.0, y_hud + 54.0, f"Đặc tính: {self._preset.description_vi[:48]}…")

        # Bottom Buttons
        y_bot = self.HEIGHT - 42.0

        # Play / Pause
        r_play = self._btn_rects["play_pause"]
        is_hov_play = (self._hovered_button == "play_pause")
        painter.setBrush(QBrush(QColor("#0284c7") if is_hov_play else QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(r_play, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        txt_play = "⏸ Tạm Dừng" if self._is_animating else "▶ Tiếp Tục"
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, txt_play)

        # Steppers: k, m, b
        self._paint_stepper(painter, "dec_k", "inc_k", f"Độ cứng k: {self._spring_k:.0f} N/m", 118.0)
        self._paint_stepper(painter, "dec_m", "inc_m", f"Khối lượng m: {self._mass_kg:.1f} kg", 258.0)
        self._paint_stepper(painter, "dec_b", "inc_b", f"Lực cản b: {self._damping_b:.1f}", 398.0)

        # Reset button
        r_rst = self._btn_rects["reset_sim"]
        is_hov_rst = (self._hovered_button == "reset_sim")
        painter.setBrush(QBrush(QColor("#334155") if is_hov_rst else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_rst, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "🔄 Đặt Lại")

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        is_hov_st = (self._hovered_button == "stamp")
        painter.setBrush(QBrush(QColor("#059669") if is_hov_st else QColor("#047857")))
        painter.setPen(QPen(QColor("#34d399"), 1.2))
        painter.drawRoundedRect(r_stamp, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Bảng")

    def _paint_stepper(self, painter: QPainter, btn_dec: str, btn_inc: str, label: str, x_start: float) -> None:
        """Helper to render decrement / increment stepper buttons."""
        r_dec = self._btn_rects[btn_dec]
        r_inc = self._btn_rects[btn_inc]

        hov_d = (self._hovered_button == btn_dec)
        painter.setBrush(QBrush(QColor("#334155") if hov_d else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_dec, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_dec, Qt.AlignmentFlag.AlignCenter, "−")

        hov_i = (self._hovered_button == btn_inc)
        painter.setBrush(QBrush(QColor("#334155") if hov_i else QColor("#1e293b")))
        painter.drawRoundedRect(r_inc, 4.0, 4.0)
        painter.drawText(r_inc, Qt.AlignmentFlag.AlignCenter, "+")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(x_start), int(r_dec.top() - 4.0), label)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        for key, r in self._btn_rects.items():
            if r.contains(pos):
                if key == "close":
                    self.closed.emit()
                    self.setVisible(False)
                    event.accept()
                    return
                elif key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                    event.accept()
                    return
                elif key == "play_pause":
                    self.toggle_animation()
                    event.accept()
                    return
                elif key == "dec_k":
                    self.set_spring_constant(self._spring_k - 5.0)
                    event.accept()
                    return
                elif key == "inc_k":
                    self.set_spring_constant(self._spring_k + 5.0)
                    event.accept()
                    return
                elif key == "dec_m":
                    self.set_mass(self._mass_kg - 0.1)
                    event.accept()
                    return
                elif key == "inc_m":
                    self.set_mass(self._mass_kg + 0.1)
                    event.accept()
                    return
                elif key == "dec_b":
                    self.set_damping(self._damping_b - 0.5)
                    event.accept()
                    return
                elif key == "inc_b":
                    self.set_damping(self._damping_b + 0.5)
                    event.accept()
                    return
                elif key == "reset_sim":
                    self.reset_simulation()
                    event.accept()
                    return
                elif key == "stamp":
                    self.stamp_to_board()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        hovered = None
        for key, r in self._btn_rects.items():
            if r.contains(pos):
                hovered = key
                break
        if hovered != self._hovered_button:
            self._hovered_button = hovered
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_button is not None:
            self._hovered_button = None
            self.update()
        super().hoverLeaveEvent(event)

    def cleanup(self) -> None:
        """Stops background animation timer and frees resources."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
