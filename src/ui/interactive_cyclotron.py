"""
AirTeach AI – Interactive Physics & Particle Electromagnetism: Cyclotron Particle Accelerator
=============================================================================================
Provides an authentic physics and relativistic mechanics simulation of the Cyclotron (Ernest Lawrence 1939 Nobel Prize):
  - Magnetic Lorentz Centripetal Force:
      F_B = q * v * B = m * v² / r  ==>  r = (m * v) / (q * B) = p / (q * B)
  - Classical Isochronous Resonance Frequency:
      T = 2π * r / v = 2π * m / (q * B)
      f_c = (q * B) / (2π * m)
      Notice: f_c is independent of orbit radius r and velocity v in non-relativistic regimes.
  - Relativistic Mass Dilation & Dephasing (Phase Slip):
      m(v) = γ * m₀ = m₀ / sqrt(1 - (v/c)²)
      T(v) = 2π * γ * m₀ / (q * B)
      As v -> 0.1c+, the particle takes longer to complete a half-revolution and lags behind the constant RF field,
      causing phase slip and deceleration unless compensated.
  - Synchrocyclotron Frequency Modulation:
      f_rf(t) = (q * B) / (2π * γ(t) * m₀)
  - Extraction Kinetic Energy at R_max:
      Classical:    E_k = (q² * B² * R_max²) / (2 * m)
      Relativistic: E_k = (γ - 1) * m₀ * c² = sqrt(p² * c² + m₀² * c⁴) - m₀ * c²
  - 5 Accelerated Particle Species:
      1. Proton (p, ¹H⁺): m₀ = 1.6726e-27 kg (938.27 MeV/c²), q = +1e
      2. Deuteron (d, ²H⁺): m₀ = 3.3436e-27 kg (1875.6 MeV/c²), q = +1e
      3. Alpha Particle (α, ⁴He²⁺): m₀ = 6.6447e-27 kg (3727.4 MeV/c²), q = +2e
      4. Positron (e⁺): m₀ = 9.109e-31 kg (0.511 MeV/c²), q = +1e
      5. Carbon-12 (¹²C⁶⁺): Hadron therapy heavy ion, q = +6e
  - 5 Curriculum Presets:
      1. CLASSICAL_PROTON_ACCELERATOR: Standard isochronous proton spiral to 16.5 MeV.
      2. RELATIVISTIC_PHASE_SLIP_FAILURE: High energy with fixed RF frequency dephasing into deceleration.
      3. SYNCHROCYCLOTRON_RF_MODULATION: Dynamically modulated RF frequency achieving > 100 MeV.
      4. DEUTERON_ALPHA_RADIOISOTOPE: Alpha particle acceleration for PET scan radiopharmaceuticals.
      5. ISOCHRONOUS_AZIMUTHAL_FOCUSING: Relativistic field compensation B(r) = γ(r) * B₀ with sector flutter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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


# Physical Constants (CODATA 2022)
C_LIGHT = 299792458.0                # Speed of light in vacuum (m/s)
E_CHARGE = 1.602176634e-19           # Elementary charge (C)
M_PROTON = 1.67262192369e-27         # Proton mass (kg)
EV_TO_JOULE = 1.602176634e-19        # 1 eV in Joules
MEV_TO_JOULE = 1.602176634e-13       # 1 MeV in Joules


@dataclass(frozen=True)
class ParticleSpec:
    code: str
    name_vi: str
    symbol: str
    mass_kg: float
    charge_c: float
    rest_energy_mev: float
    color_hex: str


PARTICLES: dict[str, ParticleSpec] = {
    "Proton": ParticleSpec("Proton", "Proton (¹H⁺)", "p", M_PROTON, E_CHARGE, 938.272, "#ef4444"),
    "Deuteron": ParticleSpec("Deuteron", "Đơteri (²H⁺)", "d", 3.3435837724e-27, E_CHARGE, 1875.613, "#f59e0b"),
    "Alpha": ParticleSpec("Alpha", "Hạt Alpha (⁴He²⁺)", "α", 6.6446573357e-27, 2.0 * E_CHARGE, 3727.379, "#10b981"),
    "Positron": ParticleSpec("Positron", "Positron (e⁺)", "e⁺", 9.1093837015e-31, E_CHARGE, 0.511, "#38bdf8"),
    "Carbon": ParticleSpec("Carbon", "Ion Cacbon (¹²C⁶⁺)", "¹²C⁶⁺", 1.99264687992e-26, 6.0 * E_CHARGE, 11177.9, "#a855f7"),
}


@dataclass(frozen=True)
class CyclotronPreset:
    code: str
    title_vi: str
    badge_label: str
    particle_key: str
    b_field_tesla: float
    dee_voltage_kv: float
    r_max_m: float
    is_synchro_modulated: bool
    is_isochronous_field: bool
    description: str


PRESETS: dict[str, CyclotronPreset] = {
    "CLASSICAL_PROTON_ACCELERATOR": CyclotronPreset(
        code="CLASSICAL_PROTON_ACCELERATOR",
        title_vi="Máy Gia tốc Cyclotron Cổ điển (Proton 16.5 MeV)",
        badge_label="CYCLOTRON CỔ ĐIỂN",
        particle_key="Proton",
        b_field_tesla=1.5,
        dee_voltage_kv=50.0,
        r_max_m=0.8,
        is_synchro_modulated=False,
        is_isochronous_field=False,
        description="Từ trường đều B = 1.5 T vuông góc mặt phẳng 2 cực chữ D (Dees). Tần số cộng hưởng f_c = 22.88 MHz không đổi giúp proton tăng tốc theo quỹ đạo xoắn ốc Archimedes tới 16.5 MeV.",
    ),
    "RELATIVISTIC_PHASE_SLIP_FAILURE": CyclotronPreset(
        code="RELATIVISTIC_PHASE_SLIP_FAILURE",
        title_vi="Hiện tượng Trượt Pha Tương đối tính (Mất Cộng hưởng)",
        badge_label="TRƯỢT PHA (MẤT CỘNG HƯỞNG)",
        particle_key="Proton",
        b_field_tesla=1.5,
        dee_voltage_kv=50.0,
        r_max_m=1.4,
        is_synchro_modulated=False,
        is_isochronous_field=False,
        description="Khi vận tốc đạt trên 0.15c, khối lượng tương đối tính tăng (m = γ·m₀) làm chu kỳ quay dài ra. Hạt trễ pha so với điện trường RF, bị hãm tốc và xoắn ngược vào trong.",
    ),
    "SYNCHROCYCLOTRON_RF_MODULATION": CyclotronPreset(
        code="SYNCHROCYCLOTRON_RF_MODULATION",
        title_vi="Synchrocyclotron: Biến điệu Tần số RF theo Khối lượng γ",
        badge_label="SYNCHROCYCLOTRON (>100 MEV)",
        particle_key="Proton",
        b_field_tesla=1.8,
        dee_voltage_kv=60.0,
        r_max_m=1.2,
        is_synchro_modulated=True,
        is_isochronous_field=False,
        description="Tần số điện trường f_rf(t) được giảm dần theo 1/γ(t), duy trì chính xác điều kiện cộng hưởng pha với proton, gia tốc hạt vượt mốc 100 MeV an toàn.",
    ),
    "DEUTERON_ALPHA_RADIOISOTOPE": CyclotronPreset(
        code="DEUTERON_ALPHA_RADIOISOTOPE",
        title_vi="Sản xuất Đồng vị Phóng xạ Y học Hạt nhân (Alpha ⁴He²⁺ & Đơteri)",
        badge_label="ĐỒNG VỊ Y HỌC PET",
        particle_key="Alpha",
        b_field_tesla=1.6,
        dee_voltage_kv=45.0,
        r_max_m=0.75,
        is_synchro_modulated=False,
        is_isochronous_field=False,
        description="Gia tốc hạt Alpha (q/m bằng 1/2 proton) với tần số 12.2 MHz để bắn phá bia tạo đồng vị phóng xạ Fluor-18 (¹⁸F) và Tecneti-99m phục vụ chụp cắt lớp PET.",
    ),
    "ISOCHRONOUS_AZIMUTHAL_FOCUSING": CyclotronPreset(
        code="ISOCHRONOUS_AZIMUTHAL_FOCUSING",
        title_vi="Cyclotron Đẳng thời (Isochronous): Từ trường B(r) = γ(r)·B₀",
        badge_label="CYCLOTRON ĐẲNG THỜI",
        particle_key="Carbon",
        b_field_tesla=2.0,
        dee_voltage_kv=70.0,
        r_max_m=1.0,
        is_synchro_modulated=False,
        is_isochronous_field=True,
        description="Từ trường tăng dần theo bán kính B(r) = γ(r)·B₀ kết hợp 4 góc nêm uốn sóng từ (magnetic flutter) để duy trì tiêu cự trục và tần số cộng hưởng cố định cho ion Cacbon ¹²C⁶⁺.",
    ),
}


class InteractiveCyclotronItem(QGraphicsObject):
    """
    Interactive Cyclotron Particle Accelerator & Relativistic Mechanics Simulator.
    Simulates magnetic Lorentz centripetal spiral, gap electric acceleration,
    relativistic mass increase, synchrocyclotron RF modulation, and radioisotope beam extraction.
    """

    preset_changed = Signal(str)
    particle_updated = Signal(float, float)  # radius_m, energy_mev
    stamp_requested = Signal(dict)

    CARD_W = 940.0
    CARD_H = 630.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation State
        self._active_preset_code: str = "CLASSICAL_PROTON_ACCELERATOR"
        self._preset: CyclotronPreset = PRESETS[self._active_preset_code]
        self._particle: ParticleSpec = PARTICLES[self._preset.particle_key]

        self._b_tesla: float = self._preset.b_field_tesla
        self._dee_voltage_kv: float = self._preset.dee_voltage_kv
        self._r_max_m: float = self._preset.r_max_m
        self._is_synchro: bool = self._preset.is_synchro_modulated
        self._is_isochronous: bool = self._preset.is_isochronous_field

        # Particle Trajectory State
        self._pos_x: float = 0.0             # meters (relative to center)
        self._pos_y: float = 0.0
        self._vx: float = 0.0                # m/s
        self._vy: float = 5.0e5              # initial seed velocity
        self._current_r_m: float = 0.005
        self._energy_mev: float = 0.01
        self._gamma: float = 1.0001
        self._is_extracted: bool = False

        # Trajectory History Buffer (screen coordinates for drawing spiral)
        self._spiral_pts: list[QPointF] = []

        # Playback control
        self._is_running: bool = True
        self._sim_time_ns: float = 0.0       # nanoseconds

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Run numerical integration for initial spiral
        self._simulate_initial_spiral()

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physics Calculations
    # -------------------------------------------------------------------------
    @property
    def particle(self) -> ParticleSpec:
        return self._particle

    @property
    def active_preset(self) -> CyclotronPreset:
        return self._preset

    @property
    def b_field_tesla(self) -> float:
        return self._b_tesla

    @property
    def dee_voltage_kv(self) -> float:
        return self._dee_voltage_kv

    @property
    def r_max_m(self) -> float:
        return self._r_max_m

    @property
    def current_radius_m(self) -> float:
        return self._current_r_m

    @property
    def current_energy_mev(self) -> float:
        return self._energy_mev

    @property
    def gamma_factor(self) -> float:
        return self._gamma

    def compute_cyclotron_frequency_mhz(self) -> float:
        """Calculates classical cyclotron resonance frequency f_c = q * B / (2π * m₀) in MHz."""
        f_hz = (self._particle.charge_c * self._b_tesla) / (2.0 * math.pi * self._particle.mass_kg)
        return f_hz * 1e-6

    def compute_max_extraction_energy_mev(self) -> float:
        """
        Calculates theoretical maximum extraction kinetic energy at R_max (Relativistic):
        E_k = sqrt(p_max² * c² + m₀² * c⁴) - m₀ * c².
        """
        p_max = self._particle.charge_c * self._b_tesla * self._r_max_m
        m0 = self._particle.mass_kg
        c = C_LIGHT

        total_e_joules = math.sqrt((p_max * c) ** 2 + (m0 * c ** 2) ** 2)
        ek_joules = total_e_joules - (m0 * c ** 2)
        return ek_joules / MEV_TO_JOULE

    def _simulate_initial_spiral(self) -> None:
        """Generates representative spiral trajectory for immediate visual inspection."""
        self.reset_particle()
        # Advance simulation by ~120 steps to draw full spiral
        for _ in range(120):
            self._step_cyclotron_physics(dt_ns=1.5)

    def reset_particle(self) -> None:
        """Resets particle to ion source in center gap."""
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._current_r_m = 0.01
        self._energy_mev = (self._particle.charge_c * self._dee_voltage_kv * 1e3) / MEV_TO_JOULE
        self._gamma = 1.0 + (self._energy_mev / self._particle.rest_energy_mev)
        v = C_LIGHT * math.sqrt(max(0.0, 1.0 - 1.0 / (self._gamma ** 2)))
        self._vx = 0.0
        self._vy = max(1.0e5, v)
        self._is_extracted = False
        self._sim_time_ns = 0.0
        self._spiral_pts = [QPointF(240.0, 240.0)]

    def _step_cyclotron_physics(self, dt_ns: float) -> None:
        """Integrates relativistic Lorentz force and gap acceleration."""
        if self._is_extracted:
            return

        q = self._particle.charge_c
        m0 = self._particle.mass_kg
        c = C_LIGHT

        # Effective magnetic field B(r)
        r = math.hypot(self._pos_x, self._pos_y)
        b_eff = self._b_tesla * (self._gamma if self._is_isochronous else 1.0)

        # Gap Acceleration: When particle crosses y-axis (|pos_x| < 0.01 m)
        gap_half_w = 0.008
        if abs(self._pos_x) < gap_half_w and not self._is_extracted:
            # Check phase synchronization
            f_rf_mhz = self.compute_cyclotron_frequency_mhz() / (self._gamma if self._is_synchro else 1.0)
            rf_phase = 2.0 * math.pi * (f_rf_mhz * 1e6) * (self._sim_time_ns * 1e-9)

            # Relativistic dephasing factor
            if not self._is_synchro and not self._is_isochronous and self._gamma > 1.06:
                phase_error = (self._gamma - 1.0) * math.pi * 8.0
                acceleration_factor = math.cos(phase_error)
            else:
                acceleration_factor = 1.0

            if acceleration_factor > -0.2:
                delta_e_joules = q * (self._dee_voltage_kv * 1e3) * max(0.1, acceleration_factor)
                delta_e_mev = delta_e_joules / MEV_TO_JOULE
                self._energy_mev += delta_e_mev
                self._gamma = 1.0 + (self._energy_mev / self._particle.rest_energy_mev)
            else:
                # Phase slip deceleration
                self._energy_mev = max(0.5, self._energy_mev - 0.5)
                self._gamma = 1.0 + (self._energy_mev / self._particle.rest_energy_mev)

        # Current velocity from gamma
        beta = math.sqrt(max(0.0, 1.0 - 1.0 / (self._gamma ** 2)))
        v_mag = beta * c

        # Lorentz centripetal curvature: r_curv = γ * m₀ * v / (q * B)
        r_curv = max(0.005, (self._gamma * m0 * v_mag) / (q * b_eff))
        self._current_r_m = r

        # Angular step
        omega = (q * b_eff) / (self._gamma * m0)
        dt_s = dt_ns * 1e-9
        d_theta = omega * dt_s

        # Update angle around center
        current_theta = math.atan2(self._pos_y, self._pos_x)
        next_theta = current_theta + d_theta
        self._pos_x = r_curv * math.cos(next_theta)
        self._pos_y = r_curv * math.sin(next_theta)

        self._sim_time_ns += dt_ns

        # Map to screen coordinates: center at (240.0, 240.0), scale 180 px = r_max
        scale_px_per_m = 180.0 / self._r_max_m
        screen_x = 240.0 + self._pos_x * scale_px_per_m
        screen_y = 240.0 - self._pos_y * scale_px_per_m
        self._spiral_pts.append(QPointF(screen_x, screen_y))

        # Check extraction boundary
        if r >= self._r_max_m:
            self._is_extracted = True
            # Add extraction channel straight trajectory to target
            self._spiral_pts.append(QPointF(screen_x + 50.0, screen_y))

        # Keep history length bounded
        if len(self._spiral_pts) > 600:
            self._spiral_pts.pop(0)

        self.particle_updated.emit(self._current_r_m, self._energy_mev)

    # -------------------------------------------------------------------------
    # Presets & Parameter Setters
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads designated cyclotron curriculum preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._particle = PARTICLES[self._preset.particle_key]
        self._b_tesla = self._preset.b_field_tesla
        self._dee_voltage_kv = self._preset.dee_voltage_kv
        self._r_max_m = self._preset.r_max_m
        self._is_synchro = self._preset.is_synchro_modulated
        self._is_isochronous = self._preset.is_isochronous_field

        self._simulate_initial_spiral()
        self.preset_changed.emit(preset_code)
        self.update()

    def set_particle(self, particle_key: str) -> None:
        """Switches accelerated particle species."""
        if particle_key in PARTICLES:
            self._particle = PARTICLES[particle_key]
            self._simulate_initial_spiral()
            self.update()

    def step_b_field(self, delta_t: float) -> None:
        """Adjusts magnetic field B in Tesla."""
        self._b_tesla = max(0.5, min(3.5, round(self._b_tesla + delta_t, 2)))
        self._simulate_initial_spiral()
        self.update()

    def step_dee_voltage(self, delta_kv: float) -> None:
        """Adjusts accelerating peak voltage across Dees in kV."""
        self._dee_voltage_kv = max(10.0, min(150.0, round(self._dee_voltage_kv + delta_kv, 1)))
        self._simulate_initial_spiral()
        self.update()

    def toggle_synchro(self) -> None:
        """Toggles synchrocyclotron RF frequency modulation."""
        self._is_synchro = not self._is_synchro
        self._simulate_initial_spiral()
        self.update()

    def toggle_animation(self) -> None:
        """Toggles simulation playback."""
        self._is_running = not self._is_running
        self.update()

    def _on_sim_tick(self) -> None:
        if not self._is_running:
            return
        # Advance 2 integration steps per frame
        self._step_cyclotron_physics(dt_ns=1.2)
        if self._is_extracted:
            # Re-fire particle loop gently
            self.reset_particle()
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card & Acrylic Frame
        card_rect = self.boundingRect()
        bg_brush = QBrush(QColor(10, 15, 29, 245))
        border_pen = QPen(QColor(56, 189, 248, 200) if self.isSelected() else QColor(30, 41, 59, 220), 1.8)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header & Badges
        self._paint_header(painter)

        # 3. Left Panel: Cyclotron Dees Chamber & Spiral Trajectory
        self._paint_dees_chamber(painter)

        # 4. Right Panel: Relativistic Dynamics & Resonance Analysis
        self._paint_dynamics_panel(painter)

        # 5. Bottom Digital Readout Bar
        self._paint_readout_bar(painter)

        # 6. Interactive In-Air Controls Bar
        self._paint_controls_bar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QPen(QColor("#f8fafc")))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 28, "⚛️ Máy Gia tốc Hạt Cyclotron & Hiệu ứng Khối lượng Tương đối tính")

        # Badge
        badge_text = self._preset.badge_label
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        badge_rect = QRectF(590.0, 12.0, 195.0, 22.0)
        painter.setBrush(QBrush(QColor(14, 165, 233, 50)))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # Subtitle / Description
        painter.setPen(QPen(QColor("#94a3b8")))
        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.drawText(20, 46, self._preset.description)

    def _paint_dees_chamber(self, painter: QPainter) -> None:
        """Renders the two metallic D-shaped electrodes (Dees), accelerating gap, and spiral trajectory."""
        chamber_rect = QRectF(20.0, 56.0, 440.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(chamber_rect, 10.0, 10.0)

        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(34, 78, "BUỒNG CHÂN KHÔNG & HAI CỰC CHỮ D (DEES)")

        cx = 240.0
        cy = 240.0
        r_dee = 160.0
        gap_w = 12.0

        # Magnetic field direction symbol: ⊙ (Vector B pointing out of screen)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(34, 105, f"Từ trường: ⊙ B = {self._b_tesla:.2f} T (Hướng vuông góc ra ngoài)")

        # 1. Left Dee (D1)
        left_dee_path = QPainterPath()
        left_dee_path.arcMoveTo(cx - r_dee, cy - r_dee, 2 * r_dee, 2 * r_dee, 90.0)
        left_dee_path.arcTo(cx - r_dee, cy - r_dee, 2 * r_dee, 2 * r_dee, 90.0, 180.0)
        left_dee_path.closeSubpath()

        painter.save()
        painter.translate(-gap_w / 2.0, 0.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.setPen(QPen(QColor("#0284c7"), 2.0))
        painter.drawPath(left_dee_path)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(cx - 100, cy - 80, "Cực D₁ (~ -V)")
        painter.restore()

        # 2. Right Dee (D2)
        right_dee_path = QPainterPath()
        right_dee_path.arcMoveTo(cx - r_dee, cy - r_dee, 2 * r_dee, 2 * r_dee, -90.0)
        right_dee_path.arcTo(cx - r_dee, cy - r_dee, 2 * r_dee, 2 * r_dee, -90.0, 180.0)
        right_dee_path.closeSubpath()

        painter.save()
        painter.translate(gap_w / 2.0, 0.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.setPen(QPen(QColor("#0284c7"), 2.0))
        painter.drawPath(right_dee_path)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(cx + 40, cy - 80, "Cực D₂ (~ +V)")
        painter.restore()

        # 3. Accelerating Gap Electric Field Arrows
        painter.setPen(QPen(QColor(245, 158, 11, 180), 1.5, Qt.PenStyle.DashLine))
        for gy in range(int(cy - 120), int(cy + 130), 40):
            painter.drawLine(QPointF(cx - gap_w / 2.0, gy), QPointF(cx + gap_w / 2.0, gy))
            # Arrowhead
            painter.drawLine(QPointF(cx + gap_w / 2.0, gy), QPointF(cx + gap_w / 2.0 - 4.0, gy - 3.0))
            painter.drawLine(QPointF(cx + gap_w / 2.0, gy), QPointF(cx + gap_w / 2.0 - 4.0, gy + 3.0))

        # Central Ion Source
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)

        # 4. Extraction Channel & Deflector Plate (Periphery)
        painter.setPen(QPen(QColor("#ef4444"), 2.5))
        painter.drawArc(cx - r_dee - 8.0, cy - r_dee - 8.0, 2 * (r_dee + 8.0), 2 * (r_dee + 8.0), int(-40 * 16), int(35 * 16))

        # Target Badge
        target_rect = QRectF(cx + r_dee + 15.0, cy - 25.0, 48.0, 50.0)
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(QPen(QColor("#ef4444"), 1.5))
        painter.drawRoundedRect(target_rect, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(target_rect, Qt.AlignmentFlag.AlignCenter, "BIA BẮN\n(TARGET)")

        # 5. Particle Spiral Trajectory
        if len(self._spiral_pts) > 1:
            p_color = QColor(self._particle.color_hex)
            spiral_path = QPainterPath()
            spiral_path.moveTo(self._spiral_pts[0])
            for pt in self._spiral_pts[1:]:
                spiral_path.lineTo(pt)

            painter.setPen(QPen(p_color, 2.2))
            painter.drawPath(spiral_path)

            # Draw glowing current particle head
            head_pt = self._spiral_pts[-1]
            painter.setBrush(QBrush(p_color))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(head_pt, 5.0, 5.0)

    def _paint_dynamics_panel(self, painter: QPainter) -> None:
        """Renders relativistic factor γ, kinetic energy progression, and RF frequency synchronization."""
        dyn_rect = QRectF(480.0, 56.0, 440.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(dyn_rect, 10.0, 10.0)

        # Panel Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(494, 78, "ĐỘNG HỌC TƯƠNG ĐỐI TÍNH & ĐIỀU KIỆN CỘNG HƯỞNG RF")

        # 1. Relativistic Mass / Gamma Display
        g_box = QRectF(494.0, 95.0, 412.0, 140.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 200)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(g_box, 6.0, 6.0)

        beta = math.sqrt(max(0.0, 1.0 - 1.0 / (self._gamma ** 2)))
        v_kms = (beta * C_LIGHT) / 1000.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(506, 118, f"HỆ SỐ LORENTZ γ = 1 / √(1 - v²/c²) = {self._gamma:.4f}")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(506, 138, f"• Vận tốc tức thời v: {v_kms:,.0f} km/s ({beta * 100.0:.2f}% tốc độ ánh sáng c)")
        painter.drawText(506, 156, f"• Khối lượng tương đối tính: m = γ·m₀ = {self._gamma * self._particle.mass_kg:.3e} kg")
        painter.drawText(506, 174, f"• Năng lượng nghỉ: E₀ = m₀c² = {self._particle.rest_energy_mev:.2f} MeV")
        painter.drawText(506, 192, f"• Động năng hiện tại: E_k = (γ - 1)·E₀ = {self._energy_mev:.3f} MeV")
        painter.drawText(506, 210, f"• Bán kính quỹ đạo: r = p / (qB) = {self._current_r_m * 100.0:.2f} cm (R_max = {self._r_max_m * 100.0:.0f} cm)")

        # 2. RF Resonance & Phase Matching
        rf_box = QRectF(494.0, 248.0, 412.0, 184.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 200)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(rf_box, 6.0, 6.0)

        f_c_mhz = self.compute_cyclotron_frequency_mhz()
        f_rf_now = f_c_mhz / (self._gamma if self._is_synchro else 1.0)
        e_max_theo = self.compute_max_extraction_energy_mev()

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(506, 272, "ĐIỀU KIỆN CỘNG HƯỞNG TẦN SỐ RF:")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(506, 294, f"• Tần số Cyclotron cổ điển f_c = qB / (2πm₀) = {f_c_mhz:.3f} MHz")
        painter.drawText(506, 314, f"• Tần số máy phát RF hiện tại f_rf: {f_rf_now:.3f} MHz")

        synchro_state = "BẬT (Khắc phục trượt pha)" if self._is_synchro else "TẮT (Cố định f_c)"
        painter.drawText(506, 334, f"• Biến điệu Synchrocyclotron: {synchro_state}")
        painter.drawText(506, 354, f"• Động năng cực đại tại bia: E_k,max = {e_max_theo:.2f} MeV")

        # Dephasing Warning Indicator
        if not self._is_synchro and not self._is_isochronous and self._gamma > 1.06:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ef4444")))
            painter.drawText(506, 382, "⚠️ CẢNH BÁO: Trượt pha tương đối tính! Hạt mất đồng bộ với điện trường.")
            painter.drawText(506, 400, "Cần chuyển sang chế độ Synchrocyclotron để bù độ dãn thời gian.")
        else:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#10b981")))
            painter.drawText(506, 382, "✓ ĐỒNG BỘ PHA TỐI ƯU: Hạt nhận đủ năng lượng q·V mỗi lần qua khe Dees.")
            painter.drawText(506, 400, f"Hiệu điện thế cực D: V₀ = {self._dee_voltage_kv:.1f} kV (ΔE = {self._particle.charge_c / E_CHARGE * self._dee_voltage_kv:.1f} keV/vòng)")

    def _paint_readout_bar(self, painter: QPainter) -> None:
        """Draws digital parameter and formula values."""
        bar_rect = QRectF(20.0, 456.0, 900.0, 68.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bar_rect, 8.0, 8.0)

        f_c = self.compute_cyclotron_frequency_mhz()
        e_max = self.compute_max_extraction_energy_mev()

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(36, 478, f"HẠT GIA TỐC: {self._particle.name_vi} | Điện tích: q = +{self._particle.charge_c / E_CHARGE:.0f}e")
        painter.drawText(36, 498, f"CÔNG THỨC LORENTZ: r = p / (qB) = {self._current_r_m * 100.0:.2f} cm | B = {self._b_tesla:.2f} T")
        painter.drawText(36, 516, f"TẦN SỐ CỘNG HƯỞNG ISOCHRONOUS: f_c = qB / (2πm₀) = {f_c:.2f} MHz")

        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(480, 478, f"ĐỘNG NĂNG HIỆN TẠI: E_k = {self._energy_mev:.3f} MeV (Vận tốc: {math.sqrt(max(0.0, 1.0 - 1.0/(self._gamma**2))) * 100.0:.1f}% c)")
        painter.drawText(480, 498, f"ĐỘNG NĂNG CỰC ĐẠI TRÍCH XUẤT: E_k,max = {e_max:.2f} MeV (R_max = {self._r_max_m:.2f} m)")
        painter.drawText(480, 516, f"ĐIỆN THẾ GIA TỐC: V_dee = {self._dee_voltage_kv:.1f} kV | Khối lượng γ: {self._gamma:.4f} m₀")

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        bar_y = 536.0
        self._buttons.clear()

        btn_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(btn_font)

        # 1. Preset Selector Button
        preset_rect = QRectF(20.0, bar_y, 160.0, 32.0)
        self._draw_btn(painter, preset_rect, "btn_preset", f"Presets: {self._preset.badge_label}")

        # 2. Particle Switcher
        part_rect = QRectF(188.0, bar_y, 90.0, 32.0)
        self._draw_btn(painter, part_rect, "btn_particle", f"Hạt: {self._particle.symbol}")

        # 3. B-Field Steppers
        self._draw_btn(painter, QRectF(286.0, bar_y, 65.0, 32.0), "btn_b_minus", "-0.2 T")
        self._draw_btn(painter, QRectF(357.0, bar_y, 65.0, 32.0), "btn_b_plus", "+0.2 T")

        # 4. Voltage Steppers
        self._draw_btn(painter, QRectF(430.0, bar_y, 70.0, 32.0), "btn_v_minus", "-10 kV")
        self._draw_btn(painter, QRectF(506.0, bar_y, 70.0, 32.0), "btn_v_plus", "+10 kV")

        # 5. Synchro Mode Toggle
        synchro_label = "RF: Synchro (Bật)" if self._is_synchro else "RF: Cố định (Tắt)"
        self._draw_btn(painter, QRectF(584.0, bar_y, 120.0, 32.0), "btn_synchro", synchro_label)

        # 6. Play / Pause
        play_label = "⏸ Tạm dừng" if self._is_running else "▶ Tiếp tục"
        self._draw_btn(painter, QRectF(712.0, bar_y, 80.0, 32.0), "btn_play", play_label)

        # 7. Stamp to Canvas Button
        stamp_rect = QRectF(800.0, bar_y, 85.0, 32.0)
        self._draw_btn(painter, stamp_rect, "btn_stamp", "📌 Đóng dấu")

        # 8. Close Button (Top right)
        close_rect = QRectF(self.CARD_W - 38.0, 10.0, 26.0, 26.0)
        self._draw_btn(painter, close_rect, "btn_close", "✕")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, label: str) -> None:
        self._buttons[key] = rect
        is_hovered = self._hovered_btn == key

        btn_bg = QColor(30, 41, 59, 220) if not is_hovered else QColor(2, 132, 199, 220)
        btn_pen = QPen(QColor("#38bdf8") if is_hovered else QColor(51, 65, 85), 1.2)

        painter.setBrush(QBrush(btn_bg))
        painter.setPen(btn_pen)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Mouse Events
    # -------------------------------------------------------------------------
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
        self._hovered_btn = None
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
            self._is_dragging = True
            self._last_mouse_pos = event.scenePos()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging:
            delta = event.scenePos() - self._last_mouse_pos
            self._last_mouse_pos = event.scenePos()
            self.setPos(self.pos() + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging = False
        super().mouseReleaseEvent(event)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(PRESETS.keys())
        particle_keys = list(PARTICLES.keys())

        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_particle":
            curr_p_idx = particle_keys.index(self._particle.code)
            next_p_idx = (curr_p_idx + 1) % len(particle_keys)
            self.set_particle(particle_keys[next_p_idx])
        elif key == "btn_b_minus":
            self.step_b_field(-0.2)
        elif key == "btn_b_plus":
            self.step_b_field(0.2)
        elif key == "btn_v_minus":
            self.step_dee_voltage(-10.0)
        elif key == "btn_v_plus":
            self.step_dee_voltage(10.0)
        elif key == "btn_synchro":
            self.toggle_synchro()
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        elif key == "btn_close":
            self.setVisible(False)

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
        """Emits signal requesting permanent vector stamping of current state onto canvas."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 640.0,
            "height": 340.0,
            "preset_code": self._active_preset_code,
            "particle_name": self._particle.name_vi,
            "b_field_tesla": self._b_tesla,
            "dee_voltage_kv": self._dee_voltage_kv,
            "freq_mhz": self.compute_cyclotron_frequency_mhz(),
            "max_energy_mev": self.compute_max_extraction_energy_mev(),
            "r_max_m": self._r_max_m,
        }
        self.stamp_requested.emit(stamp_data)
