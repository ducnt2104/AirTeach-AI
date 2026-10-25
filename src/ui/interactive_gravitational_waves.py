"""
AirTeach AI - Interactive Astrophysics: Gravitational Waves, Binary Inspiral & Interferometry
=============================================================================================
Authentic Nobel Prize in Physics (2017) Gravitational Wave Astronomy Foundations:
- Compact Binary Coalescence (CBC):
    Chirp mass:
      M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
    Total mass:
      M_tot = m1 + m2,  Symmetric mass ratio: η = (m1 * m2) / M_tot^2
    Innermost Stable Circular Orbit (ISCO) frequency:
      f_ISCO = c^3 / (6^(3/2) * π * G * M_tot)
- Gravitational Wave Chirp Dynamics (Post-Newtonian leading order):
    Time to coalescence τ = t_merger - t:
      f_GW(t) = (1 / π) * [ (5 / 256) * c^5 / (G^(5/3) * M_chirp^(5/3) * τ) ]^(3/8)
      h_amp(t) = (4 / D_L) * (G * M_chirp / c^2)^(5/3) * (π * f_GW / c)^(2/3)
      Φ_GW(t) = -2 * [ c^3 * τ / (5 * G * M_chirp) ]^(5/8) + Φ_0
      h_plus(t) = h_amp(t) * 0.5 * (1 + cos^2(ι)) * cos(Φ_GW(t))
- Black Hole Merger & Quasi-Normal Mode (QNM) Ringdown:
    Remnant mass: M_rem = (1 - ε_rad) * M_tot
    Remnant spin: a_rem ≈ 0.69
    Ringdown frequency: f_qnm ≈ (c^3 / (2π * G * M_rem)) * [1 - 0.63 * (1 - a_rem)^0.3]
    Damping time: τ_damp ≈ 4 * (1 - a_rem)^(-0.45) / (2π * f_qnm)
    h_ring(t) = h_peak * exp(-(t - t_merger) / τ_damp) * cos(2π * f_qnm * (t - t_merger))
- Laser Interferometry (LIGO / Virgo):
    Differential arm strain: ΔL(t) = L * h(t)  (L = 4 km)
    Phase shift: ΔΦ(t) = 4π * L * h(t) / λ_laser  (λ = 1064 nm)
- 5 Authentic Historic Presets:
    1. GW150914: First direct detection, 36.2 M_sun + 29.1 M_sun BBH, 410 Mpc, 3 M_sun radiated.
    2. GW170817: Binary Neutron Star (BNS) merger & kilonova, 1.46 M_sun + 1.27 M_sun, 40 Mpc.
    3. GW190521: Intermediate-mass black hole merger, 85 M_sun + 66 M_sun -> 142 M_sun.
    4. GW200105: Neutron Star - Black Hole (NSBH) merger, 8.9 M_sun + 1.9 M_sun, 280 Mpc.
    5. LISA_SMBH: Space-based detector, supermassive binary 10^6 M_sun + 10^6 M_sun, mHz band.
- Touchless webcam air gestures (air-tap, mass/distance sliders, audio-visual chirp play),
  bimanual manipulation, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
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

# Fundamental physical and astronomical constants
SPEED_OF_LIGHT: float = 299792458.0          # m/s
GRAVITATIONAL_CONST: float = 6.67430e-11     # m^3 / (kg * s^2)
SOLAR_MASS: float = 1.98847e30               # kg
PARSEC: float = 3.085677581e16               # m
MEGAPARSEC: float = PARSEC * 1e6             # m
LIGO_ARM_LENGTH: float = 4000.0              # m (4 km)
LASER_WAVELENGTH: float = 1064e-9            # m (1064 nm Nd:YAG)


@dataclass
class GravitationalWavePreset:
    """Preset configuration for historical Gravitational Wave events."""
    key: str
    name: str
    desc: str
    event_type: str
    mass1_msun: float
    mass2_msun: float
    dist_mpc: float
    radiated_energy_msun: float
    peak_strain_scale: float
    duration_sec: float


PRESETS: List[GravitationalWavePreset] = [
    GravitationalWavePreset(
        key="gw150914",
        name="GW150914: Sáp nhập 2 Lỗ đen Khối lượng Sao (LIGO 2015)",
        desc="Phát hiện sóng hấp dẫn trực tiếp đầu tiên lịch sử: 36.2 M☉ + 29.1 M☉ -> 62.3 M☉, phát xạ 3.0 M☉c²",
        event_type="Binary Black Hole (BBH)",
        mass1_msun=36.2,
        mass2_msun=29.1,
        dist_mpc=410.0,
        radiated_energy_msun=3.0,
        peak_strain_scale=1.0e-21,
        duration_sec=0.25,
    ),
    GravitationalWavePreset(
        key="gw170817",
        name="GW170817: Sáp nhập Sao Neutron & Kilonova Đa tín hiệu",
        desc="Sáp nhập 2 sao neutron 1.46 M☉ + 1.27 M☉ phát ra sóng hấp dẫn kéo dài hơn 60s kèm chớp Gamma và quang học",
        event_type="Binary Neutron Star (BNS)",
        mass1_msun=1.46,
        mass2_msun=1.27,
        dist_mpc=40.0,
        radiated_energy_msun=0.04,
        peak_strain_scale=3.5e-22,
        duration_sec=0.8,
    ),
    GravitationalWavePreset(
        key="gw190521",
        name="GW190521: Lỗ đen Khối lượng Trung bình (85 M☉ + 66 M☉)",
        desc="Sáp nhập trong 'khe hở khối lượng' tạo ra lỗ đen 142 M☉, giải phóng 8 M☉ năng lượng hấp dẫn",
        event_type="Intermediate Mass Black Hole",
        mass1_msun=85.0,
        mass2_msun=66.0,
        dist_mpc=5300.0,
        radiated_energy_msun=8.0,
        peak_strain_scale=8.0e-22,
        duration_sec=0.15,
    ),
    GravitationalWavePreset(
        key="gw200105",
        name="GW200105: Sáp nhập Sao Neutron - Lỗ đen (NSBH)",
        desc="Lỗ đen 8.9 M☉ nuốt chửng sao neutron 1.9 M☉, minh họa hiện tượng xé rách thủy triều thiên văn",
        event_type="Neutron Star - Black Hole (NSBH)",
        mass1_msun=8.9,
        mass2_msun=1.9,
        dist_mpc=280.0,
        radiated_energy_msun=0.5,
        peak_strain_scale=5.0e-22,
        duration_sec=0.35,
    ),
    GravitationalWavePreset(
        key="lisa_smbh",
        name="LISA: Sáp nhập Lỗ đen Siêu khối lượng (10⁶ M☉ + 10⁶ M☉)",
        desc="Sóng hấp dẫn tần số milli-Hertz từ tâm thiên hà dự kiến phát hiện bởi kính thiên văn không gian ESA/NASA LISA",
        event_type="Supermassive Binary (LISA)",
        mass1_msun=1.0e6,
        mass2_msun=1.0e6,
        dist_mpc=10000.0,
        radiated_energy_msun=1.0e5,
        peak_strain_scale=1.0e-19,
        duration_sec=1.5,
    ),
]


class InteractiveGravitationalWavesItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Gravitational Wave Interferometry,
    Compact Binary Inspiral, Chirp Mass Calculation, and QNM Ringdown.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 940.0
    HEIGHT: float = 600.0

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        p = PRESETS[self._preset_index]
        self._mass1_msun: float = p.mass1_msun
        self._mass2_msun: float = p.mass2_msun
        self._dist_mpc: float = p.dist_mpc
        self._disp_duration: float = p.duration_sec

        # Interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance binary orbital motion and strain replay."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources defensively."""
        try:
            if hasattr(self, "_anim_timer") and self._anim_timer is not None and self._anim_timer.isActive():
                self._anim_timer.stop()
        except (RuntimeError, ReferenceError):
            pass

    @property
    def current_preset(self) -> GravitationalWavePreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active gravitational wave preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._mass1_msun = p.mass1_msun
            self._mass2_msun = p.mass2_msun
            self._dist_mpc = p.dist_mpc
            self._disp_duration = p.duration_sec
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_mass1(self, m1: float) -> None:
        """Set component mass m1 in Solar Masses."""
        self._mass1_msun = max(1.0, min(m1, 2.0e6))
        self.property_changed.emit("mass1_msun", self._mass1_msun)
        self.update()

    def set_mass2(self, m2: float) -> None:
        """Set component mass m2 in Solar Masses."""
        self._mass2_msun = max(1.0, min(m2, 2.0e6))
        self.property_changed.emit("mass2_msun", self._mass2_msun)
        self.update()

    def set_distance_mpc(self, d_mpc: float) -> None:
        """Set luminosity distance in Megaparsecs."""
        self._dist_mpc = max(5.0, min(d_mpc, 20000.0))
        self.property_changed.emit("dist_mpc", self._dist_mpc)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset defaults."""
        self.set_preset(self._preset_index)

    # -------------------------------------------------------------------------
    # Authentic General Relativity & Gravitational Wave Physics Calculations
    # -------------------------------------------------------------------------

    def compute_total_mass_msun(self) -> float:
        """M_tot = m1 + m2 in solar masses."""
        return float(self._mass1_msun + self._mass2_msun)

    def compute_chirp_mass_msun(self) -> float:
        """
        Compute authentic chirp mass:
        M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
        """
        m1 = self._mass1_msun
        m2 = self._mass2_msun
        m_chirp = ((m1 * m2) ** 0.6) / ((m1 + m2) ** 0.2)
        return float(m_chirp)

    def compute_symmetric_mass_ratio(self) -> float:
        """η = (m1 * m2) / (m1 + m2)^2  (max 0.25 for equal masses)."""
        m1 = self._mass1_msun
        m2 = self._mass2_msun
        return float((m1 * m2) / ((m1 + m2) ** 2))

    def compute_isco_frequency_hz(self) -> float:
        """
        Innermost Stable Circular Orbit (ISCO) GW frequency:
        f_ISCO = c^3 / (6^(3/2) * π * G * M_tot)
        (≈ 4396 Hz / (M_tot / M_sun)).
        """
        m_tot_kg = self.compute_total_mass_msun() * SOLAR_MASS
        f_isco = (SPEED_OF_LIGHT ** 3) / (
            (6.0 ** 1.5) * math.pi * GRAVITATIONAL_CONST * m_tot_kg
        )
        return float(f_isco)

    def compute_ringdown_frequency_hz(self) -> float:
        """
        Quasi-Normal Mode (QNM) dominant ringdown frequency (l=2, m=2):
        f_ring ≈ (c^3 / (2π * G * M_rem)) * [1 - 0.63 * (1 - a_rem)^0.3]
        Assuming typical remnant spin a_rem ≈ 0.69 and ~5% radiated mass.
        """
        m_rem_kg = self.compute_total_mass_msun() * 0.95 * SOLAR_MASS
        a_rem = 0.69
        factor = 1.0 - 0.63 * ((1.0 - a_rem) ** 0.3)
        f_ring = ((SPEED_OF_LIGHT ** 3) / (2.0 * math.pi * GRAVITATIONAL_CONST * m_rem_kg)) * factor
        return float(f_ring)

    def compute_ringdown_damping_time_sec(self) -> float:
        """
        Ringdown exponential damping time:
        τ_damp ≈ 4 * (1 - a_rem)^(-0.45) / (2π * f_ring)
        """
        f_ring = self.compute_ringdown_frequency_hz()
        a_rem = 0.69
        tau = (4.0 * ((1.0 - a_rem) ** -0.45)) / (2.0 * math.pi * max(1e-3, f_ring))
        return float(tau)

    def compute_strain_at_time(self, t_rel: float) -> Tuple[float, float]:
        """
        Compute authentic GW strain h(t) and instantaneous GW frequency f(t)
        at time t_rel relative to merger (t_rel < 0: inspiral, t_rel >= 0: ringdown).
        Returns: (h(t), f_gw(t))
        """
        m_chirp_kg = self.compute_chirp_mass_msun() * SOLAR_MASS
        d_m = self._dist_mpc * MEGAPARSEC
        f_isco = self.compute_isco_frequency_hz()
        f_ring = self.compute_ringdown_frequency_hz()
        tau_damp = self.compute_ringdown_damping_time_sec()

        t_disp_total = self._disp_duration
        t_merger_offset = t_disp_total * 0.8
        tau = max(1e-4, t_merger_offset - t_rel)

        if t_rel < t_merger_offset:
            # 1. Post-Newtonian Inspiral Regime
            # f_gw(tau) = (1 / π) * [ (5/256) * c^5 / (G^(5/3) * M^(5/3) * tau) ]^(3/8)
            coeff = (5.0 / 256.0) * (SPEED_OF_LIGHT ** 5) / (
                (GRAVITATIONAL_CONST ** (5.0 / 3.0)) * (m_chirp_kg ** (5.0 / 3.0)) * tau
            )
            f_gw = (1.0 / math.pi) * (coeff ** 0.375)
            f_gw = min(f_gw, f_isco * 1.2)

            # Amplitude: h = (4 / D_L) * (G*M/c^2)^(5/3) * (π*f/c)^(2/3)
            gm_c2 = (GRAVITATIONAL_CONST * m_chirp_kg) / (SPEED_OF_LIGHT ** 2)
            pi_f_c = (math.pi * f_gw) / SPEED_OF_LIGHT
            h_amp = (4.0 / d_m) * (gm_c2 ** (5.0 / 3.0)) * (pi_f_c ** (2.0 / 3.0))

            # Phase: Φ(tau) = -2 * [ c^3 * tau / (5 * G * M) ]^(5/8)
            phase_arg = (SPEED_OF_LIGHT ** 3) * tau / (5.0 * GRAVITATIONAL_CONST * m_chirp_kg)
            phi = -2.0 * (max(0.0, phase_arg) ** 0.625)
            h_val = h_amp * math.cos(phi)
            return float(h_val), float(f_gw)
        else:
            # 2. Merger & Quasi-Normal Mode (QNM) Ringdown
            dt_ring = t_rel - t_merger_offset
            gm_c2 = (GRAVITATIONAL_CONST * m_chirp_kg) / (SPEED_OF_LIGHT ** 2)
            pi_f_c = (math.pi * f_isco) / SPEED_OF_LIGHT
            h_peak = (4.0 / d_m) * (gm_c2 ** (5.0 / 3.0)) * (pi_f_c ** (2.0 / 3.0))

            decay = math.exp(-min(30.0, dt_ring / max(1e-4, tau_damp)))
            h_val = h_peak * decay * math.cos(2.0 * math.pi * f_ring * dt_ring)
            return float(h_val), float(f_ring)

    def compute_arm_displacement_meters(self, h_strain: float) -> float:
        """
        Interferometer arm differential displacement:
        ΔL = L_arm * h(t)  (for L_arm = 4 km, ΔL ~ 10^-18 m, 1/1000th proton diameter).
        """
        return float(LIGO_ARM_LENGTH * h_strain)

    def compute_optical_phase_shift_rad(self, h_strain: float) -> float:
        """
        Optical phase shift in Nd:YAG laser:
        ΔΦ = 4π * ΔL / λ_laser
        """
        dl = self.compute_arm_displacement_meters(h_strain)
        return float(4.0 * math.pi * dl / LASER_WAVELENGTH)

    def step_physics(self, dt: float) -> None:
        """Advance time through binary orbital spiral and strain replay."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        cycle = self._disp_duration * 1.3
        if self._sim_time > cycle:
            self._sim_time = 0.0
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(15.0, 10.0, 910.0, 60.0)

    def _interferometer_rect(self) -> QRectF:
        return QRectF(15.0, 75.0, 280.0, 410.0)

    def _waveform_rect(self) -> QRectF:
        return QRectF(305.0, 75.0, 320.0, 410.0)

    def _spectrogram_rect(self) -> QRectF:
        return QRectF(635.0, 75.0, 290.0, 410.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, GravitationalWavePreset]]:
        h_rect = self._header_rect()
        btn_w = 95.0
        btn_h = 28.0
        gap = 6.0
        start_x = 350.0
        y = (h_rect.height() - btn_h) / 2.0
        rects = []
        for i, p in enumerate(PRESETS):
            bx = start_x + i * (btn_w + gap)
            rects.append((QRectF(bx, y, btn_w, btn_h), p))
        return rects

    def paint(
        self,
        painter: QPainter,
        option: Optional[QStyleOptionGraphicsItem] = None,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Main Background Card
        bg_brush = QBrush(QColor(8, 12, 22, 248))
        border_pen = QPen(QColor(56, 189, 248, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_interferometer_panel(painter)
        self._paint_waveform_panel(painter)
        self._paint_spectrogram_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ SÓNG HẤP DẪN & GIAO THOA KẾ LASER (LIGO / VIRGO)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        m_chirp = self.compute_chirp_mass_msun()
        f_isco = self.compute_isco_frequency_hz()
        painter.drawText(
            20,
            44,
            f"Mô hình: {p.event_type}  |  Khối lượng Chirp ℳ = {m_chirp:.2f} M☉  |  Tần số ISCO: {f_isco:.1f} Hz  |  Khoảng cách: {self._dist_mpc:.0f} Mpc",
        )

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(14, 165, 233, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(56, 189, 248), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(148, 163, 184))
            label = preset.key.upper()
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        painter.setBrush(QColor(14, 165, 233, 40))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        curr_t = min(self._sim_time, self._disp_duration)
        curr_h, _ = self.compute_strain_at_time(curr_t)
        painter.drawText(badge_rect, Qt.AlignCenter, f"h: {curr_h:.1e}")

    def _paint_interferometer_panel(self, painter: QPainter) -> None:
        """Render Left Panel: Binary Black Hole Orbit & Michelson Interferometer Arms."""
        rect = self._interferometer_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "🔭 QUỸ ĐẠO CBC & GIAO THOA KẾ")

        # 1. Binary Orbit View (Upper half)
        orbit_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 170.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(orbit_rect, 8.0, 8.0)

        cx = orbit_rect.center().x()
        cy = orbit_rect.center().y()

        # Dynamic orbital radius shrinking as merger approaches
        t_now = min(self._sim_time, self._disp_duration)
        t_merger = self._disp_duration * 0.8
        tau = max(0.0, t_merger - t_now)
        frac = min(1.0, tau / (self._disp_duration * 0.8))
        r_orbit = 14.0 + 55.0 * (frac ** 0.25)

        # Orbital frequency increases
        omega = 3.0 + 15.0 * ((1.0 - frac) ** 1.5)
        angle = self._sim_time * omega

        # Concentric Gravitational Wave Ripples in Spacetime
        painter.setBrush(Qt.NoBrush)
        for ring_i in range(5):
            phase_offset = (self._sim_time * 4.0 - ring_i * 18.0) % 80.0
            r_ripple = r_orbit + phase_offset
            alpha = max(0, int(180 * (1.0 - phase_offset / 80.0)))
            painter.setPen(QPen(QColor(56, 189, 248, alpha), 1.2, Qt.DashLine))
            painter.drawEllipse(QPointF(cx, cy), r_ripple, r_ripple * 0.6)

        # Binary bodies
        m1_rad = max(4.0, min(14.0, 5.0 + math.log10(max(1.0, self._mass1_msun)) * 2.5))
        m2_rad = max(4.0, min(14.0, 5.0 + math.log10(max(1.0, self._mass2_msun)) * 2.5))

        x1 = cx + r_orbit * math.cos(angle)
        y1 = cy + (r_orbit * 0.6) * math.sin(angle)
        x2 = cx - r_orbit * math.cos(angle)
        y2 = cy - (r_orbit * 0.6) * math.sin(angle)

        # Body 1 (Cyan)
        painter.setBrush(QColor(56, 189, 248))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(x1, y1), m1_rad, m1_rad)

        # Body 2 (Orange)
        painter.setBrush(QColor(251, 146, 60))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(x2, y2), m2_rad, m2_rad)

        # 2. Michelson Interferometer Diagram (Lower half)
        ifo_rect = QRectF(rect.left() + 15.0, rect.top() + 215.0, rect.width() - 30.0, 180.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(ifo_rect, 8.0, 8.0)

        ix = ifo_rect.left() + 45.0
        iy = ifo_rect.bottom() - 45.0

        # Laser Source -> Beam Splitter
        painter.setPen(QPen(QColor(239, 68, 68), 2.0))  # 1064 nm Laser beam
        painter.drawLine(QPointF(ix - 25.0, iy), QPointF(ix, iy))

        # Laser box
        painter.setBrush(QColor(220, 38, 38))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(ix - 35.0, iy - 6.0, 10.0, 12.0))

        # Beam Splitter (45 deg mirror)
        painter.setPen(QPen(QColor(241, 245, 249), 2.5))
        painter.drawLine(QPointF(ix - 8.0, iy + 8.0), QPointF(ix + 8.0, iy - 8.0))

        # Instantaneous strain strain stretch/squeeze on X and Y arms
        curr_h, _ = self.compute_strain_at_time(t_now)
        arm_len = 120.0
        strain_visual = (curr_h / max(1e-23, self.current_preset.peak_strain_scale)) * 8.0
        arm_x = arm_len + strain_visual
        arm_y = arm_len - strain_visual

        # X-Arm (Horizontal)
        painter.setPen(QPen(QColor(239, 68, 68), 1.8))
        painter.drawLine(QPointF(ix, iy), QPointF(ix + arm_x, iy))
        # End mirror X
        painter.setPen(QPen(QColor(96, 165, 250), 3.0))
        painter.drawLine(QPointF(ix + arm_x, iy - 12.0), QPointF(ix + arm_x, iy + 12.0))

        # Y-Arm (Vertical)
        painter.setPen(QPen(QColor(239, 68, 68), 1.8))
        painter.drawLine(QPointF(ix, iy), QPointF(ix, iy - arm_y))
        # End mirror Y
        painter.setPen(QPen(QColor(96, 165, 250), 3.0))
        painter.drawLine(QPointF(ix - 12.0, iy - arm_y), QPointF(ix + 12.0, iy - arm_y))

        # Recombined beam to photodiode
        painter.setPen(QPen(QColor(250, 204, 21), 2.0))
        painter.drawLine(QPointF(ix, iy), QPointF(ix, iy + 25.0))
        painter.setBrush(QColor(234, 179, 8))
        painter.drawEllipse(QPointF(ix, iy + 28.0), 5.0, 5.0)

        # Interferometer Arm Readout HUD
        dl_m = self.compute_arm_displacement_meters(curr_h)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(ix + 20.0, iy - 60.0, f"Nhánh X: L = 4.0 km")
        painter.drawText(ix + 20.0, iy - 45.0, f"ΔL: {dl_m:+.2e} m")
        painter.drawText(ix + 20.0, iy - 30.0, f"(~1/1000 proton)")

    def _paint_waveform_panel(self, painter: QPainter) -> None:
        """Render Center Panel: Authentic Time-Domain Waveform h(t) [Inspiral, Merger, Ringdown]."""
        rect = self._waveform_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "📈 BIÊN ĐỘ BIẾN DẠNG h(t) (CHIRP)")

        # Waveform Plot Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        # Grid lines
        mid_y = box_rect.center().y()
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(box_rect.left(), mid_y), QPointF(box_rect.right(), mid_y))

        t_total = self._disp_duration
        t_merger = t_total * 0.8
        peak_scale = max(1e-23, self.current_preset.peak_strain_scale)

        # Draw full waveform curve
        pts = []
        n_samples = 250
        for i in range(n_samples):
            t_val = (i / (n_samples - 1)) * t_total
            h_val, _ = self.compute_strain_at_time(t_val)
            norm_h = max(-1.0, min(1.0, h_val / peak_scale))
            px = box_rect.left() + (i / (n_samples - 1)) * box_rect.width()
            py = mid_y - norm_h * (box_rect.height() * 0.42)
            pts.append(QPointF(px, py))

        path = QPainterPath()
        if pts:
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawPath(path)

        # Merger line
        mx = box_rect.left() + (t_merger / t_total) * box_rect.width()
        painter.setPen(QPen(QColor(244, 63, 94, 180), 1.2, Qt.DashLine))
        painter.drawLine(QPointF(mx, box_rect.top()), QPointF(mx, box_rect.bottom()))

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(244, 63, 94))
        painter.drawText(mx + 4.0, box_rect.top() + 14.0, "SÁP NHẬP (MERGER)")

        # Current Playback Scrubber Cursor
        t_curr = min(self._sim_time, t_total)
        cx = box_rect.left() + (t_curr / t_total) * box_rect.width()
        painter.setPen(QPen(QColor(250, 204, 21), 1.8))
        painter.drawLine(QPointF(cx, box_rect.top()), QPointF(cx, box_rect.bottom()))

        # Live strain readout
        curr_h, curr_f = self.compute_strain_at_time(t_curr)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(250, 204, 21))
        painter.drawText(box_rect.left() + 10.0, box_rect.bottom() - 25.0, f"f_GW: {curr_f:.1f} Hz")
        painter.drawText(box_rect.left() + 10.0, box_rect.bottom() - 10.0, f"h(t): {curr_h:+.2e}")

    def _paint_spectrogram_panel(self, painter: QPainter) -> None:
        """Render Right Panel: Time-Frequency Chirp Spectrogram (t vs f)."""
        rect = self._spectrogram_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "📊 PHỔ THỜI GIAN-TẦN SỐ (SPECTROGRAM)")

        # Spectrogram Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        t_total = self._disp_duration
        f_max = max(50.0, self.compute_ringdown_frequency_hz() * 1.3)

        # Draw Chirp Track (f proportional to (t_merger - t)^(-3/8))
        chirp_path = QPainterPath()
        n_pts = 150
        first = True
        for i in range(n_pts):
            t_val = (i / (n_pts - 1)) * t_total
            _, f_val = self.compute_strain_at_time(t_val)
            px = box_rect.left() + (t_val / t_total) * box_rect.width()
            norm_f = max(0.0, min(1.0, f_val / f_max))
            py = box_rect.bottom() - norm_f * box_rect.height()
            if first:
                chirp_path.moveTo(px, py)
                first = False
            else:
                chirp_path.lineTo(px, py)

        # Glow trail
        painter.setPen(QPen(QColor(244, 63, 94, 80), 6.0))
        painter.drawPath(chirp_path)
        painter.setPen(QPen(QColor(251, 191, 36), 2.2))
        painter.drawPath(chirp_path)

        # Axis annotations
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(box_rect.left() + 8.0, box_rect.top() + 16.0, f"{f_max:.0f} Hz")
        painter.drawText(box_rect.left() + 8.0, box_rect.bottom() - 8.0, "0 Hz")
        painter.drawText(box_rect.right() - 40.0, box_rect.bottom() - 8.0, f"{t_total:.2f} s")

        # Current state indicator dot
        t_curr = min(self._sim_time, t_total)
        _, curr_f = self.compute_strain_at_time(t_curr)
        dot_x = box_rect.left() + (t_curr / t_total) * box_rect.width()
        dot_y = box_rect.bottom() - max(0.0, min(1.0, curr_f / f_max)) * box_rect.height()
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(250, 204, 21), 2.0))
        painter.drawEllipse(QPointF(dot_x, dot_y), 5.0, 5.0)

    def _paint_controls(self, painter: QPainter) -> None:
        """Render Bottom Control Bar: Sliders (m1, m2, d_L) and Playback Buttons."""
        rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        col_w = (rect.width() - 40.0) / 4.0

        # Slider 1: Mass 1 (M☉)
        c1_x = rect.left() + 20.0
        c1_y = rect.top() + 18.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(c1_x, c1_y, f"Khối lượng m₁: {self._mass1_msun:.1f} M☉")

        track_y = c1_y + 18.0
        track_w = col_w - 25.0
        painter.setPen(QPen(QColor(51, 65, 85), 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(c1_x, track_y), QPointF(c1_x + track_w, track_y))
        norm1 = max(0.0, min(1.0, math.log10(max(1.0, self._mass1_msun)) / 6.0))
        handle1_x = c1_x + norm1 * track_w
        painter.setBrush(QColor(56, 189, 248))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(handle1_x, track_y), 7.0, 7.0)

        # Slider 2: Mass 2 (M☉)
        c2_x = rect.left() + 20.0 + col_w
        c2_y = c1_y
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(251, 146, 60))
        painter.drawText(c2_x, c2_y, f"Khối lượng m₂: {self._mass2_msun:.1f} M☉")

        painter.setPen(QPen(QColor(51, 65, 85), 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(c2_x, track_y), QPointF(c2_x + track_w, track_y))
        norm2 = max(0.0, min(1.0, math.log10(max(1.0, self._mass2_msun)) / 6.0))
        handle2_x = c2_x + norm2 * track_w
        painter.setBrush(QColor(251, 146, 60))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(handle2_x, track_y), 7.0, 7.0)

        # Slider 3: Luminosity Distance (Mpc)
        c3_x = rect.left() + 20.0 + 2 * col_w
        c3_y = c1_y
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(167, 139, 250))
        painter.drawText(c3_x, c3_y, f"Khoảng cách D_L: {self._dist_mpc:.0f} Mpc")

        painter.setPen(QPen(QColor(51, 65, 85), 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(c3_x, track_y), QPointF(c3_x + track_w, track_y))
        norm3 = max(0.0, min(1.0, (self._dist_mpc - 5.0) / 10000.0))
        handle3_x = c3_x + norm3 * track_w
        painter.setBrush(QColor(167, 139, 250))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(handle3_x, track_y), 7.0, 7.0)

        # Buttons (Col 4): Reset & Pause
        c4_x = rect.left() + 20.0 + 3 * col_w
        btn_w = (col_w - 30.0) / 2.0
        btn1_rect = QRectF(c4_x, rect.top() + 22.0, btn_w, 38.0)
        btn2_rect = QRectF(c4_x + btn_w + 10.0, rect.top() + 22.0, btn_w, 38.0)

        # Reset button
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(btn1_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(btn1_rect, Qt.AlignCenter, "🔄 ĐẶT LẠI")

        # Pause / Play button
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(250, 204, 21), 1.2))
        painter.drawRoundedRect(btn2_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        p_txt = "▶ TIẾP TỤC" if self._is_paused else "⏸ TẠM DỪNG"
        painter.drawText(btn2_rect, Qt.AlignCenter, p_txt)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Controls buttons
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        c4_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_w = (col_w - 30.0) / 2.0
        btn1_rect = QRectF(c4_x, ctrl_rect.top() + 22.0, btn_w, 38.0)
        btn2_rect = QRectF(c4_x + btn_w + 10.0, ctrl_rect.top() + 22.0, btn_w, 38.0)

        if btn1_rect.contains(pos):
            self.reset_simulation()
            event.accept()
            return
        if btn2_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Sliders
        for i, key in enumerate(["mass1", "mass2", "dist"]):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["mass1", "mass2", "dist"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        if key == "mass1":
            log_m = norm * 6.0
            self.set_mass1(10.0 ** log_m)
        elif key == "mass2":
            log_m = norm * 6.0
            self.set_mass2(10.0 ** log_m)
        elif key == "dist":
            self.set_distance_mpc(5.0 + norm * 10000.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveGravitationalWavesItem",
            "preset_key": self.current_preset.key,
            "mass1_msun": self._mass1_msun,
            "mass2_msun": self._mass2_msun,
            "dist_mpc": self._dist_mpc,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveGravitationalWavesItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "gw150914")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "mass1_msun" in data:
            item._mass1_msun = float(data["mass1_msun"])
        if "mass2_msun" in data:
            item._mass2_msun = float(data["mass2_msun"])
        if "dist_mpc" in data:
            item._dist_mpc = float(data["dist_mpc"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
