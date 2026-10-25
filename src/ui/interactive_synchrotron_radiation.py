"""
AirTeach AI - Interactive Synchrotron Radiation & Relativistic Undulator Simulator
===================================================================================
Authentic Relativistic Electrodynamics, Lienard-Wiechert Retarded Potentials & Insertion Devices:
- Relativistic Kinematics:
    γ = E / (m_e * c^2) = 1 / sqrt(1 - β^2)
    Bending radius in dipole field B0: ρ = γ * m_e * c / (e * B0)
- Relativistic Larmor Formula & Forward Beaming Cone:
    Total radiated power: P_rad = (e^2 * c / (6 * π * ε0)) * (γ^4 / ρ^2)
    Razor-thin relativistic emission opening half-angle:
      θ_cone ≈ 1 / γ  (e.g., ~0.17 mrad for 3 GeV storage ring)
- Critical Photon Energy & Spectrum:
    ħ * ω_c = (3/2) * (ħ * c * γ^3 / ρ) = (3/2) * (e * ħ * B0 * γ^2 / m_e)
    Half of total synchrotron power emitted above ħ*ω_c, half below.
- Periodic Insertion Devices (Undulator vs Wiggler):
    Alternating periodic magnetic field: B_y(z) = B0 * cos(2*π*z / λ_u)
    Deflection parameter: K = e * B0 * λ_u / (2 * π * m_e * c) ≈ 0.9337 * B0[T] * λ_u[cm]
    * Undulator regime (K <= 1): Trajectory angle <= 1/γ. Coherent N-period interference!
      Sharp harmonic emission at:
        λ_n = (λ_u / (2 * n * γ^2)) * (1 + K^2 / 2 + γ^2 * θ^2)
    * Wiggler regime (K >> 1): Trajectory angle >> 1/γ. Continuous spectrum with 2N-fold flux.
- 5 Curriculum Presets:
    1. BENDING_MAGNET: Storage ring dipole, broad continuum from infrared to hard X-rays.
    2. PLANAR_UNDULATOR: Low-K device (K = 1.2, N = 60), razor-sharp monochromatic harmonics.
    3. MULTIPOLE_WIGGLER: High-K device (K = 15, B0 = 2.0 T), boosted hard X-ray continuum.
    4. FREE_ELECTRON_LASER: SASE FEL microbunching, ultra-intense coherent femtosecond pulses.
    5. LOW_ENERGY_SYNCHROTRON: 1.5 GeV ring, VUV / soft X-rays for photoemission spectroscopy.
- Touchless webcam air gestures (air-tap, sliders, angle deflection), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
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

# Fundamental physical constants
SPEED_OF_LIGHT: float = 299792458.0          # m/s
ELECTRON_MASS_EV: float = 510998.95           # eV/c^2
ELECTRON_MASS_KG: float = 9.1093837e-31       # kg
E_CHARGE: float = 1.602176634e-19            # C
EPSILON_0: float = 8.8541878128e-12          # F/m
PLANCK_HBAR_EV: float = 6.582119569e-16       # eV * s


@dataclass
class SynchrotronPreset:
    """Preset configuration for educational synchrotron & undulator experiments."""
    key: str
    name: str
    desc: str
    facility: str
    energy_gev: float
    device_type: str  # "bending_magnet", "undulator", "wiggler", "fel"
    b0_tesla: float
    period_cm: float
    num_periods: int
    default_angle_mrad: float


PRESETS: List[SynchrotronPreset] = [
    SynchrotronPreset(
        key="bending_magnet",
        name="Nam châm uốn cong (Bending Dipole)",
        desc="Dipole uốn chùm electron 3 GeV, bức xạ hãm synchrotron phát hình nón cực hẹp 1/γ, phổ liên tục từ IR tới Hard X-ray",
        facility="Máy gia tốc vòng bức xạ (3.0 GeV)",
        energy_gev=3.0,
        device_type="bending_magnet",
        b0_tesla=1.2,
        period_cm=0.0,
        num_periods=1,
        default_angle_mrad=0.0,
    ),
    SynchrotronPreset(
        key="planar_undulator",
        name="Bộ Undulator phẳng (Coherent Harmonics)",
        desc="Thiết bị chèn chu kỳ ngắn K = 1.2 <= 1. Giao thoa tăng cường tạo các đỉnh phổ điều hòa đơn sắc cực kỳ sắc nét",
        facility="Undulator tia X mềm (Soft X-Ray Beamline)",
        energy_gev=3.0,
        device_type="undulator",
        b0_tesla=0.45,
        period_cm=2.8,
        num_periods=60,
        default_angle_mrad=0.0,
    ),
    SynchrotronPreset(
        key="multipole_wiggler",
        name="Bộ Wiggler đa cực (Hard X-Ray High-K)",
        desc="Hệ nam châm từ trường cao K = 15 >> 1, bẻ cong góc lớn hơn 1/γ, tăng cường độ thông lượng phổ cứng lên gấp 2N lần",
        facility="Wiggler tinh thể học (Hard X-Ray Wiggler)",
        energy_gev=3.0,
        device_type="wiggler",
        b0_tesla=1.8,
        period_cm=8.5,
        num_periods=30,
        default_angle_mrad=0.0,
    ),
    SynchrotronPreset(
        key="free_electron_laser",
        name="Laser Electron Tự do (X-ray FEL SASE)",
        desc="Cơ chế vi bó điện tử tự khuếch đại SASE sinh chùm xung laser tia X kết hợp kết tụ cực sáng độ rộng xung femto-giây",
        facility="XFEL Siêu sáng (Free Electron Laser)",
        energy_gev=6.0,
        device_type="fel",
        b0_tesla=0.8,
        period_cm=3.0,
        num_periods=120,
        default_angle_mrad=0.0,
    ),
    SynchrotronPreset(
        key="low_energy_synchrotron",
        name="Máy gia tốc VUV Đại học (1.5 GeV)",
        desc="Vòng trữ năng lượng 1.5 GeV chuyên dụng cho quang phổ phát xạ quang điện tử XPS và nghiên cứu vật liệu màng mỏng",
        facility="Nguồn sáng VUV / Soft X-Ray (1.5 GeV)",
        energy_gev=1.5,
        device_type="undulator",
        b0_tesla=0.6,
        period_cm=4.2,
        num_periods=40,
        default_angle_mrad=0.0,
    ),
]


class InteractiveSynchrotronRadiationItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Synchrotron Radiation, Relativistic Larmor formula,
    retarded potential radiation cones, Undulators, Wigglers, and X-ray emission spectra.
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
        self._preset_index: int = 1  # Default to Undulator
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        p = PRESETS[self._preset_index]
        self._beam_energy_gev: float = p.energy_gev
        self._b0_tesla: float = p.b0_tesla
        self._period_cm: float = p.period_cm
        self._num_periods: int = p.num_periods
        self._obs_angle_mrad: float = p.default_angle_mrad

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance continuous electron beam pulses."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

    @property
    def current_preset(self) -> SynchrotronPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active synchrotron radiation preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._beam_energy_gev = p.energy_gev
            self._b0_tesla = p.b0_tesla
            self._period_cm = p.period_cm
            self._num_periods = p.num_periods
            self._obs_angle_mrad = p.default_angle_mrad
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_beam_energy(self, e_gev: float) -> None:
        """Set storage ring electron beam energy in GeV."""
        self._beam_energy_gev = max(0.5, min(e_gev, 10.0))
        self.property_changed.emit("energy_gev", self._beam_energy_gev)
        self.update()

    def set_magnetic_field(self, b_tesla: float) -> None:
        """Set peak magnetic field in Tesla."""
        self._b0_tesla = max(0.05, min(b_tesla, 3.0))
        self.property_changed.emit("b0_tesla", self._b0_tesla)
        self.update()

    def set_observation_angle(self, theta_mrad: float) -> None:
        """Set observation off-axis angle in milliradians."""
        self._obs_angle_mrad = max(0.0, min(theta_mrad, 10.0))
        self.property_changed.emit("angle_mrad", self._obs_angle_mrad)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset defaults."""
        self.set_preset(self._preset_index)

    # -------------------------------------------------------------------------
    # Authentic Relativistic Electrodynamics Calculations
    # -------------------------------------------------------------------------

    def compute_gamma(self) -> float:
        """Compute relativistic Lorentz factor γ = E / (m_e * c^2)."""
        e_ev = self._beam_energy_gev * 1e9
        return float(e_ev / ELECTRON_MASS_EV)

    def compute_beaming_angle(self) -> float:
        """Compute characteristic forward radiation cone half-angle θ_cone ≈ 1 / γ (mrad)."""
        gamma = self.compute_gamma()
        theta_rad = 1.0 / max(1.0, gamma)
        return float(theta_rad * 1000.0)  # in milliradians

    def compute_bending_radius(self) -> float:
        """Compute bending radius ρ = p / (e * B) ≈ E[GeV] / (0.2998 * B[T]) in meters."""
        return float(self._beam_energy_gev / (0.299792458 * max(0.01, self._b0_tesla)))

    def compute_critical_energy(self) -> float:
        """
        Compute critical photon energy in keV:
        E_c = (3/2) * (ħ * c * γ^3 / ρ) ≈ 0.665 * E[GeV]^2 * B[T].
        """
        e_gev = self._beam_energy_gev
        b_t = self._b0_tesla
        return float(0.6650 * (e_gev**2) * b_t)  # keV

    def compute_undulator_k(self) -> float:
        """Compute dimensionless deflection parameter K ≈ 0.9337 * B0[T] * λ_u[cm]."""
        if self._period_cm <= 0.0:
            return 0.0
        return float(0.9337 * self._b0_tesla * self._period_cm)

    def compute_fundamental_harmonic_energy(self) -> float:
        """
        Compute fundamental (n=1) undulator harmonic photon energy in keV:
        E_1 ≈ 0.950 * (E[GeV]^2) / ( λ_u[cm] * (1 + K^2 / 2 + γ^2 * θ^2) )
        """
        k = self.compute_undulator_k()
        gamma = self.compute_gamma()
        theta_rad = self._obs_angle_mrad * 1e-3
        denom = max(0.1, self._period_cm * (1.0 + (k**2) / 2.0 + (gamma * theta_rad)**2))
        return float(0.9496 * (self._beam_energy_gev**2) / denom)

    def step_physics(self, dt: float) -> None:
        """Advance simulation time for electron wavepacket propagation."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, 56.0)

    def _trajectory_canvas_rect(self) -> QRectF:
        return QRectF(15.0, 65.0, 520.0, 420.0)

    def _spectrum_panel_rect(self) -> QRectF:
        return QRectF(545.0, 65.0, 380.0, 420.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, SynchrotronPreset]]:
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
        bg_brush = QBrush(QColor(11, 15, 25, 245))
        border_pen = QPen(QColor(168, 85, 247, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_trajectory_canvas(painter)
        self._paint_spectrum_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ BỨC XẠ SYNCHROTRON & BỘ CHÈN UNDULATOR TƯƠNG ĐỐI TÍNH")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        gamma = self.compute_gamma()
        theta_c = self.compute_beaming_angle()
        painter.drawText(20, 44, f"Cơ sở: {p.facility}  |  Năng lượng E = {self._beam_energy_gev:.1f} GeV (γ ≈ {gamma:,.0f})  |  Góc nón θ_c ≈ {theta_c:.3f} mrad")

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(168, 85, 247, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(192, 132, 252), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            labels = ["Dipole uốn", "Undulator", "Wiggler", "Laser XFEL", "VUV 1.5GeV"]
            label = labels[i] if i < len(labels) else preset.name[:8]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        k_val = self.compute_undulator_k()
        badge_text = f"K = {k_val:.2f} ({'Undulator' if k_val <= 1.0 else 'Wiggler'})" if self._period_cm > 0 else "Bending Magnet"

        painter.setBrush(QColor(168, 85, 247, 40))
        painter.setPen(QPen(QColor(192, 132, 252), 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(216, 180, 254))
        painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

    def _paint_trajectory_canvas(self, painter: QPainter) -> None:
        """Render Left Panel: Relativistic Electron Orbit & Narrow Beaming Cone."""
        c_rect = self._trajectory_canvas_rect()

        # Canvas Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(c_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(c_rect.left(), c_rect.top() + 12.0, c_rect.width(), 20.0), Qt.AlignCenter, "🔬 QUỸ ĐẠO ELECTRON & NÓN PHÁT XẠ ĐỊNH HƯỚNG 1/γ")

        p = self.current_preset
        is_insertion = p.device_type in ["undulator", "wiggler", "fel"]

        # Trajectory Geometry
        traj_path = QPainterPath()
        start_x = c_rect.left() + 40.0
        center_y = c_rect.center().y()
        end_x = c_rect.right() - 80.0

        if not is_insertion:
            # Circular Arc (Bending Magnet)
            # Arc bends downwards
            traj_path.moveTo(start_x, center_y - 40.0)
            ctrl_x = (start_x + end_x) / 2.0
            ctrl_y = center_y + 10.0
            traj_path.quadTo(ctrl_x, ctrl_y, end_x, center_y + 70.0)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248), 2.5))
            painter.drawPath(traj_path)

            # Dipole magnet pole faces (Green/Yellow blocks)
            dipole_rect = QRectF(ctrl_x - 50.0, ctrl_y - 65.0, 100.0, 80.0)
            painter.setBrush(QColor(16, 185, 129, 40))
            painter.setPen(QPen(QColor(16, 185, 129), 1.2, Qt.DashLine))
            painter.drawRoundedRect(dipole_rect, 6.0, 6.0)

            # Tangential Synchrotron Radiation Fan
            tan_x = ctrl_x + 10.0
            tan_y = ctrl_y + 12.0

            fan_path = QPainterPath()
            fan_path.moveTo(tan_x, tan_y)
            fan_path.lineTo(c_rect.right() - 20.0, tan_y - 25.0)
            fan_path.lineTo(c_rect.right() - 20.0, tan_y + 45.0)
            fan_path.closeSubpath()

            fan_grad = QLinearGradient(tan_x, tan_y, c_rect.right() - 20.0, tan_y)
            fan_grad.setColorAt(0.0, QColor(250, 204, 21, 180))
            fan_grad.setColorAt(0.7, QColor(244, 63, 94, 90))
            fan_grad.setColorAt(1.0, QColor(168, 85, 247, 10))
            painter.setBrush(fan_grad)
            painter.setPen(Qt.NoPen)
            painter.drawPath(fan_path)

        else:
            # Sinusoidal Undulator Trajectory
            k_val = self.compute_undulator_k()
            amp = min(18.0, max(2.5, k_val * 2.2))
            num_wiggles = 7
            w_step = (end_x - start_x) / (num_wiggles * 2.0)

            traj_path.moveTo(start_x, center_y)
            for wi in range(num_wiggles * 2):
                nx = start_x + (wi + 1) * w_step
                ny = center_y + (amp if (wi % 2 == 0) else -amp)
                traj_path.lineTo(nx, ny)

            # Draw Alternating Permanent Magnet Halbach Array (Top & Bottom)
            pole_w = w_step * 1.8
            pole_h = 24.0
            for pi in range(num_wiggles * 2):
                px = start_x + pi * w_step
                is_north = (pi % 2 == 0)
                # Top pole
                painter.setBrush(QColor(225, 29, 72, 140) if is_north else QColor(37, 99, 235, 140))
                painter.setPen(Qt.NoPen)
                painter.drawRect(QRectF(px, center_y - 42.0, pole_w, pole_h))
                # Bottom pole (opposite polarity)
                painter.setBrush(QColor(37, 99, 235, 140) if is_north else QColor(225, 29, 72, 140))
                painter.drawRect(QRectF(px, center_y + 18.0, pole_w, pole_h))

            # Draw electron trajectory
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248), 2.2))
            painter.drawPath(traj_path)

            # Highly collimated forward emission beam (Laser-like X-ray pencil beam)
            beam_start_x = end_x
            beam_rect = QRectF(beam_start_x, center_y - 5.0, c_rect.right() - beam_start_x - 15.0, 10.0)
            beam_grad = QLinearGradient(beam_rect.left(), center_y, beam_rect.right(), center_y)
            beam_grad.setColorAt(0.0, QColor(250, 204, 21, 240))
            beam_grad.setColorAt(0.4, QColor(168, 85, 247, 200))
            beam_grad.setColorAt(1.0, QColor(56, 189, 248, 40))
            painter.setBrush(beam_grad)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(beam_rect, 4.0, 4.0)

        # Telemetry Text
        gamma = self.compute_gamma()
        theta_c = self.compute_beaming_angle()
        rho = self.compute_bending_radius()
        ec_kev = self.compute_critical_energy()

        hud_y = c_rect.bottom() - 65.0
        lines = [
            f"Bán kính uốn: ρ = {rho:.2f} m  |  Góc mở nón phát xạ: 1/γ = {theta_c:.3f} mrad",
            f"Năng lượng photon tới hạn: Ec = {ec_kev:.2f} keV  (λc ≈ {1.2398 / max(0.01, ec_kev):.3f} Å)",
            f"Thông số chèn K = {self.compute_undulator_k():.2f}  |  Chu kỳ λu = {self._period_cm:.2f} cm (N = {self._num_periods})",
        ]

        painter.setFont(QFont("Segoe UI", 8))
        for line in lines:
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(c_rect.left() + 20.0), int(hud_y), line)
            hud_y += 18.0

    def _paint_spectrum_panel(self, painter: QPainter) -> None:
        """Render Right Panel: Synchrotron Flux vs Photon Energy Spectrum."""
        rect = self._spectrum_panel_rect()

        # Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "📊 PHỔ BỨC XẠ SYNCHROTRON dP/dω (SPECTRUM)")

        # Plot boundary
        plot_rect = QRectF(rect.left() + 25.0, rect.top() + 45.0, rect.width() - 40.0, 260.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(plot_rect)

        # Grid lines (log scale approximation)
        painter.setPen(QPen(QColor(30, 41, 59), 1.0, Qt.DashLine))
        for gy in [0.25, 0.5, 0.75]:
            y_pos = plot_rect.top() + gy * plot_rect.height()
            painter.drawLine(int(plot_rect.left()), int(y_pos), int(plot_rect.right()), int(y_pos))

        p = self.current_preset
        is_undulator = (p.device_type == "undulator" or p.device_type == "fel")
        spec_path = QPainterPath()
        num_pts = 80

        if not is_undulator:
            # Broad Continuous Synchrotron Spectrum (Universal Function S(x))
            # Peak near 0.29 * Ec
            for i in range(num_pts + 1):
                norm_e = (i / num_pts) * 3.5  # E / Ec from 0 to 3.5
                px = plot_rect.left() + (i / num_pts) * plot_rect.width()

                # S(x) approximation ≈ 1.33 * x^(1/3) * exp(-x)
                if norm_e < 1e-4:
                    val = 0.0
                else:
                    val = 1.33 * (norm_e**(1.0 / 3.0)) * math.exp(-norm_e)
                    val = min(1.0, val / 0.45)  # normalize peak to ~1.0

                py = plot_rect.bottom() - val * (plot_rect.height() * 0.88)
                if i == 0:
                    spec_path.moveTo(px, py)
                else:
                    spec_path.lineTo(px, py)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(250, 204, 21), 2.2))
            painter.drawPath(spec_path)

            # Mark Critical Energy Ec
            ec_x = plot_rect.left() + (1.0 / 3.5) * plot_rect.width()
            painter.setPen(QPen(QColor(239, 68, 68), 1.5, Qt.DashLine))
            painter.drawLine(int(ec_x), int(plot_rect.top()), int(ec_x), int(plot_rect.bottom()))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(239, 68, 68))
            painter.drawText(QRectF(ec_x - 30.0, plot_rect.top() + 6.0, 60.0, 16.0), Qt.AlignCenter, "Ec (Tới hạn)")

        else:
            # Undulator Sharp Harmonics Spectrum (Harmonics n=1, n=3, n=5)
            # Baseline continuum
            e1_kev = self.compute_fundamental_harmonic_energy()
            spec_path.moveTo(plot_rect.left(), plot_rect.bottom())

            for i in range(num_pts + 1):
                px = plot_rect.left() + (i / num_pts) * plot_rect.width()
                norm_x = (i / num_pts) * 4.0  # units of E / E1

                # Peaks at n=1, n=3, n=5 (sync shape)
                peak_1 = 0.95 * math.exp(-((norm_x - 1.0) / 0.08)**2)
                peak_3 = 0.45 * math.exp(-((norm_x - 3.0) / 0.08)**2)
                base = 0.04 * (norm_x**0.5) * math.exp(-norm_x * 0.5)
                val = min(0.98, peak_1 + peak_3 + base)

                py = plot_rect.bottom() - val * (plot_rect.height() * 0.9)
                spec_path.lineTo(px, py)

            painter.setBrush(QColor(168, 85, 247, 40))
            painter.setPen(QPen(QColor(192, 132, 252), 2.0))
            painter.drawPath(spec_path)

            # Harmonic 1 label
            e1_px = plot_rect.left() + (1.0 / 4.0) * plot_rect.width()
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(250, 204, 21))
            painter.drawText(QRectF(e1_px - 40.0, plot_rect.top() + 15.0, 80.0, 16.0), Qt.AlignCenter, f"H1: {e1_kev:.2f} keV")

        # Telemetry & Physics Notes
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        ty = plot_rect.bottom() + 18.0
        lines = [
            f"Trục hoành: Năng lượng photon E (keV)  |  Trục tung: Thông lượng quang tử",
            f"Chế độ: {'Đơn sắc sóng điều hòa Coherent' if is_undulator else 'Phổ liên tục Broad Continuum'}",
            f"Góc lệch trục đo: θ_obs = {self._obs_angle_mrad:.2f} mrad",
        ]
        for line in lines:
            painter.drawText(int(rect.left() + 20.0), int(ty), line)
            ty += 16.0

    def _paint_controls(self, painter: QPainter) -> None:
        ctrl_rect = self._controls_rect()

        # Controls Panel Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(ctrl_rect, 10.0, 10.0)

        col_w = (ctrl_rect.width() - 40.0) / 4.0

        sliders = [
            ("energy", "Năng lượng chùm e- (GeV)", f"{self._beam_energy_gev:.2f} GeV", self._beam_energy_gev, 0.5, 8.0),
            ("b0", "Từ trường đỉnh B₀ (T)", f"{self._b0_tesla:.2f} T", self._b0_tesla, 0.1, 2.5),
            ("angle", "Góc quan sát θ (mrad)", f"{self._obs_angle_mrad:.2f} mrad", self._obs_angle_mrad, 0.0, 5.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(192, 132, 252))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill
            norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(168, 85, 247))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(168, 85, 247), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Reset button in 4th column
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        painter.setBrush(QColor(168, 85, 247, 40))
        painter.setPen(QPen(QColor(192, 132, 252), 1.2))
        painter.drawRoundedRect(btn_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(192, 132, 252))
        painter.drawText(btn_rect, Qt.AlignCenter, "🔄 ĐẶT LẠI BAN ĐẦU")

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

        # Check Reset Button
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        if btn_rect.contains(pos):
            self.reset_simulation()
            event.accept()
            return

        # Check Sliders
        for i, key in enumerate(["energy", "b0", "angle"]):
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
            slider_keys = ["energy", "b0", "angle"]
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
        if key == "energy":
            self.set_beam_energy(0.5 + norm * 7.5)
        elif key == "b0":
            self.set_magnetic_field(0.1 + norm * 2.4)
        elif key == "angle":
            self.set_observation_angle(norm * 5.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveSynchrotronRadiationItem",
            "preset_key": self.current_preset.key,
            "beam_energy_gev": self._beam_energy_gev,
            "b0_tesla": self._b0_tesla,
            "obs_angle_mrad": self._obs_angle_mrad,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveSynchrotronRadiationItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "planar_undulator")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "beam_energy_gev" in data:
            item._beam_energy_gev = float(data["beam_energy_gev"])
        if "b0_tesla" in data:
            item._b0_tesla = float(data["b0_tesla"])
        if "obs_angle_mrad" in data:
            item._obs_angle_mrad = float(data["obs_angle_mrad"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
