"""
AirTeach AI – Interactive Physics & Optics: Michelson Interferometer & Gravitational Wave Detector (LIGO)
==========================================================================================================
Provides an authentic physics and astrophysics simulation of the Michelson Interferometer
and the LIGO Laser Interferometer Gravitational-Wave Observatory:
  - Optical Path Difference (OPD):
      Δd = 2 * (L₂ - L₁)
      Phase Difference:
      ΔΦ(θ) = (2π / λ) * 2 * Δd * cos(θ) + π
  - Haidinger Circular Fringes (Vân tròn đồng tâm):
      Formed when mirror M₁ is perpendicular to mirror M₂ (virtual mirror M₁' is parallel to M₂).
      Central fringe condition: 2 * Δd = m * λ.
      Fringe intensity distribution:
      I(r) = (I_max / 2) * [1 + V * cos((2π / λ) * 2 * Δd * cos(θ) + π)]
      where θ ≈ r / f.
  - Fizeau Wedge Fringes (Vân nêm định vị):
      Formed when mirror M₂ is tilted by small angle α relative to M₁'.
      Linear fringe spacing:
      Δx = λ / (2 * α)
  - Fringe Shift Counting (Đo bước sóng & vi dịch chuyển):
      ΔN = (2 * Δd) / λ  <=>  Δd = ΔN * (λ / 2)
  - Coherence Length & White Light Zero-Path Envelope:
      Coherence Length: L_c = λ₀² / Δλ
      Visibility envelope: V(Δd) = exp(-(2 * Δd / L_c)²)
      For broadband white light (Δλ ≈ 300 nm, L_c ≈ 1.0 μm), fringes vanish beyond |Δd| > 2 μm.
  - Sodium Doublet Beats (Phách vân vạch kép Natri):
      λ₁ = 589.00 nm, λ₂ = 589.59 nm (Δλ = 0.59 nm, λ̄ = 589.295 nm).
      Visibility envelope beats:
      V(Δd) = |cos(2π * Δλ * Δd / λ̄²)|
      First destructive beat node where fringes disappear: Δd ≈ 147.1 μm; first constructive recovery: Δd ≈ 294.3 μm.
  - LIGO Gravitational Wave Detector Simulation:
      Quadrupole metric perturbation of spacetime h(t) = h₀ * cos(ω_gw * t).
      Differential arm deformation:
      ΔL(t) = L * h(t)
      With Fabry-Perot cavity round-trip bounces N_bounce ≈ 280, L_eff ≈ 1120 km:
      ΔΦ_gw(t) = (4π / λ) * L_eff * h(t)
      Modulating photodiode current at binary merger chirp frequencies (35 Hz -> 250 Hz, GW150914).
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


# Physical Constants
C_LIGHT = 299792458.0                # Speed of light in vacuum (m/s)


class InterferometerMode(str, Enum):
    CIRCULAR = "circular"            # Haidinger concentric circular fringes
    WEDGE = "wedge"                  # Fizeau localized wedge straight fringes
    LIGO = "ligo"                    # LIGO gravitational wave detector 4 km arms


@dataclass(frozen=True)
class LightSource:
    code: str
    name_vi: str
    wavelength_nm: float
    secondary_wavelength_nm: Optional[float]
    spectral_width_nm: float
    color_hex: str

    @property
    def coherence_length_um(self) -> float:
        """Coherence length L_c = λ² / Δλ in micrometers."""
        if self.spectral_width_nm <= 1e-6:
            return 1e6  # Essentially infinite for idealized laser
        return ((self.wavelength_nm * 1e-3) ** 2) / (self.spectral_width_nm * 1e-3)


SOURCES: dict[str, LightSource] = {
    "HeNe": LightSource("HeNe", "Laser He-Ne Đỏ", 632.8, None, 0.002, "#ef4444"),
    "Sodium": LightSource("Sodium", "Đèn Natri (Vạch kép D)", 589.0, 589.59, 0.02, "#f59e0b"),
    "WhiteLight": LightSource("WhiteLight", "Ánh sáng Trắng (Quang phổ liên tục)", 550.0, None, 300.0, "#f8fafc"),
    "LaserNdYAG": LightSource("LaserNdYAG", "Laser Nd:YAG Hồng ngoại LIGO", 1064.0, None, 0.0001, "#38bdf8"),
    "GreenLaser": LightSource("GreenLaser", "Laser Bán dẫn Lục", 532.0, None, 0.005, "#22c55e"),
}


@dataclass(frozen=True)
class MichelsonPreset:
    code: str
    title_vi: str
    badge_label: str
    mode: InterferometerMode
    source_key: str
    delta_d_um: float
    tilt_angle_deg: float
    ligo_strain_h0: float
    ligo_gw_freq_hz: float
    description: str


PRESETS: dict[str, MichelsonPreset] = {
    "HE_NE_CIRCULAR": MichelsonPreset(
        code="HE_NE_CIRCULAR",
        title_vi="Giao thoa Vân tròn Đồng tâm Haidinger (Laser He-Ne 632.8 nm)",
        badge_label="VÂN TRÒN HE-NE",
        mode=InterferometerMode.CIRCULAR,
        source_key="HeNe",
        delta_d_um=10.0,
        tilt_angle_deg=0.0,
        ligo_strain_h0=0.0,
        ligo_gw_freq_hz=0.0,
        description="Gương M₁ vuông góc với M₂. Chùm tia song song phản xạ tạo bản mỏng không khí, cho hệ vân tròn đồng tâm Haidinger. Khi dịch chuyển M₂, các vân tròn sinh ra hoặc chui vào tâm.",
    ),
    "SODIUM_DOUBLET_BEATS": MichelsonPreset(
        code="SODIUM_DOUBLET_BEATS",
        title_vi="Hiện tượng Phách Vân Giao thoa (Vạch kép Natri D1/D2)",
        badge_label="PHÁCH VẠCH KÉP NATRI",
        mode=InterferometerMode.CIRCULAR,
        source_key="Sodium",
        delta_d_um=147.1,
        tilt_angle_deg=0.0,
        ligo_strain_h0=0.0,
        ligo_gw_freq_hz=0.0,
        description="Đèn Natri phát hai vạch phổ sát nhau (589.00 nm và 589.59 nm). Khi dịch chuyển gương M₂, độ tương phản vân biến thiên tuần hoàn theo dạng phách; tại d ≈ 147 μm vân triệt tiêu hoàn toàn (V ≈ 0).",
    ),
    "WHITE_LIGHT_ZERO_PATH": MichelsonPreset(
        code="WHITE_LIGHT_ZERO_PATH",
        title_vi="Giao thoa Ánh sáng Trắng & Chiều dài Kết hợp Lc",
        badge_label="ÁNH SÁNG TRẮNG Δd=0",
        mode=InterferometerMode.CIRCULAR,
        source_key="WhiteLight",
        delta_d_um=0.0,
        tilt_angle_deg=0.0,
        ligo_strain_h0=0.0,
        ligo_gw_freq_hz=0.0,
        description="Ánh sáng trắng có dải phổ rộng (Δλ ≈ 300 nm) dẫn đến chiều dài kết hợp cực ngắn (Lc ≈ 1 μm). Hệ vân chỉ quan sát được trong phạm vi vài micromet quanh vị trí hiệu quang trình bằng 0.",
    ),
    "WEDGE_LINEAR_FRINGES": MichelsonPreset(
        code="WEDGE_LINEAR_FRINGES",
        title_vi="Giao thoa Vân Thẳng Nêm Không khí Fizeau",
        badge_label="VÂN NÊM THẲNG",
        mode=InterferometerMode.WEDGE,
        source_key="HeNe",
        delta_d_um=5.0,
        tilt_angle_deg=0.04,
        ligo_strain_h0=0.0,
        ligo_gw_freq_hz=0.0,
        description="Gương M₂ nghiêng một góc nhỏ α so với gương ảo M₁', tạo thành nêm không khí mỏng. Vân giao thoa là các đoạn thẳng song song cách đều nhau với khoảng vân i = λ / (2α).",
    ),
    "LIGO_GRAVITATIONAL_CHIRP": MichelsonPreset(
        code="LIGO_GRAVITATIONAL_CHIRP",
        title_vi="Đài quan sát Sóng Hấp dẫn LIGO & Biến dạng Không-Thời gian",
        badge_label="SÓNG HẤP DẪN LIGO",
        mode=InterferometerMode.LIGO,
        source_key="LaserNdYAG",
        delta_d_um=0.0,
        tilt_angle_deg=0.0,
        ligo_strain_h0=1.0e-21,
        ligo_gw_freq_hz=100.0,
        description="Mô phỏng đài quan sát LIGO với cánh tay dài 4 km. Sóng hấp dẫn dạng tứ cực làm một cánh tay dãn ra trong khi cánh tay kia co lại, gây biến điệu công suất tại photodiode theo tín hiệu Chirp.",
    ),
}


class InteractiveMichelsonInterferometerItem(QGraphicsObject):
    """
    Interactive Michelson Interferometer & LIGO Gravitational Wave Detector Simulator.
    Provides touchless air gesture interaction, authentic physical equations,
    circular/wedge fringe synthesis, and live LIGO quadrupole chirp wave monitoring.
    """

    preset_changed = Signal(str)
    path_difference_changed = Signal(float)
    wavelength_changed = Signal(float)
    mode_changed = Signal(str)
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
        self._active_preset_code: str = "HE_NE_CIRCULAR"
        self._preset: MichelsonPreset = PRESETS[self._active_preset_code]
        self._mode: InterferometerMode = self._preset.mode
        self._source: LightSource = SOURCES[self._preset.source_key]

        self._delta_d_um: float = self._preset.delta_d_um
        self._tilt_angle_deg: float = self._preset.tilt_angle_deg
        self._wavelength_nm: float = self._source.wavelength_nm

        # Animation state
        self._is_running: bool = True
        self._sim_time: float = 0.0

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physics Calculations
    # -------------------------------------------------------------------------
    @property
    def mode(self) -> InterferometerMode:
        return self._mode

    @property
    def active_preset(self) -> MichelsonPreset:
        return self._preset

    @property
    def delta_d_um(self) -> float:
        return self._delta_d_um

    @property
    def wavelength_nm(self) -> float:
        return self._wavelength_nm

    @property
    def tilt_angle_deg(self) -> float:
        return self._tilt_angle_deg

    def compute_optical_path_difference_um(self) -> float:
        """Returns total optical path difference Δ = 2 * Δd in micrometers."""
        return 2.0 * self._delta_d_um

    def compute_fringe_shift_count(self) -> float:
        """Total number of fringes shifted from Δd = 0: ΔN = 2 * Δd / λ."""
        lam_um = self._wavelength_nm * 1e-3
        if lam_um <= 1e-9:
            return 0.0
        return (2.0 * self._delta_d_um) / lam_um

    def compute_visibility(self) -> float:
        """
        Calculates fringe visibility contrast V = (I_max - I_min) / (I_max + I_min).
        Incorporates coherence decay envelope and sodium doublet beat oscillation.
        """
        d_um = abs(self._delta_d_um)
        lc_um = self._source.coherence_length_um

        # Gaussian coherence envelope decay
        coherence_decay = math.exp(-((2.0 * d_um) / lc_um) ** 2) if lc_um > 0.0 else 1.0

        if self._source.secondary_wavelength_nm is not None:
            # Sodium doublet beats
            lam1 = self._source.wavelength_nm * 1e-3      # μm
            lam2 = self._source.secondary_wavelength_nm * 1e-3  # μm
            delta_lam = abs(lam2 - lam1)
            mean_lam = (lam1 + lam2) / 2.0
            beat_phase = (2.0 * math.pi * delta_lam / (mean_lam ** 2)) * d_um
            beat_envelope = abs(math.cos(beat_phase))
            return max(0.0, min(1.0, beat_envelope * coherence_decay))

        return max(0.0, min(1.0, coherence_decay))

    def compute_wedge_fringe_spacing_mm(self) -> float:
        """Calculates linear wedge fringe spacing i = λ / (2 * α) in mm."""
        alpha_rad = math.radians(self._tilt_angle_deg)
        if alpha_rad <= 1e-9:
            return 999.0
        lam_mm = self._wavelength_nm * 1e-6
        return lam_mm / (2.0 * alpha_rad)

    def compute_circular_intensity(self, r_norm: float) -> float:
        """
        Calculates normalized intensity at normalized radius r_norm in [0, 1].
        θ ≈ r_norm * θ_max where θ_max ≈ 0.03 rad.
        I(r) = 0.5 * [1 + V * cos( (2π / λ) * 2 * Δd * cos(θ) + π )].
        """
        theta_max = 0.035  # ~2 degrees incident divergence
        theta = r_norm * theta_max
        cos_theta = math.cos(theta)

        lam_um = self._wavelength_nm * 1e-3
        phase = (2.0 * math.pi / lam_um) * (2.0 * self._delta_d_um) * cos_theta + math.pi
        v = self.compute_visibility()
        return 0.5 * (1.0 + v * math.cos(phase))

    def compute_wedge_intensity(self, x_norm: float) -> float:
        """
        Calculates normalized intensity across wedge aperture x_norm in [-1, 1].
        Phase = (2π / λ) * [2 * Δd + 2 * α * x] + π.
        """
        lam_um = self._wavelength_nm * 1e-3
        alpha_rad = math.radians(self._tilt_angle_deg)
        # Scale x over 2 mm window
        x_um = x_norm * 1000.0
        phase = (2.0 * math.pi / lam_um) * (2.0 * self._delta_d_um + 2.0 * alpha_rad * x_um) + math.pi
        v = self.compute_visibility()
        return 0.5 * (1.0 + v * math.cos(phase))

    def compute_ligo_strain(self, t_sec: float) -> tuple[float, float, float]:
        """
        Returns (strain_h, diff_arm_delta_L_m, photodiode_power_pct).
        Models binary black hole merger chirp (GW150914) with frequency sweep.
        """
        # Periodic 3.5s chirp cycle
        t_cycle = t_sec % 3.5
        t_norm = t_cycle / 3.5

        if t_norm < 0.85:
            # Inspiral phase: frequency increases from 35 Hz to 250 Hz
            f_inst = 35.0 + 215.0 * (t_norm / 0.85) ** 2
            amp = (0.2 + 0.8 * (t_norm / 0.85)) * self._preset.ligo_strain_h0
        else:
            # Merger & Ringdown phase: rapid dampening
            decay_norm = (t_norm - 0.85) / 0.15
            f_inst = 250.0 - 50.0 * decay_norm
            amp = self._preset.ligo_strain_h0 * math.exp(-decay_norm * 4.0)

        phase = 2.0 * math.pi * f_inst * t_sec
        h_t = amp * math.cos(phase)

        # LIGO physical arm length L = 4000 m
        arm_L = 4000.0
        delta_L = arm_L * h_t  # meters

        # Fabry-Perot cavity amplification N_bounce = 280
        l_eff = arm_L * 280.0
        lam_m = self._wavelength_nm * 1e-9
        delta_phi = (4.0 * math.pi / lam_m) * (l_eff * h_t)

        # Photodiode normalized power output (interference modulation)
        # Scaled visually for education: 50% baseline + modulation
        p_det = 50.0 + 45.0 * math.sin(phase)

        return h_t, delta_L, p_det

    # -------------------------------------------------------------------------
    # Preset & Parameter Setters
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads designated interferometer curriculum preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._mode = self._preset.mode
        self._source = SOURCES[self._preset.source_key]
        self._delta_d_um = self._preset.delta_d_um
        self._tilt_angle_deg = self._preset.tilt_angle_deg
        self._wavelength_nm = self._source.wavelength_nm

        self.preset_changed.emit(preset_code)
        self.mode_changed.emit(self._mode.value)
        self.path_difference_changed.emit(self._delta_d_um)
        self.wavelength_changed.emit(self._wavelength_nm)
        self.update()

    def set_mode(self, mode: InterferometerMode) -> None:
        """Switches interferometer observation mode."""
        self._mode = mode
        self.mode_changed.emit(self._mode.value)
        self.update()

    def step_mirror_position(self, delta_um: float) -> None:
        """Adjusts mirror M₂ position by delta_um (positive or negative)."""
        self._delta_d_um = round(self._delta_d_um + delta_um, 4)
        self.path_difference_changed.emit(self._delta_d_um)
        self.update()

    def step_wavelength(self, delta_nm: float) -> None:
        """Adjusts light source wavelength by delta_nm."""
        self._wavelength_nm = max(380.0, min(1200.0, round(self._wavelength_nm + delta_nm, 1)))
        self.wavelength_changed.emit(self._wavelength_nm)
        self.update()

    def step_tilt_angle(self, delta_deg: float) -> None:
        """Adjusts mirror tilt angle α for wedge mode."""
        self._tilt_angle_deg = max(0.005, min(0.5, round(self._tilt_angle_deg + delta_deg, 3)))
        self.update()

    def reset_zero_path(self) -> None:
        """Resets movable mirror to exact zero optical path difference (Δd = 0)."""
        self._delta_d_um = 0.0
        self.path_difference_changed.emit(0.0)
        self.update()

    def toggle_animation(self) -> None:
        """Toggles simulation clock playback."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if not self._is_running:
            return
        self._sim_time += 0.033
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

        # 3. Left Panel: Optical Bench Schematic / LIGO Arms
        self._paint_optical_bench(painter)

        # 4. Right Panel: Interference Viewport & Analytical Oscilloscope
        self._paint_interference_viewport(painter)

        # 5. Bottom Digital Readout Bar
        self._paint_readout_bar(painter)

        # 6. Interactive In-Air Controls Bar
        self._paint_controls_bar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QPen(QColor("#f8fafc")))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 28, "🔬 Giao thoa kế Michelson & Sóng Hấp dẫn LIGO")

        # Badge
        badge_text = self._preset.badge_label
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        badge_rect = QRectF(480.0, 12.0, 175.0, 22.0)
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

    def _paint_optical_bench(self, painter: QPainter) -> None:
        """Renders the physical Michelson interferometer bench and laser ray splitting."""
        bench_rect = QRectF(20.0, 56.0, 440.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bench_rect, 10.0, 10.0)

        # Bench Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        bench_title = "SƠ ĐỒ BĂNG QUANG HỌC & ĐƯỜNG TRUYỀN TIA SÁNG" if self._mode != InterferometerMode.LIGO else "MÔ HÌNH ĐÀI QUAN SÁT SÓNG HẤP DẪN LIGO (4 KM)"
        painter.drawText(34, 78, bench_title)

        center_x = 220.0
        center_y = 230.0

        beam_color = QColor(self._source.color_hex)
        laser_pen = QPen(beam_color, 2.2)

        if self._mode != InterferometerMode.LIGO:
            # 1. Light Source (Left)
            src_rect = QRectF(40.0, center_y - 18.0, 50.0, 36.0)
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(beam_color, 1.5))
            painter.drawRoundedRect(src_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(src_rect, Qt.AlignmentFlag.AlignCenter, "NGUỒN S")

            # 2. Beam Splitter (Center, 45° angle)
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(45.0)
            bs_rect = QRectF(-5.0, -32.0, 10.0, 64.0)
            painter.setBrush(QBrush(QColor(148, 163, 184, 80)))
            painter.setPen(QPen(QColor("#38bdf8"), 1.5))
            painter.drawRoundedRect(bs_rect, 2.0, 2.0)
            painter.restore()

            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(center_x - 45, center_y + 35, "Bán mạ (BS)")

            # 3. Fixed Mirror M1 (Top)
            m1_rect = QRectF(center_x - 30.0, 100.0, 60.0, 10.0)
            painter.setBrush(QBrush(QColor(51, 65, 85)))
            painter.setPen(QPen(QColor("#f8fafc"), 1.8))
            painter.drawRoundedRect(m1_rect, 2.0, 2.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(center_x - 22, 92, "Gương Cố định M₁")

            # 4. Movable Mirror M2 (Right)
            m2_offset_x = (self._delta_d_um / 20.0)  # visual displacement scaling
            m2_x = 380.0 + m2_offset_x
            painter.save()
            painter.translate(m2_x, center_y)
            if self._mode == InterferometerMode.WEDGE:
                painter.rotate(self._tilt_angle_deg * 25.0)  # visually exaggerated tilt
            m2_rect = QRectF(0.0, -30.0, 10.0, 60.0)
            painter.setBrush(QBrush(QColor(51, 65, 85)))
            painter.setPen(QPen(QColor("#f59e0b"), 1.8))
            painter.drawRoundedRect(m2_rect, 2.0, 2.0)
            painter.restore()

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(int(m2_x - 25), int(center_y - 36), "Gương Di động M₂")

            # 5. Observation Screen / Photodetector (Bottom)
            det_rect = QRectF(center_x - 32.0, 375.0, 64.0, 12.0)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.setPen(QPen(QColor("#22c55e"), 1.5))
            painter.drawRoundedRect(det_rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#22c55e")))
            painter.drawText(center_x - 24, 402, "Màn Quan sát E")

            # 6. Ray Paths
            painter.setPen(laser_pen)
            # Source to BS
            painter.drawLine(QPointF(90.0, center_y), QPointF(center_x, center_y))
            # BS to M1 and back
            painter.drawLine(QPointF(center_x, center_y), QPointF(center_x, 110.0))
            # BS to M2 and back
            painter.drawLine(QPointF(center_x, center_y), QPointF(m2_x, center_y))
            # BS to Detector
            recombined_pen = QPen(beam_color, 2.8)
            painter.setPen(recombined_pen)
            painter.drawLine(QPointF(center_x, center_y), QPointF(center_x, 375.0))

            # Ray Arrowheads
            self._draw_arrow(painter, 150.0, center_y, 0.0, beam_color)
            self._draw_arrow(painter, center_x, 160.0, -90.0, beam_color)
            self._draw_arrow(painter, center_x, 180.0, 90.0, beam_color)
            self._draw_arrow(painter, 300.0, center_y, 0.0, beam_color)
            self._draw_arrow(painter, 290.0, center_y, 180.0, beam_color)
            self._draw_arrow(painter, center_x, 310.0, 90.0, beam_color)

            # Labels for Arms
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(center_x + 10, 175, "Nhánh 1: d₁")
            painter.drawText(275, center_y + 18, f"Nhánh 2: d₂ (Δd = {self._delta_d_um:+.1f} μm)")

        else:
            # LIGO Gravitational Wave Mode Rendering
            h_t, delta_L, p_det = self.compute_ligo_strain(self._sim_time)
            arm_stretch = math.sin(self._sim_time * 8.0) * 8.0

            # LIGO Beam Splitter & Corner Station
            corner_rect = QRectF(center_x - 35.0, center_y - 20.0, 70.0, 70.0)
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(QColor("#38bdf8"), 1.5))
            painter.drawRoundedRect(corner_rect, 6.0, 6.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(corner_rect, Qt.AlignmentFlag.AlignCenter, "Trạm Trung tâm\nCorner Station")

            # Arm 1: North Arm (Vertical, 4 km)
            n_arm_len = 120.0 + arm_stretch
            painter.setPen(QPen(QColor("#0284c7"), 4.0))
            painter.drawLine(QPointF(center_x, center_y - 20.0), QPointF(center_x, center_y - 20.0 - n_arm_len))

            # End Test Mass 1
            etm1_rect = QRectF(center_x - 25.0, center_y - 20.0 - n_arm_len - 12.0, 50.0, 12.0)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.setPen(QPen(QColor("#38bdf8"), 1.5))
            painter.drawRoundedRect(etm1_rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(center_x - 42, int(center_y - 20.0 - n_arm_len - 18.0), f"ETM_Y: L₁ + ΔL/2")

            # Arm 2: East Arm (Horizontal, 4 km)
            e_arm_len = 130.0 - arm_stretch
            painter.setPen(QPen(QColor("#0284c7"), 4.0))
            painter.drawLine(QPointF(center_x + 35.0, center_y + 15.0), QPointF(center_x + 35.0 + e_arm_len, center_y + 15.0))

            # End Test Mass 2
            etm2_rect = QRectF(center_x + 35.0 + e_arm_len, center_y + 3.0, 12.0, 24.0)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.setPen(QPen(QColor("#f59e0b"), 1.5))
            painter.drawRoundedRect(etm2_rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(center_x + 35.0 + e_arm_len - 15.0), int(center_y - 2.0), f"ETM_X: L₂ - ΔL/2")

            # Gravitational Ripple Waves Visualization
            painter.setPen(QPen(QColor(244, 63, 94, 130), 1.2, Qt.PenStyle.DashLine))
            for rad in [40.0, 75.0, 110.0, 145.0]:
                dynamic_rad = rad + (self._sim_time * 30.0) % 35.0
                painter.drawEllipse(QPointF(center_x, center_y + 15.0), dynamic_rad, dynamic_rad * 0.75)

            # Quadrupole Strain Annotation
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor("#f43f5e")))
            painter.drawText(34, 400, f"Sóng Tứ cực Graviton: h(t) = {h_t:+.2e} | ΔL = {delta_L:+.2e} m (10⁻¹⁸ m)")

    def _paint_interference_viewport(self, painter: QPainter) -> None:
        """Renders simulated interference field (circular/wedge) or LIGO detector oscilloscope."""
        view_rect = QRectF(480.0, 56.0, 440.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(view_rect, 10.0, 10.0)

        # Panel Header
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        title = "TRƯỜNG GIAO THOA QUANG HỌC" if self._mode != InterferometerMode.LIGO else "TÍN HIỆU DAO ĐỘNG KÝ PHOTODIODE LIGO"
        painter.drawText(494, 78, title)

        if self._mode == InterferometerMode.CIRCULAR:
            self._paint_circular_fringes(painter)
        elif self._mode == InterferometerMode.WEDGE:
            self._paint_wedge_fringes(painter)
        else:
            self._paint_ligo_oscilloscope(painter)

    def _paint_circular_fringes(self, painter: QPainter) -> None:
        """Draws concentric Haidinger rings based on analytical intensity distribution."""
        cx = 700.0
        cy = 190.0
        max_r = 95.0

        # Viewport Outer Ring
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawEllipse(QPointF(cx, cy), max_r, max_r)

        # Draw concentric rings analytically
        beam_color = QColor(self._source.color_hex)
        v = self.compute_visibility()

        # Step along radius to draw antialiased rings
        steps = 65
        for i in range(steps, 0, -1):
            r_norm = i / float(steps)
            r_px = r_norm * max_r
            intensity = self.compute_circular_intensity(r_norm)

            if self._source.code == "WhiteLight" and abs(self._delta_d_um) < 2.0:
                # Chromatic dispersion colors near zero OPD
                alpha = int(intensity * 255.0)
                phase_deg = int((r_norm * 360.0 + self._delta_d_um * 90.0) % 360)
                ring_color = QColor.fromHsv(phase_deg, 200, 255, alpha)
            else:
                alpha = int(intensity * 255.0)
                ring_color = QColor(beam_color.red(), beam_color.green(), beam_color.blue(), alpha)

            painter.setBrush(QBrush(ring_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

        # Center Crosshair
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cx - max_r, cy), QPointF(cx + max_r, cy))
        painter.drawLine(QPointF(cx, cy - max_r), QPointF(cx, cy + max_r))

        # Bottom Analytical Curve: Intensity Profile I(r)
        self._paint_intensity_curve(painter, is_wedge=False)

    def _paint_wedge_fringes(self, painter: QPainter) -> None:
        """Draws parallel Fizeau straight fringes across rectangular aperture."""
        box_rect = QRectF(590.0, 95.0, 220.0, 190.0)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawRoundedRect(box_rect, 6.0, 6.0)

        beam_color = QColor(self._source.color_hex)
        steps = 80
        w_px = box_rect.width()
        h_px = box_rect.height()
        col_w = w_px / float(steps)

        for i in range(steps):
            x_norm = (i / float(steps)) * 2.0 - 1.0
            intensity = self.compute_wedge_intensity(x_norm)
            alpha = int(intensity * 255.0)
            col_color = QColor(beam_color.red(), beam_color.green(), beam_color.blue(), alpha)

            x_pos = box_rect.left() + i * col_w
            painter.setBrush(QBrush(col_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(x_pos, box_rect.top(), col_w + 0.5, h_px))

        # Re-stroke border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawRoundedRect(box_rect, 6.0, 6.0)

        # Bottom Analytical Curve: Intensity Profile I(x)
        self._paint_intensity_curve(painter, is_wedge=True)

    def _paint_intensity_curve(self, painter: QPainter, is_wedge: bool) -> None:
        """Plots the spatial intensity profile I(r) or I(x)."""
        plot_rect = QRectF(510.0, 310.0, 380.0, 120.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 200)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(plot_rect, 6.0, 6.0)

        # Axes & Grid
        painter.setPen(QPen(QColor(51, 65, 85), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(plot_rect.left() + 30.0, plot_rect.top() + 60.0), QPointF(plot_rect.right() - 10.0, plot_rect.top() + 60.0))
        painter.drawLine(QPointF(plot_rect.left() + 30.0, plot_rect.bottom() - 20.0), QPointF(plot_rect.right() - 10.0, plot_rect.bottom() - 20.0))

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(plot_rect.left() + 6.0), int(plot_rect.top() + 25.0), "I_max")
        painter.drawText(int(plot_rect.left() + 10.0), int(plot_rect.bottom() - 18.0), "0.0")

        curve_path = QPainterPath()
        pts = 90
        w_curve = plot_rect.width() - 50.0
        h_curve = 75.0
        base_y = plot_rect.bottom() - 25.0

        for i in range(pts):
            fraction = i / float(pts - 1)
            if is_wedge:
                val = self.compute_wedge_intensity(fraction * 2.0 - 1.0)
            else:
                val = self.compute_circular_intensity(fraction)
            px = plot_rect.left() + 35.0 + fraction * w_curve
            py = base_y - val * h_curve
            if i == 0:
                curve_path.moveTo(px, py)
            else:
                curve_path.lineTo(px, py)

        painter.setPen(QPen(QColor(self._source.color_hex), 1.8))
        painter.drawPath(curve_path)

        axis_lbl = "Bán kính góc r (m)" if not is_wedge else "Tọa độ nêm x (mm)"
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(plot_rect.right() - 110.0), int(plot_rect.bottom() - 6.0), axis_lbl)

    def _paint_ligo_oscilloscope(self, painter: QPainter) -> None:
        """Renders real-time LIGO binary black hole chirp strain and photodiode power oscillogram."""
        osc_rect = QRectF(500.0, 95.0, 400.0, 335.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 240)))
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.drawRoundedRect(osc_rect, 8.0, 8.0)

        # Grid lines
        painter.setPen(QPen(QColor(15, 23, 42), 1.0))
        for x_step in range(50, 400, 50):
            painter.drawLine(QPointF(osc_rect.left() + x_step, osc_rect.top()), QPointF(osc_rect.left() + x_step, osc_rect.bottom()))
        for y_step in range(40, 330, 40):
            painter.drawLine(QPointF(osc_rect.left(), osc_rect.top() + y_step), QPointF(osc_rect.right(), osc_rect.top() + y_step))

        # Draw Gravitational Wave Chirp Waveform h(t)
        mid_y1 = osc_rect.top() + 90.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(int(osc_rect.left() + 15.0), int(osc_rect.top() + 20.0), "BIẾN DẠNG KHÔNG-THỜI GIAN h(t) [CHIRP SIGNAL GW150914]")

        h_path = QPainterPath()
        pts = 140
        w_plot = osc_rect.width() - 30.0

        for i in range(pts):
            t_sample = (self._sim_time - 3.0) + (i / float(pts)) * 3.0
            h_val, _, _ = self.compute_ligo_strain(t_sample)
            px = osc_rect.left() + 15.0 + (i / float(pts)) * w_plot
            py = mid_y1 - (h_val / 1.0e-21) * 45.0
            if i == 0:
                h_path.moveTo(px, py)
            else:
                h_path.lineTo(px, py)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(h_path)

        # Draw Photodiode Interference Power Output P_det(t)
        mid_y2 = osc_rect.top() + 240.0
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(int(osc_rect.left() + 15.0), int(osc_rect.top() + 170.0), "CÔNG SUẤT QUANG TẠI PHOTODIODE P_det(t) = P₀·sin²(ΔΦ_gw/2)")

        p_path = QPainterPath()
        for i in range(pts):
            t_sample = (self._sim_time - 3.0) + (i / float(pts)) * 3.0
            _, _, p_val = self.compute_ligo_strain(t_sample)
            px = osc_rect.left() + 15.0 + (i / float(pts)) * w_plot
            py = mid_y2 - ((p_val - 50.0) / 50.0) * 45.0
            if i == 0:
                p_path.moveTo(px, py)
            else:
                p_path.lineTo(px, py)

        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawPath(p_path)

    def _paint_readout_bar(self, painter: QPainter) -> None:
        """Draws digital parameter and formula values."""
        bar_rect = QRectF(20.0, 456.0, 900.0, 68.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bar_rect, 8.0, 8.0)

        delta_d = self._delta_d_um
        delta_n = self.compute_fringe_shift_count()
        v_pct = self.compute_visibility() * 100.0
        lc = self._source.coherence_length_um
        lam = self._wavelength_nm

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(36, 478, f"HIỆU QUANG TRÌNH: Δd = 2·(L₂ - L₁) = {2.0 * delta_d:+.2f} μm (d₂ - d₁ = {delta_d:+.2f} μm)")
        painter.drawText(36, 498, f"SỐ VÂN DỊCH CHUYỂN: ΔN = 2Δd / λ = {delta_n:+.2f} vân")
        painter.drawText(36, 516, f"ĐỘ TƯƠNG PHẢN (VISIBILITY): V = {v_pct:.1f}%")

        painter.drawText(480, 478, f"BƯỚC SÓNG TỚI: λ = {lam:.1f} nm ({self._source.name_vi})")
        if self._source.secondary_wavelength_nm:
            painter.drawText(480, 498, f"VẠCH KÉP NATRI: λ₁ = {self._source.wavelength_nm:.2f} nm | λ₂ = {self._source.secondary_wavelength_nm:.2f} nm (Δλ = 0.59 nm)")
        else:
            painter.drawText(480, 498, f"CHIỀU DÀI KẾT HỢP: L_c = λ² / Δλ ≈ {lc:.2f} μm")

        if self._mode == InterferometerMode.WEDGE:
            spacing_mm = self.compute_wedge_fringe_spacing_mm()
            painter.drawText(480, 516, f"KHOẢNG VÂN NÊM: i = λ / (2α) = {spacing_mm:.3f} mm (Góc nghiêng α = {self._tilt_angle_deg:.3f}°)")
        elif self._mode == InterferometerMode.LIGO:
            h_t, delta_L, _ = self.compute_ligo_strain(self._sim_time)
            painter.drawText(480, 516, f"LIGO SÓNG HẤP DẪN: h = {h_t:+.2e} | ΔL = {delta_L:+.2e} m | L_eff = 1,120 km")
        else:
            painter.drawText(480, 516, f"CHẾ ĐỘ QUAN SÁT: Vân tròn đồng tâm Haidinger (M₁ ⊥ M₂)")

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air buttons."""
        bar_y = 536.0
        self._buttons.clear()

        btn_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(btn_font)

        # 1. Preset Selector Button
        preset_rect = QRectF(20.0, bar_y, 140.0, 32.0)
        self._draw_btn(painter, preset_rect, "btn_preset", f"Presets: {self._preset.badge_label}")

        # 2. Mode Toggle Button
        mode_label = "Vân Tròn" if self._mode == InterferometerMode.CIRCULAR else ("Vân Nêm" if self._mode == InterferometerMode.WEDGE else "LIGO")
        mode_rect = QRectF(168.0, bar_y, 100.0, 32.0)
        self._draw_btn(painter, mode_rect, "btn_mode", f"Chế độ: {mode_label}")

        # 3. Mirror M2 Steppers
        self._draw_btn(painter, QRectF(276.0, bar_y, 55.0, 32.0), "btn_d_minus_10", "-10μm")
        self._draw_btn(painter, QRectF(336.0, bar_y, 50.0, 32.0), "btn_d_minus_1", "-1μm")
        self._draw_btn(painter, QRectF(391.0, bar_y, 50.0, 32.0), "btn_d_plus_1", "+1μm")
        self._draw_btn(painter, QRectF(446.0, bar_y, 55.0, 32.0), "btn_d_plus_10", "+10μm")

        # 4. Wavelength Steppers
        self._draw_btn(painter, QRectF(508.0, bar_y, 58.0, 32.0), "btn_lam_minus", "-20nm")
        self._draw_btn(painter, QRectF(571.0, bar_y, 58.0, 32.0), "btn_lam_plus", "+20nm")

        # 5. Play / Pause
        play_label = "⏸ Tạm dừng" if self._is_running else "▶ Tiếp tục"
        self._draw_btn(painter, QRectF(636.0, bar_y, 82.0, 32.0), "btn_play", play_label)

        # 6. Zero OPD Reset
        self._draw_btn(painter, QRectF(724.0, bar_y, 70.0, 32.0), "btn_reset", "↺ Δd = 0")

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

    def _draw_arrow(self, painter: QPainter, x: float, y: float, angle_deg: float, color: QColor) -> None:
        """Helper to draw subtle directional arrow along ray paths."""
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle_deg)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        poly = [QPointF(4.0, 0.0), QPointF(-4.0, -3.5), QPointF(-4.0, 3.5)]
        painter.drawPolygon(poly)
        painter.restore()

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
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_mode":
            modes = [InterferometerMode.CIRCULAR, InterferometerMode.WEDGE, InterferometerMode.LIGO]
            curr_mode_idx = modes.index(self._mode)
            next_mode = modes[(curr_mode_idx + 1) % len(modes)]
            self.set_mode(next_mode)
        elif key == "btn_d_minus_10":
            self.step_mirror_position(-10.0)
        elif key == "btn_d_minus_1":
            self.step_mirror_position(-1.0)
        elif key == "btn_d_plus_1":
            self.step_mirror_position(1.0)
        elif key == "btn_d_plus_10":
            self.step_mirror_position(10.0)
        elif key == "btn_lam_minus":
            self.step_wavelength(-20.0)
        elif key == "btn_lam_plus":
            self.step_wavelength(20.0)
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_reset":
            self.reset_zero_path()
        elif key == "btn_stamp":
            self.stamp_diagram()
        elif key == "btn_close":
            self.setVisible(False)

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless pinch gesture targeting interactive UI buttons."""
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
            "mode": self._mode.value,
            "source_name": self._source.name_vi,
            "wavelength_nm": self._wavelength_nm,
            "delta_d_um": self._delta_d_um,
            "visibility_pct": self.compute_visibility() * 100.0,
            "fringe_count": self.compute_fringe_shift_count(),
        }
        self.stamp_requested.emit(stamp_data)
