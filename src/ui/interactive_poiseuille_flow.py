"""
AirTeach AI — Interactive Fluid Dynamics & Aerodynamics: Poiseuille Flow,
Hagen-Poiseuille Viscous Tube Flow & Reynolds Number Turbulence Simulator.

Simulates:
  - Exact Navier-Stokes analytical solution for steady laminar viscous pipe flow:
      v(r) = (ΔP / 4ηL) * (R² - r²) = v_max * (1 - r²/R²)
  - Hagen-Poiseuille 4th-power volumetric flow rate law:
      Q = (π * R⁴ * ΔP) / (8 * η * L)
  - Hydraulic resistance fluid analog to Ohm's law:
      R_h = ΔP / Q = (8 * η * L) / (π * R⁴)
  - Wall shear stress:
      τ_w = (ΔP * R) / (2 * L)
  - Reynolds number & flow regime transitions:
      Re = (ρ * v_avg * 2R) / η
      * Re < 2000: Laminar flow (smooth parabolic streamlines, silent)
      * 2000 <= Re <= 4000: Transitional flow (eddy shedding, intermittent bursts)
      * Re > 4000: Fully turbulent flow (flattened 1/7th power-law velocity profile, high dissipation)
"""

from __future__ import annotations
import math
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
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class PoiseuillePreset:
    """Represents a curriculum scenario for Poiseuille Flow & Reynolds Number."""
    code: str
    title_vi: str
    badge_label: str
    fluid_name: str
    density_kg_per_m3: float     # Fluid density in kg/m³
    viscosity_mPa_s: float       # Dynamic viscosity in mPa·s (1 mPa·s = 10^-3 Pa·s)
    default_radius_mm: float     # Pipe radius in mm
    default_delta_p_Pa: float    # Pressure drop in Pa
    pipe_length_m: float         # Pipe length in meters
    description: str


POISEUILLE_PRESETS: dict[str, PoiseuillePreset] = {
    "BLOOD_ARTERIOLE_VASODILATION": PoiseuillePreset(
        code="BLOOD_ARTERIOLE_VASODILATION",
        title_vi="Huyết Động Học Tim Mạch — Giãn Tiểu Động Mạch & Định Luật R⁴",
        badge_label="HEMODYNAMICS R⁴",
        fluid_name="Máu Người Toàn Phần (Blood)",
        density_kg_per_m3=1060.0,
        viscosity_mPa_s=3.5,
        default_radius_mm=0.05,    # 50 um
        default_delta_p_Pa=2500.0, # 2.5 kPa
        pipe_length_m=0.002,       # 2 mm
        description="Định luật Hagen-Poiseuille trong vi tuần hoàn: lưu lượng máu phụ thuộc cực nhạy vào lũy thừa bậc 4 của bán kính mạch máu (Q ∝ R⁴). Giảm bán kính chỉ 16% làm lưu lượng máu giảm một nửa!",
    ),
    "WATER_CAPILLARY_LAMINAR": PoiseuillePreset(
        code="WATER_CAPILLARY_LAMINAR",
        title_vi="Dòng Chảy Tầng Trong Ống Mao Dẫn — Thí Nghiệm Poiseuille Chuẩn",
        badge_label="WATER CAPILLARY",
        fluid_name="Nước Cất ở 20°C (Water)",
        density_kg_per_m3=1000.0,
        viscosity_mPa_s=1.002,
        default_radius_mm=0.5,     # 0.5 mm
        default_delta_p_Pa=1200.0, # 1.2 kPa
        pipe_length_m=0.20,        # 20 cm
        description="Thí nghiệm kinh điển xác định độ nhớt chất lỏng của Poiseuille (1840). Dòng chảy tầng hoàn hảo (Re ≈ 450 < 2000) với profile vận tốc parabol đối xứng chuẩn xác qua trục ống.",
    ),
    "GLYCERIN_HIGH_VISCOSITY_CREEPING": PoiseuillePreset(
        code="GLYCERIN_HIGH_VISCOSITY_CREEPING",
        title_vi="Chất Lỏng Siêu Nhớt Glyxerin — Chế Độ Chảy Chậm (Creeping Stokes Flow)",
        badge_label="GLYCERIN CREEPING",
        fluid_name="Glycerin Tinh Khiết (Glycerin)",
        density_kg_per_m3=1260.0,
        viscosity_mPa_s=1412.0,    # ~1.4 Pa·s
        default_radius_mm=4.0,     # 4 mm
        default_delta_p_Pa=30000.0,# 30 kPa
        pipe_length_m=0.10,        # 10 cm
        description="Dòng chảy nhớt chi phối hoàn toàn (Re << 1): lực ma sát trong giữa các lớp chất lỏng vượt trội so với lực quán tính, năng lượng áp suất bị tiêu tán mãnh liệt thành nhiệt lượng.",
    ),
    "TURBULENT_WATER_PIPE_BURST": PoiseuillePreset(
        code="TURBULENT_WATER_PIPE_BURST",
        title_vi="Đường Ống Thủy Lực Tốc Độ Cao — Chuyển Dịch Dòng Rối (Turbulent Flow)",
        badge_label="TURBULENT PIPE",
        fluid_name="Nước Thủy Lực Áp Cao",
        density_kg_per_m3=1000.0,
        viscosity_mPa_s=1.002,
        default_radius_mm=6.0,     # 6 mm
        default_delta_p_Pa=4500.0, # 4.5 kPa
        pipe_length_m=0.40,        # 40 cm
        description="Khi số Reynolds Re > 4000, các xoáy cuộn hỗn loạn hình thành làm phá vỡ phân tầng parabol: profile vận tốc bị dàn phẳng thành quy luật lũy thừa 1/7 và lực cản thủy lực tăng vọt.",
    ),
    "AIR_RESPIRATORY_BRONCHIOLE": PoiseuillePreset(
        code="AIR_RESPIRATORY_BRONCHIOLE",
        title_vi="Khí Động Học Phổi — Luồng Khí Qua Tiểu Phế Quản Hô Hấp",
        badge_label="BRONCHIOLE AIR",
        fluid_name="Không Khí Ẩm 37°C (Air)",
        density_kg_per_m3=1.14,
        viscosity_mPa_s=0.0185,
        default_radius_mm=0.35,    # 0.35 mm
        default_delta_p_Pa=40.0,   # 40 Pa
        pipe_length_m=0.005,       # 5 mm
        description="Cơ học thông khí phổi ở nhánh phế quản tận: đường kính hẹp nhưng chênh áp nhỏ duy trì dòng chảy tầng êm ái, bảo vệ biểu mô hô hấp chống vi chấn thương.",
    ),
}


@dataclass
class FlowTracerParticle:
    """Tracer particle animated inside the viscous tube."""
    rel_x: float          # Normalized axial position [0.0, 1.0]
    rel_r: float          # Normalized radial distance [-1.0, 1.0] from centerline
    size: float
    color: QColor
    turbulence_jitter: float = 0.0


class InteractivePoiseuilleFlowItem(QGraphicsObject):
    """
    Interactive PySide6 Graphics Item simulating Poiseuille viscous pipe flow,
    Hagen-Poiseuille 4th-power law, parabolic velocity vectors, and Reynolds number turbulence.
    """

    flow_computed = Signal(float, float, float, float)  # Q_m3_s, v_max, v_avg, Re
    stamp_requested = Signal(dict)

    CARD_W = 740.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "WATER_CAPILLARY_LAMINAR"
        self._preset: PoiseuillePreset = POISEUILLE_PRESETS[self._active_preset_code]

        self._radius_mm: float = self._preset.default_radius_mm
        self._delta_p_Pa: float = self._preset.default_delta_p_Pa
        self._viscosity_mPa_s: float = self._preset.viscosity_mPa_s
        self._pipe_length_m: float = self._preset.pipe_length_m
        self._density_kg_per_m3: float = self._preset.density_kg_per_m3

        # Tracer particles
        self._particles: list[FlowTracerParticle] = []
        self._init_tracer_particles(num_particles=45)

        # Animation state
        self._sim_time: float = 0.0
        self._is_running: bool = True

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for 60fps real-time physics animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Physical & Fluid Dynamics Properties (Navier-Stokes & Hagen-Poiseuille)
    # -------------------------------------------------------------------------
    @property
    def radius_m(self) -> float:
        """Pipe radius R in meters."""
        return max(1e-6, self._radius_mm * 1e-3)

    @property
    def dynamic_viscosity_Pa_s(self) -> float:
        """Dynamic viscosity η in Pa·s."""
        return max(1e-6, self._viscosity_mPa_s * 1e-3)

    @property
    def max_velocity_m_per_s(self) -> float:
        """
        Centerline maximum velocity for laminar pipe flow:
        v_max = (ΔP * R²) / (4 * η * L)
        """
        r = self.radius_m
        eta = self.dynamic_viscosity_Pa_s
        length = max(1e-4, self._pipe_length_m)
        dp = max(0.0, self._delta_p_Pa)
        return (dp * (r ** 2)) / (4.0 * eta * length)

    @property
    def volumetric_flow_rate_m3_per_s(self) -> float:
        """
        Hagen-Poiseuille volumetric flow rate:
        Q = (π * R⁴ * ΔP) / (8 * η * L)
        """
        r = self.radius_m
        eta = self.dynamic_viscosity_Pa_s
        length = max(1e-4, self._pipe_length_m)
        dp = max(0.0, self._delta_p_Pa)
        re = self.reynolds_number

        q_laminar = (math.pi * (r ** 4) * dp) / (8.0 * eta * length)
        if re <= 2000.0:
            return q_laminar
        elif re >= 4000.0:
            # Turbulent flow dissipation reduces effective flow rate for given pressure head
            # Q_turbulent ≈ (49/60) * π * R² * v_max_turbulent
            # We scale laminar flow with friction factor adjustment
            friction_ratio = (2000.0 / re) ** 0.25
            return q_laminar * max(0.35, friction_ratio)
        else:
            # Transitional regime interpolation
            alpha = (re - 2000.0) / 2000.0
            friction_ratio = (2000.0 / re) ** 0.25
            return q_laminar * ((1.0 - alpha) + alpha * max(0.35, friction_ratio))

    @property
    def volumetric_flow_rate_mL_per_s(self) -> float:
        """Volumetric flow rate Q in mL/s."""
        return self.volumetric_flow_rate_m3_per_s * 1e6

    @property
    def average_velocity_m_per_s(self) -> float:
        """Cross-sectional average velocity v_avg = Q / (π * R²)."""
        r = self.radius_m
        area = math.pi * (r ** 2)
        if area <= 1e-12:
            return 0.0
        return self.volumetric_flow_rate_m3_per_s / area

    @property
    def reynolds_number(self) -> float:
        """
        Reynolds number for pipe flow:
        Re = (ρ * v_avg * D) / η = (2 * ρ * v_avg * R) / η
        """
        r = self.radius_m
        rho = self._density_kg_per_m3
        eta = self.dynamic_viscosity_Pa_s
        # Estimate base laminar v_avg = v_max / 2 for Re calculation
        v_laminar_avg = self.max_velocity_m_per_s / 2.0
        re_nominal = (rho * v_laminar_avg * (2.0 * r)) / eta
        return max(0.01, re_nominal)

    @property
    def hydraulic_resistance_Pa_s_per_m3(self) -> float:
        """
        Hydraulic resistance R_h = ΔP / Q = (8 * η * L) / (π * R⁴)
        Fluid dynamic analogue to Ohm's resistance.
        """
        q = self.volumetric_flow_rate_m3_per_s
        if q <= 1e-15:
            return 1e12
        return self._delta_p_Pa / q

    @property
    def wall_shear_stress_Pa(self) -> float:
        """
        Wall shear stress τ_w = (ΔP * R) / (2 * L)
        """
        r = self.radius_m
        length = max(1e-4, self._pipe_length_m)
        return (self._delta_p_Pa * r) / (2.0 * length)

    @property
    def flow_regime_label(self) -> tuple[str, str, QColor]:
        """Returns regime name, Vietnamese description, and badge color."""
        re = self.reynolds_number
        if re < 2000.0:
            return "LAMINAR", "DÒNG CHẢY TẦNG (Êm Ả, Parabol Chuẩn)", QColor("#10b981")
        elif re <= 4000.0:
            return "TRANSITIONAL", "CHUYỂN DỊCH QUÁ ĐỘ (Xoáy Bất Ổn Định)", QColor("#f59e0b")
        else:
            return "TURBULENT", "DÒNG CHẢY RỐI (Hỗn Loạn, Profile Dàn Phẳng)", QColor("#ef4444")

    def local_velocity(self, rel_r: float) -> float:
        """
        Calculates local flow velocity v(r) at normalized radius rel_r in [-1.0, 1.0].
        rel_r = 0.0 is tube centerline, ±1.0 is tube wall (no-slip v=0).
        """
        r_norm = min(1.0, abs(rel_r))
        re = self.reynolds_number
        v_max = self.max_velocity_m_per_s

        if re <= 2000.0:
            # Classic parabolic Poiseuille profile: v(r) = v_max * (1 - r²)
            return max(0.0, v_max * (1.0 - (r_norm ** 2)))
        elif re >= 4000.0:
            # Fully turbulent 1/7th power-law profile: v(r) = v_max * (1 - r)^(1/7)
            # Wall shear layer drops sharply to 0 at r_norm = 1.0
            return max(0.0, v_max * ((1.0 - r_norm) ** (1.0 / 7.0)))
        else:
            # Transitional blend
            alpha = (re - 2000.0) / 2000.0
            v_lam = v_max * (1.0 - (r_norm ** 2))
            v_turb = v_max * ((1.0 - r_norm) ** (1.0 / 7.0))
            return max(0.0, (1.0 - alpha) * v_lam + alpha * v_turb)

    # -------------------------------------------------------------------------
    # Tracer Particle Lifecycle
    # -------------------------------------------------------------------------
    def _init_tracer_particles(self, num_particles: int = 45) -> None:
        """Initializes distributed flow tracer particles inside the tube."""
        self._particles.clear()
        palette = [
            QColor("#38bdf8"),
            QColor("#0ea5e9"),
            QColor("#06b6d4"),
            QColor("#67e8f9"),
            QColor("#22d3ee"),
        ]
        for _ in range(num_particles):
            rel_x = random.uniform(0.0, 1.0)
            rel_r = random.uniform(-0.92, 0.92)
            sz = random.uniform(2.5, 4.0)
            col = random.choice(palette)
            self._particles.append(FlowTracerParticle(rel_x, rel_r, sz, col))

    def _on_sim_tick(self) -> None:
        """Updates tracer particles and animates velocity vectors."""
        if not self._is_running:
            return

        dt = 0.016
        self._sim_time += dt
        v_max = max(0.001, self.max_velocity_m_per_s)
        re = self.reynolds_number

        for p in self._particles:
            v_local = self.local_velocity(p.rel_r)
            # Normalized axial speed proportional to local velocity
            speed_factor = (v_local / v_max) if v_max > 0 else 0.5
            p.rel_x += speed_factor * 0.018 + 0.002

            # If turbulent, introduce transverse brownian/vortex oscillations
            if re > 2000.0:
                jitter_strength = min(0.08, (re - 2000.0) / 10000.0)
                p.rel_r += random.uniform(-jitter_strength, jitter_strength) * 0.1
                p.rel_r = max(-0.94, min(0.94, p.rel_r))

            # Wrap around when exiting pipe outlet
            if p.rel_x > 1.0:
                p.rel_x = 0.0
                p.rel_r = random.uniform(-0.92, 0.92)

        self.update()

    # -------------------------------------------------------------------------
    # Controls & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in POISEUILLE_PRESETS:
            return
        self._active_preset_code = code
        self._preset = POISEUILLE_PRESETS[code]
        self._radius_mm = self._preset.default_radius_mm
        self._delta_p_Pa = self._preset.default_delta_p_Pa
        self._viscosity_mPa_s = self._preset.viscosity_mPa_s
        self._pipe_length_m = self._preset.pipe_length_m
        self._density_kg_per_m3 = self._preset.density_kg_per_m3

        self.flow_computed.emit(
            self.volumetric_flow_rate_m3_per_s,
            self.max_velocity_m_per_s,
            self.average_velocity_m_per_s,
            self.reynolds_number,
        )
        self.update()

    def set_radius_mm(self, r_mm: float) -> None:
        """Sets tube radius R in mm [0.01, 15.0]."""
        self._radius_mm = max(0.01, min(15.0, r_mm))
        self.flow_computed.emit(
            self.volumetric_flow_rate_m3_per_s,
            self.max_velocity_m_per_s,
            self.average_velocity_m_per_s,
            self.reynolds_number,
        )
        self.update()

    def set_pressure_drop_Pa(self, dp_Pa: float) -> None:
        """Sets pressure difference ΔP in Pa [1.0, 100000.0]."""
        self._delta_p_Pa = max(1.0, min(100000.0, dp_Pa))
        self.flow_computed.emit(
            self.volumetric_flow_rate_m3_per_s,
            self.max_velocity_m_per_s,
            self.average_velocity_m_per_s,
            self.reynolds_number,
        )
        self.update()

    def set_viscosity_mPa_s(self, eta_mPa_s: float) -> None:
        """Sets dynamic viscosity η in mPa·s [0.01, 5000.0]."""
        self._viscosity_mPa_s = max(0.01, min(5000.0, eta_mPa_s))
        self.flow_computed.emit(
            self.volumetric_flow_rate_m3_per_s,
            self.max_velocity_m_per_s,
            self.average_velocity_m_per_s,
            self.reynolds_number,
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets parameters to active preset defaults."""
        self._radius_mm = self._preset.default_radius_mm
        self._delta_p_Pa = self._preset.default_delta_p_Pa
        self._viscosity_mPa_s = self._preset.viscosity_mPa_s
        self.flow_computed.emit(
            self.volumetric_flow_rate_m3_per_s,
            self.max_velocity_m_per_s,
            self.average_velocity_m_per_s,
            self.reynolds_number,
        )
        self.update()

    # -------------------------------------------------------------------------
    # Layout Buttons for Touchless & Mouse Interaction
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Defines touch button bounding boxes at the bottom."""
        by = self.CARD_H - 46.0
        bw = 88.0
        bh = 32.0
        spacing = 10.0
        start_x = 18.0

        keys = [
            ("btn_preset", 130.0),
            ("btn_r_down", 68.0),
            ("btn_r_up", 68.0),
            ("btn_p_down", 72.0),
            ("btn_p_up", 72.0),
            ("btn_eta_down", 76.0),
            ("btn_eta_up", 76.0),
            ("btn_reset", 66.0),
            ("btn_stamp", 112.0),
        ]

        curr_x = start_x
        for key, width in keys:
            self._buttons[key] = QRectF(curr_x, by, width, bh)
            curr_x += width + spacing

    # -------------------------------------------------------------------------
    # Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Container Card
        card_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor("#090e17"))
        grad.setColorAt(1.0, QColor("#111a28"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Top Header & Preset Badge
        self._paint_header(painter)

        # 3. Main Viscous Pipe Simulator & Flow Field
        self._paint_pipe_flow_field(painter)

        # 4. Parabolic vs Turbulent Cross-Section Velocity Profile Plot
        self._paint_velocity_profile_plot(painter)

        # 5. Quantitative Physical Metrics & Hagen-Poiseuille Formula
        self._paint_metrics_panel(painter)

        # 6. Bottom Touchless Buttons
        self._paint_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Badge
        badge_rect = QRectF(20.0, 16.0, 175.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"🌊 {self._preset.badge_label}")

        # Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(208.0, 33.0, self._preset.title_vi)

        # Subtitle / Fluid Description
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            22.0,
            58.0,
            f"Chất lỏng: {self._preset.fluid_name} (ρ = {self._density_kg_per_m3:.1f} kg/m³, η = {self._viscosity_mPa_s:.3f} mPa·s) | Chiều dài ống L = {self._pipe_length_m * 1000.0:.1f} mm",
        )

    def _paint_pipe_flow_field(self, painter: QPainter) -> None:
        """Renders the cylindrical tube with pressure gradient, parabolic vector arrows, and tracers."""
        px = 22.0
        py = 74.0
        pw = 430.0
        ph = 250.0

        # Background frame for pipe
        painter.setBrush(QBrush(QColor("#060a12")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title of pipe view
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "1. Ống Thủy Lực & Trường Vector Vận Tốc v(r)")

        # Tube boundaries
        tube_x0 = px + 40.0
        tube_w = pw - 60.0
        tube_cy = py + ph / 2.0 + 8.0
        tube_h = 160.0
        tube_top = tube_cy - tube_h / 2.0
        tube_bottom = tube_cy + tube_h / 2.0

        # Internal fluid gradient: high pressure at inlet (amber/red) -> low pressure at outlet (cyan/blue)
        f_grad = QLinearGradient(tube_x0, 0.0, tube_x0 + tube_w, 0.0)
        f_grad.setColorAt(0.0, QColor(245, 158, 11, 40))   # High P_inlet
        f_grad.setColorAt(1.0, QColor(14, 165, 233, 40))   # Low P_outlet
        painter.setBrush(QBrush(f_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(tube_x0, tube_top, tube_w, tube_h))

        # Upper and lower solid walls (hatched lines indicating no-slip boundary)
        wall_pen = QPen(QColor("#64748b"), 3.0)
        painter.setPen(wall_pen)
        painter.drawLine(QPointF(tube_x0, tube_top), QPointF(tube_x0 + tube_w, tube_top))
        painter.drawLine(QPointF(tube_x0, tube_bottom), QPointF(tube_x0 + tube_w, tube_bottom))

        # Wall hatching
        hatch_pen = QPen(QColor("#475569"), 1.0)
        painter.setPen(hatch_pen)
        num_hatches = int(tube_w / 14.0)
        for i in range(num_hatches + 1):
            hx = tube_x0 + i * 14.0
            # top wall hatching (pointing up-right)
            painter.drawLine(QPointF(hx, tube_top), QPointF(hx + 8.0, tube_top - 8.0))
            # bottom wall hatching (pointing down-right)
            painter.drawLine(QPointF(hx, tube_bottom), QPointF(hx + 8.0, tube_bottom + 8.0))

        # No-slip labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(tube_x0 + 8.0, tube_top - 12.0, "Thành ống: Điều kiện không trượt (No-slip) v(R) = 0")
        painter.drawText(tube_x0 + 8.0, tube_bottom + 18.0, "Thành ống: Điều kiện không trượt (No-slip) v(-R) = 0")

        # Centerline (dashed)
        dash_pen = QPen(QColor("#38bdf8"), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.drawLine(QPointF(tube_x0, tube_cy), QPointF(tube_x0 + tube_w, tube_cy))

        # Centerline tag
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(tube_x0 + tube_w - 65.0, tube_cy - 4.0, "Trục r = 0 (v_max)")

        # Flow tracer particles
        for p in self._particles:
            px_coord = tube_x0 + p.rel_x * tube_w
            py_coord = tube_cy + p.rel_r * (tube_h / 2.0)
            painter.setBrush(QBrush(p.color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px_coord, py_coord), p.size / 2.0, p.size / 2.0)

        # Parabolic velocity vector arrows across two cross-sections
        arrow_cols = [tube_x0 + tube_w * 0.35, tube_x0 + tube_w * 0.70]
        v_max = max(1e-6, self.max_velocity_m_per_s)
        max_arrow_len = 54.0

        for col_x in arrow_cols:
            # Draw vertical baseline
            painter.setPen(QPen(QColor("#475569"), 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(col_x, tube_top), QPointF(col_x, tube_bottom))

            num_vectors = 11
            for vi in range(num_vectors):
                rel_r = -1.0 + vi * (2.0 / (num_vectors - 1))
                vy = tube_cy + rel_r * (tube_h / 2.0)
                v_local = self.local_velocity(rel_r)
                arrow_len = (v_local / v_max) * max_arrow_len

                # Color: cyan at center, transitioning to blue/slate at edge
                col_factor = abs(rel_r)
                vec_color = QColor(
                    int(56 * (1.0 - col_factor) + 14 * col_factor),
                    int(189 * (1.0 - col_factor) + 165 * col_factor),
                    int(248 * (1.0 - col_factor) + 233 * col_factor),
                    220,
                )
                painter.setPen(QPen(vec_color, 1.6))
                end_x = col_x + arrow_len
                painter.drawLine(QPointF(col_x, vy), QPointF(end_x, vy))

                # Arrowhead
                if arrow_len > 4.0:
                    painter.setBrush(QBrush(vec_color))
                    head = QPainterPath()
                    head.moveTo(end_x, vy)
                    head.lineTo(end_x - 4.5, vy - 2.5)
                    head.lineTo(end_x - 4.5, vy + 2.5)
                    head.closeSubpath()
                    painter.drawPath(head)

        # Inflow / Outflow pressure labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(px + 4.0, tube_cy - 12.0, "Vào (P₁)")
        painter.drawText(px + 4.0, tube_cy + 4.0, f"+{self._delta_p_Pa:.0f}Pa")

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(tube_x0 + tube_w + 4.0, tube_cy - 12.0, "Ra (P₂)")
        painter.drawText(tube_x0 + tube_w + 4.0, tube_cy + 4.0, "0 Pa")

    def _paint_velocity_profile_plot(self, painter: QPainter) -> None:
        """Renders the cross-sectional 2D velocity profile v(r) plot."""
        px = 464.0
        py = 74.0
        pw = 254.0
        ph = 250.0

        # Card box
        painter.setBrush(QBrush(QColor("#060a12")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "2. Profile Vận Tốc v(r)")

        # Coordinate axes
        ox = px + 40.0
        cy = py + ph / 2.0 + 8.0
        plot_w = pw - 65.0
        plot_h = 160.0
        top_y = cy - plot_h / 2.0
        bot_y = cy + plot_h / 2.0

        # Axes lines
        ax_pen = QPen(QColor("#64748b"), 1.2)
        painter.setPen(ax_pen)
        # Vertical r axis (-R to +R)
        painter.drawLine(QPointF(ox, top_y - 6.0), QPointF(ox, bot_y + 6.0))
        # Horizontal v axis (0 to v_max)
        painter.drawLine(QPointF(ox, cy), QPointF(ox + plot_w + 6.0, cy))

        # Axis labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(ox - 30.0, top_y + 4.0, "+R")
        painter.drawText(ox - 30.0, bot_y + 2.0, "-R")
        painter.drawText(ox - 24.0, cy - 3.0, "r=0")
        painter.drawText(ox + plot_w - 28.0, cy + 14.0, "v (m/s)")

        # Average velocity vertical dashed line v_avg
        v_max = max(1e-6, self.max_velocity_m_per_s)
        v_avg = self.average_velocity_m_per_s
        v_avg_px = ox + (v_avg / v_max) * (plot_w * 0.88)

        painter.setPen(QPen(QColor("#f59e0b"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(v_avg_px, top_y), QPointF(v_avg_px, bot_y))

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(v_avg_px - 14.0, top_y - 6.0, f"v_tb={v_avg:.2f}")

        # Continuous smooth velocity profile curve v(r)
        curve_path = QPainterPath()
        num_pts = 40
        for i in range(num_pts + 1):
            rel_r = -1.0 + i * (2.0 / num_pts)
            vy_pixel = cy + rel_r * (plot_h / 2.0)
            v_val = self.local_velocity(rel_r)
            vx_pixel = ox + (v_val / v_max) * (plot_w * 0.88)
            if i == 0:
                curve_path.moveTo(vx_pixel, vy_pixel)
            else:
                curve_path.lineTo(vx_pixel, vy_pixel)

        # Fill curve area with subtle cyan glow
        fill_path = QPainterPath(curve_path)
        fill_path.lineTo(ox, bot_y)
        fill_path.lineTo(ox, top_y)
        fill_path.closeSubpath()
        painter.setBrush(QBrush(QColor(56, 189, 248, 30)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Draw curve stroke
        painter.setPen(QPen(QColor("#38bdf8"), 2.2))
        painter.drawPath(curve_path)

        # Maximum velocity dot on centerline
        v_max_px = ox + (plot_w * 0.88)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.drawEllipse(QPointF(v_max_px, cy), 3.5, 3.5)

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(v_max_px - 40.0, cy - 8.0, f"v_max={v_max:.2f}")

        # Regime indicator badge inside plot
        regime_code, regime_desc, regime_col = self.flow_regime_label
        regime_box = QRectF(px + 12.0, py + ph - 38.0, pw - 24.0, 26.0)
        painter.setBrush(QBrush(QColor(regime_col.red(), regime_col.green(), regime_col.blue(), 35)))
        painter.setPen(QPen(regime_col, 1.0))
        painter.drawRoundedRect(regime_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(regime_col))
        painter.drawText(regime_box, Qt.AlignmentFlag.AlignCenter, f"{regime_code} (Re = {self.reynolds_number:.1f})")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Renders live quantitative parameters, Hagen-Poiseuille formula, and physics readouts."""
        px = 22.0
        py = 336.0
        pw = self.CARD_W - 44.0
        ph = 142.0

        # Background container
        painter.setBrush(QBrush(QColor("#09101c")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Header of metrics
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 14.0, py + 20.0, "3. Thông Số Vật Lý & Định Luật Hagen-Poiseuille: Q = (π·R⁴·ΔP) / (8·η·L)")

        # Column 1: Current Operating Controls
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#e2e8f0")))
        col1_x = px + 14.0
        line1_y = py + 44.0
        dy = 22.0

        painter.drawText(col1_x, line1_y, f"• Bán kính ống R: {self._radius_mm:.3f} mm (Đường kính D = {2.0 * self._radius_mm:.3f} mm)")
        painter.drawText(col1_x, line1_y + dy, f"• Chênh áp ΔP: {self._delta_p_Pa:.1f} Pa ({self._delta_p_Pa / 1000.0:.2f} kPa)")
        painter.drawText(col1_x, line1_y + 2 * dy, f"• Độ nhớt động học η: {self._viscosity_mPa_s:.3f} mPa·s")
        painter.drawText(col1_x, line1_y + 3 * dy, f"• Trở kháng thủy lực R_h: {self.hydraulic_resistance_Pa_s_per_m3:.2e} Pa·s/m³")

        # Column 2: Computed Hydrodynamic Outputs
        col2_x = px + 360.0
        painter.drawText(col2_x, line1_y, f"• Lưu lượng khối Q: {self.volumetric_flow_rate_mL_per_s:.4f} mL/s ({self.volumetric_flow_rate_m3_per_s:.3e} m³/s)")
        painter.drawText(col2_x, line1_y + dy, f"• Vận tốc cực đại v_max: {self.max_velocity_m_per_s:.4f} m/s")
        painter.drawText(col2_x, line1_y + 2 * dy, f"• Vận tốc trung bình v_tb: {self.average_velocity_m_per_s:.4f} m/s")
        painter.drawText(col2_x, line1_y + 3 * dy, f"• Ứng suất trượt thành τ_w: {self.wall_shear_stress_Pa:.3f} Pa  |  Số Reynolds: Re = {self.reynolds_number:.1f}")

        # Explanatory Educational Text at bottom
        desc_rect = QRectF(px + 12.0, py + ph + 8.0, pw - 24.0, 95.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive touch/mouse control buttons."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        labels = {
            "btn_preset": f"📋 Kịch Bản ({self._preset.badge_label})",
            "btn_r_down": "R - 10%",
            "btn_r_up": "R + 10%",
            "btn_p_down": "ΔP - 20%",
            "btn_p_up": "ΔP + 20%",
            "btn_eta_down": "η - 20%",
            "btn_eta_up": "η + 20%",
            "btn_reset": "🔄 Đặt Lại",
            "btn_stamp": "📌 Đóng Dấu Bảng",
        }

        for key, rect in self._buttons.items():
            is_hover = (key == self._hovered_btn)
            text = labels.get(key, key)

            if key == "btn_stamp":
                bg = QColor("#0284c7") if not is_hover else QColor("#0369a1")
                border = QColor("#38bdf8")
            elif key == "btn_preset":
                bg = QColor("#4338ca") if not is_hover else QColor("#3730a3")
                border = QColor("#818cf8")
            elif "up" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#38bdf8")
            elif "down" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#f43f5e")
            else:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#64748b")

            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handling
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if self._hovered_btn != old_btn:
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
                    self._on_button_clicked(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_button_clicked(self, key: str) -> None:
        """Dispatches button actions."""
        if key == "btn_preset":
            keys = list(POISEUILLE_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_r_down":
            self.set_radius_mm(self._radius_mm * 0.90)
        elif key == "btn_r_up":
            self.set_radius_mm(self._radius_mm * 1.10)
        elif key == "btn_p_down":
            self.set_pressure_drop_Pa(self._delta_p_Pa * 0.80)
        elif key == "btn_p_up":
            self.set_pressure_drop_Pa(self._delta_p_Pa * 1.20)
        elif key == "btn_eta_down":
            self.set_viscosity_mPa_s(self._viscosity_mPa_s * 0.80)
        elif key == "btn_eta_up":
            self.set_viscosity_mPa_s(self._viscosity_mPa_s * 1.20)
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        regime_code, _, _ = self.flow_regime_label
        data = {
            "preset_code": self._active_preset_code,
            "fluid_name": self._preset.fluid_name,
            "radius_mm": self._radius_mm,
            "delta_p_Pa": self._delta_p_Pa,
            "viscosity_mPa_s": self._viscosity_mPa_s,
            "pipe_length_m": self._pipe_length_m,
            "density_kg_per_m3": self._density_kg_per_m3,
            "flow_rate_mL_per_s": self.volumetric_flow_rate_mL_per_s,
            "max_velocity_m_per_s": self.max_velocity_m_per_s,
            "average_velocity_m_per_s": self.average_velocity_m_per_s,
            "reynolds_number": self.reynolds_number,
            "flow_regime": regime_code,
        }
        self.stamp_requested.emit(data)
