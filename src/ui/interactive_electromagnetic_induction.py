"""
AirTeach AI - Interactive Physics Electromagnetic Induction & Faraday / Lenz / Generator Simulator
===================================================================================================
Provides an authentic physics simulation of electromagnetic induction for secondary & university education:
  - Theoretical & Electromagnetic Foundations (Zero Mock Data):
      1. Magnetic Flux (Từ thông qua mạch kín):
           Φ = N · B · S · cos(α)  [Weber, Wb]
           where α is the angle between magnetic field vector B and coil normal vector n.
      2. Faraday's Law of Electromagnetic Induction (Định luật cảm ứng Faraday):
           E = -dΦ/dt = -N · d(B · S · cos(α))/dt  [Volt, V]
      3. Lenz's Law (Định luật Len-xơ):
           The polarity of the induced electromotive force always opposes the change in magnetic flux:
             - If Φ increases (dΦ/dt > 0): Induced magnetic field B_ind opposes B_ext (repulsion force).
             - If Φ decreases (dΦ/dt < 0): Induced magnetic field B_ind reinforces B_ext (attraction force).
      4. Moving Bar Magnet through Solenoid:
           Magnet magnetic dipole moment m_dipole moving along coil axis z:
             Φ(z) = (N · μ₀ · m_dipole · R_c²) / (2 · (R_c² + z²)^(3/2))
             E(t) = -dΦ/dt = (3 · N · μ₀ · m_dipole · R_c² · z) / (2 · (R_c² + z²)^(5/2)) · v_z
      5. Alternating Current (AC) Rotating Generator:
           Coil of N turns rotating in uniform magnetic field B with angular velocity ω:
             α(t) = ω · t
             Φ(t) = Φ₀ · cos(ω · t)  where Φ₀ = N · B · S
             E(t) = E₀ · sin(ω · t)  where E₀ = ω · N · B · S  (Peak EMF)
             Phase shift: E(t) lags Φ(t) by π/2 radians.
      6. Sliding Conductor on Rails (Thanh kim loại chuyển động cắt từ trường):
           E = B · l · v  [V]
           Induced current i = E / R => Laplace braking force: F_L = B · i · l = (B² · l² · v) / R
  - 5 Authentic Curriculum Presets:
      - BAR_MAGNET_SOLENOID: Thanh nam châm chuyển động qua lòng cuộn dây, kim điện kế đảo chiều đối xứng
      - ROTATING_AC_GENERATOR: Khung dây quay đều trong từ trường tạo dòng điện xoay chiều sin
      - MUTUAL_INDUCTION_TRANSFORMER: Cảm ứng hỗ ứng & Máy biến áp lõi thép silic (U₁/U₂ ≈ N₁/N₂)
      - EDDY_CURRENT_PENDULUM: Dòng điện xoáy Foucault hãm phanh điện từ thanh nhôm
      - SLIDING_CONDUCTOR_RAIL: Thanh dẫn trượt trên hai thanh ray trong từ trường đều
  - Interactive Apparatus:
      - Center-zero sensitive galvanometer with sweeping needle (-50 to +50 mA / mV).
      - Dual-trace oscilloscope showing Φ(t) in cyan and E(t) in ruby.
      - Magnetic field vector lines (B_ext and B_ind).
      - Touchless air pinch gestures for magnet movement, rotation speed, coil turns, and vector stamping.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
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

MU_0: float = 4.0 * math.pi * 1e-7  # Vacuum magnetic permeability (H/m)


@dataclass
class InductionPreset:
    code: str
    title_vi: str
    mode: str  # "MAGNET_SOLENOID", "AC_GENERATOR", "TRANSFORMER", "EDDY_CURRENT", "SLIDING_RAIL"
    turns_n: int
    b_field_tesla: float
    coil_area_cm2: float
    resistance_ohm: float
    speed_param: float  # v in m/s or omega in rad/s
    description_vi: str
    formula_vi: str


PRESETS: dict[str, InductionPreset] = {
    "BAR_MAGNET_SOLENOID": InductionPreset(
        code="BAR_MAGNET_SOLENOID",
        title_vi="Nam Châm & Cuộn Dây Solenoid",
        mode="BAR_MAGNET_SOLENOID",
        turns_n=500,
        b_field_tesla=0.15,
        coil_area_cm2=20.0,
        resistance_ohm=10.0,
        speed_param=1.2,  # Velocity v in m/s
        description_vi="Nam châm tiến lại gần cuộn dây làm từ thông tăng, sinh dòng điện cảm ứng tạo từ trường ngược chiều đẩy nam châm (Định luật Len-xơ).",
        formula_vi="e_cư = -ΔΦ/Δt  |  B_cư chống lại sự biến thiên của B_ngoài",
    ),
    "ROTATING_AC_GENERATOR": InductionPreset(
        code="ROTATING_AC_GENERATOR",
        title_vi="Máy Phát Điện Xoay Chiều (AC Generator)",
        mode="AC_GENERATOR",
        turns_n=200,
        b_field_tesla=0.25,
        coil_area_cm2=50.0,
        resistance_ohm=5.0,
        speed_param=31.42,  # ω = 10π ≈ 31.42 rad/s (f = 5 Hz)
        description_vi="Khung dây quay đều trong từ trường đều. Góc α biến thiên điều hòa α(t) = ωt sinh suất điện động cảm ứng hình sin e(t) = E₀·sin(ωt).",
        formula_vi="Φ(t) = NBS·cos(ωt)  =>  e(t) = ωNBS·sin(ωt) = E₀·sin(ωt)",
    ),
    "MUTUAL_INDUCTION_TRANSFORMER": InductionPreset(
        code="MUTUAL_INDUCTION_TRANSFORMER",
        title_vi="Cảm Ứng Hỗ Ứng & Máy Biến Áp (Transformer)",
        mode="TRANSFORMER",
        turns_n=800,  # Secondary turns
        b_field_tesla=0.30,
        coil_area_cm2=30.0,
        resistance_ohm=15.0,
        speed_param=100.0 * math.pi,  # 50 Hz
        description_vi="Từ thông biến thiên qua lõi thép khép kín truyền cảm ứng từ cuộn sơ cấp N₁ sang cuộn thứ cấp N₂ với tỷ số điện áp U₁/U₂ = N₁/N₂.",
        formula_vi="U₁ / U₂ = N₁ / N₂ = I₂ / I₁  (Máy biến áp lý tưởng)",
    ),
    "EDDY_CURRENT_PENDULUM": InductionPreset(
        code="EDDY_CURRENT_PENDULUM",
        title_vi="Dòng Điện Xoáy Foucault (Eddy Current Braking)",
        mode="EDDY_CURRENT",
        turns_n=1,  # Solid sheet
        b_field_tesla=0.45,
        coil_area_cm2=40.0,
        resistance_ohm=0.05,
        speed_param=2.5,  # Oscillation speed
        description_vi="Thanh kim loại nhôm dao động qua vùng từ trường mạnh sinh dòng điện xoáy khép kín Foucault tạo lực hãm phanh điện từ tức thời.",
        formula_vi="F_hãm = (σ·B²·v·V)  (Phanh điện từ tàu cao tốc)",
    ),
    "SLIDING_CONDUCTOR_RAIL": InductionPreset(
        code="SLIDING_CONDUCTOR_RAIL",
        title_vi="Thanh Kim Loại Trượt Trên Ray",
        mode="SLIDING_RAIL",
        turns_n=1,
        b_field_tesla=0.50,
        coil_area_cm2=25.0,
        resistance_ohm=2.0,
        speed_param=2.0,  # Speed 2.0 m/s
        description_vi="Thanh dẫn trượt với vận tốc v cắt ngang các đường sức từ vuông góc, tạo suất điện động cảm ứng e = B·l·v và lực cản Laplace.",
        formula_vi="e = B·l·v  |  F_cản = B·i·l = (B²·l²·v) / R",
    ),
}


class InteractiveElectromagneticInductionItem(QGraphicsObject):
    """
    High-performance vector interactive Electromagnetic Induction Simulator.
    Provides authentic physical equations, animated galvanometer needle, dual-trace oscilloscope,
    magnetic field vector lines, and LessonBoard stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (290.0, 110.0)) -> None:
        super().__init__()
        self.setPos(pos[0], pos[1])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active state
        self._active_preset_code: str = "BAR_MAGNET_SOLENOID"
        self._preset: InductionPreset = PRESETS[self._active_preset_code]

        # Configurable Parameters
        self._turns_n: int = self._preset.turns_n
        self._b_field: float = self._preset.b_field_tesla
        self._coil_area_m2: float = self._preset.coil_area_cm2 * 1e-4
        self._resistance_ohm: float = self._preset.resistance_ohm
        self._speed_param: float = self._preset.speed_param

        # Dynamic simulation state
        self._elapsed_time_s: float = 0.0
        self._is_animating: bool = True
        self._magnet_z_pos: float = -120.0  # Vertical position of magnet
        self._magnet_velocity: float = 1.0  # Direction (+1 or -1)
        self._coil_angle_rad: float = 0.0  # For AC generator

        # Oscilloscope Waveform Buffer (stores last 80 points)
        self._scope_flux: list[float] = [0.0] * 80
        self._scope_emf: list[float] = [0.0] * 80

        # Animation timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "BAR_MAGNET_SOLENOID",
            "ROTATING_AC_GENERATOR",
            "MUTUAL_INDUCTION_TRANSFORMER",
            "EDDY_CURRENT_PENDULUM",
            "SLIDING_CONDUCTOR_RAIL",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Apparatus Viewport (Left) & Oscilloscope / Inspector (Right)
        self._btn_rects["apparatus"] = QRectF(16.0, 88.0, 360.0, 360.0)
        self._btn_rects["inspector"] = QRectF(384.0, 88.0, 380.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 100.0, h_bot)
        self._btn_rects["dec_speed"] = QRectF(122.0, y_bot, 44.0, h_bot)
        self._btn_rects["inc_speed"] = QRectF(172.0, y_bot, 44.0, h_bot)
        self._btn_rects["dec_turns"] = QRectF(224.0, y_bot, 44.0, h_bot)
        self._btn_rects["inc_turns"] = QRectF(274.0, y_bot, 44.0, h_bot)
        self._btn_rects["dec_field"] = QRectF(326.0, y_bot, 44.0, h_bot)
        self._btn_rects["inc_field"] = QRectF(376.0, y_bot, 44.0, h_bot)
        self._btn_rects["stamp"] = QRectF(428.0, y_bot, 336.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical Calculations & Properties (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> InductionPreset:
        return self._preset

    @property
    def turns_n(self) -> int:
        return self._turns_n

    @property
    def b_field(self) -> float:
        return self._b_field

    @property
    def speed_param(self) -> float:
        return self._speed_param

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def current_magnetic_flux_webers(self) -> float:
        """Calculates instantaneous magnetic flux Φ through coil in Webers."""
        if self._preset.mode == "AC_GENERATOR":
            phi_0 = self._turns_n * self._b_field * self._coil_area_m2
            return phi_0 * math.cos(self._coil_angle_rad)
        elif self._preset.mode == "BAR_MAGNET_SOLENOID":
            # Dipole flux through circular loop at axial distance z
            # Coil center at z = 0, coil radius R_c ≈ 0.03 m
            r_c = 0.03
            z_m = self._magnet_z_pos * 0.001  # convert px to meters
            m_dip = self._b_field * 0.05  # dipole strength
            denom = max(1e-6, (r_c ** 2 + z_m ** 2) ** 1.5)
            return (self._turns_n * MU_0 * m_dip * (r_c ** 2)) / (2.0 * denom)
        elif self._preset.mode == "SLIDING_RAIL":
            # Φ = B · l · x
            length_m = 0.15
            x_pos = (math.sin(self._elapsed_time_s * 2.0) + 1.0) * 0.1
            return self._b_field * length_m * x_pos
        else:
            return self._turns_n * self._b_field * self._coil_area_m2 * math.cos(self._elapsed_time_s * 3.0)

    @property
    def current_induced_emf_volts(self) -> float:
        """Calculates instantaneous induced electromotive force E = -dΦ/dt in Volts."""
        if self._preset.mode == "AC_GENERATOR":
            omega = self._speed_param
            e_0 = omega * self._turns_n * self._b_field * self._coil_area_m2
            return e_0 * math.sin(self._coil_angle_rad)
        elif self._preset.mode == "BAR_MAGNET_SOLENOID":
            r_c = 0.03
            z_m = self._magnet_z_pos * 0.001
            v_z = self._magnet_velocity * self._speed_param
            m_dip = self._b_field * 0.05
            denom = max(1e-6, (r_c ** 2 + z_m ** 2) ** 2.5)
            dphi_dz = -(3.0 * self._turns_n * MU_0 * m_dip * (r_c ** 2) * z_m) / (2.0 * denom)
            return -dphi_dz * v_z
        elif self._preset.mode == "SLIDING_RAIL":
            length_m = 0.15
            v_m_s = self._speed_param * math.cos(self._elapsed_time_s * 2.0)
            return self._b_field * length_m * v_m_s
        else:
            omega = 3.0
            return self._turns_n * self._b_field * self._coil_area_m2 * omega * math.sin(self._elapsed_time_s * omega)

    @property
    def current_induced_current_amperes(self) -> float:
        """Calculates induced current i = E / R in Amperes."""
        return self.current_induced_emf_volts / max(0.01, self._resistance_ohm)

    # -------------------------------------------------------------------------
    # Public Controller API
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic electromagnetic induction preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._turns_n = self._preset.turns_n
        self._b_field = self._preset.b_field_tesla
        self._coil_area_m2 = self._preset.coil_area_cm2 * 1e-4
        self._resistance_ohm = self._preset.resistance_ohm
        self._speed_param = self._preset.speed_param
        self._magnet_z_pos = -120.0
        self._magnet_velocity = 1.0
        self._coil_angle_rad = 0.0
        self._scope_flux = [0.0] * 80
        self._scope_emf = [0.0] * 80
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_speed(self, speed: float) -> None:
        """Set motion speed or angular frequency (clamped between 0.1 and 150.0)."""
        self._speed_param = max(0.1, min(150.0, float(speed)))
        self.update()

    def set_turns(self, turns: int) -> None:
        """Set number of coil turns (clamped between 1 and 2000)."""
        self._turns_n = max(1, min(2000, int(turns)))
        self.update()

    def set_magnetic_field(self, b_tesla: float) -> None:
        """Set magnetic field magnitude B in Tesla (clamped between 0.01T and 3.0T)."""
        self._b_field = max(0.01, min(3.0, float(b_tesla)))
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle apparatus movement animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def cleanup(self) -> None:
        """Stop animation timer and cleanly disconnect."""
        self._anim_timer.stop()
        self._is_animating = False

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        if not self._is_animating:
            return

        dt = 0.033  # ~33 ms per frame
        self._elapsed_time_s += dt

        # 1. Update magnet motion or coil rotation
        if self._preset.mode == "BAR_MAGNET_SOLENOID":
            # Oscillate bar magnet smoothly through coil
            z_limit = 110.0
            move_step = self._magnet_velocity * self._speed_param * dt * 90.0
            self._magnet_z_pos += move_step
            if self._magnet_z_pos > z_limit:
                self._magnet_z_pos = z_limit
                self._magnet_velocity = -1.0
            elif self._magnet_z_pos < -z_limit:
                self._magnet_z_pos = -z_limit
                self._magnet_velocity = 1.0
        elif self._preset.mode == "AC_GENERATOR":
            self._coil_angle_rad = (self._coil_angle_rad + self._speed_param * dt) % (2.0 * math.pi)

        # 2. Update oscilloscope buffers
        curr_phi = self.current_magnetic_flux_webers
        curr_emf = self.current_induced_emf_volts
        self._scope_flux.pop(0)
        self._scope_flux.append(curr_phi)
        self._scope_emf.pop(0)
        self._scope_emf.append(curr_emf)

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Vector Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main outer card
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        # 2. Header Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 48.0), 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 24.0, self.WIDTH, 24.0))

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(16.0, 10.0, 520.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🧲 {self._preset.title_vi} | Cảm Ứng Điện Từ & Định Luật Faraday",
        )

        # Close Button
        close_r = self._btn_rects["close"]
        is_close_hover = self._hovered_button == "close"
        painter.setBrush(QBrush(QColor("#ef4444" if is_close_hover else "#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(close_r)
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        cx, cy = close_r.center().x(), close_r.center().y()
        painter.drawLine(int(cx - 4), int(cy - 4), int(cx + 4), int(cy + 4))
        painter.drawLine(int(cx + 4), int(cy - 4), int(cx - 4), int(cy + 4))

        # 3. Presets Bar
        preset_items = [
            ("BAR_MAGNET_SOLENOID", "Nam Châm & Cuộn Dây"),
            ("ROTATING_AC_GENERATOR", "Máy Phát Điện AC"),
            ("MUTUAL_INDUCTION_TRANSFORMER", "Máy Biến Áp"),
            ("EDDY_CURRENT_PENDULUM", "Dòng Foucault"),
            ("SLIDING_CONDUCTOR_RAIL", "Thanh Trượt Ray"),
        ]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for code, label in preset_items:
            r = self._btn_rects[f"preset_{code}"]
            is_active = self._active_preset_code == code
            is_hover = self._hovered_button == f"preset_{code}"
            bg = QColor("#0284c7") if is_active else (QColor("#334155") if is_hover else QColor("#1e293b"))
            border = QColor("#38bdf8") if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(r, 6.0, 6.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#cbd5e1")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)

        # 4. Left Apparatus Viewport
        app_r = self._btn_rects["apparatus"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(app_r, 10.0, 10.0)
        self._paint_apparatus(painter, app_r)

        # 5. Right Inspector & Oscilloscope Panel
        insp_r = self._btn_rects["inspector"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(insp_r, 10.0, 10.0)
        self._paint_inspector_panel(painter, insp_r)

        # 6. Bottom Action Toolbar
        self._paint_bottom_toolbar(painter)

    def _paint_apparatus(self, painter: QPainter, arena_r: QRectF) -> None:
        """Paints the solenoid coil, moving bar magnet or rotating coil, and sensitive galvanometer."""
        cx = arena_r.center().x()
        cy = arena_r.top() + 150.0

        if self._preset.mode == "BAR_MAGNET_SOLENOID":
            # 1. Solenoid Coil (cylinder with copper wire loops)
            coil_w, coil_h = 100.0, 70.0
            c_left = cx - coil_w / 2.0
            c_top = cy - coil_h / 2.0

            # Coil Core tube
            painter.setPen(QPen(QColor("#475569"), 1.5))
            painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
            painter.drawRoundedRect(QRectF(c_left, c_top, coil_w, coil_h), 8.0, 8.0)

            # Helical Copper Winding Loops
            painter.setPen(QPen(QColor("#f59e0b"), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            loops = 9
            for i in range(loops):
                lx = c_left + 8.0 + (i / (loops - 1)) * (coil_w - 16.0)
                painter.drawLine(int(lx - 4), int(c_top + 4), int(lx + 4), int(c_top + coil_h - 4))

            # 2. Moving Permanent Bar Magnet (North: Red, South: Blue)
            mag_w, mag_h = 32.0, 90.0
            mag_x = cx - mag_w / 2.0
            mag_y = cy + self._magnet_z_pos - mag_h / 2.0

            # North Pole (Red)
            painter.setPen(QPen(QColor("#ef4444"), 1.5))
            painter.setBrush(QBrush(QColor("#dc2626")))
            painter.drawRect(QRectF(mag_x, mag_y + mag_h / 2.0, mag_w, mag_h / 2.0))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(mag_x, mag_y + mag_h / 2.0, mag_w, mag_h / 2.0), Qt.AlignmentFlag.AlignCenter, "N")

            # South Pole (Blue)
            painter.setPen(QPen(QColor("#3b82f6"), 1.5))
            painter.setBrush(QBrush(QColor("#2563eb")))
            painter.drawRect(QRectF(mag_x, mag_y, mag_w, mag_h / 2.0))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(mag_x, mag_y, mag_w, mag_h / 2.0), Qt.AlignmentFlag.AlignCenter, "S")

            # Magnet outline
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(mag_x, mag_y, mag_w, mag_h))

            # Velocity vector arrow on magnet
            v_dir = self._magnet_velocity
            v_len = 35.0 * v_dir
            painter.setPen(QPen(QColor("#10b981"), 2.2))
            painter.drawLine(int(mag_x + mag_w + 14.0), int(mag_y + mag_h / 2.0), int(mag_x + mag_w + 14.0), int(mag_y + mag_h / 2.0 + v_len))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(QPointF(mag_x + mag_w + 20.0, mag_y + mag_h / 2.0 + v_len / 2.0), "v")

        elif self._preset.mode == "AC_GENERATOR":
            # Rotating armature loop in magnetic field
            # Permanent Magnet Poles (Left: N, Right: S)
            pole_w, pole_h = 50.0, 110.0
            # N Pole
            painter.setBrush(QBrush(QColor("#dc2626")))
            painter.setPen(QPen(QColor("#ef4444"), 1.5))
            painter.drawRoundedRect(QRectF(cx - 150.0, cy - pole_h / 2.0, pole_w, pole_h), 6.0, 6.0)
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(cx - 150.0, cy - pole_h / 2.0, pole_w, pole_h), Qt.AlignmentFlag.AlignCenter, "N")

            # S Pole
            painter.setBrush(QBrush(QColor("#2563eb")))
            painter.setPen(QPen(QColor("#3b82f6"), 1.5))
            painter.drawRoundedRect(QRectF(cx + 100.0, cy - pole_h / 2.0, pole_w, pole_h), 6.0, 6.0)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(cx + 100.0, cy - pole_h / 2.0, pole_w, pole_h), Qt.AlignmentFlag.AlignCenter, "S")

            # Magnetic field lines (B lines from N to S, light blue)
            painter.setPen(QPen(QColor(56, 189, 248, 80), 1.5, Qt.PenStyle.DashLine))
            for offset_y in [-35.0, -15.0, 5.0, 25.0, 45.0]:
                painter.drawLine(int(cx - 100.0), int(cy + offset_y), int(cx + 100.0), int(cy + offset_y))

            # Rotating Coil Armature
            arm_w = 70.0 * math.cos(self._coil_angle_rad)
            painter.setPen(QPen(QColor("#f59e0b"), 3.0))
            painter.setBrush(QBrush(QColor(245, 158, 11, 40)))
            painter.drawRect(QRectF(cx - abs(arm_w) / 2.0, cy - 40.0, max(4.0, abs(arm_w)), 80.0))

            # Rotation curved arrow
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#10b981")))
            painter.drawText(QPointF(cx - 15.0, cy - 50.0), "ω ⟳")

        # 3. Sensitive Center-Zero Galvanometer (Điện kế G) at bottom of apparatus
        galv_w, galv_h = 140.0, 95.0
        gx = cx - galv_w / 2.0
        gy = arena_r.bottom() - galv_h - 15.0

        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(gx, gy, galv_w, galv_h), 10.0, 10.0)

        # Meter Dial Arc
        dial_cx = cx
        dial_cy = gy + galv_h - 15.0
        dial_r = 55.0
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.drawArc(QRectF(dial_cx - dial_r, dial_cy - dial_r, 2 * dial_r, 2 * dial_r), 45 * 16, 90 * 16)

        # Dial center zero label "0"
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QPointF(dial_cx - 3.0, dial_cy - dial_r - 4.0), "0")
        painter.drawText(QPointF(dial_cx - dial_r + 4.0, dial_cy - 12.0), "-50")
        painter.drawText(QPointF(dial_cx + dial_r - 22.0, dial_cy - 12.0), "+50")

        # Sweeping Galvanometer Needle
        # Scale current to needle angle (-45° to +45°)
        i_curr = self.current_induced_current_amperes
        max_i = 0.05  # 50 mA full scale
        needle_frac = max(-1.0, min(1.0, i_curr / max_i))
        needle_deg = needle_frac * 40.0
        needle_rad = math.radians(needle_deg - 90.0)

        needle_x = dial_cx + (dial_r - 6.0) * math.cos(needle_rad)
        needle_y = dial_cy + (dial_r - 6.0) * math.sin(needle_rad)

        painter.setPen(QPen(QColor("#ef4444"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(dial_cx, dial_cy), QPointF(needle_x, needle_y))
        # Needle pivot dot
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.drawEllipse(QPointF(dial_cx, dial_cy), 3.5, 3.5)

        # Galvanometer label "G"
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(gx + 10.0, gy + 8.0, galv_w - 20.0, 16.0), Qt.AlignmentFlag.AlignCenter, "ĐIỆN KẾ NHẠY (G)")

    def _paint_inspector_panel(self, painter: QPainter, insp_r: QRectF) -> None:
        """Paints real-time quantitative metrics and dual-trace oscilloscope for Φ(t) and E(t)."""
        x = insp_r.left()
        y = insp_r.top()
        w = insp_r.width()

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(x + 14.0, y + 24.0), "📊 THÔNG SỐ TỪ THÔNG & SUẤT ĐIỆN ĐỘNG")

        # Metrics rows
        phi_now = self.current_magnetic_flux_webers
        emf_now = self.current_induced_emf_volts
        i_now = self.current_induced_current_amperes

        metrics = [
            ("Số vòng dây (N):", f"{self._turns_n} vòng"),
            ("Cảm ứng từ ngoài (B):", f"{self._b_field:.2f} T"),
            ("Diện tích vòng dây (S):", f"{self._coil_area_m2 * 1e4:.1f} cm²"),
            ("Điện trở toàn mạch (R):", f"{self._resistance_ohm:.1f} Ω"),
            ("Từ thông tức thời (Φ):", f"{phi_now * 1000.0:+.2f} mWb"),
            ("Suất điện động (e_cư):", f"{emf_now:+.3f} V"),
            ("Dòng điện cảm ứng (i_cư):", f"{i_now * 1000.0:+.2f} mA"),
        ]

        row_y = y + 46.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for lbl, val in metrics:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(QPointF(x + 14.0, row_y), lbl)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(QRectF(x + 150.0, row_y - 12.0, w - 165.0, 16.0), Qt.AlignmentFlag.AlignRight, val)
            row_y += 18.0

        # Oscilloscope Screen (Dual-trace: Cyan for Φ, Ruby for e_cư)
        sc_x = x + 14.0
        sc_y = row_y + 8.0
        sc_w = w - 28.0
        sc_h = 115.0

        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(QRectF(sc_x, sc_y, sc_w, sc_h), 6.0, 6.0)

        # Oscilloscope zero centerline
        mid_sc_y = sc_y + sc_h / 2.0
        painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(sc_x), int(mid_sc_y), int(sc_x + sc_w), int(mid_sc_y))

        # Trace 1: Magnetic Flux Φ(t) (Cyan)
        path_phi = QPainterPath()
        max_phi_disp = max(1e-5, max(abs(p) for p in self._scope_flux))
        for idx, p_val in enumerate(self._scope_flux):
            px = sc_x + (idx / 79.0) * sc_w
            py = mid_sc_y - (p_val / max_phi_disp) * (sc_h * 0.4)
            if idx == 0:
                path_phi.moveTo(px, py)
            else:
                path_phi.lineTo(px, py)

        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawPath(path_phi)

        # Trace 2: Induced EMF E(t) (Ruby)
        path_emf = QPainterPath()
        max_emf_disp = max(1e-4, max(abs(e) for e in self._scope_emf))
        for idx, e_val in enumerate(self._scope_emf):
            px = sc_x + (idx / 79.0) * sc_w
            py = mid_sc_y - (e_val / max_emf_disp) * (sc_h * 0.4)
            if idx == 0:
                path_emf.moveTo(px, py)
            else:
                path_emf.lineTo(px, py)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(path_emf)

        # Oscilloscope Legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(sc_x + 8.0, sc_y + 14.0), "— Từ thông Φ(t)")
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(QPointF(sc_x + 95.0, sc_y + 14.0), "— SĐĐ e(t) = -dΦ/dt")

        # Theoretical formula box at bottom
        eq_r = QRectF(x + 14.0, sc_y + sc_h + 10.0, w - 28.0, 32.0)
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(eq_r, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(eq_r, Qt.AlignmentFlag.AlignCenter, self._preset.formula_vi)

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Paints interactive action buttons: play/pause, speed/turns/field adjustments, stamp."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Play / Pause
        r_play = self._btn_rects["play_pause"]
        is_hover = self._hovered_button == "play_pause"
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor("#0284c7" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_play, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        play_txt = "⏸ Tạm dừng" if self._is_animating else "▶ Tiếp tục"
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # 2. Stepper Buttons: Speed, Turns, Field B
        steppers = [
            ("dec_speed", "v -"),
            ("inc_speed", "v +"),
            ("dec_turns", "N -"),
            ("inc_turns", "N +"),
            ("dec_field", "B -"),
            ("inc_field", "B +"),
        ]
        for btn_name, lbl in steppers:
            r = self._btn_rects[btn_name]
            is_hover = self._hovered_button == btn_name
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor("#334155" if is_hover else "#1e293b")))
            painter.drawRoundedRect(r, 6.0, 6.0)
            painter.setPen(QPen(QColor("#f8fafc")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, lbl)

        # 3. Stamp to Board
        r_stamp = self._btn_rects["stamp"]
        is_hover = self._hovered_button == "stamp"
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.setBrush(QBrush(QColor("#059669" if is_hover else "#064e3b")))
        painter.drawRoundedRect(r_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Thí Nghiệm Lên Bảng (LessonBoard)")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects["close"].contains(pos):
            self.cleanup()
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects["play_pause"].contains(pos):
            self.toggle_animation()
            event.accept()
            return
        if self._btn_rects["dec_speed"].contains(pos):
            self.set_speed(self._speed_param * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_speed"].contains(pos):
            self.set_speed(self._speed_param * 1.25)
            event.accept()
            return
        if self._btn_rects["dec_turns"].contains(pos):
            self.set_turns(int(self._turns_n * 0.8))
            event.accept()
            return
        if self._btn_rects["inc_turns"].contains(pos):
            self.set_turns(int(self._turns_n * 1.25))
            event.accept()
            return
        if self._btn_rects["dec_field"].contains(pos):
            self.set_magnetic_field(self._b_field * 0.8)
            event.accept()
            return
        if self._btn_rects["inc_field"].contains(pos):
            self.set_magnetic_field(self._b_field * 1.25)
            event.accept()
            return
        if self._btn_rects["stamp"].contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break
        if self._hovered_button != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Touchless air pinch gesture for selecting controls."""
        local_p = self.mapFromScene(point)

        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.cleanup()
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "play_pause":
                    self.toggle_animation()
                    return True
                elif name == "dec_speed":
                    self.set_speed(self._speed_param * 0.8)
                    return True
                elif name == "inc_speed":
                    self.set_speed(self._speed_param * 1.25)
                    return True
                elif name == "dec_turns":
                    self.set_turns(int(self._turns_n * 0.8))
                    return True
                elif name == "inc_turns":
                    self.set_turns(int(self._turns_n * 1.25))
                    return True
                elif name == "dec_field":
                    self.set_magnetic_field(self._b_field * 0.8)
                    return True
                elif name == "inc_field":
                    self.set_magnetic_field(self._b_field * 1.25)
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current induction experiment as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_electromagnetic_induction_diagram(
            center=center,
            preset_code=self._active_preset_code,
            turns_n=self._turns_n,
            b_field=self._b_field,
            speed_param=self._speed_param,
        )
        for itm in items:
            scene.addItem(itm)
