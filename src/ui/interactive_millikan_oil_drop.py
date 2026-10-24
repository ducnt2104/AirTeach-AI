"""
AirTeach AI – Interactive Modern Physics: Millikan Oil Drop Experiment & Elementary Charge Quantization
========================================================================================================
Provides an authentic physics simulation of Robert Millikan's 1909-1913 Nobel Prize experiment:
  - Microscopic Fluid Dynamics & Stokes Viscous Drag:
      F_drag = 6 * pi * eta_eff * r * v
      Cunningham Slip Correction Factor: f_corr = 1 + (A * lambda_mfp / r)
      eta_eff = eta_air / f_corr
  - Forces on Microscopic Oil Droplet (Density rho_oil ≈ 886 kg/m^3, r ≈ 1.5 μm):
      1. Gravity:  F_g = m * g = (4/3) * pi * r^3 * rho_oil * g
      2. Buoyancy: F_b = (4/3) * pi * r^3 * rho_air * g
      3. Net Gravity: F_net_g = F_g - F_b = (4/3) * pi * r^3 * (rho_oil - rho_air) * g
      4. Electric Force: F_e = q * E = q * (V / d) (parallel plates distance d = 5.0 mm)
  - Droplet Kinematic Motion:
      - Terminal downward fall velocity (V = 0):
          v_1 = F_net_g / (6 * pi * eta_eff * r)
      - Terminal upward rise velocity (V > V_balance):
          v_2 = (F_e - F_net_g) / (6 * pi * eta_eff * r)
      - Static Balanced Suspension (v = 0):
          F_e = F_net_g  ==>  V_bal = F_net_g * d / q
  - Elementary Charge Quantization:
      Every measured charge q is an integral multiple of fundamental charge:
          q = n * e, where e = 1.602176634 * 10^(-19) C, n in {-1, -2, -3, -4, -5, ...}
  - 5 Curriculum Presets:
      1. BALANCED_SUSPENSION_SINGLE_ELECTRON: Droplet with q = -1e suspended in static balance.
      2. MULTI_ELECTRON_CHARGES: Droplets with q = -1e, -2e, -3e, -4e, -5e illustrating GCD quantization.
      3. TERMINAL_VELOCITY_STOKES_FALL: Free fall under gravity and viscosity with micrometer reticle.
      4. XRAY_IONIZATION_CHARGE_JUMP: X-ray ionizing photon pulse alters electron count Δn = ±1.
      5. STOKES_CUNNINGHAM_CORRECTION: Hydrodynamic boundary slip comparison for submicron droplets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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


# Physical Constants (CODATA Recommended Values)
E_CHARGE = 1.602176634e-19           # Elementary charge in Coulombs (C)
G_ACCEL = 9.80665                    # Gravitational acceleration (m/s^2)
RHO_OIL = 886.0                      # Mineral oil density (kg/m^3)
RHO_AIR = 1.204                      # Air density at 20°C (kg/m^3)
ETA_AIR = 1.825e-5                   # Air dynamic viscosity (Pa·s)
LAMBDA_MFP = 68.0e-9                 # Air molecules mean free path (m)
CUNNINGHAM_A = 0.86                  # Empirical Cunningham slip coefficient
PLATE_DISTANCE_M = 0.0025             # Capacitor plate spacing: 2.5 mm


@dataclass(frozen=True)
class MillikanPreset:
    code: str
    title_vi: str
    badge_label: str
    electrons: int
    radius_um: float
    description: str


PRESETS: dict[str, MillikanPreset] = {
    "BALANCED_SUSPENSION_SINGLE_ELECTRON": MillikanPreset(
        code="BALANCED_SUSPENSION_SINGLE_ELECTRON",
        title_vi="Cân bằng tĩnh 1 electron đơn lẻ (q = -1e)",
        badge_label="CÂN BẰNG 1e",
        electrons=-1,
        radius_um=0.85,
        description="Lực điện trường F_e hướng lên triệt tiêu hoàn toàn trọng lực hiệu dụng, giọt dầu đứng yên lơ lửng.",
    ),
    "MULTI_ELECTRON_CHARGES": MillikanPreset(
        code="MULTI_ELECTRON_CHARGES",
        title_vi="Lượng tử hóa điện tích (q = n·e với n = 1..5)",
        badge_label="LƯỢNG TỬ HÓA",
        electrons=-3,
        radius_um=0.85,
        description="Mọi điện tích đo được đều là bội số nguyên của điện tích nguyên tố e = 1.602 × 10⁻¹⁹ C.",
    ),
    "TERMINAL_VELOCITY_STOKES_FALL": MillikanPreset(
        code="TERMINAL_VELOCITY_STOKES_FALL",
        title_vi="Rơi tự do Stokes không điện trường (V = 0)",
        badge_label="RƠI STOKES",
        electrons=-2,
        radius_um=0.90,
        description="Đo vận tốc giới hạn rơi v₁ qua thước trắc vi thị kính để xác định chính xác bán kính r và khối lượng m.",
    ),
    "XRAY_IONIZATION_CHARGE_JUMP": MillikanPreset(
        code="XRAY_IONIZATION_CHARGE_JUMP",
        title_vi="Chiếu tia X ion hóa làm nhảy điện tích (Δn = ±1)",
        badge_label="TIA X ION HÓA",
        electrons=-2,
        radius_um=0.85,
        description="Tia X ion hóa không khí, electron tự do bám vào giọt dầu làm thay đổi điện tích tức thời kèm bước nhảy vận tốc.",
    ),
    "STOKES_CUNNINGHAM_CORRECTION": MillikanPreset(
        code="STOKES_CUNNINGHAM_CORRECTION",
        title_vi="Hiệu chỉnh Cunningham trượt phân tử khí",
        badge_label="CUNNINGHAM",
        electrons=-1,
        radius_um=0.65,
        description="Với giọt dầu kích thước dưới micromet, quãng đường tự do trung bình của phân tử khí làm giảm lực cản nhớt.",
    ),
}


class InteractiveMillikanOilDropItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Robert Millikan's Oil Drop Experiment,
    elementary charge quantization (q = n*e), Stokes drag force, and terminal velocities.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    charge_changed = Signal(int)  # Emits current integer electron count n

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)

        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Simulation Preset & Droplet Parameters
        self._active_preset_code: str = "BALANCED_SUSPENSION_SINGLE_ELECTRON"
        self._preset: MillikanPreset = PRESETS[self._active_preset_code]

        self._electrons_n: int = -1              # Integer number of excess electrons (negative for oil droplets)
        self._radius_m: float = 0.85e-6          # Droplet radius (0.85 μm)
        self._voltage_v: float = 0.0             # Applied capacitor voltage in Volts (0 .. 1000 V)
        self._is_voltage_on: bool = True
        self._cunningham_enabled: bool = True

        # Droplet Kinematics & Position
        self._droplet_y_norm: float = 0.50       # 0.0 = top plate, 1.0 = bottom plate
        self._sim_velocity_mps: float = 0.0      # Current terminal velocity in m/s (+ up, - down)
        self._is_xray_active: bool = False
        self._xray_timer_ticks: int = 0

        # General Animation
        self._anim_phase: float = 0.0
        self._is_animating: bool = True

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # Set initial balancing voltage for preset
        self.load_preset(self._active_preset_code)

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physics Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def droplet_radius_um(self) -> float:
        """Droplet radius in micrometers."""
        return self._radius_m * 1e6

    @property
    def droplet_volume_m3(self) -> float:
        """Volume V = (4/3) * pi * r^3."""
        return (4.0 / 3.0) * math.pi * (self._radius_m ** 3)

    @property
    def droplet_mass_kg(self) -> float:
        """Mass m = V * rho_oil."""
        return self.droplet_volume_m3 * RHO_OIL

    @property
    def net_gravity_force_n(self) -> float:
        """Net downward gravitational force: F_net_g = V * (rho_oil - rho_air) * g."""
        delta_rho = RHO_OIL - RHO_AIR
        return self.droplet_volume_m3 * delta_rho * G_ACCEL

    @property
    def cunningham_correction_factor(self) -> float:
        """Cunningham factor f_corr = 1 + A * lambda / r."""
        if not self._cunningham_enabled:
            return 1.0
        return 1.0 + (CUNNINGHAM_A * LAMBDA_MFP / self._radius_m)

    @property
    def effective_air_viscosity(self) -> float:
        """eta_eff = eta_air / f_corr."""
        return ETA_AIR / self.cunningham_correction_factor

    @property
    def droplet_charge_c(self) -> float:
        """Total electric charge q = n * e."""
        return float(self._electrons_n) * E_CHARGE

    @property
    def electric_field_vpm(self) -> float:
        """Capacitor electric field E = V / d in V/m."""
        if not self._is_voltage_on:
            return 0.0
        return self._voltage_v / PLATE_DISTANCE_M

    @property
    def electric_force_n(self) -> float:
        """Upward electric force: F_e = -q * E (top plate is positive, attracts negative electrons)."""
        # Note: q is negative (-n * e), so with positive E, upward force is |q| * E
        return abs(self.droplet_charge_c) * self.electric_field_vpm

    @property
    def terminal_velocity_mps(self) -> float:
        """
        Steady-state terminal velocity in m/s (+ upward, - downward):
        v = (F_e - F_net_g) / (6 * pi * eta_eff * r).
        """
        stokes_coeff = 6.0 * math.pi * self.effective_air_viscosity * self._radius_m
        net_f = self.electric_force_n - self.net_gravity_force_n
        return net_f / stokes_coeff

    @property
    def balanced_voltage_v(self) -> float:
        """Voltage V required to balance droplet stationary in mid-air (v = 0)."""
        if abs(self._electrons_n) == 0:
            return 9999.0
        return (self.net_gravity_force_n * PLATE_DISTANCE_M) / (abs(self.droplet_charge_c))

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> MillikanPreset:
        return self._preset

    @property
    def electrons_count(self) -> int:
        return self._electrons_n

    @property
    def voltage_v(self) -> float:
        return self._voltage_v

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum preset and sets appropriate physical state."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._electrons_n = self._preset.electrons
        self._radius_m = self._preset.radius_um * 1e-6
        self._cunningham_enabled = (preset_code == "STOKES_CUNNINGHAM_CORRECTION")

        if preset_code == "TERMINAL_VELOCITY_STOKES_FALL":
            self._voltage_v = 0.0
            self._is_voltage_on = False
        else:
            self._is_voltage_on = True
            # Set to near balanced voltage
            self._voltage_v = round(self.balanced_voltage_v, 1)

        self._droplet_y_norm = 0.45
        self.charge_changed.emit(self._electrons_n)
        self.update()

    def set_voltage(self, volts: float) -> None:
        self._voltage_v = max(0.0, min(1000.0, round(volts, 1)))
        self._is_voltage_on = (self._voltage_v > 0.0)
        self.update()

    def set_electrons_count(self, n: int) -> None:
        # Negative integer (electrons)
        self._electrons_n = -max(1, min(10, abs(n)))
        self.charge_changed.emit(self._electrons_n)
        self.update()

    def set_droplet_radius_um(self, r_um: float) -> None:
        self._radius_m = max(0.5, min(4.0, r_um)) * 1e-6
        self.update()

    def fire_xray_pulse(self) -> None:
        """Fires an X-ray ionizing pulse, altering electron charge by ±1e."""
        self._is_xray_active = True
        self._xray_timer_ticks = 15  # ~0.5s visual beam flash
        delta = random.choice([-1, 1])
        new_n = self._electrons_n + delta
        if new_n == 0:
            new_n = -1
        self.set_electrons_count(new_n)
        self.update()

    def spray_new_droplet(self) -> None:
        """Simulates atomizer spray injecting a fresh droplet with random size and charge."""
        self._radius_m = random.uniform(1.2, 2.2) * 1e-6
        self._electrons_n = -random.randint(1, 5)
        self._droplet_y_norm = 0.20
        self.charge_changed.emit(self._electrons_n)
        self.update()

    def toggle_voltage(self) -> None:
        self._is_voltage_on = not self._is_voltage_on
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        dt = 0.033
        self._anim_phase += 0.08
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi

        if self._is_xray_active:
            self._xray_timer_ticks -= 1
            if self._xray_timer_ticks <= 0:
                self._is_xray_active = False

        # Physical velocity (m/s)
        v_mps = self.terminal_velocity_mps
        self._sim_velocity_mps = v_mps

        # Map physical velocity to normalized plate spacing motion:
        # Distance between plates = 5.0 mm = 0.005 m.
        # dt * v_mps / 0.005 gives fractional movement per tick
        # Note: +v is upward (towards norm y = 0.0), -v is downward (towards norm y = 1.0)
        frac_move = -(v_mps * dt) / PLATE_DISTANCE_M

        self._droplet_y_norm += frac_move

        # Clamping at top and bottom plates (bouncing / resetting)
        if self._droplet_y_norm < 0.08:
            self._droplet_y_norm = 0.08
        elif self._droplet_y_norm > 0.92:
            self._droplet_y_norm = 0.92

        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#070b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Capacitor Chamber & Microscope Eyepiece)
        left_w = 445.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_viewport(painter, view_rect)

        # 4. Right Control & Data Panel
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Physics Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "THÍ NGHIỆM GIỌT DẦU MILLIKAN & LƯỢNG TỬ HÓA ĐIỆN TÍCH (1909)")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Lực cản nhớt Stokes • Điện trường bản cực E=V/d • Điện tích nguyên tố e = 1.602×10⁻¹⁹ C")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # 1. Capacitor Plates (Top: Positive Anode, Bottom: Negative Cathode)
        plate_w = rect.width() - 80.0
        plate_x = rect.left() + 40.0
        top_y = rect.top() + 35.0
        bot_y = rect.bottom() - 35.0

        # Top Plate (+)
        top_plate = QRectF(plate_x, top_y, plate_w, 14.0)
        painter.setBrush(QBrush(QColor("#7f1d1d") if self._is_voltage_on else QColor("#334155")))
        painter.setPen(QPen(QColor("#ef4444") if self._is_voltage_on else QColor("#475569"), 1.2))
        painter.drawRoundedRect(top_plate, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(top_plate, Qt.AlignmentFlag.AlignCenter, f"BẢN CỰC DƯƠNG (+)   V = {self._voltage_v:.1f} V" if self._is_voltage_on else "BẢN CỰC TRÊN (V = 0)")

        # Pin-hole in top plate for oil droplet entrance
        hole_x = rect.center().x() - 10.0
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(hole_x, top_y, 20.0, 14.0))

        # Bottom Plate (-)
        bot_plate = QRectF(plate_x, bot_y - 14.0, plate_w, 14.0)
        painter.setBrush(QBrush(QColor("#1e3a8a") if self._is_voltage_on else QColor("#334155")))
        painter.setPen(QPen(QColor("#3b82f6") if self._is_voltage_on else QColor("#475569"), 1.2))
        painter.drawRoundedRect(bot_plate, 3.0, 3.0)

        painter.setPen(QColor("#ffffff"))
        painter.drawText(bot_plate, Qt.AlignmentFlag.AlignCenter, "BẢN CỰC ÂM (-)   TIẾP ĐẤT / GROUND" if self._is_voltage_on else "BẢN CỰC DƯỚI (V = 0)")

        # 2. Electric Field Lines inside capacitor
        if self._is_voltage_on and self._voltage_v > 0.0:
            painter.setPen(QPen(QColor(56, 189, 248, 45), 1.0, Qt.PenStyle.DashLine))
            field_y_start = top_y + 14.0
            field_y_end = bot_y - 14.0
            for fx in range(int(plate_x + 30.0), int(plate_x + plate_w - 20.0), 40):
                painter.drawLine(QPointF(fx, field_y_start), QPointF(fx, field_y_end))

        # 3. Eyepiece Reticle Measurement Grid (Microscope view lines)
        reticle_x = rect.center().x()
        chamber_h = (bot_y - 14.0) - (top_y + 14.0)
        painter.setPen(QPen(QColor("#334155"), 0.8))

        for tick in range(1, 10):
            ty = (top_y + 14.0) + (tick / 10.0) * chamber_h
            painter.drawLine(QPointF(reticle_x - 45.0, ty), QPointF(reticle_x + 45.0, ty))
            painter.setFont(QFont("Segoe UI", 5))
            painter.setPen(QColor("#64748b"))
            painter.drawText(QPointF(reticle_x + 50.0, ty + 3.0), f"{tick * 0.5:.1f} mm")
            painter.setPen(QPen(QColor("#334155"), 0.8))

        # 4. X-ray Ionization Beam (if active)
        if self._is_xray_active:
            beam_y = top_y + 14.0 + 0.5 * chamber_h
            beam_path = QPainterPath()
            beam_path.moveTo(rect.left(), beam_y - 18.0)
            beam_path.lineTo(rect.center().x() + 30.0, beam_y)
            beam_path.lineTo(rect.left(), beam_y + 18.0)
            beam_path.closeSubpath()

            painter.setBrush(QBrush(QColor(168, 85, 247, 70)))
            painter.setPen(QPen(QColor("#a855f7"), 1.5, Qt.PenStyle.DashLine))
            painter.drawPath(beam_path)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#c084fc"))
            painter.drawText(QPointF(rect.left() + 10.0, beam_y - 24.0), "⚡ CHÙM TIA X ION HÓA KHÔNG KHÍ")

        # 5. Oil Droplet
        drop_x = rect.center().x()
        drop_y = (top_y + 14.0) + self._droplet_y_norm * chamber_h
        r_px = max(6.0, min(14.0, self.droplet_radius_um * 4.5))

        # Radiant glowing droplet
        painter.setBrush(QBrush(QColor("#fbbf24")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(drop_x, drop_y), r_px, r_px)

        # Negative charges minus symbols inside droplet
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#000000"))
        painter.drawText(QRectF(drop_x - r_px, drop_y - r_px, 2 * r_px, 2 * r_px), Qt.AlignmentFlag.AlignCenter, f"{abs(self._electrons_n)}e⁻")

        # 6. Force Vectors on Droplet (Free Body Diagram)
        fe = self.electric_force_n
        fg = self.net_gravity_force_n
        scale_f = 2e12  # px per Newton

        # Net gravity downward (Red)
        len_fg = min(60.0, fg * scale_f)
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        painter.drawLine(QPointF(drop_x, drop_y), QPointF(drop_x, drop_y + len_fg))
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(QPointF(drop_x + 12.0, drop_y + len_fg), f"F_g={fg*1e12:.1f}pN")

        # Electric force upward (Cyan/Green)
        if self._is_voltage_on and fe > 0:
            len_fe = min(60.0, fe * scale_f)
            painter.setPen(QPen(QColor("#10b981"), 2.0))
            painter.drawLine(QPointF(drop_x, drop_y), QPointF(drop_x, drop_y - len_fe))
            painter.setPen(QColor("#10b981"))
            painter.drawText(QPointF(drop_x + 12.0, drop_y - len_fe), f"F_e={fe*1e12:.1f}pN")

        # Motion status banner
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        v_state = "LƠ LỬNG CÂN BẰNG (v = 0)" if abs(self._sim_velocity_mps) < 1e-6 else ("BAY LÊN TRÊN ↑" if self._sim_velocity_mps > 0 else "RƠI XUỐNG DƯỚI ↓")
        v_col = QColor("#10b981") if abs(self._sim_velocity_mps) < 1e-6 else (QColor("#38bdf8") if self._sim_velocity_mps > 0 else QColor("#f43f5e"))
        painter.setPen(v_col)
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), f"TRẠNG THÁI: {v_state} • v = {self._sim_velocity_mps*1e3:.2f} mm/s")

        painter.restore()

    # -------------------------------------------------------------------------
    # Right Control & Readout Panel
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THÔNG SỐ ĐIỆN & KHÍ NƯỚC:")

        # Voltage Card
        y += 18.0
        v_card = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(v_card, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 95.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"{self._voltage_v:.1f} V")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        v_bal = self.balanced_voltage_v
        painter.drawText(QRectF(ox + 105.0, y + 2.0, w - 110.0, 28.0), Qt.AlignmentFlag.AlignVCenter, f"V_cân bằng: {v_bal:.1f} V\nE = {self.electric_field_vpm/1000.0:.2f} kV/m")

        # Voltage Stepper Buttons
        y += 38.0
        btn_vm10 = QRectF(ox, y, 32.0, 20.0)
        btn_vm1 = QRectF(ox + 36.0, y, 32.0, 20.0)
        btn_vp1 = QRectF(ox + 100.0, y, 32.0, 20.0)
        btn_vp10 = QRectF(ox + 136.0, y, 32.0, 20.0)

        self._btn_rects["v_m10"] = btn_vm10
        self._btn_rects["v_m1"] = btn_vm1
        self._btn_rects["v_p1"] = btn_vp1
        self._btn_rects["v_p10"] = btn_vp10

        for b_r, b_txt in [(btn_vm10, "-10V"), (btn_vm1, "-1V"), (btn_vp1, "+1V"), (btn_vp10, "+10V")]:
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)

        # Microscopic Physics Measurements Readout
        y += 28.0
        q_c = self.droplet_charge_c
        r_um = self.droplet_radius_um
        m_kg = self.droplet_mass_kg

        rows = [
            (f"• Số electron n: {abs(self._electrons_n)} (q = {abs(q_c)*1e19:.3f} ×10⁻¹⁹ C)", QColor("#10b981")),
            (f"• Bán kính giọt r: {r_um:.2f} μm", QColor("#38bdf8")),
            (f"• Khối lượng m: {m_kg*1e15:.2f} fg (10⁻¹⁵ kg)", QColor("#cbd5e1")),
            (f"• Trọng lực F_g: {self.net_gravity_force_n*1e12:.2f} pN", QColor("#ef4444")),
            (f"• Lực điện F_e: {self.electric_force_n*1e12:.2f} pN", QColor("#10b981")),
            (f"• Vận tốc v: {self.terminal_velocity_mps*1e3:+.2f} mm/s", QColor("#fbbf24")),
            (f"• Hệ số Cunningham: {self.cunningham_correction_factor:.3f}", QColor("#94a3b8")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 16.0

        # Action Buttons: X-ray, Spray, Balance Auto
        y = rect.bottom() - 50.0
        xray_r = QRectF(ox, y, w / 2.0 - 4.0, 22.0)
        spray_r = QRectF(ox + w / 2.0 + 4.0, y, w / 2.0 - 4.0, 22.0)
        self._btn_rects["xray"] = xray_r
        self._btn_rects["spray"] = spray_r

        painter.setBrush(QBrush(QColor("#7e22ce") if not self._is_xray_active else QColor("#a855f7")))
        painter.setPen(QPen(QColor("#c084fc"), 1.0))
        painter.drawRoundedRect(xray_r, 4.0, 4.0)

        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(spray_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(xray_r, Qt.AlignmentFlag.AlignCenter, "⚡ Tia X Ion hóa")
        painter.drawText(spray_r, Qt.AlignmentFlag.AlignCenter, "💧 Xịt giọt mới")

        # Auto-Balance button
        y += 26.0
        auto_bal_r = QRectF(ox, y, w, 18.0)
        self._btn_rects["auto_balance"] = auto_bal_r
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(auto_bal_r, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(auto_bal_r, Qt.AlignmentFlag.AlignCenter, "⚖ Tự động cân bằng điện áp (V = V_bal)")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "ĐỊNH LUẬT LƯỢNG TỬ HÓA ĐIỆN TÍCH MILLIKAN (GIẢI NOBEL VẬT LÝ 1923):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Định luật Millikan: q = n·e (e = 1.602176634 × 10⁻¹⁹ C). Điện tích không liên tục mà bị lượng tử hóa thành các gói rời rạc.\n"
            f"• Lực cản nhớt Stokes F_d = 6πηrv kết hợp số hiệu chỉnh Cunningham f_corr = 1 + A·λ/r bù trừ hiện tượng trượt phân tử khí ở cấp micromet.\n"
            f"• Mô tả chế độ: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Voltage steppers
        if self._btn_rects.get("v_m10") and self._btn_rects["v_m10"].contains(local_pos):
            self.set_voltage(self._voltage_v - 10.0)
            return True
        if self._btn_rects.get("v_m1") and self._btn_rects["v_m1"].contains(local_pos):
            self.set_voltage(self._voltage_v - 1.0)
            return True
        if self._btn_rects.get("v_p1") and self._btn_rects["v_p1"].contains(local_pos):
            self.set_voltage(self._voltage_v + 1.0)
            return True
        if self._btn_rects.get("v_p10") and self._btn_rects["v_p10"].contains(local_pos):
            self.set_voltage(self._voltage_v + 10.0)
            return True

        # 3. Action buttons
        if self._btn_rects.get("xray") and self._btn_rects["xray"].contains(local_pos):
            self.fire_xray_pulse()
            return True
        if self._btn_rects.get("spray") and self._btn_rects["spray"].contains(local_pos):
            self.spray_new_droplet()
            return True
        if self._btn_rects.get("auto_balance") and self._btn_rects["auto_balance"].contains(local_pos):
            self.set_voltage(self.balanced_voltage_v)
            return True
        if self._btn_rects.get("stamp") and self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
