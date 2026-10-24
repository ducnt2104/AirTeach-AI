"""
AirTeach AI - Interactive AC Series RLC Circuit & Fresnel Phasor Diagram Simulator
===================================================================================
Provides an authentic physics and electrical engineering simulation of alternating current (AC):
  - Series RLC Circuit Equations:
      u(t) = U₀ * cos(ω * t) = U * sqrt(2) * cos(2π * f * t)
      i(t) = I₀ * cos(ω * t - φ) = I * sqrt(2) * cos(2π * f * t - φ)
  - Key Electrical Quantities:
      - Resistor: R [Ω]
      - Inductor: L [H] => Inductive reactance: Z_L = ω * L = 2π * f * L [Ω]
      - Capacitor: C [F] => Capacitive reactance: Z_C = 1 / (ω * C) = 1 / (2π * f * C) [Ω]
      - Total Impedance: Z = sqrt(R² + (Z_L - Z_C)²) [Ω]
      - RMS Current: I = U / Z [A]
      - Phase Shift: tan(φ) = (Z_L - Z_C) / R => φ = arctan((Z_L - Z_C) / R)
      - Power Factor: cos(φ) = R / Z
      - Active Power: P = U * I * cos(φ) = I² * R [W]
  - Electrical Resonance Condition (Hiện tượng Cộng Hưởng Điện):
      Z_L = Z_C <=> ω₀ = 1 / sqrt(L * C) <=> f₀ = 1 / (2π * sqrt(L * C))
      At resonance:
        Z_min = R
        I_max = U / R
        φ = 0 (voltage and current in phase)
        cos(φ) = 1.0 (maximum active power)
        Quality Factor: Q = (ω₀ * L) / R = (1 / R) * sqrt(L / C)
  - Dynamic Visual Elements:
      - Circuit schematic with AC source, R, L, C, ammeter, and oscillating current arrows.
      - Fresnel Phasor Diagram: Vector U_R (in-phase), U_L (+90°), U_C (-90°), and composite U.
      - Dual-Trace Digital Oscilloscope: Real-time traces of u(t) (Amber) and i(t) (Cyan).
      - Touchless air pinch / slider controls on f, R, L, C with instant resonance snap.
      - Vector stamping to LessonBoard via PhysicsTools.create_rlc_circuit_diagram.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

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

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


class InteractiveRlcCircuitItem(QGraphicsObject):
    """
    Interactive AC Series RLC Circuit & Fresnel Phasor Diagram Simulator item for QGraphicsScene.
    """

    rlc_updated = Signal(float, float, float, float)  # (impedance_Z, current_I, phase_deg, cos_phi)
    preset_changed = Signal(str)                       # ("RESONANCE_50HZ", etc.)
    resonance_triggered = Signal(float, float)         # (f_res_hz, Q_factor)

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 160.0),
        preset: str = "RESONANCE_50HZ",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(38.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Dimensions
        self._width: float = 840.0
        self._height: float = 560.0

        # Physical circuit parameters
        self._rms_voltage_U: float = 220.0   # RMS Voltage [V]
        self._frequency_f: float = 50.0      # Frequency f [Hz]
        self._resistor_R: float = 50.0       # Resistance R [Ω]
        self._inductor_L: float = 0.3183     # Inductance L [H] (gives Z_L ≈ 100 Ω at 50Hz)
        self._capacitor_C_uF: float = 31.83  # Capacitance C [μF] (gives Z_C ≈ 100 Ω at 50Hz)
        self._preset: str = preset

        # Visual toggles
        self._show_phasor_diagram: bool = True
        self._show_oscilloscope: bool = True
        self._show_schematic: bool = True
        self._is_animating: bool = True

        # Left canvas layout
        self._canvas_rect = QRectF(15.0, 45.0, 500.0, 500.0)

        # Animation phase timer (40 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(25)  # 40 FPS
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._sim_time_s: float = 0.0

        # Interaction drag state
        self._drag_target: Optional[str] = None  # "TITLE", "SLIDER_F", "SLIDER_R", "SLIDER_L", "SLIDER_C"
        self._drag_offset: QPointF = QPointF(0.0, 0.0)
        self._hover_target: Optional[str] = None

        # Apply initial preset and start timer
        self._apply_preset(self._preset)
        self._anim_timer.start()

    # ── Physical Calculations ─────────────────────────────────────────────────

    @property
    def omega(self) -> float:
        """Angular frequency ω = 2π * f [rad/s]."""
        return 2.0 * math.pi * self._frequency_f

    @property
    def inductive_reactance_ZL(self) -> float:
        """Inductive reactance Z_L = ω * L [Ω]."""
        return self.omega * self._inductor_L

    @property
    def capacitive_reactance_ZC(self) -> float:
        """Capacitive reactance Z_C = 1 / (ω * C) [Ω]."""
        c_farad = self._capacitor_C_uF * 1e-6
        if c_farad <= 0.0 or self.omega <= 0.0:
            return 1e6
        return 1.0 / (self.omega * c_farad)

    @property
    def total_impedance_Z(self) -> float:
        """Total circuit impedance Z = sqrt(R² + (Z_L - Z_C)²) [Ω]."""
        x_net = self.inductive_reactance_ZL - self.capacitive_reactance_ZC
        return math.sqrt(self._resistor_R * self._resistor_R + x_net * x_net)

    @property
    def rms_current_I(self) -> float:
        """RMS current I = U / Z [A]."""
        z = self.total_impedance_Z
        return self._rms_voltage_U / z if z > 0.0 else 0.0

    @property
    def phase_rad(self) -> float:
        """Phase difference φ = arctan((Z_L - Z_C) / R) in radians."""
        if self._resistor_R <= 0.0:
            return math.pi / 2.0 if self.inductive_reactance_ZL >= self.capacitive_reactance_ZC else -math.pi / 2.0
        x_net = self.inductive_reactance_ZL - self.capacitive_reactance_ZC
        return math.atan(x_net / self._resistor_R)

    @property
    def phase_deg(self) -> float:
        """Phase difference in degrees."""
        return math.degrees(self.phase_rad)

    @property
    def power_factor(self) -> float:
        """Power factor cos(φ) = R / Z."""
        z = self.total_impedance_Z
        return self._resistor_R / z if z > 0.0 else 0.0

    @property
    def active_power_P(self) -> float:
        """Active power P = U * I * cos(φ) [W]."""
        return self._rms_voltage_U * self.rms_current_I * self.power_factor

    @property
    def resonance_frequency_f0(self) -> float:
        """Resonant frequency f₀ = 1 / (2π * sqrt(LC)) [Hz]."""
        c_farad = self._capacitor_C_uF * 1e-6
        if self._inductor_L <= 0.0 or c_farad <= 0.0:
            return 0.0
        return 1.0 / (2.0 * math.pi * math.sqrt(self._inductor_L * c_farad))

    @property
    def quality_factor_Q(self) -> float:
        """Quality factor Q = (1 / R) * sqrt(L / C)."""
        c_farad = self._capacitor_C_uF * 1e-6
        if self._resistor_R <= 0.0 or c_farad <= 0.0 or self._inductor_L <= 0.0:
            return 0.0
        return (1.0 / self._resistor_R) * math.sqrt(self._inductor_L / c_farad)

    # ── Animation Timer ───────────────────────────────────────────────────────

    def _on_animation_tick(self) -> None:
        """Advances simulation time for live phasor rotation and oscilloscope wave."""
        if self._is_animating:
            self._sim_time_s += 0.0015
            self.update()

    # ── Bounding Rect & Painting ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Container
        self._paint_outer_container(painter)

        # 2. Left Electrical Canvas (Schematic + Phasor Diagram + Oscilloscope)
        self._paint_electrical_canvas(painter)

        # 3. Right Control & Telemetry HUD
        self._paint_controls_and_hud(painter)

        painter.restore()

    def _paint_outer_container(self, painter: QPainter) -> None:
        outer = self.boundingRect()

        # Slate-900 background
        painter.setPen(QPen(QColor("#334155"), 1.8))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))
        painter.drawRoundedRect(outer, 14.0, 14.0)

        # Title bar
        title_rect = QRectF(0.0, 0.0, self._width, 40.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
        painter.drawRoundedRect(title_rect, 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 26.0, self._width, 14.0))

        # Title Text
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(18.0, 8.0, 520.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "⚡ MẠCH ĐIỆN XOAY CHIỀU RLC NỐI TIẾP & GIẢN ĐỒ FRESNEL",
        )

        # Preset badge
        preset_labels = {
            "RESONANCE_50HZ": "🎯 Cộng Hưởng Điện (f = f₀ = 50Hz)",
            "INDUCTIVE_CIRCUIT": "🧲 Tính Cảm Kháng (Z_L > Z_C)",
            "CAPACITIVE_CIRCUIT": "⚡ Tính Dung Kháng (Z_C > Z_L)",
            "HIGH_Q_RESONANCE": "🔥 Hệ Số Phẩm Chất Cao (Q = 10)",
        }
        badge = preset_labels.get(self._preset, "Mạch RLC Xoay Chiều")
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(520.0, 8.0, 260.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            badge,
        )

        # Close button
        c_btn = self._get_close_button_rect()
        h_close = (self._hover_target == "CLOSE")
        painter.setBrush(QBrush(QColor("#dc2626") if h_close else QColor(239, 68, 68, 160)))
        painter.setPen(QPen(QColor("#fca5a5"), 1.2))
        painter.drawRoundedRect(c_btn, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(c_btn, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_electrical_canvas(self, painter: QPainter) -> None:
        """Paints RLC schematic (top), Fresnel Phasor diagram (middle), and Oscilloscope (bottom)."""
        painter.save()
        painter.setClipRect(self._canvas_rect)

        # Deep cosmic viewport
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.setBrush(QBrush(QColor("#030712")))
        painter.drawRoundedRect(self._canvas_rect, 8.0, 8.0)

        # ── 1. Top Section: Series RLC Circuit Schematic ──
        if self._show_schematic:
            self._paint_circuit_schematic(painter)

        # ── 2. Middle Section: Fresnel Phasor Diagram ──
        if self._show_phasor_diagram:
            self._paint_phasor_diagram(painter)

        # ── 3. Bottom Section: Dual-Trace Digital Oscilloscope ──
        if self._show_oscilloscope:
            self._paint_oscilloscope(painter)

        painter.restore()

    def _paint_circuit_schematic(self, painter: QPainter) -> None:
        """Paints schematic circuit loop with AC generator, R, L, C, and current flow."""
        y_sch = 75.0
        x_left = 45.0
        x_right = 485.0
        h_sch = 55.0

        painter.setPen(QPen(QColor("#64748b"), 1.6))

        # Top wire containing R, L, C
        painter.drawLine(QPointF(x_left, y_sch), QPointF(x_left + 45.0, y_sch))

        # Resistor R rectangle
        r_rect = QRectF(x_left + 45.0, y_sch - 8.0, 60.0, 16.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawRect(r_rect)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(r_rect, Qt.AlignmentFlag.AlignCenter, f"R {self._resistor_R:.0f}Ω")

        # Wire to Inductor
        painter.setPen(QPen(QColor("#64748b"), 1.6))
        painter.drawLine(QPointF(x_left + 105.0, y_sch), QPointF(x_left + 145.0, y_sch))

        # Inductor L (4 looped arches)
        ind_path = QPainterPath()
        ind_path.moveTo(x_left + 145.0, y_sch)
        for k in range(4):
            ax = x_left + 145.0 + k * 16.0
            ind_path.arcTo(QRectF(ax, y_sch - 12.0, 16.0, 24.0), 180.0, -180.0)
        painter.setPen(QPen(QColor("#10b981"), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(ind_path)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(x_left + 145.0, y_sch - 26.0, 64.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"L {self._inductor_L:.2f}H")

        # Wire to Capacitor
        painter.setPen(QPen(QColor("#64748b"), 1.6))
        painter.drawLine(QPointF(x_left + 209.0, y_sch), QPointF(x_left + 265.0, y_sch))

        # Capacitor C (Two parallel plates)
        c_x = x_left + 265.0
        painter.setPen(QPen(QColor("#f59e0b"), 2.2))
        painter.drawLine(QPointF(c_x, y_sch - 12.0), QPointF(c_x, y_sch + 12.0))
        painter.drawLine(QPointF(c_x + 8.0, y_sch - 12.0), QPointF(c_x + 8.0, y_sch + 12.0))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(c_x - 12.0, y_sch - 26.0, 48.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"C {self._capacitor_C_uF:.1f}μF")

        # Wire to right side and down
        painter.setPen(QPen(QColor("#64748b"), 1.6))
        painter.drawLine(QPointF(c_x + 8.0, y_sch), QPointF(x_right, y_sch))
        painter.drawLine(QPointF(x_right, y_sch), QPointF(x_right, y_sch + h_sch))

        # Bottom wire with AC source symbol
        y_bot = y_sch + h_sch
        ac_cx = (x_left + x_right) / 2.0
        painter.drawLine(QPointF(x_left, y_bot), QPointF(ac_cx - 16.0, y_bot))
        painter.drawLine(QPointF(ac_cx + 16.0, y_bot), QPointF(x_right, y_bot))

        # AC source circle with sine wave inside
        painter.setPen(QPen(QColor("#cbd5e1"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawEllipse(QPointF(ac_cx, y_bot), 16.0, 16.0)

        sine_path = QPainterPath()
        sine_path.moveTo(ac_cx - 9.0, y_bot)
        sine_path.cubicTo(ac_cx - 4.5, y_bot - 8.0, ac_cx, y_bot - 8.0, ac_cx, y_bot)
        sine_path.cubicTo(ac_cx, y_bot + 8.0, ac_cx + 4.5, y_bot + 8.0, ac_cx + 9.0, y_bot)
        painter.setPen(QPen(QColor("#fbbf24"), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(sine_path)

        # Left wire closing the loop
        painter.setPen(QPen(QColor("#64748b"), 1.6))
        painter.drawLine(QPointF(x_left, y_sch), QPointF(x_left, y_bot))

        # AC Label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(QRectF(ac_cx - 60.0, y_bot + 18.0, 120.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"u(t) = {self._rms_voltage_U:.0f}V, {self._frequency_f:.1f}Hz")

    def _paint_phasor_diagram(self, painter: QPainter) -> None:
        """Paints vector Fresnel diagram: U_R (horizontal), U_L (up), U_C (down), and composite U."""
        cx = 160.0
        cy = 265.0
        scale = 0.55  # pixels per Volt

        i_rms = self.rms_current_I
        u_r = i_rms * self._resistor_R
        u_l = i_rms * self.inductive_reactance_ZL
        u_c = i_rms * self.capacitive_reactance_ZC

        # Box border for Phasor diagram
        phasor_rect = QRectF(30.0, 160.0, 260.0, 210.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(phasor_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QRectF(38.0, 166.0, 200.0, 16.0), Qt.AlignmentFlag.AlignLeft, "GIẢN ĐỒ VÉC-TƠ FRESNEL")

        # Origin crosshair
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(phasor_rect.left() + 10.0, cy), QPointF(phasor_rect.right() - 10.0, cy))
        painter.drawLine(QPointF(cx, phasor_rect.top() + 25.0), QPointF(cx, phasor_rect.bottom() - 10.0))

        # Clamp drawing lengths to fit comfortably inside card
        px_ur = min(110.0, u_r * scale)
        px_ul = min(85.0, u_l * scale)
        px_uc = min(85.0, u_c * scale)
        px_ulc = px_ul - px_uc

        # 1. Vector U_R (along horizontal axis in Cyan)
        pen_ur = QPen(QColor("#38bdf8"), 2.0)
        painter.setPen(pen_ur)
        painter.drawLine(QPointF(cx, cy), QPointF(cx + px_ur, cy))
        self._paint_arrowhead(painter, QPointF(cx + px_ur, cy), 0.0, QColor("#38bdf8"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(QPointF(cx + px_ur + 4.0, cy + 12.0), f"U_R ({u_r:.0f}V)")

        # 2. Vector U_L (upwards in Emerald)
        if px_ul > 2.0:
            pen_ul = QPen(QColor("#10b981"), 2.0)
            painter.setPen(pen_ul)
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy - px_ul))
            self._paint_arrowhead(painter, QPointF(cx, cy - px_ul), -math.pi / 2.0, QColor("#10b981"))
            painter.drawText(QPointF(cx + 4.0, cy - px_ul + 2.0), f"U_L ({u_l:.0f}V)")

        # 3. Vector U_C (downwards in Amber)
        if px_uc > 2.0:
            pen_uc = QPen(QColor("#f59e0b"), 2.0)
            painter.setPen(pen_uc)
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + px_uc))
            self._paint_arrowhead(painter, QPointF(cx, cy + px_uc), math.pi / 2.0, QColor("#f59e0b"))
            painter.drawText(QPointF(cx + 4.0, cy + px_uc + 10.0), f"U_C ({u_c:.0f}V)")

        # 4. Total Composite Voltage Vector U (in Red/Rose)
        tip_u = QPointF(cx + px_ur, cy - px_ulc)
        pen_u = QPen(QColor("#f43f5e"), 2.4)
        painter.setPen(pen_u)
        painter.drawLine(QPointF(cx, cy), tip_u)
        angle_u = math.atan2(-px_ulc, px_ur)
        self._paint_arrowhead(painter, tip_u, angle_u, QColor("#f43f5e"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(QPointF(tip_u.x() + 4.0, tip_u.y() - 4.0), f"U ({self._rms_voltage_U:.0f}V)")

        # Angle arc for phase shift phi
        if abs(self.phase_deg) > 1.0:
            arc_r = 35.0
            painter.setPen(QPen(QColor(244, 63, 94, 160), 1.2, Qt.PenStyle.DotLine))
            painter.drawArc(
                QRectF(cx - arc_r, cy - arc_r, arc_r * 2.0, arc_r * 2.0),
                0,
                int(self.phase_deg * 16),
            )
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.drawText(QPointF(cx + 38.0, cy - (px_ulc * 0.4)), f"φ={self.phase_deg:+.1f}°")

    def _paint_oscilloscope(self, painter: QPainter) -> None:
        """Paints dual-trace digital oscilloscope showing u(t) and i(t) waveforms."""
        osc_rect = QRectF(305.0, 160.0, 200.0, 210.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor("#030712")))
        painter.drawRoundedRect(osc_rect, 6.0, 6.0)

        # Title & Legend
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(QRectF(315.0, 166.0, 90.0, 16.0), Qt.AlignmentFlag.AlignLeft, "— u(t) Điện áp")
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(410.0, 166.0, 90.0, 16.0), Qt.AlignmentFlag.AlignLeft, "— i(t) Dòng điện")

        cy = osc_rect.center().y() + 10.0
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(osc_rect.left() + 5.0, cy), QPointF(osc_rect.right() - 5.0, cy))

        # Render waveform traces
        u_path = QPainterPath()
        i_path = QPainterPath()
        first = True

        steps = 80
        w_px = osc_rect.width() - 16.0
        x_start = osc_rect.left() + 8.0

        phi = self.phase_rad
        t_offset = self._sim_time_s * 2.0 * math.pi * self._frequency_f

        for step in range(steps + 1):
            t_phase = (step / steps) * 4.0 * math.pi + t_offset
            px = x_start + (step / steps) * w_px

            # u(t) = U_0 * cos(t_phase)
            u_val = math.cos(t_phase)
            py_u = cy - u_val * 45.0

            # i(t) = I_0 * cos(t_phase - phi)
            i_val = math.cos(t_phase - phi)
            py_i = cy - i_val * 45.0

            if first:
                u_path.moveTo(px, py_u)
                i_path.moveTo(px, py_i)
                first = False
            else:
                u_path.lineTo(px, py_u)
                i_path.lineTo(px, py_i)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(u_path)

        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawPath(i_path)

    def _paint_arrowhead(self, painter: QPainter, tip: QPointF, angle_rad: float, color: QColor) -> None:
        arrow_size = 7.5
        p1 = QPointF(
            tip.x() - arrow_size * math.cos(angle_rad - math.pi / 6.0),
            tip.y() - arrow_size * math.sin(angle_rad - math.pi / 6.0),
        )
        p2 = QPointF(
            tip.x() - arrow_size * math.cos(angle_rad + math.pi / 6.0),
            tip.y() - arrow_size * math.sin(angle_rad + math.pi / 6.0),
        )
        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(p1)
        path.lineTo(p2)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _paint_controls_and_hud(self, painter: QPainter) -> None:
        """Paints parameters sliders, theoretical telemetry, presets, and action buttons."""
        hud_left = 525.0
        hud_w = 300.0

        # ── 1. Parameters Sliders Card ──
        card1_rect = QRectF(hud_left, 45.0, hud_w, 150.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card1_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 50.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "ĐIỀU KHIỂN THÔNG SỐ MẠCH RLC",
        )

        # Row 1: Frequency f Slider + Resonance Snap Button
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(QRectF(hud_left + 12.0, 70.0, 150.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Tần số f = {self._frequency_f:.1f} Hz")

        snap_btn = self._get_resonance_snap_button_rect()
        h_snap = (self._hover_target == "SNAP_RES")
        painter.setPen(QPen(QColor("#f59e0b"), 1.0))
        painter.setBrush(QBrush(QColor(245, 158, 11, 70) if h_snap else QColor(245, 158, 11, 35)))
        painter.drawRoundedRect(snap_btn, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(snap_btn, Qt.AlignmentFlag.AlignCenter, f"Snap f₀={self.resonance_frequency_f0:.1f}Hz")

        f_track, f_handle = self._get_f_slider_geometry()
        self._paint_mini_slider(painter, f_track, f_handle, QColor("#fbbf24"))

        # Row 2: Resistance R
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(hud_left + 12.0, 102.0, 180.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Điện trở R = {self._resistor_R:.1f} Ω")
        r_track, r_handle = self._get_r_slider_geometry()
        self._paint_mini_slider(painter, r_track, r_handle, QColor("#38bdf8"))

        # Row 3: Inductance L and Capacitance C (side-by-side)
        painter.drawText(QRectF(hud_left + 12.0, 134.0, 130.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"L = {self._inductor_L:.3f} H")
        painter.drawText(QRectF(hud_left + 160.0, 134.0, 130.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"C = {self._capacitor_C_uF:.1f} μF")

        l_track, l_handle = self._get_l_slider_geometry()
        self._paint_mini_slider(painter, l_track, l_handle, QColor("#10b981"))

        c_track, c_handle = self._get_c_slider_geometry()
        self._paint_mini_slider(painter, c_track, c_handle, QColor("#f59e0b"))

        # ── 2. Telemetry & Physical Metrics Card ──
        card2_rect = QRectF(hud_left, 202.0, hud_w, 118.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card2_rect, 8.0, 8.0)

        # Resonance state classification
        zl = self.inductive_reactance_ZL
        zc = self.capacitive_reactance_ZC
        diff_z = abs(zl - zc)
        if diff_z < 1.0:
            status_txt = "🎯 CỘNG HƯỞNG ĐIỆN (f ≈ f₀)"
            status_col = QColor("#10b981")
        elif zl > zc:
            status_txt = "🧲 TÍNH CẢM KHÁNG (u sớm pha hơn i)"
            status_col = QColor("#38bdf8")
        else:
            status_txt = "⚡ TÍNH DUNG KHÁNG (i sớm pha hơn u)"
            status_col = QColor("#f59e0b")

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(status_col))
        painter.drawText(QRectF(hud_left + 12.0, 206.0, hud_w - 24.0, 18.0), Qt.AlignmentFlag.AlignLeft, status_txt)

        # Big Values: Z and I
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(QRectF(hud_left + 12.0, 226.0, 140.0, 20.0), Qt.AlignmentFlag.AlignLeft, f"Z = {self.total_impedance_Z:.1f} Ω")
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(hud_left + 155.0, 226.0, 140.0, 20.0), Qt.AlignmentFlag.AlignLeft, f"I = {self.rms_current_I:.2f} A")

        # Reactance details & power factor
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(
            QRectF(hud_left + 12.0, 250.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Cảm kháng Z_L = {zl:.1f} Ω | Dung kháng Z_C = {zc:.1f} Ω",
        )
        painter.drawText(
            QRectF(hud_left + 12.0, 268.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Độ lệch pha: φ = {self.phase_deg:+.1f}° | Hệ số cos(φ) = {self.power_factor:.3f}",
        )
        painter.drawText(
            QRectF(hud_left + 12.0, 286.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"• Công suất P = {self.active_power_P:.1f} W | Phẩm chất Q = {self.quality_factor_Q:.2f}",
        )

        # ── 3. Presets (2x2 Grid) ──
        card3_rect = QRectF(hud_left, 327.0, hud_w, 98.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card3_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QRectF(hud_left + 12.0, 330.0, hud_w - 24.0, 16.0), Qt.AlignmentFlag.AlignLeft, "BÀI HỌC THỰC NGHIỆM ĐIỆN XOAY CHIỀU")

        presets = [
            ("RESONANCE_50HZ", "🎯 Cộng Hưởng 50Hz", 0, 0),
            ("INDUCTIVE_CIRCUIT", "🧲 Cảm Kháng (Z_L > Z_C)", 0, 1),
            ("CAPACITIVE_CIRCUIT", "⚡ Dung Kháng (Z_C > Z_L)", 1, 0),
            ("HIGH_Q_RESONANCE", "🔥 Phẩm Chất Cao (Q=10)", 1, 1),
        ]
        for p_id, p_label, row, col in presets:
            btn_r = self._get_preset_button_rect(row, col)
            is_active = (self._preset == p_id)
            is_hover = (self._hover_target == f"PRESET_{p_id}")
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(14, 165, 233, 60) if is_active else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 160))))
            painter.drawRoundedRect(btn_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#e2e8f0")))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, p_label)

        # ── 4. Visual Layer Toggles ──
        toggles = [
            ("PHASOR", "🔄 Véc-tơ", self._show_phasor_diagram, 0),
            ("OSC", "📈 Dao Động", self._show_oscilloscope, 1),
            ("ANIM", "▶️ Sóng Chạy", self._is_animating, 2),
        ]
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        for tog_key, tog_label, is_on, idx in toggles:
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 432.0, tog_w, 28.0)
            is_hover = (self._hover_target == f"TOGGLE_{tog_key}")
            painter.setPen(QPen(QColor("#10b981") if is_on else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(16, 185, 129, 60) if is_on else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 140))))
            painter.drawRoundedRect(tog_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#34d399") if is_on else QColor("#94a3b8")))
            painter.drawText(tog_r, Qt.AlignmentFlag.AlignCenter, tog_label)

        # ── 5. Action Buttons (Stamp & Close) ──
        stamp_btn = self._get_stamp_button_rect()
        h_stamp = (self._hover_target == "STAMP")
        painter.setPen(QPen(QColor("#0ea5e9"), 1.5))
        painter.setBrush(QBrush(QColor(14, 165, 233, 210) if h_stamp else QColor(2, 132, 199, 180)))
        painter.drawRoundedRect(stamp_btn, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(stamp_btn, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng Dấu Vector")

        close_b = self._get_close_bottom_button_rect()
        h_cb = (self._hover_target == "CLOSE_BOTTOM")
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(71, 85, 105, 200) if h_cb else QColor(51, 65, 85, 160)))
        painter.drawRoundedRect(close_b, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(close_b, Qt.AlignmentFlag.AlignCenter, "Đóng")

    def _paint_mini_slider(self, painter: QPainter, track: QRectF, handle: QRectF, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(track, 3.0, 3.0)
        filled = QRectF(track.left(), track.top(), handle.center().x() - track.left(), track.height())
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(filled, 3.0, 3.0)
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(handle)

    # ── Geometry & Layout Helpers ─────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 8.0, 24.0, 24.0)

    def _get_resonance_snap_button_rect(self) -> QRectF:
        return QRectF(525.0 + 175.0, 68.0, 110.0, 18.0)

    def _get_f_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 90.0, 276.0, 6.0)
        frac = (self._frequency_f - 10.0) / (200.0 - 10.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_r_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 122.0, 276.0, 6.0)
        frac = (self._resistor_R - 10.0) / (500.0 - 10.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_l_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 154.0, 130.0, 6.0)
        frac = (self._inductor_L - 0.05) / (1.00 - 0.05)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_c_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 160.0, 154.0, 128.0, 6.0)
        frac = (self._capacitor_C_uF - 5.0) / (100.0 - 5.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_preset_button_rect(self, row: int, col: int) -> QRectF:
        hud_left = 525.0
        w = 134.0
        h = 26.0
        x = hud_left + 12.0 + col * (w + 8.0)
        y = 350.0 + row * (h + 8.0)
        return QRectF(x, y, w, h)

    def _get_stamp_button_rect(self) -> QRectF:
        return QRectF(525.0 + 12.0, 495.0, 175.0, 36.0)

    def _get_close_bottom_button_rect(self) -> QRectF:
        return QRectF(525.0 + 195.0, 495.0, 93.0, 36.0)

    # ── Presets & Physical Setters ────────────────────────────────────────────

    def set_frequency(self, f_hz: float) -> None:
        """Sets frequency f clamped to [10.0, 200.0] Hz."""
        self._frequency_f = max(10.0, min(200.0, float(f_hz)))
        self._emit_updates()

    def set_resistance(self, r_ohm: float) -> None:
        """Sets resistance R clamped to [10.0, 500.0] Ω."""
        self._resistor_R = max(10.0, min(500.0, float(r_ohm)))
        self._emit_updates()

    def set_inductance(self, l_henry: float) -> None:
        """Sets inductance L clamped to [0.05, 1.00] H."""
        self._inductor_L = max(0.05, min(1.00, float(l_henry)))
        self._emit_updates()

    def set_capacitance(self, c_uF: float) -> None:
        """Sets capacitance C clamped to [5.0, 100.0] μF."""
        self._capacitor_C_uF = max(5.0, min(100.0, float(c_uF)))
        self._emit_updates()

    def snap_to_resonance(self) -> None:
        """Snaps frequency f directly to resonant frequency f₀."""
        f0 = self.resonance_frequency_f0
        if 10.0 <= f0 <= 200.0:
            self.set_frequency(f0)
            self.resonance_triggered.emit(f0, self.quality_factor_Q)

    def set_preset(self, preset_name: str) -> None:
        """Switches through authentic AC curriculum presets."""
        self._preset = preset_name
        self._apply_preset(preset_name)
        self.preset_changed.emit(preset_name)
        self.update()

    def _apply_preset(self, preset: str) -> None:
        p_up = preset.upper()
        if p_up == "RESONANCE_50HZ":
            self._frequency_f = 50.0
            self._resistor_R = 50.0
            self._inductor_L = 0.3183
            self._capacitor_C_uF = 31.83
        elif p_up == "INDUCTIVE_CIRCUIT":
            self._frequency_f = 60.0
            self._resistor_R = 60.0
            self._inductor_L = 0.500
            self._capacitor_C_uF = 20.0
        elif p_up == "CAPACITIVE_CIRCUIT":
            self._frequency_f = 40.0
            self._resistor_R = 60.0
            self._inductor_L = 0.150
            self._capacitor_C_uF = 15.0
        elif p_up == "HIGH_Q_RESONANCE":
            self._frequency_f = 50.0
            self._resistor_R = 10.0
            self._inductor_L = 0.3183
            self._capacitor_C_uF = 31.83

        self._emit_updates()

    def _emit_updates(self) -> None:
        self.rlc_updated.emit(self.total_impedance_Z, self.rms_current_I, self.phase_deg, self.power_factor)
        self.update()

    # ── Mouse & Air Gesture Interaction ───────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Close
        if self._get_close_button_rect().contains(pos) or self._get_close_bottom_button_rect().contains(pos):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_button_rect().contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # Resonance snap button
        if self._get_resonance_snap_button_rect().contains(pos):
            self.snap_to_resonance()
            event.accept()
            return

        # Presets
        presets = [
            ("RESONANCE_50HZ", 0, 0),
            ("INDUCTIVE_CIRCUIT", 0, 1),
            ("CAPACITIVE_CIRCUIT", 1, 0),
            ("HIGH_Q_RESONANCE", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(pos):
                self.set_preset(p_id)
                event.accept()
                return

        # Visual toggles
        hud_left = 525.0
        hud_w = 300.0
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        for idx, attr in enumerate(["_show_phasor_diagram", "_show_oscilloscope", "_is_animating"]):
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 432.0, tog_w, 28.0)
            if tog_r.contains(pos):
                setattr(self, attr, not getattr(self, attr))
                self.update()
                event.accept()
                return

        # Sliders
        f_track, f_handle = self._get_f_slider_geometry()
        if f_track.contains(pos) or f_handle.contains(pos):
            self._drag_target = "SLIDER_F"
            frac = (pos.x() - f_track.left()) / f_track.width()
            self.set_frequency(10.0 + frac * (200.0 - 10.0))
            event.accept()
            return

        r_track, r_handle = self._get_r_slider_geometry()
        if r_track.contains(pos) or r_handle.contains(pos):
            self._drag_target = "SLIDER_R"
            frac = (pos.x() - r_track.left()) / r_track.width()
            self.set_resistance(10.0 + frac * (500.0 - 10.0))
            event.accept()
            return

        l_track, l_handle = self._get_l_slider_geometry()
        if l_track.contains(pos) or l_handle.contains(pos):
            self._drag_target = "SLIDER_L"
            frac = (pos.x() - l_track.left()) / l_track.width()
            self.set_inductance(0.05 + frac * (1.00 - 0.05))
            event.accept()
            return

        c_track, c_handle = self._get_c_slider_geometry()
        if c_track.contains(pos) or c_handle.contains(pos):
            self._drag_target = "SLIDER_C"
            frac = (pos.x() - c_track.left()) / c_track.width()
            self.set_capacitance(5.0 + frac * (100.0 - 5.0))
            event.accept()
            return

        # Title bar dragging
        if QRectF(0.0, 0.0, self._width, 40.0).contains(pos):
            self._drag_target = "TITLE"
            self._drag_offset = event.scenePos() - self.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        if self._drag_target == "TITLE":
            self.setPos(event.scenePos() - self._drag_offset)
            event.accept()
            return
        elif self._drag_target == "SLIDER_F":
            f_track, _ = self._get_f_slider_geometry()
            frac = (pos.x() - f_track.left()) / f_track.width()
            self.set_frequency(10.0 + frac * (200.0 - 10.0))
            event.accept()
            return
        elif self._drag_target == "SLIDER_R":
            r_track, _ = self._get_r_slider_geometry()
            frac = (pos.x() - r_track.left()) / r_track.width()
            self.set_resistance(10.0 + frac * (500.0 - 10.0))
            event.accept()
            return
        elif self._drag_target == "SLIDER_L":
            l_track, _ = self._get_l_slider_geometry()
            frac = (pos.x() - l_track.left()) / l_track.width()
            self.set_inductance(0.05 + frac * (1.00 - 0.05))
            event.accept()
            return
        elif self._drag_target == "SLIDER_C":
            c_track, _ = self._get_c_slider_geometry()
            frac = (pos.x() - c_track.left()) / c_track.width()
            self.set_capacitance(5.0 + frac * (100.0 - 5.0))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._drag_target = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old = self._hover_target

        if self._get_close_button_rect().contains(pos):
            self._hover_target = "CLOSE"
        elif self._get_close_bottom_button_rect().contains(pos):
            self._hover_target = "CLOSE_BOTTOM"
        elif self._get_stamp_button_rect().contains(pos):
            self._hover_target = "STAMP"
        elif self._get_resonance_snap_button_rect().contains(pos):
            self._hover_target = "SNAP_RES"
        elif self._get_f_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_F"
        elif self._get_r_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_R"
        elif self._get_l_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_L"
        elif self._get_c_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_C"
        else:
            self._hover_target = None
            presets = [
                ("RESONANCE_50HZ", 0, 0),
                ("INDUCTIVE_CIRCUIT", 0, 1),
                ("CAPACITIVE_CIRCUIT", 1, 0),
                ("HIGH_Q_RESONANCE", 1, 1),
            ]
            for p_id, r, c in presets:
                if self._get_preset_button_rect(r, c).contains(pos):
                    self._hover_target = f"PRESET_{p_id}"
                    break

        if self._hover_target != old:
            self.update()

    # ── Touchless Air Gesture API ─────────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_button_rect().contains(loc) or self._get_close_bottom_button_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Resonance snap
        if self._get_resonance_snap_button_rect().contains(loc):
            self.snap_to_resonance()
            return True

        # Presets
        presets = [
            ("RESONANCE_50HZ", 0, 0),
            ("INDUCTIVE_CIRCUIT", 0, 1),
            ("CAPACITIVE_CIRCUIT", 1, 0),
            ("HIGH_Q_RESONANCE", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(loc):
                self.set_preset(p_id)
                return True

        # Frequency Slider
        f_track, _ = self._get_f_slider_geometry()
        if f_track.contains(loc):
            frac = (loc.x() - f_track.left()) / f_track.width()
            self.set_frequency(10.0 + frac * (200.0 - 10.0))
            return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── Board Integration & Vector Stamping ───────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of RLC circuit & Phasor diagram onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_rlc_circuit_diagram(
            center=(cx, cy + 30.0),
            width=500.0,
            height=320.0,
            voltage_u=self._rms_voltage_U,
            frequency_f=self._frequency_f,
            resistor_r=self._resistor_R,
            inductor_l=self._inductor_L,
            capacitor_c_uf=self._capacitor_C_uF,
            impedance_z=self.total_impedance_Z,
            current_i=self.rms_current_I,
            phase_deg=self.phase_deg,
            preset=self._preset,
        )

        for item in diagram_items:
            board.addItem(item)

        logger.info(f"AC Series RLC circuit diagram stamped onto LessonBoard (f={self._frequency_f:.1f}Hz).")

    def close(self) -> None:
        self._anim_timer.stop()
        self.setVisible(False)
        self.update()

    # ── Public Properties for Tests & State Verification ─────────────────────

    @property
    def frequency_f(self) -> float:
        return self._frequency_f

    @property
    def resistor_R(self) -> float:
        return self._resistor_R

    @property
    def inductor_L(self) -> float:
        return self._inductor_L

    @property
    def capacitor_C_uF(self) -> float:
        return self._capacitor_C_uF

    @property
    def preset(self) -> str:
        return self._preset
