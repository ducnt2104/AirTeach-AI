"""
AirTeach AI - Interactive Thermodynamics Carnot Heat Engine & PV / TS Cycle Simulator
======================================================================================
Provides an authentic thermodynamic simulation of Sadi Carnot's ideal reversible cycle:
  - Working Fluid:
      - Ideal gas: n = 1.0 mol, R = 8.314462618 J/(mol·K)
      - Specific heat ratio: γ = 1.40 (Diatomic gas, e.g. Air, N₂, O₂)
  - The Four Thermodynamic Stages:
      1. Stage 1 → 2: Reversible Isothermal Expansion at Hot Temperature T_H
          Gas absorbs heat Q_H from hot reservoir at constant T_H.
          Q_H = W₁₂ = n * R * T_H * ln(V₂ / V₁) > 0, ΔU₁₂ = 0
          Entropy change: ΔS = Q_H / T_H = n * R * ln(r)
      2. Stage 2 → 3: Reversible Adiabatic Expansion from T_H down to T_C
          Thermally insulated cylinder (Q = 0, isentropic ΔS = 0).
          Poisson relation: T_H * V₂^(γ-1) = T_C * V₃^(γ-1)
          Work done by gas: W₂₃ = n * R * (T_H - T_C) / (γ - 1) = -ΔU₂₃ > 0
      3. Stage 3 → 4: Reversible Isothermal Compression at Cold Temperature T_C
          Gas rejects heat |Q_C| to cold reservoir at constant T_C.
          Q_C = W₃₄ = -n * R * T_C * ln(V₃ / V₄) < 0, ΔU₃₄ = 0
          Entropy change: ΔS₃₄ = -n * R * ln(r) = -ΔS
      4. Stage 4 → 1: Reversible Adiabatic Compression from T_C back to T_H
          Thermally insulated cylinder (Q = 0, isentropic ΔS = 0).
          Poisson relation: T_C * V₄^(γ-1) = T_H * V₁^(γ-1)
          Work done on gas: W₄₁ = n * R * (T_C - T_H) / (γ - 1) = -W₂₃ < 0
  - Net Work Output & Carnot Thermal Efficiency:
      - W_net = Q_H + Q_C = n * R * (T_H - T_C) * ln(r) = ∮ P dV > 0
      - Maximum Carnot Efficiency:
          η = W_net / Q_H = (T_H - T_C) / T_H = 1 - (T_C / T_H)
      - Reverse Cycle (Carnot Refrigerator & Heat Pump):
          COP_ref = T_C / (T_H - T_C) = |Q_C| / W_net
          COP_pump = T_H / (T_H - T_C) = COP_ref + 1
  - Dual Visualizations:
      - P - V Diagram (Pressure vs Volume) with shaded net work area W_net.
      - T - S Diagram (Temperature vs Entropy) with perfect rectangular geometry.
      - Dynamic mechanical cylinder with translating piston head, crank rod, and flywheel.
  - Air gestures (air tap / air pinch drag) & vector stamping to LessonBoard.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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

# ── Thermodynamic Physical Constants ──────────────────────────────────────────
GAS_CONSTANT_R = 8.314462618   # J / (mol · K)
DEFAULT_MOLES_N = 1.0          # mol
GAMMA_DIATOMIC = 1.40          # cp / cv for diatomic gas (Air, N2)

PRESETS: Dict[str, Dict[str, Any]] = {
    "STANDARD_STEAM": {
        "label": "Động cơ Hơi nước Chuẩn",
        "th_k": 600.0,      # 327°C
        "tc_k": 300.0,      # 27°C
        "ratio_r": 2.0,
        "desc": "T_H = 600K, T_C = 300K => Hiệu suất Carnot lý tưởng η = 50.0%.",
    },
    "HIGH_EFFICIENCY": {
        "label": "Tua-bin Nhiệt Siêu Cao",
        "th_k": 900.0,      # 627°C
        "tc_k": 300.0,      # 27°C
        "ratio_r": 2.5,
        "desc": "T_H = 900K, T_C = 300K => Hiệu suất cực đại η = 66.7%.",
    },
    "REVERSE_REFRIGERATOR": {
        "label": "Máy lạnh & Bơm nhiệt Carnot",
        "th_k": 310.0,      # 37°C
        "tc_k": 250.0,      # -23°C
        "ratio_r": 1.8,
        "desc": "Chu trình ngược: T_C = 250K, T_H = 310K => COP làm lạnh = 4.17, COP sưởi = 5.17.",
    },
    "LOW_DELTA_T": {
        "label": "Chênh lệch Nhiệt Nhỏ",
        "th_k": 350.0,      # 77°C
        "tc_k": 300.0,      # 27°C
        "ratio_r": 1.5,
        "desc": "Chênh lệch nhiệt độ nhỏ (ΔT = 50K) => η = 14.3% (Năng lượng nhiệt đại dương OTEC).",
    },
}


class InteractiveCarnotEngineItem(QGraphicsObject):
    """
    Comprehensive Interactive Carnot Heat Engine & PV / TS Cycle Simulator for LessonBoard.
    Visualizes gas expansion/compression, heat exchange Q_H & Q_C, thermodynamic P-V & T-S
    enclosed areas, and real-time mechanical piston animation.
    """

    parameter_changed = Signal()
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (320.0, 160.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])

        self._width = 600.0
        self._height = 430.0

        # Thermodynamic State Variables
        self._th_k: float = 600.0      # Hot reservoir temperature (K)
        self._tc_k: float = 300.0      # Cold reservoir temperature (K)
        self._ratio_r: float = 2.0     # Isothermal volume expansion ratio (V2 / V1)
        self._gamma: float = GAMMA_DIATOMIC
        self._moles_n: float = DEFAULT_MOLES_N
        self._current_preset: str = "STANDARD_STEAM"
        self._diagram_mode: str = "PV_DIAGRAM"  # "PV_DIAGRAM" or "TS_DIAGRAM"

        # Animation State
        self._is_animating: bool = True
        self._cycle_phase: float = 0.0  # Phase in [0.0, 4.0) traversing stages 1, 2, 3, 4
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(25)  # 40 FPS
        self._anim_timer.timeout.connect(self._on_animation_step)
        self._anim_timer.start()

        # Dragging state
        self._dragging_slider: Optional[str] = None

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(35.0)

    # ── Geometry & Bounds ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Thermodynamic Engine Calculations ─────────────────────────────────────

    def get_temperatures(self) -> Tuple[float, float]:
        """Returns (T_H, T_C) in Kelvin with invariant T_H > T_C."""
        return self._th_k, self._tc_k

    def get_carnot_efficiency(self) -> float:
        """η = 1 - (T_C / T_H) in fraction [0.0, 1.0]."""
        if self._th_k <= 0.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self._tc_k / self._th_k)))

    def get_carnot_cop_refrigerator(self) -> float:
        """COP_ref = T_C / (T_H - T_C)."""
        delta_t = self._th_k - self._tc_k
        if delta_t <= 0.01:
            return 999.0
        return self._tc_k / delta_t

    def get_carnot_cop_heat_pump(self) -> float:
        """COP_pump = T_H / (T_H - T_C) = COP_ref + 1."""
        return self.get_carnot_cop_refrigerator() + 1.0

    def get_entropy_change_ds(self) -> float:
        """ΔS = n * R * ln(r) in J/K."""
        return self._moles_n * GAS_CONSTANT_R * math.log(max(1.01, self._ratio_r))

    def get_heat_absorbed_qh(self) -> float:
        """Q_H = n * R * T_H * ln(r) in Joules."""
        return self._th_k * self.get_entropy_change_ds()

    def get_heat_rejected_qc_magnitude(self) -> float:
        """|Q_C| = n * R * T_C * ln(r) in Joules."""
        return self._tc_k * self.get_entropy_change_ds()

    def get_net_work_wnet(self) -> float:
        """W_net = Q_H - |Q_C| = n * R * (T_H - T_C) * ln(r) in Joules."""
        return self.get_heat_absorbed_qh() - self.get_heat_rejected_qc_magnitude()

    def get_state_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculates (V, P) coordinates for the 4 key state points (1, 2, 3, 4).
        Volume in Liters, Pressure in Atmospheres (or scaled units for display).
        Base V1 = 1.0 L.
        """
        v1 = 1.0
        p1 = (self._moles_n * GAS_CONSTANT_R * self._th_k) / (v1 * 1e-3) / 101325.0  # atm

        v2 = self._ratio_r * v1
        p2 = (self._moles_n * GAS_CONSTANT_R * self._th_k) / (v2 * 1e-3) / 101325.0

        # Adiabatic expansion: T_H * V2^(gamma-1) = T_C * V3^(gamma-1)
        power = 1.0 / (self._gamma - 1.0)  # 2.5 for gamma = 1.4
        v3 = v2 * math.pow(self._th_k / max(1.0, self._tc_k), power)
        p3 = (self._moles_n * GAS_CONSTANT_R * self._tc_k) / (v3 * 1e-3) / 101325.0

        # Isothermal compression at T_C: V3 / V4 = r => V4 = V3 / r
        v4 = v3 / self._ratio_r
        p4 = (self._moles_n * GAS_CONSTANT_R * self._tc_k) / (v4 * 1e-3) / 101325.0

        return {
            "1": (v1, p1),
            "2": (v2, p2),
            "3": (v3, p3),
            "4": (v4, p4),
        }

    def get_current_cycle_state(self) -> Tuple[float, float, float, str]:
        """
        Calculates instantaneous (V(t), P(t), T(t), stage_name) along cycle phase [0.0, 4.0).
        """
        states = self.get_state_points()
        v1, p1 = states["1"]
        v2, p2 = states["2"]
        v3, p3 = states["3"]
        v4, p4 = states["4"]

        phase = self._cycle_phase % 4.0

        if phase < 1.0:
            # Stage 1 -> 2: Isothermal expansion at T_H
            frac = phase
            v = v1 + frac * (v2 - v1)
            p = (self._moles_n * GAS_CONSTANT_R * self._th_k) / (v * 1e-3) / 101325.0
            t = self._th_k
            stage = "1 → 2: Giãn đẳng nhiệt ở T_H (Hấp thụ Q_H)"
        elif phase < 2.0:
            # Stage 2 -> 3: Adiabatic expansion T_H -> T_C
            frac = phase - 1.0
            v = v2 + frac * (v3 - v2)
            # T * V^(gamma-1) = const
            t = self._th_k * math.pow(v2 / v, self._gamma - 1.0)
            p = (self._moles_n * GAS_CONSTANT_R * t) / (v * 1e-3) / 101325.0
            stage = "2 → 3: Giãn đoạn nhiệt (Sinh công W > 0, Q = 0)"
        elif phase < 3.0:
            # Stage 3 -> 4: Isothermal compression at T_C
            frac = phase - 2.0
            v = v3 - frac * (v3 - v4)
            p = (self._moles_n * GAS_CONSTANT_R * self._tc_k) / (v * 1e-3) / 101325.0
            t = self._tc_k
            stage = "3 → 4: Nén đẳng nhiệt ở T_C (Tỏa nhiệt Q_C)"
        else:
            # Stage 4 -> 1: Adiabatic compression T_C -> T_H
            frac = phase - 3.0
            v = v4 - frac * (v4 - v1)
            t = self._tc_k * math.pow(v4 / v, self._gamma - 1.0)
            p = (self._moles_n * GAS_CONSTANT_R * t) / (v * 1e-3) / 101325.0
            stage = "4 → 1: Nén đoạn nhiệt (Nhận công, Q = 0)"

        return v, p, t, stage

    # ── Getters & Setters ─────────────────────────────────────────────────────

    @property
    def th_k(self) -> float:
        return self._th_k

    def set_th(self, val: float) -> None:
        new_th = max(200.0, min(1000.0, float(val)))
        if new_th <= self._tc_k + 10.0:
            self._tc_k = max(180.0, new_th - 10.0)
        self._th_k = new_th
        self.parameter_changed.emit()
        self.update()

    @property
    def tc_k(self) -> float:
        return self._tc_k

    def set_tc(self, val: float) -> None:
        new_tc = max(180.0, min(900.0, float(val)))
        if new_tc >= self._th_k - 10.0:
            self._th_k = min(1000.0, new_tc + 10.0)
        self._tc_k = new_tc
        self.parameter_changed.emit()
        self.update()

    def set_temperatures(self, th: float, tc: float) -> None:
        """Sets both reservoir temperatures simultaneously ensuring T_H >= T_C + 10K."""
        new_tc = max(180.0, min(800.0, float(tc)))
        new_th = max(new_tc + 10.0, min(1000.0, float(th)))
        self._tc_k = new_tc
        self._th_k = new_th
        self.parameter_changed.emit()
        self.update()

    @property
    def ratio_r(self) -> float:
        return self._ratio_r

    def set_ratio(self, val: float) -> None:
        clamped = max(1.5, min(3.5, float(val)))
        if abs(clamped - self._ratio_r) > 0.02:
            self._ratio_r = clamped
            self.parameter_changed.emit()
            self.update()

    def set_diagram_mode(self, mode: str) -> None:
        if mode in ("PV_DIAGRAM", "TS_DIAGRAM") and mode != self._diagram_mode:
            self._diagram_mode = mode
            self.update()

    def toggle_animation(self) -> bool:
        self._is_animating = not self._is_animating
        self.update()
        return self._is_animating

    def apply_preset(self, preset_key: str) -> None:
        if preset_key in PRESETS:
            p = PRESETS[preset_key]
            self._current_preset = preset_key
            self._th_k = float(p["th_k"])
            self._tc_k = float(p["tc_k"])
            self._ratio_r = float(p["ratio_r"])
            self.parameter_changed.emit()
            self.update()

    # ── Animation Timer ───────────────────────────────────────────────────────

    def _on_animation_step(self) -> None:
        if self._is_animating:
            self._cycle_phase = (self._cycle_phase + 0.025) % 4.0
            self.update()

    def cleanup(self) -> None:
        """Stop animation timer on cleanup."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

    def close(self) -> None:
        self.cleanup()
        self.setVisible(False)
        self.closed.emit()

    # ── UI Regions & Layout ───────────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 10.0, 22.0, 22.0)

    def _get_stamp_button_rect(self) -> QRectF:
        return QRectF(self._width - 150.0, 10.0, 110.0, 24.0)

    def _get_preset_rects(self) -> List[Tuple[str, QRectF]]:
        keys = list(PRESETS.keys())
        rects = []
        x_start = 14.0
        y_top = 42.0
        w = 138.0
        h = 24.0
        for i, k in enumerate(keys):
            rects.append((k, QRectF(x_start + i * (w + 6.0), y_top, w, h)))
        return rects

    def _get_mode_toggle_rects(self) -> Tuple[QRectF, QRectF]:
        # Tabs for P-V vs T-S Diagram
        x = 315.0
        y = 74.0
        w = 130.0
        h = 22.0
        return (QRectF(x, y, w, h), QRectF(x + w + 6.0, y, w, h))

    def _get_slider_rects(self) -> Dict[str, Tuple[QRectF, QRectF]]:
        """Returns {name: (track_rect, handle_rect)} for T_H, T_C, and Ratio r."""
        # T_H: 320 to 1000 K
        th_track = QRectF(70.0, 370.0, 140.0, 6.0)
        th_frac = (self._th_k - 320.0) / (1000.0 - 320.0)
        th_hx = th_track.left() + th_frac * th_track.width()
        th_handle = QRectF(th_hx - 7.0, th_track.center().y() - 7.0, 14.0, 14.0)

        # T_C: 180 to 350 K
        tc_track = QRectF(70.0, 395.0, 140.0, 6.0)
        tc_frac = (self._tc_k - 180.0) / (350.0 - 180.0)
        tc_hx = tc_track.left() + tc_frac * tc_track.width()
        tc_handle = QRectF(tc_hx - 7.0, tc_track.center().y() - 7.0, 14.0, 14.0)

        # Ratio r: 1.5 to 3.5
        r_track = QRectF(330.0, 385.0, 130.0, 6.0)
        r_frac = (self._ratio_r - 1.5) / (3.5 - 1.5)
        r_hx = r_track.left() + r_frac * r_track.width()
        r_handle = QRectF(r_hx - 7.0, r_track.center().y() - 7.0, 14.0, 14.0)

        return {
            "th": (th_track, th_handle),
            "tc": (tc_track, tc_handle),
            "ratio": (r_track, r_handle),
        }

    def _get_play_pause_button_rect(self) -> QRectF:
        return QRectF(520.0, 376.0, 64.0, 24.0)

    # ── Painting & Rendering ──────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Glassmorphism Container Card
        card_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self._height)
        bg_grad.setColorAt(0.0, QColor("#090d16"))
        bg_grad.setColorAt(1.0, QColor("#050811"))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Presets & Mode Toggle Tabs
        self._paint_selectors(painter)

        # 4. Mechanical Cylinder & Piston Engine (Left Pane)
        self._paint_engine_cylinder(painter)

        # 5. Diagnostic Diagram Pane (Right Pane: P-V or T-S)
        self._paint_diagram_pane(painter)

        # 6. Interactive Sliders & Controls
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 26, "🔥 ĐỘNG CƠ NHIỆT CARNOT & CHU TRÌNH PV - TS (SADI CARNOT 1824)")

        # Stamp Button
        s_rect = self._get_stamp_button_rect()
        painter.setBrush(QBrush(QColor("#ea580c")))
        painter.setPen(QPen(QColor("#fb923c"), 1.2))
        painter.drawRoundedRect(s_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu bảng")

        # Close Button
        c_rect = self._get_close_button_rect()
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(c_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_selectors(self, painter: QPainter) -> None:
        # 1. Preset Buttons
        for key, rect in self._get_preset_rects():
            p_data = PRESETS[key]
            is_active = (key == self._current_preset)
            bg = QColor("#9a3412") if is_active else QColor("#1e293b")
            border = QColor("#f97316") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, p_data["label"])

        # 2. Mode Toggle Tabs
        tab_pv, tab_ts = self._get_mode_toggle_rects()
        for tab_rect, mode_id, label in [
            (tab_pv, "PV_DIAGRAM", "Đồ thị P - V (Áp suất)"),
            (tab_ts, "TS_DIAGRAM", "Đồ thị T - S (Entropy)"),
        ]:
            is_on = (self._diagram_mode == mode_id)
            painter.setBrush(QBrush(QColor("#c2410c") if is_on else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#fb923c") if is_on else QColor("#334155"), 1.2 if is_on else 1.0))
            painter.drawRoundedRect(tab_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_on else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_on else QColor("#94a3b8"))
            painter.drawText(tab_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_engine_cylinder(self, painter: QPainter) -> None:
        """Paints cross-section of thermodynamic cylinder, piston, and heat exchange arrows."""
        x_left = 14.0
        y_top = 104.0
        w = 260.0
        h = 246.0

        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(QRectF(x_left, y_top, w, h), 8.0, 8.0)

        # Instantaneous cycle state
        v_curr, p_curr, t_curr, stage_name = self.get_current_cycle_state()
        states = self.get_state_points()
        v_min = states["1"][0]  # 1.0 L
        v_max = states["3"][0]  # Max volume at state 3
        v_norm = (v_curr - v_min) / max(0.1, v_max - v_min)
        v_norm = max(0.0, min(1.0, v_norm))

        # Cylinder Chamber Geometry
        cyl_x = x_left + 24.0
        cyl_y = y_top + 46.0
        cyl_w = 150.0
        cyl_h = 90.0

        # Piston displacement (min at 30px, max at 130px)
        piston_x = cyl_x + 30.0 + v_norm * 85.0

        # Gas Chamber Background with dynamic temperature color
        # T_H -> Red/Orange, T_C -> Azure/Cyan
        t_frac = (t_curr - self._tc_k) / max(1.0, self._th_k - self._tc_k)
        t_frac = max(0.0, min(1.0, t_frac))
        gas_color = QColor(
            int(56 + t_frac * 199),   # R
            int(189 - t_frac * 139),  # G
            int(248 - t_frac * 228),  # B
            180,
        )

        gas_rect = QRectF(cyl_x + 6.0, cyl_y + 6.0, piston_x - cyl_x - 6.0, cyl_h - 12.0)
        painter.setBrush(QBrush(gas_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(gas_rect)

        # Cylinder Walls (U-shape: top, left, bottom)
        wall_pen = QPen(QColor("#94a3b8"), 4.0)
        painter.setPen(wall_pen)
        painter.drawLine(int(cyl_x + cyl_w), int(cyl_y), int(cyl_x), int(cyl_y))
        painter.drawLine(int(cyl_x), int(cyl_y), int(cyl_x), int(cyl_y + cyl_h))
        painter.drawLine(int(cyl_x), int(cyl_y + cyl_h), int(cyl_x + cyl_w), int(cyl_y + cyl_h))

        # Piston Head
        p_head_rect = QRectF(piston_x, cyl_y + 4.0, 14.0, cyl_h - 8.0)
        painter.setBrush(QBrush(QColor("#cbd5e1")))
        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.drawRoundedRect(p_head_rect, 2.0, 2.0)

        # Connecting Rod to Crank Flywheel
        flywheel_cx = cyl_x + cyl_w + 35.0
        flywheel_cy = cyl_y + cyl_h / 2.0
        flywheel_r = 22.0

        crank_angle = (self._cycle_phase / 4.0) * 2.0 * math.pi
        crank_x = flywheel_cx + flywheel_r * math.cos(crank_angle)
        crank_y = flywheel_cy + flywheel_r * math.sin(crank_angle)

        # Connecting Rod Line
        painter.setPen(QPen(QColor("#94a3b8"), 3.0))
        painter.drawLine(int(piston_x + 14.0), int(cyl_y + cyl_h / 2.0), int(crank_x), int(crank_y))

        # Flywheel Circle
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#64748b"), 2.0))
        painter.drawEllipse(QPointF(flywheel_cx, flywheel_cy), flywheel_r, flywheel_r)

        # Crank Pin
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(crank_x, crank_y), 4.0, 4.0)

        # Heat Flux Indicators (Q_H in stage 1, Q_C in stage 3, Insulated in stages 2 & 4)
        phase = self._cycle_phase % 4.0
        heat_banner_y = cyl_y + cyl_h + 16.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        if phase < 1.0:
            # Stage 1: Flame Heat Inflow
            painter.setPen(QColor("#f97316"))
            painter.drawText(int(cyl_x), int(heat_banner_y), "🔥 Hấp thụ nhiệt Q_H từ nguồn nóng T_H")
        elif phase < 2.0:
            # Stage 2: Adiabatic expansion
            painter.setPen(QColor("#a855f7"))
            painter.drawText(int(cyl_x), int(heat_banner_y), "🛡️ Giãn đoạn nhiệt (Cách nhiệt hoàn hảo Q = 0)")
        elif phase < 3.0:
            # Stage 3: Heat Outflow
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(int(cyl_x), int(heat_banner_y), "❄️ Nhả nhiệt |Q_C| cho nguồn lạnh T_C")
        else:
            # Stage 4: Adiabatic compression
            painter.setPen(QColor("#a855f7"))
            painter.drawText(int(cyl_x), int(heat_banner_y), "🛡️ Nén đoạn nhiệt (Cách nhiệt hoàn hảo Q = 0)")

        # State Telemetry Badge (Bottom of Left Pane)
        stat_rect = QRectF(x_left + 10.0, y_top + h - 68.0, w - 20.0, 58.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(stat_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            f"• Trạng thái tức thời: {stage_name}",
            f"• Thể tích V = {v_curr:.2f} L  |  Áp suất P = {p_curr:.1f} atm",
            f"• Nhiệt độ T = {t_curr:.1f} K ({t_curr - 273.15:+.1f}°C)",
        ]
        ty = stat_rect.top() + 16.0
        for ln in lines:
            painter.drawText(int(stat_rect.left() + 8.0), int(ty), ln)
            ty += 16.0

    def _paint_diagram_pane(self, painter: QPainter) -> None:
        """Paints either P - V Diagram or T - S Diagram."""
        x_left = 284.0
        y_top = 104.0
        w = 302.0
        h = 246.0

        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(QRectF(x_left, y_top, w, h), 8.0, 8.0)

        if self._diagram_mode == "PV_DIAGRAM":
            self._paint_pv_diagram(painter, x_left, y_top, w, h)
        else:
            self._paint_ts_diagram(painter, x_left, y_top, w, h)

    def _paint_pv_diagram(self, painter: QPainter, x_left: float, y_top: float, w: float, h: float) -> None:
        """Plots authentic P - V curves with enclosed Net Work area."""
        plot_x = x_left + 36.0
        plot_y = y_top + 16.0
        plot_w = w - 50.0
        plot_h = h - 70.0
        bottom_y = plot_y + plot_h

        # Axes
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawLine(int(plot_x), int(bottom_y), int(plot_x + plot_w), int(bottom_y))
        painter.drawLine(int(plot_x), int(plot_y), int(plot_x), int(bottom_y))

        # Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(plot_x + plot_w - 30), int(bottom_y + 14), "V (L)")
        painter.drawText(int(plot_x - 30), int(plot_y + 10), "P (atm)")

        states = self.get_state_points()
        v_min = states["1"][0]
        v_max = states["3"][0]
        p_max = states["1"][1]

        def to_plot(v: float, p: float) -> QPointF:
            px = plot_x + ((v - v_min) / max(0.1, v_max - v_min)) * (plot_w - 15.0)
            py = bottom_y - (p / max(1.0, p_max)) * (plot_h - 15.0)
            return QPointF(px, py)

        # 1. Closed Path of Cycle (1 -> 2 -> 3 -> 4 -> 1)
        cycle_path = QPainterPath()
        # Stage 1 -> 2 (Isothermal at T_H)
        v1, v2 = states["1"][0], states["2"][0]
        p_start = to_plot(v1, states["1"][1])
        cycle_path.moveTo(p_start)
        for s in range(1, 25):
            vs = v1 + (s / 24.0) * (v2 - v1)
            ps = (self._moles_n * GAS_CONSTANT_R * self._th_k) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_plot(vs, ps))

        # Stage 2 -> 3 (Adiabatic expansion)
        v3 = states["3"][0]
        for s in range(1, 25):
            vs = v2 + (s / 24.0) * (v3 - v2)
            ts = self._th_k * math.pow(v2 / vs, self._gamma - 1.0)
            ps = (self._moles_n * GAS_CONSTANT_R * ts) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_plot(vs, ps))

        # Stage 3 -> 4 (Isothermal compression at T_C)
        v4 = states["4"][0]
        for s in range(1, 25):
            vs = v3 - (s / 24.0) * (v3 - v4)
            ps = (self._moles_n * GAS_CONSTANT_R * self._tc_k) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_plot(vs, ps))

        # Stage 4 -> 1 (Adiabatic compression)
        for s in range(1, 25):
            vs = v4 - (s / 24.0) * (v4 - v1)
            ts = self._tc_k * math.pow(v4 / vs, self._gamma - 1.0)
            ps = (self._moles_n * GAS_CONSTANT_R * ts) / (vs * 1e-3) / 101325.0
            cycle_path.lineTo(to_plot(vs, ps))

        cycle_path.closeSubpath()

        # Enclosed Net Work Fill (Translucent Orange)
        work_grad = QLinearGradient(plot_x, plot_y, plot_x + plot_w, plot_y + plot_h)
        work_grad.setColorAt(0.0, QColor(249, 115, 22, 120))
        work_grad.setColorAt(1.0, QColor(234, 88, 12, 60))
        painter.setBrush(QBrush(work_grad))
        painter.setPen(QPen(QColor("#f97316"), 2.0))
        painter.drawPath(cycle_path)

        # Label points 1, 2, 3, 4
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        for k, lbl in [("1", "1 (T_H)"), ("2", "2 (T_H)"), ("3", "3 (T_C)"), ("4", "4 (T_C)")]:
            pt = to_plot(states[k][0], states[k][1])
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(pt, 3.5, 3.5)
            painter.drawText(int(pt.x() + 5), int(pt.y() - 5), lbl)

        # Current Operating Marker
        v_curr, p_curr, _, _ = self.get_current_cycle_state()
        curr_pt = to_plot(v_curr, p_curr)
        painter.setBrush(QBrush(QColor("#22c55e")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(curr_pt, 5.5, 5.5)

        # Summary Metrics below graph
        stat_y = y_top + h - 38.0
        eta = self.get_carnot_efficiency()
        wnet_kj = self.get_net_work_wnet() / 1000.0
        qh_kj = self.get_heat_absorbed_qh() / 1000.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            f"• Hiệu suất cực đại: η = 1 - T_C/T_H = {eta * 100.0:.1f}%",
            f"• Công sinh ra: W_net = {wnet_kj:.2f} kJ  |  Nhiệt hấp thụ: Q_H = {qh_kj:.2f} kJ",
        ]
        painter.drawText(int(x_left + 10), int(stat_y), lines[0])
        painter.drawText(int(x_left + 10), int(stat_y + 16), lines[1])

    def _paint_ts_diagram(self, painter: QPainter, x_left: float, y_top: float, w: float, h: float) -> None:
        """Plots authentic T - S Diagram (perfect Carnot rectangle)."""
        plot_x = x_left + 36.0
        plot_y = y_top + 16.0
        plot_w = w - 50.0
        plot_h = h - 70.0
        bottom_y = plot_y + plot_h

        # Axes: S on x, T on y
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawLine(int(plot_x), int(bottom_y), int(plot_x + plot_w), int(bottom_y))
        painter.drawLine(int(plot_x), int(plot_y), int(plot_x), int(bottom_y))

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(plot_x + plot_w - 30), int(bottom_y + 14), "S (J/K)")
        painter.drawText(int(plot_x - 30), int(plot_y + 10), "T (K)")

        # Temperature bounds
        t_max_axis = 1000.0
        y_th = bottom_y - (self._th_k / t_max_axis) * (plot_h - 15.0)
        y_tc = bottom_y - (self._tc_k / t_max_axis) * (plot_h - 15.0)

        # Entropy coordinates: S1 to S2 = S1 + ΔS
        ds = self.get_entropy_change_ds()
        x_s1 = plot_x + 35.0
        x_s2 = x_s1 + (ds / 20.0) * (plot_w - 60.0)

        # Carnot Rectangle in T-S plane: Area = (T_H - T_C) * ΔS = W_net!
        rect_path = QPainterPath()
        rect_path.moveTo(x_s1, y_th)
        rect_path.lineTo(x_s2, y_th)
        rect_path.lineTo(x_s2, y_tc)
        rect_path.lineTo(x_s1, y_tc)
        rect_path.closeSubpath()

        painter.setBrush(QBrush(QColor(168, 85, 247, 100)))
        painter.setPen(QPen(QColor("#a855f7"), 2.0))
        painter.drawPath(rect_path)

        # Stage Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(int(x_s1 - 10), int(y_th - 4), "1")
        painter.drawText(int(x_s2 + 6), int(y_th - 4), "2")
        painter.drawText(int(x_s2 + 6), int(y_tc + 12), "3")
        painter.drawText(int(x_s1 - 10), int(y_tc + 12), "4")

        # Temperature level dashed lines
        painter.setPen(QPen(QColor("#ef4444"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(plot_x), int(y_th), int(x_s1), int(y_th))
        painter.drawText(int(plot_x - 28), int(y_th + 4), f"{int(self._th_k)}K")

        painter.setPen(QPen(QColor("#38bdf8"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(plot_x), int(y_tc), int(x_s1), int(y_tc))
        painter.drawText(int(plot_x - 28), int(y_tc + 4), f"{int(self._tc_k)}K")

        # Summary Metrics below graph
        stat_y = y_top + h - 38.0
        cop_ref = self.get_carnot_cop_refrigerator()
        cop_hp = self.get_carnot_cop_heat_pump()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            f"• Độ biến thiên Entropy: ΔS = {ds:.2f} J/K (Diện tích hình chữ nhật = W_net)",
            f"• Chu trình lạnh ngược: COP_ref = {cop_ref:.2f}  |  Bơm nhiệt COP_hp = {cop_hp:.2f}",
        ]
        painter.drawText(int(x_left + 10), int(stat_y), lines[0])
        painter.drawText(int(x_left + 10), int(stat_y + 16), lines[1])

    def _paint_controls(self, painter: QPainter) -> None:
        """Paints interactive sliders for T_H, T_C, Volume Ratio, and Play/Pause button."""
        slider_dict = self._get_slider_rects()

        # 1. T_H Slider
        th_track, th_handle = slider_dict["th"]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f97316"))
        painter.drawText(16, int(th_track.center().y() + 4), "Nguồn T_H:")

        painter.setBrush(QBrush(QColor("#ea580c")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(th_track, 3.0, 3.0)

        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#ea580c"), 1.8))
        painter.drawEllipse(th_handle)

        painter.setPen(QColor("#f8fafc"))
        painter.drawText(int(th_track.right() + 8), int(th_track.center().y() + 4), f"{int(self._th_k)} K")

        # 2. T_C Slider
        tc_track, tc_handle = slider_dict["tc"]
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(16, int(tc_track.center().y() + 4), "Nguồn T_C:")

        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.drawRoundedRect(tc_track, 3.0, 3.0)

        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#0284c7"), 1.8))
        painter.drawEllipse(tc_handle)

        painter.setPen(QColor("#f8fafc"))
        painter.drawText(int(tc_track.right() + 8), int(tc_track.center().y() + 4), f"{int(self._tc_k)} K")

        # 3. Ratio r Slider
        r_track, r_handle = slider_dict["ratio"]
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(r_track.left() - 85), int(r_track.center().y() + 4), "Tỷ số nén r:")

        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(r_track, 3.0, 3.0)

        active_rect = QRectF(r_track.left(), r_track.top(), r_handle.center().x() - r_track.left(), r_track.height())
        painter.setBrush(QBrush(QColor("#a855f7")))
        painter.drawRoundedRect(active_rect, 3.0, 3.0)

        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#a855f7"), 1.8))
        painter.drawEllipse(r_handle)

        painter.setPen(QColor("#f8fafc"))
        painter.drawText(int(r_track.right() + 8), int(r_track.center().y() + 4), f"{self._ratio_r:.1f}x")

        # 4. Play/Pause Button
        pp_rect = self._get_play_pause_button_rect()
        painter.setBrush(QBrush(QColor("#047857") if self._is_animating else QColor("#b45309")))
        painter.setPen(QPen(QColor("#10b981") if self._is_animating else QColor("#f59e0b"), 1.2))
        painter.drawRoundedRect(pp_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        lbl = "⏸ Tạm dừng" if self._is_animating else "▶ Tiếp tục"
        painter.drawText(pp_rect, Qt.AlignmentFlag.AlignCenter, lbl)

    # ── Mouse & Touchless Air Gesture Handlers ────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()

        # Close
        if self._get_close_button_rect().contains(loc):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            event.accept()
            return

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.apply_preset(key)
                event.accept()
                return

        # Play/Pause
        if self._get_play_pause_button_rect().contains(loc):
            self.toggle_animation()
            event.accept()
            return

        # Mode toggles
        tab_pv, tab_ts = self._get_mode_toggle_rects()
        if tab_pv.contains(loc):
            self.set_diagram_mode("PV_DIAGRAM")
            event.accept()
            return
        elif tab_ts.contains(loc):
            self.set_diagram_mode("TS_DIAGRAM")
            event.accept()
            return

        # Sliders
        sliders = self._get_slider_rects()
        for name, (track, handle) in sliders.items():
            hit_rect = handle.adjusted(-6, -6, 6, 6)
            track_hit = track.adjusted(-6, -6, 6, 6)
            if hit_rect.contains(loc) or track_hit.contains(loc):
                self._dragging_slider = name
                self._update_slider_from_pos(name, loc.x(), track)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_slider:
            sliders = self._get_slider_rects()
            if self._dragging_slider in sliders:
                track, _ = sliders[self._dragging_slider]
                self._update_slider_from_pos(self._dragging_slider, event.pos().x(), track)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, name: str, x: float, track: QRectF) -> None:
        frac = (x - track.left()) / track.width()
        frac = max(0.0, min(1.0, frac))

        if name == "th":
            th = 320.0 + frac * (1000.0 - 320.0)
            self.set_th(th)
        elif name == "tc":
            tc = 180.0 + frac * (350.0 - 180.0)
            self.set_tc(tc)
        elif name == "ratio":
            ratio = 1.5 + frac * (3.5 - 1.5)
            self.set_ratio(ratio)

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_button_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.apply_preset(key)
                return True

        # Play/Pause
        if self._get_play_pause_button_rect().contains(loc):
            self.toggle_animation()
            return True

        # Mode toggles
        tab_pv, tab_ts = self._get_mode_toggle_rects()
        if tab_pv.contains(loc):
            self.set_diagram_mode("PV_DIAGRAM")
            return True
        elif tab_ts.contains(loc):
            self.set_diagram_mode("TS_DIAGRAM")
            return True

        # Sliders
        sliders = self._get_slider_rects()
        for name, (track, handle) in sliders.items():
            hit_rect = handle.adjusted(-12, -12, 12, 12)
            track_hit = track.adjusted(-6, -6, 6, 6)
            if hit_rect.contains(loc) or track_hit.contains(loc):
                self._update_slider_from_pos(name, loc.x(), track)
                return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── LessonBoard Stamping ──────────────────────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of Carnot engine and P-V cycle onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_carnot_engine_diagram(
            center=(cx, cy + 30.0),
            width=520.0,
            height=320.0,
            th_k=self._th_k,
            tc_k=self._tc_k,
            ratio_r=self._ratio_r,
            preset=self._current_preset,
        )
        for it in diagram_items:
            board.addItem(it)

        if hasattr(board, "commit_state_action"):
            board.commit_state_action("Stamp Carnot Heat Engine Diagram")
        logger.info(f"Carnot Engine diagram stamped to LessonBoard at ({cx:.1f}, {cy:.1f}).")
