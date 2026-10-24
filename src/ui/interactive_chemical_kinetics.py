"""
AirTeach AI - Interactive Chemical Kinetics, Arrhenius Activation Energy & Dynamic Equilibrium Simulator
========================================================================================================
Provides an authentic physical chemistry simulation of chemical kinetics and chemical equilibrium:
  - Law of Mass Action & Reversible Elementary Reaction:
      A + B  <===>  C + D
      Forward rate: vf = kf * [A] * [B]
      Reverse rate: vr = kr * [C] * [D]
      Net reaction rate: d[C]/dt = vf - vr
  - Dynamic Chemical Equilibrium (Cân bằng hóa học động):
      vf = vr => Kc = kf / kr = ([C]_eq * [D]_eq) / ([A]_eq * [B]_eq)
      Reaction quotient: Qc = ([C] * [D]) / ([A] * [B])
        - Qc < Kc: Shifts Forward (Chiều thuận ->)
        - Qc == Kc: Dynamic Equilibrium (Trạng thái cân bằng <=>)
        - Qc > Kc: Shifts Reverse (Chiều nghịch <-)
  - Arrhenius Equation & Temperature Dependence:
      k(T) = A * exp(-Ea / (R * T))
      R = 8.314 J/(mol·K)
      Ea_r = Ea_f - ΔH
      Reaction enthalpy: ΔH = Ea_f - Ea_r
        - Exothermic (ΔH < 0, Tỏa nhiệt): Increasing T shifts reverse (<-)
        - Endothermic (ΔH > 0, Thu nhiệt): Increasing T shifts forward (->)
  - Catalysis (Chất xúc tác):
      Lowers activation barrier Ea_f and Ea_r equally by ΔE_cat.
      Accelerates both kf and kr equally without altering equilibrium constant Kc!
  - Le Chatelier's Principle Perturbations (Chuyển dịch cân bằng Le Chatelier):
      Spike reactant [A], spike product [C], increase/decrease temperature T.
  - Potential Energy Profile Curve (Sơ đồ năng lượng phản ứng):
      Visualizes reactant energy, transition state barrier hill, product energy,
      reaction enthalpy ΔH, and lowered catalyzed activation path.
  - Touchless air gestures and vector diagram stamping via ChemistryTools.create_reaction_kinetics_diagram.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
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
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.ui.stem_tools import ChemistryTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

GAS_CONSTANT_R = 8.314  # J / (mol·K)

# Kinetics Reaction Presets: (name_vi, reaction_eq, delta_h_kj, ea_f_kj, a_factor)
KINETICS_PRESETS: dict[str, tuple[str, str, float, float, float]] = {
    "HABER_BOSCH": (
        "Tổng hợp Amoniac (Haber-Bosch)",
        "N₂ + 3H₂ ⇄ 2NH₃",
        -92.0,  # Exothermic
        95.0,   # Activation energy (kJ/mol)
        1.2e12,
    ),
    "HYDROGEN_IODIDE": (
        "Cân bằng Hiđro Iođua (HI)",
        "H₂ + I₂ ⇄ 2HI",
        -9.5,   # Slightly exothermic
        165.0,
        8.5e11,
    ),
    "NITROGEN_DIOXIDE": (
        "Dime hóa Khí NO₂",
        "2NO₂ (nâu đỏ) ⇄ N₂O₄ (không màu)",
        -57.2,  # Exothermic
        53.0,
        2.5e10,
    ),
    "GENERIC": (
        "Phản ứng Thuận nghịch Tổng quát",
        "A + B ⇄ C + D",
        -40.0,
        75.0,
        1.0e11,
    ),
}


class InteractiveChemicalKineticsItem(QGraphicsObject):
    """
    Interactive Chemical Kinetics, Arrhenius Activation Energy & Dynamic Equilibrium Simulator item.
    """

    kinetics_updated = Signal(float, float, float, float)  # (conc_a, conc_c, vf, vr)
    preset_changed = Signal(str)                           # ("HABER_BOSCH", "HYDROGEN_IODIDE", etc.)
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (450.0, 240.0),
        width: float = 780.0,
        height: float = 500.0,
        preset: str = "HABER_BOSCH",
        temperature_k: float = 350.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Active Preset
        self._preset = preset.upper() if preset.upper() in KINETICS_PRESETS else "HABER_BOSCH"
        spec = KINETICS_PRESETS[self._preset]
        self._delta_h = spec[2]           # kJ/mol
        self._ea_f = spec[3]              # kJ/mol
        self._a_factor = spec[4]

        # Physical Conditions
        self._temp_k = max(250.0, min(550.0, float(temperature_k)))  # Kelvin
        self._has_catalyst = False        # Catalyst active toggle
        self._catalyst_delta_ea = 35.0    # Ea reduction by catalyst (kJ/mol)

        # Chemical Concentrations [M]
        self._conc_a = 0.90               # Reactant A
        self._conc_b = 0.90               # Reactant B
        self._conc_c = 0.10               # Product C
        self._conc_d = 0.10               # Product D

        # History buffer for plotting [A](t) and [C](t)
        # Keeps last 140 state tuples: (t_s, conc_a, conc_c, vf, vr)
        self._sim_time = 0.0
        self._history: collections.deque[tuple[float, float, float, float, float]] = collections.deque(maxlen=140)
        self._record_history()

        # Animation Timer (35 FPS, dt = 0.030s)
        self._dt = 0.030
        self._is_running = True
        self._timer = QTimer()
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Layout Geometry Constants
        self._prof_x = 24.0
        self._prof_y = 52.0
        self._prof_w = 260.0
        self._prof_h = 320.0

        self._plot_x = 300.0
        self._plot_y = 52.0
        self._plot_w = self._width - self._plot_x - 24.0  # 456.0 px
        self._plot_h = 195.0

        self._hud_x = 300.0
        self._hud_y = 258.0
        self._hud_w = self._plot_w
        self._hud_h = 114.0

        # Window Controls
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Control Deck Rectangles (y = 388 to 480)
        # Row 1: Reaction Presets (y = 388, h = 26)
        self._rect_btn_haber = QRectF(24.0, 388.0, 175.0, 26.0)
        self._rect_btn_hi = QRectF(205.0, 388.0, 175.0, 26.0)
        self._rect_btn_no2 = QRectF(386.0, 388.0, 185.0, 26.0)
        self._rect_btn_generic = QRectF(577.0, 388.0, 178.0, 26.0)

        # Row 2: Perturbations & Actions (y = 422, h = 26)
        self._rect_btn_add_react = QRectF(24.0, 422.0, 120.0, 26.0)
        self._rect_btn_add_prod = QRectF(150.0, 422.0, 120.0, 26.0)
        self._rect_btn_catalyst = QRectF(276.0, 422.0, 125.0, 26.0)
        self._rect_btn_temp_dec = QRectF(407.0, 422.0, 30.0, 26.0)
        self._rect_btn_temp_inc = QRectF(497.0, 422.0, 30.0, 26.0)
        self._rect_btn_play_pause = QRectF(535.0, 422.0, 68.0, 26.0)
        self._rect_btn_reset = QRectF(609.0, 422.0, 68.0, 26.0)
        self._rect_btn_stamp = QRectF(683.0, 422.0, 72.0, 26.0)

    # ─── Public Properties & Chemical Solvers ────────────────────────────────

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def temperature_k(self) -> float:
        return self._temp_k

    @property
    def has_catalyst(self) -> bool:
        return self._has_catalyst

    @property
    def conc_a(self) -> float:
        return self._conc_a

    @property
    def conc_c(self) -> float:
        return self._conc_c

    @property
    def effective_ea_f(self) -> float:
        """Forward activation energy in kJ/mol (reduced if catalyst is active)."""
        base = self._ea_f
        if self._has_catalyst:
            return max(15.0, base - self._catalyst_delta_ea)
        return base

    @property
    def effective_ea_r(self) -> float:
        """Reverse activation energy in kJ/mol: Ea_r = Ea_f - ΔH."""
        return max(10.0, self.effective_ea_f - self._delta_h)

    @property
    def rate_constant_kf(self) -> float:
        """Forward rate constant kf via Arrhenius equation."""
        # Scale to manageable visual range: k ~ 0.05 to 5.0
        ea_j = self.effective_ea_f * 1000.0
        exponent = -ea_j / (GAS_CONSTANT_R * self._temp_k)
        # Clamped exponent to prevent overflow/underflow
        clamped_exp = max(-25.0, min(0.0, exponent * 0.15 + 2.5))
        return math.exp(clamped_exp) * 0.85

    @property
    def rate_constant_kr(self) -> float:
        """Reverse rate constant kr via Arrhenius equation."""
        ea_j = self.effective_ea_r * 1000.0
        exponent = -ea_j / (GAS_CONSTANT_R * self._temp_k)
        clamped_exp = max(-25.0, min(0.0, exponent * 0.15 + 2.5))
        return math.exp(clamped_exp) * 0.85

    @property
    def forward_rate_vf(self) -> float:
        """Forward reaction rate vf = kf * [A] * [B]."""
        return self.rate_constant_kf * self._conc_a * self._conc_b

    @property
    def reverse_rate_vr(self) -> float:
        """Reverse reaction rate vr = kr * [C] * [D]."""
        return self.rate_constant_kr * self._conc_c * self._conc_d

    @property
    def equilibrium_constant_kc(self) -> float:
        """Equilibrium constant Kc = kf / kr."""
        kr = max(1e-5, self.rate_constant_kr)
        return self.rate_constant_kf / kr

    @property
    def reaction_quotient_qc(self) -> float:
        """Reaction quotient Qc = ([C] * [D]) / ([A] * [B])."""
        denom = max(1e-5, self._conc_a * self._conc_b)
        return (self._conc_c * self._conc_d) / denom

    @property
    def le_chatelier_shift(self) -> str:
        """Determines active Le Chatelier shift direction."""
        qc = self.reaction_quotient_qc
        kc = self.equilibrium_constant_kc
        if abs(qc - kc) / max(0.01, kc) < 0.05:
            return "Cân bằng động (vf = vr)"
        elif qc < kc:
            return "Chuyển dịch Thuận (->)"
        else:
            return "Chuyển dịch Nghịch (<-)"

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ─── Controls & Perturbations ────────────────────────────────────────────

    def set_preset(self, preset_name: str) -> None:
        """Switches active chemical reaction preset and resets concentrations."""
        name_upper = preset_name.upper()
        if name_upper in KINETICS_PRESETS and name_upper != self._preset:
            self._preset = name_upper
            spec = KINETICS_PRESETS[self._preset]
            self._delta_h = spec[2]
            self._ea_f = spec[3]
            self._a_factor = spec[4]
            self.reset_simulation()
            self.preset_changed.emit(self._preset)
            self.update()

    def set_temperature(self, temp_k: float) -> None:
        """Sets temperature in Kelvin [250K to 550K]."""
        self._temp_k = max(250.0, min(550.0, float(temp_k)))
        self.update()

    def toggle_catalyst(self) -> None:
        """Toggles catalyst presence, lowering Ea and accelerating kinetics."""
        self._has_catalyst = not self._has_catalyst
        self.update()

    def add_reactants(self, amount: float = 0.35) -> None:
        """Le Chatelier perturbation: injects additional reactants A and B."""
        self._conc_a = min(1.80, self._conc_a + amount)
        self._conc_b = min(1.80, self._conc_b + amount)
        self._record_history()
        self.update()

    def add_products(self, amount: float = 0.35) -> None:
        """Le Chatelier perturbation: injects additional products C and D."""
        self._conc_c = min(1.80, self._conc_c + amount)
        self._conc_d = min(1.80, self._conc_d + amount)
        self._record_history()
        self.update()

    def toggle_play_pause(self) -> None:
        """Toggles simulation running state."""
        self._is_running = not self._is_running
        self.update()

    def reset_simulation(self) -> None:
        """Resets concentrations to initial reactants and clears history."""
        self._conc_a = 0.90
        self._conc_b = 0.90
        self._conc_c = 0.10
        self._conc_d = 0.10
        self._sim_time = 0.0
        self._history.clear()
        self._record_history()
        self.kinetics_updated.emit(self._conc_a, self._conc_c, self.forward_rate_vf, self.reverse_rate_vr)
        self.update()

    # ─── Numerical Kinetics ODE Integration ──────────────────────────────────

    def _kinetics_step(self, dt: float) -> None:
        """
        Integrates chemical rate equations:
            d[C]/dt = vf - vr = kf*[A]*[B] - kr*[C]*[D]
        """
        vf = self.forward_rate_vf
        vr = self.reverse_rate_vr
        net_rate = vf - vr

        # Rate of concentration change
        delta = net_rate * dt

        # Update concentrations with lower bound protection
        self._conc_a = max(0.02, min(2.0, self._conc_a - delta))
        self._conc_b = max(0.02, min(2.0, self._conc_b - delta))
        self._conc_c = max(0.02, min(2.0, self._conc_c + delta))
        self._conc_d = max(0.02, min(2.0, self._conc_d + delta))

        self._sim_time += dt

    def _record_history(self) -> None:
        """Records current state to history buffer."""
        self._history.append((
            self._sim_time,
            self._conc_a,
            self._conc_c,
            self.forward_rate_vf,
            self.reverse_rate_vr,
        ))

    def _on_sim_tick(self) -> None:
        """Simulation tick callback."""
        if self._is_running:
            self._kinetics_step(self._dt * 1.5)
            self._record_history()
            self.kinetics_updated.emit(
                self._conc_a, self._conc_c,
                self.forward_rate_vf, self.reverse_rate_vr
            )
        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps reaction kinetics energy profile and concentration curves onto LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        items = ChemistryTools.create_reaction_kinetics_diagram(
            center=(cx, cy),
            width=440.0,
            height=260.0,
            reaction_name=self._preset,
            temperature_k=self._temp_k,
            has_catalyst=self._has_catalyst,
            ea_kj=self.effective_ea_f,
            delta_h_kj=self._delta_h,
            conc_reactants=self._conc_a,
            conc_products=self._conc_c,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None and hasattr(scene, "add_item_with_undo"):
                target_board = scene  # type: ignore

        if target_board is not None and hasattr(target_board, "add_item_with_undo"):
            for item in items:
                target_board.add_item_with_undo(item)  # type: ignore
        elif self.scene() is not None:
            for item in items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        logger.info(
            "Stamped Chemical Kinetics diagram to LessonBoard (%s, T=%.0fK, Cat=%s)",
            self._preset, self._temp_k, self._has_catalyst
        )
        return items

    # ─── Air Gesture & Interaction Handlers ──────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """
        Handles touchless air pinch gesture.
        If pinching the energy profile or plot, modulates temperature.
        """
        local_pos = self.mapFromScene(scene_pos)

        # Check thermometer area on profile box (left side)
        if self._prof_x <= local_pos.x() <= self._prof_x + self._prof_w and self._prof_y <= local_pos.y() <= self._prof_y + self._prof_h:
            # Map vertical position to temperature: top is 550K, bottom is 250K
            ratio = 1.0 - (local_pos.y() - self._prof_y) / self._prof_h
            new_temp = 250.0 + ratio * 300.0
            self.set_temperature(new_temp)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Handles touchless air tap gesture on buttons."""
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_haber.contains(local_pos):
            self.set_preset("HABER_BOSCH")
            return True
        elif self._rect_btn_hi.contains(local_pos):
            self.set_preset("HYDROGEN_IODIDE")
            return True
        elif self._rect_btn_no2.contains(local_pos):
            self.set_preset("NITROGEN_DIOXIDE")
            return True
        elif self._rect_btn_generic.contains(local_pos):
            self.set_preset("GENERIC")
            return True
        elif self._rect_btn_add_react.contains(local_pos):
            self.add_reactants(0.35)
            return True
        elif self._rect_btn_add_prod.contains(local_pos):
            self.add_products(0.35)
            return True
        elif self._rect_btn_catalyst.contains(local_pos):
            self.toggle_catalyst()
            return True
        elif self._rect_btn_temp_dec.contains(local_pos):
            self.set_temperature(self._temp_k - 25.0)
            return True
        elif self._rect_btn_temp_inc.contains(local_pos):
            self.set_temperature(self._temp_k + 25.0)
            return True
        elif self._rect_btn_play_pause.contains(local_pos):
            self.toggle_play_pause()
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True

        return False

    def close(self) -> None:
        """Stops timer and removes item from scene."""
        self._timer.stop()
        scene = self.scene()
        if scene:
            scene.removeItem(self)

    # ─── Mouse Event Handlers ────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        local_pos = event.pos()

        # Check Close Button
        if self._rect_btn_close.contains(local_pos):
            self.close()
            event.accept()
            return

        # Check Preset Buttons
        if self._rect_btn_haber.contains(local_pos):
            self.set_preset("HABER_BOSCH")
            event.accept()
            return
        elif self._rect_btn_hi.contains(local_pos):
            self.set_preset("HYDROGEN_IODIDE")
            event.accept()
            return
        elif self._rect_btn_no2.contains(local_pos):
            self.set_preset("NITROGEN_DIOXIDE")
            event.accept()
            return
        elif self._rect_btn_generic.contains(local_pos):
            self.set_preset("GENERIC")
            event.accept()
            return

        # Check Perturbation Buttons
        if self._rect_btn_add_react.contains(local_pos):
            self.add_reactants(0.35)
            event.accept()
            return
        elif self._rect_btn_add_prod.contains(local_pos):
            self.add_products(0.35)
            event.accept()
            return
        elif self._rect_btn_catalyst.contains(local_pos):
            self.toggle_catalyst()
            event.accept()
            return
        elif self._rect_btn_temp_dec.contains(local_pos):
            self.set_temperature(self._temp_k - 25.0)
            event.accept()
            return
        elif self._rect_btn_temp_inc.contains(local_pos):
            self.set_temperature(self._temp_k + 25.0)
            event.accept()
            return

        # Check Action Buttons
        if self._rect_btn_play_pause.contains(local_pos):
            self.toggle_play_pause()
            event.accept()
            return
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            event.accept()
            return
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas()
            event.accept()
            return

        # Check click on Thermometer area
        if self._prof_x <= local_pos.x() <= self._prof_x + self._prof_w and self._prof_y <= local_pos.y() <= self._prof_y + self._prof_h:
            ratio = 1.0 - (local_pos.y() - self._prof_y) / self._prof_h
            new_temp = 250.0 + ratio * 300.0
            self.set_temperature(new_temp)
            event.accept()
            return

        super().mousePressEvent(event)

    # ─── Bounding Rect & Painting ────────────────────────────────────────────

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

        # 1. Outer Dark STEM Dashboard Panel
        self._paint_dashboard_frame(painter)

        # 2. Left Column: Reaction Potential Energy Profile (E vs Reaction Coordinate)
        self._paint_energy_profile(painter)

        # 3. Right Top: Real-Time Concentration Curves [A](t) & [C](t)
        self._paint_concentration_plot(painter)

        # 4. Right Bottom: Reaction Rates & Le Chatelier Equilibrium HUD
        self._paint_equilibrium_hud(painter)

        # 5. Bottom Interactive Control Deck
        self._paint_control_deck(painter)

        painter.restore()

    def _paint_dashboard_frame(self, painter: QPainter) -> None:
        """Paints high-tech dark STEM HUD background container."""
        bg_rect = self.boundingRect()
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))  # Slate 900
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # Header Title Bar
        header_rect = QRectF(0.0, 0.0, self._width, 42.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 220)))  # Slate 800
        painter.drawRoundedRect(header_rect, 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 20.0, self._width, 22.0))

        # Title & Theoretical Equation
        spec = KINETICS_PRESETS[self._preset]
        rxn_name, rxn_eq, rxn_dh, _, _ = spec

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(
            16.0, 26.0,
            f"🧪 ĐỘNG HÓA HỌC & CÂN BẰNG: {rxn_name}"
        )

        temp_c = self._temp_k - 273.15
        cat_str = " | Xúc tác: BẬT" if self._has_catalyst else ""
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(
            self._width - 340.0, 26.0,
            f"{rxn_eq} | T = {self._temp_k:.0f}K ({temp_c:.0f}°C){cat_str}"
        )

        # Close button [X]
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_energy_profile(self, painter: QPainter) -> None:
        """Paints potential energy profile curve with activation hill and ΔH."""
        px = self._prof_x
        py = self._prof_y
        pw = self._prof_w
        ph = self._prof_h

        prof_rect = QRectF(px, py, pw, ph)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(prof_rect, 6.0, 6.0)

        # Profile Axes
        axis_pen = QPen(QColor("#64748b"), 1.2)
        bottom_y = py + ph - 24.0
        left_x = px + 28.0

        painter.setPen(axis_pen)
        painter.drawLine(QPointF(left_x, bottom_y), QPointF(px + pw - 12.0, bottom_y))
        painter.drawLine(QPointF(left_x, py + 14.0), QPointF(left_x, bottom_y))

        # Title & Axis labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(px + 10.0, py + 18.0, "SƠ ĐỒ THẾ NĂNG (ARRHENIUS)")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(left_x + 4.0, py + 22.0, "E (kJ/mol)")
        painter.drawText(px + pw - 72.0, bottom_y + 16.0, "Tiến trình ->")

        # Energy curve calculation
        y_react = py + ph * 0.58
        ea_val = self.effective_ea_f
        peak_y = py + 38.0
        dh_px = (self._delta_h / 120.0) * (ph * 0.25)
        y_prod = y_react - dh_px

        start_x = left_x + 14.0
        end_x = px + pw - 18.0
        mid_x = (start_x + end_x) / 2.0

        # 1. Uncatalyzed Curve (Red)
        uncat_path = QPainterPath()
        uncat_path.moveTo(start_x, y_react)
        uncat_path.lineTo(start_x + 18.0, y_react)
        uncat_path.cubicTo(mid_x - 24.0, y_react, mid_x - 14.0, peak_y, mid_x, peak_y)
        uncat_path.cubicTo(mid_x + 14.0, peak_y, mid_x + 24.0, y_prod, end_x - 18.0, y_prod)
        uncat_path.lineTo(end_x, y_prod)

        painter.setPen(QPen(QColor("#ef4444"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(uncat_path)

        # 2. Catalyzed Curve (Green dashed)
        if self._has_catalyst:
            cat_peak_y = peak_y + (y_react - peak_y) * 0.38
            cat_path = QPainterPath()
            cat_path.moveTo(start_x, y_react)
            cat_path.lineTo(start_x + 18.0, y_react)
            cat_path.cubicTo(mid_x - 22.0, y_react, mid_x - 12.0, cat_peak_y, mid_x, cat_peak_y)
            cat_path.cubicTo(mid_x + 12.0, cat_peak_y, mid_x + 22.0, y_prod, end_x - 18.0, y_prod)
            cat_path.lineTo(end_x, y_prod)

            painter.setPen(QPen(QColor("#10b981"), 1.8, Qt.PenStyle.DashLine))
            painter.drawPath(cat_path)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#10b981"))
            painter.drawText(mid_x - 22.0, cat_peak_y - 8.0, "Ea_cat (Xúc tác)")

        # Reactant and Product labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(start_x - 6.0, y_react + 14.0, "Chất PƯ (A+B)")

        painter.setPen(QColor("#fbbf24"))
        painter.drawText(end_x - 48.0, y_prod + 14.0, "Sản phẩm (C+D)")

        # Delta H arrow & text
        painter.setPen(QPen(QColor("#a855f7"), 1.2))
        painter.drawLine(QPointF(end_x - 6.0, y_react), QPointF(end_x - 6.0, y_prod))
        painter.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#a855f7"))
        dh_tag = "Tỏa nhiệt" if self._delta_h < 0 else "Thu nhiệt"
        painter.drawText(end_x - 65.0, (y_react + y_prod) / 2.0 - 2.0, f"ΔH={self._delta_h:.0f}kJ ({dh_tag})")

        # Bottom Temperature gauge line
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(px + 10.0, py + ph - 8.0, f"Nhiệt độ T = {self._temp_k:.0f}K | Ea = {self.effective_ea_f:.0f} kJ/mol")

    def _paint_concentration_plot(self, painter: QPainter) -> None:
        """Paints real-time concentration progress curves [A](t) and [C](t)."""
        px = self._plot_x
        py = self._plot_y
        pw = self._plot_w
        ph = self._plot_h

        plot_rect = QRectF(px, py, pw, ph)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(plot_rect, 6.0, 6.0)

        # Plot axes constants
        axis_x_start = px + 36.0
        axis_w = pw - 52.0
        axis_y_bottom = py + ph - 24.0
        axis_y_top = py + 24.0
        axis_h = axis_y_bottom - axis_y_top

        # Gridlines and concentration ticks: 0.0M, 0.5M, 1.0M, 1.5M, 2.0M
        max_conc = 2.0
        for i in range(5):
            c_val = i * 0.50
            y_pos = axis_y_bottom - (c_val / max_conc) * axis_h

            painter.setPen(QPen(QColor(30, 41, 59, 140), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(axis_x_start, y_pos), QPointF(axis_x_start + axis_w, y_pos))

            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QColor("#64748b"))
            painter.drawText(px + 4.0, y_pos + 4.0, f"{c_val:.1f}M")

        # Solid Axes
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawLine(QPointF(axis_x_start, axis_y_bottom), QPointF(axis_x_start + axis_w, axis_y_bottom))
        painter.drawLine(QPointF(axis_x_start, axis_y_top), QPointF(axis_x_start, axis_y_bottom))

        # Title & Legend
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(px + 12.0, py + 18.0, "ĐỒ THỊ NỒNG ĐỘ THEO THỜI GIAN: [A](t) & [C](t)")

        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(px + pw - 210.0, py + 18.0, f"[A]: {self._conc_a:.2f}M | [C]: {self._conc_c:.2f}M")

        if len(self._history) < 2:
            return

        # Plot curves
        path_a = QPainterPath()
        path_c = QPainterPath()

        hist_list = list(self._history)
        n = len(hist_list)

        for i, (t_val, a_val, c_val, vf_val, vr_val) in enumerate(hist_list):
            sx = axis_x_start + (i / float(n - 1)) * axis_w
            sy_a = axis_y_bottom - (a_val / max_conc) * axis_h
            sy_c = axis_y_bottom - (c_val / max_conc) * axis_h

            if i == 0:
                path_a.moveTo(sx, sy_a)
                path_c.moveTo(sx, sy_c)
            else:
                path_a.lineTo(sx, sy_a)
                path_c.lineTo(sx, sy_c)

        # Draw Reactant [A] (Cyan)
        painter.setPen(QPen(QColor("#38bdf8"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_a)

        # Draw Product [C] (Amber)
        painter.setPen(QPen(QColor("#f59e0b"), 2.2))
        painter.drawPath(path_c)

        # Probe dots at rightmost edge
        latest_x = axis_x_start + axis_w
        latest_ya = axis_y_bottom - (self._conc_a / max_conc) * axis_h
        latest_yc = axis_y_bottom - (self._conc_c / max_conc) * axis_h

        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(QPointF(latest_x, latest_ya), 3.5, 3.5)

        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.drawEllipse(QPointF(latest_x, latest_yc), 3.5, 3.5)

        # Legend at bottom
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(px + 12.0, py + ph - 8.0, "● Chất phản ứng [A]")
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(px + 130.0, py + ph - 8.0, "● Sản phẩm [C]")

    def _paint_equilibrium_hud(self, painter: QPainter) -> None:
        """Paints live reaction rates, Kc, Qc and Le Chatelier equilibrium status."""
        hx = self._hud_x
        hy = self._hud_y
        hw = self._hud_w
        hh = self._hud_h

        hud_rect = QRectF(hx, hy, hw, hh)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(hud_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#a855f7"))
        painter.drawText(hx + 12.0, hy + 18.0, "TỐC ĐỘ PHẢN ỨNG & NGUYÊN LÝ CHUYỂN DỊCH LE CHATELIER")

        vf = self.forward_rate_vf
        vr = self.reverse_rate_vr
        kc = self.equilibrium_constant_kc
        qc = self.reaction_quotient_qc

        # Rates Bar Gauges
        max_rate = max(0.5, vf, vr) * 1.2
        bar_x = hx + 115.0
        bar_w = 120.0
        bar_h = 8.0

        def draw_rate_bar(label: str, val: float, y_pos: float, col: QColor) -> None:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(col)
            painter.drawText(hx + 12.0, y_pos + bar_h - 1.0, label)

            # Slot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(QRectF(bar_x, y_pos, bar_w, bar_h), 2.5, 2.5)

            # Fill
            fill_w = max(2.0, min(bar_w, (val / max_rate) * bar_w))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(QRectF(bar_x, y_pos, fill_w, bar_h), 2.5, 2.5)

            # Text
            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(bar_x + fill_w + 4.0, y_pos + bar_h - 1.0, f"{val:.3f}")

        draw_rate_bar("Tốc độ thuận vf:", vf, hy + 32.0, QColor("#10b981"))
        draw_rate_bar("Tốc độ nghịch vr:", vr, hy + 48.0, QColor("#f59e0b"))

        # Equilibrium constants column
        f_txt = QFont("Consolas", 7)
        painter.setFont(f_txt)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(hx + 12.0, hy + 76.0, f"Hằng số Kc = {kc:.2f}")
        painter.drawText(hx + 130.0, hy + 76.0, f"Thương số Qc = {qc:.2f}")

        # Le Chatelier Active Status Tag
        shift_str = self.le_chatelier_shift
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        tag_col = QColor("#10b981") if "Cân bằng" in shift_str else (
            QColor("#38bdf8") if "Thuận" in shift_str else QColor("#f59e0b")
        )
        painter.setPen(tag_col)
        painter.drawText(hx + 12.0, hy + 98.0, f"● Trạng thái: {shift_str}")

        # Note
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            hx + 200.0, hy + 98.0,
            "Tăng T => dịch chuyển thu nhiệt; Xúc tác tăng tốc độ cả 2 chiều"
        )

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive button deck."""
        # Row 1: Reaction Presets
        def draw_preset_btn(rect: QRectF, text: str, is_active: bool) -> None:
            col = QColor("#0284c7") if is_active else QColor("#1e293b")
            border = col.lighter(130) if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        draw_preset_btn(self._rect_btn_haber, "🌱 Haber N₂+3H₂⇄2NH₃", self._preset == "HABER_BOSCH")
        draw_preset_btn(self._rect_btn_hi, "🧪 Tạo thành H₂+I₂⇄2HI", self._preset == "HYDROGEN_IODIDE")
        draw_preset_btn(self._rect_btn_no2, "💨 Khí 2NO₂⇄N₂O₄", self._preset == "NITROGEN_DIOXIDE")
        draw_preset_btn(self._rect_btn_generic, "⚙️ Tổng quát A+B⇄C+D", self._preset == "GENERIC")

        # Row 2: Perturbations & Actions
        # 1. Add Reactant
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.drawRoundedRect(self._rect_btn_add_react, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_add_react, Qt.AlignmentFlag.AlignCenter, "➕ Thêm chất PƯ [A]")

        # 2. Add Product
        painter.setPen(QPen(QColor("#d97706"), 1.0))
        painter.setBrush(QBrush(QColor("#b45309")))
        painter.drawRoundedRect(self._rect_btn_add_prod, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_add_prod, Qt.AlignmentFlag.AlignCenter, "➕ Thêm sản phẩm [C]")

        # 3. Catalyst Toggle
        cat_col = QColor("#059669") if self._has_catalyst else QColor("#334155")
        painter.setPen(QPen(cat_col.lighter(130), 1.0))
        painter.setBrush(QBrush(cat_col))
        painter.drawRoundedRect(self._rect_btn_catalyst, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        cat_txt = "⚡ Xúc tác: BẬT" if self._has_catalyst else "⚡ Xúc tác: TẮT"
        painter.drawText(self._rect_btn_catalyst, Qt.AlignmentFlag.AlignCenter, cat_txt)

        # 4. Temperature Stepper [-] [T] [+]
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_temp_dec, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_temp_dec, Qt.AlignmentFlag.AlignCenter, "−")

        t_mid = QRectF(self._rect_btn_temp_dec.right() + 2.0, self._rect_btn_temp_dec.top(), 56.0, 26.0)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(t_mid, Qt.AlignmentFlag.AlignCenter, f"{self._temp_k:.0f}K")

        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_temp_inc, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_temp_inc, Qt.AlignmentFlag.AlignCenter, "+")

        # 5. Play / Pause
        play_txt = "⏸ Dừng" if self._is_running else "▶ Tiếp"
        painter.setPen(QPen(QColor("#0284c7"), 1.2))
        painter.setBrush(QBrush(QColor("#0369a1") if self._is_running else QColor("#0284c7")))
        painter.drawRoundedRect(self._rect_btn_play_pause, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_play_pause, Qt.AlignmentFlag.AlignCenter, play_txt)

        # 6. Reset
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_reset, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "🔄 Đặt lại")

        # 7. Stamp
        painter.setPen(QPen(QColor("#059669"), 1.2))
        painter.setBrush(QBrush(QColor("#047857")))
        painter.drawRoundedRect(self._rect_btn_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng dấu")
