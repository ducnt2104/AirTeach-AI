"""
AirTeach AI - Interactive Nuclear Physics & Radioactive Decay Chain Law Simulator
===================================================================================
Provides an authentic nuclear physics simulation of radioactive decay:
  - Radioactive Decay Law (Định luật phóng xạ):
      N(t) = N₀ * exp(-λ * t) = N₀ * 2^(-t / T½)
  - Decay Constant λ & Half-life T½:
      λ = ln(2) / T½ ≈ 0.693147 / T½
      Mean lifetime τ = 1 / λ = T½ / ln(2) ≈ 1.4427 * T½
  - Activity (Độ phóng xạ A):
      A(t) = |dN/dt| = λ * N(t) = A₀ * 2^(-t / T½)
      Unit: Becquerel (1 Bq = 1 decay/s), Curie (1 Ci = 3.7 * 10¹⁰ Bq)
  - Microscopic Monte Carlo Stochastic Lattice:
      100 individual atomic nuclei (10x10 crystal matrix).
      In each discrete time step Δt, each undecayed nucleus has probability:
          P_decay = 1 - 2^(-Δt / T½)
      When decayed:
          Atom transitions from radiant parent color into stable daughter color.
          Emits animated radiation burst: α (Helium-4), β⁻ (electron), or γ (gamma photon).
          Geiger-Müller counter audio/visual pulse tick.
  - Authentic Isotope Presets:
      1. Carbon-14 (¹⁴₆C ➔ ¹⁴₇N + e⁻ + ν̄ₑ, T½ = 5,730 years, Radiocarbon dating)
      2. Radon-222 (²²²₈₆Rn ➔ ²¹⁸₈₄Po + ⁴₂He, T½ = 3.82 days, Geological radon)
      3. Iodine-131 (¹³¹₅₃I ➔ ¹³¹₅₄Xe + e⁻ + γ, T½ = 8.02 days, Thyroid radiotherapy)
      4. Uranium-238 (²³⁸₉₂U ➔ ²³⁴₉₀Th + ⁴₂He, T½ = 4.47 billion years, Earth age)
      5. Cobalt-60 (⁶⁰₂₇Co ➔ ⁶⁰₂₈Ni + e⁻ + γ, T½ = 5.27 years, Cancer therapy)
  - Radiocarbon Dating / Specimen Age Calculator:
      Estimates specimen age: t_age = -T½ * log2(N_remaining / N₀)
  - Touchless air gestures and vector diagram stamping via PhysicsTools.create_radioactive_decay_diagram.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
import math
import random
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

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


# Isotope Specifications: (name_vi, reaction_eq, radiation_type, half_life_str, half_life_s, parent_col, daughter_col)
ISOTOPE_SPECS: dict[str, tuple[str, str, str, str, float, str, str]] = {
    "CARBON_14": (
        "Carbon-14 (¹⁴₆C)",
        "¹⁴₆C ➔ ¹⁴₇N + e⁻ + ν̄ₑ",
        "β⁻ (Beta)",
        "5,730 năm",
        5730.0 * 365.25 * 86400.0,
        "#f59e0b",  # Amber parent
        "#10b981",  # Emerald daughter (Nitrogen-14)
    ),
    "RADON_222": (
        "Radon-222 (²²²₈₆Rn)",
        "²²²₈₆Rn ➔ ²¹⁸₈₄Po + ⁴₂He",
        "α (Alpha)",
        "3.82 ngày",
        3.82 * 86400.0,
        "#ef4444",  # Red parent
        "#06b6d4",  # Cyan daughter (Polonium-218)
    ),
    "IODINE_131": (
        "Iodine-131 (¹³¹₅₃I)",
        "¹³¹₅₃I ➔ ¹³¹₅₄Xe + e⁻ + γ",
        "β⁻, γ",
        "8.02 ngày",
        8.02 * 86400.0,
        "#a855f7",  # Purple parent
        "#38bdf8",  # Sky daughter (Xenon-131)
    ),
    "URANIUM_238": (
        "Uranium-238 (²³⁸₉₂U)",
        "²³⁸₉₂U ➔ ²³⁴₉₀Th + ⁴₂He",
        "α (Alpha)",
        "4.47 tỷ năm",
        4.47e9 * 365.25 * 86400.0,
        "#eab308",  # Gold parent
        "#10b981",  # Emerald daughter (Thorium-234)
    ),
    "COBALT_60": (
        "Cobalt-60 (⁶⁰₂₇Co)",
        "⁶⁰₂₇Co ➔ ⁶⁰₂₈Ni + e⁻ + γ",
        "β⁻, γ",
        "5.27 năm",
        5.27 * 365.25 * 86400.0,
        "#3b82f6",  # Blue parent
        "#22c55e",  # Green daughter (Nickel-60)
    ),
}


@dataclass
class AtomNucleus:
    """Represents a single atomic nucleus in the Monte Carlo lattice."""
    row: int
    col: int
    is_decayed: bool = False
    flash_timer: float = 0.0          # Animation duration for radiation flash
    particle_angle: float = 0.0       # Direction of emitted alpha/beta particle


class InteractiveRadioactiveDecayItem(QGraphicsObject):
    """
    Interactive Nuclear Physics & Radioactive Decay Chain Law Simulator item for QGraphicsScene.
    """

    decay_updated = Signal(int, int, float)  # (n_remaining, n_total, elapsed_half_lives)
    isotope_changed = Signal(str)            # ("CARBON_14", "RADON_222", etc.)
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (450.0, 240.0),
        width: float = 780.0,
        height: float = 500.0,
        isotope: str = "CARBON_14",
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

        # Active Isotope
        self._isotope = isotope.upper() if isotope.upper() in ISOTOPE_SPECS else "CARBON_14"

        # Monte Carlo Lattice: 10x10 = 100 nuclei
        self._num_rows = 10
        self._num_cols = 10
        self._total_nuclei = self._num_rows * self._num_cols
        self._nuclei: list[AtomNucleus] = [
            AtomNucleus(r, c)
            for r in range(self._num_rows)
            for c in range(self._num_cols)
        ]

        # Simulation Clock & Progress
        # We define 1.0 unit of simulated half-life T½ to correspond to 12.0 seconds at 1x speed
        self._elapsed_half_lives = 0.0
        self._sim_speed = 1.0             # Speed multiplier (1x, 2x, 5x)
        self._is_running = True           # Simulation active
        self._geiger_clicks_this_tick = 0 # Flash count in current tick

        # Stochastic time history buffer: list of (t_half, n_actual)
        self._history: list[tuple[float, int]] = [(0.0, self._total_nuclei)]

        # Animation Timer (35 FPS, dt = 0.028s)
        self._dt = 0.028
        self._seconds_per_half_life = 10.0  # 10s of real time = 1 T½ at 1x speed
        self._timer = QTimer()
        self._timer.setInterval(28)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Layout Geometry Constants
        self._lat_x = 24.0
        self._lat_y = 52.0
        self._lat_w = 270.0
        self._lat_h = 320.0

        self._plot_x = 310.0
        self._plot_y = 52.0
        self._plot_w = self._width - self._plot_x - 24.0  # 446.0 px
        self._plot_h = 210.0

        # Analysis HUD Box
        self._hud_x = 310.0
        self._hud_y = 272.0
        self._hud_w = self._plot_w
        self._hud_h = 100.0

        # Window Controls
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Control Deck Rectangles (y = 388 to 480)
        # Row 1: Isotope Presets (y = 388, h = 26)
        self._rect_btn_c14 = QRectF(24.0, 388.0, 136.0, 26.0)
        self._rect_btn_rn222 = QRectF(166.0, 388.0, 142.0, 26.0)
        self._rect_btn_i131 = QRectF(314.0, 388.0, 140.0, 26.0)
        self._rect_btn_u238 = QRectF(460.0, 388.0, 144.0, 26.0)
        self._rect_btn_co60 = QRectF(610.0, 388.0, 146.0, 26.0)

        # Row 2: Actions & Speed (y = 422, h = 26)
        self._rect_btn_play_pause = QRectF(24.0, 422.0, 100.0, 26.0)
        self._rect_btn_step_thalf = QRectF(130.0, 422.0, 150.0, 26.0)
        self._rect_btn_speed_1x = QRectF(286.0, 422.0, 52.0, 26.0)
        self._rect_btn_speed_2x = QRectF(344.0, 422.0, 52.0, 26.0)
        self._rect_btn_speed_5x = QRectF(402.0, 422.0, 52.0, 26.0)
        self._rect_btn_reset = QRectF(550.0, 422.0, 100.0, 26.0)
        self._rect_btn_stamp = QRectF(656.0, 422.0, 100.0, 26.0)

    # ─── Public Properties & Physical Solvers ────────────────────────────────

    @property
    def isotope(self) -> str:
        return self._isotope

    @property
    def total_nuclei(self) -> int:
        return self._total_nuclei

    @property
    def undecayed_count(self) -> int:
        """Returns number of remaining undecayed parent nuclei."""
        return sum(1 for n in self._nuclei if not n.is_decayed)

    @property
    def decayed_count(self) -> int:
        """Returns number of decayed daughter nuclei."""
        return self._total_nuclei - self.undecayed_count

    @property
    def elapsed_half_lives(self) -> float:
        """Returns elapsed time in units of half-life (t / T½)."""
        return self._elapsed_half_lives

    @property
    def theoretical_remaining_count(self) -> float:
        """Calculates theoretical remaining nuclei count: N(t) = N0 * 2^(-t / T½)."""
        return self._total_nuclei * (2.0 ** (-self._elapsed_half_lives))

    @property
    def theoretical_fraction(self) -> float:
        """Calculates theoretical remaining fraction N(t) / N0."""
        return 2.0 ** (-self._elapsed_half_lives)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def simulation_speed(self) -> float:
        return self._sim_speed

    # ─── Isotope & Simulation Controls ───────────────────────────────────────

    def set_isotope(self, isotope_name: str) -> None:
        """Switches active radioisotope preset and resets lattice."""
        name_upper = isotope_name.upper()
        if name_upper in ISOTOPE_SPECS and name_upper != self._isotope:
            self._isotope = name_upper
            self.reset_simulation()
            self.isotope_changed.emit(self._isotope)
            self.update()

    def set_speed(self, speed_mult: float) -> None:
        """Sets simulation speed multiplier [0.5x to 10.0x]."""
        self._sim_speed = max(0.5, min(10.0, float(speed_mult)))
        self.update()

    def toggle_play_pause(self) -> None:
        """Toggles simulation running state."""
        self._is_running = not self._is_running
        self.update()

    def reset_simulation(self) -> None:
        """Resets all nuclei to undecayed parent state and clears history."""
        for n in self._nuclei:
            n.is_decayed = False
            n.flash_timer = 0.0
        self._elapsed_half_lives = 0.0
        self._history = [(0.0, self._total_nuclei)]
        self._geiger_clicks_this_tick = 0
        self.decay_updated.emit(self.undecayed_count, self._total_nuclei, self._elapsed_half_lives)
        self.update()

    def advance_one_half_life(self) -> None:
        """
        Advances the stochastic simulation by exactly 1.0 half-life instantly,
        performing Monte Carlo trials on remaining parent nuclei.
        """
        # In exactly 1 half-life, each undecayed nucleus has decay probability P = 0.50
        clicks = 0
        for n in self._nuclei:
            if not n.is_decayed and random.random() < 0.50:
                n.is_decayed = True
                n.flash_timer = 1.0
                n.particle_angle = random.uniform(0.0, 2.0 * math.pi)
                clicks += 1

        self._elapsed_half_lives += 1.0
        self._geiger_clicks_this_tick = clicks
        self._history.append((self._elapsed_half_lives, self.undecayed_count))
        self.decay_updated.emit(self.undecayed_count, self._total_nuclei, self._elapsed_half_lives)
        self.update()

    # ─── Monte Carlo Discrete Time Step ──────────────────────────────────────

    def _monte_carlo_step(self, dt_half_life: float) -> None:
        """
        Executes a physical Monte Carlo stochastic decay step:
            Decay probability: P = 1 - 2^(-dt / T½)
        """
        p_decay = 1.0 - (2.0 ** (-dt_half_life))
        clicks = 0

        for n in self._nuclei:
            # Decay particle animation cooldown
            if n.flash_timer > 0.0:
                n.flash_timer = max(0.0, n.flash_timer - 0.08)

            # Stochastic trial on undecayed nuclei
            if not n.is_decayed:
                if random.random() < p_decay:
                    n.is_decayed = True
                    n.flash_timer = 1.0
                    n.particle_angle = random.uniform(0.0, 2.0 * math.pi)
                    clicks += 1

        self._elapsed_half_lives += dt_half_life
        self._geiger_clicks_this_tick = clicks

        # Record history if time advanced by at least 0.05 half-life
        if not self._history or (self._elapsed_half_lives - self._history[-1][0]) >= 0.05:
            self._history.append((self._elapsed_half_lives, self.undecayed_count))

        self.decay_updated.emit(self.undecayed_count, self._total_nuclei, self._elapsed_half_lives)

    def _on_sim_tick(self) -> None:
        """Simulation tick callback."""
        if self._is_running:
            dt_thalf = (self._dt / self._seconds_per_half_life) * self._sim_speed
            self._monte_carlo_step(dt_thalf)
        else:
            # Still update flash fade-outs even if paused
            for n in self._nuclei:
                if n.flash_timer > 0.0:
                    n.flash_timer = max(0.0, n.flash_timer - 0.08)

        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps laboratory radioactive decay lattice and exponential curve vector diagram onto LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        spec = ISOTOPE_SPECS[self._isotope]
        half_life_str = spec[3]

        items = PhysicsTools.create_radioactive_decay_diagram(
            center=(cx, cy),
            width=440.0,
            height=260.0,
            isotope=self._isotope,
            n_undecayed=self.undecayed_count,
            n_total=self._total_nuclei,
            elapsed_t_half=self._elapsed_half_lives,
            half_life_str=half_life_str,
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
            "Stamped Radioactive Decay diagram to LessonBoard (%s, N=%d/%d, t=%.2f T½)",
            self._isotope, self.undecayed_count, self._total_nuclei, self._elapsed_half_lives
        )
        return items

    # ─── Air Gesture & Interaction Handlers ──────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """
        Handles touchless air pinch gesture.
        If pinching the decay curve area, scrubs time elapsed.
        """
        local_pos = self.mapFromScene(scene_pos)

        # Check plot area for time scrubbing
        if self._plot_x <= local_pos.x() <= self._plot_x + self._plot_w and self._plot_y <= local_pos.y() <= self._plot_y + self._plot_h:
            frac = (local_pos.x() - self._plot_x) / self._plot_w
            new_t_half = max(0.0, min(4.0, frac * 4.0))

            # Reset and fast-forward to scrubbed time
            self.reset_simulation()
            # Advance in small steps to scrubbed time
            step = 0.05
            while self._elapsed_half_lives < new_t_half:
                self._monte_carlo_step(min(step, new_t_half - self._elapsed_half_lives))
            self.update()
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Handles touchless air tap gesture on buttons."""
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_c14.contains(local_pos):
            self.set_isotope("CARBON_14")
            return True
        elif self._rect_btn_rn222.contains(local_pos):
            self.set_isotope("RADON_222")
            return True
        elif self._rect_btn_i131.contains(local_pos):
            self.set_isotope("IODINE_131")
            return True
        elif self._rect_btn_u238.contains(local_pos):
            self.set_isotope("URANIUM_238")
            return True
        elif self._rect_btn_co60.contains(local_pos):
            self.set_isotope("COBALT_60")
            return True
        elif self._rect_btn_play_pause.contains(local_pos):
            self.toggle_play_pause()
            return True
        elif self._rect_btn_step_thalf.contains(local_pos):
            self.advance_one_half_life()
            return True
        elif self._rect_btn_speed_1x.contains(local_pos):
            self.set_speed(1.0)
            return True
        elif self._rect_btn_speed_2x.contains(local_pos):
            self.set_speed(2.0)
            return True
        elif self._rect_btn_speed_5x.contains(local_pos):
            self.set_speed(5.0)
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

        # Check Isotope Buttons
        if self._rect_btn_c14.contains(local_pos):
            self.set_isotope("CARBON_14")
            event.accept()
            return
        elif self._rect_btn_rn222.contains(local_pos):
            self.set_isotope("RADON_222")
            event.accept()
            return
        elif self._rect_btn_i131.contains(local_pos):
            self.set_isotope("IODINE_131")
            event.accept()
            return
        elif self._rect_btn_u238.contains(local_pos):
            self.set_isotope("URANIUM_238")
            event.accept()
            return
        elif self._rect_btn_co60.contains(local_pos):
            self.set_isotope("COBALT_60")
            event.accept()
            return

        # Check Action Buttons
        if self._rect_btn_play_pause.contains(local_pos):
            self.toggle_play_pause()
            event.accept()
            return
        elif self._rect_btn_step_thalf.contains(local_pos):
            self.advance_one_half_life()
            event.accept()
            return
        elif self._rect_btn_speed_1x.contains(local_pos):
            self.set_speed(1.0)
            event.accept()
            return
        elif self._rect_btn_speed_2x.contains(local_pos):
            self.set_speed(2.0)
            event.accept()
            return
        elif self._rect_btn_speed_5x.contains(local_pos):
            self.set_speed(5.0)
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

        # Check Click on Curve area to scrub time
        if self._plot_x <= local_pos.x() <= self._plot_x + self._plot_w and self._plot_y <= local_pos.y() <= self._plot_y + self._plot_h:
            frac = (local_pos.x() - self._plot_x) / self._plot_w
            new_t_half = max(0.0, min(4.0, frac * 4.0))
            self.reset_simulation()
            step = 0.05
            while self._elapsed_half_lives < new_t_half:
                self._monte_carlo_step(min(step, new_t_half - self._elapsed_half_lives))
            self.update()
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

        # 2. Left Column: 10x10 Monte Carlo Atomic Lattice Matrix & Radiation Bursts
        self._paint_atomic_lattice(painter)

        # 3. Right Top: Real-Time Exponential Decay Curve N(t) Overlaid with Stochastic Data Points
        self._paint_decay_plot(painter)

        # 4. Right Bottom: Radiometric Dating & Activity HUD
        self._paint_analysis_hud(painter)

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
        spec = ISOTOPE_SPECS[self._isotope]
        iso_name, decay_eq, rad_type, half_life_str, _, _, _ = spec

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(
            16.0, 26.0,
            f"☢️ ĐỊNH LUẬT PHÓNG XẠ & CHUỖI PHÂN RÃ ({iso_name})"
        )

        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(
            self._width - 340.0, 26.0,
            f"N(t) = N₀ · 2^(-t/T½) | T½ = {half_life_str}"
        )

        # Close button [X]
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_atomic_lattice(self, painter: QPainter) -> None:
        """Paints 10x10 Monte Carlo crystal lattice with particle bursts."""
        spec = ISOTOPE_SPECS[self._isotope]
        iso_name, decay_eq, rad_type, half_life_str, _, parent_hex, daughter_hex = spec

        lat_rect = QRectF(self._lat_x, self._lat_y, self._lat_w, self._lat_h)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(lat_rect, 6.0, 6.0)

        # Title & Geiger Counter Pulse Light
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(self._lat_x + 12.0, self._lat_y + 18.0, "MẠNG TINH THỂ NGUYÊN TỬ (100 HẠT)")

        # Geiger pulse LED at top-right of lattice
        is_clicking = (self._geiger_clicks_this_tick > 0)
        led_col = QColor("#ef4444") if is_clicking else QColor("#334155")
        painter.setPen(QPen(led_col.lighter(130), 1.0))
        painter.setBrush(QBrush(led_col))
        painter.drawEllipse(QPointF(self._lat_x + self._lat_w - 20.0, self._lat_y + 14.0), 5.0, 5.0)

        # Draw 10x10 Grid
        grid_left = self._lat_x + 18.0
        grid_top = self._lat_y + 36.0
        grid_w = self._lat_w - 36.0
        grid_h = 220.0
        step_x = grid_w / float(self._num_cols - 1)
        step_y = grid_h / float(self._num_rows - 1)
        atom_r = 7.0

        for n in self._nuclei:
            ax = grid_left + n.col * step_x
            ay = grid_top + n.row * step_y

            if not n.is_decayed:
                # Undecayed Parent Nucleus (Radiant with soft halo)
                grad = QRadialGradient(ax, ay, atom_r * 1.5)
                grad.setColorAt(0.0, QColor(parent_hex))
                grad.setColorAt(1.0, QColor(parent_hex).darker(150))
                painter.setPen(QPen(QColor("#ffffff"), 0.8))
                painter.setBrush(QBrush(grad))
                painter.drawEllipse(QPointF(ax, ay), atom_r, atom_r)
            else:
                # Decayed Stable Daughter Nucleus
                painter.setPen(QPen(QColor("#334155"), 1.0))
                painter.setBrush(QBrush(QColor(daughter_hex)))
                painter.drawEllipse(QPointF(ax, ay), atom_r - 1.5, atom_r - 1.5)

                # If currently decaying (radiation burst flash)
                if n.flash_timer > 0.05:
                    burst_r = atom_r + (1.0 - n.flash_timer) * 16.0
                    alpha = int(n.flash_timer * 220)
                    painter.setPen(QPen(QColor(239, 68, 68, alpha), 1.5))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(ax, ay), burst_r, burst_r)

                    # Ejected particle beam line
                    b_len = 18.0 * (1.0 - n.flash_timer)
                    bx = ax + b_len * math.cos(n.particle_angle)
                    by = ay + b_len * math.sin(n.particle_angle)
                    painter.setPen(QPen(QColor(251, 191, 36, alpha), 2.0))
                    painter.drawLine(QPointF(ax, ay), QPointF(bx, by))

        # Bottom Legend & Counts
        n_undec = self.undecayed_count
        pct = (n_undec / float(self._total_nuclei)) * 100.0

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor(parent_hex))
        painter.drawText(self._lat_x + 12.0, self._lat_y + self._lat_h - 34.0, f"● Mẹ: {n_undec} hạt ({pct:.1f}%)")

        painter.setPen(QColor(daughter_hex))
        painter.drawText(
            self._lat_x + 140.0, self._lat_y + self._lat_h - 34.0,
            f"● Con: {self.decayed_count} hạt"
        )

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            self._lat_x + 12.0, self._lat_y + self._lat_h - 14.0,
            f"Phương trình: {decay_eq} | Tia phát: {rad_type}"
        )

    def _paint_decay_plot(self, painter: QPainter) -> None:
        """Paints real-time theoretical exponential decay curve overlaid with actual Monte Carlo counts."""
        px = self._plot_x
        py = self._plot_y
        pw = self._plot_w
        ph = self._plot_h

        plot_rect = QRectF(px, py, pw, ph)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(plot_rect, 6.0, 6.0)

        # Coordinate axes constants
        axis_x_start = px + 42.0
        axis_w = pw - 60.0
        axis_y_bottom = py + ph - 26.0
        axis_y_top = py + 26.0
        axis_h = axis_y_bottom - axis_y_top

        # Gridlines and percentage ticks
        max_th = 4.0
        pct_ticks = [(1.0, "100%"), (0.50, "50%"), (0.25, "25%"), (0.125, "12.5%")]

        for frac, lbl in pct_ticks:
            y_pos = axis_y_bottom - frac * axis_h
            painter.setPen(QPen(QColor(30, 41, 59, 140), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(axis_x_start, y_pos), QPointF(axis_x_start + axis_w, y_pos))

            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(px + 4.0, y_pos + 4.0, lbl)

        # Time ticks on horizontal axis: 0, 1T½, 2T½, 3T½, 4T½
        for th in range(int(max_th) + 1):
            x_pos = axis_x_start + (th / max_th) * axis_w
            painter.setPen(QPen(QColor(30, 41, 59, 140), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x_pos, axis_y_top), QPointF(x_pos, axis_y_bottom))

            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(x_pos - 8.0, axis_y_bottom + 16.0, "0" if th == 0 else f"{th}T½")

        # Bottom and Left Solid Axes
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawLine(QPointF(axis_x_start, axis_y_bottom), QPointF(axis_x_start + axis_w, axis_y_bottom))
        painter.drawLine(QPointF(axis_x_start, axis_y_top), QPointF(axis_x_start, axis_y_bottom))

        # Title & Current Values Readout
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(px + 12.0, py + 18.0, "ĐỒ THỊ ĐỊNH LUẬT PHÓNG XẠ: N(t) = N₀ · 2^(-t/T½)")

        cur_th = self._elapsed_half_lives
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(
            px + pw - 220.0, py + 18.0,
            f"t = {cur_th:.2f} T½ | N_thực: {self.undecayed_count}/100"
        )

        # 1. Smooth Theoretical Curve: N(t) = 100 * 2^(-t / T½)
        theo_path = QPainterPath()
        num_pts = 80
        for i in range(num_pts + 1):
            t_val = (i / float(num_pts)) * max_th
            frac = 2.0 ** (-t_val)
            cx_pt = axis_x_start + (t_val / max_th) * axis_w
            cy_pt = axis_y_bottom - (frac * axis_h)
            if i == 0:
                theo_path.moveTo(cx_pt, cy_pt)
            else:
                theo_path.lineTo(cx_pt, cy_pt)

        painter.setPen(QPen(QColor("#0284c7"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(theo_path)

        # 2. Stochastic Experimental Points from Monte Carlo History
        if len(self._history) >= 2:
            stoch_path = QPainterPath()
            for i, (t_h, n_rem) in enumerate(self._history):
                if t_h > max_th:
                    break
                sx = axis_x_start + (t_h / max_th) * axis_w
                sy = axis_y_bottom - ((n_rem / float(self._total_nuclei)) * axis_h)
                if i == 0:
                    stoch_path.moveTo(sx, sy)
                else:
                    stoch_path.lineTo(sx, sy)

            painter.setPen(QPen(QColor(245, 158, 11, 200), 1.4))
            painter.drawPath(stoch_path)

        # 3. Current Operating Marker Point
        cur_t_clamped = min(max_th, cur_th)
        marker_x = axis_x_start + (cur_t_clamped / max_th) * axis_w
        marker_y = axis_y_bottom - ((self.undecayed_count / float(self._total_nuclei)) * axis_h)

        # Vertical dashed drop line
        painter.setPen(QPen(QColor("#ef4444"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(marker_x, marker_y), QPointF(marker_x, axis_y_bottom))

        # Marker Dot
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.drawEllipse(QPointF(marker_x, marker_y), 4.0, 4.0)

        # Legend
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#0284c7"))
        painter.drawText(px + 12.0, py + ph - 8.0, "― Lý thuyết: N₀·2^(-t/T½)")
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(px + 140.0, py + ph - 8.0, "● Thực nghiệm Monte Carlo")

    def _paint_analysis_hud(self, painter: QPainter) -> None:
        """Paints Radiometric Dating Age Calculator and Activity Readout."""
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
        painter.drawText(hx + 12.0, hy + 18.0, "ỨNG DỤNG ĐỊNH TUỔI PHÓNG XẠ & ĐỘ HOẠT ĐỘ PHÓNG XẠ")

        # Radiometric age calculation based on current remaining nuclei
        spec = ISOTOPE_SPECS[self._isotope]
        iso_name, _, _, half_life_str, hl_sec, _, _ = spec

        n_rem = max(1, self.undecayed_count)
        ratio = n_rem / float(self._total_nuclei)
        # t = -T1/2 * log2(ratio)
        calc_thalf = -math.log2(ratio)

        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#cbd5e1"))

        # Column 1
        painter.drawText(hx + 12.0, hy + 38.0, f"Tỷ lệ hạt mẹ: {ratio*100.0:.1f}%")
        painter.drawText(hx + 12.0, hy + 54.0, f"Thời gian trôi: {self._elapsed_half_lives:.2f} T½")
        painter.drawText(hx + 12.0, hy + 70.0, f"Độ phóng xạ A: {ratio*100.0:.1f}% A₀")

        # Column 2: Specimen Dating Age
        if self._isotope == "CARBON_14":
            age_years = calc_thalf * 5730.0
            age_str = f"{age_years:,.0f} năm"
            app_note = "✦ Định tuổi mẫu sinh học / Khảo cổ học cổ đại"
        elif self._isotope == "URANIUM_238":
            age_gyr = calc_thalf * 4.47
            age_str = f"{age_gyr:.2f} tỷ năm"
            app_note = "✦ Định tuổi mẫu đá địa chất & Tuổi Trái Đất"
        elif self._isotope == "RADON_222":
            age_days = calc_thalf * 3.82
            age_str = f"{age_days:.2f} ngày"
            app_note = "✦ Giám sát nồng độ khí Radon trong hầm mỏ"
        elif self._isotope == "IODINE_131":
            age_days = calc_thalf * 8.02
            age_str = f"{age_days:.2f} ngày"
            app_note = "✦ Liều xạ trị ung thư tuyến giáp (Y học hạt nhân)"
        else:
            age_years = calc_thalf * 5.27
            age_str = f"{age_years:.2f} năm"
            app_note = "✦ Nguồn xạ trị Coban & tiệt trùng thiết bị y tế"

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(hx + 180.0, hy + 38.0, f"Tuổi suy ra của mẫu vật: {age_str}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(hx + 180.0, hy + 54.0, app_note)
        painter.drawText(hx + 180.0, hy + 70.0, "Công thức: t = -T½ · log₂(N/N₀)")

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive button deck."""
        # Row 1: Isotope Selectors
        def draw_iso_btn(rect: QRectF, text: str, is_active: bool, col_hex: str) -> None:
            col = QColor(col_hex) if is_active else QColor("#1e293b")
            border = col.lighter(130) if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        draw_iso_btn(self._rect_btn_c14, "🌱 Carbon-14", self._isotope == "CARBON_14", "#d97706")
        draw_iso_btn(self._rect_btn_rn222, "⛰️ Radon-222", self._isotope == "RADON_222", "#dc2626")
        draw_iso_btn(self._rect_btn_i131, "🏥 Iodine-131", self._isotope == "IODINE_131", "#9333ea")
        draw_iso_btn(self._rect_btn_u238, "🌍 Uranium-238", self._isotope == "URANIUM_238", "#ca8a04")
        draw_iso_btn(self._rect_btn_co60, "⚡ Cobalt-60", self._isotope == "COBALT_60", "#2563eb")

        # Row 2: Action Buttons
        # 1. Play / Pause
        play_txt = "⏸ Dừng" if self._is_running else "▶ Tiếp"
        painter.setPen(QPen(QColor("#0284c7"), 1.2))
        painter.setBrush(QBrush(QColor("#0369a1") if self._is_running else QColor("#0284c7")))
        painter.drawRoundedRect(self._rect_btn_play_pause, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_play_pause, Qt.AlignmentFlag.AlignCenter, play_txt)

        # 2. Step 1 T1/2
        painter.setPen(QPen(QColor("#a855f7"), 1.2))
        painter.setBrush(QBrush(QColor("#7e22ce")))
        painter.drawRoundedRect(self._rect_btn_step_thalf, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_step_thalf, Qt.AlignmentFlag.AlignCenter, "⚡ Nhảy 1 Chu kỳ T½")

        # 3. Speeds
        def draw_speed_btn(rect: QRectF, text: str, is_active: bool) -> None:
            col = QColor("#0284c7") if is_active else QColor("#334155")
            painter.setPen(QPen(col.lighter(130), 1.0))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        draw_speed_btn(self._rect_btn_speed_1x, "1x", abs(self._sim_speed - 1.0) < 0.1)
        draw_speed_btn(self._rect_btn_speed_2x, "2x", abs(self._sim_speed - 2.0) < 0.1)
        draw_speed_btn(self._rect_btn_speed_5x, "5x", abs(self._sim_speed - 5.0) < 0.1)

        # 4. Reset
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_reset, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "🔄 Đặt lại")

        # 5. Stamp
        painter.setPen(QPen(QColor("#059669"), 1.2))
        painter.setBrush(QBrush(QColor("#047857")))
        painter.drawRoundedRect(self._rect_btn_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng dấu")
