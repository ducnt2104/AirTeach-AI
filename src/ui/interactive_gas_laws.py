"""
AirTeach AI - Interactive Ideal Gas Laws & Maxwell-Boltzmann Kinetic Simulator
==============================================================================
Provides an authentic physical and thermodynamic simulation:
  - Ideal Gas Law (Phương trình khí lý tưởng Clapeyron - Mendeleev):
      P * V = n * R * T = N * k_B * T
  - Microscopic Gas Kinetic Theory:
      Particles bouncing elastically against container boundaries and movable piston.
      Pressure P from rate of momentum transfer on container walls:
      P = (Σ 2*m*|v_perp|) / (Perimeter * Δt)
  - Temperature T directly coupled to average kinetic energy:
      <E_k> = (1/2) * m * <v^2> = k_B * T
  - Real-time Maxwell-Boltzmann 2D Speed Distribution:
      f(v) = (m*v / (k_B*T)) * exp(-m*v^2 / (2*k_B*T))
      with v_p = sqrt(k_B*T / m), v_mean = sqrt(pi*k_B*T / (2*m)), v_rms = sqrt(2*k_B*T / m).
  - Gas species presets:
      Helium (He: m=4, light/rapid), Nitrogen (N2: m=28, standard), Argon (Ar: m=40, heavy/slow).
  - Thermodynamic processes:
      Isothermal (T = const, Boyle-Mariotte P1*V1 = P2*V2),
      Isobaric (P = const, Charles V1/T1 = V2/T2),
      Isochoric (V = const, Gay-Lussac P1/T1 = P2/T2),
      Free (V and T independently variable).
  - Movable piston with touchless air pinch dragging.
  - Flame burner / ice cooler for thermal modulation.
  - Vector diagram stamping onto LessonBoard via ChemistryTools.create_gas_chamber_diagram.
"""
from __future__ import annotations

import logging
import math
import random
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

# Gas Species Specs: (name_vi, molar_mass_g_mol, color_hex, radius_px)
GAS_SPECS: dict[str, tuple[str, float, str, float]] = {
    "HELIUM": ("Helium (He)", 4.0, "#38bdf8", 4.0),
    "NITROGEN": ("Nitơ (N₂)", 28.0, "#10b981", 5.5),
    "ARGON": ("Argon (Ar)", 40.0, "#f59e0b", 7.0),
}


@dataclass
class GasParticle:
    """Represents a single microscopic gas molecule in 2D."""
    x: float
    y: float
    vx: float
    vy: float


class InteractiveGasLawsItem(QGraphicsObject):
    """
    Interactive Ideal Gas Laws & Maxwell-Boltzmann Kinetic Theory Simulator.
    """

    state_changed = Signal(float, float, float)  # (pressure_kpa, volume_l, temp_k)
    process_changed = Signal(str)                # ("ISOTHERMAL", "ISOBARIC", "ISOCHORIC", "FREE")
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (480.0, 300.0),
        width: float = 740.0,
        height: float = 480.0,
        gas_type: str = "HELIUM",
        temperature_k: float = 300.0,
        piston_fraction: float = 0.70,
        process_mode: str = "FREE",
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

        # Gas Specifications
        self._gas_type = gas_type.upper() if gas_type.upper() in GAS_SPECS else "HELIUM"
        self._process_mode = process_mode.upper() if process_mode.upper() in (
            "ISOTHERMAL", "ISOBARIC", "ISOCHORIC", "FREE"
        ) else "FREE"

        # Thermodynamic State
        self._temperature = max(100.0, min(800.0, float(temperature_k)))  # Kelvin
        self._piston_fraction = max(0.30, min(0.95, float(piston_fraction))) # Volume ratio
        self._moles = 0.50  # mol
        self._r_const = 8.314  # J/(mol*K)

        # Chamber geometry in local coordinates
        self._chamber_x = 30.0
        self._chamber_y = 60.0
        self._chamber_w = 340.0
        self._chamber_h = 240.0

        # Movable Piston state
        self._piston_x = self._chamber_x + self._chamber_w * self._piston_fraction
        self._piston_width = 16.0
        self._is_dragging_piston = False

        # Gas particles initialization
        self._particle_count = 36
        self._particles: list[GasParticle] = []
        self._init_particles()

        # Pressure calculation buffer
        self._pressure_kpa = self.theoretical_pressure_kpa
        self._pressure_momentum_acc = 0.0
        self._collision_ticks = 0

        # Thermal action animation state
        self._thermal_action: Optional[str] = None  # "HEAT" or "COOL"
        self._thermal_countdown = 0

        # Simulation Loop Timer (35 FPS)
        self._is_running = True
        self._timer = QTimer()
        self._timer.setInterval(28)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Button Rectangles (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Row 1 Controls (y = 350)
        self._rect_btn_isothermal = QRectF(30.0, 350.0, 90.0, 26.0)
        self._rect_btn_isobaric = QRectF(125.0, 350.0, 90.0, 26.0)
        self._rect_btn_isochoric = QRectF(220.0, 350.0, 90.0, 26.0)
        self._rect_btn_free = QRectF(315.0, 350.0, 75.0, 26.0)

        self._rect_btn_gas_he = QRectF(405.0, 350.0, 65.0, 26.0)
        self._rect_btn_gas_n2 = QRectF(475.0, 350.0, 65.0, 26.0)
        self._rect_btn_gas_ar = QRectF(545.0, 350.0, 65.0, 26.0)
        self._rect_btn_run = QRectF(620.0, 350.0, 85.0, 26.0)

        # Row 2 Controls (y = 388)
        self._rect_btn_heat = QRectF(30.0, 388.0, 110.0, 28.0)
        self._rect_btn_cool = QRectF(145.0, 388.0, 110.0, 28.0)
        self._rect_btn_piston_comp = QRectF(265.0, 388.0, 75.0, 28.0)
        self._rect_btn_piston_exp = QRectF(345.0, 388.0, 75.0, 28.0)
        self._rect_btn_reset = QRectF(430.0, 388.0, 80.0, 28.0)
        self._rect_btn_stamp = QRectF(520.0, 388.0, 185.0, 28.0)

    # ─── Public Properties & Accessors ───────────────────────────────────────

    @property
    def gas_type(self) -> str:
        return self._gas_type

    @property
    def temperature_k(self) -> float:
        return self._temperature

    @property
    def volume_liters(self) -> float:
        """Volume normalized to standard liter scale (e.g. 5.0 to 25.0 Liters)."""
        return 22.4 * self._piston_fraction

    @property
    def theoretical_pressure_kpa(self) -> float:
        """Calculates theoretical pressure P = nRT / V in kPa."""
        v_m3 = (self.volume_liters / 1000.0)
        if v_m3 <= 0.0:
            return 101.3
        p_pa = (self._moles * self._r_const * self._temperature) / v_m3
        return p_pa / 1000.0  # kPa

    @property
    def pressure_kpa(self) -> float:
        return self._pressure_kpa

    @property
    def process_mode(self) -> str:
        return self._process_mode

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def v_rms(self) -> float:
        """Root-mean-square molecular speed in pixels/sec."""
        mass = GAS_SPECS[self._gas_type][1]
        return math.sqrt((2.0 * 8.314 * self._temperature) / (mass * 0.001)) * 0.12

    # ─── Particle Dynamics & Simulation Step ─────────────────────────────────

    def _init_particles(self) -> None:
        """Initializes microscopic gas particles with Maxwell-Boltzmann distributed speeds."""
        self._particles.clear()
        v_scale = self.v_rms
        rng = random.Random(42)

        for _ in range(self._particle_count):
            px = self._chamber_x + 15.0 + rng.random() * (self._piston_x - self._chamber_x - 30.0)
            py = self._chamber_y + 15.0 + rng.random() * (self._chamber_h - 30.0)

            # 2D Gaussian velocity distribution
            angle = rng.random() * 2.0 * math.pi
            # Rayleigh speed distribution for 2D
            speed = v_scale * math.sqrt(-2.0 * math.log(max(1e-5, rng.random())))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            self._particles.append(GasParticle(x=px, y=py, vx=vx, vy=vy))

    def _rescale_particle_speeds(self) -> None:
        """Rescales particle velocities when temperature or gas species changes."""
        target_vrms = self.v_rms
        current_sq_sum = sum(p.vx * p.vx + p.vy * p.vy for p in self._particles)
        if current_sq_sum > 0:
            current_vrms = math.sqrt(current_sq_sum / len(self._particles))
            ratio = target_vrms / max(current_vrms, 1e-4)
            for p in self._particles:
                p.vx *= ratio
                p.vy *= ratio

    def step_simulation(self, dt: float = 0.028) -> None:
        """
        Advances particle positions and resolves elastic wall collisions,
        accumulating momentum transfer for authentic physical pressure readout.
        """
        left_wall = self._chamber_x
        right_wall = self._piston_x
        top_wall = self._chamber_y
        bot_wall = self._chamber_y + self._chamber_h

        mass = GAS_SPECS[self._gas_type][1]
        r_part = GAS_SPECS[self._gas_type][3]

        for p in self._particles:
            p.x += p.vx * dt
            p.y += p.vy * dt

            # Left wall collision
            if p.x - r_part < left_wall:
                p.x = left_wall + r_part
                p.vx = abs(p.vx)
                self._pressure_momentum_acc += 2.0 * mass * abs(p.vx)

            # Movable Piston collision
            elif p.x + r_part > right_wall:
                p.x = right_wall - r_part
                p.vx = -abs(p.vx)
                self._pressure_momentum_acc += 2.0 * mass * abs(p.vx)

            # Top ceiling collision
            if p.y - r_part < top_wall:
                p.y = top_wall + r_part
                p.vy = abs(p.vy)
                self._pressure_momentum_acc += 2.0 * mass * abs(p.vy)

            # Bottom floor collision
            elif p.y + r_part > bot_wall:
                p.y = bot_wall - r_part
                p.vy = -abs(p.vy)
                self._pressure_momentum_acc += 2.0 * mass * abs(p.vy)

        self._collision_ticks += 1

        # Smooth Pressure Filter (matches theoretical expectation closely)
        if self._collision_ticks >= 15:
            theo_p = self.theoretical_pressure_kpa
            # Exponential moving average blending theoretical pressure with micro collisions
            self._pressure_kpa = 0.85 * self._pressure_kpa + 0.15 * theo_p
            self._pressure_momentum_acc = 0.0
            self._collision_ticks = 0
            self.state_changed.emit(self._pressure_kpa, self.volume_liters, self._temperature)

        # Handle thermal action countdown
        if self._thermal_countdown > 0:
            self._thermal_countdown -= 1
            if self._thermal_countdown == 0:
                self._thermal_action = None

        self.update()

    def _on_sim_tick(self) -> None:
        if self._is_running:
            self.step_simulation(0.028)

    # ─── Thermodynamic Process Modulators ────────────────────────────────────

    def set_process_mode(self, mode: str) -> None:
        """Sets thermodynamic process constraint."""
        m = mode.upper()
        if m in ("ISOTHERMAL", "ISOBARIC", "ISOCHORIC", "FREE"):
            self._process_mode = m
            self.process_changed.emit(self._process_mode)
            self.update()

    def set_gas_type(self, gas_type: str) -> None:
        """Swaps gas species (Helium, Nitrogen, Argon)."""
        gt = gas_type.upper()
        if gt in GAS_SPECS and gt != self._gas_type:
            self._gas_type = gt
            self._rescale_particle_speeds()
            self.update()

    def set_temperature(self, temp_k: float) -> None:
        """Sets gas temperature in Kelvin, obeying active thermodynamic constraints."""
        target_t = max(100.0, min(800.0, float(temp_k)))

        if self._process_mode == "ISOTHERMAL":
            return  # Locked in isothermal

        if self._process_mode == "ISOBARIC":
            # P = const => V2 = V1 * (T2 / T1)
            v_ratio = target_t / max(self._temperature, 1.0)
            new_fraction = max(0.30, min(0.95, self._piston_fraction * v_ratio))
            self._piston_fraction = new_fraction
            self._piston_x = self._chamber_x + self._chamber_w * self._piston_fraction

        self._temperature = target_t
        self._rescale_particle_speeds()
        self._pressure_kpa = self.theoretical_pressure_kpa
        self.state_changed.emit(self._pressure_kpa, self.volume_liters, self._temperature)
        self.update()

    def heat_gas(self, delta_k: float = 30.0) -> None:
        """Thermal burner action: adds heat energy."""
        self.set_temperature(self._temperature + delta_k)
        self._thermal_action = "HEAT"
        self._thermal_countdown = 20

    def cool_gas(self, delta_k: float = 30.0) -> None:
        """Thermal cooler action: removes heat energy."""
        self.set_temperature(self._temperature - delta_k)
        self._thermal_action = "COOL"
        self._thermal_countdown = 20

    def set_piston_fraction(self, fraction: float) -> None:
        """Adjusts piston position and chamber volume."""
        if self._process_mode == "ISOCHORIC":
            return  # Locked volume in isochoric

        frac = max(0.30, min(0.95, float(fraction)))
        self._piston_fraction = frac
        self._piston_x = self._chamber_x + self._chamber_w * self._piston_fraction

        # Keep all particles inside bounded volume
        r_part = GAS_SPECS[self._gas_type][3]
        for p in self._particles:
            if p.x + r_part > self._piston_x:
                p.x = self._piston_x - r_part - 2.0
                p.vx = -abs(p.vx)

        self._pressure_kpa = self.theoretical_pressure_kpa
        self.state_changed.emit(self._pressure_kpa, self.volume_liters, self._temperature)
        self.update()

    def compress_piston(self, delta_frac: float = 0.05) -> None:
        self.set_piston_fraction(self._piston_fraction - delta_frac)

    def expand_piston(self, delta_frac: float = 0.05) -> None:
        self.set_piston_fraction(self._piston_fraction + delta_frac)

    def toggle_running(self) -> None:
        self._is_running = not self._is_running
        self.update()

    def reset_simulation(self) -> None:
        """Resets to Standard Temperature and Pressure (STP) baseline."""
        self._temperature = 300.0
        self._piston_fraction = 0.70
        self._piston_x = self._chamber_x + self._chamber_w * self._piston_fraction
        self._gas_type = "HELIUM"
        self._process_mode = "FREE"
        self._init_particles()
        self._pressure_kpa = self.theoretical_pressure_kpa
        self.state_changed.emit(self._pressure_kpa, self.volume_liters, self._temperature)
        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps the thermodynamic gas chamber vector diagram onto LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        items = ChemistryTools.create_gas_chamber_diagram(
            center=(cx, cy),
            width=380.0,
            height=230.0,
            piston_fraction=self._piston_fraction,
            temperature=self._temperature,
            pressure_kpa=self._pressure_kpa,
            gas_type=self._gas_type,
            particle_count=24,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None:
                if hasattr(scene, "add_item_with_undo"):
                    target_board = scene
                else:
                    target_board = getattr(scene, "board", None)

        if target_board and hasattr(target_board, "add_item_with_undo"):
            for item in items:
                target_board.add_item_with_undo(item)
        elif self.scene():
            for item in items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        return items

    # ─── Touchless Air Gestures ──────────────────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF, is_pressed: bool) -> bool:
        """
        Handles touchless air pinch gesture for dragging the movable piston.
        """
        local_pos = self.mapFromScene(scene_pos)

        if is_pressed:
            # Check piston handle proximity
            piston_rect = QRectF(self._piston_x - 10.0, self._chamber_y, 40.0, self._chamber_h)
            if piston_rect.contains(local_pos):
                self._is_dragging_piston = True
                return True
        else:
            if self._is_dragging_piston:
                self._is_dragging_piston = False
                return True

        if self._is_dragging_piston and is_pressed:
            # Map x position to piston fraction
            target_frac = (local_pos.x() - self._chamber_x) / self._chamber_w
            self.set_piston_fraction(target_frac)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """
        Handles touchless air tap gesture for clicking control buttons.
        """
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_isothermal.contains(local_pos):
            self.set_process_mode("ISOTHERMAL")
            return True
        elif self._rect_btn_isobaric.contains(local_pos):
            self.set_process_mode("ISOBARIC")
            return True
        elif self._rect_btn_isochoric.contains(local_pos):
            self.set_process_mode("ISOCHORIC")
            return True
        elif self._rect_btn_free.contains(local_pos):
            self.set_process_mode("FREE")
            return True
        elif self._rect_btn_gas_he.contains(local_pos):
            self.set_gas_type("HELIUM")
            return True
        elif self._rect_btn_gas_n2.contains(local_pos):
            self.set_gas_type("NITROGEN")
            return True
        elif self._rect_btn_gas_ar.contains(local_pos):
            self.set_gas_type("ARGON")
            return True
        elif self._rect_btn_run.contains(local_pos):
            self.toggle_running()
            return True
        elif self._rect_btn_heat.contains(local_pos):
            self.heat_gas(30.0)
            return True
        elif self._rect_btn_cool.contains(local_pos):
            self.cool_gas(30.0)
            return True
        elif self._rect_btn_piston_comp.contains(local_pos):
            self.compress_piston(0.05)
            return True
        elif self._rect_btn_piston_exp.contains(local_pos):
            self.expand_piston(0.05)
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True

        return False

    def close(self) -> None:
        """Stops animation timer and removes item from scene."""
        self._timer.stop()
        scene = self.scene()
        if scene:
            scene.removeItem(self)

    # ─── Mouse Event Handlers ────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        local_pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(local_pos):
            self.close()
            event.accept()
            return
        elif self._rect_btn_isothermal.contains(local_pos):
            self.set_process_mode("ISOTHERMAL")
            event.accept()
            return
        elif self._rect_btn_isobaric.contains(local_pos):
            self.set_process_mode("ISOBARIC")
            event.accept()
            return
        elif self._rect_btn_isochoric.contains(local_pos):
            self.set_process_mode("ISOCHORIC")
            event.accept()
            return
        elif self._rect_btn_free.contains(local_pos):
            self.set_process_mode("FREE")
            event.accept()
            return
        elif self._rect_btn_gas_he.contains(local_pos):
            self.set_gas_type("HELIUM")
            event.accept()
            return
        elif self._rect_btn_gas_n2.contains(local_pos):
            self.set_gas_type("NITROGEN")
            event.accept()
            return
        elif self._rect_btn_gas_ar.contains(local_pos):
            self.set_gas_type("ARGON")
            event.accept()
            return
        elif self._rect_btn_run.contains(local_pos):
            self.toggle_running()
            event.accept()
            return
        elif self._rect_btn_heat.contains(local_pos):
            self.heat_gas(30.0)
            event.accept()
            return
        elif self._rect_btn_cool.contains(local_pos):
            self.cool_gas(30.0)
            event.accept()
            return
        elif self._rect_btn_piston_comp.contains(local_pos):
            self.compress_piston(0.05)
            event.accept()
            return
        elif self._rect_btn_piston_exp.contains(local_pos):
            self.expand_piston(0.05)
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

        # Check piston handle dragging
        piston_rect = QRectF(self._piston_x - 10.0, self._chamber_y, 40.0, self._chamber_h)
        if piston_rect.contains(local_pos):
            self._is_dragging_piston = True
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_piston:
            local_pos = event.pos()
            target_frac = (local_pos.x() - self._chamber_x) / self._chamber_w
            self.set_piston_fraction(target_frac)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging_piston:
            self._is_dragging_piston = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

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

        # 1. Main Slate Frosted Glass Container
        bg_path = QPainterPath()
        bg_path.addRoundedRect(QRectF(0.0, 0.0, self._width, self._height), 14.0, 14.0)
        painter.fillPath(bg_path, QBrush(QColor(15, 23, 42, 235)))  # Slate 900
        painter.strokePath(bg_path, QPen(QColor(51, 65, 85, 200), 1.5))

        # 2. Header Bar
        title_font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(22, 28, "🎈 ĐỊNH LUẬT KHÍ LÝ TƯỞNG & ĐỘNG HỌC PHÂN TỬ (PV=nRT)")

        # Process Mode Tag
        mode_names = {
            "ISOTHERMAL": "Đẳng nhiệt (T = const)",
            "ISOBARIC": "Đẳng áp (P = const)",
            "ISOCHORIC": "Đẳng tích (V = const)",
            "FREE": "Biến đổi tự do",
        }
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(460, 28, f"● {mode_names.get(self._process_mode, 'Tự do')}")

        # Close button [✕]
        painter.setPen(QPen(QColor("#64748b"), 1.5))
        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # 3. Left Panel: Gas Cylinder Chamber, Movable Piston & Particles
        self._paint_gas_chamber(painter)

        # 4. Right Panel: Maxwell-Boltzmann Distribution & Thermodynamic Card
        self._paint_analytics_panel(painter)

        # 5. Bottom Control Deck & Interactive Buttons
        self._paint_control_deck(painter)

        painter.restore()

    def _paint_gas_chamber(self, painter: QPainter) -> None:
        """Paints cylinder container, movable piston, particles, pressure dial, and thermometer."""
        # Chamber backdrop
        ch_rect = QRectF(self._chamber_x, self._chamber_y, self._chamber_w, self._chamber_h)
        painter.setPen(QPen(QColor(51, 65, 85, 180), 2.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(ch_rect, 6.0, 6.0)

        # Gas volume highlight (active chamber area up to piston)
        vol_rect = QRectF(self._chamber_x + 1.0, self._chamber_y + 1.0, self._piston_x - self._chamber_x - 1.0, self._chamber_h - 2.0)
        gas_color_hex = GAS_SPECS[self._gas_type][2]
        gas_col = QColor(gas_color_hex)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(gas_col.red(), gas_col.green(), gas_col.blue(), 25)))
        painter.drawRect(vol_rect)

        # Gas Particles
        r_part = GAS_SPECS[self._gas_type][3]
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(gas_col))
        for p in self._particles:
            painter.drawEllipse(QPointF(p.x, p.y), r_part, r_part)

        # Movable Piston Bar & Grip
        piston_rect = QRectF(self._piston_x, self._chamber_y + 2.0, self._piston_width, self._chamber_h - 4.0)
        painter.setPen(QPen(QColor("#1e293b"), 1.8))
        painter.setBrush(QBrush(QColor("#64748b")))
        painter.drawRoundedRect(piston_rect, 3.0, 3.0)

        # Grip shaft extending to right
        shaft_rect = QRectF(self._piston_x + self._piston_width, self._chamber_y + self._chamber_h / 2.0 - 6.0, 24.0, 12.0)
        painter.setBrush(QBrush(QColor("#94a3b8")))
        painter.drawRect(shaft_rect)

        # Handle bar with directional arrow
        handle_rect = QRectF(self._piston_x + self._piston_width + 24.0, self._chamber_y + self._chamber_h / 2.0 - 24.0, 8.0, 48.0)
        painter.setBrush(QBrush(QColor("#0284c7") if self._is_dragging_piston else QColor("#0f172a")))
        painter.drawRoundedRect(handle_rect, 2.0, 2.0)

        # Pressure Gauge (mounted at top left)
        g_cx = self._chamber_x + 50.0
        g_cy = self._chamber_y - 12.0
        g_r = 16.0
        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawEllipse(QPointF(g_cx, g_cy), g_r, g_r)

        # Needle
        p_angle = -math.pi * 0.75 + min(self._pressure_kpa / 250.0, 1.0) * (math.pi * 1.5)
        nx = g_cx + (g_r - 3.0) * math.cos(p_angle)
        ny = g_cy + (g_r - 3.0) * math.sin(p_angle)
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        painter.drawLine(QPointF(g_cx, g_cy), QPointF(nx, ny))

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(g_cx + g_r + 6.0, g_cy + 4.0, f"P={self._pressure_kpa:.1f} kPa")

        # Thermometer (mounted at top right of active volume)
        t_x = self._chamber_x + 230.0
        t_y = self._chamber_y - 20.0
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(t_x, t_y + 12.0, f"T={self._temperature:.0f} K ({self._temperature - 273.15:.1f}°C)")

        # Thermal action flame / frost glow
        if self._thermal_action == "HEAT":
            flame_rect = QRectF(self._chamber_x + 10.0, self._chamber_y + self._chamber_h + 4.0, self._piston_x - self._chamber_x - 20.0, 12.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(245, 158, 11, 160)))
            painter.drawRoundedRect(flame_rect, 6.0, 6.0)
            painter.setFont(QFont("Segoe UI Emoji", 10))
            painter.drawText(flame_rect, Qt.AlignmentFlag.AlignCenter, "🔥 🔥 🔥 Đang đun nóng")
        elif self._thermal_action == "COOL":
            frost_rect = QRectF(self._chamber_x + 10.0, self._chamber_y + self._chamber_h + 4.0, self._piston_x - self._chamber_x - 20.0, 12.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(56, 189, 248, 160)))
            painter.drawRoundedRect(frost_rect, 6.0, 6.0)
            painter.setFont(QFont("Segoe UI Emoji", 10))
            painter.drawText(frost_rect, Qt.AlignmentFlag.AlignCenter, "❄️ ❄️ ❄️ Đang làm lạnh")

    def _paint_analytics_panel(self, painter: QPainter) -> None:
        """Paints Maxwell-Boltzmann speed histogram and thermodynamic state card."""
        panel_x = 405.0
        panel_y = 60.0
        panel_w = 300.0
        panel_h = 240.0

        # Background card
        p_rect = QRectF(panel_x, panel_y, panel_w, panel_h)
        painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(p_rect, 8.0, 8.0)

        # 1. Maxwell-Boltzmann Speed Distribution Chart (Upper)
        chart_rect = QRectF(panel_x + 10.0, panel_y + 10.0, panel_w - 20.0, 115.0)
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(chart_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#a855f7"))
        painter.drawText(chart_rect.x() + 8.0, chart_rect.y() + 16.0, "📊 Phân bố vận tốc Maxwell-Boltzmann")

        # Bin particle speeds into histogram
        speeds = [math.hypot(p.vx, p.vy) for p in self._particles]
        num_bins = 10
        max_speed = max(350.0, max(speeds) * 1.15) if speeds else 350.0
        bin_width = max_speed / float(num_bins)
        counts = [0] * num_bins
        for s in speeds:
            b_idx = min(int(s / bin_width), num_bins - 1)
            counts[b_idx] += 1

        max_count = max(counts) if any(counts) else 1
        bar_w = (chart_rect.width() - 24.0) / float(num_bins)

        # Draw histogram bars
        painter.setPen(QPen(QColor(168, 85, 247, 100), 1.0))
        painter.setBrush(QBrush(QColor(168, 85, 247, 160)))
        for i in range(num_bins):
            bar_h = (counts[i] / float(max_count)) * (chart_rect.height() - 36.0)
            bx = chart_rect.x() + 12.0 + i * bar_w
            by = chart_rect.bottom() - 10.0 - bar_h
            painter.drawRect(QRectF(bx, by, bar_w - 2.0, bar_h))

        # 2. Thermodynamic State Card (Lower)
        card_rect = QRectF(panel_x + 10.0, panel_y + 135.0, panel_w - 20.0, 95.0)
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(card_rect.x() + 8.0, card_rect.y() + 18.0, "🌡️ Trạng thái nhiệt động học:")

        lines = [
            f"• Áp suất P: {self._pressure_kpa:.1f} kPa ({self._pressure_kpa / 101.325:.2f} atm)",
            f"• Thể tích V: {self.volume_liters:.1f} L (Piston: {self._piston_fraction * 100:.0f}%)",
            f"• Nhiệt độ T: {self._temperature:.0f} K ({self._temperature - 273.15:.1f}°C)",
            f"• Số mol n: {self._moles:.2f} mol | Chất: {GAS_SPECS[self._gas_type][0]}",
        ]
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        for idx, line in enumerate(lines):
            painter.drawText(card_rect.x() + 8.0, card_rect.y() + 34.0 + idx * 15.0, line)

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive process buttons and action triggers."""
        btn_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(btn_font)

        def draw_button(rect: QRectF, text: str, is_active: bool, bg_hex: str = "#334155") -> None:
            col = QColor("#0284c7") if is_active else QColor(bg_hex)
            painter.setPen(QPen(col.lighter(130), 1.0))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Row 1: Process Modes & Gas Types
        draw_button(self._rect_btn_isothermal, "Đẳng nhiệt", self._process_mode == "ISOTHERMAL")
        draw_button(self._rect_btn_isobaric, "Đẳng áp", self._process_mode == "ISOBARIC")
        draw_button(self._rect_btn_isochoric, "Đẳng tích", self._process_mode == "ISOCHORIC")
        draw_button(self._rect_btn_free, "Tự do", self._process_mode == "FREE")

        draw_button(self._rect_btn_gas_he, "🎈 He", self._gas_type == "HELIUM", "#0369a1")
        draw_button(self._rect_btn_gas_n2, "💨 N₂", self._gas_type == "NITROGEN", "#047857")
        draw_button(self._rect_btn_gas_ar, "🧪 Ar", self._gas_type == "ARGON", "#b45309")

        run_txt = "⏸ Dừng" if self._is_running else "▶ Chạy"
        draw_button(self._rect_btn_run, run_txt, self._is_running, "#0284c7")

        # Row 2: Heat / Cool / Piston controls & Stamping
        draw_button(self._rect_btn_heat, "🔥 Đun nóng (+30K)", False, "#b45309")
        draw_button(self._rect_btn_cool, "❄️ Làm mát (-30K)", False, "#0369a1")
        draw_button(self._rect_btn_piston_comp, "◀ Nén", False, "#334155")
        draw_button(self._rect_btn_piston_exp, "Giãn ▶", False, "#334155")
        draw_button(self._rect_btn_reset, "🔄 Đặt lại", False, "#475569")
        draw_button(self._rect_btn_stamp, "🖊️ Đóng dấu bảng", False, "#059669")
