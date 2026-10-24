"""
AirTeach AI — Interactive Physical Chemistry & Thermodynamics: Phase Diagrams Simulator.
Simulates thermodynamic P-T phase equilibrium diagrams, Clausius-Clapeyron vapor pressure,
sublimation, anomalous negative melting slope of Water (Ice Ih), triple points, critical points,
supercritical fluid phase transitions, and microscopic particle lattice dynamics.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
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
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class PhasePreset:
    code: str
    title_vi: str
    badge_label: str
    substance_name: str
    formula: str
    t_triple_c: float        # °C
    p_triple_atm: float      # atm
    t_critical_c: float      # °C
    p_critical_atm: float    # atm
    t_normal_melt_c: float   # °C at 1 atm (None if P_triple > 1 atm)
    t_normal_boil_c: float   # °C at 1 atm (None if P_triple > 1 atm)
    t_sublime_1atm_c: Optional[float]  # °C if sublimes at 1 atm
    melting_slope_negative: bool       # True for Water (density ice < water)
    delta_h_vap_kj_mol: float          # Heat of vaporization kJ/mol
    delta_h_sub_kj_mol: float          # Heat of sublimation kJ/mol
    description: str
    temp_range_c: tuple[float, float]  # (min_T, max_T) for chart
    press_range_log: tuple[float, float] # (min_log10_P, max_log10_P) in atm


PHASE_PRESETS: dict[str, PhasePreset] = {
    "WATER_H2O_ANOMALOUS_MELTING": PhasePreset(
        code="WATER_H2O_ANOMALOUS_MELTING",
        title_vi="Nước (H₂O) — Độ dốc Nóng chảy Âm Bất thường & Điểm Tam giác",
        badge_label="H₂O ANOMALOUS",
        substance_name="Nước",
        formula="H₂O",
        t_triple_c=0.01,
        p_triple_atm=0.006,
        t_critical_c=373.95,
        p_critical_atm=217.75,
        t_normal_melt_c=0.0,
        t_normal_boil_c=100.0,
        t_sublime_1atm_c=None,
        melting_slope_negative=True,
        delta_h_vap_kj_mol=40.66,
        delta_h_sub_kj_mol=51.06,
        description="Độ dốc nóng chảy dP/dT < 0 do mạng tinh thể băng rỗng có thể tích riêng lớn hơn nước lỏng (khối lượng riêng nước đá 0.917 g/cm³ < nước 1.0 g/cm³). Nén áp suất làm băng tan chảy (hiện tượng trượt băng).",
        temp_range_c=(-60.0, 420.0),
        press_range_log=(-4.0, 3.0),
    ),
    "CARBON_DIOXIDE_DRY_ICE_SUBLIMATION": PhasePreset(
        code="CARBON_DIOXIDE_DRY_ICE_SUBLIMATION",
        title_vi="Khí Cacbonic (CO₂) — Thăng hoa Băng khô & Điểm Tam giác > 1 atm",
        badge_label="CO₂ DRY ICE",
        substance_name="Khí Cacbonic",
        formula="CO₂",
        t_triple_c=-56.6,
        p_triple_atm=5.11,
        t_critical_c=31.1,
        p_critical_atm=72.8,
        t_normal_melt_c=-56.6,  # not at 1 atm
        t_normal_boil_c=None,
        t_sublime_1atm_c=-78.5,
        melting_slope_negative=False,
        delta_h_vap_kj_mol=16.6,
        delta_h_sub_kj_mol=25.2,
        description="Điểm tam giác P_t = 5.11 atm > 1 atm khiến CO₂ lỏng không thể tồn tại ở áp suất khí quyển. Băng khô thăng hoa trực tiếp từ rắn sang khí ở -78.5°C tại 1 atm.",
        temp_range_c=(-120.0, 60.0),
        press_range_log=(-2.0, 3.0),
    ),
    "ARGON_NOBLE_GAS_CLASSICAL": PhasePreset(
        code="ARGON_NOBLE_GAS_CLASSICAL",
        title_vi="Khí hiếm Argon (Ar) — Mô hình Khí lý tưởng Van der Waals Cổ điển",
        badge_label="ARGON CLASSICAL",
        substance_name="Argon",
        formula="Ar",
        t_triple_c=-189.35,  # 83.80 K
        p_triple_atm=0.68,
        t_critical_c=-122.29, # 150.86 K
        p_critical_atm=48.34,
        t_normal_melt_c=-189.3,
        t_normal_boil_c=-185.8,
        t_sublime_1atm_c=None,
        melting_slope_negative=False,
        delta_h_vap_kj_mol=6.5,
        delta_h_sub_kj_mol=7.7,
        description="Chất kinh điển theo tương tác Lennard-Jones không phân cực. Đường nóng chảy dP/dT > 0 tiêu chuẩn, khoảng chất lỏng tại 1 atm hẹp chỉ 3.5°C.",
        temp_range_c=(-220.0, -100.0),
        press_range_log=(-2.0, 2.5),
    ),
    "SULFUR_ALLOTROPIC_POLYMORPHISM": PhasePreset(
        code="SULFUR_ALLOTROPIC_POLYMORPHISM",
        title_vi="Lưu huỳnh (S) — Đa hình Đa thù hình & 4 Pha Cân bằng",
        badge_label="SULFUR ALLOTROPY",
        substance_name="Lưu huỳnh",
        formula="S₈",
        t_triple_c=115.2,
        p_triple_atm=0.00002,
        t_critical_c=1041.0,
        p_critical_atm=180.0,
        t_normal_melt_c=115.2,
        t_normal_boil_c=444.6,
        t_sublime_1atm_c=None,
        melting_slope_negative=False,
        delta_h_vap_kj_mol=45.0,
        delta_h_sub_kj_mol=54.0,
        description="Đặc trưng chuyển pha thù hình rắn-rắn giữa Lưu huỳnh Tà phương (Rhombic α-S) và Đơn tà (Monoclinic β-S) ở 95.6°C, tạo ra nhiều điểm tam giác độc đáo.",
        temp_range_c=(20.0, 500.0),
        press_range_log=(-5.0, 2.5),
    ),
    "SUPERCRITICAL_EXTRACTION_INDUSTRIAL": PhasePreset(
        code="SUPERCRITICAL_EXTRACTION_INDUSTRIAL",
        title_vi="Chất lưu Siêu tới hạn CO₂ — Ứng dụng Chiết xuất Cà phê & Dược liệu",
        badge_label="SUPERCRITICAL CO₂",
        substance_name="CO₂ Siêu tới hạn",
        formula="scCO₂",
        t_triple_c=-56.6,
        p_triple_atm=5.11,
        t_critical_c=31.1,
        p_critical_atm=72.8,
        t_normal_melt_c=-56.6,
        t_normal_boil_c=None,
        t_sublime_1atm_c=-78.5,
        melting_slope_negative=False,
        delta_h_vap_kj_mol=16.6,
        delta_h_sub_kj_mol=25.2,
        description="Vùng siêu tới hạn T > 31.1°C, P > 72.8 atm kết hợp độ nhớt thấp và khuếch tán nhanh như chất khí với khả năng hòa tan cao như chất lỏng, không để lại cặn độc hại.",
        temp_range_c=(-80.0, 80.0),
        press_range_log=(-1.0, 3.0),
    ),
}


class InteractivePhaseDiagramItem(QGraphicsObject):
    """
    Interactive P-T Phase Diagram and Microscopic Matter Simulation.
    Provides real-time Clausius-Clapeyron thermodynamic boundaries,
    phase identification (Solid, Liquid, Gas, Supercritical), anomalous melting slopes,
    and microscopic particle dynamics.
    """

    preset_changed = Signal(str)
    state_changed = Signal(str, float, float)  # phase_name, T_c, P_atm
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Active Preset
        self._active_preset_code: str = "WATER_H2O_ANOMALOUS_MELTING"
        self._preset: PhasePreset = PHASE_PRESETS[self._active_preset_code]

        # Thermodynamic State Coordinates
        self._temp_c: float = 25.0       # Current temperature °C
        self._press_atm: float = 1.0     # Current pressure atm

        # Animation & Trajectory Tracking
        self._trajectory_mode: str = "MANUAL"  # "MANUAL", "ISOBARIC_HEATING", "ISOTHERMAL_COMPRESSION", "SUPERCRITICAL_CYCLE"
        self._traj_step: int = 0
        self._is_animating: bool = True

        # Microscopic Particle System (24 particles)
        self._particles: list[dict] = []
        self._init_particles()

        # UI & Touchless Gestures
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 60 Hz physics animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> PhasePreset:
        return self._preset

    @property
    def temperature_c(self) -> float:
        return self._temp_c

    @property
    def pressure_atm(self) -> float:
        return self._press_atm

    @property
    def current_phase(self) -> str:
        return self.determine_phase(self._temp_c, self._press_atm)

    # -------------------------------------------------------------------------
    # Thermodynamic Phase Determination & Boundaries
    # -------------------------------------------------------------------------
    def vapor_pressure_clausius_clapeyron(self, t_c: float) -> float:
        """
        Calculates equilibrium vapor pressure P_vap (atm) at temperature t_c using
        integrated Clausius-Clapeyron equation:
        ln(P/P_ref) = -(ΔH_vap / R) * (1/T - 1/T_ref)
        Anchored at normal boiling point (1.0 atm) if available, otherwise triple point.
        """
        r_gas = 8.314462  # J/(mol*K)
        t_k = t_c + 273.15
        if t_k <= 0:
            return 1e-10

        if self._preset.t_normal_boil_c is not None:
            t_ref_k = self._preset.t_normal_boil_c + 273.15
            p_ref_atm = 1.0
        else:
            t_ref_k = self._preset.t_triple_c + 273.15
            p_ref_atm = self._preset.p_triple_atm

        dh_j = self._preset.delta_h_vap_kj_mol * 1000.0
        exponent = -(dh_j / r_gas) * (1.0 / t_k - 1.0 / t_ref_k)
        # Cap exponent to avoid overflow
        exponent = max(-25.0, min(15.0, exponent))
        return p_ref_atm * math.exp(exponent)

    def sublimation_pressure(self, t_c: float) -> float:
        """
        Calculates equilibrium sublimation pressure P_sub (atm) below triple point:
        ln(P/P_t) = -(ΔH_sub / R) * (1/T - 1/T_t)
        """
        r_gas = 8.314462
        t_k = t_c + 273.15
        t_t_k = self._preset.t_triple_c + 273.15
        if t_k <= 0 or t_t_k <= 0:
            return 1e-10

        dh_j = self._preset.delta_h_sub_kj_mol * 1000.0
        exponent = -(dh_j / r_gas) * (1.0 / t_k - 1.0 / t_t_k)
        exponent = max(-25.0, min(15.0, exponent))
        return self._preset.p_triple_atm * math.exp(exponent)

    def melting_temperature_at_p(self, p_atm: float) -> float:
        """
        Calculates melting temperature Tm (°C) at given pressure P (atm).
        For water: negative slope Tm decreases with pressure (regelation).
        For others: positive slope Tm increases with pressure.
        """
        p_t = self._preset.p_triple_atm
        t_t = self._preset.t_triple_c
        delta_p = p_atm - p_t

        if self._preset.melting_slope_negative:
            # Water: ~ -0.0075 °C / atm
            return t_t - 0.0075 * delta_p
        else:
            # Normal: ~ +0.025 °C / atm
            return t_t + 0.025 * delta_p

    def determine_phase(self, t_c: float, p_atm: float) -> str:
        """
        Determines current thermodynamic phase:
        - "RẮN (SOLID)"
        - "LỎNG (LIQUID)"
        - "KHÍ (GAS / VAPOR)"
        - "CHẤT LƯU SIÊU TỚI HẠN (SUPERCRITICAL FLUID)"
        """
        p_crit = self._preset.p_critical_atm
        t_crit = self._preset.t_critical_c
        p_trip = self._preset.p_triple_atm
        t_trip = self._preset.t_triple_c

        # Supercritical check
        if t_c >= t_crit and p_atm >= p_crit:
            return "CHẤT LƯU SIÊU TỚI HẠN (SUPERCRITICAL FLUID)"

        # Below triple point pressure
        if p_atm < p_trip:
            p_sub = self.sublimation_pressure(t_c)
            if p_atm > p_sub:
                return "RẮN (SOLID)"
            else:
                return "KHÍ (GAS / VAPOR)"

        # Above triple point pressure
        t_melt = self.melting_temperature_at_p(p_atm)

        if t_c < t_melt:
            return "RẮN (SOLID)"

        # To the right of melting curve
        if t_c > t_crit:
            return "KHÍ DÃN NỞ (SUPERHEATED GAS)"

        p_vap = self.vapor_pressure_clausius_clapeyron(t_c)
        if p_atm > p_vap:
            return "LỎNG (LIQUID)"
        else:
            return "KHÍ (GAS / VAPOR)"

    # -------------------------------------------------------------------------
    # Microscopic Particle System Initialization & Simulation
    # -------------------------------------------------------------------------
    def _init_particles(self) -> None:
        """Initializes 24 microscopic molecules in a lattice for rendering."""
        self._particles.clear()
        cols = 6
        rows = 4
        spacing_x = 22.0
        spacing_y = 22.0
        base_x = 35.0
        base_y = 35.0

        for r in range(rows):
            for c in range(cols):
                orig_x = base_x + c * spacing_x
                orig_y = base_y + r * spacing_y
                self._particles.append({
                    "orig_x": orig_x,
                    "orig_y": orig_y,
                    "x": orig_x,
                    "y": orig_y,
                    "vx": (random.random() - 0.5) * 2.0,
                    "vy": (random.random() - 0.5) * 2.0,
                    "phase_jitter": random.random() * 2.0 * math.pi,
                })

    def _update_particles(self) -> None:
        """Updates particle positions based on current state (Solid, Liquid, Gas, Supercritical)."""
        phase = self.current_phase
        t_c = self._temp_c
        t_k = max(10.0, t_c + 273.15)
        thermal_speed = math.sqrt(t_k / 300.0)

        box_w = 160.0
        box_h = 130.0
        box_x = 15.0
        box_y = 15.0

        for p in self._particles:
            if "RẮN" in phase:
                # Solid: localized harmonic vibration around lattice origin
                p["phase_jitter"] += 0.35 * thermal_speed
                amp = 1.8 * min(3.0, thermal_speed)
                p["x"] = p["orig_x"] + amp * math.sin(p["phase_jitter"])
                p["y"] = p["orig_y"] + amp * math.cos(p["phase_jitter"] * 1.3)

            elif "LỎNG" in phase:
                # Liquid: dense rolling fluid, wandering with surface constraint
                p["x"] += p["vx"] * thermal_speed * 0.7
                p["y"] += p["vy"] * thermal_speed * 0.7
                # Liquid gathers toward bottom half
                min_y = box_y + box_h * 0.35
                max_y = box_y + box_h - 10.0
                min_x = box_x + 10.0
                max_x = box_x + box_w - 10.0

                if p["x"] < min_x or p["x"] > max_x:
                    p["vx"] *= -1.0
                    p["x"] = max(min_x, min(max_x, p["x"]))
                if p["y"] < min_y or p["y"] > max_y:
                    p["vy"] *= -1.0
                    p["y"] = max(min_y, min(max_y, p["y"]))

            elif "KHÍ" in phase:
                # Gas: fast, free ballistic motion bouncing off all walls
                p["x"] += p["vx"] * thermal_speed * 2.2
                p["y"] += p["vy"] * thermal_speed * 2.2
                min_x = box_x + 8.0
                max_x = box_x + box_w - 8.0
                min_y = box_y + 8.0
                max_y = box_y + box_h - 8.0

                if p["x"] < min_x or p["x"] > max_x:
                    p["vx"] *= -1.0
                    p["x"] = max(min_x, min(max_x, p["x"]))
                if p["y"] < min_y or p["y"] > max_y:
                    p["vy"] *= -1.0
                    p["y"] = max(min_y, min(max_y, p["y"]))

            else:
                # Supercritical fluid: high density chaotic clustering with high velocity
                p["x"] += p["vx"] * thermal_speed * 1.5
                p["y"] += p["vy"] * thermal_speed * 1.5
                min_x = box_x + 6.0
                max_x = box_x + box_w - 6.0
                min_y = box_y + 6.0
                max_y = box_y + box_h - 6.0

                if p["x"] < min_x or p["x"] > max_x:
                    p["vx"] *= -1.0
                    p["x"] = max(min_x, min(max_x, p["x"]))
                if p["y"] < min_y or p["y"] > max_y:
                    p["vy"] *= -1.0
                    p["y"] = max(min_y, min(max_y, p["y"]))

    def _on_sim_tick(self) -> None:
        """Simulation tick for particle physics and automated trajectories."""
        self._update_particles()

        if self._is_animating and self._trajectory_mode != "MANUAL":
            self._advance_trajectory()

        self.update()

    def _advance_trajectory(self) -> None:
        """Advances automated thermodynamic phase change trajectories."""
        self._traj_step += 1
        cycle = self._traj_step % 200

        if self._trajectory_mode == "ISOBARIC_HEATING":
            # P = 1 atm, heating from -40°C to 140°C then back
            self._press_atm = 1.0
            if cycle < 100:
                self._temp_c = -40.0 + (180.0 * (cycle / 100.0))
            else:
                self._temp_c = 140.0 - (180.0 * ((cycle - 100) / 100.0))

        elif self._trajectory_mode == "ISOTHERMAL_COMPRESSION":
            # T = 20°C, compressing from 0.001 atm to 100 atm
            self._temp_c = 20.0
            if cycle < 100:
                log_p = -3.0 + 5.0 * (cycle / 100.0)
            else:
                log_p = 2.0 - 5.0 * ((cycle - 100) / 100.0)
            self._press_atm = math.pow(10.0, log_p)

        elif self._trajectory_mode == "SUPERCRITICAL_CYCLE":
            # Cycle through: Liquid -> Supercritical -> Gas -> Liquid
            if cycle < 50:
                self._temp_c = 10.0 + (35.0 * (cycle / 50.0))
                self._press_atm = 85.0
            elif cycle < 100:
                self._temp_c = 45.0
                self._press_atm = 85.0 - (70.0 * ((cycle - 50) / 50.0))
            elif cycle < 150:
                self._temp_c = 45.0 - (40.0 * ((cycle - 100) / 50.0))
                self._press_atm = 15.0
            else:
                self._temp_c = 5.0
                self._press_atm = 15.0 + (70.0 * ((cycle - 150) / 50.0))

    # -------------------------------------------------------------------------
    # Presets & Interactive Steppers
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a thermodynamic substance phase preset."""
        if preset_code not in PHASE_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PHASE_PRESETS[preset_code]
        # Set sensible initial coordinates
        if preset_code == "WATER_H2O_ANOMALOUS_MELTING":
            self._temp_c = 25.0
            self._press_atm = 1.0
        elif preset_code == "CARBON_DIOXIDE_DRY_ICE_SUBLIMATION":
            self._temp_c = -78.5
            self._press_atm = 1.0
        elif preset_code == "ARGON_NOBLE_GAS_CLASSICAL":
            self._temp_c = -187.0
            self._press_atm = 1.0
        elif preset_code == "SULFUR_ALLOTROPIC_POLYMORPHISM":
            self._temp_c = 115.0
            self._press_atm = 1.0
        elif preset_code == "SUPERCRITICAL_EXTRACTION_INDUSTRIAL":
            self._temp_c = 35.0
            self._press_atm = 80.0

        self._trajectory_mode = "MANUAL"
        self._init_particles()
        self.preset_changed.emit(preset_code)
        self.update()

    def set_temperature_pressure(self, temp_c: float, press_atm: float) -> None:
        """Sets temperature and pressure explicitly with boundary safety."""
        t_min, t_max = self._preset.temp_range_c
        p_log_min, p_log_max = self._preset.press_range_log
        self._temp_c = max(t_min, min(t_max, temp_c))
        log_p = math.log10(max(1e-6, press_atm))
        clamped_log = max(p_log_min, min(p_log_max, log_p))
        self._press_atm = math.pow(10.0, clamped_log)
        self.update()

    def step_temperature(self, delta_c: float) -> None:
        """Steps temperature by delta °C."""
        self.set_temperature_pressure(self._temp_c + delta_c, self._press_atm)

    def step_pressure_log(self, delta_log: float) -> None:
        """Multiplies or divides pressure by factor 10^delta_log."""
        log_p = math.log10(max(1e-6, self._press_atm))
        self.set_temperature_pressure(self._temp_c, math.pow(10.0, log_p + delta_log))

    def toggle_animation(self) -> None:
        """Toggles automated simulation/trajectory execution."""
        self._is_animating = not self._is_animating
        self.update()

    # -------------------------------------------------------------------------
    # UI Layout & Touchless Air Pinch Handling
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_t_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_t_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_p_down"] = QRectF(290.0, bar_y, 48.0, 34.0)
        self._buttons["btn_p_up"] = QRectF(343.0, bar_y, 48.0, 34.0)
        self._buttons["btn_mode"] = QRectF(398.0, bar_y, 140.0, 34.0)
        self._buttons["btn_p_co2"] = QRectF(546.0, bar_y, 82.0, 34.0)
        self._buttons["btn_p_sc"] = QRectF(634.0, bar_y, 82.0, 34.0)
        self._buttons["btn_play"] = QRectF(722.0, bar_y, 85.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(813.0, bar_y, 115.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        """Handles click on UI button keys."""
        preset_keys = list(PHASE_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_t_down":
            self.step_temperature(-10.0)
        elif key == "btn_t_up":
            self.step_temperature(10.0)
        elif key == "btn_p_down":
            self.step_pressure_log(-0.5)
        elif key == "btn_p_up":
            self.step_pressure_log(0.5)
        elif key == "btn_mode":
            modes = ["MANUAL", "ISOBARIC_HEATING", "ISOTHERMAL_COMPRESSION", "SUPERCRITICAL_CYCLE"]
            c_idx = modes.index(self._trajectory_mode)
            self._trajectory_mode = modes[(c_idx + 1) % len(modes)]
            self._traj_step = 0
        elif key == "btn_p_co2":
            self.load_preset("CARBON_DIOXIDE_DRY_ICE_SUBLIMATION")
        elif key == "btn_p_sc":
            self.load_preset("SUPERCRITICAL_EXTRACTION_INDUSTRIAL")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
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
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "substance_name": self._preset.substance_name,
            "formula": self._preset.formula,
            "temp_c": self._temp_c,
            "press_atm": self._press_atm,
            "current_phase": self.current_phase,
            "t_triple": self._preset.t_triple_c,
            "p_triple": self._preset.p_triple_atm,
            "t_crit": self._preset.t_critical_c,
            "p_crit": self._preset.p_critical_atm,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_phase_diagram_chart(painter)
        self._paint_microscopic_panel(painter)
        self._paint_thermodynamic_metrics(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(11, 15, 25, 252))
        grad.setColorAt(1.0, QColor(8, 12, 20, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"NHIỆT ĐỘNG HỌC & GIẢN ĐỒ PHA: {self._preset.title_vi.upper()}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(22, 48, f"Công thức: {self._preset.formula} | Điểm tam giác: ({self._preset.t_triple_c:.2f}°C, {self._preset.p_triple_atm:.4f} atm) | Điểm tới hạn: ({self._preset.t_critical_c:.1f}°C, {self._preset.p_critical_atm:.1f} atm)")

    def _paint_phase_diagram_chart(self, painter: QPainter) -> None:
        """Paints logarithmic P-T phase diagram chart with authentic boundaries."""
        cx = 20.0
        cy = 60.0
        cw = 610.0
        ch = 440.0

        # Background frame
        painter.setBrush(QBrush(QColor(4, 7, 14)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(cx, cy, cw, ch, 8.0)

        # Coordinate mappings
        t_min, t_max = self._preset.temp_range_c
        p_log_min, p_log_max = self._preset.press_range_log

        def t_to_x(t: float) -> float:
            ratio = (t - t_min) / (t_max - t_min)
            return cx + 55.0 + ratio * (cw - 80.0)

        def p_to_y(p: float) -> float:
            log_p = math.log10(max(1e-6, p))
            ratio = (log_p - p_log_min) / (p_log_max - p_log_min)
            return cy + ch - 40.0 - ratio * (ch - 65.0)

        # Draw Grid & Axis Labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(51, 65, 85), 0.8, Qt.PenStyle.DashLine))

        # Horizontal log P lines: 0.001, 0.01, 0.1, 1, 10, 100, 1000
        p_ticks = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for pt in p_ticks:
            log_p = math.log10(pt)
            if p_log_min <= log_p <= p_log_max:
                y = p_to_y(pt)
                painter.drawLine(cx + 50.0, y, cx + cw - 15.0, y)
                painter.setPen(QPen(QColor(148, 163, 184)))
                painter.drawText(cx + 6.0, y + 4.0, f"{pt:g} atm")
                painter.setPen(QPen(QColor(51, 65, 85), 0.8, Qt.PenStyle.DashLine))

        # 1 atm Isobar line (Highlight)
        y_1atm = p_to_y(1.0)
        painter.setPen(QPen(QColor(245, 158, 11, 160), 1.4, Qt.PenStyle.DotLine))
        painter.drawLine(cx + 50.0, y_1atm, cx + cw - 15.0, y_1atm)
        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.drawText(cx + cw - 70.0, y_1atm - 4.0, "1 atm (Chuẩn)")

        # Vertical Temperature lines
        t_step = 50.0 if (t_max - t_min) > 200.0 else 20.0
        curr_t = math.ceil(t_min / t_step) * t_step
        while curr_t <= t_max:
            x = t_to_x(curr_t)
            painter.setPen(QPen(QColor(51, 65, 85), 0.8, Qt.PenStyle.DashLine))
            painter.drawLine(x, cy + 20.0, x, cy + ch - 35.0)
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(x - 14.0, cy + ch - 18.0, f"{int(curr_t)}°C")
            curr_t += t_step

        # Key Points
        t_trip = self._preset.t_triple_c
        p_trip = self._preset.p_triple_atm
        t_crit = self._preset.t_critical_c
        p_crit = self._preset.p_critical_atm

        x_trip, y_trip = t_to_x(t_trip), p_to_y(p_trip)
        x_crit, y_crit = t_to_x(t_crit), p_to_y(p_crit)

        # Draw Phase Regions Shading
        # 1. Solid Region (left of melting + sublimation)
        solid_path = QPainterPath()
        solid_path.moveTo(t_to_x(t_min), p_to_y(math.pow(10, p_log_max)))
        # Melting line down to triple point
        solid_path.lineTo(x_trip, y_trip)
        # Sublimation line down to t_min
        for step in range(25):
            t_s = t_trip - (step / 24.0) * (t_trip - t_min)
            p_s = self.sublimation_pressure(t_s)
            solid_path.lineTo(t_to_x(t_s), p_to_y(p_s))
        solid_path.lineTo(t_to_x(t_min), p_to_y(math.pow(10, p_log_min)))
        solid_path.closeSubpath()
        painter.fillPath(solid_path, QBrush(QColor(30, 58, 95, 80)))

        # 2. Liquid Region (between melting, vaporization, and critical point)
        liquid_path = QPainterPath()
        liquid_path.moveTo(x_trip, y_trip)
        # Melting curve up
        liquid_path.lineTo(t_to_x(self.melting_temperature_at_p(math.pow(10, p_log_max))), p_to_y(math.pow(10, p_log_max)))
        # Top boundary to critical temperature
        liquid_path.lineTo(x_crit, p_to_y(math.pow(10, p_log_max)))
        liquid_path.lineTo(x_crit, y_crit)
        # Vaporization curve back to triple point
        for step in range(25, -1, -1):
            t_v = t_trip + (step / 25.0) * (t_crit - t_trip)
            p_v = self.vapor_pressure_clausius_clapeyron(t_v)
            liquid_path.lineTo(t_to_x(t_v), p_to_y(p_v))
        liquid_path.closeSubpath()
        painter.fillPath(liquid_path, QBrush(QColor(15, 76, 92, 90)))

        # 3. Supercritical Fluid Region (T > T_crit and P > P_crit)
        super_path = QPainterPath()
        super_path.moveTo(x_crit, y_crit)
        super_path.lineTo(t_to_x(t_max), y_crit)
        super_path.lineTo(t_to_x(t_max), p_to_y(math.pow(10, p_log_max)))
        super_path.lineTo(x_crit, p_to_y(math.pow(10, p_log_max)))
        super_path.closeSubpath()
        painter.fillPath(super_path, QBrush(QColor(59, 29, 84, 100)))

        # Draw Phase Boundary Curves
        # Sublimation Curve (Solid - Gas): Amber
        sub_path = QPainterPath()
        sub_path.moveTo(t_to_x(t_min), p_to_y(self.sublimation_pressure(t_min)))
        for step in range(1, 30):
            t_s = t_min + (step / 29.0) * (t_trip - t_min)
            sub_path.lineTo(t_to_x(t_s), p_to_y(self.sublimation_pressure(t_s)))
        painter.setPen(QPen(QColor(245, 158, 11), 2.2))
        painter.drawPath(sub_path)

        # Vaporization Curve (Liquid - Gas): Cyan ending at Critical Point
        vap_path = QPainterPath()
        vap_path.moveTo(x_trip, y_trip)
        for step in range(1, 30):
            t_v = t_trip + (step / 29.0) * (t_crit - t_trip)
            vap_path.lineTo(t_to_x(t_v), p_to_y(self.vapor_pressure_clausius_clapeyron(t_v)))
        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(vap_path)

        # Melting Curve (Solid - Liquid): Emerald Green
        # Highlights negative slope for water, positive for others
        melt_path = QPainterPath()
        melt_path.moveTo(x_trip, y_trip)
        p_top = math.pow(10.0, p_log_max)
        t_top = self.melting_temperature_at_p(p_top)
        melt_path.lineTo(t_to_x(t_top), p_to_y(p_top))
        painter.setPen(QPen(QColor(16, 185, 129), 2.5))
        painter.drawPath(melt_path)

        # Labels on Diagram
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        # Solid Label
        painter.setPen(QPen(QColor(147, 197, 253)))
        painter.drawText(t_to_x(t_min + 15.0), p_to_y(10.0), "RẮN (SOLID)")

        # Liquid Label
        painter.setPen(QPen(QColor(45, 212, 191)))
        painter.drawText(t_to_x((t_trip + t_crit) / 2.0 - 20.0), p_to_y(max(15.0, p_trip * 5.0)), "LỎNG (LIQUID)")

        # Gas Label
        painter.setPen(QPen(QColor(253, 186, 116)))
        painter.drawText(t_to_x((t_trip + t_crit) / 2.0), p_to_y(min(0.05, p_trip * 0.2)), "KHÍ (GAS / VAPOR)")

        # Supercritical Label
        painter.setPen(QPen(QColor(216, 180, 254)))
        painter.drawText(x_crit + 15.0, y_crit - 25.0, "CHẤT LƯU SIÊU TỚI HẠN\n(SUPERCRITICAL FLUID)")

        # Triple Point Marker
        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(x_trip, y_trip), 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(x_trip + 8.0, y_trip + 4.0, f"Điểm Tam giác T ({t_trip:g}°C, {p_trip:g} atm)")

        # Critical Point Marker
        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(x_crit, y_crit), 6.0, 6.0)
        painter.drawText(x_crit - 45.0, y_crit - 10.0, f"Điểm Tới hạn C ({t_crit:g}°C, {p_crit:g} atm)")

        # Current State Indicator Dot (Red Glowing Pointer)
        curr_x = t_to_x(self._temp_c)
        curr_y = p_to_y(self._press_atm)

        painter.setBrush(QBrush(QColor(239, 68, 68, 60)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(curr_x, curr_y), 12.0, 12.0)

        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.setPen(QPen(QColor(255, 255, 255), 2.0))
        painter.drawEllipse(QPointF(curr_x, curr_y), 5.5, 5.5)

    def _paint_microscopic_panel(self, painter: QPainter) -> None:
        """Paints microscopic particle arrangement viewport (Solid lattice vs Fluid vs Gas)."""
        mx = 645.0
        my = 60.0
        mw = 285.0
        mh = 190.0

        # Viewport background
        painter.setBrush(QBrush(QColor(4, 7, 14)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "CẤU TRÚC VI MÔ PHÂN TỬ:")

        # Container Box for Particles
        box_x = mx + 15.0
        box_y = my + 30.0
        box_w = 255.0
        box_h = 145.0
        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 6.0)

        # Draw molecules
        phase = self.current_phase
        if "RẮN" in phase:
            mol_color = QColor(147, 197, 253)
        elif "LỎNG" in phase:
            mol_color = QColor(45, 212, 191)
        elif "KHÍ" in phase:
            mol_color = QColor(253, 186, 116)
        else:
            mol_color = QColor(216, 180, 254)

        scale_x = (box_w - 20.0) / 160.0
        scale_y = (box_h - 20.0) / 130.0

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        for p in self._particles:
            px = box_x + 10.0 + (p["x"] - 15.0) * scale_x
            py = box_y + 10.0 + (p["y"] - 15.0) * scale_y
            painter.setBrush(QBrush(mol_color))
            painter.drawEllipse(QPointF(px, py), 5.0, 5.0)

        # State overlay badge
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(mol_color))
        painter.drawText(box_x + 10.0, box_y + box_h - 10.0, f"Trạng thái: {phase.split(' ')[0]}")

    def _paint_thermodynamic_metrics(self, painter: QPainter) -> None:
        """Paints analytical state readouts and Clausius-Clapeyron values."""
        rx = 645.0
        ry = 260.0
        rw = 285.0
        rh = 240.0

        painter.setBrush(QBrush(QColor(4, 7, 14)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(rx, ry, rw, rh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(rx + 12.0, ry + 22.0, "THÔNG SỐ TRẠNG THÁI HIỆN TẠI:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(226, 232, 240)))
        t_k = self._temp_c + 273.15
        p_atm = self._press_atm
        p_bar = p_atm * 1.01325
        p_kpa = p_atm * 101.325

        p_vap = self.vapor_pressure_clausius_clapeyron(self._temp_c)

        lines = [
            f"• Nhiệt độ: {self._temp_c:.2f} °C ({t_k:.2f} K)",
            f"• Áp suất: {p_atm:g} atm ({p_bar:.3f} bar / {p_kpa:.1f} kPa)",
            f"• Pha nhiệt động: {self.current_phase}",
            f"• Áp suất hơi bão hòa Clausius-Clapeyron:",
            f"   P_vap({self._temp_c:.1f}°C) = {p_vap:.4g} atm",
            f"• Nhiệt hóa hơi ΔH_vap: {self._preset.delta_h_vap_kj_mol} kJ/mol",
            f"• Nhiệt thăng hoa ΔH_sub: {self._preset.delta_h_sub_kj_mol} kJ/mol",
            f"• Quỹ đạo mô phỏng: {self._trajectory_mode}",
        ]
        curr_y = ry + 42.0
        for line in lines:
            painter.drawText(rx + 12.0, curr_y, line)
            curr_y += 18.0

        # Description note
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        desc_rect = QRectF(rx + 12.0, curr_y + 4.0, rw - 24.0, 45.0)
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Paints interactive gesture buttons."""
        self._layout_buttons()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Preset Selector
        self._draw_btn(painter, self._buttons["btn_preset"], "btn_preset", f"Preset: {self._preset.badge_label}")

        # 2. Temperature Steppers
        self._draw_btn(painter, self._buttons["btn_t_down"], "btn_t_down", "-T")
        self._draw_btn(painter, self._buttons["btn_t_up"], "btn_t_up", "+T")

        # 3. Pressure Steppers
        self._draw_btn(painter, self._buttons["btn_p_down"], "btn_p_down", "-P")
        self._draw_btn(painter, self._buttons["btn_p_up"], "btn_p_up", "+P")

        # 4. Trajectory Mode Switcher
        mode_label = "Chế độ: Thủ công" if self._trajectory_mode == "MANUAL" else f"Q.Đạo: {self._trajectory_mode[:7]}"
        self._draw_btn(painter, self._buttons["btn_mode"], "btn_mode", mode_label)

        # 5. Quick Presets
        self._draw_btn(painter, self._buttons["btn_p_co2"], "btn_p_co2", "Băng khô CO₂")
        self._draw_btn(painter, self._buttons["btn_p_sc"], "btn_p_sc", "scCO₂ Siêu tới hạn")

        # 6. Play / Pause
        play_label = "⏸ Dừng" if self._is_animating else "▶ Tiếp tục"
        self._draw_btn(painter, self._buttons["btn_play"], "btn_play", play_label)

        # 7. Stamp Diagram
        self._draw_btn(painter, self._buttons["btn_stamp"], "btn_stamp", "📌 Chèn Vector")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, text: str) -> None:
        is_hovered = self._hovered_btn == key
        bg_col = QColor(30, 41, 59, 220) if not is_hovered else QColor(51, 65, 85, 240)
        border_col = QColor(56, 189, 248) if is_hovered else QColor(71, 85, 105)

        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.2))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor(248, 250, 252) if is_hovered else QColor(203, 213, 225)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # Mouse fallback
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        new_hover = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                new_hover = key
                break
        if new_hover != self._hovered_btn:
            self._hovered_btn = new_hover
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._handle_button_click(key)
                event.accept()
                return
        super().mousePressEvent(event)
