"""
AirTeach AI — Interactive Thermodynamics & Statistical Mechanics:
Maxwell-Boltzmann Molecular Speed Distribution, Mean Free Path & Kinetic Theory of Gases.
Milestone 149 (Task I.136).

Provides real-time interactive simulation of:
1. Maxwell-Boltzmann probability density function:
   f(v) = 4 * pi * (M / (2 * pi * R * T))^(3/2) * v^2 * exp(-M * v^2 / (2 * R * T))
2. Three characteristic speeds:
   - Most Probable Speed: v_p = sqrt(2 * R * T / M)
   - Mean / Average Speed: <v> = sqrt(8 * R * T / (pi * M))
   - Root-Mean-Square Speed: v_rms = sqrt(3 * R * T / M)
   Invariant ordering: v_p < <v> < v_rms
3. Kinetic gas container with 40-50 animated colliding molecules color-coded by kinetic energy.
4. Mean free path lambda = (k_B * T) / (sqrt(2) * pi * d^2 * P) and collision rate.
5. 5 Authentic Curriculum Presets: Helium at 300K, Nitrogen air, High-temp Hydrogen at 1000K,
   Cryogenic Xenon at 150K, and Venusian Oxygen at 737K.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsObject,
    QGraphicsItem,
)

# Fundamental Physical Constants
R_GAS_CONST: float = 8.314462618        # Universal gas constant in J/(mol·K)
K_BOLTZMANN: float = 1.380649e-23       # Boltzmann constant in J/K
N_AVOGADRO: float = 6.02214076e23       # Avogadro constant in mol^-1
EV_TO_JOULE: float = 1.602176634e-19    # Joules per electronvolt


@dataclass(frozen=True)
class GasPreset:
    """Represents a curriculum thermodynamics scenario for kinetic gas theory."""
    code: str
    title_vi: str
    gas_name: str
    chemical_formula: str
    molar_mass_kg_mol: float        # M in kg/mol (e.g. He: 0.004003 kg/mol)
    molecular_diameter_m: float     # Collision diameter d in meters
    default_temp_K: float           # Absolute temperature T in Kelvin
    default_pressure_atm: float     # Pressure P in atmospheres
    accent_color: QColor            # Visual accent color
    description: str


GAS_PRESETS: dict[str, GasPreset] = {
    "HELIUM_GAS_ROOM_TEMP_300K": GasPreset(
        code="HELIUM_GAS_ROOM_TEMP_300K",
        title_vi="Khí Heli ở Nhiệt độ Phòng 300K (Helium ⁴He)",
        gas_name="Helium (Khí hiếm đơn nguyên tử He)",
        chemical_formula="⁴He",
        molar_mass_kg_mol=0.0040026,
        molecular_diameter_m=2.1e-10,
        default_temp_K=300.0,
        default_pressure_atm=1.0,
        accent_color=QColor(0, 210, 255), # Bright cyan
        description="Khí hiếm nhẹ, đơn nguyên tử; tốc độ phân tử trung bình rất cao (v_rms ~ 1367 m/s) do khối lượng mol nhỏ.",
    ),
    "NITROGEN_AIR_EARTH_ATMOSPHERE": GasPreset(
        code="NITROGEN_AIR_EARTH_ATMOSPHERE",
        title_vi="Khí Nitơ Không khí Trái Đất 293K (Nitrogen N₂)",
        gas_name="Nitrogen (Khí Nitơ thành phần chính khí quyển)",
        chemical_formula="N₂",
        molar_mass_kg_mol=0.0280134,
        molecular_diameter_m=3.7e-10,
        default_temp_K=293.15,
        default_pressure_atm=1.0,
        accent_color=QColor(90, 160, 255), # Sky blue
        description="Thành phần chính của không khí (78%); mô hình chuẩn cho động học khí thực tế trong giảng dạy phổ thông.",
    ),
    "HYDROGEN_HIGH_TEMP_FURNACE_1000K": GasPreset(
        code="HYDROGEN_HIGH_TEMP_FURNACE_1000K",
        title_vi="Khí Hydro Nhiệt độ Cao 1000K (Lightest Gas H₂)",
        gas_name="Hydrogen (Khí Hydro lò nhiệt phân)",
        chemical_formula="H₂",
        molar_mass_kg_mol=0.002016,
        molecular_diameter_m=2.7e-10,
        default_temp_K=1000.0,
        default_pressure_atm=1.0,
        accent_color=QColor(255, 170, 0), # Thermal gold
        description="Khí nhẹ nhất vũ trụ; ở 1000K tốc độ vượt 3500 m/s giải thích cơ chế thoát khí quyển hành tinh (Jeans Escape).",
    ),
    "XENON_HEAVY_NOBLE_GAS_COLD": GasPreset(
        code="XENON_HEAVY_NOBLE_GAS_COLD",
        title_vi="Khí Xenon Nặng Lạnh 150K (Heavy Noble Gas Xe)",
        gas_name="Xenon (Khí hiếm nặng động cơ đẩy ion)",
        chemical_formula="Xe",
        molar_mass_kg_mol=0.131293,
        molecular_diameter_m=4.1e-10,
        default_temp_K=150.0,
        default_pressure_atm=1.0,
        accent_color=QColor(180, 100, 255), # Deep violet
        description="Khí hiếm nặng khối lượng phân tử lớn; đường cong phân bố rất hẹp và tập trung ở dải vận tốc thấp.",
    ),
    "OXYGEN_SURFACE_VENUS_737K": GasPreset(
        code="OXYGEN_SURFACE_VENUS_737K",
        title_vi="Khí Oxy Bề Mặt Sao Kim 737K (Venusian Oxygen O₂)",
        gas_name="Oxygen (Khí Oxy nhiệt độ cao Sao Kim)",
        chemical_formula="O₂",
        molar_mass_kg_mol=0.031999,
        molecular_diameter_m=3.6e-10,
        default_temp_K=737.0,
        default_pressure_atm=2.0,
        accent_color=QColor(255, 75, 75), # Fiery red
        description="Mô phỏng động học chất khí trong điều kiện nhiệt độ cực hạn trên bề mặt Sao Kim.",
    ),
}


class InteractiveMaxwellBoltzmannItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating Maxwell-Boltzmann
    molecular speed distribution, characteristic speeds (vp, v_avg, v_rms),
    mean free path, and kinetic gas container collisions.
    """

    # Signals
    distribution_computed = Signal(dict)
    stamp_requested = Signal(dict)

    CARD_W: float = 680.0
    CARD_H: float = 410.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "HELIUM_GAS_ROOM_TEMP_300K"
        self._preset: GasPreset = GAS_PRESETS[self._active_preset_code]

        self._temperature_K: float = self._preset.default_temp_K
        self._pressure_atm: float = self._preset.default_pressure_atm
        self._molar_mass: float = self._preset.molar_mass_kg_mol

        # 2D Kinetic Gas Container Simulation
        self._molecules: list[dict] = []
        self._init_molecules(45)

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None

        # Timer for 60fps real-time particle animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def _init_molecules(self, count: int) -> None:
        """Initializes molecules in the 2D container with random Maxwell-Boltzmann sampled speeds."""
        self._molecules.clear()
        vp = self.most_probable_speed_m_s
        for _ in range(count):
            angle = random.uniform(0.0, 2.0 * math.pi)
            # Sample approx speed around vp
            speed_ratio = random.weibullvariate(1.0, 2.0)
            self._molecules.append({
                "x": random.uniform(30.0, 310.0),
                "y": random.uniform(85.0, 260.0),
                "vx": math.cos(angle) * speed_ratio,
                "vy": math.sin(angle) * speed_ratio,
                "radius": 3.0,
            })

    # ------------------------------------------------------------------
    # Physics Properties & Analytical Laws
    # ------------------------------------------------------------------

    @property
    def most_probable_speed_m_s(self) -> float:
        """
        Most probable speed v_p = sqrt(2 * R * T / M).
        Speed corresponding to the maximum of f(v).
        """
        t = max(1.0, self._temperature_K)
        m = max(1e-4, self._molar_mass)
        return math.sqrt(2.0 * R_GAS_CONST * t / m)

    @property
    def average_speed_m_s(self) -> float:
        """
        Mean / average speed <v> = sqrt(8 * R * T / (pi * M)).
        """
        t = max(1.0, self._temperature_K)
        m = max(1e-4, self._molar_mass)
        return math.sqrt(8.0 * R_GAS_CONST * t / (math.pi * m))

    @property
    def rms_speed_m_s(self) -> float:
        """
        Root-mean-square speed v_rms = sqrt(3 * R * T / M).
        Directly related to average kinetic energy: <Ek> = 0.5 * m * v_rms^2 = 1.5 * k_B * T.
        """
        t = max(1.0, self._temperature_K)
        m = max(1e-4, self._molar_mass)
        return math.sqrt(3.0 * R_GAS_CONST * t / m)

    def pdf_f_v(self, v: float) -> float:
        """
        Maxwell-Boltzmann probability density function:
        f(v) = 4 * pi * (M / (2 * pi * R * T))^(3/2) * v^2 * exp(-M * v^2 / (2 * R * T))
        """
        if v < 0.0:
            return 0.0
        t = max(1.0, self._temperature_K)
        m = max(1e-4, self._molar_mass)
        prefactor = 4.0 * math.pi * ((m / (2.0 * math.pi * R_GAS_CONST * t)) ** 1.5)
        exponent = - (m * v * v) / (2.0 * R_GAS_CONST * t)
        if exponent < -50.0:
            return 0.0
        return prefactor * (v * v) * math.exp(exponent)

    @property
    def mean_kinetic_energy_J(self) -> float:
        """Average translational kinetic energy per molecule <Ek> = 1.5 * k_B * T (Joules)."""
        return 1.5 * K_BOLTZMANN * max(1.0, self._temperature_K)

    @property
    def mean_kinetic_energy_eV(self) -> float:
        """Average translational kinetic energy in electronvolts (eV)."""
        return self.mean_kinetic_energy_J / EV_TO_JOULE

    @property
    def mean_free_path_nm(self) -> float:
        """
        Mean free path lambda = (k_B * T) / (sqrt(2) * pi * d^2 * P) in nanometers.
        """
        t = max(1.0, self._temperature_K)
        p_pa = max(1.0, self._pressure_atm * 101325.0)
        d = self._preset.molecular_diameter_m
        denom = math.sqrt(2.0) * math.pi * (d * d) * p_pa
        return (K_BOLTZMANN * t / denom) * 1e9 if denom > 0.0 else 0.0

    @property
    def collision_frequency_GHz(self) -> float:
        """Collision frequency Z = <v> / lambda in GHz (10^9 s^-1)."""
        lam_m = self.mean_free_path_nm * 1e-9
        if lam_m <= 0.0:
            return 0.0
        return (self.average_speed_m_s / lam_m) * 1e-9

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches curriculum scenario."""
        if preset_code in GAS_PRESETS:
            self._active_preset_code = preset_code
            self._preset = GAS_PRESETS[preset_code]
            self._temperature_K = self._preset.default_temp_K
            self._pressure_atm = self._preset.default_pressure_atm
            self._molar_mass = self._preset.molar_mass_kg_mol
            self._emit_computed_values()
            self.update()

    def set_temperature_K(self, temp: float) -> None:
        """Sets temperature in Kelvin within [50.0, 2500.0] K."""
        self._temperature_K = max(50.0, min(2500.0, round(temp, 1)))
        self._emit_computed_values()
        self.update()

    def set_pressure_atm(self, p_atm: float) -> None:
        """Sets pressure within [0.1, 10.0] atmospheres."""
        self._pressure_atm = max(0.1, min(10.0, round(p_atm, 2)))
        self._emit_computed_values()
        self.update()

    def reset_defaults(self) -> None:
        """Restores default values of current preset."""
        self.set_preset(self._active_preset_code)

    def _emit_computed_values(self) -> None:
        """Emits real-time computed physics data."""
        data = {
            "preset_code": self._active_preset_code,
            "gas_name": self._preset.gas_name,
            "formula": self._preset.chemical_formula,
            "temperature_K": self._temperature_K,
            "pressure_atm": self._pressure_atm,
            "molar_mass_kg_mol": self._molar_mass,
            "vp_m_s": self.most_probable_speed_m_s,
            "v_avg_m_s": self.average_speed_m_s,
            "v_rms_m_s": self.rms_speed_m_s,
            "mean_free_path_nm": self.mean_free_path_nm,
            "collision_freq_GHz": self.collision_frequency_GHz,
        }
        self.distribution_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Simulation tick for bouncing gas particles in the 2D chamber."""
        # Container bounds
        c_min_x = 30.0
        c_max_x = 340.0
        c_min_y = 65.0
        c_max_y = 230.0

        # Scale particle animation speed with sqrt(T)
        speed_factor = math.sqrt(self._temperature_K / 300.0) * 1.8

        for m in self._molecules:
            m["x"] += m["vx"] * speed_factor
            m["y"] += m["vy"] * speed_factor

            # Elastic bounce off horizontal walls
            if m["x"] <= c_min_x:
                m["x"] = c_min_x
                m["vx"] = abs(m["vx"])
            elif m["x"] >= c_max_x:
                m["x"] = c_max_x
                m["vx"] = -abs(m["vx"])

            # Elastic bounce off vertical walls
            if m["y"] <= c_min_y:
                m["y"] = c_min_y
                m["vy"] = abs(m["vy"])
            elif m["y"] >= c_max_y:
                m["y"] = c_max_y
                m["vy"] = -abs(m["vy"])

        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        """Arranges interactive control buttons in bottom panel."""
        ctrl_x = 385.0
        row1_y = 248.0
        btn_w_half = 72.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["gas_cycle"] = QRectF(ctrl_x + 104.0, row1_y, 172.0, btn_h)

        row2_y = 278.0
        self._buttons["t_dec"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["t_inc"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["t_room"] = QRectF(ctrl_x + 158.0, row2_y, 118.0, btn_h)

        row3_y = 308.0
        self._buttons["p_dec"] = QRectF(ctrl_x, row3_y, btn_w_half, btn_h)
        self._buttons["p_inc"] = QRectF(ctrl_x + 78.0, row3_y, btn_w_half, btn_h)
        self._buttons["t_hot"] = QRectF(ctrl_x + 158.0, row3_y, 118.0, btn_h)

        row4_y = 342.0
        self._buttons["reset"] = QRectF(ctrl_x, row4_y, 100.0, 26.0)
        self._buttons["stamp"] = QRectF(ctrl_x + 108.0, row4_y, 168.0, 26.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Rendering Pipeline
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option=None, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        self._paint_card_background(painter)

        # 2. Header and Title Banner
        self._paint_header(painter)

        # 3. Left Side: 2D Kinetic Gas Chamber with Colliding Molecules
        self._paint_kinetic_chamber(painter)

        # 4. Right Side: Maxwell-Boltzmann Speed Distribution Curve f(v)
        self._paint_distribution_curve(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Card background frame."""
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        """Title header and gas badge."""
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "🔥 PHÂN BỐ VẬN TỐC PHÂN TỬ MAXWELL-BOLTZMANN")

        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        t_c = self._temperature_K - 273.15
        painter.drawText(20, 44, f"Khí: {self._preset.gas_name} | T = {self._temperature_K:.1f} K ({t_c:+.1f}°C) | P = {self._pressure_atm:.1f} atm")

        # Badge (Top Right)
        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        accent = self._preset.accent_color
        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        badge_txt = f"{self._preset.chemical_formula} • M={self._molar_mass*1e3:.2f} g/mol"
        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(accent)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_kinetic_chamber(self, painter: QPainter) -> None:
        """Renders 2D chamber showing colliding gas molecules."""
        box = QRectF(20.0, 56.0, 345.0, 180.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.2))
        painter.setBrush(QBrush(QColor(8, 12, 20, 220)))
        painter.drawRoundedRect(box, 8.0, 8.0)

        # Title of chamber
        font_lbl = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_lbl)
        painter.setPen(QColor(120, 150, 190))
        painter.drawText(int(box.left() + 10.0), int(box.top() + 16.0), "BÌNH CHỨA KHÍ LÝ TƯỞNG (2D KINETIC GAS CHAMBER)")

        # Render colliding molecules
        accent = self._preset.accent_color
        for m in self._molecules:
            speed = math.sqrt(m["vx"] ** 2 + m["vy"] ** 2)
            # Color code: slow -> cyan/blue; medium -> yellow; fast -> red/white
            if speed < 0.7:
                col = QColor(0, 180, 255, 230)
            elif speed < 1.3:
                col = QColor(255, 220, 50, 240)
            else:
                col = QColor(255, 60, 60, 255)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(m["x"], m["y"]), m["radius"], m["radius"])

        # Bottom info bar in chamber
        info_rect = QRectF(box.left() + 10.0, box.bottom() - 32.0, box.width() - 20.0, 24.0)
        painter.setPen(QPen(QColor(40, 60, 85), 1.0))
        painter.setBrush(QBrush(QColor(12, 18, 30, 200)))
        painter.drawRoundedRect(info_rect, 4.0, 4.0)

        font_inf = QFont("Segoe UI", 7, QFont.Weight.Normal)
        painter.setFont(font_inf)
        painter.setPen(QColor(180, 210, 235))
        painter.drawText(
            info_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"Quãng đường tự do trung bình λ = {self.mean_free_path_nm:.1f} nm | Tần số va chạm Z = {self.collision_frequency_GHz:.2f} GHz",
        )

        # Telemetry panel below chamber (left bottom: y = 245 to 395)
        self._paint_telemetry_table(painter)

    def _paint_telemetry_table(self, painter: QPainter) -> None:
        """Renders exact characteristic speeds and energy metrics."""
        t_box = QRectF(20.0, 245.0, 345.0, 150.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(10, 15, 25, 220)))
        painter.drawRoundedRect(t_box, 8.0, 8.0)

        font_h = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_h)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(t_box.left() + 12.0), int(t_box.top() + 18.0), "⚡ VẬN TỐC ĐẶC TRƯNG (CHARACTERISTIC SPEEDS)")

        font_k = QFont("Segoe UI", 7, QFont.Weight.Normal)
        font_v = QFont("Consolas", 8, QFont.Weight.Bold)

        vp = self.most_probable_speed_m_s
        v_avg = self.average_speed_m_s
        v_rms = self.rms_speed_m_s

        lines = [
            ("Tốc độ có xác suất lớn nhất (v_p):", f"{vp:6.1f} m/s", QColor("#f87171")),
            ("Tốc độ trung bình số học (<v>):", f"{v_avg:6.1f} m/s", QColor("#4ade80")),
            ("Tốc độ căn quân phương (v_rms):", f"{v_rms:6.1f} m/s", QColor("#38bdf8")),
            ("Động năng tịnh tiến trung bình <Ek>:", f"{self.mean_kinetic_energy_eV*1e3:5.2f} meV", QColor("#fbbf24")),
            ("Hệ thức bất biến tỷ lệ:", "v_p < <v> < v_rms  (1 : 1.128 : 1.225)", QColor("#e2e8f0")),
        ]

        curr_y = t_box.top() + 38.0
        for lbl, val_str, val_col in lines:
            painter.setFont(font_k)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(t_box.left() + 12.0), int(curr_y), lbl)

            painter.setFont(font_v)
            painter.setPen(val_col)
            painter.drawText(int(t_box.left() + 215.0), int(curr_y), val_str)
            curr_y += 22.0

    def _paint_distribution_curve(self, painter: QPainter) -> None:
        """Renders Maxwell-Boltzmann PDF f(v) curve on the right panel."""
        plot_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(plot_rect, 8.0, 8.0)

        # Plot axes inside plot_rect
        ax_x0 = plot_rect.left() + 35.0
        ax_y0 = plot_rect.bottom() - 25.0
        ax_w = plot_rect.width() - 50.0
        ax_h = plot_rect.height() - 50.0

        # Draw grid axes
        painter.setPen(QPen(QColor(50, 68, 95), 1.0))
        painter.drawLine(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0) # v axis
        painter.drawLine(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h) # f(v) axis

        font_ax = QFont("Segoe UI", 6, QFont.Weight.Normal)
        painter.setFont(font_ax)
        painter.setPen(QColor(130, 150, 180))
        painter.drawText(int(ax_x0 + ax_w - 40.0), int(ax_y0 + 15.0), "v (m/s) →")
        painter.drawText(int(ax_x0 - 25.0), int(ax_y0 - ax_h + 8.0), "f(v)")

        # Determine plot domain: max_v based on rms speed
        v_rms = self.rms_speed_m_s
        max_v = max(1500.0, min(6000.0, v_rms * 2.8))

        # Peak of f(v) occurs at vp
        vp = self.most_probable_speed_m_s
        f_max = max(1e-6, self.pdf_f_v(vp))

        # Sample curve points
        curve_path = QPainterPath()
        points: list[QPointF] = []
        n_steps = 70
        for i in range(n_steps + 1):
            v_val = (i / n_steps) * max_v
            f_val = self.pdf_f_v(v_val)
            px = ax_x0 + (v_val / max_v) * ax_w
            py = ax_y0 - (f_val / (f_max * 1.15)) * ax_h
            points.append(QPointF(px, py))

        if points:
            curve_path.moveTo(points[0])
            for pt in points[1:]:
                curve_path.lineTo(pt)

            # Draw shaded area under curve
            area_path = QPainterPath(curve_path)
            area_path.lineTo(points[-1].x(), ax_y0)
            area_path.lineTo(points[0].x(), ax_y0)
            area_path.closeSubpath()

            grad = QLinearGradient(0.0, ax_y0 - ax_h, 0.0, ax_y0)
            accent = self._preset.accent_color
            grad.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 120))
            grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 20))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawPath(area_path)

            # Draw outline curve
            painter.setPen(QPen(accent, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(curve_path)

        # Draw vertical lines for vp, v_avg, v_rms
        def draw_v_marker(v_val: float, color: QColor, label: str):
            if v_val <= max_v:
                mx = ax_x0 + (v_val / max_v) * ax_w
                painter.setPen(QPen(color, 1.2, Qt.PenStyle.DashLine))
                painter.drawLine(mx, ax_y0, mx, ax_y0 - ax_h * 0.95)

                font_m = QFont("Segoe UI", 6, QFont.Weight.Bold)
                painter.setFont(font_m)
                painter.setPen(color)
                painter.drawText(int(mx - 8.0), int(ax_y0 - ax_h * 0.98), label)

        draw_v_marker(vp, QColor("#f87171"), "v_p")
        draw_v_marker(self.average_speed_m_s, QColor("#4ade80"), "<v>")
        draw_v_marker(v_rms, QColor("#38bdf8"), "v_rms")

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive buttons."""
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "gas_cycle": f"Khí ({self._preset.chemical_formula})",
            "t_dec": "T: −50 K",
            "t_inc": "T: +50 K",
            "t_room": "T: 300K Phòng",
            "p_dec": "P: −0.5 atm",
            "p_inc": "P: +0.5 atm",
            "t_hot": "T: 1000K Nóng",
            "reset": "🔄 Mặc định",
            "stamp": "📌 Đóng dấu bài học",
        }

        font_btn = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_btn)

        for btn_key, rect in self._buttons.items():
            is_hover = (self._hovered_btn == btn_key)
            if btn_key == "stamp":
                border_col = QColor(0, 230, 180) if is_hover else QColor(0, 180, 140)
                fill_col = QColor(0, 140, 100, 180) if is_hover else QColor(0, 100, 75, 150)
                txt_col = QColor(255, 255, 255)
            elif btn_key == "reset":
                border_col = QColor(220, 100, 100) if is_hover else QColor(160, 70, 70)
                fill_col = QColor(140, 50, 50, 160) if is_hover else QColor(90, 35, 35, 140)
                txt_col = QColor(255, 230, 230)
            elif btn_key in ("t_room", "t_hot"):
                border_col = QColor(240, 180, 50) if is_hover else QColor(190, 140, 30)
                fill_col = QColor(160, 110, 20, 160) if is_hover else QColor(100, 70, 15, 140)
                txt_col = QColor(255, 250, 220)
            else:
                border_col = QColor(80, 140, 220) if is_hover else QColor(45, 75, 120)
                fill_col = QColor(35, 60, 95, 180) if is_hover else QColor(22, 38, 62, 160)
                txt_col = QColor(225, 238, 255)

            painter.setPen(QPen(border_col, 1.0))
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(txt_col)
            label = button_labels.get(btn_key, btn_key)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

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
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _handle_button_click(self, btn_key: str) -> None:
        """Executes corresponding action for UI control clicks."""
        preset_keys = list(GAS_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key in ("preset_next", "gas_cycle"):
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "t_dec":
            self.set_temperature_K(self._temperature_K - 50.0)
        elif btn_key == "t_inc":
            self.set_temperature_K(self._temperature_K + 50.0)
        elif btn_key == "t_room":
            self.set_temperature_K(300.0)
        elif btn_key == "t_hot":
            self.set_temperature_K(1000.0)
        elif btn_key == "p_dec":
            self.set_pressure_atm(self._pressure_atm - 0.50)
        elif btn_key == "p_inc":
            self.set_pressure_atm(self._pressure_atm + 0.50)
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "gas_name": self._preset.gas_name,
            "formula": self._preset.chemical_formula,
            "temperature_K": self._temperature_K,
            "pressure_atm": self._pressure_atm,
            "molar_mass_kg_mol": self._molar_mass,
            "vp_m_s": self.most_probable_speed_m_s,
            "v_avg_m_s": self.average_speed_m_s,
            "v_rms_m_s": self.rms_speed_m_s,
            "mean_free_path_nm": self.mean_free_path_nm,
            "collision_freq_GHz": self.collision_frequency_GHz,
        }
        self.stamp_requested.emit(data)
