"""
AirTeach AI — Interactive Solid State Physics & Semiconductor Electronics:
Hall Effect, Lorentz Drift Velocity, Carrier Density & Hall Voltage Simulator.
Milestone 147 (Task I.134).

Provides real-time interactive simulation of:
1. Charge carrier drift under external electric field: v_d = J / (n * |q|)
2. Transverse Lorentz deflection force: F_B = q * (v_d x B)
3. Steady-state equilibrium Hall electric field: E_H = -v_d * B
4. Analytical Hall voltage: V_H = (I * B) / (n * q * d) = R_H * (I * B) / d
5. Hall coefficient: R_H = 1 / (n * q) distinguishing electrons (n-type) vs holes (p-type)
6. Authentic curriculum presets: InSb sensor, p-type Germanium, Copper metallic foil,
   GaAs 2DEG quantum well, and Silicon automotive sensor.
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
E_CHARGE_C: float = 1.602176634e-19  # Elementary charge in Coulombs


@dataclass(frozen=True)
class HallPreset:
    """Represents a curriculum scenario for the Hall Effect."""
    code: str
    title_vi: str
    material_name: str
    carrier_type: str             # "electron" (n-type / metal) or "hole" (p-type)
    carrier_charge_sign: int      # -1 for electron, +1 for hole
    carrier_density_m3: float     # Carrier concentration n in m^-3
    default_thickness_um: float   # Slab thickness d in micrometers
    default_current_mA: float     # Longitudinal current I in mA
    default_b_field_T: float      # Perpendicular magnetic field B in Tesla
    carrier_mobility_m2_Vs: float # Electron/hole mobility mu
    accent_color: QColor          # Color theme for the semiconductor material
    description: str


HALL_PRESETS: dict[str, HallPreset] = {
    "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR": HallPreset(
        code="INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR",
        title_vi="Cảm biến Từ InSb (Antimonua Indi - Loại n)",
        material_name="Indium Antimonide (InSb n-type)",
        carrier_type="electron",
        carrier_charge_sign=-1,
        carrier_density_m3=2.0e22,       # 2.0 x 10^16 cm^-3 = 2.0 x 10^22 m^-3
        default_thickness_um=10.0,       # 10 micrometers thin film
        default_current_mA=20.0,         # 20 mA
        default_b_field_T=0.50,          # 0.50 Tesla
        carrier_mobility_m2_Vs=7.70,     # Ultra-high mobility 77,000 cm^2/(V*s)
        accent_color=QColor(0, 180, 216),# Electric Cyan
        description="Vật liệu bán dẫn có độ linh động điện tử kỷ lục; ứng dụng chế tạo đầu dò từ trường cực nhạy.",
    ),
    "GERMANIUM_P_TYPE_SEMICONDUCTOR": HallPreset(
        code="GERMANIUM_P_TYPE_SEMICONDUCTOR",
        title_vi="Bán dẫn Germanium Loại p (P-type Germanium)",
        material_name="Germanium p-type (P-doped with In)",
        carrier_type="hole",
        carrier_charge_sign=+1,
        carrier_density_m3=5.0e21,       # 5.0 x 10^21 m^-3
        default_thickness_um=50.0,       # 50 micrometers
        default_current_mA=15.0,         # 15 mA
        default_b_field_T=0.40,          # 0.40 Tesla
        carrier_mobility_m2_Vs=0.19,     # Hole mobility 1,900 cm^2/(V*s)
        accent_color=QColor(230, 57, 70),# Crimson Ruby
        description="Bán dẫn pha tạp thụ nhận (Acceptor); hạt mang điện đa số là lỗ trống dẫn đến điện thế Hall dương.",
    ),
    "COPPER_THIN_FOIL_METAL": HallPreset(
        code="COPPER_THIN_FOIL_METAL",
        title_vi="Lá Đồng Kim Loại (Copper Foil Conductor)",
        material_name="Copper (Kim loại Đồng nguyên chất)",
        carrier_type="electron",
        carrier_charge_sign=-1,
        carrier_density_m3=8.49e28,      # 8.49 x 10^28 m^-3 (1 conduction electron per Cu atom)
        default_thickness_um=25.0,       # 25 micrometers
        default_current_mA=200.0,        # 200 mA
        default_b_field_T=1.00,          # 1.00 Tesla
        carrier_mobility_m2_Vs=0.0045,   # Metallic mobility
        accent_color=QColor(244, 162, 97),# Copper Amber
        description="Kim loại dẫn điện mật độ điện tử tự do khổng lồ; điện thế Hall siêu nhỏ ở mức micro-Volt (µV).",
    ),
    "GALLIUM_ARSENIDE_HEMT_2DEG": HallPreset(
        code="GALLIUM_ARSENIDE_HEMT_2DEG",
        title_vi="Khí Điện Tử 2 Chiều GaAs HEMT (2DEG Quantum Well)",
        material_name="Gallium Arsenide Heterostructure",
        carrier_type="electron",
        carrier_charge_sign=-1,
        carrier_density_m3=3.5e22,       # Equivalent 3D density from 2D sheet 3.5e15 m^-2
        default_thickness_um=10.0,       # 10 micrometers
        default_current_mA=10.0,         # 10 mA
        default_b_field_T=0.80,          # 0.80 Tesla
        carrier_mobility_m2_Vs=1.50,     # 15,000 cm^2/(V*s)
        accent_color=QColor(155, 93, 229),# Quantum Violet
        description="Cấu trúc dị thể lượng tử giếng thế trong transistor cao tần HEMT và hiệu ứng Hall lượng tử.",
    ),
    "SILICON_N_TYPE_INTEGRATED_SENSOR": HallPreset(
        code="SILICON_N_TYPE_INTEGRATED_SENSOR",
        title_vi="Cảm biến Hall Tích hợp Silicon (Si IC Sensor)",
        material_name="Silicon n-type (Phốt-pho pha tạp)",
        carrier_type="electron",
        carrier_charge_sign=-1,
        carrier_density_m3=1.2e21,       # 1.2 x 10^21 m^-3
        default_thickness_um=20.0,       # 20 micrometers
        default_current_mA=10.0,         # 10 mA
        default_b_field_T=0.30,          # 0.30 Tesla
        carrier_mobility_m2_Vs=0.14,     # Electron mobility 1,400 cm^2/(V*s)
        accent_color=QColor(42, 157, 143),# Silicon Teal
        description="Mạch tích hợp cảm biến Hall thương mại trong ô tô, đo tốc độ góc trục khuỷu và vị trí góc rotor.",
    ),
}


class InteractiveHallEffectItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating the Hall Effect in conductors
    and semiconductors with real-time drift velocity, Lorentz deflection, and Hall voltage.
    """

    # Signals
    hall_computed = Signal(dict)
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
        self._active_preset_code: str = "INDIUM_ANTIMONIDE_IN_SB_HALL_SENSOR"
        self._preset: HallPreset = HALL_PRESETS[self._active_preset_code]

        self._current_mA: float = self._preset.default_current_mA
        self._b_field_T: float = self._preset.default_b_field_T
        self._thickness_um: float = self._preset.default_thickness_um
        self._slab_width_mm: float = 5.0   # Fixed width w = 5.0 mm
        self._carrier_charge_sign: int = self._preset.carrier_charge_sign
        self._carrier_density_m3: float = self._preset.carrier_density_m3

        # Animated drifting carriers
        self._carriers: list[dict] = []
        self._init_carriers(28)

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for smooth 60fps particle drift animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def _init_carriers(self, count: int) -> None:
        """Initializes drifting charge carrier positions and animation phases."""
        self._carriers.clear()
        for _ in range(count):
            self._carriers.append({
                "u": random.uniform(0.0, 1.0),       # Normalized x along slab (0 to 1)
                "v": random.uniform(0.1, 0.9),       # Normalized y along transverse height
                "phase": random.uniform(0.0, 2.0 * math.pi),
                "speed": random.uniform(0.85, 1.15),
            })

    # ------------------------------------------------------------------
    # Physics Properties & Exact Mathematical Laws
    # ------------------------------------------------------------------

    @property
    def carrier_charge_coulombs(self) -> float:
        """Exact carrier charge q = sign * e."""
        return float(self._carrier_charge_sign) * E_CHARGE_C

    @property
    def current_amperes(self) -> float:
        """Electric current I in Amperes."""
        return self._current_mA * 1e-3

    @property
    def thickness_meters(self) -> float:
        """Slab thickness d in meters."""
        return max(1.0, self._thickness_um) * 1e-6

    @property
    def width_meters(self) -> float:
        """Slab width w in meters."""
        return self._slab_width_mm * 1e-3

    @property
    def cross_section_area_m2(self) -> float:
        """Slab cross-section area A = w * d in m^2."""
        return self.width_meters * self.thickness_meters

    @property
    def current_density_A_m2(self) -> float:
        """Current density J = I / (w * d) in A/m^2."""
        area = self.cross_section_area_m2
        return self.current_amperes / area if area > 0.0 else 0.0

    @property
    def drift_velocity_m_s(self) -> float:
        """
        Carrier drift velocity:
        v_d = J / (n * |q|)
        """
        n = max(1.0, self._carrier_density_m3)
        denom = n * E_CHARGE_C
        return self.current_density_A_m2 / denom if denom > 0.0 else 0.0

    @property
    def hall_coefficient_m3_C(self) -> float:
        """
        Hall coefficient:
        R_H = 1 / (n * q)
        Negative for electrons, positive for holes.
        """
        denom = self._carrier_density_m3 * self.carrier_charge_coulombs
        return 1.0 / denom if abs(denom) > 1e-30 else 0.0

    @property
    def hall_voltage_volts(self) -> float:
        """
        Exact Analytical Hall Voltage:
        V_H = (I * B) / (n * q * d) = R_H * (I * B) / d
        """
        i_amp = self.current_amperes
        b_tesla = self._b_field_T
        d_m = self.thickness_meters
        n = max(1.0, self._carrier_density_m3)
        q = self.carrier_charge_coulombs

        denom = n * q * d_m
        if abs(denom) < 1e-30:
            return 0.0
        return (i_amp * b_tesla) / denom

    @property
    def hall_field_V_m(self) -> float:
        """
        Equilibrium transverse Hall electric field:
        E_H = V_H / w = v_d * B
        """
        return self.hall_voltage_volts / self.width_meters

    @property
    def lorentz_force_newtons(self) -> float:
        """
        Magnetic Lorentz force on a single charge carrier:
        F_B = |q| * v_d * |B|
        """
        return E_CHARGE_C * self.drift_velocity_m_s * abs(self._b_field_T)

    # ------------------------------------------------------------------
    # Simulation & State Mutations
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Selects curriculum preset and resets physical values to standard scenario."""
        if preset_code in HALL_PRESETS:
            self._active_preset_code = preset_code
            self._preset = HALL_PRESETS[preset_code]
            self._current_mA = self._preset.default_current_mA
            self._b_field_T = self._preset.default_b_field_T
            self._thickness_um = self._preset.default_thickness_um
            self._carrier_charge_sign = self._preset.carrier_charge_sign
            self._carrier_density_m3 = self._preset.carrier_density_m3
            self._emit_computed_values()
            self.update()

    def set_b_field_Tesla(self, b_val: float) -> None:
        """Sets external magnetic field B within [-3.0, 3.0] Tesla."""
        self._b_field_T = max(-3.0, min(3.0, round(b_val, 2)))
        self._emit_computed_values()
        self.update()

    def invert_b_field(self) -> None:
        """Inverts magnetic field direction (B -> -B)."""
        self.set_b_field_Tesla(-self._b_field_T)

    def set_current_mA(self, i_val: float) -> None:
        """Sets electric current within [1.0, 500.0] mA."""
        self._current_mA = max(1.0, min(500.0, round(i_val, 1)))
        self._emit_computed_values()
        self.update()

    def set_thickness_um(self, d_val: float) -> None:
        """Sets slab thickness within [2.0, 250.0] micrometers."""
        self._thickness_um = max(2.0, min(250.0, round(d_val, 1)))
        self._emit_computed_values()
        self.update()

    def toggle_carrier_type(self) -> None:
        """Toggles between electron (-e) and hole (+e) carriers."""
        self._carrier_charge_sign = -self._carrier_charge_sign
        self._emit_computed_values()
        self.update()

    def reset_defaults(self) -> None:
        """Restores default values of current preset."""
        self.set_preset(self._active_preset_code)

    def _emit_computed_values(self) -> None:
        """Emits real-time computed physics data dictionary."""
        data = {
            "preset_code": self._active_preset_code,
            "material_name": self._preset.material_name,
            "carrier_type": "hole" if self._carrier_charge_sign > 0 else "electron",
            "current_mA": self._current_mA,
            "b_field_T": self._b_field_T,
            "thickness_um": self._thickness_um,
            "carrier_density_m3": self._carrier_density_m3,
            "drift_velocity_m_s": self.drift_velocity_m_s,
            "hall_coefficient_m3_C": self.hall_coefficient_m3_C,
            "hall_voltage_V": self.hall_voltage_volts,
            "hall_field_V_m": self.hall_field_V_m,
            "lorentz_force_N": self.lorentz_force_newtons,
        }
        self.hall_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Animation update tick for carrier motion and Lorentz drift."""
        # Calculate visual drift speed proportional to drift velocity
        base_speed = 0.008
        if self._current_mA > 0.0:
            base_speed *= min(2.5, max(0.4, self._current_mA / 20.0))

        # Lorentz deflection factor: depends on sign(q) * B
        # For electrons (q < 0) moving in +x with B in +z: deflect toward -y (downwards)
        # For holes (q > 0) moving in +x with B in +z: deflect toward -y (downwards)
        # Transverse equilibrium deflection shift:
        deflection_target = -0.25 * math.tanh(self._b_field_T * 1.5)

        for c in self._carriers:
            c["u"] += base_speed * c["speed"]
            if c["u"] > 1.0:
                c["u"] -= 1.0
                c["v"] = random.uniform(0.15, 0.85)

            # Smoothly bias carrier transverse position v toward deflection edge
            v_bias = 0.5 + deflection_target
            c["v"] = 0.96 * c["v"] + 0.04 * (v_bias + 0.3 * (random.uniform(-0.5, 0.5)))
            c["v"] = max(0.08, min(0.92, c["v"]))

        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Interaction Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        """Positions interactive control buttons in the right panel and bottom footer."""
        # Preset buttons (top-right of controls)
        ctrl_x = 390.0
        row1_y = 248.0
        btn_w_half = 68.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["toggle_carrier"] = QRectF(ctrl_x + 104.0, row1_y, 166.0, btn_h)

        row2_y = 278.0
        self._buttons["b_dec"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["b_inc"] = QRectF(ctrl_x + 72.0, row2_y, btn_w_half, btn_h)
        self._buttons["b_invert"] = QRectF(ctrl_x + 148.0, row2_y, 122.0, btn_h)

        row3_y = 308.0
        self._buttons["i_dec"] = QRectF(ctrl_x, row3_y, btn_w_half, btn_h)
        self._buttons["i_inc"] = QRectF(ctrl_x + 72.0, row3_y, btn_w_half, btn_h)
        self._buttons["d_dec"] = QRectF(ctrl_x + 148.0, row3_y, 58.0, btn_h)
        self._buttons["d_inc"] = QRectF(ctrl_x + 212.0, row3_y, 58.0, btn_h)

        row4_y = 342.0
        self._buttons["reset"] = QRectF(ctrl_x, row4_y, 100.0, 26.0)
        self._buttons["stamp"] = QRectF(ctrl_x + 108.0, row4_y, 162.0, 26.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Rendering Pipeline (QPainter Vector Drawing)
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option=None, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card with sleek frosted glass gradient
        self._paint_card_background(painter)

        # 2. Header and Title Banner
        self._paint_header(painter)

        # 3. 3D Isometric / Perspective Semiconductor Slab & Physical Fields
        self._paint_hall_slab_3d(painter)

        # 4. Digital Multimeter & Telemetry Panel (Right Side)
        self._paint_telemetry_panel(painter)

        # 5. Control Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Renders main card frame with subtle border and gradient."""
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        """Renders title and carrier status badge."""
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "⚡ HIỆU ỨNG HALL & ĐỘ LINH ĐỘNG LORENTZ")

        # Preset name subtitle
        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        painter.drawText(20, 44, f"Kịch bản: {self._preset.title_vi} ({self._preset.material_name})")

        # Carrier Type Badge (Top Right)
        is_hole = (self._carrier_charge_sign > 0)
        badge_txt = "🔴 Loại p (Lỗ trống - Hole)" if is_hole else "🔵 Loại n (Điện tử - Electron)"
        badge_color = QColor(230, 57, 70) if is_hole else QColor(0, 180, 216)

        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        painter.setPen(QPen(badge_color, 1.2))
        painter.setBrush(QBrush(QColor(badge_color.red(), badge_color.green(), badge_color.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(badge_color)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_hall_slab_3d(self, painter: QPainter) -> None:
        """
        Renders a 3D isometric perspective view of the semiconductor slab:
        - Geometric faces (Top, Front, Side)
        - External Magnetic Field B vectors with flux arrows
        - Longitudinal current I entering left and exiting right
        - Transverse Hall electric field E_H and voltage probes V_H
        - Animated drifting charge carriers deflecting under Lorentz force
        """
        # Canvas boundaries for 3D slab
        slab_box = QRectF(20.0, 56.0, 350.0, 335.0)

        # Clip container background
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(8, 12, 20, 200)))
        painter.drawRoundedRect(slab_box, 8.0, 8.0)

        # 3D Coordinates of the slab vertices
        # Origin (front-bottom-left corner of the slab)
        ox = 90.0
        oy = 245.0
        length_px = 160.0    # Along X-axis (Current direction, tilted slightly)
        height_px = 55.0     # Along Y-axis (Thickness / transverse)
        depth_px = 75.0      # Along Z-axis (Depth / Width, tilted up-right)
        tilt_x = 0.866 * depth_px
        tilt_y = -0.500 * depth_px

        # Vertices:
        # Front face: A(ox, oy), B(ox + length_px, oy), C(ox + length_px, oy - height_px), D(ox, oy - height_px)
        p_A = QPointF(ox, oy)
        p_B = QPointF(ox + length_px, oy)
        p_C = QPointF(ox + length_px, oy - height_px)
        p_D = QPointF(ox, oy - height_px)

        # Back face: A', B', C', D' shifted by (tilt_x, tilt_y)
        p_A_back = QPointF(ox + tilt_x, oy + tilt_y)
        p_B_back = QPointF(ox + length_px + tilt_x, oy + tilt_y)
        p_C_back = QPointF(ox + length_px + tilt_x, oy - height_px + tilt_y)
        p_D_back = QPointF(ox + tilt_x, oy - height_px + tilt_y)

        # 1. Back lines & bottom shadow
        painter.setPen(QPen(QColor(30, 45, 65, 120), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(p_A, p_A_back)
        painter.drawLine(p_A_back, p_B_back)
        painter.drawLine(p_A_back, p_D_back)

        # 2. Draw Magnetic Field B arrows passing vertically through the slab
        self._paint_b_field_vectors(painter, p_D, p_C, p_C_back, p_D_back)

        # 3. Slab Faces
        # Top Face (D, C, C_back, D_back)
        top_path = QPainterPath()
        top_path.moveTo(p_D)
        top_path.lineTo(p_C)
        top_path.lineTo(p_C_back)
        top_path.lineTo(p_D_back)
        top_path.closeSubpath()

        mat_color = self._preset.accent_color
        top_grad = QLinearGradient(p_D, p_C_back)
        top_grad.setColorAt(0.0, QColor(mat_color.red(), mat_color.green(), mat_color.blue(), 180))
        top_grad.setColorAt(1.0, QColor(mat_color.red() // 2, mat_color.green() // 2, mat_color.blue() // 2, 160))
        painter.setPen(QPen(QColor(180, 220, 255, 180), 1.2))
        painter.setBrush(QBrush(top_grad))
        painter.drawPath(top_path)

        # Front Face (A, B, C, D)
        front_path = QPainterPath()
        front_path.moveTo(p_A)
        front_path.lineTo(p_B)
        front_path.lineTo(p_C)
        front_path.lineTo(p_D)
        front_path.closeSubpath()

        front_grad = QLinearGradient(p_D, p_A)
        front_grad.setColorAt(0.0, QColor(mat_color.red() // 2, mat_color.green() // 2, mat_color.blue() // 2, 210))
        front_grad.setColorAt(1.0, QColor(mat_color.red() // 4, mat_color.green() // 4, mat_color.blue() // 4, 230))
        painter.setBrush(QBrush(front_grad))
        painter.drawPath(front_path)

        # Right Side Face (B, B_back, C_back, C)
        side_path = QPainterPath()
        side_path.moveTo(p_B)
        side_path.lineTo(p_B_back)
        side_path.lineTo(p_C_back)
        side_path.lineTo(p_C)
        side_path.closeSubpath()

        side_grad = QLinearGradient(p_B, p_C_back)
        side_grad.setColorAt(0.0, QColor(mat_color.red() // 3, mat_color.green() // 3, mat_color.blue() // 3, 220))
        side_grad.setColorAt(1.0, QColor(mat_color.red() // 5, mat_color.green() // 5, mat_color.blue() // 5, 240))
        painter.setBrush(QBrush(side_grad))
        painter.drawPath(side_path)

        # 4. Animated Drifting Charge Carriers inside the Top/Front face
        self._paint_drifting_carriers(painter, p_D, p_C, p_C_back, p_D_back, p_A, p_B)

        # 5. Longitudinal Electric Current I (Golden Lead In / Out)
        self._paint_current_flow(painter, p_A, p_D, p_B_back, p_C_back)

        # 6. Hall Voltage Probes and Leads (Red V+ on top edge, Blue V- on bottom edge)
        self._paint_voltage_probes(painter, p_C, p_B, p_D, p_A)

        # 7. Surface Accumulated Charge Symbols (+ and - signs)
        self._paint_surface_charges(painter, p_D, p_C, p_A, p_B)

    def _paint_b_field_vectors(
        self,
        painter: QPainter,
        p_D: QPointF,
        p_C: QPointF,
        p_C_back: QPointF,
        p_D_back: QPointF,
    ) -> None:
        """Renders vertical magnetic field flux lines and direction indicator."""
        b_val = self._b_field_T
        if abs(b_val) < 0.01:
            return

        is_upward = (b_val > 0.0)
        arrow_len = 45.0
        dy = -arrow_len if is_upward else arrow_len

        b_color = QColor(190, 80, 255, 230)
        painter.setPen(QPen(b_color, 1.8))
        painter.setBrush(QBrush(b_color))

        # 3 magnetic field arrows penetrating the slab top face
        sample_points = [
            QPointF((p_D.x() + p_C.x()) * 0.5 + 20.0, (p_D.y() + p_C.y()) * 0.5 - 15.0),
            QPointF((p_D.x() + p_C.x()) * 0.5 - 25.0, (p_D.y() + p_C.y()) * 0.5 - 5.0),
            QPointF((p_D.x() + p_C.x()) * 0.5 + 40.0, (p_D.y() + p_C.y()) * 0.5 + 5.0),
        ]

        for pt in sample_points:
            p_start = QPointF(pt.x(), pt.y() - dy * 0.3)
            p_end = QPointF(pt.x(), pt.y() + dy)
            painter.drawLine(p_start, p_end)

            # Draw arrowhead
            ah_size = 5.0
            ah_dir = -1.0 if is_upward else 1.0
            p1 = QPointF(p_end.x() - ah_size * 0.7, p_end.y() - ah_dir * ah_size)
            p2 = QPointF(p_end.x() + ah_size * 0.7, p_end.y() - ah_dir * ah_size)
            arrow_poly = [p_end, p1, p2]
            painter.drawPolygon(arrow_poly)

        # Label: B-vector
        font_b = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_b)
        painter.setPen(b_color)
        lbl_y = p_D_back.y() - 15.0 if is_upward else p_D.y() + 45.0
        painter.drawText(int(p_D_back.x() + 30.0), int(lbl_y), f"B⃗ = {b_val:+.2f} T (Từ trường vuông góc)")

    def _paint_drifting_carriers(
        self,
        painter: QPainter,
        p_D: QPointF,
        p_C: QPointF,
        p_C_back: QPointF,
        p_D_back: QPointF,
        p_A: QPointF,
        p_B: QPointF,
    ) -> None:
        """Draws animated drifting carriers on the top isometric plane."""
        is_hole = (self._carrier_charge_sign > 0)
        c_color = QColor(255, 80, 80, 240) if is_hole else QColor(50, 220, 255, 240)
        core_color = QColor(255, 255, 255)

        for c in self._carriers:
            u = c["u"]
            v = c["v"]

            # Interpolate isometric point on top face:
            # D -> C along u; D -> D_back along v
            pt_x = (1.0 - u) * ((1.0 - v) * p_D.x() + v * p_D_back.x()) + u * ((1.0 - v) * p_C.x() + v * p_C_back.x())
            pt_y = (1.0 - u) * ((1.0 - v) * p_D.y() + v * p_D_back.y()) + u * ((1.0 - v) * p_C.y() + v * p_C_back.y())

            # Draw halo glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(c_color))
            painter.drawEllipse(QPointF(pt_x, pt_y), 3.2, 3.2)

            # Draw bright center core
            painter.setBrush(QBrush(core_color))
            painter.drawEllipse(QPointF(pt_x, pt_y), 1.2, 1.2)

    def _paint_current_flow(
        self,
        painter: QPainter,
        p_A: QPointF,
        p_D: QPointF,
        p_B_back: QPointF,
        p_C_back: QPointF,
    ) -> None:
        """Renders current flow I entering from the left and leaving from the right."""
        i_color = QColor(255, 200, 0, 240)  # Gold
        painter.setPen(QPen(i_color, 2.0))
        painter.setBrush(QBrush(i_color))

        # Left Inflow Arrow (I in)
        mid_in_x = p_A.x() - 40.0
        mid_in_y = (p_A.y() + p_D.y()) * 0.5
        target_in_x = p_A.x()
        painter.drawLine(QPointF(mid_in_x, mid_in_y), QPointF(target_in_x, mid_in_y))
        painter.drawPolygon([
            QPointF(target_in_x, mid_in_y),
            QPointF(target_in_x - 7.0, mid_in_y - 4.5),
            QPointF(target_in_x - 7.0, mid_in_y + 4.5),
        ])

        # Right Outflow Arrow (I out)
        mid_out_x = p_B_back.x() + 40.0
        mid_out_y = (p_B_back.y() + p_C_back.y()) * 0.5
        start_out_x = p_B_back.x()
        painter.drawLine(QPointF(start_out_x, mid_out_y), QPointF(mid_out_x, mid_out_y))
        painter.drawPolygon([
            QPointF(mid_out_x, mid_out_y),
            QPointF(mid_out_x - 7.0, mid_out_y - 4.5),
            QPointF(mid_out_x - 7.0, mid_out_y + 4.5),
        ])

        font_i = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_i)
        painter.setPen(i_color)
        painter.drawText(int(mid_in_x - 15.0), int(mid_in_y - 8.0), f"I = {self._current_mA:.1f} mA →")

    def _paint_voltage_probes(
        self,
        painter: QPainter,
        p_C: QPointF,
        p_B: QPointF,
        p_D: QPointF,
        p_A: QPointF,
    ) -> None:
        """Renders Hall voltage measurement probe contacts and dashed leads."""
        vh_val = self.hall_voltage_volts
        probe_top = QPointF((p_D.x() + p_C.x()) * 0.5, (p_D.y() + p_C.y()) * 0.5)
        probe_bottom = QPointF((p_A.x() + p_B.x()) * 0.5, (p_A.y() + p_B.y()) * 0.5)

        # Contact dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 50, 50)))  # Red V+
        painter.drawEllipse(probe_top, 4.0, 4.0)

        painter.setBrush(QBrush(QColor(50, 120, 255))) # Blue V-
        painter.drawEllipse(probe_bottom, 4.0, 4.0)

        # Dashed wires leading to Voltmeter indicator
        pen_wire = QPen(QColor(200, 200, 200, 160), 1.2, Qt.PenStyle.DashDotLine)
        painter.setPen(pen_wire)

        vm_box = QRectF(40.0, 310.0, 310.0, 40.0)
        painter.drawLine(probe_top, QPointF(vm_box.center().x() - 30.0, vm_box.top()))
        painter.drawLine(probe_bottom, QPointF(vm_box.center().x() + 30.0, vm_box.top()))

        # Voltmeter Display Box
        painter.setPen(QPen(QColor(0, 220, 180), 1.4))
        painter.setBrush(QBrush(QColor(10, 25, 30, 230)))
        painter.drawRoundedRect(vm_box, 6.0, 6.0)

        # Voltmeter readout text
        font_vm = QFont("Consolas", 10, QFont.Weight.Bold)
        painter.setFont(font_vm)
        painter.setPen(QColor(0, 240, 200))

        # Format VH nicely (µV vs mV vs V)
        abs_v = abs(vh_val)
        if abs_v < 1e-4:
            vh_str = f"{vh_val * 1e6:+.2f} µV"
        elif abs_v < 1.0:
            vh_str = f"{vh_val * 1e3:+.2f} mV"
        else:
            vh_str = f"{vh_val:+.4f} V"

        painter.drawText(
            vm_box,
            Qt.AlignmentFlag.AlignCenter,
            f"V_H = {vh_str}  (E_H = {self.hall_field_V_m:.2f} V/m)",
        )

    def _paint_surface_charges(
        self,
        painter: QPainter,
        p_D: QPointF,
        p_C: QPointF,
        p_A: QPointF,
        p_B: QPointF,
    ) -> None:
        """Renders accumulated surface charges (+ and - symbols) along opposite edges."""
        vh = self.hall_voltage_volts
        if abs(vh) < 1e-9:
            return

        # If VH > 0: top is positive (+), bottom is negative (-)
        # If VH < 0: top is negative (-), bottom is positive (+)
        top_sign = "+" if vh > 0 else "−"
        bottom_sign = "−" if vh > 0 else "+"
        top_color = QColor(255, 80, 80) if top_sign == "+" else QColor(50, 160, 255)
        bottom_color = QColor(255, 80, 80) if bottom_sign == "+" else QColor(50, 160, 255)

        font_sign = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_sign)

        # 3 charges along top front edge
        painter.setPen(top_color)
        for frac in (0.3, 0.5, 0.7):
            cx = (1.0 - frac) * p_D.x() + frac * p_C.x()
            cy = (1.0 - frac) * p_D.y() + frac * p_C.y() - 6.0
            painter.drawText(int(cx - 4.0), int(cy + 4.0), top_sign)

        # 3 charges along bottom front edge
        painter.setPen(bottom_color)
        for frac in (0.3, 0.5, 0.7):
            cx = (1.0 - frac) * p_A.x() + frac * p_B.x()
            cy = (1.0 - frac) * p_A.y() + frac * p_B.y() + 10.0
            painter.drawText(int(cx - 4.0), int(cy + 4.0), bottom_sign)

    def _paint_telemetry_panel(self, painter: QPainter) -> None:
        """Renders comprehensive physical formulas, semiconductor parameters, and telemetry."""
        panel_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(panel_rect, 8.0, 8.0)

        font_hdr = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_hdr)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(panel_rect.left() + 12.0), int(panel_rect.top() + 20.0), "📊 THÔNG SỐ VẬT LÝ & ĐỊNH LUẬT HALL")

        # Telemetry metrics
        font_txt = QFont("Segoe UI", 8, QFont.Weight.Normal)
        font_val = QFont("Consolas", 8, QFont.Weight.Bold)

        metrics = [
            ("Công thức Hall:", "V_H = (I · B) / (n · q · d)"),
            ("Hệ số Hall R_H:", f"{self.hall_coefficient_m3_C:+.3e} m³/C"),
            ("Mật độ hạt n:", f"{self._carrier_density_m3:.2e} m⁻³"),
            ("Vận tốc trôi v_d:", f"{self.drift_velocity_m_s:.2e} m/s"),
            ("Lực Lorentz F_B:", f"{self.lorentz_force_newtons:.2e} N"),
            ("Chiều dày phiến d:", f"{self._thickness_um:.1f} µm"),
            ("Độ linh động µ:", f"{self._preset.carrier_mobility_m2_Vs:.3f} m²/(V·s)"),
        ]

        y_pos = panel_rect.top() + 38.0
        for label, val_str in metrics:
            painter.setFont(font_txt)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(panel_rect.left() + 12.0), int(y_pos), label)

            painter.setFont(font_val)
            painter.setPen(QColor(230, 245, 255))
            painter.drawText(int(panel_rect.left() + 115.0), int(y_pos), val_str)
            y_pos += 20.0

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive buttons with hover state feedback."""
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "toggle_carrier": "Đổi Hạt (e⁻ ↔ h⁺)",
            "b_dec": "B: −0.1T",
            "b_inc": "B: +0.1T",
            "b_invert": "Đảo Cực B⃗",
            "i_dec": "I: −10mA",
            "i_inc": "I: +10mA",
            "d_dec": "d: −5µm",
            "d_inc": "d: +5µm",
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
            elif btn_key == "toggle_carrier":
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
    # Mouse Events & Touchless Air Gesture Mapping
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
        preset_keys = list(HALL_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "toggle_carrier":
            self.toggle_carrier_type()
        elif btn_key == "b_dec":
            self.set_b_field_Tesla(self._b_field_T - 0.10)
        elif btn_key == "b_inc":
            self.set_b_field_Tesla(self._b_field_T + 0.10)
        elif btn_key == "b_invert":
            self.invert_b_field()
        elif btn_key == "i_dec":
            self.set_current_mA(self._current_mA - 10.0)
        elif btn_key == "i_inc":
            self.set_current_mA(self._current_mA + 10.0)
        elif btn_key == "d_dec":
            self.set_thickness_um(self._thickness_um - 5.0)
        elif btn_key == "d_inc":
            self.set_thickness_um(self._thickness_um + 5.0)
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal with full physical parameters for vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "material_name": self._preset.material_name,
            "carrier_type": "hole" if self._carrier_charge_sign > 0 else "electron",
            "current_mA": self._current_mA,
            "b_field_T": self._b_field_T,
            "thickness_um": self._thickness_um,
            "carrier_density_m3": self._carrier_density_m3,
            "hall_voltage_V": self.hall_voltage_volts,
            "hall_coefficient_m3_C": self.hall_coefficient_m3_C,
            "drift_velocity_m_s": self.drift_velocity_m_s,
        }
        self.stamp_requested.emit(data)
