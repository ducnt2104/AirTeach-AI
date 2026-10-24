"""
AirTeach AI — Interactive Nuclear Physics & Quantum Mechanics:
Alpha, Beta, Gamma Radiation Penetration, Exponential Attenuation Law & Half-Value Layer (HVL) Simulator.
Milestone 148 (Task I.135).

Provides real-time interactive simulation of:
1. Three classical radiation types: Alpha (α, ⁴He²⁺), Beta (β⁻, high-energy e⁻), Gamma (γ, MeV photons).
2. Beer-Lambert-Bouguer exponential attenuation law: I(x) = I_0 * exp(-µ * x)
3. Half-Value Layer: HVL = ln(2) / µ ≈ 0.69315 / µ
4. Tenth-Value Layer: TVL = ln(10) / µ ≈ 2.30259 / µ
5. Transmission fraction and dose reduction percentage: %T = (I / I_0) * 100%
6. 5 Curriculum Presets: Cobalt-60 in Lead, Cesium-137 in Concrete, Strontium-90 in Aluminum,
   Americium-241 in Paper/Air, and Iodine-131 in Water/Tissue.
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


@dataclass(frozen=True)
class RadiationPreset:
    """Represents a curriculum nuclear physics scenario."""
    code: str
    title_vi: str
    isotope_name: str
    radiation_type: str             # "alpha", "beta", "gamma"
    energy_MeV: float               # Dominant particle/photon energy in MeV
    absorber_material: str          # Lead, Concrete, Aluminum, Paper, Water
    density_g_cm3: float            # Absorber mass density in g/cm^3
    linear_mu_cm_inv: float         # Linear attenuation coefficient mu in cm^-1
    default_thickness_cm: float     # Default barrier thickness x in cm
    default_intensity_cps: float    # Incident radiation intensity I_0 in counts per second
    particle_symbol: str            # α, β⁻, γ
    accent_color: QColor            # Visual color of the beam
    description: str


RADIATION_PRESETS: dict[str, RadiationPreset] = {
    "COBALT_60_GAMMA_LEAD_SHIELDING": RadiationPreset(
        code="COBALT_60_GAMMA_LEAD_SHIELDING",
        title_vi="Cobalt-60 Tia Gamma & Tấm Chắn Chì (Lead Pb)",
        isotope_name="Cobalt-60 (⁶⁰Co, γ = 1.25 MeV)",
        radiation_type="gamma",
        energy_MeV=1.25,
        absorber_material="Chì (Lead - Pb)",
        density_g_cm3=11.34,
        linear_mu_cm_inv=0.655,        # HVL = 1.058 cm
        default_thickness_cm=2.50,
        default_intensity_cps=1000.0,
        particle_symbol="γ",
        accent_color=QColor(180, 80, 255), # High energy violet photon
        description="Đồng vị phóng xạ y học và công nghiệp; phát tia gamma năng lượng cao đòi hỏi lớp chì dày để che chắn an toàn.",
    ),
    "CESIUM_137_GAMMA_CONCRETE_BARRIER": RadiationPreset(
        code="CESIUM_137_GAMMA_CONCRETE_BARRIER",
        title_vi="Cesium-137 Tia Gamma & Bức Tường Bê Tông (Concrete)",
        isotope_name="Cesium-137 (¹³⁷Cs, γ = 0.662 MeV)",
        radiation_type="gamma",
        energy_MeV=0.662,
        absorber_material="Bê tông Che chắn (Concrete)",
        density_g_cm3=2.35,
        linear_mu_cm_inv=0.178,        # HVL = 3.894 cm
        default_thickness_cm=8.00,
        default_intensity_cps=800.0,
        particle_symbol="γ",
        accent_color=QColor(130, 110, 255),
        description="Sản phẩm phân hạch hạt nhân; tường bê tông dày trong lò phản ứng và hầm chứa chất thải phóng xạ.",
    ),
    "STRONTIUM_90_BETA_ALUMINUM_ABSORBER": RadiationPreset(
        code="STRONTIUM_90_BETA_ALUMINUM_ABSORBER",
        title_vi="Strontium-90 Tia Beta & Tấm Nhôm (Aluminum Al)",
        isotope_name="Strontium-90 / Y-90 (⁹⁰Sr, β⁻ = 2.28 MeV)",
        radiation_type="beta",
        energy_MeV=2.28,
        absorber_material="Nhôm (Aluminum - Al)",
        density_g_cm3=2.70,
        linear_mu_cm_inv=18.5,         # Very high effective attenuation for electrons
        default_thickness_cm=0.30,     # 3 mm Al
        default_intensity_cps=1200.0,
        particle_symbol="β⁻",
        accent_color=QColor(0, 190, 240), # High-speed electron cyan
        description="Nguồn phát hạt electron năng lượng cao; tấm nhôm mỏng vài milimet hấp thụ hoàn toàn mà không sinh nhiều bức xạ hãm (Bremsstrahlung).",
    ),
    "AMERICIUM_241_ALPHA_PAPER_AIR": RadiationPreset(
        code="AMERICIUM_241_ALPHA_PAPER_AIR",
        title_vi="Americium-241 Hạt Alpha & Tờ Giấy Mỏng (Paper Barrier)",
        isotope_name="Americium-241 (²⁴¹Am, α = 5.48 MeV)",
        radiation_type="alpha",
        energy_MeV=5.48,
        absorber_material="Giấy Cellulose (Paper Barrier)",
        density_g_cm3=0.80,
        linear_mu_cm_inv=950.0,        # Almost step-function absorption
        default_thickness_cm=0.010,    # 0.1 mm paper
        default_intensity_cps=1500.0,
        particle_symbol="α",
        accent_color=QColor(255, 180, 0), # Heavy helium ion gold
        description="Nguồn alpha trong đầu báo khói gia đình; bị chặn đứng hoàn toàn bởi một tờ giấy mỏng hoặc vài cm không khí.",
    ),
    "IODINE_131_THYROID_WATER_TISSUE": RadiationPreset(
        code="IODINE_131_THYROID_WATER_TISSUE",
        title_vi="Iodine-131 Tia Gamma & Mô Cơ Thể (Water / Tissue)",
        isotope_name="Iodine-131 (¹³¹I, γ = 0.364 MeV)",
        radiation_type="gamma",
        energy_MeV=0.364,
        absorber_material="Mô người / Nước (Tissue Equivalent)",
        density_g_cm3=1.00,
        linear_mu_cm_inv=0.111,        # HVL = 6.24 cm
        default_thickness_cm=5.00,
        default_intensity_cps=600.0,
        particle_symbol="γ",
        accent_color=QColor(230, 60, 110), # Medical radiotherapy magenta
        description="Ứng dụng điều trị tuyến giáp trong y học hạt nhân; mô mềm và nước làm suy giảm liều gamma theo định luật hàm mũ.",
    ),
}


class InteractiveRadiationShieldingItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating Nuclear Radiation Penetration,
    Beer-Lambert-Bouguer Attenuation, Half-Value Layer (HVL), and Geiger Counter detection.
    """

    # Signals
    radiation_computed = Signal(dict)
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
        self._active_preset_code: str = "COBALT_60_GAMMA_LEAD_SHIELDING"
        self._preset: RadiationPreset = RADIATION_PRESETS[self._active_preset_code]

        self._thickness_cm: float = self._preset.default_thickness_cm
        self._i0_cps: float = self._preset.default_intensity_cps
        self._custom_mu_cm_inv: float = self._preset.linear_mu_cm_inv

        # Particle / wave emission stream
        self._particles: list[dict] = []
        self._init_particles(32)

        # Geiger tube detector click pulse state
        self._gm_click_timer: float = 0.0

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None

        # Timer for 60fps real-time radiation animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def _init_particles(self, count: int) -> None:
        """Initializes simulated radiation particles traveling from source to detector."""
        self._particles.clear()
        for _ in range(count):
            self._particles.append({
                "x_norm": random.uniform(0.0, 1.0),
                "y_jitter": random.uniform(-14.0, 14.0),
                "speed": random.uniform(0.015, 0.025),
                "alive": True,
            })

    # ------------------------------------------------------------------
    # Physics Properties & Analytical Laws
    # ------------------------------------------------------------------

    @property
    def linear_mu(self) -> float:
        """Linear attenuation coefficient mu in cm^-1."""
        return max(1e-4, self._custom_mu_cm_inv)

    @property
    def half_value_layer_cm(self) -> float:
        """
        Half-Value Layer (HVL):
        HVL = ln(2) / mu ≈ 0.693147 / mu
        """
        return math.log(2.0) / self.linear_mu

    @property
    def tenth_value_layer_cm(self) -> float:
        """
        Tenth-Value Layer (TVL):
        TVL = ln(10) / mu ≈ 2.302585 / mu
        """
        return math.log(10.0) / self.linear_mu

    @property
    def transmission_fraction(self) -> float:
        """
        Transmitted radiation fraction I(x) / I_0:
        For gamma/beta: exp(-mu * x)
        For alpha: complete cutoff at range boundary (0.012 cm in paper)
        """
        if self._preset.radiation_type == "alpha":
            # Alpha particles are completely stopped by paper (> 0.008 cm)
            if self._thickness_cm >= 0.008:
                return 0.0
            else:
                return max(0.0, 1.0 - (self._thickness_cm / 0.008))

        # Standard exponential attenuation:
        return math.exp(-self.linear_mu * max(0.0, self._thickness_cm))

    @property
    def transmitted_intensity_cps(self) -> float:
        """Transmitted intensity I(x) in counts per second."""
        return self._i0_cps * self.transmission_fraction

    @property
    def transmission_percent(self) -> float:
        """Percentage of radiation transmitted (%T)."""
        return self.transmission_fraction * 100.0

    @property
    def absorption_percent(self) -> float:
        """Percentage of radiation absorbed (%A)."""
        return 100.0 - self.transmission_percent

    @property
    def mass_attenuation_cm2_g(self) -> float:
        """Mass attenuation coefficient mu / rho in cm^2/g."""
        rho = max(1e-3, self._preset.density_g_cm3)
        return self.linear_mu / rho

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches curriculum scenario."""
        if preset_code in RADIATION_PRESETS:
            self._active_preset_code = preset_code
            self._preset = RADIATION_PRESETS[preset_code]
            self._thickness_cm = self._preset.default_thickness_cm
            self._i0_cps = self._preset.default_intensity_cps
            self._custom_mu_cm_inv = self._preset.linear_mu_cm_inv
            self._emit_computed_values()
            self.update()

    def set_thickness_cm(self, x_val: float) -> None:
        """Sets shield thickness x within [0.0, 15.0] cm."""
        self._thickness_cm = max(0.0, min(15.0, round(x_val, 3)))
        self._emit_computed_values()
        self.update()

    def set_intensity_cps(self, i0_val: float) -> None:
        """Sets incident source intensity I_0 within [50.0, 5000.0] cps."""
        self._i0_cps = max(50.0, min(5000.0, round(i0_val, 1)))
        self._emit_computed_values()
        self.update()

    def reset_defaults(self) -> None:
        """Restores default values of current preset."""
        self.set_preset(self._active_preset_code)

    def _emit_computed_values(self) -> None:
        """Emits telemetry dictionary."""
        data = {
            "preset_code": self._active_preset_code,
            "isotope_name": self._preset.isotope_name,
            "radiation_type": self._preset.radiation_type,
            "energy_MeV": self._preset.energy_MeV,
            "absorber_material": self._preset.absorber_material,
            "thickness_cm": self._thickness_cm,
            "i0_cps": self._i0_cps,
            "i_trans_cps": self.transmitted_intensity_cps,
            "transmission_pct": self.transmission_percent,
            "hvl_cm": self.half_value_layer_cm,
            "tvl_cm": self.tenth_value_layer_cm,
            "linear_mu_cm_inv": self.linear_mu,
        }
        self.radiation_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Animation update tick for traveling radiation particles and detector response."""
        t_frac = self.transmission_fraction

        for p in self._particles:
            p["x_norm"] += p["speed"]

            # When particle enters absorber barrier (midpoint region x_norm from 0.40 to 0.65)
            if 0.40 <= p["x_norm"] <= 0.65:
                # If absorbed based on barrier transmission probability
                if random.random() > t_frac:
                    p["alive"] = False

            if p["x_norm"] > 1.0:
                p["x_norm"] = 0.0
                p["y_jitter"] = random.uniform(-14.0, 14.0)
                p["alive"] = True

        # Geiger tube ticker flash decay
        if self._gm_click_timer > 0.0:
            self._gm_click_timer = max(0.0, self._gm_click_timer - 0.05)
        elif random.random() < min(0.8, self.transmitted_intensity_cps / 400.0):
            self._gm_click_timer = 1.0

        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Interaction Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        """Arranges interactive control buttons in right panel."""
        ctrl_x = 390.0
        row1_y = 248.0
        btn_w_half = 72.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["type_cycle"] = QRectF(ctrl_x + 104.0, row1_y, 166.0, btn_h)

        row2_y = 278.0
        self._buttons["x_dec"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["x_inc"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["x_zero"] = QRectF(ctrl_x + 158.0, row2_y, 112.0, btn_h)

        row3_y = 308.0
        self._buttons["i_dec"] = QRectF(ctrl_x, row3_y, btn_w_half, btn_h)
        self._buttons["i_inc"] = QRectF(ctrl_x + 78.0, row3_y, btn_w_half, btn_h)
        self._buttons["add_hvl"] = QRectF(ctrl_x + 158.0, row3_y, 112.0, btn_h)

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

        # 1. Background Card
        self._paint_card_background(painter)

        # 2. Header and Title Banner
        self._paint_header(painter)

        # 3. Radiation Bench Schematic (Source -> Shield Barrier -> Geiger Counter)
        self._paint_radiation_bench(painter)

        # 4. Telemetry and Attenuation Curve Panel (Right Side)
        self._paint_telemetry_and_curve(painter)

        # 5. Control Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Frosted slate card background."""
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        """Renders title and radiation badge."""
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "☢ ĐỘ XUYÊN SÂU PHÓNG XẠ & ĐỊNH LUẬT SUY GIẢM HÀM MŨ")

        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        painter.drawText(20, 44, f"Nguồn: {self._preset.isotope_name} | Tấm chắn: {self._preset.absorber_material}")

        # Radiation Type Badge (Top Right)
        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        accent = self._preset.accent_color
        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        badge_txt = f"Tia {self._preset.particle_symbol} ({self._preset.radiation_type.capitalize()}) • {self._preset.energy_MeV:.2f} MeV"
        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(accent)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_radiation_bench(self, painter: QPainter) -> None:
        """
        Renders the experimental bench:
        1. Lead Collimator Housing with Trefoil ☢ logo on left
        2. Absorber Shield Slab with scaled thickness x in the middle
        3. Geiger-Müller (GM) tube detector with ticker on right
        4. Particle trajectory stream through the bench
        """
        bench_box = QRectF(20.0, 56.0, 350.0, 335.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(8, 12, 20, 200)))
        painter.drawRoundedRect(bench_box, 8.0, 8.0)

        # Bench floor rail
        rail_y = 220.0
        painter.setPen(QPen(QColor(40, 55, 75), 2.0))
        painter.drawLine(35.0, rail_y, 355.0, rail_y)

        # 1. Radioactive Source Collimator (Left: x = 35 to 85)
        src_rect = QRectF(35.0, rail_y - 65.0, 50.0, 65.0)
        src_grad = QLinearGradient(src_rect.topLeft(), src_rect.bottomRight())
        src_grad.setColorAt(0.0, QColor(70, 75, 85))
        src_grad.setColorAt(1.0, QColor(40, 45, 52))
        painter.setPen(QPen(QColor(100, 110, 125), 1.5))
        painter.setBrush(QBrush(src_grad))
        painter.drawRoundedRect(src_rect, 4.0, 4.0)

        # Trefoil symbol ☢ on source
        font_rad = QFont("Segoe UI", 16, QFont.Weight.Bold)
        painter.setFont(font_rad)
        painter.setPen(QColor(255, 200, 0))
        painter.drawText(src_rect, Qt.AlignmentFlag.AlignCenter, "☢")

        font_s = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_s)
        painter.setPen(QColor(255, 210, 80))
        painter.drawText(int(src_rect.left()), int(src_rect.top() - 8.0), f"Nguồn I₀={self._i0_cps:.0f}")

        # 2. Shielding Absorber Barrier (Center: x = 150 to 150 + thickness_px)
        # Visual thickness scales from 4px to 65px
        thick_px = max(4.0, min(75.0, self._thickness_cm * 6.5))
        barrier_x = 160.0
        barrier_rect = QRectF(barrier_x, rail_y - 75.0, thick_px, 75.0)

        mat_color = self._preset.accent_color
        bar_grad = QLinearGradient(barrier_rect.topLeft(), barrier_rect.bottomRight())
        if "Chì" in self._preset.absorber_material:
            bar_grad.setColorAt(0.0, QColor(85, 95, 110))
            bar_grad.setColorAt(1.0, QColor(50, 58, 70))
            border_c = QColor(140, 160, 185)
        elif "Bê tông" in self._preset.absorber_material:
            bar_grad.setColorAt(0.0, QColor(145, 140, 130))
            bar_grad.setColorAt(1.0, QColor(95, 90, 82))
            border_c = QColor(190, 180, 170)
        elif "Nhôm" in self._preset.absorber_material:
            bar_grad.setColorAt(0.0, QColor(180, 200, 215))
            bar_grad.setColorAt(1.0, QColor(120, 140, 155))
            border_c = QColor(210, 230, 245)
        else: # Paper or Water
            bar_grad.setColorAt(0.0, QColor(240, 240, 245))
            bar_grad.setColorAt(1.0, QColor(180, 190, 205))
            border_c = QColor(255, 255, 255)

        painter.setPen(QPen(border_c, 1.4))
        painter.setBrush(QBrush(bar_grad))
        painter.drawRoundedRect(barrier_rect, 3.0, 3.0)

        # Barrier label
        font_b = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_b)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(
            int(barrier_x - 10.0),
            int(barrier_rect.top() - 8.0),
            f"x = {self._thickness_cm:.2f} cm ({self._preset.absorber_material[:6]})",
        )

        # 3. Geiger-Müller (GM) Tube Detector (Right: x = 285 to 345)
        gm_rect = QRectF(285.0, rail_y - 50.0, 58.0, 38.0)
        gm_grad = QLinearGradient(gm_rect.topLeft(), gm_rect.bottomRight())
        gm_grad.setColorAt(0.0, QColor(30, 40, 55))
        gm_grad.setColorAt(1.0, QColor(15, 22, 32))
        painter.setPen(QPen(QColor(0, 220, 180), 1.4))
        painter.setBrush(QBrush(gm_grad))
        painter.drawRoundedRect(gm_rect, 6.0, 6.0)

        # GM mica window face
        painter.setPen(QPen(QColor(255, 120, 0), 2.0))
        painter.drawLine(gm_rect.left(), gm_rect.top() + 6.0, gm_rect.left(), gm_rect.bottom() - 6.0)

        # Ticker LED Indicator
        led_color = QColor(255, 50, 50) if self._gm_click_timer > 0.3 else QColor(40, 15, 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(led_color))
        painter.drawEllipse(QPointF(gm_rect.right() - 10.0, gm_rect.top() + 10.0), 3.5, 3.5)

        # GM Detector label
        font_gm = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font_gm)
        painter.setPen(QColor(0, 240, 200))
        painter.drawText(int(gm_rect.left()), int(gm_rect.top() - 8.0), "Ống Geiger-Müller")

        # 4. Traveling Particles and Attenuated Ray Streams
        self._paint_particles(painter, src_rect, barrier_rect, gm_rect)

        # 5. Bottom Digital Readout Panel (Transmitted Intensity & Dose)
        self._paint_digital_counter(painter, bench_box)

    def _paint_particles(self, painter: QPainter, src_r: QRectF, bar_r: QRectF, gm_r: QRectF) -> None:
        """Renders animated radiation beam particles traveling across the bench."""
        beam_y = (src_r.top() + src_r.bottom()) * 0.5
        x_start = src_r.right()
        x_end = gm_r.left()
        total_len = x_end - x_start

        accent = self._preset.accent_color
        rad_type = self._preset.radiation_type

        for p in self._particles:
            cur_x = x_start + p["x_norm"] * total_len
            cur_y = beam_y + p["y_jitter"]

            # If particle was absorbed inside barrier, don't draw past barrier
            if not p["alive"] and cur_x > bar_r.left():
                continue

            if rad_type == "alpha":
                # Heavy gold alpha sphere
                painter.setPen(QPen(QColor(255, 230, 100), 1.0))
                painter.setBrush(QBrush(accent))
                painter.drawEllipse(QPointF(cur_x, cur_y), 3.5, 3.5)
            elif rad_type == "beta":
                # Fast zigzag cyan electron
                painter.setPen(QPen(accent, 1.5))
                painter.drawLine(cur_x - 3.0, cur_y - 1.5, cur_x + 3.0, cur_y + 1.5)
            else: # Gamma photon sine wave snippet
                painter.setPen(QPen(accent, 1.4))
                wave_y = cur_y + 3.0 * math.sin(cur_x * 0.4)
                painter.drawPoint(QPointF(cur_x, wave_y))
                painter.drawEllipse(QPointF(cur_x, wave_y), 1.5, 1.5)

    def _paint_digital_counter(self, painter: QPainter, bench_box: QRectF) -> None:
        """Renders digital counter display at bottom of radiation bench."""
        disp_rect = QRectF(bench_box.left() + 15.0, bench_box.bottom() - 85.0, bench_box.width() - 30.0, 70.0)
        painter.setPen(QPen(QColor(0, 180, 150), 1.2))
        painter.setBrush(QBrush(QColor(6, 20, 24, 230)))
        painter.drawRoundedRect(disp_rect, 6.0, 6.0)

        i_trans = self.transmitted_intensity_cps
        pct_t = self.transmission_percent

        font_lbl = QFont("Segoe UI", 8, QFont.Weight.Bold)
        font_num = QFont("Consolas", 12, QFont.Weight.Bold)

        painter.setFont(font_lbl)
        painter.setPen(QColor(150, 210, 200))
        painter.drawText(int(disp_rect.left() + 14.0), int(disp_rect.top() + 22.0), "CƯỜNG ĐỘ BỨC XẠ TRUYỀN QUA (I):")

        painter.setFont(font_num)
        painter.setPen(QColor(0, 255, 200))
        painter.drawText(
            int(disp_rect.left() + 14.0),
            int(disp_rect.top() + 45.0),
            f"I(x) = {i_trans:6.1f} cps  ({pct_t:5.2f}% I₀)",
        )

        font_sm = QFont("Segoe UI", 7, QFont.Weight.Normal)
        painter.setFont(font_sm)
        painter.setPen(QColor(200, 230, 220))
        hvl_count = self._thickness_cm / self.half_value_layer_cm
        painter.drawText(
            int(disp_rect.left() + 14.0),
            int(disp_rect.top() + 62.0),
            f"Lớp chắn tương đương: {hvl_count:.2f} × HVL | Giảm liều: {self.absorption_percent:.1f}%",
        )

    def _paint_telemetry_and_curve(self, painter: QPainter) -> None:
        """Renders analytical parameters, formulas, and attenuation curve graph."""
        panel_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(panel_rect, 8.0, 8.0)

        font_hdr = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_hdr)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(panel_rect.left() + 12.0), int(panel_rect.top() + 20.0), "📊 THÔNG SỐ & ĐỊNH LUẬT SUY GIẢM")

        font_txt = QFont("Segoe UI", 8, QFont.Weight.Normal)
        font_val = QFont("Consolas", 8, QFont.Weight.Bold)

        metrics = [
            ("Định luật suy giảm:", "I(x) = I₀ · e^(−µ·x)"),
            ("Hệ số suy giảm µ:", f"{self.linear_mu:.4f} cm⁻¹"),
            ("Bề dày bán giảm (HVL):", f"{self.half_value_layer_cm:.3f} cm"),
            ("Bề dày 1/10 (TVL):", f"{self.tenth_value_layer_cm:.3f} cm"),
            ("Độ suy giảm khối (µ/ρ):", f"{self.mass_attenuation_cm2_g:.4f} cm²/g"),
            ("Tỷ lệ hấp thụ (%A):", f"{self.absorption_percent:.2f}%"),
            ("Năng lượng hạt/photon:", f"{self._preset.energy_MeV:.3f} MeV"),
        ]

        y_pos = panel_rect.top() + 38.0
        for label, val_str in metrics:
            painter.setFont(font_txt)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(panel_rect.left() + 12.0), int(y_pos), label)

            painter.setFont(font_val)
            painter.setPen(QColor(230, 245, 255))
            painter.drawText(int(panel_rect.left() + 130.0), int(y_pos), val_str)
            y_pos += 20.0

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive buttons."""
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "type_cycle": f"Kịch bản ({self._preset.particle_symbol})",
            "x_dec": "x: −0.5cm",
            "x_inc": "x: +0.5cm",
            "x_zero": "Tháo chắn (x=0)",
            "i_dec": "I₀: −100",
            "i_inc": "I₀: +100",
            "add_hvl": "+1 HVL chắn",
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
            elif btn_key == "add_hvl":
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
        preset_keys = list(RADIATION_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key in ("preset_prev",):
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key in ("preset_next", "type_cycle"):
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "x_dec":
            step = 0.002 if self._preset.radiation_type == "alpha" else 0.50
            self.set_thickness_cm(self._thickness_cm - step)
        elif btn_key == "x_inc":
            step = 0.002 if self._preset.radiation_type == "alpha" else 0.50
            self.set_thickness_cm(self._thickness_cm + step)
        elif btn_key == "x_zero":
            self.set_thickness_cm(0.0)
        elif btn_key == "i_dec":
            self.set_intensity_cps(self._i0_cps - 100.0)
        elif btn_key == "i_inc":
            self.set_intensity_cps(self._i0_cps + 100.0)
        elif btn_key == "add_hvl":
            self.set_thickness_cm(self._thickness_cm + self.half_value_layer_cm)
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "isotope_name": self._preset.isotope_name,
            "radiation_type": self._preset.radiation_type,
            "energy_MeV": self._preset.energy_MeV,
            "absorber_material": self._preset.absorber_material,
            "thickness_cm": self._thickness_cm,
            "i0_cps": self._i0_cps,
            "i_trans_cps": self.transmitted_intensity_cps,
            "transmission_pct": self.transmission_percent,
            "hvl_cm": self.half_value_layer_cm,
            "tvl_cm": self.tenth_value_layer_cm,
            "linear_mu_cm_inv": self.linear_mu,
        }
        self.stamp_requested.emit(data)
