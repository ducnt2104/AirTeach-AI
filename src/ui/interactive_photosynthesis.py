"""
AirTeach AI - Interactive Biology Photosynthesis Light Reactions & Calvin-Benson Cycle Simulator
================================================================================================
Provides an authentic biological and biochemical simulation of oxygenic photosynthesis:
  - Theoretical Foundations (Zero Mock Data):
      1. Light-Dependent Reactions (Pha Sáng - Màng Thylakoid):
           - Photolysis of water at Oxygen-Evolving Complex (OEC, Mn₄CaO₅ cluster):
               2 H₂O + 4 hν → O₂↑ + 4 H⁺_lumen + 4 e⁻
           - Z-scheme Non-cyclic Electron Transport Chain (ETC):
               PS II (P680) → Pheophytin → Plastoquinone (PQ) → Cytochrome b₆f complex
               → Plastocyanin (PC) → PS I (P700) → Ferredoxin (Fd) → FNR (NADP⁺ reductase).
               Net reduction: NADP⁺ + 2 e⁻ + H⁺_stroma → NADPH
           - Chemiosmotic Photophosphorylation (Peter Mitchell / Paul Boyer Rotary Motor):
               Proton accumulation in lumen creates electrochemical proton gradient:
               Δp = Δψ - (2.303·R·T/F)·ΔpH  (Lumen pH ≈ 5.5, Stroma pH ≈ 8.0).
               Rotary F₀F₁-ATP synthase channels protons down gradient to phosphorylate ADP:
               ADP + P_i + 3 H⁺_lumen → ATP + H₂O + 3 H⁺_stroma
      2. Light-Independent Reactions / Calvin-Benson Cycle (Pha Tối - Stroma):
           - Stage 1: Carbon Fixation (Cố định CO₂):
               RuBP (Ribulose-1,5-bisphosphate, 5C) + CO₂ (1C) —[RuBisCO]→ 2 × 3-PGA (3-Phosphoglycerate, 3C)
           - Stage 2: Reduction (Khử):
               3-PGA + ATP → 1,3-BPG;  1,3-BPG + NADPH + H⁺ → G3P (Glyceraldehyde-3-phosphate, 3C)
           - Stage 3: Regeneration of RuBP (Tái sinh chất nhận):
               5 × G3P (15C) + 3 ATP → 3 × RuBP (15C) + 3 ADP + 2 P_i
           - Net stoichiometric synthesis of 1 Glucose (C₆H₁₂O₆):
               6 CO₂ + 18 ATP + 12 NADPH + 12 H₂O → C₆H₁₂O₆ + 18 ADP + 12 NADP⁺ + 6 H⁺ + 18 P_i
      3. Biochemical Kinematics (Farquhar-von Caemmerer-Berry FvCB & Non-Rectangular Hyperbola):
           - Electron transport rate J(I) = (α·I) / sqrt(1 + (α·I / J_max)²)
           - Rubisco-limited carboxylation W_c = V_cmax · (C_i - Γ*) / (C_i + K_c·(1 + O/K_o))
           - RuBP regeneration-limited rate W_j = J · (C_i - Γ*) / (4.5·C_i + 10.5·Γ*)
           - Gross assimilation A_gross = min(W_c, W_j)
           - Dark respiration R_d = 0.015 · V_cmax · 2.0^((T - 25) / 10)
           - Net assimilation A_net = A_gross - R_d  (μmol CO₂ · m⁻² · s⁻¹)
  - 5 Authentic Curriculum Presets:
      - SUN_PLANT_HIGH_LIGHT: Cây ưa sáng (C3 Chuẩn) đạt đỉnh quang hợp dưới nắng mạnh
      - SHADE_PLANT_LOW_LIGHT: Cây ưa bóng thích nghi tán rừng, điểm bù ánh sáng thấp
      - DROUGHT_STOMATAL_CLOSURE: Hạn hán đóng khí khổng, CO₂ cạn kiệt, tăng quang hô hấp
      - C4_MAIZE_CONCENTRATING: Thực vật C4 (Ngô) bơm CO₂ tế bào bao bó mạch loại bỏ quang hô hấp
      - GREENHOUSE_ENRICHED_CO2: Nhà kính công nghệ cao làm giàu CO₂ tăng gấp đôi sinh khối
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Biological & Physical Constants
# -----------------------------------------------------------------------------
FARADAY_F = 96485.3321  # C / mol
GAS_CONSTANT_R = 8.31446  # J / (mol·K)
OXYGEN_CONC_MMOL = 210.0  # mmol / mol (Ambient O2)
RUBISCO_KC_UMOL = 404.0  # μmol / mol (Michaelis constant for CO2)
RUBISCO_KO_MMOL = 248.0  # mmol / mol (Inhibition constant for O2)
GAMMA_STAR_25C = 42.75  # μmol / mol (CO2 compensation point without respiration)
QUANTUM_YIELD_ALPHA = 0.38  # Effective quantum yield of electron transport


@dataclass(frozen=True)
class PhotosynthesisPreset:
    """Biochemical and environmental parameter preset for photosynthesis simulator."""

    code: str
    title_vi: str
    plant_type: str  # "C3_SUN", "C3_SHADE", "C3_DROUGHT", "C4_MAIZE", "C3_GREENHOUSE"
    light_intensity: float  # μmol photons · m⁻² · s⁻¹
    co2_ppm: float  # ppm (μmol CO2 / mol air)
    temp_c: float  # degrees Celsius
    vcmax: float  # μmol CO2 · m⁻² · s⁻¹ (Maximum Rubisco carboxylation capacity)
    jmax: float  # μmol e⁻ · m⁻² · s⁻¹ (Maximum electron transport capacity)
    stomata_conductance: float  # mol air · m⁻² · s⁻¹
    description_vi: str
    formula_vi: str


PRESETS: dict[str, PhotosynthesisPreset] = {
    "SUN_PLANT_HIGH_LIGHT": PhotosynthesisPreset(
        code="SUN_PLANT_HIGH_LIGHT",
        title_vi="Cây Ưa Sáng — Điểm Bão Hòa Quang Hợp Cao (Sun Plant C3)",
        plant_type="C3_SUN",
        light_intensity=1200.0,
        co2_ppm=420.0,
        temp_c=25.0,
        vcmax=95.0,
        jmax=180.0,
        stomata_conductance=0.35,
        description_vi="Cây ưa sáng phiến lá dày, nhiều lớp tế bào mô giậu, hàm lượng diệp lục a/b cao, quang hợp cực đại dưới ánh nắng chói chang.",
        formula_vi="6 CO₂ + 12 H₂O + Ánh sáng —[Diệp lục]→ C₆H₁₂O₆ + 6 O₂↑ + 6 H₂O",
    ),
    "SHADE_PLANT_LOW_LIGHT": PhotosynthesisPreset(
        code="SHADE_PLANT_LOW_LIGHT",
        title_vi="Cây Ưa Bóng — Điểm Bù Ánh Sáng Thấp (Shade Plant C3)",
        plant_type="C3_SHADE",
        light_intensity=150.0,
        co2_ppm=420.0,
        temp_c=22.0,
        vcmax=40.0,
        jmax=85.0,
        stomata_conductance=0.18,
        description_vi="Cây ưa bóng dưới tán rừng lá mỏng, hạt lục lạp to, tỉ lệ diệp lục b cao giúp hấp thụ hiệu quả ánh sáng tán xạ bước sóng ngắn.",
        formula_vi="I_c ≈ 15-20 μmol/m²/s  (Điểm bù ánh sáng cực thấp, tích lũy sinh khối hiệu quả)",
    ),
    "DROUGHT_STOMATAL_CLOSURE": PhotosynthesisPreset(
        code="DROUGHT_STOMATAL_CLOSURE",
        title_vi="Hạn Hán & Khí Khổng Đóng — Quang Hô Hấp Tăng (Photorespiration)",
        plant_type="C3_DROUGHT",
        light_intensity=1400.0,
        co2_ppm=160.0,
        temp_c=35.0,
        vcmax=75.0,
        jmax=140.0,
        stomata_conductance=0.04,
        description_vi="Thiếu nước kích hoạt hormone ABA làm khí khổng đóng chặt tránh mất hơi nước. CO₂ cạn kiệt, RuBisCO chuyển sang gắn O₂ gây lãng phí năng lượng.",
        formula_vi="RuBP + O₂ —[RuBisCO Oxygenase]→ 3-PGA + 2-Phosphoglycolate (Tiêu tốn ATP/NADPH)",
    ),
    "C4_MAIZE_CONCENTRATING": PhotosynthesisPreset(
        code="C4_MAIZE_CONCENTRATING",
        title_vi="Thực Vật C4 (Ngô/Mía) — Bơm CO₂ Hatch-Slack Không Quang Hô Hấp",
        plant_type="C4_MAIZE",
        light_intensity=1600.0,
        co2_ppm=420.0,
        temp_c=35.0,
        vcmax=125.0,
        jmax=245.0,
        stomata_conductance=0.30,
        description_vi="Giải phẫu lá Kranz: Enzim PEP-carboxylase cố định CO₂ ở tế bào mô giậu thành axit 4C rồi bơm nồng độ cao vào tế bào bao bó mạch, triệt tiêu quang hô hấp.",
        formula_vi="PEP (3C) + HCO₃⁻ —[PEP-Carboxylase]→ OAA (4C) → Malate → Bơm CO₂ vào bao bó mạch",
    ),
    "GREENHOUSE_ENRICHED_CO2": PhotosynthesisPreset(
        code="GREENHOUSE_ENRICHED_CO2",
        title_vi="Nhà Kính Công Nghệ Cao Làm Giàu CO₂ 1000 ppm (CO₂ Enrichment)",
        plant_type="C3_GREENHOUSE",
        light_intensity=1200.0,
        co2_ppm=1000.0,
        temp_c=28.0,
        vcmax=110.0,
        jmax=220.0,
        stomata_conductance=0.38,
        description_vi="Nông nghiệp chính xác tăng nồng độ CO₂ lên 1000 ppm trong nhà kính khử hoàn toàn giới hạn cơ chất của RuBisCO, tăng vọt 50-80% năng suất cây trồng.",
        formula_vi="[CO₂] = 1000 ppm >> K_m(CO₂) → Bão hòa RuBisCO, tối đa hóa tổng hợp Glucose",
    ),
}


class InteractivePhotosynthesisItem(QGraphicsObject):
    """
    Interactive PySide6 QGraphicsItem simulating both Thylakoid Light Reactions
    (PS II, ETC, PS I, ATP Synthase) and Stroma Calvin-Benson Cycle with authentic FvCB kinetics.
    """

    WIDTH = 720.0
    HEIGHT = 500.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: Union[tuple[float, float], QPointF] = (220.0, 90.0)) -> None:
        super().__init__()
        if isinstance(pos, QPointF):
            self.setPos(pos)
        elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
            self.setPos(float(pos[0]), float(pos[1]))
        else:
            self.setPos(220.0, 90.0)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active Preset and Parameters
        self._active_preset_code: str = "SUN_PLANT_HIGH_LIGHT"
        self._preset: PhotosynthesisPreset = PRESETS[self._active_preset_code]

        self._light_intensity: float = self._preset.light_intensity
        self._co2_ppm: float = self._preset.co2_ppm
        self._temp_c: float = self._preset.temp_c
        self._vcmax: float = self._preset.vcmax
        self._jmax: float = self._preset.jmax
        self._gs: float = self._preset.stomata_conductance

        # Computed Physiological Rates
        self._a_net: float = 0.0
        self._a_gross: float = 0.0
        self._r_dark: float = 0.0
        self._j_rate: float = 0.0
        self._o2_rate: float = 0.0
        self._atp_rate: float = 0.0
        self._glucose_rate: float = 0.0
        self._ci_ppm: float = 0.0
        self._recompute_kinetics()

        # Animation State
        self._anim_time_s: float = 0.0
        self._rotor_angle_deg: float = 0.0
        self._calvin_angle_deg: float = 0.0
        self._is_animating: bool = True

        # Flowing particles: list of (x, y, speed, type)
        # types: "photon", "electron", "proton_h", "o2_bubble", "sugar"
        self._particles: list[dict] = []
        self._init_particles()

        # Animation timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Initialize click and air-pinch bounding boxes."""
        w, h = self.WIDTH, self.HEIGHT
        self._btn_rects.clear()

        # Header close button
        self._btn_rects["close"] = QRectF(w - 36.0, 10.0, 26.0, 26.0)

        # 5 Preset Buttons (top toolbar)
        p_keys = list(PRESETS.keys())
        p_w = (w - 28.0) / len(p_keys)
        for i, k in enumerate(p_keys):
            self._btn_rects[f"preset_{k}"] = QRectF(14.0 + i * p_w, 44.0, p_w - 4.0, 24.0)

        # Bottom toolbar buttons
        y_bot = h - 42.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 95.0, 30.0)
        self._btn_rects["dec_light"] = QRectF(118.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_light"] = QRectF(152.0, y_bot, 32.0, 30.0)
        self._btn_rects["dec_co2"] = QRectF(258.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_co2"] = QRectF(292.0, y_bot, 32.0, 30.0)
        self._btn_rects["dec_temp"] = QRectF(398.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_temp"] = QRectF(432.0, y_bot, 32.0, 30.0)
        self._btn_rects["stamp"] = QRectF(w - 128.0, y_bot, 112.0, 30.0)

    def _init_particles(self) -> None:
        """Seed initial animated particles for light and stroma cycles."""
        self._particles.clear()
        # Photons streaming into PSII & PSI
        for i in range(8):
            self._particles.append({
                "type": "photon",
                "x": 40.0 + (i % 4) * 20.0,
                "y": 100.0 + i * 15.0,
                "vx": 0.5,
                "vy": 2.2,
                "phase": i * 0.4,
            })
        # O2 bubbles rising from OEC
        for i in range(6):
            self._particles.append({
                "type": "o2_bubble",
                "x": 65.0 + (i % 3) * 12.0,
                "y": 260.0 - i * 20.0,
                "vy": -1.2,
                "phase": i * 0.6,
            })
        # Protons H+ accumulating in lumen
        for i in range(12):
            self._particles.append({
                "type": "proton_h",
                "x": 80.0 + (i * 22.0) % 240.0,
                "y": 280.0 + (i * 11.0) % 60.0,
                "vx": 0.4 * ((i % 2) * 2 - 1),
                "vy": 0.3 * (((i + 1) % 2) * 2 - 1),
            })

    def _recompute_kinetics(self) -> None:
        """Computes authentic Farquhar-von Caemmerer-Berry (FvCB) photosynthetic rate."""
        # 1. Internal CO2 Ci estimation from ambient CO2 and stomatal conductance
        # In C3 plants, Ci/Ca ≈ 0.7 under normal conditions, dropping under drought
        ratio_ci_ca = max(0.20, min(0.85, 0.70 * (self._gs / 0.35)))
        if self._preset.plant_type == "C4_MAIZE":
            # C4 Hatch-Slack pump concentrates CO2 inside bundle sheath to 1200+ ppm
            self._ci_ppm = max(self._co2_ppm * 2.2, 850.0)
        else:
            self._ci_ppm = self._co2_ppm * ratio_ci_ca

        # Temperature response functions (Arrhenius/Q10)
        tk = self._temp_c + 273.15
        temp_factor = 2.0 ** ((self._temp_c - 25.0) / 10.0)
        vcmax_t = self._vcmax * temp_factor * max(0.1, 1.0 - ((self._temp_c - 28.0) / 18.0) ** 2)
        jmax_t = self._jmax * temp_factor * max(0.1, 1.0 - ((self._temp_c - 25.0) / 17.0) ** 2)

        # 2. Electron Transport Rate J(I) via non-rectangular hyperbola
        i_flux = self._light_intensity
        alpha_eff = QUANTUM_YIELD_ALPHA
        self._j_rate = (alpha_eff * i_flux) / math.sqrt(1.0 + ((alpha_eff * i_flux) / max(1.0, jmax_t)) ** 2)

        # 3. Rubisco carboxylation kinetics
        ci = max(0.0, self._ci_ppm)
        gamma_star = GAMMA_STAR_25C * temp_factor
        km_app = RUBISCO_KC_UMOL * (1.0 + OXYGEN_CONC_MMOL / RUBISCO_KO_MMOL)

        # Rubisco-limited carboxylation W_c
        if self._preset.plant_type == "C4_MAIZE":
            w_c = vcmax_t  # RuBisCO fully saturated in C4 bundle sheath
            w_j = self._j_rate / 4.0
        else:
            w_c = (vcmax_t * max(0.0, ci - gamma_star)) / max(1.0, ci + km_app)
            w_j = (self._j_rate * max(0.0, ci - gamma_star)) / max(1.0, 4.5 * ci + 10.5 * gamma_star)

        # 4. Gross and Net Assimilation
        self._a_gross = max(0.0, min(w_c, w_j))
        self._r_dark = max(0.2, 0.015 * vcmax_t)
        self._a_net = self._a_gross - self._r_dark

        # O2 production, ATP synthesis, and Glucose synthesis rates
        # 4 electrons per O2, 3 H+ per ATP, 6 CO2 per Glucose
        self._o2_rate = max(0.0, self._j_rate / 4.0)
        self._atp_rate = max(0.0, self._j_rate * 0.75)
        self._glucose_rate = max(0.0, self._a_net / 6.0)

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> PhotosynthesisPreset:
        return self._preset

    @property
    def light_intensity(self) -> float:
        return self._light_intensity

    @property
    def co2_ppm(self) -> float:
        return self._co2_ppm

    @property
    def temp_c(self) -> float:
        return self._temp_c

    @property
    def net_photosynthetic_rate(self) -> float:
        """Net carbon assimilation rate A_net (μmol CO2 · m⁻² · s⁻¹)."""
        return self._a_net

    @property
    def gross_photosynthetic_rate(self) -> float:
        """Gross carbon assimilation rate A_gross (μmol CO2 · m⁻² · s⁻¹)."""
        return self._a_gross

    @property
    def dark_respiration_rate(self) -> float:
        """Mitochondrial dark respiration rate R_d (μmol CO2 · m⁻² · s⁻¹)."""
        return self._r_dark

    @property
    def electron_transport_rate(self) -> float:
        """Linear electron transport rate J (μmol e⁻ · m⁻² · s⁻¹)."""
        return self._j_rate

    @property
    def o2_evolution_rate(self) -> float:
        """Oxygen gas evolution rate from water photolysis (μmol O2 · m⁻² · s⁻¹)."""
        return self._o2_rate

    @property
    def glucose_synthesis_rate(self) -> float:
        """Rate of net hexose/glucose synthesis (μmol C6H12O6 · m⁻² · s⁻¹)."""
        return self._glucose_rate

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    # -------------------------------------------------------------------------
    # Public Controller Actions
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic photosynthesis curriculum preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._light_intensity = self._preset.light_intensity
        self._co2_ppm = self._preset.co2_ppm
        self._temp_c = self._preset.temp_c
        self._vcmax = self._preset.vcmax
        self._jmax = self._preset.jmax
        self._gs = self._preset.stomata_conductance
        self._recompute_kinetics()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_light_intensity(self, intensity: float) -> None:
        """Set photon flux density (clamped between 0.0 and 2200.0 μmol · m⁻² · s⁻¹)."""
        self._light_intensity = max(0.0, min(2200.0, float(intensity)))
        self._recompute_kinetics()
        self.update()

    def set_co2_ppm(self, ppm: float) -> None:
        """Set ambient CO2 concentration (clamped between 50.0 and 1600.0 ppm)."""
        self._co2_ppm = max(50.0, min(1600.0, float(ppm)))
        self._recompute_kinetics()
        self.update()

    def set_temperature(self, temp_c: float) -> None:
        """Set leaf temperature (clamped between 5.0°C and 45.0°C)."""
        self._temp_c = max(5.0, min(45.0, float(temp_c)))
        self._recompute_kinetics()
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle live biochemical animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def stamp_to_board(self) -> None:
        """Stamps active Photosynthesis diagram as permanent vector items on LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import BiologyTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = BiologyTools.create_photosynthesis_diagram(
            center=center,
            preset_code=self._active_preset_code,
            light_intensity=self._light_intensity,
            co2_ppm=self._co2_ppm,
            temp_c=self._temp_c,
        )
        for itm in items:
            scene.addItem(itm)

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling for interactive control."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Control deck buttons
        if self._btn_rects["close"].contains(local_pos):
            self.closed.emit()
            self.setVisible(False)
            return True

        if self._btn_rects["play_pause"].contains(local_pos):
            self.toggle_animation()
            return True

        if self._btn_rects["dec_light"].contains(local_pos):
            self.set_light_intensity(self._light_intensity - 150.0)
            return True
        if self._btn_rects["inc_light"].contains(local_pos):
            self.set_light_intensity(self._light_intensity + 150.0)
            return True

        if self._btn_rects["dec_co2"].contains(local_pos):
            self.set_co2_ppm(self._co2_ppm - 50.0)
            return True
        if self._btn_rects["inc_co2"].contains(local_pos):
            self.set_co2_ppm(self._co2_ppm + 50.0)
            return True

        if self._btn_rects["dec_temp"].contains(local_pos):
            self.set_temperature(self._temp_c - 2.0)
            return True
        if self._btn_rects["inc_temp"].contains(local_pos):
            self.set_temperature(self._temp_c + 2.0)
            return True

        if self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Advance biochemical particle motions and rotary ATP synthase."""
        if not self._is_animating:
            return

        dt = 0.033
        self._anim_time_s += dt

        # Rotary speeds scaled by biochemical rates
        atp_speed = max(1.0, min(15.0, self._atp_rate * 0.15))
        self._rotor_angle_deg = (self._rotor_angle_deg + atp_speed * 11.5) % 360.0

        calvin_speed = max(0.5, min(8.0, self._a_net * 0.25))
        self._calvin_angle_deg = (self._calvin_angle_deg + calvin_speed * 4.3) % 360.0

        # Update flowing particles
        for p in self._particles:
            if p["type"] == "photon":
                p["y"] += p["vy"] * (self._light_intensity / 800.0)
                if p["y"] > 220.0:
                    p["y"] = 100.0
            elif p["type"] == "o2_bubble":
                p["y"] += p["vy"] * max(0.3, self._o2_rate * 0.1)
                if p["y"] < 130.0:
                    p["y"] = 260.0
            elif p["type"] == "proton_h":
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                if p["x"] < 50.0 or p["x"] > 300.0:
                    p["vx"] = -p["vx"]
                if p["y"] < 255.0 or p["y"] > 345.0:
                    p["vy"] = -p["vy"]

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#0a0f1d")))
        pen_border = QPen(QColor("#10b981"), 2.0) if self.isSelected() else QPen(QColor("#1e293b"), 1.6)
        painter.setPen(pen_border)
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Preset Selector Buttons
        self._paint_preset_toolbar(painter)

        # 4. Left Viewport: Thylakoid Membrane & Light Reactions (Pha Sáng)
        self._paint_thylakoid_viewport(painter)

        # 5. Right Viewport: Calvin-Benson Cycle (Pha Tối)
        self._paint_calvin_viewport(painter)

        # 6. Bottom Inspector Deck & Controls
        self._paint_bottom_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Header with badge and close icon."""
        # Icon / Badge
        badge_rect = QRectF(14.0, 12.0, 24.0, 24.0)
        painter.setBrush(QBrush(QColor("#10b981")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "🍃")

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(46.0, 29.0, "QUANG HỢP: PHA SÁNG THYLAKOID & CHU TRÌNH CALVIN-BENSON")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(480.0, 29.0, f"Mô hình FvCB | Loại: {self._preset.plant_type}")

        # Close button
        r_cl = self._btn_rects["close"]
        hov = (self._hovered_button == "close")
        painter.setBrush(QBrush(QColor("#dc2626") if hov else QColor("#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r_cl, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_cl, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_preset_toolbar(self, painter: QPainter) -> None:
        """5 Preset buttons with active indicators."""
        labels = {
            "SUN_PLANT_HIGH_LIGHT": "☀️ Cây Ưa Sáng",
            "SHADE_PLANT_LOW_LIGHT": "🌿 Cây Ưa Bóng",
            "DROUGHT_STOMATAL_CLOSURE": "🏜️ Hạn Hán Đóng Khổng",
            "C4_MAIZE_CONCENTRATING": "🌽 Thực Vật C4 (Ngô)",
            "GREENHOUSE_ENRICHED_CO2": "🏡 Nhà Kính CO₂ Cao",
        }
        for code, r in self._btn_rects.items():
            if not code.startswith("preset_"):
                continue
            pk = code.replace("preset_", "")
            is_act = (pk == self._active_preset_code)
            is_hov = (self._hovered_button == code)

            col_bg = QColor("#065f46") if is_act else (QColor("#1e293b") if is_hov else QColor("#0f172a"))
            col_border = QColor("#10b981") if is_act else QColor("#334155")
            painter.setBrush(QBrush(col_bg))
            painter.setPen(QPen(col_border, 1.2 if is_act else 0.8))
            painter.drawRoundedRect(r, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_act else QFont.Weight.Normal))
            painter.setPen(QColor("#34d399") if is_act else QColor("#cbd5e1"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, labels.get(pk, pk))

    def _paint_thylakoid_viewport(self, painter: QPainter) -> None:
        """Left viewport: Thylakoid lipid bilayer, PSII, PSI, ETC, and ATP synthase."""
        vp_rect = QRectF(14.0, 76.0, 340.0, 290.0)
        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        # Labels: Stroma (top) vs Thylakoid Lumen (bottom)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#6ee7b7"))
        painter.drawText(24.0, 94.0, "STROMA (NGOÀI MÀNG - pH ≈ 8.0)")
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(24.0, 355.0, "LUMEN XOANG THYLAKOID (pH ≈ 5.5 - TÍCH LŨY H⁺)")

        # Thylakoid Lipid Bilayer
        mem_y = 200.0
        mem_rect = QRectF(14.0, mem_y, 340.0, 40.0)
        painter.setBrush(QBrush(QColor(16, 185, 129, 35)))
        painter.setPen(QPen(QColor("#059669"), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(mem_rect)

        # 1. Photosystem II (P680)
        ps2_rect = QRectF(50.0, mem_y - 15.0, 45.0, 70.0)
        painter.setBrush(QBrush(QColor("#047857")))
        painter.setPen(QPen(QColor("#34d399"), 1.8))
        painter.drawRoundedRect(ps2_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(ps2_rect, Qt.AlignmentFlag.AlignCenter, "PS II\nP680")

        # Oxygen Evolving Complex (OEC) at lumen face of PSII
        oec_rect = QRectF(53.0, mem_y + 55.0, 39.0, 18.0)
        painter.setBrush(QBrush(QColor("#b91c1c")))
        painter.setPen(QPen(QColor("#f87171"), 1.0))
        painter.drawRoundedRect(oec_rect, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(oec_rect, Qt.AlignmentFlag.AlignCenter, "OEC Mn₄")

        # 2. Cytochrome b6f Complex (Proton Pump)
        cyt_rect = QRectF(125.0, mem_y - 10.0, 42.0, 60.0)
        painter.setBrush(QBrush(QColor("#1e3a8a")))
        painter.setPen(QPen(QColor("#60a5fa"), 1.5))
        painter.drawRoundedRect(cyt_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(cyt_rect, Qt.AlignmentFlag.AlignCenter, "Cyt\nb₆f")

        # 3. Photosystem I (P700)
        ps1_rect = QRectF(195.0, mem_y - 15.0, 45.0, 70.0)
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#6ee7b7"), 1.8))
        painter.drawRoundedRect(ps1_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(ps1_rect, Qt.AlignmentFlag.AlignCenter, "PS I\nP700")

        # 4. F0F1 - ATP Synthase Rotary Motor
        atp_x = 285.0
        # F0 channel in membrane
        f0_rect = QRectF(atp_x - 14.0, mem_y, 28.0, 40.0)
        painter.setBrush(QBrush(QColor("#78350f")))
        painter.setPen(QPen(QColor("#f59e0b"), 1.5))
        painter.drawRoundedRect(f0_rect, 4.0, 4.0)

        # F1 rotary catalytic knob in stroma
        f1_cx, f1_cy = atp_x, mem_y - 25.0
        f1_r = 18.0
        painter.setBrush(QBrush(QColor("#d97706")))
        painter.setPen(QPen(QColor("#fde68a"), 2.0))
        painter.drawEllipse(QPointF(f1_cx, f1_cy), f1_r, f1_r)

        # Rotor spokes
        painter.save()
        painter.translate(f1_cx, f1_cy)
        painter.rotate(self._rotor_angle_deg)
        painter.setPen(QPen(QColor("#451a03"), 2.5))
        painter.drawLine(-12, 0, 12, 0)
        painter.drawLine(0, -12, 0, 12)
        painter.restore()

        # ATP Badge
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(atp_x - 30.0, mem_y - 58.0, 60.0, 16.0), Qt.AlignmentFlag.AlignCenter, "⚡ ATP")

        # Animated streaming particles
        for p in self._particles:
            if p["type"] == "photon":
                painter.setBrush(QBrush(QColor("#fde047")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(p["x"], p["y"]), 3.0, 3.0)
            elif p["type"] == "o2_bubble":
                painter.setBrush(QBrush(QColor(56, 189, 248, 180)))
                painter.setPen(QPen(QColor("#38bdf8"), 1.0))
                painter.drawEllipse(QPointF(p["x"], p["y"]), 4.0, 4.0)
            elif p["type"] == "proton_h":
                painter.setBrush(QBrush(QColor("#fbbf24")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(p["x"], p["y"]), 2.2, 2.2)

    def _paint_calvin_viewport(self, painter: QPainter) -> None:
        """Right viewport: Calvin-Benson cycle metabolic ring and sugar export."""
        vp_rect = QRectF(366.0, 76.0, 340.0, 290.0)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(376.0, 94.0, "PHA TỐI: CHU TRÌNH CALVIN-BENSON (STROMA)")

        ccx, ccy = 536.0, 215.0
        cr = 75.0

        # Circular Cycle Pathway
        path_pen = QPen(QColor("#0284c7"), 3.0, Qt.PenStyle.SolidLine)
        painter.setPen(path_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(ccx, ccy), cr, cr)

        # Rotating phase markers around circle
        phases = [
            (0.0, "1. CỐ ĐỊNH CO₂\n(RuBisCO)", "#10b981"),
            (120.0, "2. KHỬ 3-PGA\n(+ATP/NADPH)", "#f59e0b"),
            (240.0, "3. TÁI SINH RuBP\n(+ATP)", "#a855f7"),
        ]
        for ang_offset, phase_txt, col_hex in phases:
            rad = math.radians(self._calvin_angle_deg + ang_offset)
            px = ccx + cr * math.cos(rad)
            py = ccy + cr * math.sin(rad)

            # Node bubble
            painter.setBrush(QBrush(QColor(col_hex)))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(QPointF(px, py), 9.0, 9.0)

        # RuBisCO enzyme badge in center
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawEllipse(QPointF(ccx, ccy), 34.0, 34.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QRectF(ccx - 30.0, ccy - 20.0, 60.0, 40.0), Qt.AlignmentFlag.AlignCenter, "RuBisCO\nEnzyme")

        # CO2 Inflow arrow at top
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ec4899"))
        painter.drawText(QRectF(ccx - 40.0, 108.0, 80.0, 16.0), Qt.AlignmentFlag.AlignCenter, f"⬇ CO₂: {self._ci_ppm:.0f} ppm")

        # G3P / Glucose Outflow Badge at bottom right
        sugar_rect = QRectF(ccx + 50.0, ccy + 55.0, 100.0, 36.0)
        painter.setBrush(QBrush(QColor("#065f46")))
        painter.setPen(QPen(QColor("#34d399"), 1.5))
        painter.drawRoundedRect(sugar_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(sugar_rect, Qt.AlignmentFlag.AlignCenter, f"🍬 Glucose C₆H₁₂O₆\n{self._glucose_rate:.2f} μmol/m²/s")

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Bottom HUD metrics and touchless parameter sliders/buttons."""
        y_hud = self.HEIGHT - 120.0
        hud_rect = QRectF(14.0, y_hud, self.WIDTH - 28.0, 72.0)
        painter.setBrush(QBrush(QColor("#050b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(hud_rect, 8.0, 8.0)

        # Real-time metrics grid
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(24.0, y_hud + 18.0, f"Cường độ quang hợp thuần (A_net): {self._a_net:.2f} μmol CO₂/m²/s")

        painter.setPen(QColor("#38bdf8"))
        painter.drawText(24.0, y_hud + 36.0, f"Quang hợp thô (A_gross): {self._a_gross:.2f}  |  Hô hấp tối (R_d): {self._r_dark:.2f}")

        painter.setPen(QColor("#fbbf24"))
        painter.drawText(24.0, y_hud + 54.0, f"Dòng e⁻ vận chuyển (J): {self._j_rate:.1f} μmol/m²/s  |  Tốc độ thoát O₂: {self._o2_rate:.2f} μmol O₂/m²/s")

        painter.setPen(QColor("#94a3b8"))
        painter.drawText(390.0, y_hud + 18.0, f"Quang năng I: {self._light_intensity:.0f} μmol/m²/s  |  CO₂: {self._co2_ppm:.0f} ppm  |  T: {self._temp_c:.1f}°C")
        painter.drawText(390.0, y_hud + 36.0, f"Công thức: {self._preset.formula_vi}")
        painter.drawText(390.0, y_hud + 54.0, f"Đặc tính: {self._preset.description_vi[:48]}…")

        # Bottom Buttons
        y_bot = self.HEIGHT - 42.0

        # Play / Pause
        r_play = self._btn_rects["play_pause"]
        is_hov_play = (self._hovered_button == "play_pause")
        painter.setBrush(QBrush(QColor("#0284c7") if is_hov_play else QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(r_play, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        txt_play = "⏸ Tạm Dừng" if self._is_animating else "▶ Tiếp Tục"
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, txt_play)

        # Light adjustments: - / +
        self._paint_stepper(painter, "dec_light", "inc_light", f"Ánh sáng: {self._light_intensity:.0f}", 118.0)

        # CO2 adjustments: - / +
        self._paint_stepper(painter, "dec_co2", "inc_co2", f"CO₂: {self._co2_ppm:.0f}", 258.0)

        # Temp adjustments: - / +
        self._paint_stepper(painter, "dec_temp", "inc_temp", f"Nhiệt độ: {self._temp_c:.0f}°C", 398.0)

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        is_hov_st = (self._hovered_button == "stamp")
        painter.setBrush(QBrush(QColor("#059669") if is_hov_st else QColor("#047857")))
        painter.setPen(QPen(QColor("#34d399"), 1.2))
        painter.drawRoundedRect(r_stamp, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Bảng")

    def _paint_stepper(self, painter: QPainter, btn_dec: str, btn_inc: str, label: str, x_start: float) -> None:
        """Helper to render decrement / increment stepper buttons."""
        r_dec = self._btn_rects[btn_dec]
        r_inc = self._btn_rects[btn_inc]

        # Dec button
        hov_d = (self._hovered_button == btn_dec)
        painter.setBrush(QBrush(QColor("#334155") if hov_d else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_dec, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_dec, Qt.AlignmentFlag.AlignCenter, "−")

        # Inc button
        hov_i = (self._hovered_button == btn_inc)
        painter.setBrush(QBrush(QColor("#334155") if hov_i else QColor("#1e293b")))
        painter.drawRoundedRect(r_inc, 4.0, 4.0)
        painter.drawText(r_inc, Qt.AlignmentFlag.AlignCenter, "+")

        # Label above/beside
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(x_start), int(r_dec.top() - 4.0), label)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        for key, r in self._btn_rects.items():
            if r.contains(pos):
                if key == "close":
                    self.closed.emit()
                    self.setVisible(False)
                    event.accept()
                    return
                elif key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                    event.accept()
                    return
                elif key == "play_pause":
                    self.toggle_animation()
                    event.accept()
                    return
                elif key == "dec_light":
                    self.set_light_intensity(self._light_intensity - 150.0)
                    event.accept()
                    return
                elif key == "inc_light":
                    self.set_light_intensity(self._light_intensity + 150.0)
                    event.accept()
                    return
                elif key == "dec_co2":
                    self.set_co2_ppm(self._co2_ppm - 50.0)
                    event.accept()
                    return
                elif key == "inc_co2":
                    self.set_co2_ppm(self._co2_ppm + 50.0)
                    event.accept()
                    return
                elif key == "dec_temp":
                    self.set_temperature(self._temp_c - 2.0)
                    event.accept()
                    return
                elif key == "inc_temp":
                    self.set_temperature(self._temp_c + 2.0)
                    event.accept()
                    return
                elif key == "stamp":
                    self.stamp_to_board()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        hovered = None
        for key, r in self._btn_rects.items():
            if r.contains(pos):
                hovered = key
                break
        if hovered != self._hovered_button:
            self._hovered_button = hovered
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_button is not None:
            self._hovered_button = None
            self.update()
        super().hoverLeaveEvent(event)

    def cleanup(self) -> None:
        """Stops background animation timer and frees resources."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
