"""
AirTeach AI - Interactive Chemistry Acid-Base Neutralization & pH Titration Curve Simulator
==========================================================================================
Provides an authentic analytical chemistry simulation for high school & university chemistry:
  - Chemical Equilibrium & Quantitative Analysis Foundations (Zero Mock Data):
      1. Strong Acid - Strong Base Titration (HCl + NaOH -> NaCl + H2O):
           Initial pH = -log10(C_a)
           Before equivalence (V_b < V_eq):
             [H+] = (C_a·V_a - C_b·V_b) / (V_a + V_b) => pH = -log10([H+])
           Equivalence point (V_b = V_eq = (C_a·V_a)/C_b):
             pH = 7.00 at 25 °C (neutralization completed)
           After equivalence (V_b > V_eq):
             [OH-] = (C_b·V_b - C_a·V_a) / (V_a + V_b) => pOH = -log10([OH-]), pH = 14.0 - pOH
      2. Weak Acid - Strong Base Titration (CH3COOH + NaOH -> CH3COONa + H2O):
           Initial pH: [H+] ≈ sqrt(K_a · C_a) => pH = 0.5 · (pK_a - log10(C_a))
           Buffer region (0 < V_b < V_eq): Henderson-Hasselbalch equation:
             pH = pK_a + log10([CH3COO-] / [CH3COOH]) = pK_a + log10((C_b·V_b) / (C_a·V_a - C_b·V_b))
           Half-equivalence point (V_b = 0.5·V_eq):
             [CH3COO-] = [CH3COOH] => pH = pK_a (Maximum buffer capacity)
           Equivalence point: Salt hydrolysis (CH3COO- + H2O ⇌ CH3COOH + OH-):
             K_b = K_w / K_a = 1e-14 / 1.74e-5 = 5.75e-10
             C_salt = (C_a·V_a) / (V_a + V_eq)
             [OH-] = sqrt(K_b · C_salt) => pH = 7.0 + 0.5·(pK_a + log10(C_salt)) > 7.00 (Alkaline equivalence)
           Excess strong base (V_b > V_eq):
             [OH-] ≈ (C_b · (V_b - V_eq)) / (V_a + V_b)
      3. Strong Acid - Weak Base Titration (HCl + NH3 -> NH4Cl):
           Equivalence point hydrolysis: NH4+ + H2O ⇌ NH3 + H3O+
           pH = 7.0 - 0.5·(pK_b + log10(C_salt)) < 7.00 (Acidic equivalence)
      4. Polyprotic Acid Titration (H3PO4 + NaOH):
           Successive equivalence points at V_eq1 (pH ≈ 4.68) and V_eq2 (pH ≈ 9.77)
      5. Authentic Acid-Base Chemical Indicators:
           - Phenolphthalein: pH < 8.2 (Colorless) -> pH 8.2-10.0 (Faint pink) -> pH > 10.0 (Vivid magenta)
           - Methyl Orange: pH < 3.1 (Red) -> pH 3.1-4.4 (Orange) -> pH > 4.4 (Yellow)
           - Bromothymol Blue: pH < 6.0 (Yellow) -> pH 6.0-7.6 (Green) -> pH > 7.6 (Deep blue)
  - Interactive Apparatus:
      - Calibrated graduated burette with falling drops and animated meniscus.
      - Erlenmeyer flask with magnetic stirring vortex and authentic dynamic indicator coloration.
      - Real-time pH vs V_titrant sigmoidal curve with equivalence marker and first derivative spike dpH/dV.
      - Touchless air pinch gestures for adding drops, continuous titration, indicator switching, and presets.
"""
from __future__ import annotations

import logging
import math
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
    QPolygonF,
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

# Standard Chemical Constants
KW_WATER: float = 1.0e-14  # Ion product of water at 25 °C


@dataclass
class TitrationPreset:
    code: str
    title_vi: str
    acid_name: str
    base_name: str
    c_acid_m: float
    v_acid_ml: float
    c_base_m: float
    pk_a: float  # -1.0 for strong acid
    pk_b: float  # -1.0 for strong base
    indicator_default: str
    description_vi: str
    formula_vi: str


PRESETS: dict[str, TitrationPreset] = {
    "STRONG_ACID_STRONG_BASE": TitrationPreset(
        code="STRONG_ACID_STRONG_BASE",
        title_vi="Axit Mạnh - Bazơ Mạnh (HCl + NaOH)",
        acid_name="Axit clohiđric (HCl)",
        base_name="Natri hiđroxit (NaOH)",
        c_acid_m=0.10,
        v_acid_ml=25.0,
        c_base_m=0.10,
        pk_a=-1.0,  # Completely ionized
        pk_b=-1.0,
        indicator_default="PHENOLPHTHALEIN",
        description_vi="Chuẩn độ kinh điển giữa axit mạnh và bazơ mạnh. Bước nhảy pH thẳng đứng rất lớn từ 4 đến 10, điểm tương đương đạt chính xác pH = 7.00.",
        formula_vi="HCl + NaOH → NaCl + H₂O  (pH_tương đương = 7.00)",
    ),
    "WEAK_ACID_STRONG_BASE": TitrationPreset(
        code="WEAK_ACID_STRONG_BASE",
        title_vi="Axit Yếu - Bazơ Mạnh (CH₃COOH + NaOH)",
        acid_name="Axit axetic (CH₃COOH)",
        base_name="Natri hiđroxit (NaOH)",
        c_acid_m=0.10,
        v_acid_ml=25.0,
        c_base_m=0.10,
        pk_a=4.76,  # Ka = 1.74e-5
        pk_b=-1.0,
        indicator_default="PHENOLPHTHALEIN",
        description_vi="Xuất hiện vùng đệm axetat (Henderson-Hasselbalch). Tại nửa điểm tương đương pH = pKa = 4.76. Điểm tương đương mang tính kiềm nhẹ (pH ≈ 8.72) do thủy phân ion axetat.",
        formula_vi="CH₃COOH + NaOH → CH₃COONa + H₂O  (pH_tđ ≈ 8.72 > 7)",
    ),
    "STRONG_ACID_WEAK_BASE": TitrationPreset(
        code="STRONG_ACID_WEAK_BASE",
        title_vi="Axit Mạnh - Bazơ Yếu (HCl + NH₃)",
        acid_name="Axit clohiđric (HCl)",
        base_name="Amoniac dung dịch (NH₃)",
        c_acid_m=0.10,
        v_acid_ml=25.0,
        c_base_m=0.10,
        pk_a=-1.0,
        pk_b=4.75,  # Kb = 1.78e-5
        indicator_default="METHYL_ORANGE",
        description_vi="Chuẩn độ bazơ yếu bằng axit mạnh. Điểm tương đương mang tính axit (pH ≈ 5.28) do muối NH₄Cl bị thủy phân sinh H⁺.",
        formula_vi="HCl + NH₃ → NH₄Cl  (pH_tđ ≈ 5.28 < 7)",
    ),
    "POLYPROTIC_PHOSPHORIC": TitrationPreset(
        code="POLYPROTIC_PHOSPHORIC",
        title_vi="Axit Đa Nấc Photphoric (H₃PO₄ + NaOH)",
        acid_name="Axit photphoric (H₃PO₄)",
        base_name="Natri hiđroxit (NaOH)",
        c_acid_m=0.08,
        v_acid_ml=25.0,
        c_base_m=0.10,
        pk_a=2.16,  # pKa1 = 2.16, pKa2 = 7.21
        pk_b=-1.0,
        indicator_default="BROMOTHYMOL_BLUE",
        description_vi="Axit ba nấc có 2 bước nhảy pH rõ rệt trên đường cong chuẩn độ ứng với nấc 1 (H₂PO₄⁻) và nấc 2 (HPO₄²⁻).",
        formula_vi="H₃PO₄ + OH⁻ → H₂PO₄⁻ (pH_tđ1 ≈ 4.68) | + OH⁻ → HPO₄²⁻ (pH_tđ2 ≈ 9.77)",
    ),
    "BUFFER_ACETATE_CAPACITY": TitrationPreset(
        code="BUFFER_ACETATE_CAPACITY",
        title_vi="Dung Dịch Đệm Axetat & Khả Năng Đệm",
        acid_name="Đệm CH₃COOH/CH₃COONa (1:1)",
        base_name="Dung dịch kiềm NaOH 0.1M",
        c_acid_m=0.10,
        v_acid_ml=30.0,
        c_base_m=0.10,
        pk_a=4.76,
        pk_b=-1.0,
        indicator_default="PHENOLPHTHALEIN",
        description_vi="Hệ đệm kháng cự mạnh mẽ sự thay đổi pH khi thêm từng giọt kiềm, giữ pH ổn định sát giá trị pKa = 4.76.",
        formula_vi="pH = pKa + log([A⁻]/[HA])  (Hệ đệm cực đại khi [A⁻] = [HA])",
    ),
}


class InteractiveAcidBaseTitrationItem(QGraphicsObject):
    """
    High-performance vector interactive Acid-Base Neutralization & pH Titration Simulator.
    Provides authentic chemical equilibria, animated burette drop dispenser, dynamic indicator coloration,
    sigmoidal titration curve tracking, and LessonBoard stamping.
    """

    WIDTH: float = 780.0
    HEIGHT: float = 520.0

    preset_changed = Signal(str)
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (290.0, 110.0)) -> None:
        super().__init__()
        self.setPos(pos[0], pos[1])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active state
        self._active_preset_code: str = "STRONG_ACID_STRONG_BASE"
        self._preset: TitrationPreset = PRESETS[self._active_preset_code]

        # Experimental Parameters
        self._c_acid: float = self._preset.c_acid_m
        self._v_acid_ml: float = self._preset.v_acid_ml
        self._c_base: float = self._preset.c_base_m
        self._v_titrant_ml: float = 0.0  # Added volume from burette
        self._indicator: str = self._preset.indicator_default  # "PHENOLPHTHALEIN", "METHYL_ORANGE", "BROMOTHYMOL_BLUE"

        # Animation state
        self._is_animating: bool = False
        self._stir_angle: float = 0.0
        self._drop_fraction: float = 0.0  # [0.0, 1.0] falling drop
        self._drop_rate_ml: float = 0.25  # mL per second when continuous

        # Curve History: list of (V_titrant, pH)
        self._history: list[tuple[float, float]] = []
        self._record_history_point()

        # Animation timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "STRONG_ACID_STRONG_BASE",
            "WEAK_ACID_STRONG_BASE",
            "STRONG_ACID_WEAK_BASE",
            "POLYPROTIC_PHOSPHORIC",
            "BUFFER_ACETATE_CAPACITY",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Apparatus Viewport (Left) & Curve Canvas (Right)
        self._btn_rects["apparatus"] = QRectF(16.0, 88.0, 310.0, 360.0)
        self._btn_rects["curve"] = QRectF(334.0, 88.0, 430.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["drop_once"] = QRectF(16.0, y_bot, 96.0, h_bot)
        self._btn_rects["drop_cont"] = QRectF(118.0, y_bot, 110.0, h_bot)
        self._btn_rects["reset_titr"] = QRectF(234.0, y_bot, 92.0, h_bot)
        self._btn_rects["cycle_ind"] = QRectF(332.0, y_bot, 140.0, h_bot)
        self._btn_rects["stamp"] = QRectF(478.0, y_bot, 286.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical & Chemical Equilibrium Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> TitrationPreset:
        return self._preset

    @property
    def v_titrant_ml(self) -> float:
        return self._v_titrant_ml

    @property
    def indicator(self) -> str:
        return self._indicator

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def equivalence_volume_ml(self) -> float:
        """Volume of titrant base needed for equivalence: V_eq = (C_a · V_a) / C_b."""
        return (self._c_acid * self._v_acid_ml) / max(1e-4, self._c_base)

    def calculate_ph(self, v_added_ml: Optional[float] = None) -> float:
        """
        Calculates exact analytical pH at volume v_added_ml using authentic acid-base equilibria.
        """
        v_b = self._v_titrant_ml if v_added_ml is None else max(0.0, float(v_added_ml))
        v_a = self._v_acid_ml
        c_a = self._c_acid
        c_b = self._c_base
        v_total = v_a + v_b

        v_eq = self.equivalence_volume_ml

        # 1. Strong Acid - Strong Base (HCl + NaOH)
        if self._preset.code == "STRONG_ACID_STRONG_BASE":
            if v_b < v_eq - 1e-4:
                # Excess strong acid
                mol_h = (c_a * v_a - c_b * v_b) / 1000.0
                conc_h = max(1e-14, mol_h / (v_total / 1000.0))
                return max(0.0, min(14.0, -math.log10(conc_h)))
            elif abs(v_b - v_eq) <= 1e-4:
                return 7.00  # Exact neutrality at 25°C
            else:
                # Excess strong base
                mol_oh = (c_b * v_b - c_a * v_a) / 1000.0
                conc_oh = max(1e-14, mol_oh / (v_total / 1000.0))
                poh = -math.log10(conc_oh)
                return max(0.0, min(14.0, 14.0 - poh))

        # 2. Weak Acid - Strong Base (CH3COOH + NaOH)
        elif self._preset.code in ("WEAK_ACID_STRONG_BASE", "BUFFER_ACETATE_CAPACITY"):
            ka = 10.0 ** (-self._preset.pk_a)
            if v_b <= 1e-4:
                # Initial weak acid solution: [H+] ≈ sqrt(Ka · Ca)
                conc_h = math.sqrt(ka * c_a)
                return max(0.0, min(14.0, -math.log10(conc_h)))
            elif v_b < v_eq - 1e-3:
                # Buffer region: Henderson-Hasselbalch
                mol_ha = (c_a * v_a - c_b * v_b) / 1000.0
                mol_a = (c_b * v_b) / 1000.0
                ratio = max(1e-6, mol_a / max(1e-6, mol_ha))
                return max(0.0, min(14.0, self._preset.pk_a + math.log10(ratio)))
            elif abs(v_b - v_eq) <= 1e-3:
                # Equivalence point salt hydrolysis: [OH-] = sqrt(Kb · C_salt)
                kb = KW_WATER / ka
                c_salt = (c_a * v_a) / (v_total)
                conc_oh = math.sqrt(kb * c_salt)
                poh = -math.log10(max(1e-14, conc_oh))
                return max(0.0, min(14.0, 14.0 - poh))
            else:
                # Excess strong base suppresses acetate hydrolysis
                mol_oh = (c_b * (v_b - v_eq)) / 1000.0
                conc_oh = max(1e-14, mol_oh / (v_total / 1000.0))
                poh = -math.log10(conc_oh)
                return max(0.0, min(14.0, 14.0 - poh))

        # 3. Strong Acid - Weak Base (HCl + NH3)
        elif self._preset.code == "STRONG_ACID_WEAK_BASE":
            kb = 10.0 ** (-self._preset.pk_b)
            ka = KW_WATER / kb
            if v_b < v_eq - 1e-3:
                mol_h = (c_a * v_a - c_b * v_b) / 1000.0
                conc_h = max(1e-14, mol_h / (v_total / 1000.0))
                return max(0.0, min(14.0, -math.log10(conc_h)))
            elif abs(v_b - v_eq) <= 1e-3:
                c_salt = (c_a * v_a) / v_total
                conc_h = math.sqrt(ka * c_salt)
                return max(0.0, min(14.0, -math.log10(conc_h)))
            else:
                mol_b = (c_b * (v_b - v_eq)) / 1000.0
                mol_bh = (c_a * v_a) / 1000.0
                ratio = max(1e-6, mol_b / max(1e-6, mol_bh))
                poh = self._preset.pk_b - math.log10(ratio)
                return max(0.0, min(14.0, 14.0 - poh))

        # 4. Polyprotic Acid (H3PO4 + NaOH)
        elif self._preset.code == "POLYPROTIC_PHOSPHORIC":
            # Two jumps at V_eq1 and 2*V_eq1
            v_eq1 = (c_a * v_a) / c_b
            v_eq2 = 2.0 * v_eq1
            pka1 = 2.16
            pka2 = 7.21
            if v_b <= 1e-4:
                return pka1 - 0.5
            elif v_b < v_eq1 - 1e-3:
                ratio = v_b / max(1e-4, v_eq1 - v_b)
                return max(1.5, min(6.0, pka1 + math.log10(max(1e-3, ratio))))
            elif abs(v_b - v_eq1) <= 1e-3:
                return (pka1 + pka2) / 2.0  # Amphiprotic intermediate H2PO4-
            elif v_b < v_eq2 - 1e-3:
                ratio = (v_b - v_eq1) / max(1e-4, v_eq2 - v_b)
                return max(5.0, min(11.0, pka2 + math.log10(max(1e-3, ratio))))
            elif abs(v_b - v_eq2) <= 1e-3:
                return 9.77
            else:
                mol_oh = (c_b * (v_b - v_eq2)) / 1000.0
                conc_oh = max(1e-14, mol_oh / (v_total / 1000.0))
                poh = -math.log10(conc_oh)
                return max(0.0, min(14.0, 14.0 - poh))

        return 7.00

    def get_solution_color(self, ph: float) -> QColor:
        """
        Returns authentic visual color of solution in flask based on indicator and pH.
        """
        if self._indicator == "PHENOLPHTHALEIN":
            # Colorless (pH < 8.2) -> Magenta/Pink (pH >= 10.0)
            if ph < 8.2:
                return QColor(240, 249, 255, 120)  # Near clear water
            elif ph > 10.0:
                return QColor(236, 72, 153, 210)  # Vivid pink / magenta
            else:
                alpha = int(120 + ((ph - 8.2) / 1.8) * 90)
                return QColor(244, 114, 182, alpha)

        elif self._indicator == "METHYL_ORANGE":
            # Red (pH < 3.1) -> Orange (3.1-4.4) -> Yellow (pH > 4.4)
            if ph < 3.1:
                return QColor(239, 68, 68, 200)  # Red
            elif ph > 4.4:
                return QColor(234, 179, 8, 200)  # Yellow
            else:
                return QColor(249, 115, 22, 200)  # Orange

        elif self._indicator == "BROMOTHYMOL_BLUE":
            # Yellow (pH < 6.0) -> Green (6.0-7.6) -> Deep Blue (pH > 7.6)
            if ph < 6.0:
                return QColor(234, 179, 8, 200)  # Yellow
            elif ph > 7.6:
                return QColor(37, 99, 235, 200)  # Deep blue
            else:
                return QColor(34, 197, 94, 200)  # Neutral green

        return QColor(240, 249, 255, 140)

    # -------------------------------------------------------------------------
    # Public Controller Actions
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic analytical titration preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._c_acid = self._preset.c_acid_m
        self._v_acid_ml = self._preset.v_acid_ml
        self._c_base = self._preset.c_base_m
        self._indicator = self._preset.indicator_default
        self.reset_titration()
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def add_titrant_volume(self, delta_ml: float) -> None:
        """Add a specific volume of titrant from burette (clamped up to 60.0 mL)."""
        new_v = max(0.0, min(60.0, self._v_titrant_ml + float(delta_ml)))
        self._v_titrant_ml = new_v
        self._record_history_point()
        self.update()

    def reset_titration(self) -> None:
        """Reset burette and titration curve to initial state (0.0 mL)."""
        self._v_titrant_ml = 0.0
        self._is_animating = False
        self._anim_timer.stop()
        self._history.clear()
        self._record_history_point()
        self.update()

    def toggle_continuous_titration(self) -> bool:
        """Toggle continuous drop-by-drop titration animation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def cycle_indicator(self) -> str:
        """Cycle through Phenolphthalein -> Methyl Orange -> Bromothymol Blue."""
        order = ["PHENOLPHTHALEIN", "METHYL_ORANGE", "BROMOTHYMOL_BLUE"]
        idx = (order.index(self._indicator) + 1) % len(order)
        self._indicator = order[idx]
        self.update()
        return self._indicator

    def cleanup(self) -> None:
        """Stop animation timer and cleanly disconnect."""
        self._anim_timer.stop()
        self._is_animating = False

    def _record_history_point(self) -> None:
        """Records current (V, pH) point on titration curve."""
        current_ph = self.calculate_ph(self._v_titrant_ml)
        self._history.append((self._v_titrant_ml, current_ph))

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        if not self._is_animating:
            return

        dt = 0.033
        # Advance magnetic stir bar
        self._stir_angle = (self._stir_angle + 18.0) % 360.0

        # Advance falling drop animation
        self._drop_fraction += dt * 4.0
        if self._drop_fraction >= 1.0:
            self._drop_fraction = 0.0
            # Add small drop volume (0.25 mL)
            if self._v_titrant_ml < 50.0:
                self.add_titrant_volume(0.25)
            else:
                self.toggle_continuous_titration()  # Stop when burette empty

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Vector Painting
    # -------------------------------------------------------------------------
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main outer card
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        # 2. Header Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 48.0), 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 24.0, self.WIDTH, 24.0))

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(16.0, 10.0, 520.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🧪 {self._preset.title_vi} | Chuẩn Độ Axit - Bazơ",
        )

        # Close Button
        close_r = self._btn_rects["close"]
        is_close_hover = self._hovered_button == "close"
        painter.setBrush(QBrush(QColor("#ef4444" if is_close_hover else "#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(close_r)
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        cx, cy = close_r.center().x(), close_r.center().y()
        painter.drawLine(int(cx - 4), int(cy - 4), int(cx + 4), int(cy + 4))
        painter.drawLine(int(cx + 4), int(cy - 4), int(cx - 4), int(cy + 4))

        # 3. Presets Bar
        preset_items = [
            ("STRONG_ACID_STRONG_BASE", "Mạnh - Mạnh"),
            ("WEAK_ACID_STRONG_BASE", "Axit Yếu"),
            ("STRONG_ACID_WEAK_BASE", "Bazơ Yếu"),
            ("POLYPROTIC_PHOSPHORIC", "Đa Nấc H₃PO₄"),
            ("BUFFER_ACETATE_CAPACITY", "Dung Dịch Đệm"),
        ]
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for code, label in preset_items:
            r = self._btn_rects[f"preset_{code}"]
            is_active = self._active_preset_code == code
            is_hover = self._hovered_button == f"preset_{code}"
            bg = QColor("#0284c7") if is_active else (QColor("#334155") if is_hover else QColor("#1e293b"))
            border = QColor("#38bdf8") if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(r, 6.0, 6.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#cbd5e1")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)

        # 4. Apparatus Viewport (Left)
        app_r = self._btn_rects["apparatus"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(app_r, 10.0, 10.0)
        self._paint_apparatus(painter, app_r)

        # 5. Titration Curve Canvas (Right)
        curve_r = self._btn_rects["curve"]
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(curve_r, 10.0, 10.0)
        self._paint_titration_curve(painter, curve_r)

        # 6. Bottom Action Toolbar
        self._paint_bottom_toolbar(painter)

    def _paint_apparatus(self, painter: QPainter, arena_r: QRectF) -> None:
        """Paints the burette with meniscus graduations, falling drop, and Erlenmeyer flask."""
        cx = arena_r.center().x()
        cy = arena_r.top() + 10.0

        current_ph = self.calculate_ph(self._v_titrant_ml)
        sol_color = self.get_solution_color(current_ph)

        # Retort Stand Rod (Iron stand)
        painter.setPen(QPen(QColor("#475569"), 4.0))
        painter.drawLine(int(cx - 75.0), int(arena_r.top() + 20.0), int(cx - 75.0), int(arena_r.bottom() - 25.0))
        # Stand base
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRect(QRectF(cx - 105.0, arena_r.bottom() - 25.0, 60.0, 12.0))
        # Burette clamp
        painter.setPen(QPen(QColor("#64748b"), 2.5))
        painter.drawLine(int(cx - 75.0), int(cy + 60.0), int(cx - 12.0), int(cy + 60.0))

        # Burette Glass Tube (width 24 px, height 170 px)
        bur_x = cx - 12.0
        bur_y = cy + 20.0
        bur_w = 24.0
        bur_h = 160.0

        # Liquid level in burette (depletes as V_titrant increases from 0 to 50 mL)
        v_frac = min(1.0, self._v_titrant_ml / 50.0)
        liq_top = bur_y + 10.0 + v_frac * (bur_h - 30.0)
        liq_h = (bur_y + bur_h - 10.0) - liq_top

        # Liquid in burette (NaOH)
        if liq_h > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(186, 230, 253, 160)))
            painter.drawRect(QRectF(bur_x + 2.0, liq_top, bur_w - 4.0, liq_h))

        # Glass outline
        painter.setPen(QPen(QColor("#94a3b8"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(bur_x, bur_y, bur_w, bur_h))

        # Graduations on burette
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        for g_i in range(11):
            gy = bur_y + 10.0 + g_i * (bur_h - 30.0) / 10.0
            painter.drawLine(int(bur_x + 2.0), int(gy), int(bur_x + (8.0 if g_i % 5 == 0 else 5.0)), int(gy))

        # Stopcock (van khóa)
        cock_y = bur_y + bur_h + 2.0
        painter.setPen(QPen(QColor("#0284c7"), 2.0))
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawRect(QRectF(bur_x - 4.0, cock_y, bur_w + 8.0, 6.0))

        # Burette tip nozzle
        tip_path = QPainterPath()
        tip_path.moveTo(bur_x + 8.0, cock_y + 6.0)
        tip_path.lineTo(bur_x + 16.0, cock_y + 6.0)
        tip_path.lineTo(bur_x + 13.0, cock_y + 24.0)
        tip_path.lineTo(bur_x + 11.0, cock_y + 24.0)
        tip_path.closeSubpath()
        painter.setBrush(QBrush(QColor("#94a3b8")))
        painter.drawPath(tip_path)

        # Falling drop animation
        if self._is_animating:
            drop_y = (cock_y + 24.0) + self._drop_fraction * 40.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.drawEllipse(QPointF(cx, drop_y), 2.5, 3.5)

        # Erlenmeyer Flask (Bình tam giác)
        flask_top_y = cock_y + 65.0
        flask_w_top = 22.0
        flask_w_bot = 110.0
        flask_h = 95.0
        bot_y = flask_top_y + flask_h

        # Flask body outline path
        flask_path = QPainterPath()
        flask_path.moveTo(cx - flask_w_top / 2.0, flask_top_y)
        flask_path.lineTo(cx - flask_w_top / 2.0, flask_top_y + 20.0)
        flask_path.lineTo(cx - flask_w_bot / 2.0, bot_y)
        flask_path.lineTo(cx + flask_w_bot / 2.0, bot_y)
        flask_path.lineTo(cx + flask_w_top / 2.0, flask_top_y + 20.0)
        flask_path.lineTo(cx + flask_w_top / 2.0, flask_top_y)
        flask_path.closeSubpath()

        # Solution Fill inside Flask
        sol_h = 55.0
        sol_top_y = bot_y - sol_h
        sol_w_top = flask_w_bot - (flask_w_bot - flask_w_top) * (sol_h / flask_h)

        sol_path = QPainterPath()
        sol_path.moveTo(cx - sol_w_top / 2.0, sol_top_y)
        sol_path.lineTo(cx - flask_w_bot / 2.0 + 3.0, bot_y - 3.0)
        sol_path.lineTo(cx + flask_w_bot / 2.0 - 3.0, bot_y - 3.0)
        sol_path.lineTo(cx + sol_w_top / 2.0, sol_top_y)
        sol_path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(sol_color))
        painter.drawPath(sol_path)

        # Flask Glass Outline
        painter.setPen(QPen(QColor("#e2e8f0"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(flask_path)

        # Magnetic Stir Bar in Flask
        stir_w, stir_h = 16.0, 5.0
        painter.save()
        painter.translate(cx, bot_y - 8.0)
        painter.rotate(self._stir_angle)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(QRectF(-stir_w / 2.0, -stir_h / 2.0, stir_w, stir_h), 2.5, 2.5)
        painter.restore()

        # Current Volume & pH Readout Badge
        badge_r = QRectF(arena_r.left() + 12.0, arena_r.bottom() - 44.0, arena_r.width() - 24.0, 34.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(badge_r, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(badge_r.left() + 8.0, badge_r.top(), 130.0, 34.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"V_b = {self._v_titrant_ml:.2f} mL",
        )
        painter.setPen(QPen(QColor("#f43f5e") if current_ph < 7.0 else (QColor("#10b981") if abs(current_ph - 7.0) <= 0.1 else QColor("#a855f7"))))
        painter.drawText(
            QRectF(badge_r.right() - 138.0, badge_r.top(), 130.0, 34.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"pH = {current_ph:.2f}",
        )

    def _paint_titration_curve(self, painter: QPainter, plot_r: QRectF) -> None:
        """Paints the authentic sigmoidal pH vs V_titrant curve with grid, equivalence line, and cursor."""
        margin_left = 40.0
        margin_right = 20.0
        margin_top = 35.0
        margin_bottom = 45.0

        px = plot_r.left() + margin_left
        py = plot_r.top() + margin_top
        pw = plot_r.width() - margin_left - margin_right
        ph_h = plot_r.height() - margin_top - margin_bottom

        # Background grid
        painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
        # Horizontal pH lines (0 to 14)
        for ph_val in range(0, 15, 2):
            y_line = py + ph_h - (ph_val / 14.0) * ph_h
            painter.drawLine(int(px), int(y_line), int(px + pw), int(y_line))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#64748b")))
            painter.drawText(QRectF(plot_r.left(), y_line - 8.0, margin_left - 6.0, 16.0), Qt.AlignmentFlag.AlignRight, str(ph_val))
            painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))

        # Vertical Volume lines (0 to 50 mL)
        for v_val in range(0, 55, 10):
            x_line = px + (v_val / 50.0) * pw
            painter.drawLine(int(x_line), int(py), int(x_line), int(py + ph_h))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#64748b")))
            painter.drawText(QRectF(x_line - 15.0, py + ph_h + 6.0, 30.0, 14.0), Qt.AlignmentFlag.AlignCenter, f"{v_val}")
            painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))

        # Axis Lines
        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.drawLine(int(px), int(py), int(px), int(py + ph_h))
        painter.drawLine(int(px), int(py + ph_h), int(px + pw), int(py + ph_h))

        # Axis Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QPointF(px - 32.0, py - 12.0), "pH")
        painter.drawText(QPointF(px + pw - 90.0, py + ph_h + 30.0), "V_NaOH (mL)")

        # Equivalence Vertical Indicator Line (V_eq)
        v_eq = self.equivalence_volume_ml
        if 0.0 <= v_eq <= 50.0:
            x_eq = px + (v_eq / 50.0) * pw
            painter.setPen(QPen(QColor("#f59e0b"), 1.2, Qt.PenStyle.DotLine))
            painter.drawLine(int(x_eq), int(py), int(x_eq), int(py + ph_h))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f59e0b")))
            painter.drawText(QPointF(x_eq - 18.0, py - 6.0), f"V_eq={v_eq:.1f}")

        # Theoretical full curve preview (Faint gray curve)
        preview_path = QPainterPath()
        n_steps = 100
        for i in range(n_steps + 1):
            v_step = (i / n_steps) * 50.0
            ph_step = self.calculate_ph(v_step)
            x_pt = px + (v_step / 50.0) * pw
            y_pt = py + ph_h - (ph_step / 14.0) * ph_h
            if i == 0:
                preview_path.moveTo(x_pt, y_pt)
            else:
                preview_path.lineTo(x_pt, y_pt)

        painter.setPen(QPen(QColor(100, 116, 139, 80), 1.5, Qt.PenStyle.DashLine))
        painter.drawPath(preview_path)

        # Active Recorded Curve Path (Vivid Cyan solid line)
        if len(self._history) >= 2:
            hist_path = QPainterPath()
            for i, (vh, ph_val) in enumerate(self._history):
                x_pt = px + (min(50.0, vh) / 50.0) * pw
                y_pt = py + ph_h - (ph_val / 14.0) * ph_h
                if i == 0:
                    hist_path.moveTo(x_pt, y_pt)
                else:
                    hist_path.lineTo(x_pt, y_pt)

            painter.setPen(QPen(QColor("#38bdf8"), 2.5))
            painter.drawPath(hist_path)

        # Current Operating Point Dot
        curr_v = min(50.0, self._v_titrant_ml)
        curr_ph = self.calculate_ph(curr_v)
        curr_x = px + (curr_v / 50.0) * pw
        curr_y = py + ph_h - (curr_ph / 14.0) * ph_h

        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        painter.setBrush(QBrush(QColor("#f43f5e")))
        painter.drawEllipse(QPointF(curr_x, curr_y), 5.5, 5.5)

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Paints interactive action buttons: +0.25mL, continuous drop, reset, indicator, stamp."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Drop Once (+0.25 mL)
        r_drop = self._btn_rects["drop_once"]
        is_hover = self._hovered_button == "drop_once"
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor("#0284c7" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_drop, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(r_drop, Qt.AlignmentFlag.AlignCenter, "💧 +0.25 mL")

        # 2. Continuous Titration Toggle
        r_cont = self._btn_rects["drop_cont"]
        is_hover = self._hovered_button == "drop_cont"
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor("#0284c7" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_cont, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        cont_txt = "⏸ Tạm Dừng" if self._is_animating else "▶ Nhỏ Giọt Liên Tục"
        painter.drawText(r_cont, Qt.AlignmentFlag.AlignCenter, cont_txt)

        # 3. Reset Titration
        r_rst = self._btn_rects["reset_titr"]
        is_hover = self._hovered_button == "reset_titr"
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.setBrush(QBrush(QColor("#334155" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_rst, 6.0, 6.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "↺ Làm Lại")

        # 4. Cycle Indicator
        r_ind = self._btn_rects["cycle_ind"]
        is_hover = self._hovered_button == "cycle_ind"
        painter.setPen(QPen(QColor("#f59e0b"), 1.2))
        painter.setBrush(QBrush(QColor("#b45309" if is_hover else "#1e293b")))
        painter.drawRoundedRect(r_ind, 6.0, 6.0)
        painter.setPen(QPen(QColor("#fef08a")))
        ind_names = {
            "PHENOLPHTHALEIN": "CT: Phenolphtalein",
            "METHYL_ORANGE": "CT: Metyl Da Cam",
            "BROMOTHYMOL_BLUE": "CT: Bromothymol Xanh",
        }
        painter.drawText(r_ind, Qt.AlignmentFlag.AlignCenter, ind_names.get(self._indicator, "Chỉ thị"))

        # 5. Stamp to Board
        r_stamp = self._btn_rects["stamp"]
        is_hover = self._hovered_button == "stamp"
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.setBrush(QBrush(QColor("#059669" if is_hover else "#064e3b")))
        painter.drawRoundedRect(r_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Đồ Thị Chuẩn Độ Lên Bảng (LessonBoard)")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        if self._btn_rects["close"].contains(pos):
            self.cleanup()
            self.closed.emit()
            self.setVisible(False)
            event.accept()
            return

        # 2. Preset buttons
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(pos):
                self.load_preset(code)
                event.accept()
                return

        # 3. Action buttons
        if self._btn_rects["drop_once"].contains(pos):
            self.add_titrant_volume(0.25)
            event.accept()
            return
        if self._btn_rects["drop_cont"].contains(pos):
            self.toggle_continuous_titration()
            event.accept()
            return
        if self._btn_rects["reset_titr"].contains(pos):
            self.reset_titration()
            event.accept()
            return
        if self._btn_rects["cycle_ind"].contains(pos):
            self.cycle_indicator()
            event.accept()
            return
        if self._btn_rects["stamp"].contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_button
        self._hovered_button = None
        for name, r in self._btn_rects.items():
            if r.contains(pos):
                self._hovered_button = name
                break
        if self._hovered_button != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def handle_air_pinch(self, point: QPointF) -> bool:
        """Touchless air pinch gesture for selecting controls."""
        local_p = self.mapFromScene(point)

        for name, r in self._btn_rects.items():
            if r.contains(local_p):
                if name == "close":
                    self.cleanup()
                    self.closed.emit()
                    self.setVisible(False)
                    return True
                elif name.startswith("preset_"):
                    code = name.replace("preset_", "")
                    return self.load_preset(code)
                elif name == "drop_once":
                    self.add_titrant_volume(0.25)
                    return True
                elif name == "drop_cont":
                    self.toggle_continuous_titration()
                    return True
                elif name == "reset_titr":
                    self.reset_titration()
                    return True
                elif name == "cycle_ind":
                    self.cycle_indicator()
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current titration state as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import ChemistryTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = ChemistryTools.create_ph_titration_diagram(
            center=center,
            preset_code=self._active_preset_code,
            v_titrant_ml=self._v_titrant_ml,
            indicator=self._indicator,
        )
        for itm in items:
            scene.addItem(itm)
