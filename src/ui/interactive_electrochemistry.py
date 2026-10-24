"""
AirTeach AI - Interactive Chemistry Electrochemistry & Galvanic Voltaic / Electrolysis Cell Simulator
==================================================================================================
Provides an authentic interactive electrochemistry simulation for high school & university chemistry:
  - Theoretical & Thermodynamic Foundations (Zero Mock Data):
      1. Standard Reduction Potentials E° (vs Standard Hydrogen Electrode SHE):
           Zn²⁺ + 2e⁻ ⇌ Zn(s)   E° = -0.763 V (Anode oxidation)
           Cu²⁺ + 2e⁻ ⇌ Cu(s)   E° = +0.342 V (Cathode reduction)
           PbO₂ + 4H⁺ + SO₄²⁻ + 2e⁻ ⇌ PbSO₄ + 2H₂O  E° = +1.691 V
           PbSO₄ + 2e⁻ ⇌ Pb + SO₄²⁻   E° = -0.358 V
           2H⁺ + 2e⁻ ⇌ H₂(g)   E° = 0.000 V
           O₂ + 4H⁺ + 4e⁻ ⇌ 2H₂O  E° = +1.229 V
      2. Galvanic / Voltaic Cell Electromotive Force (Pin Galvani & Suất điện động):
           E°_cell = E°_cathode - E°_anode
           Nernst Equation at T = 298.15 K:
             E_cell = E°_cell - (R·T / (n·F)) · ln(Q) = E°_cell - (0.05916 / n) · log₁₀(Q)
           Gibbs Free Energy: ΔG° = -n · F · E°_cell < 0 (Spontaneous galvanic discharge)
      3. Electrolytic Cell & Faraday's Laws of Electrolysis (Điện phân & Định luật Faraday):
           Non-spontaneous driven by external DC source (E_applied > E_cell).
           Mass of deposited/dissolved substance:
             m = (A · I · t) / (n · F)
             where F = 96485.33 C/mol, I in Amperes, t in seconds, A is molar mass (g/mol).
           Faraday gas volume at STP:
             V_gas = (I · t / (n · F)) · 22.414 L/mol (or 24.79 L/mol at 25°C, 1 bar).
      4. Salt Bridge & Electroneutrality (Cầu muối & Cân bằng điện tích):
           U-tube salt bridge with inert electrolyte (KNO₃ or KCl):
           K⁺ cations migrate to cathode compartment, NO₃⁻ anions migrate to anode compartment.
  - 5 Authentic Curriculum Presets:
      - DANIELL_CELL_ZN_CU: Pin Daniell Zn-Cu kinh điển (E° = 1.105 V, ΔG° = -213.2 kJ)
      - LEAD_ACID_ACCUMULATOR: Ắc-quy chì-axit Pb-PbO₂ (E_cell ≈ 2.05 V, có chế độ Xả/Nạp)
      - ELECTROLYSIS_CUSO4: Điện phân dung dịch CuSO₄ với điện cực than chì Graphit
      - WATER_SPLITTING_PEM: Điện phân tách nước tạo khí H₂ và O₂ (Tỷ lệ V(H₂):V(O₂) = 2:1)
      - CONCENTRATION_CELL: Pin nồng độ Cu (0.01M || 1.0M, E_cell = 0.0592 V hiệu ứng Nernst)
  - Classroom Controls & Gestures:
      - Solution concentration sliders ([Zn²⁺], [Cu²⁺] from 0.001M to 3.0M).
      - Electrolysis current control (I = 0.5A to 5.0A).
      - Dynamic particle flow: electron dots on wire, ion migration in electrolyte, rising gas bubbles.
      - Play / Pause simulation toggle.
      - Vector diagram stamping onto LessonBoard.
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

# Standard Physical & Chemical Constants
FARADAY_F: float = 96485.33  # C/mol (Coulombs per mole of electrons)
GAS_CONST_R: float = 8.31446  # J/(mol·K)
TEMP_STANDARD_K: float = 298.15  # 25 °C in Kelvin
MOLAR_VOLUME_GAS_STP: float = 22.414  # L/mol at 0°C, 1 atm
MOLAR_MASS_ZN: float = 65.38  # g/mol
MOLAR_MASS_CU: float = 63.546  # g/mol
MOLAR_MASS_PB: float = 207.2  # g/mol
MOLAR_MASS_H2: float = 2.016  # g/mol
MOLAR_MASS_O2: float = 31.998  # g/mol


@dataclass
class ElectroPreset:
    code: str
    title_vi: str
    cell_type: str  # "GALVANIC", "ELECTROLYSIS", "CONCENTRATION"
    anode_material: str
    anode_half_rxn: str
    e0_anode: float  # V vs SHE
    cathode_material: str
    cathode_half_rxn: str
    e0_cathode: float  # V vs SHE
    n_electrons: int
    anode_sol_name: str
    anode_conc_m: float
    cathode_sol_name: str
    cathode_conc_m: float
    molar_mass_dep: float
    description_vi: str
    formula_vi: str


PRESETS: dict[str, ElectroPreset] = {
    "DANIELL_CELL_ZN_CU": ElectroPreset(
        code="DANIELL_CELL_ZN_CU",
        title_vi="Pin Galvani Daniell (Zn - Cu)",
        cell_type="GALVANIC",
        anode_material="Kẽm (Zn)",
        anode_half_rxn="Zn(s) → Zn²⁺ + 2e⁻",
        e0_anode=-0.763,
        cathode_material="Đồng (Cu)",
        cathode_half_rxn="Cu²⁺ + 2e⁻ → Cu(s)",
        e0_cathode=0.342,
        n_electrons=2,
        anode_sol_name="Dung dịch ZnSO₄",
        anode_conc_m=1.0,
        cathode_sol_name="Dung dịch CuSO₄",
        cathode_conc_m=1.0,
        molar_mass_dep=MOLAR_MASS_CU,
        description_vi="Pin Galvani cổ điển tạo dòng điện tự phát nhờ chênh lệch thế oxi hóa khử. Kẽm tan ở cực âm, đồng bám ở cực dương.",
        formula_vi="Zn + Cu²⁺ → Zn²⁺ + Cu  (E° = +1.105 V, ΔG° = -213.2 kJ)",
    ),
    "LEAD_ACID_ACCUMULATOR": ElectroPreset(
        code="LEAD_ACID_ACCUMULATOR",
        title_vi="Ắc-quy Chì-Axit (Lead-Acid Cell)",
        cell_type="GALVANIC",
        anode_material="Chì xốp (Pb)",
        anode_half_rxn="Pb + HSO₄⁻ → PbSO₄ + H⁺ + 2e⁻",
        e0_anode=-0.358,
        cathode_material="Chì đioxit (PbO₂)",
        cathode_half_rxn="PbO₂ + HSO₄⁻ + 3H⁺ + 2e⁻ → PbSO₄ + 2H₂O",
        e0_cathode=1.691,
        n_electrons=2,
        anode_sol_name="Dung dịch H₂SO₄",
        anode_conc_m=4.5,
        cathode_sol_name="Dung dịch H₂SO₄",
        cathode_conc_m=4.5,
        molar_mass_dep=MOLAR_MASS_PB,
        description_vi="Nguồn điện hóa học thứ cấp phổ biến trong công nghiệp ô tô, có khả năng sạc lại thuận nghịch hàng nghìn chu kỳ.",
        formula_vi="Pb + PbO₂ + 2H₂SO₄ ⇌ 2PbSO₄ + 2H₂O  (E_cell ≈ 2.05 V)",
    ),
    "ELECTROLYSIS_CUSO4": ElectroPreset(
        code="ELECTROLYSIS_CUSO4",
        title_vi="Điện Phân Dung Dịch CuSO₄ (Graphite)",
        cell_type="ELECTROLYSIS",
        anode_material="Graphit trơ (C)",
        anode_half_rxn="2H₂O → O₂↑ + 4H⁺ + 4e⁻",
        e0_anode=1.229,
        cathode_material="Graphit trơ (C)",
        cathode_half_rxn="Cu²⁺ + 2e⁻ → Cu(s)↓",
        e0_cathode=0.342,
        n_electrons=2,
        anode_sol_name="Dung dịch CuSO₄ + H₂SO₄",
        anode_conc_m=0.5,
        cathode_sol_name="Dung dịch CuSO₄",
        cathode_conc_m=1.0,
        molar_mass_dep=MOLAR_MASS_CU,
        description_vi="Điện phân cưỡng bức dưới nguồn điện một chiều. Đồng đỏ bám vào catot, khí oxi sủi bọt thoát ra ở anot.",
        formula_vi="2CuSO₄ + 2H₂O → 2Cu↓ + O₂↑ + 2H₂SO₄  (Định luật Faraday)",
    ),
    "WATER_SPLITTING_PEM": ElectroPreset(
        code="WATER_SPLITTING_PEM",
        title_vi="Điện Phân Nước Tạo Khí H₂ & O₂",
        cell_type="ELECTROLYSIS",
        anode_material="Bạch kim / Titan (Pt)",
        anode_half_rxn="2H₂O → O₂↑ + 4H⁺ + 4e⁻",
        e0_anode=1.229,
        cathode_material="Bạch kim / Niken (Pt)",
        cathode_half_rxn="2H⁺ + 2e⁻ → H₂↑",
        e0_cathode=0.000,
        n_electrons=4,
        anode_sol_name="Dung dịch KOH / H₂SO₄",
        anode_conc_m=1.0,
        cathode_sol_name="Dung dịch KOH / H₂SO₄",
        cathode_conc_m=1.0,
        molar_mass_dep=MOLAR_MASS_H2,
        description_vi="Sản xuất hydro xanh sạch cho năng lượng tương lai. Thể tích khí H₂ thu được ở catot gấp đôi thể tích khí O₂ ở anot.",
        formula_vi="2H₂O → 2H₂↑ + O₂↑  (V_H₂ : V_O₂ = 2 : 1)",
    ),
    "CONCENTRATION_CELL": ElectroPreset(
        code="CONCENTRATION_CELL",
        title_vi="Pin Nồng Độ Đồng (Cu 0.01M || 1.0M)",
        cell_type="CONCENTRATION",
        anode_material="Đồng (Cu pha loãng)",
        anode_half_rxn="Cu(s) → Cu²⁺(0.01M) + 2e⁻",
        e0_anode=0.342,
        cathode_material="Đồng (Cu đậm đặc)",
        cathode_half_rxn="Cu²⁺(1.0M) + 2e⁻ → Cu(s)",
        e0_cathode=0.342,
        n_electrons=2,
        anode_sol_name="CuSO₄ loãng 0.01M",
        anode_conc_m=0.01,
        cathode_sol_name="CuSO₄ đặc 1.0M",
        cathode_conc_m=1.0,
        molar_mass_dep=MOLAR_MASS_CU,
        description_vi="Hai điện cực cùng một kim loại nhúng vào hai nồng độ khác nhau. Suất điện động phát sinh thuần túy từ phương trình Nernst.",
        formula_vi="E = (0.0592 / 2) · log₁₀(C₂ / C₁) = 0.0592 V",
    ),
}


class InteractiveElectrochemistryItem(QGraphicsObject):
    """
    High-performance vector interactive Electrochemistry Galvanic & Electrolysis Simulator.
    Supports touchless air gestures, electron/ion particle animations, and LessonBoard stamping.
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
        self._active_preset_code: str = "DANIELL_CELL_ZN_CU"
        self._preset: ElectroPreset = PRESETS[self._active_preset_code]

        # Configurable Parameters
        self._anode_conc_m: float = self._preset.anode_conc_m
        self._cathode_conc_m: float = self._preset.cathode_conc_m
        self._current_amperes: float = 1.5  # Current in A
        self._elapsed_time_s: float = 0.0
        self._is_animating: bool = True

        # Dynamic particles for visualization
        # Electrons on top wire: list of position fractions [0.0, 1.0]
        self._electrons: list[float] = [i / 16.0 for i in range(16)]
        # Gas bubbles in beakers: list of (x, y, radius, speed)
        self._bubbles: list[list[float]] = []
        for i in range(18):
            self._bubbles.append([0.0, i / 18.0, 2.5 + (i % 3) * 1.0, 0.4 + (i % 4) * 0.15])

        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)  # ~30 FPS
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Pre-populate interactive hit areas for mouse and touchless air pinch."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)

        # Presets bar (5 buttons)
        preset_codes = [
            "DANIELL_CELL_ZN_CU",
            "LEAD_ACID_ACCUMULATOR",
            "ELECTROLYSIS_CUSO4",
            "WATER_SPLITTING_PEM",
            "CONCENTRATION_CELL",
        ]
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        # Main electrochemical cell canvas viewport
        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 470.0, 360.0)

        # Action toolbar at bottom
        y_bot = 460.0
        h_bot = 32.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 120.0, h_bot)
        self._btn_rects["dec_conc"] = QRectF(144.0, y_bot, 44.0, h_bot)
        self._btn_rects["inc_conc"] = QRectF(194.0, y_bot, 44.0, h_bot)
        self._btn_rects["dec_curr"] = QRectF(246.0, y_bot, 44.0, h_bot)
        self._btn_rects["inc_curr"] = QRectF(296.0, y_bot, 44.0, h_bot)
        self._btn_rects["reset_time"] = QRectF(348.0, y_bot, 110.0, h_bot)
        self._btn_rects["stamp"] = QRectF(466.0, y_bot, 298.0, h_bot)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    # -------------------------------------------------------------------------
    # Physical & Chemical Calculations (Zero Mock)
    # -------------------------------------------------------------------------
    @property
    def standard_emf_volts(self) -> float:
        """Standard cell potential E°_cell = E°_cathode - E°_anode."""
        return self._preset.e0_cathode - self._preset.e0_anode

    @property
    def cell_emf_volts(self) -> float:
        """
        Nernst equation for cell electromotive force at 298.15 K:
        E = E° - (0.05916 / n) * log10(Q)
        where Q = [Anode cation] / [Cathode cation]
        """
        if self._preset.cell_type == "ELECTROLYSIS":
            # For electrolysis, E_applied must overcome back-EMF
            return abs(self.standard_emf_volts) + 0.35  # Overpotential

        q_ratio = max(1e-5, self._anode_conc_m) / max(1e-5, self._cathode_conc_m)
        nernst_drop = (0.05916 / max(1, self._preset.n_electrons)) * math.log10(q_ratio)
        return self.standard_emf_volts - nernst_drop

    @property
    def gibbs_free_energy_kj(self) -> float:
        """Standard Gibbs free energy change ΔG° = -n * F * E° in kJ/mol."""
        delta_g_joules = -self._preset.n_electrons * FARADAY_F * self.standard_emf_volts
        return delta_g_joules / 1000.0  # kJ/mol

    @property
    def deposited_mass_grams(self) -> float:
        """
        Faraday's Law of Electrolysis: m = (A * I * t) / (n * F).
        Calculates mass deposited on cathode or dissolved from anode.
        """
        q_charge = self._current_amperes * self._elapsed_time_s
        n = max(1, self._preset.n_electrons)
        m = (self._preset.molar_mass_dep * q_charge) / (n * FARADAY_F)
        return m

    @property
    def evolved_gas_volume_liters(self) -> float:
        """Calculates volume of gas evolved at STP via Faraday's Law."""
        q_charge = self._current_amperes * self._elapsed_time_s
        n = max(1, self._preset.n_electrons)
        moles_gas = q_charge / (n * FARADAY_F)
        return moles_gas * MOLAR_VOLUME_GAS_STP

    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> ElectroPreset:
        return self._preset

    @property
    def current_amperes(self) -> float:
        return self._current_amperes

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def anode_conc_m(self) -> float:
        return self._anode_conc_m

    @property
    def cathode_conc_m(self) -> float:
        return self._cathode_conc_m

    def calculate_cell_potential(self) -> float:
        """Alias returning Nernst cell electromotive force in Volts."""
        return self.cell_emf_volts

    def calculate_deposited_mass_g(self) -> float:
        """Alias returning deposited mass in grams via Faraday's Law."""
        return self.deposited_mass_grams

    def calculate_gas_volumes_stp_liters(self) -> tuple[float, float]:
        """
        Calculates gas volumes evolved at cathode and anode at STP (Liters).
        For water splitting:
          Cathode: 2H+ + 2e- -> H2 (1 mol H2 per 2e-)
          Anode: 2H2O -> O2 + 4H+ + 4e- (1 mol O2 per 4e-)
          Ratio V(H2):V(O2) = 2:1
        """
        q = self._current_amperes * max(1.0, self._elapsed_time_s)
        # H2 at cathode: n=2
        moles_h2 = q / (2.0 * FARADAY_F)
        v_h2 = moles_h2 * MOLAR_VOLUME_GAS_STP
        # O2 at anode: n=4
        moles_o2 = q / (4.0 * FARADAY_F)
        v_o2 = moles_o2 * MOLAR_VOLUME_GAS_STP
        return (v_h2, v_o2)

    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic curriculum electrochemical preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._anode_conc_m = self._preset.anode_conc_m
        self._cathode_conc_m = self._preset.cathode_conc_m
        self._elapsed_time_s = 0.0
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def set_concentrations(self, anode_conc: float, cathode_conc: float) -> None:
        """Adjust electrolyte concentrations (clamped between 0.001M and 5.0M)."""
        self._anode_conc_m = max(0.001, min(5.0, float(anode_conc)))
        self._cathode_conc_m = max(0.001, min(5.0, float(cathode_conc)))
        self.update()

    def set_current(self, current_a: float) -> None:
        """Adjust electric current (clamped between 0.1A and 10.0A)."""
        self._current_amperes = max(0.1, min(10.0, float(current_a)))
        self.update()

    def toggle_animation(self) -> bool:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def reset_time(self) -> None:
        self._elapsed_time_s = 0.0
        self.update()

    def cleanup(self) -> None:
        self._anim_timer.stop()
        self._is_animating = False

    # -------------------------------------------------------------------------
    # Animation Ticks
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        if not self._is_animating:
            return

        dt = 0.033  # ~33 ms
        self._elapsed_time_s += dt * 5.0  # Time lapse multiplier 5x for classroom demonstration

        # Move electrons along circuit wire (Anode -> Cathode in Galvanic, Cathode -> Anode in Electrolysis)
        flow_speed = max(0.05, self._current_amperes * 0.08) * dt
        for i in range(len(self._electrons)):
            self._electrons[i] = (self._electrons[i] + flow_speed) % 1.0

        # Move gas bubbles upward
        for b in self._bubbles:
            b[1] -= b[3] * dt * 0.5
            if b[1] < 0.0:
                b[1] = 1.0

        self.update()

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------
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
        card_path = QPainterPath()
        card_path.addRoundedRect(bg_rect, 14.0, 14.0)
        painter.fillPath(card_path, QColor(15, 23, 42, 245))  # Deep Slate #0f172a
        painter.setPen(QPen(QColor(56, 189, 248, 180), 1.5))  # Sky cyan border
        painter.drawPath(card_path)

        # 2. Header
        self._paint_header(painter)

        # 3. Presets Bar
        self._paint_presets_bar(painter)

        # 4. Main Cell Viewport Canvas
        self._paint_cell_canvas(painter)

        # 5. Right Analytical Metrics Card
        self._paint_analytics_card(painter)

        # 6. Action Toolbar
        self._paint_action_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(
            QRectF(16.0, 12.0, 560.0, 24.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "⚡ ĐIỆN HÓA HỌC: PIN GALVANI, THẾ NERNST & ĐIỆN PHÂN FARADAY",
        )

        sub_font = QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(16.0, 34.0, 560.0, 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"Mô hình: {self._preset.title_vi}  •  {self._preset.formula_vi}",
        )

        # Close button
        close_rect = self._btn_rects["close"]
        is_hover = (self._hovered_button == "close")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(239, 68, 68, 200 if is_hover else 120))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

    def _paint_presets_bar(self, painter: QPainter) -> None:
        preset_items = [
            ("DANIELL_CELL_ZN_CU", "Pin Daniell (Zn-Cu)"),
            ("LEAD_ACID_ACCUMULATOR", "Ắc-quy Chì-Axit"),
            ("ELECTROLYSIS_CUSO4", "Điện Phân CuSO₄"),
            ("WATER_SPLITTING_PEM", "Điện Phân Nước"),
            ("CONCENTRATION_CELL", "Pin Nồng Độ Cu"),
        ]
        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font)

        for code, label in preset_items:
            rect = self._btn_rects[f"preset_{code}"]
            is_active = (self._active_preset_code == code)
            is_hover = (self._hovered_button == f"preset_{code}")

            if is_active:
                painter.setBrush(QColor(14, 165, 233, 220))  # Sky-500
                painter.setPen(QPen(QColor(255, 255, 255), 1.2))
            elif is_hover:
                painter.setBrush(QColor(51, 65, 85, 220))
                painter.setPen(QPen(QColor(56, 189, 248), 1.0))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _paint_cell_canvas(self, painter: QPainter) -> None:
        canvas_r = self._btn_rects["canvas"]

        # Viewport background
        painter.setBrush(QColor(10, 15, 26, 230))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(canvas_r, 10.0, 10.0)

        # Beaker geometry coordinates
        bw = 140.0
        bh = 170.0
        b_bottom = canvas_r.bottom() - 28.0
        b1_left = canvas_r.left() + 45.0
        b2_left = canvas_r.right() - 45.0 - bw

        # 1. Beaker 1 (Left - Anode compartment)
        self._paint_beaker(
            painter,
            QRectF(b1_left, b_bottom - bh, bw, bh),
            solution_name=self._preset.anode_sol_name,
            conc_str=f"{self._anode_conc_m:.2f} M",
            color_sol=QColor(56, 189, 248, 40) if "Zn" in self._preset.code else QColor(148, 163, 184, 40),
            is_anode=True,
        )

        # 2. Beaker 2 (Right - Cathode compartment)
        self._paint_beaker(
            painter,
            QRectF(b2_left, b_bottom - bh, bw, bh),
            solution_name=self._preset.cathode_sol_name,
            conc_str=f"{self._cathode_conc_m:.2f} M",
            color_sol=QColor(14, 165, 233, 90),  # Rich copper blue CuSO4
            is_anode=False,
        )

        # 3. U-Tube Salt Bridge connecting the two beakers
        self._paint_salt_bridge(painter, b1_left + bw - 25.0, b2_left + 25.0, b_bottom - bh + 20.0)

        # 4. Electrodes in Beakers
        # Anode plate (Zinc / Pb / Pt)
        anode_x = b1_left + 40.0
        anode_w = 20.0
        anode_top = b_bottom - bh - 20.0
        anode_h = bh - 10.0
        self._paint_electrode(painter, anode_x, anode_top, anode_w, anode_h, self._preset.anode_material, is_anode=True)

        # Cathode plate (Copper / PbO2 / Pt)
        cath_x = b2_left + bw - 60.0
        cath_w = 20.0
        cath_top = b_bottom - bh - 20.0
        cath_h = bh - 10.0
        self._paint_electrode(painter, cath_x, cath_top, cath_w, cath_h, self._preset.cathode_material, is_anode=False)

        # 5. External Circuit Wire & Voltmeter / Power Supply
        wire_y = canvas_r.top() + 35.0
        mid_wire_x = (anode_x + cath_x + cath_w) / 2.0

        painter.setPen(QPen(QColor(245, 158, 11), 2.2))  # Copper wire
        # From Anode top up to wire_y, over to meter, then to Cathode top
        wire_path = QPainterPath()
        wire_path.moveTo(anode_x + anode_w / 2.0, anode_top)
        wire_path.lineTo(anode_x + anode_w / 2.0, wire_y)
        wire_path.lineTo(cath_x + cath_w / 2.0, wire_y)
        wire_path.lineTo(cath_x + cath_w / 2.0, cath_top)
        painter.drawPath(wire_path)

        # Digital Voltmeter / DC Source at center
        meter_w = 80.0
        meter_h = 36.0
        meter_r = QRectF(mid_wire_x - meter_w / 2.0, wire_y - meter_h / 2.0, meter_w, meter_h)
        painter.setBrush(QColor(15, 23, 42))
        painter.setPen(QPen(QColor(245, 158, 11), 1.8))
        painter.drawRoundedRect(meter_r, 6.0, 6.0)

        # Digital LCD readout
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.setPen(QColor(34, 197, 94) if self.cell_emf_volts > 0 else QColor(239, 68, 68))
        disp_v = f"{self.cell_emf_volts:+.3f}V"
        painter.drawText(meter_r, int(Qt.AlignmentFlag.AlignCenter), disp_v)

        # Meter label
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(148, 163, 184))
        meter_title = "VÔN KẾ (EMF)" if self._preset.cell_type != "ELECTROLYSIS" else "NGUỒN DC"
        painter.drawText(QRectF(mid_wire_x - 50.0, wire_y - 28.0, 100.0, 12.0), int(Qt.AlignmentFlag.AlignCenter), meter_title)

        # 6. Animated Electron Dots on Circuit Wire
        dot_r = 3.5
        for frac in self._electrons:
            # Parametric position along wire path:
            # Segment 1: vertical up (len = anode_top - wire_y)
            # Segment 2: horizontal across (len = cath_x - anode_x)
            # Segment 3: vertical down (len = cath_top - wire_y)
            l1 = anode_top - wire_y
            l2 = (cath_x + cath_w / 2.0) - (anode_x + anode_w / 2.0)
            l3 = cath_top - wire_y
            total_l = l1 + l2 + l3

            dist = frac * total_l
            if dist < l1:
                px = anode_x + anode_w / 2.0
                py = anode_top - dist
            elif dist < l1 + l2:
                px = (anode_x + anode_w / 2.0) + (dist - l1)
                py = wire_y
            else:
                px = cath_x + cath_w / 2.0
                py = wire_y + (dist - l1 - l2)

            painter.setBrush(QColor(56, 189, 248))  # Cyan electron
            painter.setPen(QPen(QColor(255, 255, 255), 0.8))
            painter.drawEllipse(QPointF(px, py), dot_r, dot_r)

    def _paint_beaker(
        self,
        painter: QPainter,
        rect: QRectF,
        solution_name: str,
        conc_str: str,
        color_sol: QColor,
        is_anode: bool,
    ) -> None:
        # Glass wall
        glass_pen = QPen(QColor(148, 163, 184, 180), 2.0)
        beaker_path = QPainterPath()
        beaker_path.moveTo(rect.left(), rect.top())
        beaker_path.lineTo(rect.left(), rect.bottom() - 10.0)
        beaker_path.quadTo(rect.left(), rect.bottom(), rect.left() + 10.0, rect.bottom())
        beaker_path.lineTo(rect.right() - 10.0, rect.bottom())
        beaker_path.quadTo(rect.right(), rect.bottom(), rect.right(), rect.bottom() - 10.0)
        beaker_path.lineTo(rect.right(), rect.top())

        # Solution fill
        sol_top = rect.top() + 35.0
        sol_rect = QRectF(rect.left() + 2.0, sol_top, rect.width() - 4.0, rect.bottom() - sol_top - 2.0)
        painter.setBrush(color_sol)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(sol_rect, 4.0, 4.0)

        # Glass outline
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(glass_pen)
        painter.drawPath(beaker_path)

        # Labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 4.0, rect.width(), 14.0), int(Qt.AlignmentFlag.AlignCenter), solution_name)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 18.0, rect.width(), 12.0), int(Qt.AlignmentFlag.AlignCenter), conc_str)

    def _paint_salt_bridge(self, painter: QPainter, x1: float, x2: float, top_y: float) -> None:
        """U-tube salt bridge with KNO3 electrolyte and counter-ion labels."""
        bridge_w = 18.0
        bridge_h = 75.0

        bridge_path = QPainterPath()
        # Inverted U-tube
        bridge_path.moveTo(x1, top_y + bridge_h)
        bridge_path.lineTo(x1, top_y)
        bridge_path.lineTo(x2, top_y)
        bridge_path.lineTo(x2, top_y + bridge_h)
        bridge_path.lineTo(x2 - bridge_w, top_y + bridge_h)
        bridge_path.lineTo(x2 - bridge_w, top_y + bridge_w)
        bridge_path.lineTo(x1 + bridge_w, top_y + bridge_w)
        bridge_path.lineTo(x1 + bridge_w, top_y + bridge_h)
        bridge_path.closeSubpath()

        painter.setBrush(QColor(241, 245, 249, 70))  # Frosted glass
        painter.setPen(QPen(QColor(148, 163, 184), 1.4))
        painter.drawPath(bridge_path)

        # Salt bridge label
        mid_x = (x1 + x2) / 2.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(QRectF(mid_x - 45.0, top_y - 14.0, 90.0, 12.0), int(Qt.AlignmentFlag.AlignCenter), "CẦU MUỐI (KNO₃)")

    def _paint_electrode(
        self,
        painter: QPainter,
        x: float,
        y: float,
        w: float,
        h: float,
        material: str,
        is_anode: bool,
    ) -> None:
        # Material color
        if "Zn" in material or "Kẽm" in material:
            color_plate = QColor(148, 163, 184)  # Silvery zinc
        elif "Cu" in material or "Đồng" in material:
            color_plate = QColor(245, 158, 11)   # Copper amber
        elif "PbO" in material:
            color_plate = QColor(71, 85, 105)    # Dark PbO2
        elif "Pb" in material:
            color_plate = QColor(100, 116, 139)  # Dull grey lead
        else:
            color_plate = QColor(51, 65, 85)     # Graphite

        plate_r = QRectF(x, y, w, h)
        painter.setBrush(color_plate)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        painter.drawRoundedRect(plate_r, 3.0, 3.0)

        # Polarity badge (+ or -)
        badge_r = QRectF(x - 2.0, y - 18.0, w + 4.0, 14.0)
        pol_txt = "− Anot" if is_anode else "+ Catot"
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(239, 68, 68) if not is_anode else QColor(56, 189, 248))
        painter.drawText(badge_r, int(Qt.AlignmentFlag.AlignCenter), pol_txt)

    def _paint_analytics_card(self, painter: QPainter) -> None:
        card_r = QRectF(496.0, 88.0, 268.0, 360.0)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(card_r, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(248, 250, 252))
        painter.drawText(QRectF(card_r.left() + 12.0, card_r.top() + 10.0, 240.0, 20.0), "📊 NHIỆT ĐỘNG & FARADAY")

        curr_y = card_r.top() + 36.0
        lines = [
            ("Suất điện động chuẩn E°:", f"{self.standard_emf_volts:+.3f} V"),
            ("Thế Nernst thực tế E_cell:", f"{self.cell_emf_volts:+.3f} V"),
            ("Năng lượng Gibbs ΔG°:", f"{self.gibbs_free_energy_kj:.1f} kJ/mol"),
            ("Số electron trao đổi n:", f"{self._preset.n_electrons} e⁻"),
            ("Cường độ dòng điện I:", f"{self._current_amperes:.2f} A"),
            ("Thời gian điện hóa t:", f"{self._elapsed_time_s:.1f} s"),
            ("Khối lượng kim loại Δm:", f"{self.deposited_mass_grams * 1000.0:.2f} mg"),
        ]

        if "WATER" in self._preset.code or "CUSO4" in self._preset.code:
            lines.append(("Thể tích khí thoát V_gas:", f"{self.evolved_gas_volume_liters * 1000.0:.1f} mL"))

        painter.setFont(QFont("Segoe UI", 8))
        for lbl, val in lines:
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(QPointF(card_r.left() + 12.0, curr_y), lbl)
            painter.setPen(QColor(56, 189, 248) if "V" in val or "kJ" in val else QColor(248, 250, 252))
            painter.drawText(QRectF(card_r.right() - 110.0, curr_y - 12.0, 100.0, 16.0), int(Qt.AlignmentFlag.AlignRight), val)
            curr_y += 18.0

        # Equations divider
        curr_y += 4.0
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(card_r.left() + 12.0, curr_y, card_r.right() - 12.0, curr_y)
        curr_y += 14.0

        # Half reactions
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(245, 158, 11))
        painter.drawText(QPointF(card_r.left() + 12.0, curr_y), "BÁN PHẢN ỨNG ĐIỆN CỰC:")
        curr_y += 16.0

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(QPointF(card_r.left() + 12.0, curr_y), f"Anot:  {self._preset.anode_half_rxn}")
        curr_y += 16.0
        painter.drawText(QPointF(card_r.left() + 12.0, curr_y), f"Catot: {self._preset.cathode_half_rxn}")
        curr_y += 20.0

        # Formula
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(QRectF(card_r.left() + 12.0, curr_y, 244.0, 48.0), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.TextWordWrap), self._preset.description_vi)

    def _paint_action_toolbar(self, painter: QPainter) -> None:
        btns = [
            ("play_pause", "⏸ Tạm dừng" if self._is_animating else "▶ Tiếp tục"),
            ("dec_conc", "− C"),
            ("inc_conc", "+ C"),
            ("dec_curr", "− I"),
            ("inc_curr", "+ I"),
            ("reset_time", "🔄 Đặt lại t"),
            ("stamp", "📌 Đóng dấu Vector lên Bảng"),
        ]
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)

        for name, label in btns:
            rect = self._btn_rects[name]
            is_hover = (self._hovered_button == name)
            if name == "stamp":
                painter.setBrush(QColor(217, 119, 6, 230 if is_hover else 180))  # Amber-600
                painter.setPen(QPen(QColor(251, 191, 36), 1.0))
            elif name == "play_pause":
                painter.setBrush(QColor(14, 165, 233, 220 if is_hover else 160))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))
            else:
                painter.setBrush(QColor(51, 65, 85, 220 if is_hover else 160))
                painter.setPen(QPen(QColor(148, 163, 184), 1.0))

            painter.drawRoundedRect(rect, 6.0, 6.0)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

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
        if self._btn_rects["play_pause"].contains(pos):
            self.toggle_animation()
            event.accept()
            return
        if self._btn_rects["dec_conc"].contains(pos):
            self.set_concentrations(self._anode_conc_m * 0.5, self._cathode_conc_m)
            event.accept()
            return
        if self._btn_rects["inc_conc"].contains(pos):
            self.set_concentrations(self._anode_conc_m * 2.0, self._cathode_conc_m)
            event.accept()
            return
        if self._btn_rects["dec_curr"].contains(pos):
            self.set_current(self._current_amperes - 0.5)
            event.accept()
            return
        if self._btn_rects["inc_curr"].contains(pos):
            self.set_current(self._current_amperes + 0.5)
            event.accept()
            return
        if self._btn_rects["reset_time"].contains(pos):
            self.reset_time()
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
                elif name == "play_pause":
                    self.toggle_animation()
                    return True
                elif name == "dec_conc":
                    self.set_concentrations(self._anode_conc_m * 0.5, self._cathode_conc_m)
                    return True
                elif name == "inc_conc":
                    self.set_concentrations(self._anode_conc_m * 2.0, self._cathode_conc_m)
                    return True
                elif name == "dec_curr":
                    self.set_current(self._current_amperes - 0.5)
                    return True
                elif name == "inc_curr":
                    self.set_current(self._current_amperes + 0.5)
                    return True
                elif name == "reset_time":
                    self.reset_time()
                    return True
                elif name == "stamp":
                    self.stamp_to_board()
                    return True

        return False

    def stamp_to_board(self) -> None:
        """Stamps current electrochemical cell as permanent vector diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import ChemistryTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = ChemistryTools.create_electrochemistry_diagram(
            center=center,
            preset_code=self._active_preset_code,
            anode_conc=self._anode_conc_m,
            cathode_conc=self._cathode_conc_m,
            current_a=self._current_amperes,
        )
        for itm in items:
            scene.addItem(itm)
