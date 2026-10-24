"""
AirTeach AI - Interactive Chemical Equilibrium & Le Chatelier's Principle Simulator
=====================================================================================
Authentic thermodynamic equilibrium modeling for reversible chemical reactions.
Simulates temperature dependence via van 't Hoff equation and Gibbs free energy,
pressure shifts according to Delta n_gas, reaction quotient Q vs K_p,
and equilibrium extent of reaction xi with high-precision numerical solving.

Chemical Physics Models:
  1. Standard Gibbs Free Energy:
       ΔG°(T) = ΔH° - T * ΔS°
  2. Equilibrium Constant:
       K_p(T) = exp(-ΔG°(T) / (R * T))
  3. van 't Hoff Equation:
       d(ln K_p) / dT = ΔH° / (R * T^2)
  4. Mole Fraction Equilibrium Relation:
       K_x = K_p * P^(-Δn_gas)
  5. Haber-Bosch Synthesis:
       N2(g) + 3H2(g) <=> 2NH3(g), ΔH° = -92.2 kJ/mol, ΔS° = -198.7 J/(mol*K)
       Δn_gas = 2 - 4 = -2 => K_x = K_p * P^2
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent

# Physical & Thermodynamic Constants
R_GAS_CONST = 8.314462618  # J / (mol * K)
ATM_TO_BAR = 1.01325


@dataclass(frozen=True)
class ReactionPreset:
    name: str
    equation_display: str
    delta_h_kj: float  # kJ/mol
    delta_s_j: float  # J/(mol * K)
    delta_n_gas: int  # n_products - n_reactants
    default_t_k: float  # Kelvin
    default_p_atm: float  # atm
    reactants: List[str]
    products: List[str]
    description: str


REACTION_PRESETS: Dict[str, ReactionPreset] = {
    "HABER_BOSCH": ReactionPreset(
        name="Tổng Hợp Amoniac (Haber-Bosch)",
        equation_display="N₂(k) + 3H₂(k) ⇌ 2NH₃(k)",
        delta_h_kj=-92.22,
        delta_s_j=-198.75,
        delta_n_gas=-2,
        default_t_k=673.15,  # 400 °C
        default_p_atm=200.0,
        reactants=["N₂", "H₂"],
        products=["NH₃"],
        description="Phản ứng tỏa nhiệt (ΔH < 0, giảm số mol khí). Tăng áp suất và giảm nhiệt độ làm tăng hiệu suất NH₃.",
    ),
    "NO2_N2O4_EQUILIBRIUM": ReactionPreset(
        name="Cân Bằng Dinitrogen Tetroxide",
        equation_display="2NO₂(k, nâu đỏ) ⇌ N₂O₄(k, không màu)",
        delta_h_kj=-57.20,
        delta_s_j=-175.83,
        delta_n_gas=-1,
        default_t_k=298.15,  # 25 °C
        default_p_atm=2.0,
        reactants=["NO₂"],
        products=["N₂O₄"],
        description="Phản ứng tỏa nhiệt. Khí NO₂ có màu nâu đỏ, khi ngâm vào nước đá cân bằng chuyển dịch tạo N₂O₄ không màu.",
    ),
    "CONTACT_SO3_SYNTHESIS": ReactionPreset(
        name="Oxi Hóa SO₂ (Công Nghiệp Axit Sunfuric)",
        equation_display="2SO₂(k) + O₂(k) ⇌ 2SO₃(k)",
        delta_h_kj=-197.78,
        delta_s_j=-187.96,
        delta_n_gas=-1,
        default_t_k=723.15,  # 450 °C
        default_p_atm=1.5,
        reactants=["SO₂", "O₂"],
        products=["SO₃"],
        description="Giai đoạn mấu chốt sản xuất H₂SO₄ với xúc tác V₂O₅. Tỏa nhiệt mạnh, thuận lợi ở nhiệt độ vừa phải.",
    ),
    "HYDROGEN_IODIDE": ReactionPreset(
        name="Tổng Hợp Hiđro Iotua (Đẳng Tích Khí)",
        equation_display="H₂(k) + I₂(k) ⇌ 2HI(k)",
        delta_h_kj=9.48,  # Slightly endothermic
        delta_s_j=21.60,
        delta_n_gas=0,
        default_t_k=700.0,
        default_p_atm=5.0,
        reactants=["H₂", "I₂"],
        products=["HI"],
        description="Δn_gas = 0: Áp suất hoàn toàn KHÔNG làm chuyển dịch cân bằng. Nhiệt độ cao làm dịch chuyển theo chiều thuận.",
    ),
}


class InteractiveChemicalEquilibriumItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Chemical Equilibrium, Le Chatelier's Principle,
    and thermodynamic gas-phase reactions.
    """

    stamp_requested = Signal(dict)
    equilibrium_computed = Signal(dict)

    def __init__(
        self,
        preset_key: str = "HABER_BOSCH",
        temp_k: Optional[float] = None,
        pressure_atm: Optional[float] = None,
        pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        if pos is not None:
            self.setPos(pos[0], pos[1])

        self._active_preset_key = preset_key if preset_key in REACTION_PRESETS else "HABER_BOSCH"
        preset = REACTION_PRESETS[self._active_preset_key]

        self._temp_k = float(temp_k if temp_k is not None else preset.default_t_k)
        self._pressure_atm = float(pressure_atm if pressure_atm is not None else preset.default_p_atm)
        self._catalyst_enabled = True

        self._width = 690.0
        self._height = 430.0

        # UI interaction rects
        self._t_slider_rect = QRectF(0, 0, 0, 0)
        self._p_slider_rect = QRectF(0, 0, 0, 0)
        self._catalyst_btn_rect = QRectF(0, 0, 0, 0)
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)
        self._preset_btn_rects: Dict[str, QRectF] = {}

        self._is_dragging_t = False
        self._is_dragging_p = False

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Thermodynamic Calculations ──────────────────────────────────────────

    @property
    def temp_k(self) -> float:
        return self._temp_k

    @property
    def temp_c(self) -> float:
        return self._temp_k - 273.15

    @property
    def pressure_atm(self) -> float:
        return self._pressure_atm

    @property
    def catalyst_enabled(self) -> bool:
        return self._catalyst_enabled

    def set_temp_k(self, temp_k: float) -> None:
        self._temp_k = max(250.0, min(1000.0, float(temp_k)))
        self.equilibrium_computed.emit(self.compute_equilibrium())
        self.update()

    def set_pressure_atm(self, p_atm: float) -> None:
        self._pressure_atm = max(0.2, min(400.0, float(p_atm)))
        self.equilibrium_computed.emit(self.compute_equilibrium())
        self.update()

    def toggle_catalyst(self) -> None:
        self._catalyst_enabled = not self._catalyst_enabled
        self.update()

    def apply_preset(self, preset_key: str) -> None:
        if preset_key in REACTION_PRESETS:
            self._active_preset_key = preset_key
            preset = REACTION_PRESETS[preset_key]
            self._temp_k = preset.default_t_k
            self._pressure_atm = preset.default_p_atm
            self.equilibrium_computed.emit(self.compute_equilibrium())
            self.update()

    def compute_equilibrium(self) -> Dict[str, float]:
        """
        Solves thermodynamic equilibrium for the active chemical reaction preset.
        Returns Delta G, K_p, K_x, extent of reaction xi, and species mole fractions.
        """
        preset = REACTION_PRESETS[self._active_preset_key]
        t = self._temp_k
        p = self._pressure_atm

        # 1. Delta G°(T) = Delta H° - T * Delta S°
        delta_h_j = preset.delta_h_kj * 1000.0  # J/mol
        delta_s_j = preset.delta_s_j  # J/(mol*K)
        delta_g_j = delta_h_j - t * delta_s_j
        delta_g_kj = delta_g_j / 1000.0

        # 2. K_p(T) = exp(-Delta G° / RT)
        exponent = -delta_g_j / (R_GAS_CONST * t)
        exponent_clamped = max(-60.0, min(60.0, exponent))
        kp = math.exp(exponent_clamped)

        # 3. K_x = K_p * P^(-Delta n_gas)
        # Note: pressure in bar ≈ atm
        p_factor = math.pow(p, -preset.delta_n_gas)
        kx = kp * p_factor

        # 4. Solve extent of reaction xi in [0, 1]
        # Specific algebra per stoichiometry:
        if self._active_preset_key == "HABER_BOSCH":
            # N2 + 3H2 <=> 2NH3. Initial: 1 mol N2, 3 mol H2.
            # Q_x(xi) = 4*xi^2 * (4 - 2*xi)^2 / [27 * (1 - xi)^4] = 16*xi^2*(2 - xi)^2 / [27*(1-xi)^4]
            # Monotonic in xi in (0, 1). We solve via bisection.
            low, high = 1e-5, 0.9999
            for _ in range(35):
                mid = (low + high) * 0.5
                num = 16.0 * (mid**2) * ((2.0 - mid) ** 2)
                den = 27.0 * ((1.0 - mid) ** 4)
                q_val = num / max(1e-12, den)
                if q_val < kx:
                    low = mid
                else:
                    high = mid
            xi = (low + high) * 0.5
            total_mol = 4.0 - 2.0 * xi
            x_n2 = (1.0 - xi) / total_mol
            x_h2 = 3.0 * (1.0 - xi) / total_mol
            x_nh3 = (2.0 * xi) / total_mol
            mole_fractions = {"N₂": x_n2, "H₂": x_h2, "NH₃": x_nh3}
            yield_pct = (2.0 * xi / 2.0) * 100.0  # max possible NH3 is 2 mol

        elif self._active_preset_key == "NO2_N2O4_EQUILIBRIUM":
            # 2NO2 <=> N2O4. Initial: 2 mol NO2.
            # At xi: n(NO2) = 2 - 2*xi, n(N2O4) = xi. Total = 2 - xi.
            # Q_x = [xi / (2 - xi)] / [(2 - 2*xi)/(2 - xi)]^2 = xi * (2 - xi) / [4*(1 - xi)^2]
            low, high = 1e-5, 0.9999
            for _ in range(35):
                mid = (low + high) * 0.5
                q_val = (mid * (2.0 - mid)) / max(1e-12, 4.0 * ((1.0 - mid) ** 2))
                if q_val < kx:
                    low = mid
                else:
                    high = mid
            xi = (low + high) * 0.5
            total_mol = 2.0 - xi
            x_no2 = (2.0 - 2.0 * xi) / total_mol
            x_n2o4 = xi / total_mol
            mole_fractions = {"NO₂": x_no2, "N₂O₄": x_n2o4}
            yield_pct = xi * 100.0

        elif self._active_preset_key == "CONTACT_SO3_SYNTHESIS":
            # 2SO2 + O2 <=> 2SO3. Initial: 2 mol SO2, 1 mol O2.
            # At xi: n(SO2) = 2 - 2*xi, n(O2) = 1 - xi, n(SO3) = 2*xi. Total = 3 - xi.
            low, high = 1e-5, 0.9999
            for _ in range(35):
                mid = (low + high) * 0.5
                # Q_x = [2xi / (3 - xi)]^2 / { [2(1-xi)/(3-xi)]^2 * [(1-xi)/(3-xi)] } = xi^2 * (3 - xi) / [(1 - xi)^3]
                q_val = ((mid**2) * (3.0 - mid)) / max(1e-12, (1.0 - mid) ** 3)
                if q_val < kx:
                    low = mid
                else:
                    high = mid
            xi = (low + high) * 0.5
            total_mol = 3.0 - xi
            x_so2 = (2.0 - 2.0 * xi) / total_mol
            x_o2 = (1.0 - xi) / total_mol
            x_so3 = (2.0 * xi) / total_mol
            mole_fractions = {"SO₂": x_so2, "O₂": x_o2, "SO₃": x_so3}
            yield_pct = xi * 100.0

        else:  # HYDROGEN_IODIDE
            # H2 + I2 <=> 2HI. Initial: 1 mol H2, 1 mol I2.
            # At xi: n(H2) = 1 - xi, n(I2) = 1 - xi, n(HI) = 2*xi. Total = 2.
            # Q_x = (2*xi)^2 / (1 - xi)^2 = 4*xi^2 / (1 - xi)^2 = K_x
            # sqrt(K_x) = 2*xi / (1 - xi) => xi = sqrt(K_x) / (2 + sqrt(K_x))
            sqrt_kx = math.sqrt(max(1e-12, kx))
            xi = sqrt_kx / (2.0 + sqrt_kx)
            x_h2 = (1.0 - xi) / 2.0
            x_i2 = (1.0 - xi) / 2.0
            x_hi = xi
            mole_fractions = {"H₂": x_h2, "I₂": x_i2, "HI": x_hi}
            yield_pct = xi * 100.0

        return {
            "temp_k": t,
            "pressure_atm": p,
            "delta_h_kj": preset.delta_h_kj,
            "delta_s_j": preset.delta_s_j,
            "delta_g_kj": delta_g_kj,
            "kp": kp,
            "kx": kx,
            "xi": xi,
            "yield_pct": yield_pct,
            "mole_fractions": mole_fractions,
        }

    def request_stamp(self) -> None:
        eq = self.compute_equilibrium()
        preset = REACTION_PRESETS[self._active_preset_key]
        data = {
            "center": (self.x(), self.y()),
            "preset_key": self._active_preset_key,
            "preset_name": preset.name,
            "equation_display": preset.equation_display,
            "description": preset.description,
            "temp_k": self._temp_k,
            "temp_c": self.temp_c,
            "pressure_atm": self._pressure_atm,
            "delta_g_kj": eq["delta_g_kj"],
            "kp": eq["kp"],
            "kx": eq["kx"],
            "xi": eq["xi"],
            "yield_pct": eq["yield_pct"],
            "mole_fractions": eq["mole_fractions"],
            "catalyst_enabled": self._catalyst_enabled,
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        eq = self.compute_equilibrium()
        preset = REACTION_PRESETS[self._active_preset_key]

        hw = self._width / 2.0
        hh = self._height / 2.0

        # 1. Outer Container
        bg_grad = QLinearGradient(-hw, -hh, -hw, hh)
        bg_grad.setColorAt(0.0, QColor(13, 20, 36))
        bg_grad.setColorAt(1.0, QColor(7, 11, 22))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(30, 48, 78), 1.5))
        painter.drawRoundedRect(-hw, -hh, self._width, self._height, 10.0, 10.0)

        # 2. Header
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(-hw + 20, -hh + 12, self._width - 40, 22),
            Qt.AlignLeft,
            f"⚗ CÂN BẰNG HÓA HỌC & NGUYÊN LÝ LE CHATELIER: {preset.name.upper()}",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(-hw + 20, -hh + 34, self._width - 40, 18),
            Qt.AlignLeft,
            f"Phương trình: {preset.equation_display} | ΔH° = {preset.delta_h_kj:+.1f} kJ/mol | ΔS° = {preset.delta_s_j:+.1f} J/(mol·K)",
        )

        # 3. Reaction Presets Navigation Tabs
        self._draw_preset_tabs(painter, -hw + 20, -hh + 54)

        # 4. Left Panel: Reaction Chamber & Mole Fraction Graph
        chamber_rect = QRectF(-hw + 20, -hh + 84, 380, 230)
        self._draw_reaction_chamber(painter, chamber_rect, eq, preset)

        # 5. Right Panel: Thermodynamic Metrics & Formulas
        thermo_rect = QRectF(chamber_rect.right() + 14, -hh + 84, self._width - chamber_rect.width() - 54, 230)
        self._draw_thermodynamic_panel(painter, thermo_rect, eq, preset)

        # 6. Bottom Controls: T slider, P slider, Catalyst toggle, Stamp button
        controls_rect = QRectF(-hw + 20, hh - 100, self._width - 40, 88)
        self._draw_controls(painter, controls_rect, eq)

    def _draw_preset_tabs(self, painter: QPainter, x: float, y: float) -> None:
        self._preset_btn_rects.clear()
        tab_w = 158.0
        tab_h = 22.0

        keys = list(REACTION_PRESETS.keys())
        for i, key in enumerate(keys):
            preset = REACTION_PRESETS[key]
            rect = QRectF(x + i * (tab_w + 5), y, tab_w, tab_h)
            self._preset_btn_rects[key] = rect

            is_active = key == self._active_preset_key
            if is_active:
                painter.setBrush(QBrush(QColor(14, 116, 144)))
                painter.setPen(QPen(QColor(56, 189, 248), 1.5))
            else:
                painter.setBrush(QBrush(QColor(18, 27, 44)))
                painter.setPen(QPen(QColor(36, 52, 78), 1.0))

            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(241, 245, 249) if is_active else QColor(148, 163, 184))
            short_name = preset.name.split(" (")[0]
            painter.drawText(rect, Qt.AlignCenter, short_name)

    def _draw_reaction_chamber(
        self, painter: QPainter, rect: QRectF, eq: Dict[str, float], preset: ReactionPreset
    ) -> None:
        painter.setBrush(QBrush(QColor(10, 16, 28)))
        painter.setPen(QPen(QColor(30, 48, 75), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Section 1: Gas Container Visualization
        box_w = 160.0
        box_h = 130.0
        box_x = rect.left() + 14.0
        box_y = rect.top() + 14.0
        gas_box = QRectF(box_x, box_y, box_w, box_h)

        # Piston and chamber color
        # In NO2/N2O4, NO2 is brown, N2O4 colorless
        if self._active_preset_key == "NO2_N2O4_EQUILIBRIUM":
            no2_frac = eq["mole_fractions"].get("NO₂", 0.5)
            brown_alpha = int(max(30, min(240, no2_frac * 255)))
            chamber_brush = QBrush(QColor(180, 83, 9, brown_alpha))
        else:
            yield_pct = eq["yield_pct"]
            blue_alpha = int(max(30, min(220, yield_pct * 2.2)))
            chamber_brush = QBrush(QColor(14, 165, 233, blue_alpha))

        painter.setBrush(chamber_brush)
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(gas_box, 4.0, 4.0)

        # Draw Piston Lid (reflecting pressure P)
        piston_depth = max(6.0, min(30.0, (self._pressure_atm / 400.0) * 30.0))
        piston_rect = QRectF(box_x, box_y, box_w, piston_depth)
        painter.setBrush(QBrush(QColor(71, 85, 105)))
        painter.setPen(QPen(QColor(148, 163, 184), 1.0))
        painter.drawRect(piston_rect)

        # Center gas label
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            gas_box,
            Qt.AlignCenter,
            f"Bình Phản Ứng\n{self._pressure_atm:.1f} atm | {self.temp_c:.0f} °C",
        )

        # Section 2: Mole Fractions Bar Chart
        chart_x = box_x + box_w + 20.0
        chart_y = box_y
        chart_w = rect.right() - chart_x - 14.0
        chart_h = box_h

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(chart_x, chart_y, chart_w, 16), Qt.AlignLeft, "📊 THÀNH PHẦN (% MOL)")

        bar_y = chart_y + 24.0
        colors = [QColor(244, 63, 94), QColor(59, 130, 246), QColor(34, 197, 94), QColor(234, 179, 8)]

        mole_fractions = eq["mole_fractions"]
        for idx, (species, frac) in enumerate(mole_fractions.items()):
            col = colors[idx % len(colors)]
            # Label
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(QRectF(chart_x, bar_y, 45, 14), Qt.AlignLeft, f"x({species})")

            # Bar track
            track_w = chart_w - 95.0
            track_rect = QRectF(chart_x + 48, bar_y + 2, track_w, 10)
            painter.setBrush(QBrush(QColor(24, 34, 53)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 5.0, 5.0)

            # Fill bar
            fill_w = max(2.0, frac * track_w)
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(QRectF(track_rect.left(), track_rect.top(), fill_w, 10), 5.0, 5.0)

            # Percentage text
            painter.setPen(col)
            painter.drawText(QRectF(track_rect.right() + 6, bar_y, 40, 14), Qt.AlignRight, f"{frac * 100.0:.1f}%")

            bar_y += 22.0

        # Section 3: Le Chatelier Dynamic Shift Arrow & Description
        status_box = QRectF(rect.left() + 14, rect.bottom() - 65, rect.width() - 28, 52)
        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(30, 48, 75), 1.0))
        painter.drawRoundedRect(status_box, 4.0, 4.0)

        shift_color = QColor(34, 197, 94) if eq["yield_pct"] > 50.0 else QColor(251, 146, 60)
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(shift_color)
        painter.drawText(
            QRectF(status_box.left() + 8, status_box.top() + 6, status_box.width() - 16, 16),
            Qt.AlignLeft,
            f"⚡ HIỆU SUẤT TẠO SẢN PHẨM: {eq['yield_pct']:.1f}% | CÂN BẰNG ĐỘNG",
        )

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(status_box.left() + 8, status_box.top() + 24, status_box.width() - 16, 24),
            Qt.AlignLeft | Qt.TextWordWrap,
            preset.description,
        )

    def _draw_thermodynamic_panel(
        self, painter: QPainter, rect: QRectF, eq: Dict[str, float], preset: ReactionPreset
    ) -> None:
        painter.setBrush(QBrush(QColor(10, 16, 28)))
        painter.setPen(QPen(QColor(30, 48, 75), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, 16),
            Qt.AlignLeft,
            "🔬 THÔNG SỐ NHIỆT ĐỘNG HỌC",
        )

        rows = [
            ("Nhiệt độ (T):", f"{eq['temp_k']:.1f} K ({self.temp_c:.1f} °C)", QColor(245, 158, 11)),
            ("Áp suất tổng (P):", f"{eq['pressure_atm']:.1f} atm", QColor(56, 189, 248)),
            ("Thế Gibbs (ΔG°):", f"{eq['delta_g_kj']:+.2f} kJ/mol", QColor(168, 85, 247)),
            ("Hằng số K_p:", f"{eq['kp']:.3e}", QColor(74, 222, 128)),
            ("Hằng số K_x:", f"{eq['kx']:.3e}", QColor(236, 72, 153)),
            ("Độ chuyển hóa (ξ):", f"{eq['xi']:.3f} mol", QColor(34, 197, 94)),
            ("Xúc tác (Catalyst):", "Kích hoạt (Tăng tốc v_f, v_r)" if self._catalyst_enabled else "Không có", QColor(250, 204, 21)),
        ]

        y = rect.top() + 28.0
        painter.setFont(QFont("Segoe UI", 7))
        for label, val, col in rows:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 10, y, 110, 14), Qt.AlignLeft, label)
            painter.setPen(col)
            painter.drawText(QRectF(rect.right() - 135, y, 125, 14), Qt.AlignRight, val)
            y += 18.0

        # Equations Box
        eq_box = QRectF(rect.left() + 8, y + 4, rect.width() - 16, 68)
        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(24, 38, 60), 1.0))
        painter.drawRoundedRect(eq_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(eq_box.left() + 6, eq_box.top() + 4, eq_box.width() - 12, 12), Qt.AlignLeft, "CÔNG THỨC NHIỆT ĐỘNG:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(203, 213, 225))
        formulas = [
            "• ΔG°(T) = ΔH° - T·ΔS°",
            "• K_p(T) = exp(-ΔG° / RT)",
            "• van 't Hoff: d(ln K_p)/dT = ΔH° / (RT²)",
        ]
        fy = eq_box.top() + 18.0
        for form in formulas:
            painter.drawText(QRectF(eq_box.left() + 6, fy, eq_box.width() - 12, 12), Qt.AlignLeft, form)
            fy += 15.0

    def _draw_controls(self, painter: QPainter, rect: QRectF, eq: Dict[str, float]) -> None:
        painter.setBrush(QBrush(QColor(12, 18, 32)))
        painter.setPen(QPen(QColor(30, 48, 75), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # 1. Temperature Slider T (250 K to 1000 K)
        t_y = rect.top() + 14.0
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(245, 158, 11))
        painter.drawText(QRectF(rect.left() + 12, t_y - 2, 120, 16), Qt.AlignLeft, f"Nhiệt độ T: {self.temp_c:.0f} °C ({self._temp_k:.0f} K)")

        t_track = QRectF(rect.left() + 140, t_y + 2, 220, 8)
        self._t_slider_rect = t_track
        painter.setBrush(QBrush(QColor(24, 34, 53)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(t_track, 4.0, 4.0)

        t_norm = (self._temp_k - 250.0) / 750.0
        t_fill_w = max(0.0, min(1.0, t_norm)) * t_track.width()
        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.drawRoundedRect(QRectF(t_track.left(), t_track.top(), t_fill_w, 8), 4.0, 4.0)

        # 2. Pressure Slider P (0.2 to 400 atm)
        p_y = rect.top() + 44.0
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(rect.left() + 12, p_y - 2, 120, 16), Qt.AlignLeft, f"Áp suất P: {self._pressure_atm:.1f} atm")

        p_track = QRectF(rect.left() + 140, p_y + 2, 220, 8)
        self._p_slider_rect = p_track
        painter.setBrush(QBrush(QColor(24, 34, 53)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(p_track, 4.0, 4.0)

        p_norm = (self._pressure_atm - 0.2) / 399.8
        p_fill_w = max(0.0, min(1.0, p_norm)) * p_track.width()
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(QRectF(p_track.left(), p_track.top(), p_fill_w, 8), 4.0, 4.0)

        # 3. Catalyst Toggle Button
        cat_btn = QRectF(rect.left() + 380, rect.top() + 14, 120, 26)
        self._catalyst_btn_rect = cat_btn
        if self._catalyst_enabled:
            painter.setBrush(QBrush(QColor(16, 185, 129, 60)))
            painter.setPen(QPen(QColor(16, 185, 129), 1.5))
        else:
            painter.setBrush(QBrush(QColor(24, 34, 53)))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(cat_btn, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(241, 245, 249) if self._catalyst_enabled else QColor(148, 163, 184))
        painter.drawText(cat_btn, Qt.AlignCenter, "⚡ Xúc Tác: " + ("BẬT" if self._catalyst_enabled else "TẮT"))

        # 4. Stamp Button
        stamp_btn = QRectF(rect.right() - 130, rect.top() + 14, 120, 26)
        self._stamp_btn_rect = stamp_btn
        painter.setBrush(QBrush(QColor(14, 116, 144)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(stamp_btn, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(stamp_btn, Qt.AlignCenter, "📌 ĐÓNG DẤU BÀI DẠY")

        # Instruction Hint
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(100, 116, 139))
        painter.drawText(
            QRectF(rect.left() + 380, rect.top() + 48, rect.width() - 390, 20),
            Qt.AlignLeft,
            "💡 Kéo thanh trượt để quan sát cân bằng chuyển dịch theo Le Chatelier.",
        )

    # ── Mouse Interaction ────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset Tabs
        for key, rect in self._preset_btn_rects.items():
            if rect.contains(pos):
                self.apply_preset(key)
                event.accept()
                return

        # Check Catalyst Toggle
        if self._catalyst_btn_rect.contains(pos):
            self.toggle_catalyst()
            event.accept()
            return

        # Check Stamp Button
        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        # Check T Slider
        if self._t_slider_rect.contains(pos) or (
            abs(pos.y() - self._t_slider_rect.center().y()) < 12
            and self._t_slider_rect.left() <= pos.x() <= self._t_slider_rect.right()
        ):
            self._is_dragging_t = True
            norm = (pos.x() - self._t_slider_rect.left()) / self._t_slider_rect.width()
            self.set_temp_k(250.0 + norm * 750.0)
            event.accept()
            return

        # Check P Slider
        if self._p_slider_rect.contains(pos) or (
            abs(pos.y() - self._p_slider_rect.center().y()) < 12
            and self._p_slider_rect.left() <= pos.x() <= self._p_slider_rect.right()
        ):
            self._is_dragging_p = True
            norm = (pos.x() - self._p_slider_rect.left()) / self._p_slider_rect.width()
            self.set_pressure_atm(0.2 + norm * 399.8)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._is_dragging_t:
            norm = (pos.x() - self._t_slider_rect.left()) / self._t_slider_rect.width()
            self.set_temp_k(250.0 + norm * 750.0)
            event.accept()
            return
        elif self._is_dragging_p:
            norm = (pos.x() - self._p_slider_rect.left()) / self._p_slider_rect.width()
            self.set_pressure_atm(0.2 + norm * 399.8)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging_t = False
        self._is_dragging_p = False
        super().mouseReleaseEvent(event)
