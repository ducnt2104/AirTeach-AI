"""
AirTeach AI – Interactive Chemistry: Chemical Kinetics, Arrhenius Law & Reaction Coordinate Energy Profiles
============================================================================================================
Provides an authentic physical chemistry simulation of reaction rates, activation energies, and catalysis:
  - Arrhenius Equation (1889):
      k = A * exp(-E_a / (R * T))
      where:
        - k is the reaction rate constant (s⁻¹)
        - A is the pre-exponential frequency factor (collisions per second)
        - E_a is the activation energy (kJ/mol)
        - R = 8.314462618 J/(mol·K) is the universal gas constant
        - T is absolute temperature (Kelvin)
  - Linear Arrhenius Relation (Arrhenius Plot):
      ln(k) = ln(A) - (E_a / R) * (1 / T)
      The slope of ln(k) vs 1/T is -E_a / R, allowing experimental extraction of E_a.
  - Transition State Theory & Potential Energy Profile:
      E(q) along reaction coordinate q in [0, 1]:
      - Reactants state: E_reactants
      - Transition State [X‡] peak: E_transition = E_reactants + E_a
      - Products state: E_products = E_reactants + ΔH
      - Reaction Enthalpy ΔH:
          - ΔH < 0: Exothermic (Tỏa nhiệt)
          - ΔH > 0: Endothermic (Thu nhiệt)
  - Catalyst Mechanism (Xúc tác):
      Lowers activation energy: E_a,cat < E_a.
      Speedup factor: Ratio = k_cat / k_uncat = exp((E_a - E_a,cat) / (R * T)).
      Catalyst does NOT alter thermodynamic enthalpy ΔH or equilibrium constant K_eq.
  - 5 Curriculum Presets:
      1. EXOTHERMIC_CATALYSIS_DECOMPOSITION: Decomposition of H₂O₂ (MnO₂ catalyst, ΔH = -98 kJ/mol).
      2. ENDOTHERMIC_SYNTHESIS_HI: Synthesis of Hydrogen Iodide (H₂ + I₂ ⇌ 2HI, ΔH = +26.5 kJ/mol).
      3. TEMPERATURE_EFFECT_MAXWELL_BOLTZMANN: Fraction of molecules exceeding E_a as T rises (300K to 600K).
      4. ARRHENIUS_PLOT_LINEAR_FIT: Linear ln(k) vs 1/T plot with negative slope -E_a/R.
      5. ENZYME_SUBSTRATE_BIOCATALYSIS: Biological enzyme lowering E_a by 60 kJ/mol (10¹⁰× speedup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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


# Physical Constants (CODATA 2022)
R_GAS = 8.314462618                  # Universal Gas Constant (J / (mol·K))


@dataclass(frozen=True)
class KineticsPreset:
    code: str
    title_vi: str
    badge_label: str
    reaction_str: str
    ea_uncat_kj: float               # Uncatalyzed activation energy (kJ/mol)
    ea_cat_kj: float                 # Catalyzed activation energy (kJ/mol)
    delta_h_kj: float                # Reaction enthalpy (kJ/mol, negative = exo, positive = endo)
    temp_k: float                    # Default temperature (Kelvin)
    description: str


PRESETS: dict[str, KineticsPreset] = {
    "EXOTHERMIC_CATALYSIS_DECOMPOSITION": KineticsPreset(
        code="EXOTHERMIC_CATALYSIS_DECOMPOSITION",
        title_vi="Phân hủy H₂O₂ Xúc tác MnO₂ (Phản ứng Tỏa nhiệt ΔH < 0)",
        badge_label="TỎA NHIỆT + XÚC TÁC",
        reaction_str="2 H₂O₂ ➔ 2 H₂O + O₂ ↑ (Xúc tác MnO₂)",
        ea_uncat_kj=75.0,
        ea_cat_kj=28.0,
        delta_h_kj=-98.0,
        temp_k=298.15,
        description="Phản ứng phân hủy H₂O₂ tỏa nhiệt mạnh (ΔH = -98 kJ/mol). Chất xúc tác MnO₂ hạ thấp năng lượng hoạt hóa từ 75 xuống 28 kJ/mol, làm tốc độ phản ứng tăng hàng triệu lần.",
    ),
    "ENDOTHERMIC_SYNTHESIS_HI": KineticsPreset(
        code="ENDOTHERMIC_SYNTHESIS_HI",
        title_vi="Tổng hợp Hydro Iodua HI (Phản ứng Thu nhiệt ΔH > 0)",
        badge_label="THU NHIỆT ΔH > 0",
        reaction_str="H₂ + I₂ ⇌ 2 HI",
        ea_uncat_kj=165.0,
        ea_cat_kj=110.0,
        delta_h_kj=+26.5,
        temp_k=550.0,
        description="Phản ứng thu nhiệt có năng lượng sản phẩm cao hơn chất đầu (ΔH = +26.5 kJ/mol). Cần nhiệt độ cao để phân tử vượt qua hàng rào thế năng hoạt hóa.",
    ),
    "TEMPERATURE_EFFECT_MAXWELL_BOLTZMANN": KineticsPreset(
        code="TEMPERATURE_EFFECT_MAXWELL_BOLTZMANN",
        title_vi="Ảnh hưởng của Nhiệt độ đến Tốc độ Phản ứng (Hệ số Van't Hoff)",
        badge_label="NHIỆT ĐỘ TĂNG TỐC",
        reaction_str="Phân bố Động năng Phân tử Maxwell-Boltzmann",
        ea_uncat_kj=60.0,
        ea_cat_kj=30.0,
        delta_h_kj=-45.0,
        temp_k=450.0,
        description="Khi tăng nhiệt độ T, phần trăm số phân tử có động năng vượt ngưỡng E_a tăng theo hàm mũ exp(-E_a/RT), giải thích quy tắc kinh nghiệm Van't Hoff (tăng 10°C tốc độ tăng 2-4 lần).",
    ),
    "ARRHENIUS_PLOT_LINEAR_FIT": KineticsPreset(
        code="ARRHENIUS_PLOT_LINEAR_FIT",
        title_vi="Đồ thị Tuyến tính Arrhenius: ln(k) theo 1/T",
        badge_label="ĐỒ THỊ ARRHENIUS",
        reaction_str="ln(k) = ln(A) - (E_a / R) · (1/T)",
        ea_uncat_kj=85.0,
        ea_cat_kj=45.0,
        delta_h_kj=-60.0,
        temp_k=350.0,
        description="Đồ thị hàm ln(k) theo 1/T là một đường thẳng dốc xuống có hệ số góc bằng -E_a/R, là phương pháp thực nghiệm kinh điển để đo năng lượng hoạt hóa.",
    ),
    "ENZYME_SUBSTRATE_BIOCATALYSIS": KineticsPreset(
        code="ENZYME_SUBSTRATE_BIOCATALYSIS",
        title_vi="Cơ chế Xúc tác Sinh học Enzyme (Ổ khóa - Chìa khóa)",
        badge_label="ENZYME SINH HỌC",
        reaction_str="Enzyme + Cơ chất ➔ Phức hệ E-S ➔ Sản phẩm",
        ea_uncat_kj=105.0,
        ea_cat_kj=35.0,
        delta_h_kj=-50.0,
        temp_k=310.15,                      # 37°C body temperature
        description="Xúc tác enzyme sinh học làm giảm cực mạnh năng lượng hoạt hóa (giảm 70 kJ/mol), tăng tốc phản ứng sinh hóa gấp 10¹² lần ở nhiệt độ cơ thể người 37°C.",
    ),
}


class InteractiveReactionKineticsItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Chemical Kinetics:
    Arrhenius Law, Activation Energy profiles, Catalysis, and Energy Coordinates.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    preset_changed = Signal(str)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)

        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Simulation Preset & Physical Configuration
        self._active_preset_code: str = "EXOTHERMIC_CATALYSIS_DECOMPOSITION"
        self._preset: KineticsPreset = PRESETS[self._active_preset_code]

        self._ea_uncat_kj: float = self._preset.ea_uncat_kj
        self._ea_cat_kj: float = self._preset.ea_cat_kj
        self._delta_h_kj: float = self._preset.delta_h_kj
        self._temp_k: float = self._preset.temp_k
        self._has_catalyst: bool = True

        # Pre-exponential factor A
        self._a_factor: float = 1.0e11

        # Animation State (crossing particles)
        self._particles: list[dict[str, float]] = []
        self._anim_phase: float = 0.0
        self._is_animating: bool = True

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physical Chemistry Mathematics (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def current_ea_kj(self) -> float:
        """Active activation energy in kJ/mol (uncatalyzed or catalyzed)."""
        return self._ea_cat_kj if self._has_catalyst else self._ea_uncat_kj

    @property
    def rate_constant_uncat(self) -> float:
        """Arrhenius rate constant k_uncat = A * exp(-E_a,uncat / (R * T))."""
        ea_j = self._ea_uncat_kj * 1000.0
        exp_arg = -ea_j / (R_GAS * self._temp_k)
        # Prevent float underflow/overflow
        if exp_arg < -80.0:
            return 0.0
        return self._a_factor * math.exp(exp_arg)

    @property
    def rate_constant_cat(self) -> float:
        """Arrhenius rate constant k_cat = A * exp(-E_a,cat / (R * T))."""
        ea_j = self._ea_cat_kj * 1000.0
        exp_arg = -ea_j / (R_GAS * self._temp_k)
        if exp_arg < -80.0:
            return 0.0
        return self._a_factor * math.exp(exp_arg)

    @property
    def current_rate_constant(self) -> float:
        return self.rate_constant_cat if self._has_catalyst else self.rate_constant_uncat

    @property
    def catalysis_speedup_ratio(self) -> float:
        """Ratio k_cat / k_uncat = exp((E_a,uncat - E_a,cat) / (R * T))."""
        delta_ea_j = (self._ea_uncat_kj - self._ea_cat_kj) * 1000.0
        exp_arg = delta_ea_j / (R_GAS * self._temp_k)
        if exp_arg > 70.0:
            return math.exp(70.0)
        return math.exp(exp_arg)

    @property
    def arrhenius_slope(self) -> float:
        """Slope of ln(k) vs 1/T: -E_a / R."""
        ea_j = self.current_ea_kj * 1000.0
        return -ea_j / R_GAS

    def energy_profile_val(self, q: float, is_cat: bool) -> float:
        """
        Calculates potential energy E(q) along reaction coordinate q in [0.0, 1.0].
        Guarantees C1 smooth transition with exact peak at E_react + E_a:
        - For q <= 0.5: E(q) rises from E_react to E_react + E_a
        - For q > 0.5: E(q) falls from E_react + E_a to E_react + ΔH
        """
        if q < 0.0 or q > 1.0:
            return 0.0
        e_react = 80.0  # Base energy for reactants (kJ/mol)
        ea = self._ea_cat_kj if is_cat else self._ea_uncat_kj
        dh = self._delta_h_kj

        if q <= 0.5:
            return e_react + ea * (math.sin(math.pi * q) ** 2)
        else:
            t = 2.0 * (q - 0.5)  # in [0, 1]
            drop = ea - dh
            return (e_react + ea) - drop * (math.sin(0.5 * math.pi * t) ** 2)

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> KineticsPreset:
        return self._preset

    @property
    def temp_k(self) -> float:
        return self._temp_k

    @property
    def temp_c(self) -> float:
        return self._temp_k - 273.15

    @property
    def has_catalyst(self) -> bool:
        return self._has_catalyst

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum kinetics preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._ea_uncat_kj = self._preset.ea_uncat_kj
        self._ea_cat_kj = self._preset.ea_cat_kj
        self._delta_h_kj = self._preset.delta_h_kj
        self._temp_k = self._preset.temp_k
        self._has_catalyst = True
        self._particles.clear()
        self.preset_changed.emit(preset_code)
        self.update()

    def set_temperature_k(self, t_k: float) -> None:
        self._temp_k = max(200.0, min(900.0, round(t_k, 1)))
        self.update()

    def toggle_catalyst(self) -> None:
        self._has_catalyst = not self._has_catalyst
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._anim_phase += 0.08
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi

        # Spawn reactive particles crossing barrier
        if len(self._particles) < 18:
            import random
            if len(self._particles) == 0 or self._particles[-1]["q"] > 0.12:
                # Speed proportional to temperature and presence of catalyst
                speed = 0.025 * (self._temp_k / 300.0) * (1.5 if self._has_catalyst else 1.0)
                self._particles.append({"q": 0.0, "speed": speed})

        # Advance particles
        new_particles = []
        for p in self._particles:
            p["q"] += p["speed"]
            if p["q"] < 1.0:
                new_particles.append(p)
        self._particles = new_particles

        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#080c14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Reaction Coordinate Potential Energy Profiles)
        left_w = 430.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_energy_viewport(painter, view_rect)

        # 4. Right Chemical Kinetics & Arrhenius Telemetry Panel
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Chemistry Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "ĐỘNG HÓA HỌC: ĐỊNH LUẬT ARRHENIUS & HÀNG RÀO THẾ NĂNG")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Phương trình Arrhenius k=A·exp(-E_a/RT) • Năng lượng hoạt hóa E_a • Nhiệt phản ứng ΔH • Xúc tác")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_energy_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # Coordinate Axes
        axis_x0 = rect.left() + 45.0
        axis_x1 = rect.right() - 20.0
        axis_y0 = rect.bottom() - 35.0
        axis_y_top = rect.top() + 35.0

        painter.setPen(QPen(QColor("#475569"), 1.2))
        # Y-axis (Potential Energy E)
        painter.drawLine(QPointF(axis_x0, axis_y0), QPointF(axis_x0, axis_y_top))
        # X-axis (Reaction Coordinate q)
        painter.drawLine(QPointF(axis_x0, axis_y0), QPointF(axis_x1, axis_y0))

        # Axis arrows and labels
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(axis_x0 - 32.0, axis_y_top + 10.0), "Thế năng\nE (kJ/mol)")
        painter.drawText(QPointF(axis_x1 - 85.0, axis_y0 + 20.0), "Tiến trình phản ứng ➔")

        # Energy curve mapping scale
        coord_w = axis_x1 - axis_x0 - 20.0
        max_e_val = 240.0
        scale_y = (axis_y0 - axis_y_top - 20.0) / max_e_val

        # 1. Uncatalyzed Curve (Red dashed/solid)
        path_uncat = QPainterPath()
        n_pts = 100
        for i in range(n_pts + 1):
            q_val = i / float(n_pts)
            e_val = self.energy_profile_val(q_val, is_cat=False)
            px = axis_x0 + 10.0 + q_val * coord_w
            py = axis_y0 - e_val * scale_y
            if i == 0:
                path_uncat.moveTo(px, py)
            else:
                path_uncat.lineTo(px, py)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#ef4444"), 2.0, Qt.PenStyle.DashLine if self._has_catalyst else Qt.PenStyle.SolidLine))
        painter.drawPath(path_uncat)

        # 2. Catalyzed Curve (Green solid, if catalyst present)
        if self._has_catalyst:
            path_cat = QPainterPath()
            for i in range(n_pts + 1):
                q_val = i / float(n_pts)
                e_val = self.energy_profile_val(q_val, is_cat=True)
                px = axis_x0 + 10.0 + q_val * coord_w
                py = axis_y0 - e_val * scale_y
                if i == 0:
                    path_cat.moveTo(px, py)
                else:
                    path_cat.lineTo(px, py)

            painter.setPen(QPen(QColor("#10b981"), 2.4))
            painter.drawPath(path_cat)

        # 3. Transition State [X‡] and Energy Levels
        # Reactants baseline (q = 0.05)
        e_react_y = axis_y0 - self.energy_profile_val(0.05, False) * scale_y
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(axis_x0 + 14.0, e_react_y - 8.0), "CHẤT ĐẦU (R)")

        # Products baseline (q = 0.95)
        e_prod_y = axis_y0 - self.energy_profile_val(0.95, False) * scale_y
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(axis_x0 + coord_w - 75.0, e_prod_y - 8.0), "SẢN PHẨM (P)")

        # Reaction Enthalpy ΔH indicator
        dh_x = axis_x0 + coord_w - 20.0
        painter.setPen(QPen(QColor("#fbbf24"), 1.2, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(dh_x, e_react_y), QPointF(dh_x, e_prod_y))

        dh_str = f"ΔH = {self._delta_h_kj:+.1f} kJ"
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(dh_x - 40.0, (e_react_y + e_prod_y) / 2.0 + 4.0), dh_str)

        # Peak Transition State [X‡]
        peak_q = 0.5
        peak_e = self.energy_profile_val(peak_q, self._has_catalyst)
        peak_px = axis_x0 + 10.0 + peak_q * coord_w
        peak_py = axis_y0 - peak_e * scale_y

        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(QPointF(peak_px, peak_py), 4.5, 4.5)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(QPointF(peak_px - 22.0, peak_py - 10.0), "Trạng thái [X‡]")

        # 4. Animated Reacting Molecules crossing barrier
        if self._is_animating and self._particles:
            for p in self._particles:
                q_p = p["q"]
                e_p = self.energy_profile_val(q_p, self._has_catalyst)
                px_p = axis_x0 + 10.0 + q_p * coord_w
                py_p = axis_y0 - e_p * scale_y

                painter.setBrush(QBrush(QColor("#38bdf8") if q_p < 0.5 else QColor("#10b981")))
                painter.setPen(QPen(QColor("#ffffff"), 0.8))
                painter.drawEllipse(QPointF(px_p, py_p), 3.5, 3.5)

        # 5. Header in viewport
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), f"PHẢN ỨNG: {self._preset.reaction_str}")

        # Legend badge
        leg_r = QRectF(rect.right() - 180.0, rect.top() + 8.0, 170.0, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRoundedRect(leg_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(QPointF(leg_r.left() + 6.0, leg_r.top() + 14.0), "┅ Không xúc tác")
        painter.setPen(QColor("#10b981"))
        painter.drawText(QPointF(leg_r.left() + 85.0, leg_r.top() + 14.0), "━ Có xúc tác")

        painter.restore()

    # -------------------------------------------------------------------------
    # Right Telemetry & Interactive Controls
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THÔNG SỐ ĐỘNG HÓA HỌC:")

        # Temperature & Catalyst Status Card
        y += 18.0
        card_r = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(card_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 75.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"{self._temp_k:.0f} K")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        cat_txt = "Đang bật chất xúc tác (Hạ E_a)" if self._has_catalyst else "Không dùng chất xúc tác"
        painter.drawText(QRectF(ox + 85.0, y + 2.0, w - 90.0, 28.0), Qt.AlignmentFlag.AlignVCenter, f"T = {self.temp_c:.1f} °C\n{cat_txt}")

        # Quantitative Measurements Readout
        y += 40.0
        ea_cur = self.current_ea_kj
        speedup = self.catalysis_speedup_ratio
        k_val = self.current_rate_constant
        k_str = f"{k_val:.2e} s⁻¹" if k_val > 0 else "0.0 s⁻¹"
        speedup_str = f"{speedup:,.0f}× lần" if speedup < 1e7 else f"{speedup:.1e}× lần"

        rows = [
            (f"• E_a chưa xúc tác: {self._ea_uncat_kj:.1f} kJ/mol", QColor("#ef4444")),
            (f"• E_a có xúc tác: {self._ea_cat_kj:.1f} kJ/mol", QColor("#10b981")),
            (f"• E_a hiện tại: {ea_cur:.1f} kJ/mol", QColor("#fbbf24")),
            (f"• Nhiệt phản ứng ΔH: {self._delta_h_kj:+.1f} kJ/mol", QColor("#38bdf8")),
            (f"• Hằng số tốc độ k: {k_str}", QColor("#cbd5e1")),
            (f"• Tốc độ tăng gấp: {speedup_str}", QColor("#10b981")),
            (f"• Bản chất nhiệt: {'Tỏa nhiệt (ΔH < 0)' if self._delta_h_kj < 0 else 'Thu nhiệt (ΔH > 0)'}", QColor("#e2e8f0")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 15.0

        # Action Buttons: Toggle Catalyst, Adjust Temperature +/-
        y = rect.bottom() - 48.0
        cat_btn = QRectF(ox, y, w, 20.0)
        self._btn_rects["toggle_catalyst"] = cat_btn

        painter.setBrush(QBrush(QColor("#047857") if self._has_catalyst else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981") if self._has_catalyst else QColor("#475569"), 1.0))
        painter.drawRoundedRect(cat_btn, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(cat_btn, Qt.AlignmentFlag.AlignCenter, "🧪 Bật Xúc tác" if not self._has_catalyst else "🧪 Đang dùng Xúc tác (Bấm để tắt)")

        y += 24.0
        t_m = QRectF(ox, y, (w - 8.0) / 2.0, 20.0)
        t_p = QRectF(ox + (w - 8.0) / 2.0 + 8.0, y, (w - 8.0) / 2.0, 20.0)
        self._btn_rects["temp_minus"] = t_m
        self._btn_rects["temp_plus"] = t_p

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(t_m, 3.0, 3.0)
        painter.drawRoundedRect(t_p, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(t_m, Qt.AlignmentFlag.AlignCenter, "Nhiệt độ -25K")
        painter.drawText(t_p, Qt.AlignmentFlag.AlignCenter, "Nhiệt độ +25K")

    # -------------------------------------------------------------------------
    # Bottom Chemistry Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "QUY LUẬT ĐỘNG HÓA HỌC & ĐỊNH LUẬT ARRHENIUS (1889):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Phương trình Arrhenius k = A·exp(-E_a/RT) chứng minh hằng số tốc độ tăng theo hàm mũ khi nhiệt độ tăng.\n"
            f"• Chất xúc tác mở ra con đường phản ứng mới có năng lượng hoạt hóa E_a thấp hơn, làm tăng vọt tốc độ mà không làm đổi ΔH.\n"
            f"• Trường hợp: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Toggle catalyst
        if self._btn_rects.get("toggle_catalyst") and self._btn_rects["toggle_catalyst"].contains(local_pos):
            self.toggle_catalyst()
            return True

        # 3. Temperature steppers
        if self._btn_rects.get("temp_minus") and self._btn_rects["temp_minus"].contains(local_pos):
            self.set_temperature_k(self._temp_k - 25.0)
            return True
        if self._btn_rects.get("temp_plus") and self._btn_rects["temp_plus"].contains(local_pos):
            self.set_temperature_k(self._temp_k + 25.0)
            return True

        # 4. Stamp button
        if self._btn_rects.get("stamp") and self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
