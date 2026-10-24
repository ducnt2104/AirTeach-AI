"""
AirTeach AI — Interactive Mechanical Engineering & Materials Science:
Stress-Strain Curve, Hooke's Law Elasticity, Young's Modulus, Yield Strength & Fracture Simulator.
Milestone 152 (Task I.139).

Provides real-time interactive simulation of:
1. Hooke's Law Linear Elasticity: sigma = E * epsilon (where epsilon <= epsilon_yield).
2. Modulus of Resilience: U_r = sigma_y^2 / (2 * E).
3. Plastic deformation & strain hardening up to Ultimate Tensile Strength (UTS, sigma_uts).
4. Diffuse necking initiation (Considere criterion) and catastrophic ductile/brittle fracture.
5. Modulus of Toughness: U_T = integral(sigma * d_epsilon) from 0 to epsilon_fracture.
6. 3D Dog-bone tensile specimen stretching, localized necking, and cup-and-cone rupture.
7. 5 Authentic Curriculum Presets: ASTM A36 Structural Steel, 7075-T6 Aerospace Aluminum,
   Ti-6Al-4V Grade 5 Titanium, Class 30 Gray Cast Iron (brittle), and C11000 Annealed Copper.
"""

from __future__ import annotations

import math
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
class MaterialPreset:
    """Represents a curriculum materials science tensile testing scenario."""
    code: str
    title_vi: str
    material_name: str
    youngs_modulus_GPa: float       # E in GPa (10^9 Pa)
    yield_strength_MPa: float       # sigma_y (0.2% offset) in MPa
    uts_strength_MPa: float         # Ultimate Tensile Strength in MPa
    fracture_strength_MPa: float    # Breaking stress in MPa
    yield_strain_pct: float         # Strain at yield in %
    uts_strain_pct: float           # Strain at UTS in %
    fracture_strain_pct: float      # Elongation at break in %
    poisson_ratio: float            # nu (lateral contraction)
    is_brittle: bool                # True for cast iron/ceramic (negligible plastic strain)
    accent_color: QColor            # UI theme color
    description: str

    @property
    def poissons_ratio(self) -> float:
        return self.poisson_ratio



MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "STRUCTURAL_STEEL_A36_DUCTILE": MaterialPreset(
        code="STRUCTURAL_STEEL_A36_DUCTILE",
        title_vi="Thép Kết Cấu ASTM A36 (Low Carbon Steel - Dẻo Dai)",
        material_name="ASTM A36 Structural Carbon Steel",
        youngs_modulus_GPa=200.0,
        yield_strength_MPa=250.0,
        uts_strength_MPa=450.0,
        fracture_strength_MPa=360.0,
        yield_strain_pct=0.125,        # 250 / 200000 = 0.00125 = 0.125%
        uts_strain_pct=15.0,
        fracture_strain_pct=22.0,
        poisson_ratio=0.30,
        is_brittle=False,
        accent_color=QColor(0, 210, 255), # Steel cyan
        description="Thép xây dựng kinh điển; có thềm chảy dẻo rõ rệt, biến cứng cơ học mạnh mẽ và độ dai phá hủy cao.",
    ),
    "AEROSPACE_ALUMINUM_7075_T6": MaterialPreset(
        code="AEROSPACE_ALUMINUM_7075_T6",
        title_vi="Hợp Kim Nhôm Hàng Không 7075-T6 (High Strength Al)",
        material_name="Aluminum Alloy 7075-T6 (Al-Zn-Mg)",
        youngs_modulus_GPa=71.7,
        yield_strength_MPa=503.0,
        uts_strength_MPa=572.0,
        fracture_strength_MPa=480.0,
        yield_strain_pct=0.701,        # 503 / 71700 = 0.701%
        uts_strain_pct=7.5,
        fracture_strain_pct=11.0,
        poisson_ratio=0.33,
        is_brittle=False,
        accent_color=QColor(255, 170, 0), # Aerospace amber
        description="Hợp kim nhôm siêu bền tôi già hóa; tỷ số độ bền trên khối lượng vượt trội ứng dụng trong khung thân máy bay.",
    ),
    "TITANIUM_ALLOY_TI_6AL_4V_GRADE5": MaterialPreset(
        code="TITANIUM_ALLOY_TI_6AL_4V_GRADE5",
        title_vi="Hợp Kim Titan Ti-6Al-4V Grade 5 (Y Sinh & Vũ Trụ)",
        material_name="Titanium Ti-6Al-4V (Grade 5)",
        youngs_modulus_GPa=114.0,
        yield_strength_MPa=880.0,
        uts_strength_MPa=950.0,
        fracture_strength_MPa=790.0,
        yield_strain_pct=0.772,
        uts_strain_pct=9.0,
        fracture_strain_pct=14.0,
        poisson_ratio=0.34,
        is_brittle=False,
        accent_color=QColor(180, 100, 255), # Titanium violet
        description="Hợp kim titan hai pha alpha-beta bền nhiệt và chống ăn mòn tuyệt hảo, chế tạo turbine phản lực và khớp cấy ghép y tế.",
    ),
    "GRAY_CAST_IRON_CLASS30_BRITTLE": MaterialPreset(
        code="GRAY_CAST_IRON_CLASS30_BRITTLE",
        title_vi="Gang Xám ASTM Class 30 (Gray Cast Iron - Giòn)",
        material_name="Gray Cast Iron (ASTM Class 30)",
        youngs_modulus_GPa=100.0,
        yield_strength_MPa=214.0,       # Fractures right near UTS
        uts_strength_MPa=214.0,
        fracture_strength_MPa=214.0,
        yield_strain_pct=0.214,
        uts_strain_pct=0.60,
        fracture_strain_pct=0.60,       # Negligible elongation
        poisson_ratio=0.26,
        is_brittle=True,
        accent_color=QColor(239, 71, 111), # Brittle red
        description="Vật liệu giòn chứa phiến graphit; phá hủy giòn gãy đột ngột mà không có biến dạng thắt thân dẻo.",
    ),
    "COPPER_C11000_ELECTROLYTIC_TOUGH_PITCH": MaterialPreset(
        code="COPPER_C11000_ELECTROLYTIC_TOUGH_PITCH",
        title_vi="Đồng Đỏ Tinh Khiết C11000 (Annealed Copper - Siêu Dẻo)",
        material_name="Electrolytic Tough Pitch Copper (ETP C11000)",
        youngs_modulus_GPa=117.0,
        yield_strength_MPa=70.0,
        uts_strength_MPa=220.0,
        fracture_strength_MPa=160.0,
        yield_strain_pct=0.060,
        uts_strain_pct=32.0,
        fracture_strain_pct=45.0,       # Extreme 45% elongation
        poisson_ratio=0.34,
        is_brittle=False,
        accent_color=QColor(244, 162, 97), # Copper rose
        description="Đồng ủ dẻo cao cấp; giới hạn chảy thấp nhưng độ giãn dài đứt tới 45%, minh họa xuất sắc cơ chế biến cứng cơ học.",
    ),
}


class InteractiveStressStrainItem(QGraphicsObject):
    """
    High-fidelity interactive QGraphicsObject simulating tensile testing (ASTM E8),
    Hooke's law elasticity, Young's modulus, yield point, strain hardening,
    diffuse necking, and ductile/brittle fracture.
    """

    # Signals
    stress_computed = Signal(dict)
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
        self._active_preset_code: str = "STRUCTURAL_STEEL_A36_DUCTILE"
        self._preset: MaterialPreset = MATERIAL_PRESETS[self._active_preset_code]

        # Specimen dimensions: gauge length L0 = 50.0 mm, diameter d0 = 10.0 mm
        self._gauge_length_mm: float = 50.0
        self._specimen_diameter_mm: float = 10.0

        # Current applied strain epsilon in percent (%)
        self._current_strain_pct: float = self._preset.yield_strain_pct * 0.80

        # Animated hydraulic pull state
        self._is_pulling: bool = False
        self._pull_speed: float = 0.05  # % strain per tick

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None

        # Timer for 60fps real-time tensile test deformation animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(20)

    # ------------------------------------------------------------------
    # Physics Properties & Analytical Mechanics of Materials Equations
    # ------------------------------------------------------------------

    @property
    def cross_section_area_mm2(self) -> float:
        """Original cross-sectional area A_0 = pi * d_0^2 / 4."""
        r = self._specimen_diameter_mm * 0.5
        return math.pi * r * r

    @property
    def elongation_mm(self) -> float:
        """Gauge elongation Delta L = (epsilon / 100) * L_0 in mm."""
        return (self._current_strain_pct / 100.0) * self._gauge_length_mm

    @property
    def is_fractured(self) -> bool:
        """True if current strain has reached or exceeded fracture elongation."""
        return self._current_strain_pct >= self._preset.fracture_strain_pct

    def stress_at_strain_MPa(self, eps_pct: float) -> float:
        """
        Calculates engineering stress sigma (MPa) as a function of engineering strain eps_pct (%):
        1. Linear Elastic: sigma = E * eps (up to yield)
        2. Plastic Hardening: power law hardening up to UTS
        3. Post-UTS Necking: linear engineering stress drop to fracture
        """
        p = self._preset
        if eps_pct <= 0.0:
            return 0.0

        if eps_pct >= p.fracture_strain_pct:
            return 0.0 # Fractured specimen cannot sustain load

        eps_frac = eps_pct / 100.0
        e_gpa = p.youngs_modulus_GPa

        # 1. Linear Elastic Regime (sigma <= sigma_y)
        eps_y_pct = p.yield_strain_pct
        if eps_pct <= eps_y_pct:
            # Hooke's Law: sigma = E * eps (E in GPa, convert to MPa: * 1000)
            return (e_gpa * 1000.0) * eps_frac

        # Brittle materials rupture immediately after yield/UTS
        if p.is_brittle:
            return p.uts_strength_MPa

        # 2. Plastic Strain Hardening Regime (eps_y < eps <= eps_uts)
        eps_uts_pct = p.uts_strain_pct
        if eps_pct <= eps_uts_pct:
            t = (eps_pct - eps_y_pct) / max(1e-4, eps_uts_pct - eps_y_pct)
            # Power law curve exponent n = 0.45
            hardening = math.pow(max(0.0, min(1.0, t)), 0.45)
            return p.yield_strength_MPa + (p.uts_strength_MPa - p.yield_strength_MPa) * hardening

        # 3. Post-UTS Necking Regime (eps_uts < eps < eps_f)
        eps_f_pct = p.fracture_strain_pct
        t_neck = (eps_pct - eps_uts_pct) / max(1e-4, eps_f_pct - eps_uts_pct)
        return p.uts_strength_MPa - (p.uts_strength_MPa - p.fracture_strength_MPa) * min(1.0, t_neck)

    @property
    def current_stress_MPa(self) -> float:
        """Current engineering tensile stress in MPa."""
        return self.stress_at_strain_MPa(self._current_strain_pct)

    @property
    def tensile_force_kN(self) -> float:
        """Applied tensile force F = sigma * A_0 in kilonewtons (kN)."""
        sigma = self.current_stress_MPa
        area = self.cross_section_area_mm2
        # Force in Newtons: sigma (N/mm^2) * area (mm^2) -> divide by 1000 for kN
        return (sigma * area) / 1000.0

    @property
    def modulus_of_resilience_MJ_m3(self) -> float:
        """Modulus of resilience U_r = sigma_y^2 / (2 * E) in MJ/m^3."""
        sy = self._preset.yield_strength_MPa
        e = self._preset.youngs_modulus_GPa * 1000.0 # in MPa
        return (sy * sy) / (2.0 * e)

    @property
    def modulus_of_resilience_kJ_m3(self) -> float:
        """Modulus of resilience U_r in kJ/m^3 (1 MJ/m^3 = 1000 kJ/m^3)."""
        return self.modulus_of_resilience_MJ_m3 * 1000.0

    @property
    def gauge_diameter_mm(self) -> float:
        """Original gauge diameter d_0 in mm."""
        return self._specimen_diameter_mm

    @property
    def gauge_length_mm(self) -> float:
        """Original gauge length L_0 in mm."""
        return self._gauge_length_mm


    @property
    def modulus_of_toughness_MJ_m3(self) -> float:
        """Modulus of toughness U_T ≈ ((sigma_y + 2 * sigma_uts) / 3) * (eps_f / 100) in MJ/m^3."""
        p = self._preset
        avg_stress = (p.yield_strength_MPa + 2.0 * p.uts_strength_MPa) / 3.0
        return avg_stress * (p.fracture_strain_pct / 100.0)

    @property
    def deformation_regime_name(self) -> str:
        """Returns Vietnamese description of the current mechanical state."""
        if self.is_fractured:
            return "Đã đứt gãy hoàn toàn (Fracture)"
        eps = self._current_strain_pct
        p = self._preset
        if eps <= p.yield_strain_pct:
            return "Đàn hồi tuyến tính (Hooke Elastic)"
        elif eps <= p.uts_strain_pct:
            return "Biến dạng dẻo & Biến cứng (Plastic Hardening)"
        else:
            return "Thắt thân cục bộ (Necking Instability)"

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches curriculum material scenario."""
        if preset_code in MATERIAL_PRESETS:
            self._active_preset_code = preset_code
            self._preset = MATERIAL_PRESETS[preset_code]
            self._current_strain_pct = self._preset.yield_strain_pct * 0.80
            self._is_pulling = False
            self._emit_computed_values()
            self.update()

    def set_material_preset(self, preset_code: str) -> None:
        """Alias for set_preset."""
        self.set_preset(preset_code)

    def set_strain_pct(self, strain: float) -> None:
        """Sets engineering strain in percent [0.0, 50.0] %."""
        self._current_strain_pct = max(0.0, min(50.0, round(strain, 6)))
        self._emit_computed_values()
        self.update()

    def pull_step(self, delta_pct: float) -> None:
        """Increases or decreases strain by delta_pct."""
        self.set_strain_pct(self._current_strain_pct + delta_pct)

    def step_strain(self, delta_pct: float) -> None:
        """Alias for pull_step."""
        self.pull_step(delta_pct)

    def cycle_preset(self) -> None:
        """Cycles to the next material preset."""
        self._handle_button_click("preset_next")

    def toggle_auto_pull(self) -> None:
        """Toggles continuous motorized tensile pulling."""
        self._is_pulling = not self._is_pulling

    def reset_defaults(self) -> None:
        """Restores default values."""
        self.set_preset(self._active_preset_code)

    def reset_test(self) -> None:
        """Resets specimen strain to zero."""
        self.set_strain_pct(0.0)


    def _emit_computed_values(self) -> None:
        """Emits real-time materials science telemetry."""
        data = {
            "preset_code": self._active_preset_code,
            "material_name": self._preset.material_name,
            "youngs_modulus_GPa": self._preset.youngs_modulus_GPa,
            "yield_strength_MPa": self._preset.yield_strength_MPa,
            "uts_strength_MPa": self._preset.uts_strength_MPa,
            "current_strain_pct": self._current_strain_pct,
            "current_stress_MPa": self.current_stress_MPa,
            "tensile_force_kN": self.tensile_force_kN,
            "elongation_mm": self.elongation_mm,
            "is_fractured": self.is_fractured,
            "regime": self.deformation_regime_name,
        }
        self.stress_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Deformation animation tick."""
        if self._is_pulling and not self.is_fractured:
            self._current_strain_pct += self._pull_speed
            if self.is_fractured:
                self._is_pulling = False
            self._emit_computed_values()
        self.update()

    # ------------------------------------------------------------------
    # UI Layout & Buttons
    # ------------------------------------------------------------------

    def _layout_buttons(self) -> None:
        ctrl_x = 385.0
        row1_y = 248.0
        btn_w_half = 72.0
        btn_h = 24.0

        self._buttons["preset_prev"] = QRectF(ctrl_x, row1_y, 45.0, btn_h)
        self._buttons["preset_next"] = QRectF(ctrl_x + 52.0, row1_y, 45.0, btn_h)
        self._buttons["auto_pull"] = QRectF(ctrl_x + 104.0, row1_y, 172.0, btn_h)

        row2_y = 278.0
        self._buttons["pull_more"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["pull_less"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["set_yield"] = QRectF(ctrl_x + 158.0, row2_y, 118.0, btn_h)

        row3_y = 308.0
        self._buttons["set_uts"] = QRectF(ctrl_x, row3_y, 85.0, btn_h)
        self._buttons["set_break"] = QRectF(ctrl_x + 92.0, row3_y, 85.0, btn_h)
        self._buttons["unload_zero"] = QRectF(ctrl_x + 184.0, row3_y, 92.0, btn_h)

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

        # 3. Left Side: Dog-bone Tensile Testing Machine Specimen
        self._paint_tensile_specimen(painter)

        # 4. Right Side: Engineering Stress-Strain Curve sigma(epsilon)
        self._paint_stress_strain_curve(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        rect = self.boundingRect()
        bg_grad = QLinearGradient(0.0, 0.0, rect.width(), rect.height())
        bg_grad.setColorAt(0.0, QColor(14, 18, 28, 245))
        bg_grad.setColorAt(0.6, QColor(18, 24, 38, 245))
        bg_grad.setColorAt(1.0, QColor(12, 16, 26, 250))

        painter.setPen(QPen(QColor(45, 62, 90, 220), 1.5))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(20, 28, "⚙ ĐƯỜNG CONG ỨNG SUẤT - BIẾN DẠNG & THỬ KÉO VẬT LIỆU")

        font_sub = QFont("Segoe UI", 8, QFont.Weight.Normal)
        painter.setFont(font_sub)
        painter.setPen(QColor(160, 180, 210))
        reg_txt = self.deformation_regime_name
        painter.drawText(20, 44, f"Vật liệu: {self._preset.material_name} | Trạng thái: {reg_txt}")

        # Badge (Top Right)
        badge_rect = QRectF(self.CARD_W - 200.0, 14.0, 180.0, 22.0)
        accent = self._preset.accent_color
        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 40)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        badge_txt = f"E = {self._preset.youngs_modulus_GPa:.1f} GPa • σ_y = {self._preset.yield_strength_MPa:.0f} MPa"
        font_badge = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_badge)
        painter.setPen(accent)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_txt)

    def _paint_tensile_specimen(self, painter: QPainter) -> None:
        """
        Renders the tensile testing machine:
        - Top & Bottom hydraulic grips
        - Dog-bone specimen with dynamic elongation Delta L and localized necking
        - Tensile force F vector arrows
        """
        box = QRectF(20.0, 56.0, 345.0, 180.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.2))
        painter.setBrush(QBrush(QColor(8, 12, 20, 220)))
        painter.drawRoundedRect(box, 8.0, 8.0)

        cx = (box.left() + box.right()) * 0.5 - 40.0
        cy = (box.top() + box.bottom()) * 0.5

        # Specimen elongation in px
        elong_px = min(28.0, (self._current_strain_pct / self._preset.fracture_strain_pct) * 22.0)
        is_fractured = self.is_fractured

        # Grips (Top and Bottom hydraulic jaws)
        grip_w = 46.0
        grip_h = 18.0

        top_grip_y = cy - 50.0 - elong_px * 0.5
        bot_grip_y = cy + 32.0 + elong_px * 0.5

        painter.setPen(QPen(QColor(100, 120, 145), 1.5))
        painter.setBrush(QBrush(QColor(40, 50, 65)))
        painter.drawRoundedRect(QRectF(cx - grip_w * 0.5, top_grip_y, grip_w, grip_h), 3.0, 3.0)
        painter.drawRoundedRect(QRectF(cx - grip_w * 0.5, bot_grip_y, grip_w, grip_h), 3.0, 3.0)

        # Force arrows
        if not is_fractured and self.current_stress_MPa > 0.0:
            painter.setPen(QPen(QColor(255, 200, 0), 2.0))
            # Pull upwards on top grip
            painter.drawLine(cx, top_grip_y - 2.0, cx, top_grip_y - 14.0)
            painter.drawLine(cx, top_grip_y - 14.0, cx - 4.0, top_grip_y - 8.0)
            painter.drawLine(cx, top_grip_y - 14.0, cx + 4.0, top_grip_y - 8.0)

            # Pull downwards on bottom grip
            painter.drawLine(cx, bot_grip_y + grip_h + 2.0, cx, bot_grip_y + grip_h + 14.0)
            painter.drawLine(cx, bot_grip_y + grip_h + 14.0, cx - 4.0, bot_grip_y + grip_h + 8.0)
            painter.drawLine(cx, bot_grip_y + grip_h + 14.0, cx + 4.0, bot_grip_y + grip_h + 8.0)

        # Dog-bone specimen gauge section
        gauge_top = top_grip_y + grip_h
        gauge_bot = bot_grip_y
        gauge_h = gauge_bot - gauge_top

        # Specimen width narrows under Poisson lateral contraction & necking
        neck_factor = 1.0
        if self._current_strain_pct > self._preset.uts_strain_pct and not is_fractured:
            neck_frac = (self._current_strain_pct - self._preset.uts_strain_pct) / max(1e-4, self._preset.fracture_strain_pct - self._preset.uts_strain_pct)
            neck_factor = 1.0 - 0.45 * neck_frac

        gauge_w = 18.0 * neck_factor

        specimen_color = self._preset.accent_color
        if is_fractured:
            specimen_color = QColor(140, 40, 40)
        elif self._current_strain_pct > self._preset.yield_strain_pct:
            specimen_color = QColor(255, 120, 50) # Plastic yield glow

        if not is_fractured:
            # Solid intact stretched specimen
            spec_rect = QRectF(cx - gauge_w * 0.5, gauge_top, gauge_w, gauge_h)
            painter.setPen(QPen(specimen_color.lighter(130), 1.2))
            painter.setBrush(QBrush(specimen_color))
            painter.drawRoundedRect(spec_rect, 2.0, 2.0)
        else:
            # Fractured: two separated halves with cup-and-cone break
            half_h = gauge_h * 0.42
            top_half = QRectF(cx - gauge_w * 0.5, gauge_top, gauge_w, half_h)
            bot_half = QRectF(cx - gauge_w * 0.5, gauge_bot - half_h, gauge_w, half_h)

            painter.setPen(QPen(QColor(255, 80, 80), 1.2))
            painter.setBrush(QBrush(QColor(160, 40, 50)))
            painter.drawRect(top_half)
            painter.drawRect(bot_half)

            # Fracture label
            font_fr = QFont("Segoe UI", 7, QFont.Weight.Bold)
            painter.setFont(font_fr)
            painter.setPen(QColor(255, 80, 80))
            painter.drawText(int(cx - 30.0), int(cy + 4.0), "ĐỨT GÃY (RUPTURE)")

        # Right side of chamber: Digital load cell display
        dc_rect = QRectF(cx + 45.0, cy - 40.0, 130.0, 75.0)
        painter.setPen(QPen(QColor(0, 220, 180), 1.2))
        painter.setBrush(QBrush(QColor(6, 20, 25, 230)))
        painter.drawRoundedRect(dc_rect, 6.0, 6.0)

        font_lc = QFont("Segoe UI", 7, QFont.Weight.Bold)
        font_num = QFont("Consolas", 10, QFont.Weight.Bold)

        painter.setFont(font_lc)
        painter.setPen(QColor(150, 210, 200))
        painter.drawText(int(dc_rect.left() + 8.0), int(dc_rect.top() + 16.0), "CẢM BIẾN TẢI KÉO (LOAD CELL):")

        painter.setFont(font_num)
        painter.setPen(QColor(0, 255, 200))
        painter.drawText(int(dc_rect.left() + 8.0), int(dc_rect.top() + 38.0), f"F = {self.tensile_force_kN:5.2f} kN")

        painter.setFont(font_lc)
        painter.setPen(QColor(255, 210, 80))
        painter.drawText(
            int(dc_rect.left() + 8.0),
            int(dc_rect.top() + 58.0),
            f"ΔL = {self.elongation_mm:4.2f} mm ({self._current_strain_pct:4.2f}%)",
        )

        # Telemetry panel below box (y = 245 to 395)
        self._paint_telemetry_table(painter)

    def _paint_telemetry_table(self, painter: QPainter) -> None:
        """Renders exact materials science parameters table."""
        t_box = QRectF(20.0, 245.0, 345.0, 150.0)
        painter.setPen(QPen(QColor(35, 48, 70), 1.0))
        painter.setBrush(QBrush(QColor(10, 15, 25, 220)))
        painter.drawRoundedRect(t_box, 8.0, 8.0)

        font_h = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_h)
        painter.setPen(QColor(240, 200, 100))
        painter.drawText(int(t_box.left() + 12.0), int(t_box.top() + 18.0), "⚙ THÔNG SỐ CƠ HỌC VẬT LIỆU & BIẾN DẠNG")

        font_k = QFont("Segoe UI", 7, QFont.Weight.Normal)
        font_v = QFont("Consolas", 8, QFont.Weight.Bold)

        p = self._preset
        lines = [
            ("Ứng suất kỹ thuật hiện tại σ:", f"{self.current_stress_MPa:6.1f} MPa", QColor("#38bdf8")),
            ("Độ biến dạng tỷ đối ε:", f"{self._current_strain_pct:5.2f} %", QColor("#fbbf24")),
            ("Mô đun đàn hồi Young E:", f"{p.youngs_modulus_GPa:.1f} GPa", QColor("#4ade80")),
            ("Giới hạn chảy dẻo σ_y (0.2%):", f"{p.yield_strength_MPa:.0f} MPa", QColor("#22d3ee")),
            ("Độ bền kéo cực hạn (UTS):", f"{p.uts_strength_MPa:.0f} MPa", QColor("#f43f5e")),
        ]

        curr_y = t_box.top() + 38.0
        for lbl, val_str, val_col in lines:
            painter.setFont(font_k)
            painter.setPen(QColor(160, 180, 205))
            painter.drawText(int(t_box.left() + 12.0), int(curr_y), lbl)

            painter.setFont(font_v)
            painter.setPen(val_col)
            painter.drawText(int(t_box.left() + 195.0), int(curr_y), val_str)
            curr_y += 22.0

    def _paint_stress_strain_curve(self, painter: QPainter) -> None:
        """Renders standard engineering stress-strain curve sigma(epsilon)."""
        plot_rect = QRectF(385.0, 56.0, 275.0, 180.0)
        painter.setPen(QPen(QColor(40, 56, 82), 1.0))
        painter.setBrush(QBrush(QColor(12, 17, 28, 220)))
        painter.drawRoundedRect(plot_rect, 8.0, 8.0)

        # Plot axes
        ax_x0 = plot_rect.left() + 35.0
        ax_y0 = plot_rect.bottom() - 25.0
        ax_w = plot_rect.width() - 50.0
        ax_h = plot_rect.height() - 50.0

        painter.setPen(QPen(QColor(50, 68, 95), 1.0))
        painter.drawLine(ax_x0, ax_y0, ax_x0 + ax_w, ax_y0) # epsilon axis
        painter.drawLine(ax_x0, ax_y0, ax_x0, ax_y0 - ax_h) # sigma axis

        font_ax = QFont("Segoe UI", 6, QFont.Weight.Normal)
        painter.setFont(font_ax)
        painter.setPen(QColor(130, 150, 180))
        painter.drawText(int(ax_x0 + ax_w - 40.0), int(ax_y0 + 15.0), "ε (%) →")
        painter.drawText(int(ax_x0 - 25.0), int(ax_y0 - ax_h + 8.0), "σ (MPa)")

        p = self._preset
        max_eps = p.fracture_strain_pct * 1.15
        max_sig = p.uts_strength_MPa * 1.20

        # Sample full curve points
        curve_path = QPainterPath()
        points: list[QPointF] = []
        n_steps = 75

        for i in range(n_steps + 1):
            eps_val = (i / n_steps) * p.fracture_strain_pct
            sig_val = self.stress_at_strain_MPa(eps_val)

            px = ax_x0 + (eps_val / max_eps) * ax_w
            py = ax_y0 - (sig_val / max_sig) * ax_h
            points.append(QPointF(px, py))

        if points:
            curve_path.moveTo(points[0])
            for pt in points[1:]:
                curve_path.lineTo(pt)

            # Area under curve (Toughness)
            area_path = QPainterPath(curve_path)
            area_path.lineTo(points[-1].x(), ax_y0)
            area_path.lineTo(points[0].x(), ax_y0)
            area_path.closeSubpath()

            grad = QLinearGradient(0.0, ax_y0 - ax_h, 0.0, ax_y0)
            accent = self._preset.accent_color
            grad.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 100))
            grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 15))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawPath(area_path)

            painter.setPen(QPen(accent, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(curve_path)

        # Mark yield point (σ_y) and UTS point (σ_uts)
        def draw_marker(eps_m: float, sig_m: float, color: QColor, label: str):
            mx = ax_x0 + (eps_m / max_eps) * ax_w
            my = ax_y0 - (sig_m / max_sig) * ax_h
            painter.setPen(QPen(color, 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(mx, ax_y0, mx, my)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(mx, my), 3.0, 3.0)

            font_m = QFont("Segoe UI", 6, QFont.Weight.Bold)
            painter.setFont(font_m)
            painter.setPen(color)
            painter.drawText(int(mx - 8.0), int(my - 5.0), label)

        draw_marker(p.yield_strain_pct, p.yield_strength_MPa, QColor("#38bdf8"), "σ_y")
        draw_marker(p.uts_strain_pct, p.uts_strength_MPa, QColor("#fbbf24"), "UTS")
        draw_marker(p.fracture_strain_pct, p.fracture_strength_MPa, QColor("#ef4444"), "Đứt")

        # Dynamic operating point cursor
        cur_eps = min(p.fracture_strain_pct, self._current_strain_pct)
        cur_sig = self.stress_at_strain_MPa(cur_eps)
        cur_x = ax_x0 + (cur_eps / max_eps) * ax_w
        cur_y = ax_y0 - (cur_sig / max_sig) * ax_h

        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.setBrush(QBrush(QColor(255, 60, 60) if self.is_fractured else QColor(0, 255, 200)))
        painter.drawEllipse(QPointF(cur_x, cur_y), 4.5, 4.5)

    def _paint_buttons(self, painter: QPainter) -> None:
        button_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "auto_pull": "⏸ Tạm dừng" if self._is_pulling else "▶ Kéo tự động",
            "pull_more": "Kéo thêm (+)",
            "pull_less": "Nhả tải (−)",
            "set_yield": "Điểm chảy σ_y",
            "set_uts": "Cực hạn UTS",
            "set_break": "Kéo đứt 💥",
            "unload_zero": "Về ε = 0",
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
            elif btn_key in ("auto_pull", "set_break"):
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
        preset_keys = list(MATERIAL_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "auto_pull":
            self.toggle_auto_pull()
        elif btn_key == "pull_more":
            step = 0.50 if not self._preset.is_brittle else 0.05
            self.pull_step(step)
        elif btn_key == "pull_less":
            step = 0.50 if not self._preset.is_brittle else 0.05
            self.pull_step(-step)
        elif btn_key == "set_yield":
            self.set_strain_pct(self._preset.yield_strain_pct)
        elif btn_key == "set_uts":
            self.set_strain_pct(self._preset.uts_strain_pct)
        elif btn_key == "set_break":
            self.set_strain_pct(self._preset.fracture_strain_pct)
        elif btn_key == "unload_zero":
            self.set_strain_pct(0.0)
        elif btn_key == "reset":
            self.reset_defaults()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "material_name": self._preset.material_name,
            "youngs_modulus_GPa": self._preset.youngs_modulus_GPa,
            "yield_strength_MPa": self._preset.yield_strength_MPa,
            "uts_strength_MPa": self._preset.uts_strength_MPa,
            "current_strain_pct": self._current_strain_pct,
            "current_stress_MPa": self.current_stress_MPa,
            "tensile_force_kN": self.tensile_force_kN,
            "elongation_mm": self.elongation_mm,
            "is_fractured": self.is_fractured,
            "regime": self.deformation_regime_name,
        }
        self.stamp_requested.emit(data)
