"""
AirTeach AI — Interactive Biochemistry & Pharmacology: Michaelis-Menten Enzyme Kinetics,
Lineweaver-Burk Double Reciprocal Plot & Enzyme Inhibition Simulator.
Simulates:
  - Michaelis-Menten hyperbolic saturation equation: v_0 = (V_max * [S]) / (K_m + [S])
  - Lineweaver-Burk double reciprocal transformation: 1/v_0 = (K_m / V_max) * (1/[S]) + 1/V_max
  - Pharmacological enzyme inhibition mechanisms:
      * Competitive: K_m_app = K_m * (1 + [I]/K_i), V_max unchanged
      * Non-Competitive (Allosteric): V_max_app = V_max / (1 + [I]/K_i), K_m unchanged
      * Uncompetitive: Both K_m and V_max reduced by factor (1 + [I]/K_i')
  - Active site fractional saturation theta = [S] / (K_m + [S])
  - Catalytic turnover number k_cat and specificity efficiency k_cat / K_m
"""

from __future__ import annotations
import math
from dataclasses import dataclass
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
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class EnzymePreset:
    """Represents a curriculum scenario for Enzyme Kinetics & Pharmacology."""
    code: str
    title_vi: str
    badge_label: str
    enzyme_name: str
    substrate_name: str
    km_mM: float             # Michaelis constant in mM
    vmax_uM_per_s: float     # Maximum velocity in uM/s
    kcat_per_s: float        # Turnover number in s^-1
    ki_uM: float             # Inhibitor dissociation constant in uM
    default_inhibitor_uM: float
    inhibition_type: str     # "NONE", "COMPETITIVE", "NON_COMPETITIVE", "UNCOMPETITIVE"
    description: str


ENZYME_PRESETS: dict[str, EnzymePreset] = {
    "CHYMOTRYPSIN_PEPTIDE_CLEAVAGE": EnzymePreset(
        code="CHYMOTRYPSIN_PEPTIDE_CLEAVAGE",
        title_vi="Chymotrypsin — Động học Chuẩn Cắt Liên kết Peptide (Không Ức chế)",
        badge_label="CHYMOTRYPSIN",
        enzyme_name="Chymotrypsin",
        substrate_name="ATEE (Este Tyrosine)",
        km_mM=1.5,
        vmax_uM_per_s=100.0,
        kcat_per_s=120.0,
        ki_uM=5.0,
        default_inhibitor_uM=0.0,
        inhibition_type="NONE",
        description="Enzyme tiêu hóa Chymotrypsin (Serine protease) kinh điển. Đồ thị động học Michaelis-Menten dạng hyperbol bão hòa: tại nồng độ cơ chất [S] = Km = 1.5 mM, vận tốc phản ứng đạt chính xác 50% Vmax (v0 = 50 uM/s).",
    ),
    "COMPETITIVE_INHIBITION_STATIN": EnzymePreset(
        code="COMPETITIVE_INHIBITION_STATIN",
        title_vi="Ức chế Cạnh tranh (Competitive) — Thuốc Statin & HMG-CoA Reductase",
        badge_label="COMPETITIVE STATIN",
        enzyme_name="HMG-CoA Reductase",
        substrate_name="HMG-CoA",
        km_mM=2.0,
        vmax_uM_per_s=100.0,
        kcat_per_s=80.0,
        ki_uM=4.0,
        default_inhibitor_uM=8.0,  # [I] = 2*Ki => alpha = 3 => Km_app = 6.0 mM
        inhibition_type="COMPETITIVE",
        description="Cơ chế ức chế cạnh tranh kinh điển của thuốc hạ mỡ máu Statin (Atorvastatin/Lipitor). Chất ức chế cạnh tranh trực tiếp vị trí gắn cơ chất tại trung tâm hoạt động: Km biểu kiến tăng vọt (từ 2.0 lên 6.0 mM) trong khi Vmax không đổi vì có thể bị giải phóng khi [S] cực lớn.",
    ),
    "NONCOMPETITIVE_ALLOSTERIC_INHIBITION": EnzymePreset(
        code="NONCOMPETITIVE_ALLOSTERIC_INHIBITION",
        title_vi="Ức chế Phi Cạnh tranh (Non-Competitive) — Điều hòa Dị lập thể Allosteric",
        badge_label="ALLOSTERIC PFK",
        enzyme_name="Phosphofructokinase-1 (PFK-1)",
        substrate_name="Fructose-6-phosphate",
        km_mM=1.8,
        vmax_uM_per_s=100.0,
        kcat_per_s=90.0,
        ki_uM=5.0,
        default_inhibitor_uM=5.0,  # alpha = 2 => Vmax_app = 50 uM/s
        inhibition_type="NON_COMPETITIVE",
        description="Ức chế phi cạnh tranh (dị lập thể) điều hòa đường phân Glycolysis. Chất ức chế gắn vào vị trí allosteric khác với trung tâm hoạt động làm thay đổi cấu hình không gian: Vmax giảm một nửa (từ 100 xuống 50 uM/s) trong khi Km giữ nguyên vì ái lực gắn cơ chất không bị cản trở.",
    ),
    "UNCOMPETITIVE_INHIBITION_MODEL": EnzymePreset(
        code="UNCOMPETITIVE_INHIBITION_MODEL",
        title_vi="Ức chế Bất Cạnh tranh (Uncompetitive) — Gắn vào Phức hệ [ES]",
        badge_label="UNCOMPETITIVE",
        enzyme_name="Inositol Monophosphatase",
        substrate_name="Inositol phosphate",
        km_mM=3.0,
        vmax_uM_per_s=100.0,
        kcat_per_s=75.0,
        ki_uM=6.0,
        default_inhibitor_uM=6.0,  # alpha = 2 => both Km and Vmax halved
        inhibition_type="UNCOMPETITIVE",
        description="Ức chế bất cạnh tranh: Chất ức chế chỉ gắn vào phức hệ enzyme-cơ chất [ES] đã hình thành mà không gắn vào enzyme tự do. Cả Km và Vmax đều giảm theo cùng tỷ số, tạo thành các đường thẳng song song tuyệt đối trên đồ thị Lineweaver-Burk.",
    ),
    "CATALASE_ULTRA_FAST_DIFFUSION_LIMIT": EnzymePreset(
        code="CATALASE_ULTRA_FAST_DIFFUSION_LIMIT",
        title_vi="Catalase — Hoàn hảo về Xúc tác & Giới hạn Khuếch tán Phân tử",
        badge_label="CATALASE DIFFUSION",
        enzyme_name="Catalase",
        substrate_name="H₂O₂ (Hydrogen Peroxide)",
        km_mM=25.0,
        vmax_uM_per_s=120.0,
        kcat_per_s=400000.0,
        ki_uM=10.0,
        default_inhibitor_uM=0.0,
        inhibition_type="NONE",
        description="Catalase phân hủy H2O2 thành H2O và O2 với hằng số xúc tác phi thường k_cat ~ 4×10⁵ s⁻¹. Hiệu suất xúc tác k_cat / Km tiệm cận giới hạn khuếch tán phân tử ~ 10⁸ - 10⁹ M⁻¹s⁻¹ (mỗi phân tử enzyme xử lý hàng triệu phân tử cơ chất mỗi giây).",
    ),
}


class InteractiveEnzymeKineticsItem(QGraphicsObject):
    """
    Interactive Biochemistry & Pharmacology: Enzyme Kinetics & Inhibition Simulator.
    Simulates:
      - Michaelis-Menten hyperbolic curve v_0([S])
      - Lineweaver-Burk double reciprocal linear plot 1/v_0 vs 1/[S]
      - Inhibition modes: Competitive, Non-Competitive, Uncompetitive, None
      - Active site saturation theta = [S] / (K_m + [S])
      - Animated enzyme-substrate binding pocket with inhibitor competition
    """

    preset_changed = Signal(str)
    kinetics_computed = Signal(float, float, float)  # v0, km_app, vmax_app
    stamp_requested = Signal(dict)

    CARD_W = 960.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "COMPETITIVE_INHIBITION_STATIN"
        self._preset: EnzymePreset = ENZYME_PRESETS[self._active_preset_code]

        self._km_mM: float = self._preset.km_mM
        self._vmax_uM_per_s: float = self._preset.vmax_uM_per_s
        self._kcat_per_s: float = self._preset.kcat_per_s
        self._ki_uM: float = self._preset.ki_uM
        self._inhibitor_uM: float = self._preset.default_inhibitor_uM
        self._inhibition_type: str = self._preset.inhibition_type

        # Substrate concentration [S] in mM
        self._substrate_mM: float = 3.0

        # Animation states
        self._pocket_phase: float = 0.0
        self._is_running: bool = True

        # UI & Touchless buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 30 fps simulation timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physical & Biochemical Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> EnzymePreset:
        return self._preset

    @property
    def alpha_inhibitor_factor(self) -> float:
        """alpha = 1 + [I] / K_i."""
        if self._inhibition_type == "NONE" or self._ki_uM <= 1e-6:
            return 1.0
        return 1.0 + (self._inhibitor_uM / self._ki_uM)

    @property
    def apparent_km_mM(self) -> float:
        """Calculates apparent Michaelis constant K_m^app based on inhibition mode."""
        alpha = self.alpha_inhibitor_factor
        if self._inhibition_type == "COMPETITIVE":
            return self._km_mM * alpha
        elif self._inhibition_type == "UNCOMPETITIVE":
            return self._km_mM / alpha
        else:  # NON_COMPETITIVE or NONE
            return self._km_mM

    @property
    def apparent_vmax_uM_per_s(self) -> float:
        """Calculates apparent maximum velocity V_max^app based on inhibition mode."""
        alpha = self.alpha_inhibitor_factor
        if self._inhibition_type in ("NON_COMPETITIVE", "UNCOMPETITIVE"):
            return self._vmax_uM_per_s / alpha
        else:  # COMPETITIVE or NONE
            return self._vmax_uM_per_s

    @property
    def current_velocity_v0(self) -> float:
        """Calculates initial velocity v_0 = (V_max_app * [S]) / (K_m_app + [S])."""
        s = max(0.0, self._substrate_mM)
        km_app = self.apparent_km_mM
        vmax_app = self.apparent_vmax_uM_per_s
        denom = km_app + s
        if denom <= 1e-6:
            return 0.0
        return (vmax_app * s) / denom

    @property
    def uninhibited_velocity_v0(self) -> float:
        """Calculates baseline uninhibited velocity v_0."""
        s = max(0.0, self._substrate_mM)
        denom = self._km_mM + s
        if denom <= 1e-6:
            return 0.0
        return (self._vmax_uM_per_s * s) / denom

    @property
    def active_site_saturation_fraction(self) -> float:
        """Fractional saturation theta = [S] / (K_m_app + [S])."""
        s = max(0.0, self._substrate_mM)
        km_app = self.apparent_km_mM
        denom = km_app + s
        if denom <= 1e-6:
            return 0.0
        return s / denom

    @property
    def fractional_saturation_theta(self) -> float:
        """Alias for active_site_saturation_fraction."""
        return self.active_site_saturation_fraction

    @property
    def catalytic_efficiency_M_per_s(self) -> float:
        """Catalytic efficiency k_cat / K_m in M^-1 s^-1."""
        km_M = self._km_mM / 1000.0
        if km_M <= 1e-9:
            return 0.0
        return self._kcat_per_s / km_M

    # -------------------------------------------------------------------------
    # Simulation Control & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in ENZYME_PRESETS:
            return
        self._active_preset_code = code
        self._preset = ENZYME_PRESETS[code]
        self._km_mM = self._preset.km_mM
        self._vmax_uM_per_s = self._preset.vmax_uM_per_s
        self._kcat_per_s = self._preset.kcat_per_s
        self._ki_uM = self._preset.ki_uM
        self._inhibitor_uM = self._preset.default_inhibitor_uM
        self._inhibition_type = self._preset.inhibition_type
        self._substrate_mM = 3.0

        self._layout_buttons()
        self.preset_changed.emit(code)
        self.kinetics_computed.emit(
            self.current_velocity_v0,
            self.apparent_km_mM,
            self.apparent_vmax_uM_per_s,
        )
        self.update()

    def set_substrate_concentration(self, s_mM: float) -> None:
        """Sets substrate concentration [S] in mM [0.1, 25.0]."""
        self._substrate_mM = max(0.1, min(25.0, s_mM))
        self.kinetics_computed.emit(
            self.current_velocity_v0,
            self.apparent_km_mM,
            self.apparent_vmax_uM_per_s,
        )
        self.update()

    def set_inhibitor_concentration(self, i_uM: float) -> None:
        """Sets inhibitor concentration [I] in uM [0.0, 50.0]."""
        self._inhibitor_uM = max(0.0, min(50.0, i_uM))
        self.kinetics_computed.emit(
            self.current_velocity_v0,
            self.apparent_km_mM,
            self.apparent_vmax_uM_per_s,
        )
        self.update()

    def toggle_inhibition_type(self) -> None:
        """Cycles through inhibition modes: NONE -> COMPETITIVE -> NON_COMPETITIVE -> UNCOMPETITIVE."""
        modes = ["NONE", "COMPETITIVE", "NON_COMPETITIVE", "UNCOMPETITIVE"]
        idx = modes.index(self._inhibition_type) if self._inhibition_type in modes else 0
        self._inhibition_type = modes[(idx + 1) % len(modes)]
        if self._inhibition_type != "NONE" and self._inhibitor_uM <= 0.1:
            self._inhibitor_uM = self._ki_uM
        self.kinetics_computed.emit(
            self.current_velocity_v0,
            self.apparent_km_mM,
            self.apparent_vmax_uM_per_s,
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets substrate, inhibitor, and mode to preset defaults."""
        self._substrate_mM = 3.0
        self._inhibitor_uM = self._preset.default_inhibitor_uM
        self._inhibition_type = self._preset.inhibition_type
        self.kinetics_computed.emit(
            self.current_velocity_v0,
            self.apparent_km_mM,
            self.apparent_vmax_uM_per_s,
        )
        self.update()

    def _on_sim_tick(self) -> None:
        """Animation tick for enzyme active site pocket pulsation."""
        if not self._is_running:
            return
        self._pocket_phase += 0.12
        if self._pocket_phase > 2.0 * math.pi:
            self._pocket_phase -= 2.0 * math.pi
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Touchless Hit-Testing
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Lays out interactive touchless action buttons."""
        self._buttons.clear()
        by = 595.0
        bh = 38.0

        self._buttons["btn_preset"] = QRectF(20.0, by, 160.0, bh)
        self._buttons["btn_s_down"] = QRectF(188.0, by, 68.0, bh)
        self._buttons["btn_s_up"] = QRectF(262.0, by, 68.0, bh)
        self._buttons["btn_i_down"] = QRectF(338.0, by, 66.0, bh)
        self._buttons["btn_i_up"] = QRectF(410.0, by, 66.0, bh)
        self._buttons["btn_mode"] = QRectF(484.0, by, 168.0, bh)
        self._buttons["btn_reset"] = QRectF(660.0, by, 96.0, bh)
        self._buttons["btn_stamp"] = QRectF(856.0, by, 84.0, bh)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Painting Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_michaelis_menten_panel(painter)
        self._paint_lineweaver_burk_panel(painter)
        self._paint_active_site_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Renders main dark acrylic glass card background."""
        grad = QLinearGradient(0.0, 0.0, self.CARD_W, self.CARD_H)
        grad.setColorAt(0.0, QColor(10, 15, 29, 250))
        grad.setColorAt(1.0, QColor(4, 7, 16, 252))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(self.boundingRect(), 16.0, 16.0)

    def _paint_header(self, painter: QPainter) -> None:
        """Renders card header, badges, and curriculum title."""
        hx = 20.0
        hy = 20.0

        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(hx, hy + 18.0, "ĐỘNG HỌC ENZYME MICHAELIS - MENTEN & ĐỒ THỊ LINEWEAVER - BURK")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(hx, hy + 35.0, self._preset.title_vi)

        # Badges on right side
        bx = self.CARD_W - 20.0
        badges = [
            (f"v₀ = {self.current_velocity_v0:.1f} μM/s", QColor(16, 185, 129)),
            (f"Km_app = {self.apparent_km_mM:.2f} mM", QColor(14, 165, 233)),
            (f"Vmax_app = {self.apparent_vmax_uM_per_s:.1f} μM/s", QColor(168, 85, 247)),
            (self._inhibition_type, QColor(245, 158, 11)),
        ]
        for text, color in badges:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            text_w = painter.fontMetrics().horizontalAdvance(text) + 16.0
            bx -= text_w
            badge_rect = QRectF(bx, hy + 8.0, text_w, 22.0)
            painter.setBrush(QBrush(color.darker(240)))
            painter.setPen(QPen(color, 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QPen(color.lighter(130)))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
            bx -= 8.0

    def _paint_michaelis_menten_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Left Panel: Hyperbolic Michaelis-Menten Saturation Curve v_0 vs [S].
        Visualizes V_max asymptote, K_m at V_max/2, and inhibition curve shift.
        """
        mx = 20.0
        my = 70.0
        mw = 460.0
        mh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(mx, my, mw, mh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 18.0, "ĐƯỜNG CONG BÃO HÒA HYPERBOL MICHAELIS - MENTEN:")

        # Plot frame
        gx = mx + 40.0
        gy = my + 35.0
        gw = mw - 55.0
        gh = mh - 65.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(gx, gy, gw, gh))

        # Axis limits: [S] from 0 to 20 mM, v0 from 0 to 120 uM/s
        max_s = 20.0
        max_v = 120.0

        def map_x(s_val: float) -> float:
            return gx + (s_val / max_s) * gw

        def map_y(v_val: float) -> float:
            return gy + gh - (v_val / max_v) * gh

        # Grid lines
        painter.setFont(QFont("Segoe UI", 6))
        for s_tick in (4, 8, 12, 16, 20):
            cx = map_x(float(s_tick))
            painter.setPen(QPen(QColor(30, 41, 59, 140), 0.8, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(cx, gy), QPointF(cx, gy + gh))
            painter.setPen(QPen(QColor(100, 116, 139)))
            painter.drawText(cx - 8.0, gy + gh + 14.0, f"{s_tick}mM")

        for v_tick in (30, 60, 90, 120):
            cy = map_y(float(v_tick))
            painter.setPen(QPen(QColor(30, 41, 59, 140), 0.8, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(gx, cy), QPointF(gx + gw, cy))
            painter.setPen(QPen(QColor(100, 116, 139)))
            painter.drawText(mx + 8.0, cy + 4.0, str(v_tick))

        # X & Y Axis Labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(gx + gw / 2.0 - 45.0, gy + gh + 26.0, "Nồng độ Cơ chất [S] (mM)")
        painter.drawText(mx + 6.0, gy + 12.0, "v₀ (μM/s)")

        # 1. Uninhibited Baseline Curve (Emerald / Cyan dashed line)
        if self._inhibition_type != "NONE":
            base_path = QPainterPath()
            first = True
            for i in range(51):
                s_eval = (i / 50.0) * max_s
                v_eval = (self._vmax_uM_per_s * s_eval) / (self._km_mM + s_eval)
                px = map_x(s_eval)
                py = map_y(v_eval)
                if first:
                    base_path.moveTo(px, py)
                    first = False
                else:
                    base_path.lineTo(px, py)
            painter.setPen(QPen(QColor(16, 185, 129, 160), 1.6, Qt.PenStyle.DashLine))
            painter.drawPath(base_path)

        # 2. Active Curve (with inhibition if applicable)
        act_path = QPainterPath()
        first = True
        km_app = self.apparent_km_mM
        vmax_app = self.apparent_vmax_uM_per_s

        for i in range(51):
            s_eval = (i / 50.0) * max_s
            v_eval = (vmax_app * s_eval) / (km_app + s_eval)
            px = map_x(s_eval)
            py = map_y(v_eval)
            if first:
                act_path.moveTo(px, py)
                first = False
            else:
                act_path.lineTo(px, py)

        # Filled area under active curve
        fill_path = QPainterPath(act_path)
        fill_path.lineTo(gx + gw, gy + gh)
        fill_path.lineTo(gx, gy + gh)
        fill_path.closeSubpath()

        curve_grad = QLinearGradient(gx, gy, gx, gy + gh)
        curve_grad.setColorAt(0.0, QColor(56, 189, 248, 110))
        curve_grad.setColorAt(1.0, QColor(56, 189, 248, 10))
        painter.setBrush(QBrush(curve_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(act_path)

        # Asymptote V_max^app line
        vmax_y = map_y(vmax_app)
        painter.setPen(QPen(QColor(168, 85, 247, 180), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(gx, vmax_y), QPointF(gx + gw, vmax_y))

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(192, 132, 252)))
        painter.drawText(gx + gw - 65.0, vmax_y - 4.0, f"Vmax = {vmax_app:.0f}")

        # Half-maximal velocity V_max/2 and K_m marker
        v_half = vmax_app / 2.0
        v_half_y = map_y(v_half)
        km_x = map_x(km_app)

        if km_app <= max_s:
            painter.setPen(QPen(QColor(234, 179, 8, 160), 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(gx, v_half_y), QPointF(km_x, v_half_y))
            painter.drawLine(QPointF(km_x, v_half_y), QPointF(km_x, gy + gh))

            painter.setBrush(QBrush(QColor(234, 179, 8)))
            painter.setPen(QPen(QColor(0, 0, 0), 0.8))
            painter.drawEllipse(QPointF(km_x, v_half_y), 3.0, 3.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(234, 179, 8)))
            painter.drawText(km_x - 14.0, gy + gh - 6.0, f"Km={km_app:.1f}")

        # Live Operating Point: ([S], current_v0)
        s_cur = self._substrate_mM
        v_cur = self.current_velocity_v0
        cur_x = map_x(s_cur)
        cur_y = map_y(v_cur)

        # Coordinate dropped lines
        painter.setPen(QPen(QColor(245, 158, 11, 200), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cur_x, gy + gh), QPointF(cur_x, cur_y))
        painter.drawLine(QPointF(gx, cur_y), QPointF(cur_x, cur_y))

        # Glowing dot
        rad_op = QRadialGradient(cur_x, cur_y, 9.0)
        rad_op.setColorAt(0.0, QColor(245, 158, 11, 255))
        rad_op.setColorAt(0.6, QColor(245, 158, 11, 150))
        rad_op.setColorAt(1.0, QColor(245, 158, 11, 0))
        painter.setBrush(QBrush(rad_op))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cur_x, cur_y), 9.0, 9.0)

        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.drawEllipse(QPointF(cur_x, cur_y), 4.0, 4.0)

    def _paint_lineweaver_burk_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Right Panel: Lineweaver-Burk Double Reciprocal Linear Plot 1/v_0 vs 1/[S].
        Highlights intercepts: y-int = 1/Vmax, x-int = -1/Km, Slope = Km/Vmax.
        """
        lx = 500.0
        ly = 70.0
        lw = 440.0
        lh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(lx, ly, lw, lh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(lx + 12.0, ly + 18.0, "ĐỒ THỊ NGHỊCH ĐẢO KÉP LINEWEAVER - BURK (1/v₀ vs 1/[S]):")

        # Plot frame (Origin offset to right to show negative x-axis intercept -1/Km)
        gx = lx + 35.0
        gy = ly + 35.0
        gw = lw - 50.0
        gh = lh - 65.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(gx, gy, gw, gh))

        # Range of 1/[S]: from -1.0 to +2.5 mM^-1
        # Range of 1/v0: from 0.0 to 0.06 (uM/s)^-1
        min_inv_s = -1.2
        max_inv_s = 2.4
        span_inv_s = max_inv_s - min_inv_s
        max_inv_v = 0.06

        def map_lx(inv_s: float) -> float:
            return gx + ((inv_s - min_inv_s) / span_inv_s) * gw

        def map_ly(inv_v: float) -> float:
            return gy + gh - (inv_v / max_inv_v) * gh

        # Vertical Y-axis at 1/[S] = 0
        origin_x = map_lx(0.0)
        painter.setPen(QPen(QColor(100, 116, 139), 1.4))
        painter.drawLine(QPointF(origin_x, gy), QPointF(origin_x, gy + gh))

        # Horizontal X-axis at 1/v0 = 0 (bottom of plot)
        painter.drawLine(QPointF(gx, gy + gh), QPointF(gx + gw, gy + gh))

        # X & Y Axis Labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(origin_x + 12.0, gy + 14.0, "1/v₀ (μM/s)⁻¹")
        painter.drawText(gx + gw - 65.0, gy + gh - 6.0, "1/[S] (mM⁻¹)")

        # 1. Uninhibited Lineweaver-Burk Line (Dashed Green)
        if self._inhibition_type != "NONE":
            slope_base = self._km_mM / self._vmax_uM_per_s
            y_int_base = 1.0 / self._vmax_uM_per_s
            x_int_base = -1.0 / self._km_mM

            x1_b = map_lx(x_int_base)
            y1_b = map_ly(0.0)
            x2_b = map_lx(2.0)
            y2_b = map_ly(slope_base * 2.0 + y_int_base)

            painter.setPen(QPen(QColor(16, 185, 129, 160), 1.6, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x1_b, y1_b), QPointF(x2_b, y2_b))

        # 2. Active Lineweaver-Burk Line (Cyan / Amber)
        km_app = self.apparent_km_mM
        vmax_app = self.apparent_vmax_uM_per_s
        slope_act = km_app / vmax_app
        y_int_act = 1.0 / vmax_app
        x_int_act = -1.0 / km_app

        x1_a = map_lx(max(min_inv_s, x_int_act))
        y1_a = map_ly(0.0)
        x2_a = map_lx(2.2)
        y2_a = map_ly(slope_act * 2.2 + y_int_act)

        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawLine(QPointF(x1_a, y1_a), QPointF(x2_a, y2_a))

        # Mark Intercepts:
        # y-intercept (1/Vmax)
        y_int_px = origin_x
        y_int_py = map_ly(y_int_act)
        painter.setBrush(QBrush(QColor(168, 85, 247)))
        painter.setPen(QPen(QColor(0, 0, 0), 0.8))
        painter.drawEllipse(QPointF(y_int_px, y_int_py), 3.5, 3.5)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(192, 132, 252)))
        painter.drawText(y_int_px + 6.0, y_int_py + 4.0, f"1/Vmax = {y_int_act:.3f}")

        # x-intercept (-1/Km)
        if x_int_act >= min_inv_s:
            x_int_px = map_lx(x_int_act)
            x_int_py = map_ly(0.0)
            painter.setBrush(QBrush(QColor(234, 179, 8)))
            painter.setPen(QPen(QColor(0, 0, 0), 0.8))
            painter.drawEllipse(QPointF(x_int_px, x_int_py), 3.5, 3.5)

            painter.setPen(QPen(QColor(234, 179, 8)))
            painter.drawText(x_int_px - 22.0, x_int_py - 6.0, f"-1/Km={x_int_act:.2f}")

        # Current Reciprocal Operating Point: (1/[S], 1/v0)
        inv_s_cur = 1.0 / self._substrate_mM
        inv_v_cur = 1.0 / max(0.01, self.current_velocity_v0)
        cur_lx = map_lx(inv_s_cur)
        cur_ly = map_ly(inv_v_cur)

        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.drawEllipse(QPointF(cur_lx, cur_ly), 4.0, 4.0)

        # Callout annotation of Slope
        slope_box = QRectF(lx + 45.0, ly + 45.0, 150.0, 36.0)
        painter.setBrush(QBrush(QColor(9, 14, 25, 220)))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.0))
        painter.drawRoundedRect(slope_box, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(226, 232, 240)))
        painter.drawText(slope_box.x() + 6.0, slope_box.y() + 14.0, f"Hệ số góc: Km/Vmax = {slope_act:.4f}")
        painter.drawText(slope_box.x() + 6.0, slope_box.y() + 28.0, f"1/Km = {1.0/km_app:.3f} | 1/Vmax = {y_int_act:.3f}")

    def _paint_active_site_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Left Panel: 2D/3D Enzyme Pocket, Substrate/Inhibitor Competition & Saturation Meter.
        """
        ax = 20.0
        ay = 360.0
        aw = 460.0
        ah = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(ax, ay, aw, ah), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ax + 12.0, ay + 18.0, "TRUNG TÂM HOẠT ĐỘNG (ACTIVE SITE) & ĐỘ BÃO HÒA:")

        # Enzyme Macromolecule Geometry (Center of left sub-panel)
        cx = ax + 110.0
        cy = ay + 115.0

        # Pulsating Enzyme Globular Body
        r_enz = 55.0 + 1.5 * math.sin(self._pocket_phase)
        enz_grad = QRadialGradient(cx - 15.0, cy - 15.0, r_enz)
        enz_grad.setColorAt(0.0, QColor("#1e293b"))
        enz_grad.setColorAt(0.7, QColor("#0f172a"))
        enz_grad.setColorAt(1.0, QColor("#020617"))

        painter.setBrush(QBrush(enz_grad))
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawEllipse(QPointF(cx, cy), r_enz, r_enz)

        # Binding Pocket Cutout (Active Site)
        pocket_w = 34.0
        pocket_h = 24.0
        pocket_rect = QRectF(cx - pocket_w / 2.0, cy - r_enz * 0.4, pocket_w, pocket_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#64748b"), 1.2, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(pocket_rect, 6.0, 6.0)

        # Draw Substrate (Green hexagon) or Inhibitor (Red pentagon) inside/approaching pocket
        is_inhibited = self._inhibition_type != "NONE" and self._inhibitor_uM > 0.1
        if is_inhibited and self._inhibition_type == "COMPETITIVE":
            # Red Inhibitor occupies pocket
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawRoundedRect(QRectF(cx - 12.0, cy - r_enz * 0.35, 24.0, 16.0), 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(cx - 12.0, cy - r_enz * 0.35, 24.0, 16.0), Qt.AlignmentFlag.AlignCenter, "I")
        else:
            # Green Substrate occupies pocket
            painter.setBrush(QBrush(QColor("#22c55e")))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawRoundedRect(QRectF(cx - 12.0, cy - r_enz * 0.35, 24.0, 16.0), 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(QRectF(cx - 12.0, cy - r_enz * 0.35, 24.0, 16.0), Qt.AlignmentFlag.AlignCenter, "S")

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(cx - 24.0, cy + 25.0, self._preset.enzyme_name)

        # Right sub-panel: Fractional Saturation Bar & Turnover Metrics
        rx = ax + 230.0
        ry = ay + 38.0

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(rx, ry, "TỶ LỆ BÃO HÒA TRUNG TÂM HOẠT ĐỘNG (θ):")

        # Saturation Progress Bar
        theta = self.active_site_saturation_fraction
        bar_w = 200.0
        bar_h = 20.0
        bar_x = rx
        bar_y = ry + 12.0

        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))

        fill_w = theta * bar_w
        painter.setBrush(QBrush(QColor(16, 185, 129)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(bar_x, bar_y, fill_w, bar_h))

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(QRectF(bar_x, bar_y, bar_w, bar_h), Qt.AlignmentFlag.AlignCenter, f"θ = {theta * 100:.1f}%")

        # Turnover info lines
        lines = [
            f"• Số vòng quay xúc tác: k_cat = {self._kcat_per_s:,.0f} s⁻¹",
            f"• Hiệu suất xúc tác: k_cat/Km = {self.catalytic_efficiency_M_per_s:.2e} M⁻¹s⁻¹",
            f"• Nồng độ Cơ chất [S]: {self._substrate_mM:.2f} mM",
            f"• Nồng độ Chất ức chế [I]: {self._inhibitor_uM:.2f} μM (Ki = {self._ki_uM:.2f} μM)",
            f"• Cơ chế: {self._inhibition_type}",
        ]

        row_y = bar_y + 36.0
        painter.setFont(QFont("Segoe UI", 6))
        for l in lines:
            painter.setPen(QPen(QColor(226, 232, 240) if "k_cat" in l else QColor(148, 163, 184)))
            painter.drawText(rx, row_y, l)
            row_y += 18.0

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Right Panel: Rigorous Pharmacological & Kinetic Equations.
        """
        mx = 500.0
        my = 360.0
        mw = 440.0
        mh = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(mx, my, mw, mh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 18.0, "CÔNG THỨC ĐỘNG HỌC & DƯỢC LÝ HỌC ENZYME:")

        metrics = [
            ("Phương trình Michaelis-Menten:", f"v₀ = (Vmax·[S]) / (Km + [S]) = {self.current_velocity_v0:.2f} μM/s"),
            ("Hằng số Michaelis Biểu kiến (Km_app):", f"Km_app = {self.apparent_km_mM:.2f} mM (Gốc: {self._km_mM:.2f} mM)"),
            ("Vận tốc Tối đa Biểu kiến (Vmax_app):", f"Vmax_app = {self.apparent_vmax_uM_per_s:.1f} μM/s (Gốc: {self._vmax_uM_per_s:.1f})"),
            ("Hệ số Ức chế Dược lý (alpha = 1 + [I]/Ki):", f"α = {self.alpha_inhibitor_factor:.2f} ([I] = {self._inhibitor_uM:.1f} μM)"),
            ("Độ Bão hòa Vị trí Hoạt động (Saturation θ):", f"θ = [S] / (Km + [S]) = {self.active_site_saturation_fraction * 100:.1f}%"),
            ("Hiệu suất Xúc tác Chuyên biệt (Specificity):", f"k_cat/Km = {self.catalytic_efficiency_M_per_s:.2e} M⁻¹s⁻¹"),
            ("Kiểu Ức chế Dược lý (Inhibition Mode):", f"{self._inhibition_type}"),
        ]

        row_y = my + 40.0
        line_h = 24.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 12.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            val_col = QColor(16, 185, 129) if "v₀" in name or "Km_app" in name else QColor(226, 232, 240)
            painter.setPen(QPen(val_col))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture control buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_s_down", "[S] -", QColor(51, 65, 85)),
            ("btn_s_up", "[S] +", QColor(51, 65, 85)),
            ("btn_i_down", "[I] -", QColor(51, 65, 85)),
            ("btn_i_up", "[I] +", QColor(51, 65, 85)),
            ("btn_mode", f"Chế độ: {self._inhibition_type}", QColor(245, 158, 11)),
            ("btn_reset", "Mặc định", QColor(168, 85, 247)),
            ("btn_stamp", "Ghim", QColor(16, 185, 129)),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for key, text, color in buttons_info:
            rect = self._buttons[key]
            hovered = self._hovered_btn == key
            btn_col = color.lighter(125) if hovered else color

            painter.setBrush(QBrush(btn_col))
            painter.setPen(QPen(QColor(255, 255, 255, 160) if hovered else QColor(0, 0, 0, 80), 1.0))
            painter.drawRoundedRect(rect, 6.0, 6.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Touch & Mouse Event Handling
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if self._hovered_btn != old_btn:
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
                    self._on_button_clicked(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_button_clicked(self, key: str) -> None:
        """Dispatches button actions."""
        if key == "btn_preset":
            keys = list(ENZYME_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_s_down":
            self.set_substrate_concentration(self._substrate_mM - 0.5)
        elif key == "btn_s_up":
            self.set_substrate_concentration(self._substrate_mM + 0.5)
        elif key == "btn_i_down":
            self.set_inhibitor_concentration(self._inhibitor_uM - 1.0)
        elif key == "btn_i_up":
            self.set_inhibitor_concentration(self._inhibitor_uM + 1.0)
        elif key == "btn_mode":
            self.toggle_inhibition_type()
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "enzyme_name": self._preset.enzyme_name,
            "substrate_name": self._preset.substrate_name,
            "km_mM": self.apparent_km_mM,
            "vmax_uM_per_s": self.apparent_vmax_uM_per_s,
            "substrate_mM": self._substrate_mM,
            "v0_uM_per_s": self.current_velocity_v0,
            "inhibition_type": self._inhibition_type,
            "alpha": self.alpha_inhibitor_factor,
        }
        self.stamp_requested.emit(data)
