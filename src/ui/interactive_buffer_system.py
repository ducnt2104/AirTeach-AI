"""
AirTeach AI — Interactive Analytical Chemistry: Acid-Base Buffer Systems,
Henderson-Hasselbalch Equation & Van Slyke Buffer Capacity Simulator.
Simulates:
  - Conjugate acid-base pair equilibrium HA + H2O <=> A- + H3O+
  - Henderson-Hasselbalch equation: pH = pKa + log10([A-] / [HA])
  - Van Slyke buffer capacity: beta = 2.303 * ([H+] + [OH-] + (C_total * Ka * [H+]) / (Ka + [H+])^2)
  - Maximum buffer capacity beta_max = 0.575 * C_total at pH = pKa
  - Real-time acid (HCl) / base (NaOH) titration perturbation and resistance
  - Speciation fraction curves alpha_HA(pH) and alpha_A-(pH)
  - Universal pH indicator chromophore liquid beaker and digital pH meter
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
class BufferPreset:
    """Represents a curriculum scenario for Acid-Base Buffer Systems."""
    code: str
    title_vi: str
    badge_label: str
    acid_name: str
    base_name: str
    pKa: float
    default_c_total_M: float
    default_base_acid_ratio: float  # [A-] / [HA]
    description: str

    @property
    def Ka(self) -> float:
        return 10.0 ** (-self.pKa)


BUFFER_PRESETS: dict[str, BufferPreset] = {
    "ACETIC_ACID_ACETATE_BUFFER": BufferPreset(
        code="ACETIC_ACID_ACETATE_BUFFER",
        title_vi="Hệ Đệm Axit Axetic / Natri Axetat (CH₃COOH / CH₃COO⁻) — pKa = 4.76",
        badge_label="ACETATE pKa 4.76",
        acid_name="CH₃COOH",
        base_name="CH₃COO⁻",
        pKa=4.76,
        default_c_total_M=0.20,
        default_base_acid_ratio=1.0,  # Equimolar: pH = pKa
        description="Hệ đệm axit axetic/natri axetat kinh điển trong phòng thí nghiệm hóa học phân tích. Khả năng đệm tối ưu trong khoảng pH từ 3.76 đến 5.76, dung lượng đệm cực đại beta_max đạt đỉnh tại pH = pKa = 4.76.",
    ),
    "BLOOD_CARBONATE_PHYSIOLOGY": BufferPreset(
        code="BLOOD_CARBONATE_PHYSIOLOGY",
        title_vi="Hệ Đệm Bicarbonate Huyết tương Máu Người (H₂CO₃ / HCO₃⁻) — pH Sinh lý 7.40",
        badge_label="BLOOD pH 7.40",
        acid_name="H₂CO₃ (CO₂ hòa tan)",
        base_name="HCO₃⁻",
        pKa=6.10,
        default_c_total_M=0.0252,  # 25.2 mM in physiological blood plasma
        default_base_acid_ratio=20.0,  # [HCO3-] / [H2CO3] = 20:1 => pH = 6.10 + log(20) = 7.40
        description="Hệ đệm Bicarbonate sinh lý điều hòa pH máu người ổn định ở mức 7.35 - 7.45. Dù pH máu (7.40) cao hơn pKa (6.10), tỷ lệ bazơ/axit cao [HCO₃⁻]/[H₂CO₃] = 20:1 mang lại khả năng chống toan (acidosis) cực lớn trước các sản phẩm phụ chuyển hóa axit.",
    ),
    "PHOSPHATE_CELLULAR_BUFFER": BufferPreset(
        code="PHOSPHATE_CELLULAR_BUFFER",
        title_vi="Hệ Đệm Phosphat Nội bào (H₂PO₄⁻ / HPO₄²⁻) — pKa₂ = 7.20",
        badge_label="PHOSPHATE pKa 7.20",
        acid_name="H₂PO₄⁻",
        base_name="HPO₄²⁻",
        pKa=7.20,
        default_c_total_M=0.10,
        default_base_acid_ratio=1.0,  # pH = 7.20
        description="Hệ đệm Dihydrophosphat / Monohydrophosphat nội bào đóng vai trò duy trì pH dịch tế bào và ống thận ở mức trung tính gần 7.20, cực kỳ gần sát với pKa2 = 7.20 giúp đạt hiệu suất đệm tối đa.",
    ),
    "AMMONIA_AMMONIUM_ALKALINE": BufferPreset(
        code="AMMONIA_AMMONIUM_ALKALINE",
        title_vi="Hệ Đệm Bazơ Amoniac / Amoni (NH₄⁺ / NH₃) — pKa = 9.25",
        badge_label="AMMONIA pKa 9.25",
        acid_name="NH₄⁺",
        base_name="NH₃",
        pKa=9.25,
        default_c_total_M=0.20,
        default_base_acid_ratio=1.0,  # pH = 9.25
        description="Hệ đệm kiềm Amoni / Amoniac (pKa = 9.25) dùng phổ biến trong chuẩn độ tạo phức Chelat EDTA xác định độ cứng canxi/magie trong nước và phân tích cation nhóm III.",
    ),
    "UNBUFFERED_WATER_COMPARISON": BufferPreset(
        code="UNBUFFERED_WATER_COMPARISON",
        title_vi="Nước Cất Không Đệm (Unbuffered Pure Water) — So sánh Kháng Axit/Bazơ",
        badge_label="PURE WATER",
        acid_name="H₂O (H⁺)",
        base_name="OH⁻",
        pKa=7.00,
        default_c_total_M=0.000001,  # Trace buffer capacity
        default_base_acid_ratio=1.0,  # pH = 7.00
        description="Minh họa trực quan sự thiếu vắng dung lượng đệm của nước cất tinh khiết (beta chỉ khoảng 4.6×10⁻⁷ M/pH). Khi thêm chỉ một lượng nhỏ axit mạnh (HCl 0.1 M), pH giảm đột ngột từ 7.00 xuống ~4.00, trong khi hệ đệm thực sự chống lại sự biến đổi này.",
    ),
}


class InteractiveBufferSystemItem(QGraphicsObject):
    """
    Interactive Analytical Chemistry: Acid-Base Buffer Systems & Van Slyke Capacity Simulator.
    Simulates:
      - Henderson-Hasselbalch equation: pH = pKa + log([A-] / [HA])
      - Van Slyke buffer capacity: beta = 2.303 * ([H+] + [OH-] + (C_total * Ka * [H+]) / (Ka + [H+])^2)
      - Dynamic addition of strong acid (HCl) and strong base (NaOH)
      - Chemical speciation fractions alpha_HA and alpha_A-
      - Chromophore color transition based on Universal Indicator pH scale
    """

    preset_changed = Signal(str)
    buffer_computed = Signal(float, float, float)  # pH, beta, ratio
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

        # Buffer State
        self._active_preset_code: str = "ACETIC_ACID_ACETATE_BUFFER"
        self._preset: BufferPreset = BUFFER_PRESETS[self._active_preset_code]

        self._c_total_M: float = self._preset.default_c_total_M
        self._base_acid_ratio: float = self._preset.default_base_acid_ratio

        # Added perturbing strong acid/base amounts in millimoles per liter (mM)
        self._added_acid_mM: float = 0.0
        self._added_base_mM: float = 0.0

        # Animation state for beaker fluid bubbles & wave
        self._wave_phase: float = 0.0
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
    # Physical & Analytical Chemical Formulations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> BufferPreset:
        return self._preset

    @property
    def pKa(self) -> float:
        return self._preset.pKa

    @property
    def Ka(self) -> float:
        return self._preset.Ka

    @property
    def effective_ratio(self) -> float:
        """
        Calculates effective [A-] / [HA] ratio after accounting for added strong acid/base.
        Initial [HA] = C_total / (1 + ratio), Initial [A-] = C_total * ratio / (1 + ratio).
        Adding base consumes HA and produces A-: [HA] -= delta_b, [A-] += delta_b.
        Adding acid consumes A- and produces HA: [A-] -= delta_a, [HA] += delta_a.
        """
        r0 = max(0.001, self._base_acid_ratio)
        c_tot = max(1e-6, self._c_total_M)

        ha0 = c_tot / (1.0 + r0)
        a0 = c_tot * r0 / (1.0 + r0)

        # Net perturbation: delta_b in M
        delta_b = (self._added_base_mM - self._added_acid_mM) / 1000.0

        ha_eff = max(1e-6, ha0 - delta_b)
        a_eff = max(1e-6, a0 + delta_b)

        return a_eff / ha_eff

    @property
    def current_pH(self) -> float:
        """
        Henderson-Hasselbalch: pH = pKa + log10([A-] / [HA]).
        Bounded between 0.5 and 13.5.
        """
        if self._active_preset_code == "UNBUFFERED_WATER_COMPARISON":
            # For pure water, direct strong acid/base calculation
            net_mM = self._added_base_mM - self._added_acid_mM
            if abs(net_mM) < 1e-4:
                return 7.00
            elif net_mM > 0:
                oh = net_mM / 1000.0
                return min(13.5, 14.0 + math.log10(max(1e-7, oh)))
            else:
                h = abs(net_mM) / 1000.0
                return max(0.5, -math.log10(max(1e-7, h)))

        r = self.effective_ratio
        ph = self.pKa + math.log10(r)
        return max(0.5, min(13.5, ph))

    @property
    def hydrogen_ion_concentration_M(self) -> float:
        """[H+] = 10^(-pH)."""
        return 10.0 ** (-self.current_pH)

    @property
    def hydroxide_ion_concentration_M(self) -> float:
        """[OH-] = 10^(-(14 - pH))."""
        return 10.0 ** (-(14.0 - self.current_pH))

    @property
    def van_slyke_capacity(self) -> float:
        """
        Van Slyke buffer capacity (M / pH):
        beta = 2.303 * ( [H+] + [OH-] + (C_total * Ka * [H+]) / (Ka + [H+])^2 )
        """
        return self.calculate_buffer_capacity_at_pH(self.current_pH)

    def calculate_buffer_capacity_at_pH(self, eval_pH: float) -> float:
        """Calculates Van Slyke capacity beta at any given arbitrary pH value."""
        h = 10.0 ** (-eval_pH)
        oh = (1.0e-14) / max(1e-15, h)
        ka = self.Ka
        c_tot = self._c_total_M

        if self._active_preset_code == "UNBUFFERED_WATER_COMPARISON":
            return 2.303 * (h + oh)

        denom = (ka + h) ** 2.0
        buffer_term = (c_tot * ka * h) / denom if denom > 1e-30 else 0.0
        beta = 2.303 * (h + oh + buffer_term)
        return min(2.5, beta)

    @property
    def maximum_buffer_capacity(self) -> float:
        """Global maximum buffer capacity beta_max = 0.575 * C_total (at pH = pKa)."""
        return 0.575 * self._c_total_M

    @property
    def is_in_effective_buffer_range(self) -> bool:
        """Effective buffer range is within pKa ± 1.0."""
        return abs(self.current_pH - self.pKa) <= 1.0

    @property
    def speciation_fraction_HA(self) -> float:
        """alpha_HA = [HA] / C_total = [H+] / (Ka + [H+])."""
        h = self.hydrogen_ion_concentration_M
        return h / (self.Ka + h)

    @property
    def speciation_fraction_A(self) -> float:
        """alpha_A- = [A-] / C_total = Ka / (Ka + [H+])."""
        h = self.hydrogen_ion_concentration_M
        return self.Ka / (self.Ka + h)

    def get_indicator_color(self, ph: float) -> QColor:
        """
        Returns authentic Universal Indicator chromophore color corresponding to pH:
        pH < 3: Red (#ef4444)
        pH 4-5: Orange (#f97316)
        pH 6: Yellow-Green (#eab308)
        pH 7: Emerald Green (#22c55e)
        pH 8-9: Cyan-Blue (#06b6d4)
        pH 10-11: Deep Indigo (#3b82f6)
        pH >= 12: Violet-Purple (#a855f7)
        """
        if ph <= 3.0:
            return QColor("#ef4444")
        elif ph <= 5.0:
            return QColor("#f97316")
        elif ph <= 6.5:
            return QColor("#eab308")
        elif ph <= 7.5:
            return QColor("#22c55e")
        elif ph <= 9.0:
            return QColor("#06b6d4")
        elif ph <= 11.0:
            return QColor("#3b82f6")
        else:
            return QColor("#a855f7")

    # -------------------------------------------------------------------------
    # Simulation Control & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in BUFFER_PRESETS:
            return
        self._active_preset_code = code
        self._preset = BUFFER_PRESETS[code]
        self._c_total_M = self._preset.default_c_total_M
        self._base_acid_ratio = self._preset.default_base_acid_ratio
        self._added_acid_mM = 0.0
        self._added_base_mM = 0.0

        self._layout_buttons()
        self.preset_changed.emit(code)
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def add_strong_acid_drop(self, delta_mM: float = 5.0) -> None:
        """Simulates addition of strong acid (HCl) dropwise."""
        self._added_acid_mM += delta_mM
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def add_strong_base_drop(self, delta_mM: float = 5.0) -> None:
        """Simulates addition of strong base (NaOH) dropwise."""
        self._added_base_mM += delta_mM
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def reset_perturbations(self) -> None:
        """Resets added acid and base to restore baseline equilibrium."""
        self._added_acid_mM = 0.0
        self._added_base_mM = 0.0
        self._base_acid_ratio = self._preset.default_base_acid_ratio
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def adjust_ratio(self, delta: float) -> None:
        """Adjusts baseline [A-] / [HA] ratio."""
        new_ratio = max(0.01, min(100.0, self._base_acid_ratio + delta))
        self._base_acid_ratio = new_ratio
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def adjust_total_concentration(self, delta_M: float) -> None:
        """Adjusts total analytical buffer concentration C_total [0.01, 1.0 M]."""
        self._c_total_M = max(0.01, min(1.0, self._c_total_M + delta_M))
        self.buffer_computed.emit(self.current_pH, self.van_slyke_capacity, self.effective_ratio)
        self.update()

    def _on_sim_tick(self) -> None:
        """Animation tick for beaker liquid surface and bubbling."""
        if not self._is_running:
            return
        self._wave_phase += 0.15
        if self._wave_phase > 2.0 * math.pi:
            self._wave_phase -= 2.0 * math.pi
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
        self._buttons["btn_add_acid"] = QRectF(188.0, by, 105.0, bh)
        self._buttons["btn_add_base"] = QRectF(301.0, by, 105.0, bh)
        self._buttons["btn_ratio_down"] = QRectF(414.0, by, 72.0, bh)
        self._buttons["btn_ratio_up"] = QRectF(494.0, by, 72.0, bh)
        self._buttons["btn_c_down"] = QRectF(574.0, by, 62.0, bh)
        self._buttons["btn_c_up"] = QRectF(644.0, by, 62.0, bh)
        self._buttons["btn_reset"] = QRectF(714.0, by, 130.0, bh)
        self._buttons["btn_stamp"] = QRectF(852.0, by, 88.0, bh)

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
        self._paint_van_slyke_panel(painter)
        self._paint_speciation_and_beaker_panel(painter)
        self._paint_stress_test_comparison_panel(painter)
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
        painter.drawText(hx, hy + 18.0, "HỆ ĐỆM AXIT - BAZƠ & DUNG LƯỢNG VAN SLYKE (BUFFER CAPACITY)")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(hx, hy + 35.0, self._preset.title_vi)

        # Badges on right side
        bx = self.CARD_W - 20.0
        badges = [
            (f"β_max = {self.maximum_buffer_capacity:.3f} M/pH", QColor(168, 85, 247)),
            (f"C_total = {self._c_total_M:.2f} M", QColor(14, 165, 233)),
            (f"pKa = {self.pKa:.2f}", QColor(16, 185, 129)),
            (self._preset.badge_label, QColor(245, 158, 11)),
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

    def _paint_van_slyke_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Left Panel: Van Slyke Buffer Capacity Curve beta(pH) & Operating Point.
        Visualizes the bell-shaped capacity peak at pH = pKa and effective buffer zone [pKa-1, pKa+1].
        """
        vx = 20.0
        vy = 70.0
        vw = 460.0
        vh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(vx, vy, vw, vh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(vx + 12.0, vy + 18.0, "ĐỒ THỊ DUNG LƯỢNG ĐỆM VAN SLYKE β(pH):")

        # Status badge: Effective range
        is_eff = self.is_in_effective_buffer_range
        status_txt = "TRONG KHOẢNG ĐỆM (pKa ± 1)" if is_eff else "NGOÀI KHOẢNG ĐỆM HIỆU DỤNG"
        status_col = QColor(16, 185, 129) if is_eff else QColor(244, 63, 94)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        badge_w = painter.fontMetrics().horizontalAdvance(status_txt) + 14.0
        badge_rect = QRectF(vx + vw - badge_w - 12.0, vy + 6.0, badge_w, 18.0)
        painter.setBrush(QBrush(status_col.darker(250)))
        painter.setPen(QPen(status_col, 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)
        painter.setPen(QPen(status_col.lighter(130)))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, status_txt)

        # Plot frame
        gx = vx + 40.0
        gy = vy + 35.0
        gw = vw - 55.0
        gh = vh - 65.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(gx, gy, gw, gh))

        # Axis limits: pH from 1.0 to 13.0, beta from 0.0 to max_beta_vis
        min_ph = 1.0
        max_ph = 13.0
        span_ph = max_ph - min_ph
        max_beta_vis = max(0.20, self.maximum_buffer_capacity * 1.5)

        def map_x(ph: float) -> float:
            return gx + ((ph - min_ph) / span_ph) * gw

        def map_y(beta_val: float) -> float:
            clamped = min(max_beta_vis, beta_val)
            return gy + gh - (clamped / max_beta_vis) * gh

        # Highlight Effective Buffer Zone [pKa - 1, pKa + 1]
        pka_left = map_x(max(min_ph, self.pKa - 1.0))
        pka_right = map_x(min(max_ph, self.pKa + 1.0))
        zone_rect = QRectF(pka_left, gy, pka_right - pka_left, gh)

        painter.setBrush(QBrush(QColor(16, 185, 129, 35)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(zone_rect)

        # pH grid lines (every 2 pH units)
        painter.setFont(QFont("Segoe UI", 6))
        for p in range(2, 14, 2):
            cx = map_x(float(p))
            painter.setPen(QPen(QColor(30, 41, 59, 140), 0.8, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(cx, gy), QPointF(cx, gy + gh))
            painter.setPen(QPen(QColor(100, 116, 139)))
            painter.drawText(cx - 6.0, gy + gh + 14.0, str(p))

        # X-axis label
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(gx + gw / 2.0 - 25.0, gy + gh + 26.0, "Chỉ số pH")

        # Y-axis label
        painter.drawText(vx + 6.0, gy + 12.0, "β (M/pH)")

        # Plot Van Slyke curve
        curve_path = QPainterPath()
        num_pts = 120
        first = True
        for i in range(num_pts + 1):
            eval_p = min_ph + (i / num_pts) * span_ph
            beta_v = self.calculate_buffer_capacity_at_pH(eval_p)
            px = map_x(eval_p)
            py = map_y(beta_v)
            if first:
                curve_path.moveTo(px, py)
                first = False
            else:
                curve_path.lineTo(px, py)

        # Draw filled gradient under curve
        fill_path = QPainterPath(curve_path)
        fill_path.lineTo(gx + gw, gy + gh)
        fill_path.lineTo(gx, gy + gh)
        fill_path.closeSubpath()

        curve_grad = QLinearGradient(gx, gy, gx, gy + gh)
        curve_grad.setColorAt(0.0, QColor(14, 165, 233, 110))
        curve_grad.setColorAt(1.0, QColor(14, 165, 233, 10))
        painter.setBrush(QBrush(curve_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(curve_path)

        # Draw pKa maximum capacity line
        pka_x = map_x(self.pKa)
        pka_y = map_y(self.maximum_buffer_capacity)

        painter.setPen(QPen(QColor(168, 85, 247, 180), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(pka_x, gy), QPointF(pka_x, gy + gh))

        # Peak indicator dot
        painter.setBrush(QBrush(QColor(168, 85, 247)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.drawEllipse(QPointF(pka_x, pka_y), 3.5, 3.5)

        # Peak label
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(192, 132, 252)))
        painter.drawText(pka_x - 28.0, pka_y - 6.0, f"pKa = {self.pKa:.2f}")

        # Current Operating Point: (current_pH, van_slyke_capacity)
        cur_x = map_x(self.current_pH)
        cur_y = map_y(self.van_slyke_capacity)

        # Vertical line at current pH
        painter.setPen(QPen(QColor(245, 158, 11, 220), 1.6, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cur_x, gy), QPointF(cur_x, gy + gh))

        # Glowing operating point
        rad_op = QRadialGradient(cur_x, cur_y, 10.0)
        rad_op.setColorAt(0.0, QColor(245, 158, 11, 255))
        rad_op.setColorAt(0.6, QColor(245, 158, 11, 160))
        rad_op.setColorAt(1.0, QColor(245, 158, 11, 0))
        painter.setBrush(QBrush(rad_op))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cur_x, cur_y), 10.0, 10.0)

        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.drawEllipse(QPointF(cur_x, cur_y), 4.0, 4.0)

        # Coordinate callout badge
        badge_op = QRectF(cur_x - 36.0, gy - 16.0, 72.0, 14.0)
        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_op, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(badge_op, Qt.AlignmentFlag.AlignCenter, f"pH = {self.current_pH:.2f}")

    def _paint_speciation_and_beaker_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Right Panel: Chemical Speciation Curves alpha(pH) and Universal Indicator Liquid Beaker.
        """
        bx = 500.0
        by = 70.0
        bw = 440.0
        bh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(bx, by, bw, bh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(bx + 12.0, by + 18.0, "PHÂN BỐ CẤU TỬ (SPECIATION) & BÌNH DUNG DỊCH ĐỆM:")

        # Left side: Speciation Plot (180 px wide)
        sx = bx + 34.0
        sy = by + 35.0
        sw = 175.0
        sh = bh - 65.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(sx, sy, sw, sh))

        # Alpha curves: HA (red/orange) and A- (blue/cyan)
        pka = self.pKa
        path_ha = QPainterPath()
        path_a = QPainterPath()

        num_pts = 60
        for i in range(num_pts + 1):
            eval_p = 1.0 + (i / num_pts) * 12.0
            h = 10.0 ** (-eval_p)
            ka = self.Ka
            denom = ka + h
            alpha_ha = h / denom if denom > 0 else 0.0
            alpha_a = ka / denom if denom > 0 else 0.0

            px = sx + ((eval_p - 1.0) / 12.0) * sw
            py_ha = sy + sh - alpha_ha * sh
            py_a = sy + sh - alpha_a * sh

            if i == 0:
                path_ha.moveTo(px, py_ha)
                path_a.moveTo(px, py_a)
            else:
                path_ha.lineTo(px, py_ha)
                path_a.lineTo(px, py_a)

        painter.setPen(QPen(QColor("#f43f5e"), 1.8))
        painter.drawPath(path_ha)

        painter.setPen(QPen(QColor("#06b6d4"), 1.8))
        painter.drawPath(path_a)

        # Crossover indicator at pH = pKa (alpha = 0.5)
        cross_x = sx + ((pka - 1.0) / 12.0) * sw
        cross_y = sy + sh * 0.5
        painter.setPen(QPen(QColor(168, 85, 247, 180), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cross_x, sy), QPointF(cross_x, sy + sh))

        painter.setBrush(QBrush(QColor(168, 85, 247)))
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.drawEllipse(QPointF(cross_x, cross_y), 3.0, 3.0)

        # Speciation Legend
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(sx + 6.0, sy + 14.0, f"α({self._preset.acid_name}) = {self.speciation_fraction_HA * 100:.0f}%")
        painter.setPen(QPen(QColor("#06b6d4")))
        painter.drawText(sx + 6.0, sy + 28.0, f"α({self._preset.base_name}) = {self.speciation_fraction_A * 100:.0f}%")

        # Right side: Laboratory Chemistry Beaker & Digital pH Meter
        beaker_x = bx + 245.0
        beaker_y = by + 45.0
        beaker_w = 160.0
        beaker_h = 165.0

        # Glass Beaker outline
        painter.setPen(QPen(QColor(148, 163, 184, 180), 2.0))
        painter.drawLine(QPointF(beaker_x, beaker_y), QPointF(beaker_x, beaker_y + beaker_h))
        painter.drawLine(QPointF(beaker_x, beaker_y + beaker_h), QPointF(beaker_x + beaker_w, beaker_y + beaker_h))
        painter.drawLine(QPointF(beaker_x + beaker_w, beaker_y + beaker_h), QPointF(beaker_x + beaker_w, beaker_y))

        # Liquid level inside beaker (65% full)
        liquid_top_y = beaker_y + beaker_h * 0.35
        liquid_col = self.get_indicator_color(self.current_pH)

        # Liquid wave path
        liquid_path = QPainterPath()
        liquid_path.moveTo(beaker_x + 2.0, beaker_y + beaker_h - 2.0)
        liquid_path.lineTo(beaker_x + beaker_w - 2.0, beaker_y + beaker_h - 2.0)
        liquid_path.lineTo(beaker_x + beaker_w - 2.0, liquid_top_y)

        # Oscillating wave crest
        wave_pts = 10
        for i in range(wave_pts, -1, -1):
            wx = beaker_x + 2.0 + (i / wave_pts) * (beaker_w - 4.0)
            wy = liquid_top_y + 2.5 * math.sin((i / wave_pts) * 2.0 * math.pi + self._wave_phase)
            liquid_path.lineTo(wx, wy)

        liquid_path.closeSubpath()

        # Fill liquid with translucent indicator color
        liquid_grad = QLinearGradient(beaker_x, liquid_top_y, beaker_x, beaker_y + beaker_h)
        liquid_grad.setColorAt(0.0, QColor(liquid_col.red(), liquid_col.green(), liquid_col.blue(), 190))
        liquid_grad.setColorAt(1.0, QColor(liquid_col.red(), liquid_col.green(), liquid_col.blue(), 230))
        painter.setBrush(QBrush(liquid_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(liquid_path)

        # Digital pH Meter Floating Inset inside beaker
        meter_w = 120.0
        meter_h = 42.0
        meter_x = beaker_x + (beaker_w - meter_w) / 2.0
        meter_y = beaker_y + beaker_h + 12.0

        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(QRectF(meter_x, meter_y, meter_w, meter_h), 6.0, 6.0)

        # Big LCD pH readout
        painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        painter.setPen(QPen(liquid_col.lighter(130)))
        painter.drawText(QRectF(meter_x, meter_y + 2.0, meter_w, 24.0), Qt.AlignmentFlag.AlignCenter, f"{self.current_pH:.2f}")

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(QRectF(meter_x, meter_y + 24.0, meter_w, 14.0), Qt.AlignmentFlag.AlignCenter, "CHỈ THỊ VẠN NĂNG")

    def _paint_stress_test_comparison_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Left Panel: Strong Acid/Base Addition Titration Stress Test.
        Compares pH shift of buffered solution vs unbuffered pure water.
        """
        tx = 20.0
        ty = 360.0
        tw = 460.0
        th = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(tx, ty, tw, th), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(tx + 12.0, ty + 18.0, "THỬ THÁCH KHÁNG AXIT/BAZƠ (STRESS TEST PERTURBATION):")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        net_perturbation_mM = self._added_base_mM - self._added_acid_mM
        painter.drawText(tx + 12.0, ty + 32.0, f"Axit HCl thêm: {self._added_acid_mM:.1f} mM | Bazơ NaOH thêm: {self._added_base_mM:.1f} mM (Net: {net_perturbation_mM:+.1f} mM)")

        # Comparison Bars: Buffer vs Pure Water
        bar_y1 = ty + 50.0
        bar_y2 = ty + 105.0
        bar_w_max = 280.0
        bar_x = tx + 130.0

        # 1. Buffered Solution Response
        # Initial pH at baseline
        r0 = max(0.001, self._base_acid_ratio)
        initial_buffer_pH = self.pKa + math.log10(r0)
        final_buffer_pH = self.current_pH
        delta_buffer_pH = final_buffer_pH - initial_buffer_pH

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(tx + 12.0, bar_y1 + 18.0, "DUNG DỊCH ĐỆM:")

        # Bar background track
        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(bar_x, bar_y1 + 4.0, bar_w_max, 22.0))

        # Fill bar length proportional to pH / 14
        fill_w_buf = (final_buffer_pH / 14.0) * bar_w_max
        painter.setBrush(QBrush(self.get_indicator_color(final_buffer_pH)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(bar_x, bar_y1 + 4.0, fill_w_buf, 22.0))

        # Value readout
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(bar_x + bar_w_max + 8.0, bar_y1 + 19.0, f"pH = {final_buffer_pH:.2f} (ΔpH = {delta_buffer_pH:+.2f})")

        # 2. Pure Water Response (Extreme Sensitivity)
        if abs(net_perturbation_mM) < 1e-4:
            water_pH = 7.00
        elif net_perturbation_mM > 0:
            water_pH = min(13.5, 14.0 + math.log10(max(1e-7, net_perturbation_mM / 1000.0)))
        else:
            water_pH = max(0.5, -math.log10(max(1e-7, abs(net_perturbation_mM) / 1000.0)))
        delta_water_pH = water_pH - 7.00

        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(tx + 12.0, bar_y2 + 18.0, "NƯỚC CẤT (H₂O):")

        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(bar_x, bar_y2 + 4.0, bar_w_max, 22.0))

        fill_w_water = (water_pH / 14.0) * bar_w_max
        painter.setBrush(QBrush(self.get_indicator_color(water_pH)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(bar_x, bar_y2 + 4.0, fill_w_water, 22.0))

        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(bar_x + bar_w_max + 8.0, bar_y2 + 19.0, f"pH = {water_pH:.2f} (ΔpH = {delta_water_pH:+.2f})")

        # 3. Scientific Analysis Box
        info_rect = QRectF(tx + 12.0, ty + 145.0, tw - 24.0, 60.0)
        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(info_rect, 6.0, 6.0)

        lines = [
            f"• Tác động Axit: H⁺ + {self._preset.base_name} → {self._preset.acid_name} (Bazơ liên hợp trung hòa H⁺)",
            f"• Tác động Bazơ: OH⁻ + {self._preset.acid_name} → {self._preset.base_name} + H₂O (Axit liên hợp trung hòa OH⁻)",
            f"• Khả năng chống biến đổi pH của hệ đệm vượt trội gấp ~{(abs(delta_water_pH) / max(0.001, abs(delta_buffer_pH))):.0f}x so với nước cất!",
        ]
        row_y = info_rect.y() + 16.0
        painter.setFont(QFont("Segoe UI", 6))
        for l in lines:
            painter.setPen(QPen(QColor(226, 232, 240) if "vượt trội" in l else QColor(148, 163, 184)))
            painter.drawText(info_rect.x() + 8.0, row_y, l)
            row_y += 18.0

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Right Panel: Rigorous Quantitative Analytical Buffer Formulations.
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
        painter.drawText(mx + 12.0, my + 18.0, "CÔNG THỨC HENDERSON-HASSELBALCH & VAN SLYKE:")

        metrics = [
            ("Phương trình Henderson-Hasselbalch:", f"pH = pKa + log([A⁻]/[HA]) = {self.current_pH:.2f}"),
            ("Hằng số phân ly Axit (pKa / Ka):", f"pKa = {self.pKa:.2f} (Ka = {self.Ka:.2e})"),
            ("Tỷ lệ Nồng độ Bazơ / Axit ([A⁻]/[HA]):", f"[A⁻]/[HA] = {self.effective_ratio:.3f} (log = {math.log10(self.effective_ratio):+.2f})"),
            ("Tổng Nồng độ Hệ Đệm (Total Concentration):", f"C_total = [HA] + [A⁻] = {self._c_total_M:.3f} M"),
            ("Dung lượng Đệm Van Slyke Thực tế (β):", f"β = {self.van_slyke_capacity:.4f} M/pH"),
            ("Dung lượng Đệm Cực đại Lý thuyết (β_max):", f"β_max = 0.575·C_total = {self.maximum_buffer_capacity:.4f} M/pH"),
            ("Khoảng Đệm Hiệu dụng (Effective Range):", f"pH ∈ [{self.pKa - 1.0:.2f}, {self.pKa + 1.0:.2f}] ({'Hiệu dụng' if self.is_in_effective_buffer_range else 'Vượt ngưỡng'})"),
        ]

        row_y = my + 40.0
        line_h = 24.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            val_col = QColor(16, 185, 129) if "Hiệu dụng" in val or "β_max" in name else QColor(226, 232, 240)
            painter.setPen(QPen(val_col))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture control buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_add_acid", "+ HCl (Axit)", QColor(239, 68, 68)),
            ("btn_add_base", "+ NaOH (Bazơ)", QColor(59, 130, 246)),
            ("btn_ratio_down", "Tỷ lệ -", QColor(51, 65, 85)),
            ("btn_ratio_up", "Tỷ lệ +", QColor(51, 65, 85)),
            ("btn_c_down", "C -", QColor(51, 65, 85)),
            ("btn_c_up", "C +", QColor(51, 65, 85)),
            ("btn_reset", "Hoàn nguyên đệm", QColor(245, 158, 11)),
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
            keys = list(BUFFER_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_add_acid":
            self.add_strong_acid_drop(5.0)
        elif key == "btn_add_base":
            self.add_strong_base_drop(5.0)
        elif key == "btn_ratio_down":
            self.adjust_ratio(-0.2)
        elif key == "btn_ratio_up":
            self.adjust_ratio(+0.2)
        elif key == "btn_c_down":
            self.adjust_total_concentration(-0.05)
        elif key == "btn_c_up":
            self.adjust_total_concentration(+0.05)
        elif key == "btn_reset":
            self.reset_perturbations()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "acid_name": self._preset.acid_name,
            "base_name": self._preset.base_name,
            "pKa": self.pKa,
            "c_total_M": self._c_total_M,
            "ratio": self.effective_ratio,
            "current_pH": self.current_pH,
            "van_slyke_capacity": self.van_slyke_capacity,
            "beta_max": self.maximum_buffer_capacity,
        }
        self.stamp_requested.emit(data)
