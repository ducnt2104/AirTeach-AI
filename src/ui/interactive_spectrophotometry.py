"""
AirTeach AI — Interactive Analytical Chemistry & Spectroscopy:
Spectrophotometry, Beer-Lambert Law & Chemical Absorbance Simulator.

Simulates:
  - Beer-Lambert Law (Định luật Bouguer-Lambert-Beer):
      A = -log10(T) = -log10(I / I_0) = ε * c * l
      T = I / I_0 = 10^(-ε * c * l)
      %T = 100 * 10^(-A)
  - Molar absorptivity (extinction coefficient) ε (L/(mol·cm))
  - Cuvette optical path length l (cm)
  - Analyte molar concentration c (mol/L or mM)
  - Spectral absorption profile A(λ) with Gaussian peak at λ_max
  - Complementary color theory (transmitted vs absorbed color)
  - High-concentration deviation & stray light breakdown:
      A_apparent = -log10(10^(-A_true) + s_stray)
"""

from __future__ import annotations
import math
import random
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
class SpectroPreset:
    """Represents a curriculum scenario for Spectrophotometry & Beer-Lambert Law."""
    code: str
    title_vi: str
    badge_label: str
    analyte_name: str
    lambda_max_nm: float          # Peak absorption wavelength in nm
    epsilon_L_mol_cm: float       # Molar absorptivity at lambda_max (L/(mol·cm))
    default_conc_mM: float        # Default concentration in mM (1 mM = 10^-3 M)
    default_path_cm: float        # Cuvette path length in cm (standard 1.0 cm)
    liquid_color: QColor          # Solution observed visual color
    absorbed_color: QColor        # Color absorbed by chromophore
    description: str


SPECTRO_PRESETS: dict[str, SpectroPreset] = {
    "POTASSIUM_PERMANGANATE_KMnO4": SpectroPreset(
        code="POTASSIUM_PERMANGANATE_KMnO4",
        title_vi="Kali Pemanganat (KMnO₄) — Hấp Thụ Xanh Lục, Hiện Màu Tím Tía",
        badge_label="KMnO₄ PURPLE",
        analyte_name="KMnO₄ (Potassium Permanganate)",
        lambda_max_nm=525.0,
        epsilon_L_mol_cm=2400.0,
        default_conc_mM=0.40,         # A = 2400 * 0.0004 * 1.0 = 0.96
        default_path_cm=1.0,
        liquid_color=QColor("#a855f7"),      # Deep purple / magenta
        absorbed_color=QColor("#22c55e"),    # Green light absorbed
        description="Chất oxy hóa pemanganat kinh điển. Dung dịch có màu tím tía do hấp thụ mạnh ánh sáng màu lục (λ_max = 525 nm). Hệ số hấp thụ mol lớn ε = 2400 cho phép định lượng ở nồng độ sub-millimolar.",
    ),
    "COPPER_SULFATE_PENTAHYDRATE": SpectroPreset(
        code="COPPER_SULFATE_PENTAHYDRATE",
        title_vi="Đồng(II) Sunfat ([Cu(H₂O)₆]²⁺) — Hấp Thụ Đỏ/Cận Hồng Ngoại, Hiện Màu Lam",
        badge_label="CuSO₄ BLUE",
        analyte_name="[Cu(H₂O)₆]²⁺ (Copper Sulfate)",
        lambda_max_nm=810.0,
        epsilon_L_mol_cm=12.5,
        default_conc_mM=40.0,         # A = 12.5 * 0.040 * 1.0 = 0.50
        default_path_cm=1.0,
        liquid_color=QColor("#0ea5e9"),      # Cyan blue
        absorbed_color=QColor("#ef4444"),    # Red light absorbed
        description="Phức chất aquo đồng(II) bát diện. Bước chuyển mức năng lượng d-d bị cấm Laporte dẫn đến hệ số hấp thụ nhỏ (ε ≈ 12.5), đòi hỏi nồng độ cao (10 - 100 mM) để đo độ hấp thụ tối ưu A ≈ 0.2 - 0.8.",
    ),
    "PROTEIN_BRADFORD_COOMASSIE": SpectroPreset(
        code="PROTEIN_BRADFORD_COOMASSIE",
        title_vi="Định Lượng Protein Bradford — Phức Hệ Xanh Coomassie G-250 (595 nm)",
        badge_label="BRADFORD 595nm",
        analyte_name="Protein BSA + Coomassie Blue",
        lambda_max_nm=595.0,
        epsilon_L_mol_cm=43000.0,
        default_conc_mM=0.015,        # 15 uM, A = 43000 * 1.5e-5 * 1.0 = 0.645
        default_path_cm=1.0,
        liquid_color=QColor("#2563eb"),      # Intense royal blue
        absorbed_color=QColor("#eab308"),    # Yellow-orange absorbed
        description="Phương pháp định lượng protein Bradford sinh hóa. Thuốc nhuộm Coomassie Brilliant Blue chuyển từ dạng cation đỏ (465 nm) sang dạng anion liên kết protein màu lam đậm tại 595 nm.",
    ),
    "DNA_UV_260_NUCLEIC_ACID": SpectroPreset(
        code="DNA_UV_260_NUCLEIC_ACID",
        title_vi="Đo Nồng Độ ADN Quang Phổ Tử Ngoại — Đỉnh Hấp Thụ UV 260 nm",
        badge_label="DNA UV 260nm",
        analyte_name="ADN Chuỗi Kép (dsDNA)",
        lambda_max_nm=260.0,
        epsilon_L_mol_cm=6600.0,
        default_conc_mM=0.10,         # A = 6600 * 1e-4 * 1.0 = 0.66
        default_path_cm=1.0,
        liquid_color=QColor("#e2e8f0"),      # Visually clear/colorless
        absorbed_color=QColor("#8b5cf6"),    # UV radiation absorbed
        description="Các bazơ nitơ (adenine, guanine, cytosine, thymine) hấp thụ cực mạnh bức xạ tử ngoại ở 260 nm nhờ hệ liên kết pi liên hợp thơm. Tỷ số A260/A280 = 1.8 là tiêu chuẩn vàng đánh giá độ tinh sạch ADN.",
    ),
    "BEER_LAMBERT_HIGH_CONC_DEVIATION": SpectroPreset(
        code="BEER_LAMBERT_HIGH_CONC_DEVIATION",
        title_vi="Độ Lệch Định Luật Beer Ở Nồng Độ Cao — Tương Tác Tĩnh Điện & Ánh Sáng Tán Xạ",
        badge_label="NON-LINEAR LIMIT",
        analyte_name="Phẩm Nhuộm Nồng Độ Cao",
        lambda_max_nm=500.0,
        epsilon_L_mol_cm=1500.0,
        default_conc_mM=3.0,          # Theoretical A = 4.5, but deviates to plateau ~2.3
        default_path_cm=1.0,
        liquid_color=QColor("#dc2626"),      # Deep crimson red
        absorbed_color=QColor("#06b6d4"),    # Cyan absorbed
        description="Khi nồng độ vượt quá 0.01 M, lực tương tác tĩnh điện giữa các phân tử mang điện và hiện tượng ánh sáng phân tán lạc (stray light) làm đường chuẩn bị cong gãy, vi phạm tính tuyến tính của định luật Beer.",
    ),
}


class InteractiveSpectrophotometryItem(QGraphicsObject):
    """
    Interactive PySide6 Graphics Item simulating a UV-Vis Spectrophotometer,
    cuvette light ray transmission, Beer-Lambert absorbance, and standard calibration curves.
    """

    spectro_computed = Signal(float, float, float)  # absorbance, transmittance_pct, conc_mM
    stamp_requested = Signal(dict)

    CARD_W = 740.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "POTASSIUM_PERMANGANATE_KMnO4"
        self._preset: SpectroPreset = SPECTRO_PRESETS[self._active_preset_code]

        self._conc_mM: float = self._preset.default_conc_mM
        self._path_cm: float = self._preset.default_path_cm
        self._wavelength_nm: float = self._preset.lambda_max_nm
        self._stray_light_pct: float = 0.05  # 0.05% instrumental stray light

        # Light beam animation phase
        self._beam_phase: float = 0.0
        self._is_running: bool = True

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for 60fps real-time light pulse animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Physical & Analytical Spectroscopy Calculations (Beer-Lambert Law)
    # -------------------------------------------------------------------------
    @property
    def current_epsilon(self) -> float:
        """
        Molar absorptivity ε at current wavelength λ, modeled with Gaussian absorption band:
        ε(λ) = ε_max * exp(-0.5 * ((λ - λ_max) / σ)²)
        """
        lam = self._wavelength_nm
        lam_max = self._preset.lambda_max_nm
        sigma = 35.0  # Bandwidth in nm
        gaussian_factor = math.exp(-0.5 * (((lam - lam_max) / sigma) ** 2))
        return self._preset.epsilon_L_mol_cm * gaussian_factor

    @property
    def concentration_M(self) -> float:
        """Concentration c in M (mol/L)."""
        return max(1e-6, self._conc_mM * 1e-3)

    @property
    def true_absorbance(self) -> float:
        """
        Ideal Beer-Lambert absorbance:
        A = ε * c * l
        """
        c = self.concentration_M
        l = max(0.1, self._path_cm)
        eps = self.current_epsilon
        return eps * c * l

    @property
    def apparent_absorbance(self) -> float:
        """
        Measured absorbance with chemical non-ideality & instrumental stray light:
        A_apparent = -log10(10^(-A_true) + s_stray)
        """
        a_true = self.true_absorbance
        # Chemical association factor at high concentration > 0.005 M
        c = self.concentration_M
        if c > 0.005:
            # Non-linear suppression factor
            dev_factor = 1.0 / (1.0 + 0.15 * ((c - 0.005) * 100.0) ** 1.3)
            a_true *= dev_factor

        t_true = 10.0 ** (-a_true)
        stray = self._stray_light_pct / 100.0
        t_measured = max(1e-5, min(1.0, t_true + stray))
        return -math.log10(t_measured)

    @property
    def transmittance_fraction(self) -> float:
        """Transmittance T = I / I_0 = 10^(-A)."""
        a = self.apparent_absorbance
        return max(0.0, min(1.0, 10.0 ** (-a)))

    @property
    def transmittance_percent(self) -> float:
        """Transmittance in percent: %T = T * 100%."""
        return self.transmittance_fraction * 100.0

    # -------------------------------------------------------------------------
    # Animation & Lifecycle
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        """Light beam pulsation animation."""
        if not self._is_running:
            return
        self._beam_phase += 0.08
        if self._beam_phase > 2.0 * math.pi:
            self._beam_phase -= 2.0 * math.pi
        self.update()

    # -------------------------------------------------------------------------
    # Controls & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in SPECTRO_PRESETS:
            return
        self._active_preset_code = code
        self._preset = SPECTRO_PRESETS[code]
        self._conc_mM = self._preset.default_conc_mM
        self._path_cm = self._preset.default_path_cm
        self._wavelength_nm = self._preset.lambda_max_nm

        self.spectro_computed.emit(
            self.apparent_absorbance,
            self.transmittance_percent,
            self._conc_mM,
        )
        self.update()

    def set_concentration_mM(self, c_mM: float) -> None:
        """Sets analyte concentration c in mM [0.001, 100.0]."""
        self._conc_mM = max(0.001, min(100.0, c_mM))
        self.spectro_computed.emit(
            self.apparent_absorbance,
            self.transmittance_percent,
            self._conc_mM,
        )
        self.update()

    def set_path_length_cm(self, l_cm: float) -> None:
        """Sets cuvette path length l in cm [0.1, 5.0]."""
        self._path_cm = max(0.1, min(5.0, l_cm))
        self.spectro_computed.emit(
            self.apparent_absorbance,
            self.transmittance_percent,
            self._conc_mM,
        )
        self.update()

    def set_wavelength_nm(self, lam_nm: float) -> None:
        """Sets monochromator wavelength λ in nm [200.0, 900.0]."""
        self._wavelength_nm = max(200.0, min(900.0, lam_nm))
        self.spectro_computed.emit(
            self.apparent_absorbance,
            self.transmittance_percent,
            self._conc_mM,
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets parameters to active preset defaults."""
        self._conc_mM = self._preset.default_conc_mM
        self._path_cm = self._preset.default_path_cm
        self._wavelength_nm = self._preset.lambda_max_nm
        self.spectro_computed.emit(
            self.apparent_absorbance,
            self.transmittance_percent,
            self._conc_mM,
        )
        self.update()

    # -------------------------------------------------------------------------
    # Layout Buttons
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Defines touch button bounding boxes at the bottom."""
        by = self.CARD_H - 46.0
        bw = 88.0
        bh = 32.0
        spacing = 10.0
        start_x = 18.0

        keys = [
            ("btn_preset", 130.0),
            ("btn_c_down", 72.0),
            ("btn_c_up", 72.0),
            ("btn_l_down", 68.0),
            ("btn_l_up", 68.0),
            ("btn_lam_down", 72.0),
            ("btn_lam_up", 72.0),
            ("btn_reset", 66.0),
            ("btn_stamp", 112.0),
        ]

        curr_x = start_x
        for key, width in keys:
            self._buttons[key] = QRectF(curr_x, by, width, bh)
            curr_x += width + spacing

    # -------------------------------------------------------------------------
    # Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Card Container
        card_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor("#080c14"))
        grad.setColorAt(1.0, QColor("#0f172a"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header
        self._paint_header(painter)

        # 3. Left Panel: Optical Bench & Cuvette Transmission
        self._paint_optical_bench(painter)

        # 4. Right Panel: Calibration Curve (A vs c) & Spectrum A(λ)
        self._paint_calibration_curve(painter)

        # 5. Bottom Metrics Panel
        self._paint_metrics_panel(painter)

        # 6. Bottom Touchless Buttons
        self._paint_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Badge
        badge_rect = QRectF(20.0, 16.0, 185.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"🧪 {self._preset.badge_label}")

        # Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(218.0, 33.0, self._preset.title_vi)

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            22.0,
            58.0,
            f"Dung dịch: {self._preset.analyte_name} | Bước sóng đo λ = {self._wavelength_nm:.0f} nm (λ_max = {self._preset.lambda_max_nm:.0f} nm) | Cu-vét: l = {self._path_cm:.1f} cm",
        )

    def _paint_optical_bench(self, painter: QPainter) -> None:
        """Renders the physical optical spectrophotometer path: Lamp -> Monochromator -> Cuvette -> Detector."""
        px = 22.0
        py = 74.0
        pw = 400.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "1. Đường Truyền Quang Học & Cu-vét Dung Dịch")

        cy = py + ph / 2.0 + 8.0

        # Component 1: Light Source Lamp
        lamp_x = px + 20.0
        lamp_y = cy - 25.0
        lamp_w = 34.0
        lamp_h = 50.0

        lamp_grad = QRadialGradient(lamp_x + lamp_w / 2.0, cy, 20.0)
        lamp_grad.setColorAt(0.0, QColor("#ffffff"))
        lamp_grad.setColorAt(0.4, QColor("#fef08a"))
        lamp_grad.setColorAt(1.0, QColor("#ca8a04"))
        painter.setBrush(QBrush(lamp_grad))
        painter.setPen(QPen(QColor("#eab308"), 1.2))
        painter.drawRoundedRect(QRectF(lamp_x, lamp_y, lamp_w, lamp_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#000000")))
        painter.drawText(QRectF(lamp_x, lamp_y, lamp_w, lamp_h), Qt.AlignmentFlag.AlignCenter, "NGUỒN\nSÁNG")

        # Component 2: Monochromator Prism/Grating
        mono_x = lamp_x + lamp_w + 35.0
        mono_y = cy - 30.0
        mono_w = 40.0
        mono_h = 60.0

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        tri_path = QPainterPath()
        tri_path.moveTo(mono_x, mono_y + mono_h)
        tri_path.lineTo(mono_x + mono_w / 2.0, mono_y)
        tri_path.lineTo(mono_x + mono_w, mono_y + mono_h)
        tri_path.closeSubpath()
        painter.drawPath(tri_path)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(mono_x - 2.0, mono_y + mono_h + 12.0, f"Lăng Kính\n{self._wavelength_nm:.0f}nm")

        # Polychromatic incident beam from Lamp to Monochromator
        painter.setPen(QPen(QColor("#fef08a"), 3.0))
        painter.drawLine(QPointF(lamp_x + lamp_w, cy), QPointF(mono_x, cy))

        # Component 3: Sample Quartz Cuvette
        cuv_w = max(24.0, self._path_cm * 45.0)  # Width proportional to path length l
        cuv_h = 95.0
        cuv_x = mono_x + mono_w + 55.0
        cuv_y = cy - cuv_h / 2.0

        # Monochromatic incident beam I_0 from Monochromator to Cuvette
        # Color of beam corresponds to selected wavelength
        beam_col = self._preset.absorbed_color
        i0_pen = QPen(beam_col, 3.5)
        painter.setPen(i0_pen)
        painter.drawLine(QPointF(mono_x + mono_w, cy), QPointF(cuv_x, cy))

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(beam_col))
        painter.drawText(mono_x + mono_w + 14.0, cy - 8.0, "I₀ (100%)")

        # Liquid inside cuvette
        # Opacity proportional to concentration
        liq_col = self._preset.liquid_color
        alpha_val = int(min(240, max(40, self._conc_mM * 50.0)))
        painter.setBrush(QBrush(QColor(liq_col.red(), liq_col.green(), liq_col.blue(), alpha_val)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(cuv_x + 3.0, cuv_y + 4.0, cuv_w - 6.0, cuv_h - 6.0))

        # Glass cuvette walls
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#e2e8f0"), 2.5))
        painter.drawRect(QRectF(cuv_x, cuv_y, cuv_w, cuv_h))

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(cuv_x - 4.0, cuv_y - 8.0, f"Cu-vét (l={self._path_cm:.1f}cm)")

        # Component 4: Photodiode Detector
        det_x = px + pw - 48.0
        det_y = cy - 25.0
        det_w = 26.0
        det_h = 50.0

        # Transmitted beam I from Cuvette to Detector
        t_pct = self.transmittance_percent
        # Transmitted beam thickness / alpha proportional to %T
        t_alpha = int(max(15, min(255, 2.55 * t_pct)))
        trans_pen = QPen(QColor(beam_col.red(), beam_col.green(), beam_col.blue(), t_alpha), max(1.0, 3.5 * (t_pct / 100.0)))
        painter.setPen(trans_pen)
        painter.drawLine(QPointF(cuv_x + cuv_w, cy), QPointF(det_x, cy))

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(beam_col))
        painter.drawText(cuv_x + cuv_w + 12.0, cy - 8.0, f"I ({t_pct:.1f}%)")

        # Detector Box
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.drawRoundedRect(QRectF(det_x, det_y, det_w, det_h), 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(QRectF(det_x, det_y, det_w, det_h), Qt.AlignmentFlag.AlignCenter, "QUANG\nĐIỆN")

        # Bottom Color Explanation
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(px + 12.0, py + ph - 8.0, f"Màu truyền qua quan sát được (Transmitted): đối màu của ánh sáng bị hấp thụ.")

    def _paint_calibration_curve(self, painter: QPainter) -> None:
        """Renders standard Beer-Lambert linear calibration curve (A vs c)."""
        px = 434.0
        py = 74.0
        pw = 284.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "2. Đường Chuẩn Độ Hấp Thụ (A vs c)")

        # Plot axes
        ox = px + 40.0
        oy = py + ph - 34.0
        ax_w = pw - 55.0
        ax_h = 160.0

        ax_pen = QPen(QColor("#64748b"), 1.2)
        painter.setPen(ax_pen)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + ax_w + 6.0, oy))       # Concentration axis
        painter.drawLine(QPointF(ox, oy), QPointF(ox, oy - ax_h - 6.0))       # Absorbance axis

        # Labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(ox + ax_w - 36.0, oy + 15.0, "c (mM)")
        painter.drawText(ox - 30.0, oy - ax_h - 8.0, "Độ Hấp Thụ (A)")

        # Grid lines
        for gi in range(1, 4):
            gy = oy - gi * (ax_h / 3.0)
            painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(ox, gy), QPointF(ox + ax_w, gy))
            a_val = gi * (2.0 / 3.0)
            painter.drawText(ox - 24.0, gy + 4.0, f"{a_val:.1f}")

        # Calibration line A = ε * l * c
        # Scale: max A = 2.0 mapped to ax_h, max c mapped to ax_w
        max_c = max(1.0, self._preset.default_conc_mM * 2.5)
        max_a = 2.0

        # Draw linear line
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        line_path = QPainterPath()
        line_path.moveTo(ox, oy)
        end_a = min(max_a, (self.current_epsilon * (max_c * 1e-3) * self._path_cm))
        end_x = ox + ax_w
        end_y = oy - (end_a / max_a) * ax_h
        line_path.lineTo(end_x, end_y)
        painter.drawPath(line_path)

        # Current Operating Point dot
        cur_c = self._conc_mM
        cur_a = self.apparent_absorbance
        pt_x = ox + (cur_c / max_c) * ax_w
        pt_y = oy - (min(max_a, cur_a) / max_a) * ax_h

        # Glowing halo
        painter.setBrush(QBrush(QColor(56, 189, 248, 80)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(pt_x, pt_y), 6.0, 6.0)

        # Dot
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.drawEllipse(QPointF(pt_x, pt_y), 3.5, 3.5)

        # Point tag
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(pt_x + 6.0, pt_y - 6.0, f"A={cur_a:.3f}")

        # Slope equation label
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(ox + 8.0, oy - ax_h + 14.0, f"Độ dốc k = ε·l = {self.current_epsilon * self._path_cm:.0f} M⁻¹")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Renders live quantitative parameters, Beer-Lambert calculations, and readouts."""
        px = 22.0
        py = 336.0
        pw = self.CARD_W - 44.0
        ph = 142.0

        # Background container
        painter.setBrush(QBrush(QColor("#09101c")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Header of metrics
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 14.0, py + 20.0, "3. Thông Số Quang Phổ & Định Luật Beer-Lambert: A = -log₁₀(T) = ε · c · l")

        # Column 1
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#e2e8f0")))
        col1_x = px + 14.0
        line1_y = py + 44.0
        dy = 22.0

        painter.drawText(col1_x, line1_y, f"• Nồng độ chất tan c: {self._conc_mM:.3f} mM ({self.concentration_M:.3e} M)")
        painter.drawText(col1_x, line1_y + dy, f"• Bề dày truyền quang cu-vét l: {self._path_cm:.2f} cm")
        painter.drawText(col1_x, line1_y + 2 * dy, f"• Hệ số hấp thụ mol ε: {self.current_epsilon:,.1f} L/(mol·cm)")
        painter.drawText(col1_x, line1_y + 3 * dy, f"• Bước sóng đo λ: {self._wavelength_nm:.0f} nm (λ_max = {self._preset.lambda_max_nm:.0f} nm)")

        # Column 2
        col2_x = px + 360.0
        a_val = self.apparent_absorbance
        t_val = self.transmittance_percent
        painter.drawText(col2_x, line1_y, f"• Độ hấp thụ quang (Absorbance): A = {a_val:.4f} (Mật độ quang OD)")
        painter.drawText(col2_x, line1_y + dy, f"• Độ truyền qua (Transmittance): %T = {t_val:.2f}% (T = {self.transmittance_fraction:.4f})")
        
        opt_status = "TỐI ƯU (0.2 ≤ A ≤ 0.8)" if (0.2 <= a_val <= 0.8) else ("QUÁ ĐẬM (A > 1.0, %T < 10%)" if a_val > 1.0 else "QUÁ LOÃNG (A < 0.2)")
        opt_col = QColor("#10b981") if (0.2 <= a_val <= 0.8) else QColor("#f59e0b")
        painter.setPen(QPen(opt_col))
        painter.drawText(col2_x, line1_y + 2 * dy, f"• Vùng đo tối ưu quang kế: {opt_status}")

        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(col2_x, line1_y + 3 * dy, f"• Ánh sáng lạc (Stray light): s = {self._stray_light_pct:.2f}%  |  A_true = {self.true_absorbance:.4f}")

        # Explanatory Educational Text at bottom
        desc_rect = QRectF(px + 12.0, py + ph + 8.0, pw - 24.0, 95.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive touch/mouse control buttons."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        labels = {
            "btn_preset": f"📋 Kịch Bản ({self._preset.badge_label})",
            "btn_c_down": "c - 20%",
            "btn_c_up": "c + 20%",
            "btn_l_down": "l - 0.2cm",
            "btn_l_up": "l + 0.2cm",
            "btn_lam_down": "λ - 25nm",
            "btn_lam_up": "λ + 25nm",
            "btn_reset": "🔄 Đặt Lại",
            "btn_stamp": "📌 Đóng Dấu Bảng",
        }

        for key, rect in self._buttons.items():
            is_hover = (key == self._hovered_btn)
            text = labels.get(key, key)

            if key == "btn_stamp":
                bg = QColor("#0284c7") if not is_hover else QColor("#0369a1")
                border = QColor("#38bdf8")
            elif key == "btn_preset":
                bg = QColor("#4338ca") if not is_hover else QColor("#3730a3")
                border = QColor("#818cf8")
            elif "up" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#38bdf8")
            elif "down" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#f43f5e")
            else:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#64748b")

            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handling
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
            keys = list(SPECTRO_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_c_down":
            self.set_concentration_mM(self._conc_mM * 0.80)
        elif key == "btn_c_up":
            self.set_concentration_mM(self._conc_mM * 1.20)
        elif key == "btn_l_down":
            self.set_path_length_cm(self._path_cm - 0.20)
        elif key == "btn_l_up":
            self.set_path_length_cm(self._path_cm + 0.20)
        elif key == "btn_lam_down":
            self.set_wavelength_nm(self._wavelength_nm - 25.0)
        elif key == "btn_lam_up":
            self.set_wavelength_nm(self._wavelength_nm + 25.0)
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "analyte_name": self._preset.analyte_name,
            "conc_mM": self._conc_mM,
            "path_cm": self._path_cm,
            "wavelength_nm": self._wavelength_nm,
            "epsilon": self.current_epsilon,
            "absorbance": self.apparent_absorbance,
            "transmittance_pct": self.transmittance_percent,
        }
        self.stamp_requested.emit(data)
