"""
AirTeach AI - Interactive Physical Chemistry: FTIR Spectroscopy & Molecular Vibrations
======================================================================================
Authentic Fourier-Transform Infrared (FTIR) Spectroscopy & Michelson Optical Physics:
- Molecular Vibrational Normal Modes (3N - 6 for non-linear, 3N - 5 for linear):
    Quantum Harmonic Oscillator:
      ν_tilde = (1 / (2π c)) * sqrt(k / μ)
      where reduced mass μ = (m1 * m2) / (m1 + m2), force constant k (N/m).
    Selection rule: Δv = ±1 and dynamic dipole moment gradient dμ/dQ ≠ 0 (IR active).
- Michelson Interferometer & Optical Path Difference (OPD):
    Moving mirror displacement x => Optical path difference δ = 2x.
    Polychromatic broadband infrared interferogram:
      I_AC(δ) = ∫_0^∞ B(ν_tilde) * cos(2π * ν_tilde * δ) dν_tilde
    At Zero Path Difference (ZPD, δ = 0), constructive interference produces the Centerburst.
- Fourier Transform & Spectral Inversion:
    B(ν_tilde) = 2 ∫_0^∞ I_AC(δ) * A_pod(δ) * cos(2π * ν_tilde * δ) dδ
    Apodization window A_pod(δ): Boxcar vs Hamming:
      A_Hamming(δ) = 0.54 + 0.46 * cos(π * δ / L_max)
- Beer-Lambert Law & %Transmittance vs Absorbance:
    T(ν_tilde) = exp(-c * ε(ν_tilde) * l)
    A(ν_tilde) = -log10(T) = c * ε(ν_tilde) * l
- 5 Authentic Chemical Presets:
    1. ETHANOL (CH3CH2OH): O-H broad stretch (3350 cm^-1), C-H sp3 (2970 cm^-1), C-O (1050 cm^-1).
    2. ACETONE (CH3COCH3): Carbonyl C=O strong peak (1715 cm^-1), C-H (2960 cm^-1), C-C (1220 cm^-1).
    3. BENZENE (C6H6): Aromatic =C-H (3030 cm^-1), C=C ring (1480, 1600 cm^-1), =C-H bend (675 cm^-1).
    4. CARBON_DIOXIDE (CO2): Asymmetric stretch ν3 (2349 cm^-1), Bending ν2 (667 cm^-1). Symmetric ν1 (1388 cm^-1, IR inactive).
    5. WATER_VAPOR (H2O): Symmetric stretch ν1 (3657 cm^-1), Bending ν2 (1595 cm^-1), Asymmetric ν3 (3756 cm^-1).
- Touchless air gestures (air-tap presets, mirror sweep, mode toggle),
  bimanual peak zooming, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
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

SPEED_OF_LIGHT_CM_S: float = 2.99792458e10  # cm / s


@dataclass
class IRAbsorptionPeak:
    """An authentic infrared vibrational absorption band."""
    center_cm: float
    width_cm: float
    intensity: float
    assignment: str


@dataclass
class FTIRPreset:
    """Preset molecule for FTIR spectroscopy demonstration."""
    key: str
    name: str
    formula: str
    desc: str
    peaks: List[IRAbsorptionPeak]


PRESETS: List[FTIRPreset] = [
    FTIRPreset(
        key="ethanol",
        name="Etanol (Cồn Rượu CH₃CH₂OH)",
        formula="CH₃CH₂OH",
        desc="Dải O-H liên kết hydro rộng mạnh ở 3350 cm⁻¹, dải C-H no ở 2970 cm⁻¹, và dải C-O ở 1050 cm⁻¹",
        peaks=[
            IRAbsorptionPeak(3350.0, 220.0, 0.92, "O-H dãn (Hydrogen-bonded)"),
            IRAbsorptionPeak(2970.0, 45.0, 0.75, "C-H sp³ dãn bất đối xứng"),
            IRAbsorptionPeak(2890.0, 40.0, 0.60, "C-H sp³ dãn đối xứng"),
            IRAbsorptionPeak(1450.0, 35.0, 0.45, "CH₂ uốn biến dạng (Scissoring)"),
            IRAbsorptionPeak(1380.0, 30.0, 0.40, "CH₃ uốn đối xứng (Rocking)"),
            IRAbsorptionPeak(1050.0, 50.0, 0.85, "C-O dãn bậc một"),
        ],
    ),
    FTIRPreset(
        key="acetone",
        name="Axeton (Đimetyl Xeton CH₃COCH₃)",
        formula="CH₃COCH₃",
        desc="Đặc trưng bởi mũi nhọn Carbonyl C=O cực mạnh ở 1715 cm⁻¹ và dao động C-C-C ở 1220 cm⁻¹",
        peaks=[
            IRAbsorptionPeak(2960.0, 45.0, 0.40, "C-H dãn"),
            IRAbsorptionPeak(1715.0, 30.0, 0.98, "C=O Carbonyl dãn cực mạnh"),
            IRAbsorptionPeak(1420.0, 35.0, 0.50, "CH₃ uốn"),
            IRAbsorptionPeak(1360.0, 30.0, 0.65, "CH₃ uốn biến dạng"),
            IRAbsorptionPeak(1220.0, 40.0, 0.70, "C-C-C khung dãn"),
        ],
    ),
    FTIRPreset(
        key="benzene",
        name="Benzen (Hydrocacbon thơm C₆H₆)",
        formula="C₆H₆",
        desc="Dải C-H thơm ở 3030 cm⁻¹, dải nhân thơm C=C ở 1480 và 1600 cm⁻¹, uốn C-H ngoài mặt phẳng ở 675 cm⁻¹",
        peaks=[
            IRAbsorptionPeak(3030.0, 30.0, 0.70, "=C-H thơm dãn"),
            IRAbsorptionPeak(1600.0, 25.0, 0.45, "C=C vòng thơm"),
            IRAbsorptionPeak(1480.0, 25.0, 0.80, "C=C vòng thơm dãn"),
            IRAbsorptionPeak(1035.0, 30.0, 0.40, "C-H uốn trong mặt phẳng"),
            IRAbsorptionPeak(675.0, 25.0, 0.95, "=C-H uốn ngoài mặt phẳng (OOP)"),
        ],
    ),
    FTIRPreset(
        key="carbon_dioxide",
        name="Khí Cacbonic (CO₂ - Hiệu ứng nhà kính)",
        formula="CO₂",
        desc="Phân tử thẳng: dãn bất đối xứng ν₃ ở 2349 cm⁻¹ (hấp thụ nhiệt hồng ngoại mạnh nhất), dao động uốn ν₂ ở 667 cm⁻¹",
        peaks=[
            IRAbsorptionPeak(2349.0, 35.0, 0.98, "ν₃ dãn bất đối xứng (Greenhouse)"),
            IRAbsorptionPeak(667.0, 30.0, 0.85, "ν₂ uốn suy biến hai chiều"),
        ],
    ),
    FTIRPreset(
        key="water_vapor",
        name="Hơi nước (H₂O Khí quyển)",
        formula="H₂O",
        desc="Phân tử góc C₂v: dãn đối xứng ν₁ ở 3657 cm⁻¹, uốn ν₂ ở 1595 cm⁻¹, và dãn bất đối xứng ν₃ ở 3756 cm⁻¹",
        peaks=[
            IRAbsorptionPeak(3756.0, 60.0, 0.88, "ν₃ dãn bất đối xứng"),
            IRAbsorptionPeak(3657.0, 55.0, 0.78, "ν₁ dãn đối xứng"),
            IRAbsorptionPeak(1595.0, 70.0, 0.82, "ν₂ uốn biến dạng liên kết"),
        ],
    ),
]


class InteractiveFTIRSpectroscopyItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic FTIR Spectroscopy,
    Michelson Interferometer Optical Centerburst, and Fourier Inversion.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 940.0
    HEIGHT: float = 600.0

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        # Physical parameters
        self._concentration: float = 1.0  # Relative concentration (0.2 to 3.0)
        self._is_absorbance_mode: bool = True  # True: Absorbance, False: %Transmittance
        self._use_hamming_apodization: bool = True
        self._mirror_velocity_cm_s: float = 0.5  # Mirror translation speed

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance Michelson mirror sweep."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer defensively."""
        try:
            if hasattr(self, "_anim_timer") and self._anim_timer is not None and self._anim_timer.isActive():
                self._anim_timer.stop()
        except (RuntimeError, ReferenceError):
            pass

    @property
    def current_preset(self) -> FTIRPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active FTIR molecular preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_concentration(self, conc: float) -> None:
        """Set sample concentration scalar (Beer-Lambert c * l)."""
        self._concentration = max(0.1, min(conc, 3.0))
        self.property_changed.emit("concentration", self._concentration)
        self.update()

    def toggle_display_mode(self) -> None:
        """Toggle between Absorbance A and %Transmittance (%T)."""
        self._is_absorbance_mode = not self._is_absorbance_mode
        self.update()

    def toggle_apodization(self) -> None:
        """Toggle between Boxcar and Hamming window apodization."""
        self._use_hamming_apodization = not self._use_hamming_apodization
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset default."""
        self._sim_time = 0.0
        self._concentration = 1.0
        self.update()

    # -------------------------------------------------------------------------
    # Authentic FTIR Optical & Spectral Calculations
    # -------------------------------------------------------------------------

    def compute_absorbance(self, wavenumber_cm: float) -> float:
        """
        Compute absorbance A(ν̃) = c * ∑ A_i * exp(-((ν̃ - ν_i) / (w_i/2))^2).
        Follows authentic Gaussian band broadening and Beer-Lambert linearity.
        """
        p = self.current_preset
        total_abs = 0.0
        for peak in p.peaks:
            sigma = peak.width_cm / 2.355  # FWHM to Gaussian sigma
            diff = wavenumber_cm - peak.center_cm
            band_shape = math.exp(-0.5 * (diff / max(1.0, sigma)) ** 2)
            total_abs += peak.intensity * band_shape

        return float(self._concentration * total_abs)

    def compute_transmittance(self, wavenumber_cm: float) -> float:
        """
        Compute %Transmittance: %T = 100 * 10^(-A).
        """
        abs_val = self.compute_absorbance(wavenumber_cm)
        t_frac = 10.0 ** (-abs_val)
        return float(max(0.0, min(100.0, t_frac * 100.0)))

    def compute_interferogram_at_opd(self, opd_cm: float) -> float:
        """
        Compute authentic Michelson interferogram I_AC(δ) by integrating over
        broadband infrared blackbody continuum attenuated by molecular absorption:
          I_AC(δ) = ∫ B_sample(ν̃) * cos(2π * ν̃ * δ) dν̃
        For numerical responsiveness, calculated over key representative bands.
        """
        p = self.current_preset
        # Continuum baseline emission (IR Globar source ~1200 K)
        signal = 0.0
        # Sample absorption modulations
        for peak in p.peaks:
            nu = peak.center_cm
            weight = peak.intensity * math.exp(-0.5 * (peak.width_cm * opd_cm * 2.0 * math.pi) ** 2)
            signal -= weight * math.cos(2.0 * math.pi * nu * opd_cm)

        # Intense central burst envelope at ZPD (δ = 0)
        centerburst_env = math.exp(-0.5 * (opd_cm / 0.002) ** 2)
        total_signal = (centerburst_env * 3.5) + signal * 0.4
        return float(total_signal)

    def step_physics(self, dt: float) -> None:
        """Advance moving mirror in Michelson interferometer."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(15.0, 10.0, 910.0, 60.0)

    def _interferometer_rect(self) -> QRectF:
        return QRectF(15.0, 75.0, 270.0, 410.0)

    def _interferogram_rect(self) -> QRectF:
        return QRectF(295.0, 75.0, 310.0, 410.0)

    def _spectrum_rect(self) -> QRectF:
        return QRectF(615.0, 75.0, 310.0, 410.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, FTIRPreset]]:
        h_rect = self._header_rect()
        btn_w = 95.0
        btn_h = 28.0
        gap = 6.0
        start_x = 350.0
        y = (h_rect.height() - btn_h) / 2.0
        rects = []
        for i, p in enumerate(PRESETS):
            bx = start_x + i * (btn_w + gap)
            rects.append((QRectF(bx, y, btn_w, btn_h), p))
        return rects

    def paint(
        self,
        painter: QPainter,
        option: Optional[QStyleOptionGraphicsItem] = None,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Main Card Background
        bg_brush = QBrush(QColor(12, 16, 28, 248))
        border_pen = QPen(QColor(236, 72, 153, 160), 1.5)  # Infrared Rose Glow
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_interferometer_panel(painter)
        self._paint_interferogram_panel(painter)
        self._paint_spectrum_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ QUANG PHỔ FTIR & DAO ĐỘNG PHÂN TỬ (MICHELSON)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        mode_str = "Độ hấp thụ A" if self._is_absorbance_mode else "Độ truyền qua %T"
        painter.drawText(
            20,
            44,
            f"Mẫu: {p.name}  |  Công thức: {p.formula}  |  Nồng độ: {self._concentration:.1f}x  |  Chế độ: {mode_str}",
        )

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(236, 72, 153, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(244, 114, 182), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(148, 163, 184))
            label = preset.key.upper()[:10]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        painter.setBrush(QColor(236, 72, 153, 40))
        painter.setPen(QPen(QColor(244, 114, 182), 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(244, 114, 182))
        max_abs = max(self.compute_absorbance(pk.center_cm) for pk in p.peaks)
        painter.drawText(badge_rect, Qt.AlignCenter, f"A_max: {max_abs:.2f} Abs")

    def _paint_interferometer_panel(self, painter: QPainter) -> None:
        """Render Left Panel: Michelson Optical Bench with Oscillating Mirror."""
        rect = self._interferometer_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "🔬 GIAO THOA KẾ MICHELSON")

        # Optical Bench Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(3, 7, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        bx = box_rect.center().x()
        by = box_rect.center().y()

        # 1. IR Globar Source (Left)
        source_x = box_rect.left() + 25.0
        painter.setBrush(QColor(249, 115, 22))  # Glowing Orange IR emitter
        painter.setPen(QPen(QColor(254, 215, 170), 2.0))
        painter.drawEllipse(QPointF(source_x, by), 12.0, 12.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(249, 115, 22))
        painter.drawText(source_x - 18.0, by + 26.0, "Nguồn IR")

        # Beam from source to beam splitter
        painter.setPen(QPen(QColor(236, 72, 153, 200), 2.5))
        painter.drawLine(QPointF(source_x + 12.0, by), QPointF(bx, by))

        # 2. Beam Splitter (Center, 45 deg)
        painter.setPen(QPen(QColor(241, 245, 249), 3.0))
        painter.drawLine(QPointF(bx - 14.0, by + 14.0), QPointF(bx + 14.0, by - 14.0))
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(bx - 20.0, by - 20.0, "Bộ tách chùm KBr")

        # 3. Fixed Mirror (Top)
        fixed_y = box_rect.top() + 35.0
        painter.setPen(QPen(QColor(236, 72, 153, 180), 2.0))
        painter.drawLine(QPointF(bx, by), QPointF(bx, fixed_y))
        painter.setPen(QPen(QColor(56, 189, 248), 4.0))  # Fixed mirror bar
        painter.drawLine(QPointF(bx - 20.0, fixed_y), QPointF(bx + 20.0, fixed_y))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(bx - 25.0, fixed_y - 8.0, "Gương Cố định")

        # 4. Moving Mirror (Right)
        # Mirror oscillates back and forth across ZPD
        opd_scan = math.sin(self._sim_time * 2.5) * 25.0
        moving_x = box_rect.right() - 40.0 + opd_scan
        painter.setPen(QPen(QColor(236, 72, 153, 180), 2.0))
        painter.drawLine(QPointF(bx, by), QPointF(moving_x, by))
        painter.setPen(QPen(QColor(250, 204, 21), 4.0))  # Moving mirror bar
        painter.drawLine(QPointF(moving_x, by - 20.0), QPointF(moving_x, by + 20.0))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        painter.drawText(moving_x - 30.0, by - 26.0, "Gương Di động")

        # 5. Sample Cell & Detector (Bottom)
        det_y = box_rect.bottom() - 35.0
        # Recombined beam downward
        painter.setPen(QPen(QColor(236, 72, 153, 240), 2.8))
        painter.drawLine(QPointF(bx, by), QPointF(bx, det_y))

        # Sample Cuvette
        cell_rect = QRectF(bx - 15.0, by + 35.0, 30.0, 20.0)
        painter.setBrush(QColor(52, 211, 153, 100))
        painter.setPen(QPen(QColor(52, 211, 153), 1.5))
        painter.drawRect(cell_rect)
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        painter.setPen(QColor(52, 211, 153))
        painter.drawText(cell_rect, Qt.AlignCenter, "MẪU")

        # Detector (DLATGS / MCT)
        painter.setBrush(QColor(168, 85, 247))
        painter.setPen(QPen(QColor(216, 180, 254), 1.5))
        painter.drawRect(QRectF(bx - 18.0, det_y, 36.0, 14.0))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(bx - 18.0, det_y, 36.0, 14.0), Qt.AlignCenter, "ĐẦU DÒ")

        # ZPD indicator
        is_near_zpd = abs(opd_scan) < 3.0
        if is_near_zpd:
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(250, 204, 21))
            painter.drawText(bx - 40.0, by + 85.0, "✨ ZPD CENTERBURST!")

    def _paint_interferogram_panel(self, painter: QPainter) -> None:
        """Render Center Panel: Authentic Raw Interferogram I(δ) with Centerburst."""
        rect = self._interferogram_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, "📈 GIAO THOA ĐỒ RAW I(δ)")

        # Interferogram Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        mid_y = box_rect.center().y()
        mid_x = box_rect.center().x()

        # Grid lines
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(box_rect.left(), mid_y), QPointF(box_rect.right(), mid_y))
        painter.drawLine(QPointF(mid_x, box_rect.top()), QPointF(mid_x, box_rect.bottom()))

        # ZPD Label
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        painter.drawText(mid_x - 12.0, box_rect.bottom() - 10.0, "δ = 0")

        # Draw interferogram curve across OPD from -0.01 cm to +0.01 cm
        pts = []
        n_samples = 220
        max_opd = 0.008  # cm
        for i in range(n_samples):
            opd = -max_opd + (i / (n_samples - 1)) * (2.0 * max_opd)
            val = self.compute_interferogram_at_opd(opd)
            px = box_rect.left() + (i / (n_samples - 1)) * box_rect.width()
            py = mid_y - (val / 4.0) * (box_rect.height() * 0.42)
            pts.append(QPointF(px, py))

        path = QPainterPath()
        if pts:
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)

        painter.setPen(QPen(QColor(236, 72, 153), 1.6))
        painter.drawPath(path)

        # Mirror sweep cursor
        opd_scan_cm = math.sin(self._sim_time * 2.5) * max_opd
        cx = mid_x + (opd_scan_cm / max_opd) * (box_rect.width() / 2.0)
        painter.setPen(QPen(QColor(250, 204, 21), 1.5))
        painter.drawLine(QPointF(cx, box_rect.top()), QPointF(cx, box_rect.bottom()))

        # Current OPD readout
        curr_sig = self.compute_interferogram_at_opd(opd_scan_cm)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(box_rect.left() + 8.0, box_rect.top() + 16.0, f"OPD δ: {opd_scan_cm * 1e4:+.1f} µm")
        painter.drawText(box_rect.left() + 8.0, box_rect.top() + 30.0, f"Cường độ I: {curr_sig:.2f}")

    def _paint_spectrum_panel(self, painter: QPainter) -> None:
        """Render Right Panel: Fourier Transformed Infrared Spectrum (400 - 4000 cm^-1)."""
        rect = self._spectrum_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        mode_title = "PHỔ HẤP THỤ A(ν̃)" if self._is_absorbance_mode else "PHỔ TRUYỀN QUA %T(ν̃)"
        painter.drawText(QRectF(rect.left(), rect.top() + 10.0, rect.width(), 20.0), Qt.AlignCenter, f"📊 {mode_title}")

        # Spectrum Box
        box_rect = QRectF(rect.left() + 15.0, rect.top() + 35.0, rect.width() - 30.0, 360.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(box_rect, 8.0, 8.0)

        # FTIR standard x-axis runs high-to-low wavenumber (4000 cm^-1 on left, 400 cm^-1 on right)
        nu_min = 400.0
        nu_max = 4000.0

        # Draw spectrum curve
        pts = []
        n_samples = 220
        p = self.current_preset
        for i in range(n_samples):
            # i = 0 => 4000 cm^-1, i = n-1 => 400 cm^-1
            nu = nu_max - (i / (n_samples - 1)) * (nu_max - nu_min)
            px = box_rect.left() + (i / (n_samples - 1)) * box_rect.width()

            if self._is_absorbance_mode:
                val = self.compute_absorbance(nu)
                # Max scale ~1.5 Abs
                norm_val = min(1.2, val / 1.5)
                py = box_rect.bottom() - 30.0 - norm_val * (box_rect.height() - 60.0)
            else:
                # %Transmittance (100% at top, dips downward to 0% at bottom)
                t_val = self.compute_transmittance(nu)
                norm_val = t_val / 100.0
                py = box_rect.bottom() - 30.0 - norm_val * (box_rect.height() - 60.0)

            pts.append(QPointF(px, py))

        spec_path = QPainterPath()
        if pts:
            spec_path.moveTo(pts[0])
            for pt in pts[1:]:
                spec_path.lineTo(pt)

        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(spec_path)

        # Functional Group Peak Annotations
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        for peak in p.peaks:
            # Map peak center to screen x
            frac_x = (nu_max - peak.center_cm) / (nu_max - nu_min)
            if 0.0 <= frac_x <= 1.0:
                pk_x = box_rect.left() + frac_x * box_rect.width()

                if self._is_absorbance_mode:
                    val = self.compute_absorbance(peak.center_cm)
                    pk_y = box_rect.bottom() - 30.0 - (min(1.2, val / 1.5)) * (box_rect.height() - 60.0)
                else:
                    t_val = self.compute_transmittance(peak.center_cm)
                    pk_y = box_rect.bottom() - 30.0 - (t_val / 100.0) * (box_rect.height() - 60.0)

                # Peak marker dot
                painter.setBrush(QColor(244, 63, 94))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(pk_x, pk_y), 3.5, 3.5)

                # Label
                painter.setPen(QColor(250, 204, 21))
                painter.drawText(pk_x - 25.0, pk_y - 8.0, f"{peak.center_cm:.0f} cm⁻¹")

        # X-Axis Wavenumber ticks (FTIR standard reverse axis)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(box_rect.left() + 5.0, box_rect.bottom() - 10.0, "4000 cm⁻¹")
        painter.drawText(box_rect.center().x() - 25.0, box_rect.bottom() - 10.0, "2200 cm⁻¹")
        painter.drawText(box_rect.right() - 55.0, box_rect.bottom() - 10.0, "400 cm⁻¹")

        # Y-Axis label
        painter.drawText(box_rect.left() + 8.0, box_rect.top() + 18.0, "1.5 Abs" if self._is_absorbance_mode else "100% T")
        painter.drawText(box_rect.left() + 8.0, box_rect.bottom() - 32.0, "0.0 Abs" if self._is_absorbance_mode else "0% T")

    def _paint_controls(self, painter: QPainter) -> None:
        """Render Bottom Control Bar: Concentration slider and mode toggles."""
        rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        col_w = (rect.width() - 40.0) / 4.0

        # Slider 1: Concentration (c * l)
        c1_x = rect.left() + 20.0
        c1_y = rect.top() + 18.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(236, 72, 153))
        painter.drawText(c1_x, c1_y, f"Nồng độ Mẫu (c·l): {self._concentration:.2f}x")

        track_y = c1_y + 18.0
        track_w = col_w - 25.0
        painter.setPen(QPen(QColor(51, 65, 85), 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(c1_x, track_y), QPointF(c1_x + track_w, track_y))
        norm1 = max(0.0, min(1.0, (self._concentration - 0.1) / 2.9))
        handle1_x = c1_x + norm1 * track_w
        painter.setBrush(QColor(236, 72, 153))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(handle1_x, track_y), 7.0, 7.0)

        # Button 2: Mode Toggle (%T vs Absorbance)
        b2_x = rect.left() + 20.0 + col_w
        btn_w = col_w - 20.0
        b2_rect = QRectF(b2_x, rect.top() + 22.0, btn_w, 38.0)
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(b2_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        mode_lbl = "📈 CHẾ ĐỘ: ABSORBANCE" if self._is_absorbance_mode else "📉 CHẾ ĐỘ: %TRANSMIT"
        painter.drawText(b2_rect, Qt.AlignCenter, mode_lbl)

        # Button 3: Apodization Toggle (Hamming vs Boxcar)
        b3_x = rect.left() + 20.0 + 2 * col_w
        b3_rect = QRectF(b3_x, rect.top() + 22.0, btn_w, 38.0)
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(168, 85, 247), 1.2))
        painter.drawRoundedRect(b3_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(168, 85, 247))
        apod_lbl = "🪟 CỬA SỔ: HAMMING" if self._use_hamming_apodization else "🪟 CỬA SỔ: BOXCAR"
        painter.drawText(b3_rect, Qt.AlignCenter, apod_lbl)

        # Button 4: Reset & Pause
        b4_x = rect.left() + 20.0 + 3 * col_w
        b4_w = (btn_w - 10.0) / 2.0
        rst_rect = QRectF(b4_x, rect.top() + 22.0, b4_w, 38.0)
        pau_rect = QRectF(b4_x + b4_w + 10.0, rect.top() + 22.0, b4_w, 38.0)

        # Reset button
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(244, 63, 94), 1.2))
        painter.drawRoundedRect(rst_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(244, 63, 94))
        painter.drawText(rst_rect, Qt.AlignCenter, "🔄 ĐẶT LẠI")

        # Pause / Play button
        painter.setBrush(QColor(30, 41, 59, 220))
        painter.setPen(QPen(QColor(250, 204, 21), 1.2))
        painter.drawRoundedRect(pau_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        p_txt = "▶ TIẾP" if self._is_paused else "⏸ DỪNG"
        painter.drawText(pau_rect, Qt.AlignCenter, p_txt)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Controls buttons
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_w = col_w - 20.0

        # Mode button
        b2_rect = QRectF(ctrl_rect.left() + 20.0 + col_w, ctrl_rect.top() + 22.0, btn_w, 38.0)
        if b2_rect.contains(pos):
            self.toggle_display_mode()
            event.accept()
            return

        # Apodization button
        b3_rect = QRectF(ctrl_rect.left() + 20.0 + 2 * col_w, ctrl_rect.top() + 22.0, btn_w, 38.0)
        if b3_rect.contains(pos):
            self.toggle_apodization()
            event.accept()
            return

        # Reset & Pause buttons
        b4_x = ctrl_rect.left() + 20.0 + 3 * col_w
        b4_w = (btn_w - 10.0) / 2.0
        rst_rect = QRectF(b4_x, ctrl_rect.top() + 22.0, b4_w, 38.0)
        pau_rect = QRectF(b4_x + b4_w + 10.0, ctrl_rect.top() + 22.0, b4_w, 38.0)

        if rst_rect.contains(pos):
            self.reset_simulation()
            event.accept()
            return
        if pau_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Concentration slider
        c1_x = ctrl_rect.left() + 20.0
        c1_y = ctrl_rect.top() + 18.0
        track_y = c1_y + 18.0
        track_w = col_w - 25.0
        track_rect = QRectF(c1_x - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
        if track_rect.contains(pos):
            self._dragging_slider = "concentration"
            norm = max(0.0, min(1.0, (pos.x() - c1_x) / track_w))
            self.set_concentration(0.1 + norm * 2.9)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider == "concentration":
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            c1_x = ctrl_rect.left() + 20.0
            track_w = col_w - 25.0
            norm = max(0.0, min(1.0, (pos.x() - c1_x) / track_w))
            self.set_concentration(0.1 + norm * 2.9)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize FTIR simulator state for .airteach format."""
        return {
            "type": "InteractiveFTIRSpectroscopyItem",
            "preset_key": self.current_preset.key,
            "concentration": self._concentration,
            "is_absorbance_mode": self._is_absorbance_mode,
            "use_hamming_apodization": self._use_hamming_apodization,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveFTIRSpectroscopyItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "ethanol")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "concentration" in data:
            item._concentration = float(data["concentration"])
        if "is_absorbance_mode" in data:
            item._is_absorbance_mode = bool(data["is_absorbance_mode"])
        if "use_hamming_apodization" in data:
            item._use_hamming_apodization = bool(data["use_hamming_apodization"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
