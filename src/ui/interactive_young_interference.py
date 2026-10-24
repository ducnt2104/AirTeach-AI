"""
AirTeach AI - Interactive Wave Optics & Young's Double-Slit Interference / Grating Simulator
=============================================================================================
Provides an authentic wave optics simulation of light wave interference and diffraction:
  - Young's Double-Slit Geometry:
      - Wavelength λ ∈ [380 nm, 780 nm]
      - Slit separation a ∈ [0.1 mm, 2.0 mm]
      - Screen distance D ∈ [0.5 m, 3.0 m]
      - Slit width b ∈ [0.02 mm, 0.2 mm]
  - Fundamental Physical Equations:
      - Fringe spacing (Khoảng vân):
          i = (λ * D) / a
      - Geometric path difference (Hiệu đường đi):
          Δd = d₂ - d₁ ≈ (a * x) / D
      - Phase difference (Hiệu số pha):
          Δφ = 2π * Δd / λ = (2π * a * x) / (λ * D)
      - Position of Bright Fringes (Vân sáng, cực đại giao thoa):
          x_k = k * i = k * (λ * D) / a,  k ∈ ℤ
      - Position of Dark Fringes (Vân tối, cực tiểu giao thoa):
          x_k' = (k + 0.5) * i = (k + 0.5) * (λ * D) / a,  k ∈ ℤ
  - Physical Intensity Distribution (Multi-Slit & Single-Slit Diffraction Envelope):
      - Single-slit diffraction factor:
          α = π * b * x / (λ * D)
          I_diff(α) = sinc²(α) = [sin(α) / α]²
      - N-slit interference factor:
          β = 2π * a * x / (λ * D)
          I_interf(β) = [sin(N * β / 2) / sin(β / 2)]²
      - Total Normalized Intensity:
          I(x) = I_diff(α) * [1 / N²] * I_interf(β)
  - Color Physics:
      - Continuous spectrum conversion from wavelength λ (nm) to authentic sRGB color.
      - White light dispersion model: superposition of RGB spectral components
        revealing central white fringe and overlapping rainbow dispersion fringes!
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


def wavelength_to_rgb(wavelength_nm: float) -> tuple[int, int, int]:
    """
    Converts a wavelength in nm [380.0, 780.0] to authentic (R, G, B) color
    with human visual sensitivity falloff near spectral boundaries.
    """
    wl = max(380.0, min(780.0, float(wavelength_nm)))

    if wl < 440.0:
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif wl < 490.0:
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif wl < 510.0:
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif wl < 580.0:
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif wl < 645.0:
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # Vision sensitivity falloff at edges
    if wl < 420.0:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif wl > 700.0:
        factor = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0)
    else:
        factor = 1.0

    gamma = 0.80
    r_val = int(255 * ((max(0.0, r) * factor) ** gamma))
    g_val = int(255 * ((max(0.0, g) * factor) ** gamma))
    b_val = int(255 * ((max(0.0, b) * factor) ** gamma))

    return (max(0, min(255, r_val)), max(0, min(255, g_val)), max(0, min(255, b_val)))


class InteractiveYoungInterferenceItem(QGraphicsObject):
    """
    Interactive Young's Double-Slit & Grating Wave Optics Simulator item for QGraphicsScene.
    """

    optics_updated = Signal(float, float, float)  # (wavelength_nm, slit_a_mm, fringe_spacing_mm)
    preset_changed = Signal(str)                   # ("RED_HE_NE_LASER", etc.)
    probe_moved = Signal(float, float, float)      # (x_mm, order_k, intensity)

    def __init__(
        self,
        pos: tuple[float, float] = (300.0, 160.0),
        preset: str = "RED_HE_NE_LASER",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(38.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Dimensions
        self._width: float = 840.0
        self._height: float = 560.0

        # Physical optics parameters
        self._wavelength_nm: float = 632.8  # Red He-Ne standard
        self._slit_a_mm: float = 0.50       # Slit separation a [mm]
        self._slit_b_mm: float = 0.05       # Slit width b [mm]
        self._distance_D_m: float = 1.50    # Distance to screen D [m]
        self._num_slits: int = 2            # Number of slits N (2 for Young, >2 for grating)
        self._is_white_light: bool = False  # White light continuous dispersion toggle
        self._preset: str = preset

        # Interactive Measurement Crosshair Probe (in mm on observation screen)
        self._probe_x_mm: float = 0.0       # Probe position x on screen

        # Feature display toggles
        self._show_intensity_graph: bool = True
        self._show_probe: bool = True
        self._show_light_rays: bool = True

        # Left canvas layout
        self._canvas_rect = QRectF(15.0, 45.0, 500.0, 500.0)

        # Interaction drag state
        self._drag_target: Optional[str] = None  # "TITLE", "PROBE", "SLIDER_WL", "SLIDER_A", "SLIDER_D"
        self._drag_offset: QPointF = QPointF(0.0, 0.0)
        self._hover_target: Optional[str] = None

        # Apply initial preset
        self._apply_preset(self._preset)

    # ── Physical Calculations ─────────────────────────────────────────────────

    @property
    def fringe_spacing_mm(self) -> float:
        """Calculates fringe spacing i = (lambda * D) / a in millimeters."""
        if self._slit_a_mm <= 0.0:
            return 0.0
        # lambda in nm (1e-6 mm), D in m (1e3 mm), a in mm
        # i = (lambda_nm * 1e-6 * D_m * 1e3) / a_mm = (lambda_nm * D_m * 1e-3) / a_mm
        return (self._wavelength_nm * self._distance_D_m * 1e-3) / self._slit_a_mm

    def calculate_intensity(self, x_mm: float, wl_nm: Optional[float] = None) -> float:
        """
        Calculates normalized physical light intensity I(x) ∈ [0.0, 1.0] at position x_mm.
        Takes into account both N-slit interference and single-slit diffraction envelope.
        """
        wl = wl_nm if wl_nm is not None else self._wavelength_nm
        if wl <= 0.0 or self._distance_D_m <= 0.0 or self._slit_a_mm <= 0.0:
            return 0.0

        # Angles and phase terms:
        # lambda in mm: wl_mm = wl * 1e-6
        # D in mm: D_mm = self._distance_D_m * 1e3
        wl_mm = wl * 1e-6
        D_mm = self._distance_D_m * 1e3

        # Single-slit diffraction factor: alpha = pi * b * x / (lambda * D)
        alpha = (math.pi * self._slit_b_mm * x_mm) / (wl_mm * D_mm)
        if abs(alpha) < 1e-7:
            diff_factor = 1.0
        else:
            diff_factor = (math.sin(alpha) / alpha) ** 2

        # N-slit interference factor: beta = 2 * pi * a * x / (lambda * D)
        beta = (2.0 * math.pi * self._slit_a_mm * x_mm) / (wl_mm * D_mm)
        half_beta = beta / 2.0

        if self._num_slits == 2:
            # 2-slit interference: cos²(beta / 2)
            interf_factor = (math.cos(half_beta)) ** 2
        else:
            # General N-slit grating interference: [sin(N * beta / 2) / (N * sin(beta / 2))]²
            sin_half = math.sin(half_beta)
            if abs(sin_half) < 1e-6:
                interf_factor = 1.0
            else:
                interf_factor = (math.sin(self._num_slits * half_beta) / (self._num_slits * sin_half)) ** 2

        return max(0.0, min(1.0, diff_factor * interf_factor))

    # ── Bounding Rect & Painting ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Paints complete Wave Optics & Interference simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Container
        self._paint_outer_container(painter)

        # 2. Left Optical Bench & Interference Patterns
        self._paint_optics_canvas(painter)

        # 3. Right Control & Theory HUD
        self._paint_controls_and_hud(painter)

        painter.restore()

    def _paint_outer_container(self, painter: QPainter) -> None:
        """Paints outer frosted glass card and top title bar."""
        outer = self.boundingRect()

        # Slate-900 background
        painter.setPen(QPen(QColor("#334155"), 1.8))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))
        painter.drawRoundedRect(outer, 14.0, 14.0)

        # Title bar
        title_rect = QRectF(0.0, 0.0, self._width, 40.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 230)))
        painter.drawRoundedRect(title_rect, 14.0, 14.0)
        painter.drawRect(QRectF(0.0, 26.0, self._width, 14.0))

        # Title Text
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(18.0, 8.0, 520.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "🌈 QUANG HỌC SÓNG: GIAO THOA KHE YOUNG & CÁCH TỬ NHIỄU XẠ",
        )

        # Preset badge
        preset_labels = {
            "RED_HE_NE_LASER": "🔴 Laser He-Ne 633nm",
            "GREEN_DPSS_LASER": "🟢 Laser DPSS 532nm",
            "VIOLET_LASER": "🟣 Laser Tím 405nm",
            "WHITE_LIGHT_RAINBOW": "🌈 Tán Sắc Ánh Sáng Trắng",
            "DIFFRACTION_GRATING": "⚡ Cách Tử Nhiễu Xạ N=5",
        }
        badge = preset_labels.get(self._preset, "Giao Thoa Ánh Sáng")
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(
            QRectF(520.0, 8.0, 260.0, 24.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            badge,
        )

        # Close button
        c_btn = self._get_close_button_rect()
        h_close = (self._hover_target == "CLOSE")
        painter.setBrush(QBrush(QColor("#dc2626") if h_close else QColor(239, 68, 68, 160)))
        painter.setPen(QPen(QColor("#fca5a5"), 1.2))
        painter.drawRoundedRect(c_btn, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(c_btn, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_optics_canvas(self, painter: QPainter) -> None:
        """Paints optical bench diagram (top) and photorealistic interference screen & graph (bottom)."""
        painter.save()
        painter.setClipRect(self._canvas_rect)

        # Deep cosmic viewport
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.setBrush(QBrush(QColor("#030712")))
        painter.drawRoundedRect(self._canvas_rect, 8.0, 8.0)

        # ── 1. Top Half: Optical Bench Apparatus ──
        self._paint_optical_apparatus(painter)

        # ── 2. Bottom Half: Interference Screen & Intensity Graph ──
        self._paint_interference_screen(painter)

        painter.restore()

    def _paint_optical_apparatus(self, painter: QPainter) -> None:
        """Paints laser source, double-slit barrier, light rays, and screen."""
        laser_color = self._get_source_color()

        # Laser Emitter Box
        box_rect = QRectF(30.0, 110.0, 65.0, 50.0)
        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(box_rect, 6.0, 6.0)

        # Laser Aperture Nozzle
        nozzle = QRectF(95.0, 126.0, 12.0, 18.0)
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRect(nozzle)

        # Laser Label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(laser_color))
        lbl = "LASER" if not self._is_white_light else "WHITE"
        painter.drawText(box_rect, Qt.AlignmentFlag.AlignCenter, lbl)

        # Double-Slit Diaphragm Barrier
        barrier_x = 210.0
        barrier_y = 65.0
        barrier_h = 140.0
        painter.setPen(QPen(QColor("#64748b"), 2.0))
        painter.drawLine(QPointF(barrier_x, barrier_y), QPointF(barrier_x, barrier_y + 55.0))
        painter.drawLine(QPointF(barrier_x, barrier_y + 85.0), QPointF(barrier_x, barrier_y + barrier_h))

        # Slit dots/openings indicator
        slit_gap = 14.0
        s1_y = 135.0 - slit_gap / 2.0
        s2_y = 135.0 + slit_gap / 2.0
        painter.setPen(QPen(QColor("#94a3b8"), 1.0))
        painter.drawLine(QPointF(barrier_x - 3.0, s1_y), QPointF(barrier_x + 3.0, s1_y))
        painter.drawLine(QPointF(barrier_x - 3.0, s2_y), QPointF(barrier_x + 3.0, s2_y))

        # Slit Labels: S1, S2, a
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QPointF(barrier_x - 22.0, s1_y - 2.0), "S₁")
        painter.drawText(QPointF(barrier_x - 22.0, s2_y + 10.0), "S₂")
        painter.drawText(QPointF(barrier_x - 4.0, barrier_y - 6.0), f"a={self._slit_a_mm:.2f}mm")

        # Laser incident beam to slits
        pen_beam = QPen(laser_color, 1.8)
        painter.setPen(pen_beam)
        painter.drawLine(QPointF(107.0, 135.0), QPointF(barrier_x, 135.0))

        # Target Observation Screen line (at right side of apparatus)
        screen_x = 475.0
        painter.setPen(QPen(QColor("#cbd5e1"), 2.5))
        painter.drawLine(QPointF(screen_x, barrier_y), QPointF(screen_x, barrier_y + barrier_h))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(QPointF(screen_x - 12.0, barrier_y - 6.0), "Màn E")

        # Rays from S1 and S2 to screen
        if self._show_light_rays:
            ray_pen = QPen(QColor(laser_color.red(), laser_color.green(), laser_color.blue(), 100), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(ray_pen)
            # Central ray (x = 0)
            center_y = 135.0
            painter.drawLine(QPointF(barrier_x, s1_y), QPointF(screen_x, center_y))
            painter.drawLine(QPointF(barrier_x, s2_y), QPointF(screen_x, center_y))

            # Rays to probe position if active
            probe_screen_y = center_y + (self._probe_x_mm * 12.0)
            if barrier_y <= probe_screen_y <= barrier_y + barrier_h:
                probe_pen = QPen(QColor("#38bdf8"), 1.2, Qt.PenStyle.DotLine)
                painter.setPen(probe_pen)
                painter.drawLine(QPointF(barrier_x, s1_y), QPointF(screen_x, probe_screen_y))
                painter.drawLine(QPointF(barrier_x, s2_y), QPointF(screen_x, probe_screen_y))

        # Distance D dimension line
        painter.setPen(QPen(QColor("#94a3b8"), 1.0, Qt.PenStyle.DotLine))
        dim_y = barrier_y + barrier_h + 10.0
        painter.drawLine(QPointF(barrier_x, dim_y), QPointF(screen_x, dim_y))
        painter.drawText(QPointF((barrier_x + screen_x) / 2.0 - 20.0, dim_y + 14.0), f"D = {self._distance_D_m:.2f} m")

    def _paint_interference_screen(self, painter: QPainter) -> None:
        """Paints photorealistic interference fringe strip and intensity curve I(x)."""
        strip_x = 30.0
        strip_y = 440.0
        strip_w = 470.0
        strip_h = 42.0

        graph_y = 230.0
        graph_h = 175.0

        # Screen coordinates mapping:
        # Center x = 0 is at strip_x + strip_w / 2.0
        cx = strip_x + strip_w / 2.0
        # Scale: pixels per millimeter on observation screen
        px_per_mm = 25.0

        # ── 1. Photorealistic Optical Screen Strip ──
        strip_rect = QRectF(strip_x, strip_y, strip_w, strip_h)
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor("#020617")))
        painter.drawRoundedRect(strip_rect, 6.0, 6.0)

        # Render fringes across the strip width
        strip_steps = int(strip_w)
        for px_col in range(strip_steps):
            x_screen_mm = (px_col - strip_w / 2.0) / px_per_mm
            col_x = strip_x + px_col

            if self._is_white_light:
                # White light synthesis: sum RGB intensities at respective wavelengths
                i_red = self.calculate_intensity(x_screen_mm, 650.0)
                i_green = self.calculate_intensity(x_screen_mm, 532.0)
                i_blue = self.calculate_intensity(x_screen_mm, 450.0)
                col_color = QColor(
                    min(255, int(255 * i_red)),
                    min(255, int(255 * i_green)),
                    min(255, int(255 * i_blue)),
                )
            else:
                intensity = self.calculate_intensity(x_screen_mm)
                src_col = self._get_source_color()
                col_color = QColor(
                    int(src_col.red() * intensity),
                    int(src_col.green() * intensity),
                    int(src_col.blue() * intensity),
                )

            painter.setPen(QPen(col_color, 1.0))
            painter.drawLine(QPointF(col_x, strip_y + 2.0), QPointF(col_x, strip_y + strip_h - 2.0))

        # Screen border
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(strip_rect, 6.0, 6.0)

        # Screen Label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(strip_x + 8.0, strip_y + 4.0, 200.0, 16.0), Qt.AlignmentFlag.AlignLeft, "Màn Quan Sát Thực Tế (Screen)")

        # ── 2. Intensity Curve I(x) ──
        if self._show_intensity_graph:
            # Baseline & zero line
            base_y = graph_y + graph_h - 10.0
            top_y = graph_y + 15.0

            painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(strip_x, base_y), QPointF(strip_x + strip_w, base_y))
            painter.drawLine(QPointF(cx, top_y - 5.0), QPointF(cx, base_y + 5.0))

            # Draw I(x) curve path
            curve_path = QPainterPath()
            first = True
            step_count = 200
            for i_step in range(step_count + 1):
                px_pos = strip_x + (i_step / step_count) * strip_w
                x_mm = (px_pos - cx) / px_per_mm
                if self._is_white_light:
                    intens = (
                        self.calculate_intensity(x_mm, 650.0)
                        + self.calculate_intensity(x_mm, 532.0)
                        + self.calculate_intensity(x_mm, 450.0)
                    ) / 3.0
                else:
                    intens = self.calculate_intensity(x_mm)

                py = base_y - intens * (graph_h - 30.0)
                if first:
                    curve_path.moveTo(px_pos, py)
                    first = False
                else:
                    curve_path.lineTo(px_pos, py)

            curve_color = self._get_source_color() if not self._is_white_light else QColor("#f8fafc")
            painter.setPen(QPen(curve_color, 2.0))
            painter.drawPath(curve_path)

            # Intensity axis labels
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(QPointF(strip_x + 6.0, top_y + 8.0), "I = I₀")
            painter.drawText(QPointF(strip_x + 6.0, base_y - 4.0), "I = 0")
            painter.drawText(QPointF(cx + 4.0, base_y + 14.0), "x = 0 (Vân sáng trung tâm k=0)")

            # Order Markers k = ±1, ±2 on screen
            i_val = self.fringe_spacing_mm
            if i_val > 0.1:
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                for k in [-3, -2, -1, 1, 2, 3]:
                    kx = cx + k * i_val * px_per_mm
                    if strip_x <= kx <= strip_x + strip_w:
                        painter.setPen(QPen(QColor("#e2e8f0")))
                        painter.drawText(QPointF(kx - 10.0, strip_y + strip_h + 14.0), f"k={k:+d}")

        # ── 3. Interactive Measurement Probe Crosshair ──
        if self._show_probe:
            probe_px = cx + self._probe_x_mm * px_per_mm
            if strip_x <= probe_px <= strip_x + strip_w:
                # Vertical guide line
                painter.setPen(QPen(QColor("#38bdf8"), 1.5, Qt.PenStyle.DashDotLine))
                painter.drawLine(QPointF(probe_px, graph_y), QPointF(probe_px, strip_y + strip_h))

                # Handle bead on the screen strip
                probe_y = strip_y + strip_h / 2.0
                painter.setPen(QPen(QColor("#ffffff"), 1.8))
                painter.setBrush(QBrush(QColor("#0284c7")))
                painter.drawEllipse(QPointF(probe_px, probe_y), 6.5, 6.5)

                # Probe Readout Badge
                i_val = self.fringe_spacing_mm
                k_val = self._probe_x_mm / i_val if i_val > 0.0 else 0.0
                intens_probe = self.calculate_intensity(self._probe_x_mm)
                badge_txt = f"x = {self._probe_x_mm:+.2f} mm | k = {k_val:+.2f} | I = {intens_probe:.2f}I₀"

                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.setPen(QPen(QColor("#38bdf8")))
                painter.drawText(QPointF(probe_px - 40.0, graph_y - 6.0), badge_txt)

    def _get_source_color(self) -> QColor:
        """Returns QColor corresponding to active optical source."""
        if self._is_white_light:
            return QColor("#f8fafc")
        r, g, b = wavelength_to_rgb(self._wavelength_nm)
        return QColor(r, g, b)

    def _paint_controls_and_hud(self, painter: QPainter) -> None:
        """Paints parameter sliders, theory metrics, presets, and action buttons."""
        hud_left = 525.0
        hud_w = 300.0

        # ── 1. Optical Parameters Card (Sliders) ──
        card1_rect = QRectF(hud_left, 45.0, hud_w, 140.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card1_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 50.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "THÔNG SỐ QUANG HỌC KHE YOUNG",
        )

        # Wavelength Row
        src_col = self._get_source_color()
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(src_col))
        wl_txt = f"λ = {self._wavelength_nm:.1f} nm" if not self._is_white_light else "λ = Ánh Sáng Trắng (400-750nm)"
        painter.drawText(QRectF(hud_left + 12.0, 68.0, hud_w - 24.0, 18.0), Qt.AlignmentFlag.AlignLeft, wl_txt)

        # Wavelength Slider
        wl_track, wl_handle = self._get_wl_slider_geometry()
        # Draw rainbow gradient along the wavelength track
        grad = QLinearGradient(wl_track.topLeft(), wl_track.topRight())
        for stop_wl in [380, 440, 490, 510, 580, 645, 780]:
            r, g, b = wavelength_to_rgb(stop_wl)
            frac = (stop_wl - 380) / (780 - 380)
            grad.setColorAt(frac, QColor(r, g, b))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(wl_track, 4.0, 4.0)

        # Wavelength Handle
        hover_wl = (self._hover_target == "SLIDER_WL")
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(src_col if not hover_wl else QColor("#ffffff")))
        painter.drawEllipse(wl_handle)

        # Slit separation a row
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(hud_left + 12.0, 102.0, 140.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Khoảng cách a = {self._slit_a_mm:.2f} mm")
        painter.drawText(QRectF(hud_left + 160.0, 102.0, 120.0, 16.0), Qt.AlignmentFlag.AlignLeft, f"Khoảng cách D = {self._distance_D_m:.2f} m")

        # Slit a Slider
        a_track, a_handle = self._get_a_slider_geometry()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(a_track, 3.0, 3.0)
        filled_a = QRectF(a_track.left(), a_track.top(), a_handle.center().x() - a_track.left(), a_track.height())
        painter.setBrush(QBrush(QColor("#0ea5e9")))
        painter.drawRoundedRect(filled_a, 3.0, 3.0)
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.drawEllipse(a_handle)

        # Distance D Slider
        d_track, d_handle = self._get_d_slider_geometry()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRoundedRect(d_track, 3.0, 3.0)
        filled_d = QRectF(d_track.left(), d_track.top(), d_handle.center().x() - d_track.left(), d_track.height())
        painter.setBrush(QBrush(QColor("#10b981")))
        painter.drawRoundedRect(filled_d, 3.0, 3.0)
        painter.setPen(QPen(QColor("#34d399"), 1.2))
        painter.setBrush(QBrush(QColor("#059669")))
        painter.drawEllipse(d_handle)

        # Multi-slit count row: [2 khe], [3 khe], [5 khe], [10 khe]
        slit_counts = [(2, "2 Khe"), (3, "3 Khe"), (5, "5 Khe"), (10, "Cách Tử 10")]
        btn_w = (hud_w - 24.0 - 18.0) / 4.0
        for idx, (cnt, lbl_cnt) in enumerate(slit_counts):
            btn_r = QRectF(hud_left + 12.0 + idx * (btn_w + 6.0), 146.0, btn_w, 24.0)
            is_active = (self._num_slits == cnt)
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(14, 165, 233, 60) if is_active else QColor(30, 41, 59, 160)))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#cbd5e1")))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl_cnt)

        # ── 2. Theoretical Optics Metrics Card ──
        card2_rect = QRectF(hud_left, 192.0, hud_w, 116.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card2_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 196.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "KẾT QUẢ GIAO THOA & VỊ TRÍ VÂN",
        )

        i_val = self.fringe_spacing_mm
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(
            QRectF(hud_left + 12.0, 216.0, hud_w - 24.0, 22.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Khoảng vân: i = (λ·D)/a = {i_val:.3f} mm",
        )

        # Probe statistics
        k_val = self._probe_x_mm / i_val if i_val > 0.0 else 0.0
        delta_d_nm = (self._slit_a_mm * abs(self._probe_x_mm) / self._distance_D_m) * 1e3
        intens_p = self.calculate_intensity(self._probe_x_mm)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(
            QRectF(hud_left + 12.0, 240.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"• Tọa độ đầu đo: x = {self._probe_x_mm:+.2f} mm (Bậc k = {k_val:+.2f})",
        )
        painter.drawText(
            QRectF(hud_left + 12.0, 258.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"• Hiệu đường đi: Δd = |d₂ - d₁| = {delta_d_nm:.1f} nm ({delta_d_nm / self._wavelength_nm:.2f}λ)",
        )
        painter.drawText(
            QRectF(hud_left + 12.0, 276.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"• Cường độ sáng: I/I₀ = {intens_p:.3f} ({'Cực đại' if intens_p > 0.95 else ('Cực tiểu' if intens_p < 0.05 else 'Trung gian')})",
        )

        # ── 3. Presets (2x2 + 1) ──
        card3_rect = QRectF(hud_left, 315.0, hud_w, 102.0)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(30, 41, 59, 210)))
        painter.drawRoundedRect(card3_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            QRectF(hud_left + 12.0, 318.0, hud_w - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "BÀI THỰC HÀNH QUANG HỌC",
        )

        presets = [
            ("RED_HE_NE_LASER", "🔴 He-Ne 633nm", 0, 0),
            ("GREEN_DPSS_LASER", "🟢 DPSS 532nm", 0, 1),
            ("VIOLET_LASER", "🟣 Violet 405nm", 1, 0),
            ("WHITE_LIGHT_RAINBOW", "🌈 Ánh Sáng Trắng", 1, 1),
        ]
        for p_id, p_label, row, col in presets:
            btn_r = self._get_preset_button_rect(row, col)
            is_active = (self._preset == p_id)
            is_hover = (self._hover_target == f"PRESET_{p_id}")
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(14, 165, 233, 60) if is_active else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 160))))
            painter.drawRoundedRect(btn_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#e2e8f0")))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, p_label)

        # ── 4. Visual Toggles ──
        toggles = [
            ("GRAPH", "📉 Đồ Thị I", self._show_intensity_graph, 0),
            ("PROBE", "🎯 Đầu Đo x", self._show_probe, 1),
            ("RAYS", "📏 Tia Sáng", self._show_light_rays, 2),
        ]
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        for tog_key, tog_label, is_on, idx in toggles:
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 424.0, tog_w, 28.0)
            is_hover = (self._hover_target == f"TOGGLE_{tog_key}")
            painter.setPen(QPen(QColor("#10b981") if is_on else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(16, 185, 129, 60) if is_on else (QColor(51, 65, 85, 200) if is_hover else QColor(30, 41, 59, 140))))
            painter.drawRoundedRect(tog_r, 5.0, 5.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#34d399") if is_on else QColor("#94a3b8")))
            painter.drawText(tog_r, Qt.AlignmentFlag.AlignCenter, tog_label)

        # ── 5. Action Buttons (Stamp & Close) ──
        stamp_btn = self._get_stamp_button_rect()
        h_stamp = (self._hover_target == "STAMP")
        painter.setPen(QPen(QColor("#0ea5e9"), 1.5))
        painter.setBrush(QBrush(QColor(14, 165, 233, 210) if h_stamp else QColor(2, 132, 199, 180)))
        painter.drawRoundedRect(stamp_btn, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(stamp_btn, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng Dấu Vector")

        close_b = self._get_close_bottom_button_rect()
        h_cb = (self._hover_target == "CLOSE_BOTTOM")
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(71, 85, 105, 200) if h_cb else QColor(51, 65, 85, 160)))
        painter.drawRoundedRect(close_b, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(close_b, Qt.AlignmentFlag.AlignCenter, "Đóng")

    # ── Geometry & Layout Helpers ─────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 8.0, 24.0, 24.0)

    def _get_wl_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 88.0, 276.0, 8.0)
        frac = (self._wavelength_nm - 380.0) / (780.0 - 380.0)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 6.0, hy - 6.0, 12.0, 12.0)

    def _get_a_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 12.0, 122.0, 130.0, 6.0)
        frac = (self._slit_a_mm - 0.10) / (2.00 - 0.10)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_d_slider_geometry(self) -> tuple[QRectF, QRectF]:
        hud_left = 525.0
        track = QRectF(hud_left + 160.0, 122.0, 128.0, 6.0)
        frac = (self._distance_D_m - 0.50) / (3.00 - 0.50)
        hx = track.left() + max(0.0, min(1.0, frac)) * track.width()
        hy = track.center().y()
        return track, QRectF(hx - 5.0, hy - 5.0, 10.0, 10.0)

    def _get_preset_button_rect(self, row: int, col: int) -> QRectF:
        hud_left = 525.0
        w = 134.0
        h = 28.0
        x = hud_left + 12.0 + col * (w + 8.0)
        y = 338.0 + row * (h + 8.0)
        return QRectF(x, y, w, h)

    def _get_stamp_button_rect(self) -> QRectF:
        return QRectF(525.0 + 12.0, 495.0, 175.0, 36.0)

    def _get_close_bottom_button_rect(self) -> QRectF:
        return QRectF(525.0 + 195.0, 495.0, 93.0, 36.0)

    # ── Presets & Physical Setters ────────────────────────────────────────────

    def set_wavelength(self, wl_nm: float) -> None:
        """Sets wavelength in nanometers [380, 780] and disables white light mode."""
        self._wavelength_nm = max(380.0, min(780.0, float(wl_nm)))
        self._is_white_light = False
        self.optics_updated.emit(self._wavelength_nm, self._slit_a_mm, self.fringe_spacing_mm)
        self.update()

    def set_slit_separation(self, a_mm: float) -> None:
        """Sets slit separation a in millimeters [0.1, 2.0]."""
        self._slit_a_mm = max(0.10, min(2.00, float(a_mm)))
        self.optics_updated.emit(self._wavelength_nm, self._slit_a_mm, self.fringe_spacing_mm)
        self.update()

    def set_distance_D(self, d_m: float) -> None:
        """Sets distance from slits to screen D in meters [0.5, 3.0]."""
        self._distance_D_m = max(0.50, min(3.00, float(d_m)))
        self.optics_updated.emit(self._wavelength_nm, self._slit_a_mm, self.fringe_spacing_mm)
        self.update()

    def set_num_slits(self, n: int) -> None:
        """Sets number of slits N (e.g. 2, 3, 5, 10)."""
        self._num_slits = max(2, int(n))
        self.update()

    def set_probe_x(self, x_mm: float) -> None:
        """Sets measurement crosshair probe position on screen in mm."""
        self._probe_x_mm = max(-9.0, min(9.0, float(x_mm)))
        i_val = self.fringe_spacing_mm
        k_val = self._probe_x_mm / i_val if i_val > 0.0 else 0.0
        self.probe_moved.emit(self._probe_x_mm, k_val, self.calculate_intensity(self._probe_x_mm))
        self.update()

    def set_preset(self, preset_name: str) -> None:
        """Switches through authentic curriculum optics experiments."""
        self._preset = preset_name
        self._apply_preset(preset_name)
        self.preset_changed.emit(preset_name)
        self.update()

    def _apply_preset(self, preset: str) -> None:
        """Configures optical parameters for curriculum preset."""
        p_up = preset.upper()
        if p_up == "RED_HE_NE_LASER":
            self._wavelength_nm = 632.8
            self._slit_a_mm = 0.50
            self._distance_D_m = 1.50
            self._num_slits = 2
            self._is_white_light = False
        elif p_up == "GREEN_DPSS_LASER":
            self._wavelength_nm = 532.0
            self._slit_a_mm = 0.50
            self._distance_D_m = 1.50
            self._num_slits = 2
            self._is_white_light = False
        elif p_up == "VIOLET_LASER":
            self._wavelength_nm = 405.0
            self._slit_a_mm = 0.50
            self._distance_D_m = 1.50
            self._num_slits = 2
            self._is_white_light = False
        elif p_up == "WHITE_LIGHT_RAINBOW":
            self._is_white_light = True
            self._slit_a_mm = 0.40
            self._distance_D_m = 1.50
            self._num_slits = 2
        elif p_up == "DIFFRACTION_GRATING":
            self._wavelength_nm = 532.0
            self._slit_a_mm = 0.25
            self._distance_D_m = 1.50
            self._num_slits = 5
            self._is_white_light = False

        self._probe_x_mm = 0.0
        self.optics_updated.emit(self._wavelength_nm, self._slit_a_mm, self.fringe_spacing_mm)

    # ── Mouse & Air Gesture Interaction ───────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Close
        if self._get_close_button_rect().contains(pos) or self._get_close_bottom_button_rect().contains(pos):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_button_rect().contains(pos):
            self.stamp_to_board()
            event.accept()
            return

        # Presets
        presets = [
            ("RED_HE_NE_LASER", 0, 0),
            ("GREEN_DPSS_LASER", 0, 1),
            ("VIOLET_LASER", 1, 0),
            ("WHITE_LIGHT_RAINBOW", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(pos):
                self.set_preset(p_id)
                event.accept()
                return

        # Slit count buttons [2, 3, 5, 10]
        hud_left = 525.0
        hud_w = 300.0
        btn_w = (hud_w - 24.0 - 18.0) / 4.0
        for idx, (cnt, _) in enumerate([(2, "2"), (3, "3"), (5, "5"), (10, "10")]):
            btn_r = QRectF(hud_left + 12.0 + idx * (btn_w + 6.0), 146.0, btn_w, 24.0)
            if btn_r.contains(pos):
                self.set_num_slits(cnt)
                event.accept()
                return

        # Visual toggles
        tog_w = (hud_w - 24.0 - 16.0) / 3.0
        for idx, attr in enumerate(["_show_intensity_graph", "_show_probe", "_show_light_rays"]):
            tog_r = QRectF(hud_left + 12.0 + idx * (tog_w + 8.0), 424.0, tog_w, 28.0)
            if tog_r.contains(pos):
                setattr(self, attr, not getattr(self, attr))
                self.update()
                event.accept()
                return

        # Sliders
        wl_track, wl_handle = self._get_wl_slider_geometry()
        if wl_track.contains(pos) or wl_handle.contains(pos):
            self._drag_target = "SLIDER_WL"
            frac = (pos.x() - wl_track.left()) / wl_track.width()
            self.set_wavelength(380.0 + frac * (780.0 - 380.0))
            event.accept()
            return

        a_track, a_handle = self._get_a_slider_geometry()
        if a_track.contains(pos) or a_handle.contains(pos):
            self._drag_target = "SLIDER_A"
            frac = (pos.x() - a_track.left()) / a_track.width()
            self.set_slit_separation(0.10 + frac * (2.00 - 0.10))
            event.accept()
            return

        d_track, d_handle = self._get_d_slider_geometry()
        if d_track.contains(pos) or d_handle.contains(pos):
            self._drag_target = "SLIDER_D"
            frac = (pos.x() - d_track.left()) / d_track.width()
            self.set_distance_D(0.50 + frac * (3.00 - 0.50))
            event.accept()
            return

        # Interactive Probe on screen strip
        strip_x = 30.0
        strip_y = 230.0
        strip_w = 470.0
        strip_h = 252.0
        if QRectF(strip_x, strip_y, strip_w, strip_h).contains(pos):
            self._drag_target = "PROBE"
            cx = strip_x + strip_w / 2.0
            px_per_mm = 25.0
            self.set_probe_x((pos.x() - cx) / px_per_mm)
            event.accept()
            return

        # Title bar dragging
        if QRectF(0.0, 0.0, self._width, 40.0).contains(pos):
            self._drag_target = "TITLE"
            self._drag_offset = event.scenePos() - self.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        if self._drag_target == "TITLE":
            self.setPos(event.scenePos() - self._drag_offset)
            event.accept()
            return
        elif self._drag_target == "SLIDER_WL":
            wl_track, _ = self._get_wl_slider_geometry()
            frac = (pos.x() - wl_track.left()) / wl_track.width()
            self.set_wavelength(380.0 + frac * (780.0 - 380.0))
            event.accept()
            return
        elif self._drag_target == "SLIDER_A":
            a_track, _ = self._get_a_slider_geometry()
            frac = (pos.x() - a_track.left()) / a_track.width()
            self.set_slit_separation(0.10 + frac * (2.00 - 0.10))
            event.accept()
            return
        elif self._drag_target == "SLIDER_D":
            d_track, _ = self._get_d_slider_geometry()
            frac = (pos.x() - d_track.left()) / d_track.width()
            self.set_distance_D(0.50 + frac * (3.00 - 0.50))
            event.accept()
            return
        elif self._drag_target == "PROBE":
            cx = 30.0 + 470.0 / 2.0
            px_per_mm = 25.0
            self.set_probe_x((pos.x() - cx) / px_per_mm)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._drag_target = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old = self._hover_target

        if self._get_close_button_rect().contains(pos):
            self._hover_target = "CLOSE"
        elif self._get_close_bottom_button_rect().contains(pos):
            self._hover_target = "CLOSE_BOTTOM"
        elif self._get_stamp_button_rect().contains(pos):
            self._hover_target = "STAMP"
        elif self._get_wl_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_WL"
        elif self._get_a_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_A"
        elif self._get_d_slider_geometry()[1].contains(pos):
            self._hover_target = "SLIDER_D"
        else:
            self._hover_target = None
            presets = [
                ("RED_HE_NE_LASER", 0, 0),
                ("GREEN_DPSS_LASER", 0, 1),
                ("VIOLET_LASER", 1, 0),
                ("WHITE_LIGHT_RAINBOW", 1, 1),
            ]
            for p_id, r, c in presets:
                if self._get_preset_button_rect(r, c).contains(pos):
                    self._hover_target = f"PRESET_{p_id}"
                    break

        if self._hover_target != old:
            self.update()

    # ── Touchless Air Gesture API ─────────────────────────────────────────────

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        """Handles touchless air pinch gesture for dragging sliders or selecting presets."""
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_button_rect().contains(loc) or self._get_close_bottom_button_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Presets
        presets = [
            ("RED_HE_NE_LASER", 0, 0),
            ("GREEN_DPSS_LASER", 0, 1),
            ("VIOLET_LASER", 1, 0),
            ("WHITE_LIGHT_RAINBOW", 1, 1),
        ]
        for p_id, r, c in presets:
            if self._get_preset_button_rect(r, c).contains(loc):
                self.set_preset(p_id)
                return True

        # Sliders
        wl_track, _ = self._get_wl_slider_geometry()
        if wl_track.contains(loc):
            frac = (loc.x() - wl_track.left()) / wl_track.width()
            self.set_wavelength(380.0 + frac * (780.0 - 380.0))
            return True

        a_track, _ = self._get_a_slider_geometry()
        if a_track.contains(loc):
            frac = (loc.x() - a_track.left()) / a_track.width()
            self.set_slit_separation(0.10 + frac * (2.00 - 0.10))
            return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── Board Integration & Vector Stamping ───────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of wave optics interference onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_wave_interference_diagram(
            center=(cx, cy + 30.0),
            width=500.0,
            height=320.0,
            wavelength_nm=self._wavelength_nm,
            slit_a_mm=self._slit_a_mm,
            distance_D_m=self._distance_D_m,
            fringe_spacing_mm=self.fringe_spacing_mm,
            preset=self._preset,
            probe_x_mm=self._probe_x_mm,
        )

        for item in diagram_items:
            board.addItem(item)

        logger.info(f"Wave Optics interference diagram stamped onto LessonBoard (λ={self._wavelength_nm:.1f}nm).")

    def close(self) -> None:
        self.setVisible(False)
        self.update()

    # ── Public Properties for Tests & State Verification ─────────────────────

    @property
    def wavelength_nm(self) -> float:
        return self._wavelength_nm

    @property
    def slit_a_mm(self) -> float:
        return self._slit_a_mm

    @property
    def distance_D_m(self) -> float:
        return self._distance_D_m

    @property
    def num_slits(self) -> int:
        return self._num_slits

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def probe_x_mm(self) -> float:
        return self._probe_x_mm

    @property
    def is_white_light(self) -> bool:
        return self._is_white_light
