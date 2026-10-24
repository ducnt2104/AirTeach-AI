"""
AirTeach AI — Interactive Optics & Laser Physics: Fabry-Pérot Interferometer & Cavity Resonator.
Simulates multi-beam interference, Airy transmission function T(δ) = 1 / (1 + F sin²(δ/2)),
mirror reflectance R, coefficient of finesse F = 4R/(1-R)², cavity finesse F = π√R / (1-R),
Free Spectral Range (FSR) Δν_FSR = c / (2nd cos θ), resonance linewidth FWHM δν = Δν_FSR / F,
longitudinal laser cavity modes, and concentric circular interference fringes.
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


C_LIGHT = 299792458.0  # Speed of light in vacuum (m/s)


@dataclass(frozen=True)
class FabryPerotPreset:
    code: str
    title_vi: str
    badge_label: str
    wavelength_nm: float       # Central wavelength λ₀ in nanometers
    cavity_d_um: float         # Mirror separation distance d in micrometers
    reflectance_r: float       # Mirror reflectance R (0.0 to 0.999)
    refractive_index_n: float  # Cavity medium index (1.0 for vacuum/air)
    incident_angle_deg: float  # Angle of incidence θ in degrees
    description: str


FABRY_PEROT_PRESETS: dict[str, FabryPerotPreset] = {
    "HELIUM_NEON_LASER_CAVITY": FabryPerotPreset(
        code="HELIUM_NEON_LASER_CAVITY",
        title_vi="Hốc Cộng hưởng Laser He-Ne (632.8 nm) — Gương Điện môi Phản xạ Cao",
        badge_label="LASER He-Ne",
        wavelength_nm=632.8,
        cavity_d_um=20.0,
        reflectance_r=0.96,
        refractive_index_n=1.0,
        incident_angle_deg=0.0,
        description="Gương phẳng phủ màng điện môi phản xạ siêu cao (R = 96%). Độ sắc nét Finesse cao F ≈ 77 tạo ra các đỉnh truyền qua siêu hẹp, lọc chọn các mode dọc (longitudinal modes) phát xạ laser.",
    ),
    "HIGH_RESOLUTION_SPECTROMETER": FabryPerotPreset(
        code="HIGH_RESOLUTION_SPECTROMETER",
        title_vi="Quang phổ kế Fabry-Pérot — Phân giải Vạch Đôi Natri (589.0 nm)",
        badge_label="SPECTROMETER",
        wavelength_nm=589.0,
        cavity_d_um=35.0,
        reflectance_r=0.90,
        refractive_index_n=1.0,
        incident_angle_deg=0.5,
        description="Phân giải vạch quang phổ kép nguyên tử Natri D1/D2 (589.0 nm & 589.6 nm). Khoảng phổ tự do FSR và bề rộng nửa cực đại FWHM cho phép tách các bước sóng chênh lệch dưới 0.6 nm.",
    ),
    "LOW_FINESSE_ETALON": FabryPerotPreset(
        code="LOW_FINESSE_ETALON",
        title_vi="Bản Chuẩn Etalon Thủy tinh Không tráng phủ — Giao thoa Finesse Thấp",
        badge_label="LOW-F ETALON",
        wavelength_nm=532.0,
        cavity_d_um=15.0,
        reflectance_r=0.30,
        refractive_index_n=1.52,
        incident_angle_deg=0.0,
        description="Bản song phẳng quang học với hệ số phản xạ Fresnel tự nhiên (R ≈ 30%). Hàm Airy suy biến thành dạng gần sóng sin điều hòa tương tự giao thoa 2 chùm tia Michelson.",
    ),
    "OPTICAL_FIBER_TELECOM_DWDM": FabryPerotPreset(
        code="OPTICAL_FIBER_TELECOM_DWDM",
        title_vi="Bộ Lọc Ghép Kênh Bước sóng Quang Viễn thông DWDM C-Band (1550 nm)",
        badge_label="TELECOM DWDM",
        wavelength_nm=1550.0,
        cavity_d_um=25.0,
        reflectance_r=0.94,
        refractive_index_n=1.45,
        incident_angle_deg=0.0,
        description="Bộ lọc vi hốc quang sợi chọn lọc bước sóng viễn thông băng C chuẩn ITU-T 100 GHz grid. Đỉnh truyền qua truyền tải tín hiệu quang với tổn hao xen cực thấp và độ cách ly kênh cao.",
    ),
    "GRAVITATIONAL_WAVE_LIGO_ARM": FabryPerotPreset(
        code="GRAVITATIONAL_WAVE_LIGO_ARM",
        title_vi="Hốc Fabry-Pérot Tích lũy Năng lượng Quang Giao thoa kế Sóng Hấp dẫn",
        badge_label="LIGO CAVITY",
        wavelength_nm=1064.0,
        cavity_d_um=45.0,
        reflectance_r=0.99,
        refractive_index_n=1.0,
        incident_angle_deg=0.0,
        description="Hốc cộng hưởng siêu Finesse (F ≈ 312, R = 99%). Ánh sáng laser phản xạ hàng trăm lần trong khoang quang học làm tăng quãng đường quang học hiệu dụng, cực kỳ nhạy với biến dạng picometer.",
    ),
}


class InteractiveFabryPerotItem(QGraphicsObject):
    """
    Interactive Fabry-Pérot Interferometer & Optical Cavity Resonator.
    Renders multi-beam interference ray paths, Airy transmission function T(δ),
    Finesse, Free Spectral Range (FSR), linewidth FWHM, and concentric circular fringes.
    """

    preset_changed = Signal(str)
    optics_computed = Signal(float, float, float)  # finesse, fsr_ghz, fwhm_ghz
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset & Parameters
        self._active_preset_code: str = "HELIUM_NEON_LASER_CAVITY"
        self._preset: FabryPerotPreset = FABRY_PEROT_PRESETS[self._active_preset_code]

        self._lambda_nm: float = self._preset.wavelength_nm
        self._d_um: float = self._preset.cavity_d_um
        self._r: float = self._preset.reflectance_r
        self._n: float = self._preset.refractive_index_n
        self._theta_deg: float = self._preset.incident_angle_deg

        # Animation state
        self._wave_phase: float = 0.0
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for optical wave propagation animation (30 fps)
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_anim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physical Optics Properties & Equations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> FabryPerotPreset:
        return self._preset

    @property
    def wavelength_nm(self) -> float:
        return self._lambda_nm

    @property
    def cavity_d_um(self) -> float:
        return self._d_um

    @property
    def reflectance_r(self) -> float:
        return self._r

    @property
    def coefficient_of_finesse_f(self) -> float:
        """Coefficient of Finesse F = 4R / (1 - R)²."""
        if self._r >= 1.0:
            return 1e6
        denom = (1.0 - self._r) ** 2
        return (4.0 * self._r) / denom

    @property
    def cavity_finesse(self) -> float:
        """Cavity Finesse Script-F = π * √R / (1 - R) = π * √F / 2."""
        if self._r >= 1.0:
            return 1e6
        return (math.pi * math.sqrt(self._r)) / (1.0 - self._r)

    @property
    def round_trip_phase_shift(self) -> float:
        """
        Round-trip phase shift δ = (4π n d cos θ) / λ.
        d in meters, λ in meters.
        """
        d_m = self._d_um * 1e-6
        lambda_m = self._lambda_nm * 1e-9
        theta_rad = math.radians(self._theta_deg)
        return (4.0 * math.pi * self._n * d_m * math.cos(theta_rad)) / lambda_m

    @property
    def airy_transmittance(self) -> float:
        """
        Airy transmission function T(δ) = 1 / (1 + F sin²(δ/2)).
        Yields 1.0 at resonance (δ = 2mπ).
        """
        delta = self.round_trip_phase_shift
        f_coeff = self.coefficient_of_finesse_f
        denom = 1.0 + f_coeff * (math.sin(delta / 2.0) ** 2)
        return 1.0 / denom

    @property
    def free_spectral_range_ghz(self) -> float:
        """
        Free Spectral Range (FSR) in frequency:
        Δν_FSR = c / (2 n d cos θ) in GHz.
        """
        d_m = self._d_um * 1e-6
        theta_rad = math.radians(self._theta_deg)
        cos_theta = max(1e-4, math.cos(theta_rad))
        fsr_hz = C_LIGHT / (2.0 * self._n * d_m * cos_theta)
        return fsr_hz * 1e-9

    @property
    def free_spectral_range_nm(self) -> float:
        """
        Free Spectral Range in wavelength:
        Δλ_FSR = λ₀² / (2 n d cos θ).
        """
        d_m = self._d_um * 1e-6
        lambda_m = self._lambda_nm * 1e-9
        theta_rad = math.radians(self._theta_deg)
        cos_theta = max(1e-4, math.cos(theta_rad))
        fsr_m = (lambda_m ** 2) / (2.0 * self._n * d_m * cos_theta)
        return fsr_m * 1e9

    @property
    def resonance_linewidth_fwhm_ghz(self) -> float:
        """Resonance linewidth FWHM δν = Δν_FSR / Script-F in GHz."""
        fin = max(0.1, self.cavity_finesse)
        return self.free_spectral_range_ghz / fin

    @property
    def resonance_linewidth_fwhm_nm(self) -> float:
        """Resonance linewidth FWHM δλ = Δλ_FSR / Script-F in nm."""
        fin = max(0.1, self.cavity_finesse)
        return self.free_spectral_range_nm / fin

    @property
    def longitudinal_mode_order_m(self) -> float:
        """Interference order m = 2 n d cos θ / λ."""
        d_m = self._d_um * 1e-6
        lambda_m = self._lambda_nm * 1e-9
        theta_rad = math.radians(self._theta_deg)
        return (2.0 * self._n * d_m * math.cos(theta_rad)) / lambda_m

    def calc_airy_transmittance_for_wavelength(self, w_nm: float) -> float:
        """Calculates Airy transmittance for arbitrary probe wavelength w_nm."""
        d_m = self._d_um * 1e-6
        lambda_m = w_nm * 1e-9
        theta_rad = math.radians(self._theta_deg)
        delta = (4.0 * math.pi * self._n * d_m * math.cos(theta_rad)) / lambda_m
        f_coeff = self.coefficient_of_finesse_f
        denom = 1.0 + f_coeff * (math.sin(delta / 2.0) ** 2)
        return 1.0 / denom

    def calc_airy_transmittance_for_angle(self, theta_rad: float) -> float:
        """Calculates Airy transmittance for arbitrary incidence angle theta_rad."""
        d_m = self._d_um * 1e-6
        lambda_m = self._lambda_nm * 1e-9
        delta = (4.0 * math.pi * self._n * d_m * math.cos(theta_rad)) / lambda_m
        f_coeff = self.coefficient_of_finesse_f
        denom = 1.0 + f_coeff * (math.sin(delta / 2.0) ** 2)
        return 1.0 / denom

    # -------------------------------------------------------------------------
    # Animation & Parameter Controls
    # -------------------------------------------------------------------------
    def _on_anim_tick(self) -> None:
        if self._is_running:
            self._wave_phase = (self._wave_phase + 0.15) % (2.0 * math.pi)
        self.update()

    def load_preset(self, preset_code: str) -> None:
        """Loads a curriculum optics preset."""
        if preset_code not in FABRY_PEROT_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = FABRY_PEROT_PRESETS[preset_code]
        self._lambda_nm = self._preset.wavelength_nm
        self._d_um = self._preset.cavity_d_um
        self._r = self._preset.reflectance_r
        self._n = self._preset.refractive_index_n
        self._theta_deg = self._preset.incident_angle_deg

        self.preset_changed.emit(preset_code)
        self.optics_computed.emit(self.cavity_finesse, self.free_spectral_range_ghz, self.resonance_linewidth_fwhm_ghz)
        self.update()

    def step_distance_um(self, delta_um: float) -> None:
        """Steps mirror cavity separation distance d in micrometers."""
        self._d_um = max(2.0, min(200.0, self._d_um + delta_um))
        self.optics_computed.emit(self.cavity_finesse, self.free_spectral_range_ghz, self.resonance_linewidth_fwhm_ghz)
        self.update()

    def step_wavelength_nm(self, delta_nm: float) -> None:
        """Steps central wavelength λ in nanometers."""
        self._lambda_nm = max(350.0, min(1800.0, self._lambda_nm + delta_nm))
        self.optics_computed.emit(self.cavity_finesse, self.free_spectral_range_ghz, self.resonance_linewidth_fwhm_ghz)
        self.update()

    def step_reflectance_r(self, delta_r: float) -> None:
        """Steps mirror reflectance R."""
        self._r = max(0.10, min(0.995, round(self._r + delta_r, 3)))
        self.optics_computed.emit(self.cavity_finesse, self.free_spectral_range_ghz, self.resonance_linewidth_fwhm_ghz)
        self.update()

    def toggle_animation(self) -> None:
        """Toggles wave propagation animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_d_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_d_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_w_down"] = QRectF(290.0, bar_y, 45.0, 34.0)
        self._buttons["btn_w_up"] = QRectF(340.0, bar_y, 45.0, 34.0)
        self._buttons["btn_r_down"] = QRectF(392.0, bar_y, 45.0, 34.0)
        self._buttons["btn_r_up"] = QRectF(442.0, bar_y, 45.0, 34.0)
        self._buttons["btn_p_laser"] = QRectF(494.0, bar_y, 80.0, 34.0)
        self._buttons["btn_p_spec"] = QRectF(578.0, bar_y, 90.0, 34.0)
        self._buttons["btn_p_telecom"] = QRectF(672.0, bar_y, 85.0, 34.0)
        self._buttons["btn_play"] = QRectF(762.0, bar_y, 75.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(842.0, bar_y, 88.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(FABRY_PEROT_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_d_down":
            self.step_distance_um(-1.0)
        elif key == "btn_d_up":
            self.step_distance_um(1.0)
        elif key == "btn_w_down":
            self.step_wavelength_nm(-10.0)
        elif key == "btn_w_up":
            self.step_wavelength_nm(10.0)
        elif key == "btn_r_down":
            self.step_reflectance_r(-0.05)
        elif key == "btn_r_up":
            self.step_reflectance_r(0.05)
        elif key == "btn_p_laser":
            self.load_preset("HELIUM_NEON_LASER_CAVITY")
        elif key == "btn_p_spec":
            self.load_preset("HIGH_RESOLUTION_SPECTROMETER")
        elif key == "btn_p_telecom":
            self.load_preset("OPTICAL_FIBER_TELECOM_DWDM")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "wavelength_nm": self._lambda_nm,
            "cavity_d_um": self._d_um,
            "reflectance_r": self._r,
            "finesse": self.cavity_finesse,
            "fsr_ghz": self.free_spectral_range_ghz,
            "fwhm_ghz": self.resonance_linewidth_fwhm_ghz,
            "transmittance": self.airy_transmittance,
            "order_m": self.longitudinal_mode_order_m,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_cavity_diagram_panel(painter)
        self._paint_airy_spectrum_panel(painter)
        self._paint_fringes_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(9, 14, 25, 252))
        grad.setColorAt(1.0, QColor(5, 8, 16, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"QUANG HỌC LASER & GIAO THOA KẾ FABRY-PÉROT: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(
            22,
            48,
            f"λ = {self._lambda_nm:.1f} nm | Khoảng cách gương d = {self._d_um:.1f} μm | Phản xạ R = {self._r:.3f} | Finesse F = {self.cavity_finesse:.1f} | FSR = {self.free_spectral_range_ghz:.1f} GHz | FWHM = {self.resonance_linewidth_fwhm_ghz:.2f} GHz",
        )

    def _paint_cavity_diagram_panel(self, painter: QPainter) -> None:
        """Renders 2-mirror cavity with multi-beam reflections and wave propagation."""
        cx = 20.0
        cy = 60.0
        cw = 440.0
        ch = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(cx, cy, cw, ch, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(cx + 12.0, cy + 20.0, "KHOANG QUANG HỌC & ĐA PHẢN XẠ NHIỀU CHÙM TIA:")

        # Mirror positions
        m1_x = cx + 110.0
        m2_x = cx + 330.0
        m_top = cy + 40.0
        m_bot = cy + 220.0
        m_thick = 16.0

        # Mirror 1 (M1)
        m1_grad = QLinearGradient(m1_x - m_thick, m_top, m1_x, m_top)
        m1_grad.setColorAt(0.0, QColor(71, 85, 105, 180))
        m1_grad.setColorAt(1.0, QColor(56, 189, 248, 220))
        painter.setBrush(QBrush(m1_grad))
        painter.setPen(QPen(QColor(148, 163, 184), 1.2))
        painter.drawRect(m1_x - m_thick, m_top, m_thick, m_bot - m_top)

        # Mirror 2 (M2)
        m2_grad = QLinearGradient(m2_x, m_top, m2_x + m_thick, m_top)
        m2_grad.setColorAt(0.0, QColor(56, 189, 248, 220))
        m2_grad.setColorAt(1.0, QColor(71, 85, 105, 180))
        painter.setBrush(QBrush(m2_grad))
        painter.drawRect(m2_x, m_top, m_thick, m_bot - m_top)

        # Labels for mirrors
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(226, 232, 240)))
        painter.drawText(m1_x - 30.0, m_top - 6.0, f"M1 (R={self._r:.2f})")
        painter.drawText(m2_x - 5.0, m_top - 6.0, f"M2 (R={self._r:.2f})")

        # Cavity distance d dimension line
        dim_y = cy + 215.0
        painter.setPen(QPen(QColor(250, 204, 21), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(m1_x, dim_y, m2_x, dim_y)
        painter.drawText((m1_x + m2_x) / 2.0 - 25.0, dim_y - 4.0, f"d = {self._d_um:.1f} μm")

        # Laser beam color according to wavelength
        beam_color = self._get_wavelength_color(self._lambda_nm)

        # Incident Beam (from left)
        in_y = cy + 85.0
        p_in = QPainterPath()
        p_in.moveTo(cx + 20.0, in_y)
        p_in.lineTo(m1_x, in_y)
        painter.setPen(QPen(beam_color, 2.5))
        painter.drawPath(p_in)
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(cx + 25.0, in_y - 6.0, "Tia tới I_in")

        # Zig-zag multi-reflections in cavity
        p_bounce = QPainterPath()
        p_bounce.moveTo(m1_x, in_y)
        # 4 internal bounces
        p_bounce.lineTo(m2_x, in_y + 25.0)
        p_bounce.lineTo(m1_x, in_y + 50.0)
        p_bounce.lineTo(m2_x, in_y + 75.0)
        p_bounce.lineTo(m1_x, in_y + 100.0)
        painter.setPen(QPen(beam_color.lighter(130), 2.0))
        painter.drawPath(p_bounce)

        # Transmitted beams (T1, T2) exiting from M2
        t_trans = self.airy_transmittance
        trans_alpha = max(40, min(255, int(t_trans * 255)))
        t_color = QColor(beam_color.red(), beam_color.green(), beam_color.blue(), trans_alpha)
        painter.setPen(QPen(t_color, 2.0))
        painter.drawLine(m2_x + m_thick, in_y + 25.0, cx + 420.0, in_y + 25.0)
        painter.drawLine(m2_x + m_thick, in_y + 75.0, cx + 420.0, in_y + 75.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(t_color))
        painter.drawText(cx + 355.0, in_y + 20.0, f"T_Airy = {t_trans*100:.1f}%")

        # Wave oscillation in cavity
        sin_path = QPainterPath()
        n_steps = 60
        dx = (m2_x - m1_x) / n_steps
        amp = 12.0 * math.sqrt(t_trans)
        for i in range(n_steps + 1):
            wx = m1_x + i * dx
            wy = (cy + 170.0) + amp * math.sin(i * 0.4 + self._wave_phase)
            if i == 0:
                sin_path.moveTo(wx, wy)
            else:
                sin_path.lineTo(wx, wy)
        painter.setPen(QPen(QColor(56, 189, 248, 190), 1.5))
        painter.drawPath(sin_path)
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(m1_x + 30.0, cy + 195.0, "Sóng đứng cộng hưởng trong hốc")

    def _paint_airy_spectrum_panel(self, painter: QPainter) -> None:
        """Plots the Airy transmission curve T(δ) over 3 Free Spectral Ranges."""
        sx = 480.0
        sy = 60.0
        sw = 450.0
        sh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(sx, sy, sw, sh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(sx + 12.0, sy + 20.0, "QUANG PHỔ HÀM AIRY: T(δ) = 1 / [1 + F sin²(δ/2)]:")

        # Plot area
        px = sx + 45.0
        py = sy + 35.0
        pw = sw - 65.0
        ph = sh - 65.0

        # Axes
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X-axis
        painter.drawLine(px, py, px, py + ph)            # Y-axis

        # Grid lines and Y labels (0.0, 0.5, 1.0)
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(100, 116, 139)))
        for frac, label in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
            ly = py + ph - frac * ph
            painter.drawLine(px, ly, px + pw, ly)
            painter.drawText(px - 28.0, ly + 3.0, label)

        # Plot Airy curve over 3 periods (-3π to +3π)
        # Center peak corresponds to current resonance
        airy_path = QPainterPath()
        f_coeff = self.coefficient_of_finesse_f
        delta_span = 6.0 * math.pi
        steps = int(pw)

        for step in range(steps + 1):
            delta = -3.0 * math.pi + (step / float(steps)) * delta_span
            t_val = 1.0 / (1.0 + f_coeff * (math.sin(delta / 2.0) ** 2))
            gx = px + step
            gy = py + ph - t_val * ph
            if step == 0:
                airy_path.moveTo(gx, gy)
            else:
                airy_path.lineTo(gx, gy)

        # Fill under curve
        fill_path = QPainterPath(airy_path)
        fill_path.lineTo(px + pw, py + ph)
        fill_path.lineTo(px, py + ph)
        fill_path.closeSubpath()
        grad = QLinearGradient(px, py, px, py + ph)
        grad.setColorAt(0.0, QColor(56, 189, 248, 120))
        grad.setColorAt(1.0, QColor(56, 189, 248, 10))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Curve stroke
        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawPath(airy_path)

        # FWHM bracket on central peak
        center_x = px + pw / 2.0
        half_y = py + ph / 2.0
        # Angular half-width: δ_half = 2 / √F
        half_width_px = (2.0 / max(0.1, math.sqrt(f_coeff))) * (pw / delta_span)
        bracket_left = max(px, center_x - half_width_px)
        bracket_right = min(px + pw, center_x + half_width_px)

        painter.setPen(QPen(QColor(244, 63, 94), 1.5))
        painter.drawLine(bracket_left, half_y, bracket_right, half_y)
        painter.drawText(center_x - 18.0, half_y - 5.0, "FWHM")

        # FSR indication between peak -2π and 0
        peak1_x = px + pw / 2.0 - (2.0 * math.pi / delta_span) * pw
        peak2_x = px + pw / 2.0
        fsr_y = py + 15.0
        painter.setPen(QPen(QColor(250, 204, 21), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(peak1_x, fsr_y, peak2_x, fsr_y)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText((peak1_x + peak2_x) / 2.0 - 15.0, fsr_y - 4.0, f"FSR = {self.free_spectral_range_ghz:.1f} GHz")

        # X-axis labels
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(peak1_x - 14.0, py + ph + 14.0, "δ = -2π")
        painter.drawText(peak2_x - 10.0, py + ph + 14.0, "δ = 0")
        peak3_x = px + pw / 2.0 + (2.0 * math.pi / delta_span) * pw
        painter.drawText(peak3_x - 10.0, py + ph + 14.0, "δ = +2π")

    def _paint_fringes_panel(self, painter: QPainter) -> None:
        """Renders 2D concentric circular Fabry-Pérot interference fringes."""
        fx = 20.0
        fy = 315.0
        fw = 440.0
        fh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(fx, fy, fw, fh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(fx + 12.0, fy + 20.0, "VÂN GIAO THOA TRÒN ĐỒNG TÂM TRÊN TIÊU DIỆN (θ):")

        # Screen center
        sc_x = fx + fw / 2.0
        sc_y = fy + fh / 2.0 + 8.0
        max_r = min(fw, fh) / 2.0 - 22.0

        # Background circular screen
        screen_grad = QRadialGradient(sc_x, sc_y, max_r)
        screen_grad.setColorAt(0.0, QColor(10, 15, 28))
        screen_grad.setColorAt(1.0, QColor(2, 4, 10))
        painter.setBrush(QBrush(screen_grad))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawEllipse(QPointF(sc_x, sc_y), max_r, max_r)

        # Draw concentric transmission rings
        # Ring condition: 2 n d cos θ = m λ
        # For small θ: cos θ ≈ 1 - θ²/2 => r_p ~ sqrt(p)
        beam_color = self._get_wavelength_color(self._lambda_nm)
        f_coeff = self.coefficient_of_finesse_f
        n_rings = 4

        for p in range(1, n_rings + 1):
            ring_radius = max_r * math.sqrt(p / float(n_rings + 1))
            # Ring sharpness depends on Finesse
            # High finesse -> thin ring; low finesse -> thick ring
            pen_width = max(1.2, min(9.0, 18.0 / math.sqrt(max(1.0, f_coeff))))
            ring_pen = QPen(beam_color, pen_width)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(sc_x, sc_y), ring_radius, ring_radius)

            # Order label
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(226, 232, 240, 180)))
            painter.drawText(sc_x + ring_radius - 8.0, sc_y - 3.0, f"m-{p}")

        # Center dot
        painter.setBrush(QBrush(beam_color))
        painter.drawEllipse(QPointF(sc_x, sc_y), 3.0, 3.0)

        # Subtitle
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(fx + 14.0, fy + fh - 10.0, f"Bán kính vân: r_p ∝ √(p) | Bề rộng vân: Δr ∝ 1/√F | F = {self.cavity_finesse:.1f}")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Displays rigorous analytical optics equations and quantitative metrics."""
        mx = 480.0
        my = 315.0
        mw = 450.0
        mh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "THÔNG SỐ QUANG HỌC LƯỢNG HÓA CHÍNH XÁC:")

        metrics = [
            ("Hệ số Phản xạ Gương (Reflectance):", f"R = {self._r:.3f} ({self._r*100:.1f}%)"),
            ("Hệ số Finesse (Coeff of Finesse):", f"F = 4R / (1 - R)² = {self.coefficient_of_finesse_f:.1f}"),
            ("Độ Sắc nét Khoang (Cavity Finesse):", f"F = π√R / (1 - R) = {self.cavity_finesse:.2f}"),
            ("Khoảng Phổ Tự do Tần số (FSR ν):", f"Δν_FSR = c / (2nd) = {self.free_spectral_range_ghz:.2f} GHz"),
            ("Khoảng Phổ Tự do Bước sóng (FSR λ):", f"Δλ_FSR = λ² / (2nd) = {self.free_spectral_range_nm:.3f} nm"),
            ("Độ rộng Nửa Cực đại (FWHM Linewidth):", f"δν = Δν / F = {self.resonance_linewidth_fwhm_ghz:.3f} GHz"),
            ("Hệ số Phẩm chất Quang (Q-Factor):", f"Q = ν₀ / δν = {(C_LIGHT / (self._lambda_nm * 1e-9)) / (self.resonance_linewidth_fwhm_ghz * 1e9):.1e}"),
            ("Bậc Giao thoa Dọc (Cavity Mode m):", f"m = 2nd / λ = {self.longitudinal_mode_order_m:.1f}"),
            ("Hệ số Truyền qua Airy Hiện tại:", f"T(δ) = {self.airy_transmittance*100:.2f}% | R(δ) = {(1.0-self.airy_transmittance)*100:.2f}%"),
        ]

        row_y = my + 44.0
        line_h = 20.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_d_down", "d -", QColor(51, 65, 85)),
            ("btn_d_up", "d +", QColor(51, 65, 85)),
            ("btn_w_down", "λ -", QColor(51, 65, 85)),
            ("btn_w_up", "λ +", QColor(51, 65, 85)),
            ("btn_r_down", "R -", QColor(51, 65, 85)),
            ("btn_r_up", "R +", QColor(51, 65, 85)),
            ("btn_p_laser", "He-Ne", QColor(30, 41, 59)),
            ("btn_p_spec", "Natri D", QColor(30, 41, 59)),
            ("btn_p_telecom", "DWDM", QColor(30, 41, 59)),
            ("btn_play", "Tạm dừng" if self._is_running else "Tiếp tục", QColor(245, 158, 11)),
            ("btn_stamp", "Ghim Bảng", QColor(16, 185, 129)),
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

    @staticmethod
    def _get_wavelength_color(lambda_nm: float) -> QColor:
        """Converts optical wavelength λ (nm) to representative RGB color."""
        if lambda_nm < 440:
            return QColor(138, 43, 226)   # Violet
        elif lambda_nm < 490:
            return QColor(0, 191, 255)    # Blue / Cyan
        elif lambda_nm < 560:
            return QColor(34, 197, 94)    # Green
        elif lambda_nm < 595:
            return QColor(234, 179, 8)    # Yellow
        elif lambda_nm < 630:
            return QColor(249, 115, 22)   # Orange
        elif lambda_nm <= 750:
            return QColor(239, 68, 68)    # Red (He-Ne 632.8 nm)
        else:
            return QColor(217, 70, 239)   # Infrared / Telecom false color
