"""
AirTeach AI — Interactive Materials Science & Crystallography: Bragg's Law, X-Ray Diffraction (XRD) & Crystal Lattice Planes.
Simulates Bragg's diffraction law n*lambda = 2*d*sin(theta), interplanar spacing d_hkl = a / sqrt(h^2 + k^2 + l^2),
systematic extinction selection rules for cubic crystal systems (FCC, BCC, Diamond Cubic, Rock Salt Halite),
Debye-Scherrer crystallite size peak broadening beta = K*lambda / (D*cos(theta)),
powder XRD diffractogram spectrum I(2theta), live goniometer scanning, and 3D Miller plane slices.
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
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class MillerPlane:
    """Represents a specific crystallographic lattice plane (hkl)."""
    h: int
    k: int
    l: int
    relative_intensity: float  # Normalized 0.0 to 100.0%
    label: str

    @property
    def sum_sq(self) -> int:
        """Quadratic sum h^2 + k^2 + l^2."""
        return self.h * self.h + self.k * self.k + self.l * self.l

    def d_spacing_nm(self, lattice_a_nm: float) -> float:
        """Calculates interplanar spacing d_hkl = a / sqrt(h^2 + k^2 + l^2) in nm."""
        sq = self.sum_sq
        if sq <= 0:
            return 0.0
        return lattice_a_nm / math.sqrt(float(sq))

    def bragg_theta_rad(self, wavelength_nm: float, lattice_a_nm: float, n: int = 1) -> Optional[float]:
        """Calculates Bragg angle theta_B in radians from sin(theta_B) = n*lambda / (2*d_hkl)."""
        d = self.d_spacing_nm(lattice_a_nm)
        if d <= 1e-6:
            return None
        sin_theta = (n * wavelength_nm) / (2.0 * d)
        if sin_theta > 1.0 or sin_theta < -1.0:
            return None
        return math.asin(sin_theta)

    def two_theta_deg(self, wavelength_nm: float, lattice_a_nm: float, n: int = 1) -> Optional[float]:
        """Calculates detector diffraction angle 2*theta_B in degrees."""
        th_rad = self.bragg_theta_rad(wavelength_nm, lattice_a_nm, n)
        if th_rad is None:
            return None
        return math.degrees(2.0 * th_rad)


@dataclass(frozen=True)
class BraggDiffractionPreset:
    """Preset curriculum scenario for Materials Science & XRD Crystallography."""
    code: str
    title_vi: str
    badge_label: str
    crystal_type: str  # "DIAMOND_CUBIC", "FCC", "BCC", "ROCK_SALT", "NANOCRYSTAL"
    lattice_a_nm: float  # Lattice parameter a in nm
    wavelength_nm: float  # Default X-ray wavelength in nm (0.15406 for Cu-Ka)
    crystallite_size_nm: float  # Volume-averaged crystallite grain size D in nm
    active_plane_idx: int
    allowed_planes: tuple[MillerPlane, ...]
    description: str


BRAGG_PRESETS: dict[str, BraggDiffractionPreset] = {
    "SILICON_POWDER_STANDARD": BraggDiffractionPreset(
        code="SILICON_POWDER_STANDARD",
        title_vi="Silic Đơn tinh thể/Bột — Tiêu chuẩn SRM 640 & Mạng Kim cương (Diamond Cubic)",
        badge_label="SILICON SRM",
        crystal_type="DIAMOND_CUBIC",
        lattice_a_nm=0.5431,  # 5.431 Å
        wavelength_nm=0.15406,  # Cu-Ka1
        crystallite_size_nm=45.0,
        active_plane_idx=0,
        allowed_planes=(
            MillerPlane(1, 1, 1, 100.0, "(111)"),
            MillerPlane(2, 2, 0, 55.0, "(220)"),
            MillerPlane(3, 1, 1, 30.0, "(311)"),
            MillerPlane(4, 0, 0, 6.0, "(400)"),
            MillerPlane(3, 3, 1, 11.0, "(331)"),
            MillerPlane(4, 2, 2, 12.0, "(422)"),
        ),
        description="Mẫu chuẩn NIST SRM 640. Tinh thể Si cấu trúc kim cương (Fd-3m) tuân theo quy tắc triệt tiêu có hệ thống: h,k,l không hỗn tạp và h+k+l = 4n khi toàn chẵn (các mặt (200), (222) bị triệt tiêu hoàn toàn). Đỉnh (111) cực đại tại 2θ = 28.44°.",
    ),
    "COPPER_FCC_POLYCRYSTAL": BraggDiffractionPreset(
        code="COPPER_FCC_POLYCRYSTAL",
        title_vi="Đồng Kim loại Đa tinh thể — Lập phương Tâm mặt (FCC) & Mặt Xếp chặt {111}",
        badge_label="COPPER FCC",
        crystal_type="FCC",
        lattice_a_nm=0.3615,  # 3.615 Å
        wavelength_nm=0.15406,
        crystallite_size_nm=35.0,
        active_plane_idx=0,
        allowed_planes=(
            MillerPlane(1, 1, 1, 100.0, "(111)"),
            MillerPlane(2, 0, 0, 46.0, "(200)"),
            MillerPlane(2, 2, 0, 20.0, "(220)"),
            MillerPlane(3, 1, 1, 17.0, "(311)"),
            MillerPlane(2, 2, 2, 8.0, "(222)"),
        ),
        description="Kim loại Cu mạng lập phương tâm mặt (FCC, a = 0.3615 nm). Điều kiện nhiễu xạ: h,k,l cùng chẵn hoặc cùng lẻ. Đỉnh phản xạ mạnh nhất từ họ mặt xếp chặt nhất {111} tại 2θ = 43.30°, theo sau bởi {200} tại 50.43°.",
    ),
    "ALPHA_IRON_BCC_FERRITE": BraggDiffractionPreset(
        code="ALPHA_IRON_BCC_FERRITE",
        title_vi="Sắt Ferrite Alpha — Lập phương Tâm khối (BCC) & Mặt Phản xạ {110}",
        badge_label="IRON BCC",
        crystal_type="BCC",
        lattice_a_nm=0.2866,  # 2.866 Å
        wavelength_nm=0.15406,
        crystallite_size_nm=30.0,
        active_plane_idx=0,
        allowed_planes=(
            MillerPlane(1, 1, 0, 100.0, "(110)"),
            MillerPlane(2, 0, 0, 20.0, "(200)"),
            MillerPlane(2, 1, 1, 30.0, "(211)"),
            MillerPlane(2, 2, 0, 10.0, "(220)"),
            MillerPlane(3, 1, 0, 12.0, "(310)"),
        ),
        description="Pha Ferrite alpha-Fe lập phương tâm khối (BCC, a = 0.2866 nm). Quy tắc tắt có hệ thống cho phép phản xạ khi h+k+l = 2n (tổng chẵn). Mặt phẳng có mật độ nguyên tử cao nhất là {110} tạo đỉnh vượt trội tại 2θ = 44.67°.",
    ),
    "SODIUM_CHLORIDE_ROCK_SALT": BraggDiffractionPreset(
        code="SODIUM_CHLORIDE_ROCK_SALT",
        title_vi="Muối ăn NaCl — Cấu trúc Halite & Yếu tố Cấu trúc F_hkl (Bragg 1913)",
        badge_label="NACL HALITE",
        crystal_type="ROCK_SALT",
        lattice_a_nm=0.5640,  # 5.640 Å
        wavelength_nm=0.15406,
        crystallite_size_nm=50.0,
        active_plane_idx=1,  # (200) is strongest
        allowed_planes=(
            MillerPlane(1, 1, 1, 13.0, "(111)"),  # Weak: F ~ f_Cl - f_Na
            MillerPlane(2, 0, 0, 100.0, "(200)"), # Strong: F ~ f_Cl + f_Na
            MillerPlane(2, 2, 0, 55.0, "(220)"),
            MillerPlane(3, 1, 1, 2.0, "(311)"),   # Weak
            MillerPlane(2, 2, 2, 15.0, "(222)"),
            MillerPlane(4, 0, 0, 6.0, "(400)"),
        ),
        description="Mô hình Halite lịch sử xác định bởi William Henry Bragg & William Lawrence Bragg (Giải Nobel 1915). Minh họa rõ yếu tố cấu trúc: mặt (111) có cường độ rất yếu do sóng tán xạ từ Na+ và Cl- ngược pha triệt tiêu một phần (F ~ f_Cl - f_Na), trong khi mặt (200) cùng pha cộng hưởng cực đại (F ~ f_Cl + f_Na).",
    ),
    "NANOCRYSTALLINE_GRAIN_SIZE": BraggDiffractionPreset(
        code="NANOCRYSTALLINE_GRAIN_SIZE",
        title_vi="Vật liệu Hạt Nano — Hiệu ứng Làm rộng Vạch Debye-Scherrer (D = 5.0 nm)",
        badge_label="NANO SCHERRER",
        crystal_type="NANOCRYSTAL",
        lattice_a_nm=0.3524,  # Ni FCC
        wavelength_nm=0.15406,
        crystallite_size_nm=5.0,  # Ultrafine grain size
        active_plane_idx=0,
        allowed_planes=(
            MillerPlane(1, 1, 1, 100.0, "(111)"),
            MillerPlane(2, 0, 0, 42.0, "(200)"),
            MillerPlane(2, 2, 0, 21.0, "(220)"),
        ),
        description="Hiệu ứng làm rộng vạch Scherrer của vật liệu nano (D = 5.0 nm so với hạt vi mô D = 50 nm). Khi kích thước hạt tinh thể giảm xuống cấp độ nanomet, số lượng mặt phẳng tán xạ hữu hạn làm độ mở rộng góc vạch nhiễu xạ FWHM beta tăng vọt tỷ lệ nghịch với D theo phương trình: beta = K*lambda / (D*cos theta).",
    ),
}


class InteractiveBraggDiffractionItem(QGraphicsObject):
    """
    Interactive Materials Science & Crystallography: Bragg's Law & XRD Powder Diffractometer.
    Simulates:
      - Bragg's Equation: n * lambda = 2 * d * sin(theta)
      - Interplanar Spacing: d_hkl = a / sqrt(h^2 + k^2 + l^2)
      - Debye-Scherrer Peak Broadening: beta = K * lambda / (D * cos(theta))
      - Extinction Selection Rules for Cubic Systems (FCC, BCC, Diamond, Rock Salt)
      - Real-time Phase Interference along Incident and Diffracted X-ray Rays
      - Powder Diffractogram Spectrum I(2theta) and Live Detector Angle Scanning
    """

    preset_changed = Signal(str)
    diffraction_computed = Signal(float, float, float)  # two_theta_deg, d_spacing_nm, fwhm_deg
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
        self._active_preset_code: str = "SILICON_POWDER_STANDARD"
        self._preset: BraggDiffractionPreset = BRAGG_PRESETS[self._active_preset_code]

        self._lattice_a_nm: float = self._preset.lattice_a_nm
        self._wavelength_nm: float = self._preset.wavelength_nm
        self._crystallite_d_nm: float = self._preset.crystallite_size_nm
        self._active_plane_idx: int = self._preset.active_plane_idx

        # Initialize incident angle theta to theoretical Bragg angle of active plane
        th_bragg = self.active_plane.bragg_theta_rad(self._wavelength_nm, self._lattice_a_nm)
        self._incident_theta_deg: float = math.degrees(th_bragg) if th_bragg is not None else 14.22

        # Animation states
        self._wave_phase: float = 0.0
        self._is_running: bool = True
        self._is_auto_scanning: bool = False
        self._scan_speed_deg_per_tick: float = 0.25

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
    # Physical & Crystallographic Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> BraggDiffractionPreset:
        return self._preset

    @property
    def active_plane(self) -> MillerPlane:
        idx = max(0, min(self._active_plane_idx, len(self._preset.allowed_planes) - 1))
        return self._preset.allowed_planes[idx]

    @property
    def d_spacing_nm(self) -> float:
        return self.active_plane.d_spacing_nm(self._lattice_a_nm)

    @property
    def d_spacing_angstrom(self) -> float:
        return self.d_spacing_nm * 10.0

    @property
    def theoretical_bragg_theta_deg(self) -> float:
        th = self.active_plane.bragg_theta_rad(self._wavelength_nm, self._lattice_a_nm)
        return math.degrees(th) if th is not None else 0.0

    @property
    def theoretical_two_theta_deg(self) -> float:
        return 2.0 * self.theoretical_bragg_theta_deg

    @property
    def current_two_theta_deg(self) -> float:
        return 2.0 * self._incident_theta_deg

    @property
    def optical_path_difference_nm(self) -> float:
        """Delta = 2 * d * sin(theta)."""
        theta_rad = math.radians(self._incident_theta_deg)
        return 2.0 * self.d_spacing_nm * math.sin(theta_rad)

    @property
    def phase_difference_rad(self) -> float:
        """Phase difference phi = 2*pi * Delta / lambda."""
        if self._wavelength_nm <= 1e-6:
            return 0.0
        return (2.0 * math.pi * self.optical_path_difference_nm) / self._wavelength_nm

    @property
    def scherrer_fwhm_rad(self) -> float:
        """
        Debye-Scherrer peak width beta = K * lambda / (D * cos(theta_B)), with K = 0.9.
        """
        th_rad = math.radians(self.theoretical_bragg_theta_deg)
        cos_th = max(0.05, math.cos(th_rad))
        k_shape = 0.9
        return (k_shape * self._wavelength_nm) / (self._crystallite_d_nm * cos_th)

    @property
    def scherrer_fwhm_deg(self) -> float:
        """FWHM beta in degrees."""
        return math.degrees(self.scherrer_fwhm_rad)

    @property
    def total_observed_fwhm_deg(self) -> float:
        """Observed FWHM combining crystallite size broadening and instrumental broadening (0.08 deg)."""
        b_size = self.scherrer_fwhm_deg
        b_inst = 0.08
        return math.sqrt(b_size * b_size + b_inst * b_inst)

    @property
    def is_bragg_resonance(self) -> bool:
        """Determines if current incident angle satisfies constructive Bragg condition within FWHM/2."""
        delta_deg = abs(self._incident_theta_deg - self.theoretical_bragg_theta_deg)
        return delta_deg <= (self.total_observed_fwhm_deg / 2.0)

    @property
    def interference_intensity_factor(self) -> float:
        """
        Calculates normalized constructive interference intensity factor [0.0, 1.0].
        Follows a Lorentzian / pseudo-Voigt profile centered at theoretical Bragg angle.
        """
        delta_deg = abs(self._incident_theta_deg - self.theoretical_bragg_theta_deg)
        hwhm_deg = max(0.04, self.total_observed_fwhm_deg / 2.0)
        # Lorentzian profile
        return 1.0 / (1.0 + (delta_deg / hwhm_deg) ** 2)

    def calculate_diffractogram_intensity(self, two_theta_eval_deg: float) -> float:
        """
        Calculates total XRD diffracted intensity at angle 2theta by summing contributions
        from all allowed crystallographic reflection planes with Scherrer peak broadening.
        """
        intensity = 2.0  # Background baseline intensity (counts/noise)
        for plane in self._preset.allowed_planes:
            peak_2th = plane.two_theta_deg(self._wavelength_nm, self._lattice_a_nm)
            if peak_2th is None:
                continue
            delta_2th = two_theta_eval_deg - peak_2th
            # Compute plane-specific Scherrer width
            th_plane_rad = math.radians(peak_2th / 2.0)
            cos_th = max(0.05, math.cos(th_plane_rad))
            beta_rad = (0.9 * self._wavelength_nm) / (self._crystallite_d_nm * cos_th)
            beta_deg = math.degrees(beta_rad)
            fwhm = math.sqrt(beta_deg * beta_deg + 0.08 * 0.08)
            hwhm = fwhm / 2.0

            # Lorentzian peak shape
            peak_profile = 1.0 / (1.0 + (delta_2th / hwhm) ** 2)
            intensity += plane.relative_intensity * peak_profile

        return min(110.0, intensity)

    # -------------------------------------------------------------------------
    # Simulation Control & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in BRAGG_PRESETS:
            return
        self._active_preset_code = code
        self._preset = BRAGG_PRESETS[code]
        self._lattice_a_nm = self._preset.lattice_a_nm
        self._wavelength_nm = self._preset.wavelength_nm
        self._crystallite_d_nm = self._preset.crystallite_size_nm
        self._active_plane_idx = self._preset.active_plane_idx

        # Reset incident angle to the new active Bragg angle
        th_bragg = self.active_plane.bragg_theta_rad(self._wavelength_nm, self._lattice_a_nm)
        if th_bragg is not None:
            self._incident_theta_deg = math.degrees(th_bragg)

        self._layout_buttons()
        self.preset_changed.emit(code)
        self.diffraction_computed.emit(
            self.current_two_theta_deg,
            self.d_spacing_nm,
            self.total_observed_fwhm_deg,
        )
        self.update()

    def set_incident_theta(self, theta_deg: float) -> None:
        """Sets the incident X-ray angle theta in degrees [5.0, 50.0]."""
        self._incident_theta_deg = max(5.0, min(50.0, theta_deg))
        self.diffraction_computed.emit(
            self.current_two_theta_deg,
            self.d_spacing_nm,
            self.total_observed_fwhm_deg,
        )
        self.update()

    def set_active_plane_index(self, idx: int) -> None:
        """Selects the target crystallographic plane index."""
        count = len(self._preset.allowed_planes)
        if count == 0:
            return
        self._active_plane_idx = idx % count
        # Auto-align incident angle to this plane's Bragg angle
        th_bragg = self.active_plane.bragg_theta_rad(self._wavelength_nm, self._lattice_a_nm)
        if th_bragg is not None:
            self._incident_theta_deg = math.degrees(th_bragg)
        self.diffraction_computed.emit(
            self.current_two_theta_deg,
            self.d_spacing_nm,
            self.total_observed_fwhm_deg,
        )
        self.update()

    def toggle_wavelength(self) -> None:
        """Toggles X-ray source between Cu-Ka (0.15406 nm) and Mo-Ka (0.07107 nm)."""
        if abs(self._wavelength_nm - 0.15406) < 0.001:
            self._wavelength_nm = 0.07107  # Mo-Ka
        else:
            self._wavelength_nm = 0.15406  # Cu-Ka
        # Re-tune incident angle to Bragg angle
        th_bragg = self.active_plane.bragg_theta_rad(self._wavelength_nm, self._lattice_a_nm)
        if th_bragg is not None:
            self._incident_theta_deg = math.degrees(th_bragg)
        self.update()

    def adjust_crystallite_size(self, delta_nm: float) -> None:
        """Adjusts crystallite grain size D in nm [2.0, 100.0]."""
        self._crystallite_d_nm = max(2.0, min(100.0, self._crystallite_d_nm + delta_nm))
        self.diffraction_computed.emit(
            self.current_two_theta_deg,
            self.d_spacing_nm,
            self.total_observed_fwhm_deg,
        )
        self.update()

    def toggle_auto_scan(self) -> None:
        """Toggles automatic continuous 2theta goniometer rotation scan."""
        self._is_auto_scanning = not self._is_auto_scanning
        self.update()

    def _on_sim_tick(self) -> None:
        """Performs animation tick for X-ray wave oscillations and goniometer sweep."""
        if not self._is_running:
            return

        self._wave_phase += 0.25
        if self._wave_phase > 2.0 * math.pi:
            self._wave_phase -= 2.0 * math.pi

        if self._is_auto_scanning:
            # Advance 2theta by scan speed, angle theta is half
            new_2th = self.current_two_theta_deg + self._scan_speed_deg_per_tick
            if new_2th > 95.0:
                new_2th = 15.0
            self._incident_theta_deg = new_2th / 2.0

        self.update()

    # -------------------------------------------------------------------------
    # Layout & Touchless Hit-Testing
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Lays out interactive touchless action buttons."""
        self._buttons.clear()
        by = 595.0
        bh = 38.0

        # Control buttons
        self._buttons["btn_preset"] = QRectF(20.0, by, 160.0, bh)
        self._buttons["btn_plane"] = QRectF(188.0, by, 110.0, bh)
        self._buttons["btn_th_down"] = QRectF(306.0, by, 58.0, bh)
        self._buttons["btn_th_up"] = QRectF(370.0, by, 58.0, bh)
        self._buttons["btn_tune"] = QRectF(436.0, by, 84.0, bh)
        self._buttons["btn_xray"] = QRectF(528.0, by, 94.0, bh)
        self._buttons["btn_d_down"] = QRectF(630.0, by, 54.0, bh)
        self._buttons["btn_d_up"] = QRectF(690.0, by, 54.0, bh)
        self._buttons["btn_scan"] = QRectF(752.0, by, 96.0, bh)
        self._buttons["btn_stamp"] = QRectF(856.0, by, 84.0, bh)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Painting & Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_lattice_interference_panel(painter)
        self._paint_diffractogram_panel(painter)
        self._paint_unit_cell_panel(painter)
        self._paint_crystallography_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        """Renders main acrylic dark glass card background."""
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
        painter.drawText(hx, hy + 18.0, "NHIỄU XẠ TIA X BRAGG & MẶT MẠNG TINH THỂ (XRD CRYSTALLOGRAPHY)")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(hx, hy + 35.0, self._preset.title_vi)

        # Badges on right side
        bx = self.CARD_W - 20.0
        badges = [
            (f"D = {self._crystallite_d_nm:.1f} nm", QColor(168, 85, 247)),
            (f"λ = {self._wavelength_nm * 10.0:.3f} Å ({'Cu-Kα' if abs(self._wavelength_nm - 0.15406) < 0.001 else 'Mo-Kα'})", QColor(14, 165, 233)),
            (f"a = {self._lattice_a_nm * 10.0:.3f} Å", QColor(16, 185, 129)),
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

    def _paint_lattice_interference_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Left Panel: Crystal Lattice Planes & Incident/Diffracted X-Ray Waves.
        Demonstrates geometric path difference Delta = 2*d*sin(theta) and wave phase interference.
        """
        px = 20.0
        py = 70.0
        pw = 460.0
        ph = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 10.0, 10.0)

        # Panel title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(px + 12.0, py + 18.0, "MẶT MẠNG TINH THỂ & GIAO THOA SÓNG TIA X:")

        # Status badge: Constructive vs Destructive
        resonance = self.is_bragg_resonance
        factor = self.interference_intensity_factor
        status_txt = f"CỘNG HƯỞNG BRAGG (I = {factor * 100:.0f}%)" if resonance else f"LỆCH PHA TRIỆT TIÊU (I = {factor * 100:.0f}%)"
        status_col = QColor(16, 185, 129) if resonance else QColor(244, 63, 94)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        badge_w = painter.fontMetrics().horizontalAdvance(status_txt) + 14.0
        badge_rect = QRectF(px + pw - badge_w - 12.0, py + 6.0, badge_w, 18.0)
        painter.setBrush(QBrush(status_col.darker(250)))
        painter.setPen(QPen(status_col, 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)
        painter.setPen(QPen(status_col.lighter(130)))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, status_txt)

        # Lattice Geometry parameters
        plane_y1 = py + 115.0  # Top plane (Row 1)
        # Visual plane spacing proportional to d_spacing_nm
        d_vis = max(40.0, min(80.0, self.d_spacing_nm * 250.0))
        plane_y2 = plane_y1 + d_vis  # Row 2
        plane_y3 = plane_y2 + d_vis  # Row 3

        # Center scattering atoms
        atom_x_center = px + pw * 0.50
        atom_spacing_x = 42.0

        # Draw 3 crystal planes (horizontal lines + periodic atom spheres)
        rows = [plane_y1, plane_y2, plane_y3]
        for r_idx, ry in enumerate(rows):
            # Plane dashed reference line
            painter.setPen(QPen(QColor(51, 65, 85, 160), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(px + 20.0, ry), QPointF(px + pw - 20.0, ry))

            # Row label
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(100, 116, 139)))
            painter.drawText(px + 22.0, ry - 4.0, f"Mặt {self.active_plane.label} #{r_idx + 1}")

            # Draw 9 atoms per row
            for col in range(-4, 5):
                ax = atom_x_center + col * atom_spacing_x
                if px + 10.0 <= ax <= px + pw - 10.0:
                    rad_grad = QRadialGradient(ax - 2.0, ry - 2.0, 9.0)
                    if (col == 0 and r_idx in (0, 1)):
                        # Key scattering atom targets
                        rad_grad.setColorAt(0.0, QColor(56, 189, 248, 255))
                        rad_grad.setColorAt(0.7, QColor(14, 116, 144, 230))
                        rad_grad.setColorAt(1.0, QColor(8, 47, 73, 220))
                    else:
                        rad_grad.setColorAt(0.0, QColor(148, 163, 184, 220))
                        rad_grad.setColorAt(0.7, QColor(71, 85, 105, 200))
                        rad_grad.setColorAt(1.0, QColor(30, 41, 59, 180))

                    painter.setBrush(QBrush(rad_grad))
                    painter.setPen(QPen(QColor(226, 232, 240, 160), 0.8))
                    painter.drawEllipse(QPointF(ax, ry), 7.0, 7.0)

        # Dimension marker: d_hkl
        dim_x = px + pw - 35.0
        painter.setPen(QPen(QColor(245, 158, 11), 1.4))
        painter.drawLine(QPointF(dim_x, plane_y1), QPointF(dim_x, plane_y2))
        painter.drawLine(QPointF(dim_x - 4.0, plane_y1), QPointF(dim_x + 4.0, plane_y1))
        painter.drawLine(QPointF(dim_x - 4.0, plane_y2), QPointF(dim_x + 4.0, plane_y2))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(dim_x - 52.0, (plane_y1 + plane_y2) / 2.0 + 4.0, f"d = {self.d_spacing_nm:.3f} nm")

        # Ray Geometry Calculations
        # Angle theta relative to horizontal surface
        th_rad = math.radians(self._incident_theta_deg)
        cos_th = max(0.1, math.cos(th_rad))
        sin_th = max(0.08, math.sin(th_rad))

        # Atom 1 on Plane 1: (atom_x_center, plane_y1)
        p1 = QPointF(atom_x_center, plane_y1)
        # Atom 2 on Plane 2: (atom_x_center, plane_y2)
        p2 = QPointF(atom_x_center, plane_y2)

        # Ray ray_length
        ray_len = 160.0
        dir_in = QPointF(cos_th, sin_th)
        dir_out = QPointF(cos_th, -sin_th)

        # Source origins of incoming rays
        src1 = QPointF(p1.x() - ray_len * dir_in.x(), p1.y() - ray_len * dir_in.y())
        src2 = QPointF(p2.x() - ray_len * dir_in.x(), p2.y() - ray_len * dir_in.y())

        # Destination endpoints of diffracted rays
        dest1 = QPointF(p1.x() + ray_len * dir_out.x(), p1.y() + ray_len * dir_out.y())
        dest2 = QPointF(p2.x() + ray_len * dir_out.x(), p2.y() + ray_len * dir_out.y())

        # Construct normal projections from p1 onto Ray 2 (Path difference segments)
        # Leg 1: From p1 perpendicular to incoming Ray 2
        # Vector p2 -> src2 has direction (-cos_th, -sin_th). Distance along Ray 2 is d_vis * sin_th
        proj_in = QPointF(p2.x() - (d_vis * sin_th) * cos_th, p2.y() - (d_vis * sin_th) * sin_th)
        # Leg 2: From p1 perpendicular to diffracted Ray 2
        proj_out = QPointF(p2.x() + (d_vis * sin_th) * cos_th, p2.y() - (d_vis * sin_th) * sin_th)

        # Draw path difference construction (perpendiculars and delta segment)
        painter.setPen(QPen(QColor(234, 179, 8, 200), 1.2, Qt.PenStyle.DotLine))
        painter.drawLine(p1, proj_in)
        painter.drawLine(p1, proj_out)

        # Highlight path difference segment along Ray 2: proj_in -> p2 -> proj_out
        painter.setPen(QPen(QColor(245, 158, 11, 230), 2.4))
        painter.drawLine(proj_in, p2)
        painter.drawLine(p2, proj_out)

        # Label Delta = 2*d*sin(theta)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.drawText(atom_x_center - 32.0, plane_y2 + 22.0, f"Δ = 2d·sin θ = {self.optical_path_difference_nm:.3f} nm")

        # Draw incident rays with animated waves
        self._paint_xray_beam(painter, src1, p1, self._wave_phase, is_outgoing=False, intensity=1.0)
        self._paint_xray_beam(painter, src2, p2, self._wave_phase, is_outgoing=False, intensity=1.0)

        # Draw diffracted rays with phase-modulated waves
        # Ray 1 outgoing phase
        phase_out1 = self._wave_phase
        # Ray 2 outgoing phase shifted by theoretical optical path difference
        phase_out2 = self._wave_phase - self.phase_difference_rad
        out_int = 0.25 + 0.75 * factor

        self._paint_xray_beam(painter, p1, dest1, phase_out1, is_outgoing=True, intensity=out_int)
        self._paint_xray_beam(painter, p2, dest2, phase_out2, is_outgoing=True, intensity=out_int)

        # Draw Incident Angle Arc at Atom 1
        arc_r = 45.0
        painter.setPen(QPen(QColor(56, 189, 248, 180), 1.2))
        painter.drawArc(QRectF(p1.x() - arc_r, p1.y() - arc_r, arc_r * 2.0, arc_r * 2.0), int((180.0 - self._incident_theta_deg) * 16), int(self._incident_theta_deg * 16))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(p1.x() - 65.0, p1.y() - 10.0, f"θ = {self._incident_theta_deg:.1f}°")

    def _paint_xray_beam(
        self,
        painter: QPainter,
        start_pt: QPointF,
        end_pt: QPointF,
        phase: float,
        is_outgoing: bool,
        intensity: float,
    ) -> None:
        """Renders an oscillating X-ray wave packet along a vector ray."""
        dx = end_pt.x() - start_pt.x()
        dy = end_pt.y() - start_pt.y()
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            return

        ux = dx / dist
        uy = dy / dist
        # Normal perpendicular vector
        nx = -uy
        ny = ux

        # Central ray line
        beam_alpha = int(120 * intensity)
        painter.setPen(QPen(QColor(56, 189, 248, beam_alpha), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(start_pt, end_pt)

        # Wave ripples
        wave_path = QPainterPath()
        num_pts = int(dist / 2.5)
        wavelength_px = 22.0  # Visual wave spatial period
        k = 2.0 * math.pi / wavelength_px
        amp = 4.5 * intensity

        first = True
        for i in range(num_pts + 1):
            s = (i / num_pts) * dist
            offset = amp * math.sin(k * s - phase)
            px = start_pt.x() + s * ux + offset * nx
            py = start_pt.y() + s * uy + offset * ny
            if first:
                wave_path.moveTo(px, py)
                first = False
            else:
                wave_path.lineTo(px, py)

        # Color: Cyan if constructive, Reddish if destructive
        if is_outgoing:
            if self.is_bragg_resonance:
                wave_col = QColor(16, 185, 129, int(230 * intensity))
            else:
                wave_col = QColor(244, 63, 94, int(180 * intensity))
        else:
            wave_col = QColor(56, 189, 248, 220)

        painter.setPen(QPen(wave_col, 1.6))
        painter.drawPath(wave_path)

    def _paint_diffractogram_panel(self, painter: QPainter) -> None:
        """
        Renders Top-Right Panel: Powder XRD Diffractogram Spectrum I(2theta).
        Visualizes calculated peaks with Debye-Scherrer broadening, goniometer detector line, and Miller labels.
        """
        dx = 500.0
        dy = 70.0
        dw = 440.0
        dh = 275.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(dx, dy, dw, dh), 10.0, 10.0)

        # Panel title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(dx + 12.0, dy + 18.0, "PHỔ NHIỄU XẠ BỘT I(2θ) & LÀM RỘNG VẠCH SCHERRER:")

        # Coordinate box for plotting
        gx = dx + 42.0
        gy = dy + 35.0
        gw = dw - 58.0
        gh = dh - 65.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(QRectF(gx, gy, gw, gh))

        # Diffractometer axis limits: 2theta from 10.0 deg to 95.0 deg
        min_2th = 10.0
        max_2th = 95.0
        span_2th = max_2th - min_2th

        def map_x(th2: float) -> float:
            return gx + ((th2 - min_2th) / span_2th) * gw

        def map_y(intensity_val: float) -> float:
            return gy + gh - (intensity_val / 110.0) * gh

        # Grid lines (every 10 deg)
        painter.setFont(QFont("Segoe UI", 6))
        for t2 in range(10, 100, 10):
            cx = map_x(float(t2))
            painter.setPen(QPen(QColor(30, 41, 59, 140), 0.8, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(cx, gy), QPointF(cx, gy + gh))
            painter.setPen(QPen(QColor(100, 116, 139)))
            painter.drawText(cx - 10.0, gy + gh + 14.0, f"{t2}°")

        # X-axis label
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(gx + gw / 2.0 - 45.0, gy + gh + 26.0, "Góc Nhiễu xạ 2θ (Độ)")

        # Y-axis label
        painter.drawText(dx + 6.0, gy + 12.0, "I (cps)")

        # Continuous Diffractogram Curve
        spec_path = QPainterPath()
        spec_pts = 180
        first_pt = True

        for i in range(spec_pts + 1):
            t2 = min_2th + (i / spec_pts) * span_2th
            intens = self.calculate_diffractogram_intensity(t2)
            px_val = map_x(t2)
            py_val = map_y(intens)
            if first_pt:
                spec_path.moveTo(px_val, py_val)
                first_pt = False
            else:
                spec_path.lineTo(px_val, py_val)

        # Fill under curve
        fill_path = QPainterPath(spec_path)
        fill_path.lineTo(gx + gw, gy + gh)
        fill_path.lineTo(gx, gy + gh)
        fill_path.closeSubpath()

        spec_grad = QLinearGradient(gx, gy, gx, gy + gh)
        spec_grad.setColorAt(0.0, QColor(14, 165, 233, 120))
        spec_grad.setColorAt(0.7, QColor(14, 165, 233, 30))
        spec_grad.setColorAt(1.0, QColor(14, 165, 233, 0))
        painter.setBrush(QBrush(spec_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Plot curve stroke
        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(spec_path)

        # Peak markers and Miller labels (hkl)
        for p_idx, plane in enumerate(self._preset.allowed_planes):
            pk_2th = plane.two_theta_deg(self._wavelength_nm, self._lattice_a_nm)
            if pk_2th is None or not (min_2th <= pk_2th <= max_2th):
                continue
            pk_x = map_x(pk_2th)
            pk_y = map_y(plane.relative_intensity + 2.0)

            is_active = (p_idx == self._active_plane_idx)
            lbl_col = QColor(245, 158, 11) if is_active else QColor(226, 232, 240)

            # Circle on top of peak
            painter.setBrush(QBrush(lbl_col))
            painter.setPen(QPen(QColor(0, 0, 0), 1.0))
            painter.drawEllipse(QPointF(pk_x, pk_y), 3.5, 3.5)

            # Text label
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(lbl_col))
            painter.drawText(pk_x - 12.0, pk_y - 6.0, plane.label)

        # Live Goniometer Detector Angle vertical indicator line: current 2theta
        cur_2th = self.current_two_theta_deg
        if min_2th <= cur_2th <= max_2th:
            det_x = map_x(cur_2th)
            painter.setPen(QPen(QColor(245, 158, 11, 230), 1.8, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(det_x, gy), QPointF(det_x, gy + gh))

            # Detector badge on top
            det_rect = QRectF(det_x - 22.0, gy - 16.0, 44.0, 14.0)
            painter.setBrush(QBrush(QColor(245, 158, 11)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(det_rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(det_rect, Qt.AlignmentFlag.AlignCenter, f"{cur_2th:.1f}°")

        # Scherrer Broadening Readout Overlay inside diffractogram
        scherrer_box = QRectF(gx + 8.0, gy + 8.0, 175.0, 38.0)
        painter.setBrush(QBrush(QColor(9, 14, 25, 220)))
        painter.setPen(QPen(QColor(168, 85, 247, 160), 1.0))
        painter.drawRoundedRect(scherrer_box, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(192, 132, 252)))
        painter.drawText(scherrer_box.x() + 6.0, scherrer_box.y() + 14.0, f"Độ rộng Scherrer β: {self.scherrer_fwhm_deg:.3f}° FWHM")
        painter.drawText(scherrer_box.x() + 6.0, scherrer_box.y() + 28.0, f"Kích thước hạt D: {self._crystallite_d_nm:.1f} nm (K = 0.9)")

    def _paint_unit_cell_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Left Panel: 3D Isometric Cubic Unit Cell & Active Miller Plane Slice.
        Visualizes the cubic crystal lattice and the intersecting (hkl) plane.
        """
        ux = 20.0
        uy = 360.0
        uw = 460.0
        uh = 220.0

        painter.setBrush(QBrush(QColor(7, 11, 22)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(QRectF(ux, uy, uw, uh), 10.0, 10.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ux + 12.0, uy + 18.0, f"Ô CƠ SỞ 3D & MẶT PHẲNG MILLER {self.active_plane.label}:")

        # Isometric 3D projection parameters
        # Origin at center of 3D display region
        cx = ux + 130.0
        cy = uy + 130.0
        side = 80.0  # Cube visual edge length

        # Isometric basis vectors: x -> down-left, y -> down-right, z -> straight up
        iso_x = QPointF(-side * math.cos(math.radians(30)), side * math.sin(math.radians(30)))
        iso_y = QPointF(side * math.cos(math.radians(30)), side * math.sin(math.radians(30)))
        iso_z = QPointF(0.0, -side)

        def proj(x_frac: float, y_frac: float, z_frac: float) -> QPointF:
            return QPointF(
                cx + x_frac * iso_x.x() + y_frac * iso_y.x() + z_frac * iso_z.x(),
                cy + x_frac * iso_x.y() + y_frac * iso_y.y() + z_frac * iso_z.y(),
            )

        # 8 vertices of cube [0, 1]
        v000 = proj(0, 0, 0)
        v100 = proj(1, 0, 0)
        v010 = proj(0, 1, 0)
        v110 = proj(1, 1, 0)
        v001 = proj(0, 0, 1)
        v101 = proj(1, 0, 1)
        v011 = proj(0, 1, 1)
        v111 = proj(1, 1, 1)

        # Draw wireframe cube edges (dashed rear edges, solid front)
        rear_pen = QPen(QColor(51, 65, 85, 140), 1.0, Qt.PenStyle.DashLine)
        front_pen = QPen(QColor(100, 116, 139), 1.4)

        # Rear edges from v000
        painter.setPen(rear_pen)
        painter.drawLine(v000, v100)
        painter.drawLine(v000, v010)
        painter.drawLine(v000, v001)

        # Draw translucent Miller Plane polygon inside cube
        h, k, l = self.active_plane.h, self.active_plane.k, self.active_plane.l
        plane_poly = QPolygonF()

        if (h, k, l) == (1, 1, 1):
            # Cuts through (1,0,0), (0,1,0), (0,0,1)
            plane_poly.append(v100)
            plane_poly.append(v010)
            plane_poly.append(v001)
        elif (h, k, l) == (1, 1, 0):
            # Cuts through (1,0,0), (0,1,0), (0,1,1), (1,0,1)
            plane_poly.append(v100)
            plane_poly.append(v010)
            plane_poly.append(v011)
            plane_poly.append(v101)
        elif (h, k, l) == (2, 0, 0):
            # Mid-plane at x = 0.5
            plane_poly.append(proj(0.5, 0, 0))
            plane_poly.append(proj(0.5, 1, 0))
            plane_poly.append(proj(0.5, 1, 1))
            plane_poly.append(proj(0.5, 0, 1))
        elif (h, k, l) == (2, 2, 0):
            # Diagonal mid-plane
            plane_poly.append(proj(0.5, 0, 0))
            plane_poly.append(proj(0, 0.5, 0))
            plane_poly.append(proj(0, 0.5, 1))
            plane_poly.append(proj(0.5, 0, 1))
        elif (h, k, l) == (3, 1, 1):
            # Acute plane cutting (0.33, 0, 0), (0, 1, 0), (0, 0, 1)
            plane_poly.append(proj(0.33, 0, 0))
            plane_poly.append(proj(0, 1, 0))
            plane_poly.append(proj(0, 0, 1))
        else:
            # Default face or generic cut
            plane_poly.append(v100)
            plane_poly.append(v010)
            plane_poly.append(v001)

        # Draw shaded Miller plane
        painter.setBrush(QBrush(QColor(245, 158, 11, 100)))
        painter.setPen(QPen(QColor(245, 158, 11, 220), 1.6))
        painter.drawPolygon(plane_poly)

        # Draw front wireframe edges
        painter.setPen(front_pen)
        painter.drawLine(v100, v110)
        painter.drawLine(v010, v110)
        painter.drawLine(v100, v101)
        painter.drawLine(v010, v011)
        painter.drawLine(v001, v101)
        painter.drawLine(v001, v011)
        painter.drawLine(v101, v111)
        painter.drawLine(v011, v111)
        painter.drawLine(v110, v111)

        # Draw corner and basis atoms
        atom_pts = [v000, v100, v010, v110, v001, v101, v011, v111]
        for ap in atom_pts:
            painter.setBrush(QBrush(QColor(56, 189, 248)))
            painter.setPen(QPen(QColor(0, 0, 0), 1.0))
            painter.drawEllipse(ap, 3.5, 3.5)

        # Extra basis atoms based on crystal type
        ctype = self._preset.crystal_type
        if ctype in ("FCC", "ROCK_SALT", "DIAMOND_CUBIC", "NANOCRYSTAL"):
            # Face centers
            face_centers = [
                proj(0.5, 0.5, 0.0),
                proj(0.5, 0.5, 1.0),
                proj(0.5, 0.0, 0.5),
                proj(0.5, 1.0, 0.5),
                proj(0.0, 0.5, 0.5),
                proj(1.0, 0.5, 0.5),
            ]
            for fc in face_centers:
                painter.setBrush(QBrush(QColor(16, 185, 129)))
                painter.setPen(QPen(QColor(0, 0, 0), 0.8))
                painter.drawEllipse(fc, 3.0, 3.0)

        if ctype == "BCC":
            # Body center
            bc = proj(0.5, 0.5, 0.5)
            painter.setBrush(QBrush(QColor(234, 179, 8)))
            painter.setPen(QPen(QColor(0, 0, 0), 1.0))
            painter.drawEllipse(bc, 4.0, 4.0)

        # Crystallographic Mathematical Formulations on Right Side of Panel
        rx = ux + 250.0
        ry = uy + 38.0

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(rx, ry, "CÔNG THỨC KHOẢNG CÁCH GIAN MẶT:")

        sq = self.active_plane.sum_sq
        lines = [
            f"• Chỉ số Miller: ({self.active_plane.h} {self.active_plane.k} {self.active_plane.l})",
            f"• Tổng bình phương: h² + k² + l² = {sq}",
            f"• d_hkl = a / √(h² + k² + l²)",
            f"  = {self._lattice_a_nm * 10.0:.3f} / √{sq} = {self.d_spacing_angstrom:.3f} Å",
            f"• Khoảng cách d: {self.d_spacing_nm:.4f} nm",
            f"• Mạng tinh thể: {self._preset.crystal_type}",
            f"• Hằng số mạng a: {self._lattice_a_nm * 10.0:.3f} Å ({self._lattice_a_nm:.4f} nm)",
        ]

        row_y = ry + 20.0
        for line in lines:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(226, 232, 240) if "d_hkl" in line or "d:" in line else QColor(148, 163, 184)))
            painter.drawText(rx, row_y, line)
            row_y += 18.0

    def _paint_crystallography_metrics_panel(self, painter: QPainter) -> None:
        """
        Renders Bottom-Right Panel: Rigorous Crystallography & Diffraction Metrics.
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
        painter.drawText(mx + 12.0, my + 18.0, "THÔNG SỐ VẬT LÝ & ĐỊNH LUẬT BRAGG:")

        # Compute key metrics
        metrics = [
            ("Nguồn bức xạ X-Ray (Wavelength λ):", f"λ = {self._wavelength_nm:.5f} nm ({self._wavelength_nm * 10.0:.3f} Å)"),
            ("Góc phản xạ Bragg lý thuyết (θ_B):", f"θ_B = {self.theoretical_bragg_theta_deg:.2f}° (2θ_B = {self.theoretical_two_theta_deg:.2f}°)"),
            ("Góc tới hiện tại (Current Incident θ):", f"θ = {self._incident_theta_deg:.2f}° (2θ = {self.current_two_theta_deg:.2f}°)"),
            ("Hiệu quang trình (Path Difference Δ):", f"Δ = 2d·sin(θ) = {self.optical_path_difference_nm:.4f} nm"),
            ("Tỷ số Quang trình / Bước sóng (Δ/λ):", f"Δ/λ = {self.optical_path_difference_nm / self._wavelength_nm:.3f} ({'Cộng hưởng nguyên' if self.is_bragg_resonance else 'Lệch pha'})"),
            ("Độ rộng phổ Scherrer (FWHM β):", f"β = {self.scherrer_fwhm_deg:.3f}° (D = {self._crystallite_d_nm:.1f} nm)"),
            ("Trạng thái Tán xạ (Interference State):", f"{'GIAO THOA CỰC ĐẠI (Đỉnh Phản xạ)' if self.is_bragg_resonance else 'GIAO THOA TRIỆT TIÊU'}"),
        ]

        row_y = my + 40.0
        line_h = 24.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            val_col = QColor(16, 185, 129) if "CỰC ĐẠI" in val or "Cộng hưởng" in val else QColor(226, 232, 240)
            painter.setPen(QPen(val_col))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture control buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_plane", f"Mặt {self.active_plane.label}", QColor(168, 85, 247)),
            ("btn_th_down", "θ -", QColor(51, 65, 85)),
            ("btn_th_up", "θ +", QColor(51, 65, 85)),
            ("btn_tune", "Chuẩn θ_B", QColor(30, 41, 59)),
            ("btn_xray", "Cu-Kα" if abs(self._wavelength_nm - 0.15406) < 0.001 else "Mo-Kα", QColor(51, 65, 85)),
            ("btn_d_down", "D -", QColor(51, 65, 85)),
            ("btn_d_up", "D +", QColor(51, 65, 85)),
            ("btn_scan", "Dừng quét" if self._is_auto_scanning else "Quét 2θ", QColor(245, 158, 11)),
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
    # Touch & Mouse Air Gesture Event Handling
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
            # Cycle through presets
            keys = list(BRAGG_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_plane":
            # Cycle through planes
            self.set_active_plane_index(self._active_plane_idx + 1)
        elif key == "btn_th_down":
            self.set_incident_theta(self._incident_theta_deg - 0.25)
        elif key == "btn_th_up":
            self.set_incident_theta(self._incident_theta_deg + 0.25)
        elif key == "btn_tune":
            # Direct tune to theoretical Bragg angle
            self.set_incident_theta(self.theoretical_bragg_theta_deg)
        elif key == "btn_xray":
            self.toggle_wavelength()
        elif key == "btn_d_down":
            self.adjust_crystallite_size(-5.0)
        elif key == "btn_d_up":
            self.adjust_crystallite_size(+5.0)
        elif key == "btn_scan":
            self.toggle_auto_scan()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "crystal_type": self._preset.crystal_type,
            "lattice_a_nm": self._lattice_a_nm,
            "wavelength_nm": self._wavelength_nm,
            "crystallite_d_nm": self._crystallite_d_nm,
            "active_plane_label": self.active_plane.label,
            "d_spacing_nm": self.d_spacing_nm,
            "bragg_theta_deg": self.theoretical_bragg_theta_deg,
            "two_theta_deg": self.theoretical_two_theta_deg,
            "scherrer_fwhm_deg": self.scherrer_fwhm_deg,
        }
        self.stamp_requested.emit(data)
