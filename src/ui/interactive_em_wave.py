"""
AirTeach AI — Interactive Physics & Electrodynamics: Maxwell's Electromagnetic Waves Simulator.
Simulates transverse electromagnetic wave propagation (E ⊥ B ⊥ k), Poynting energy flux vector (S = E x B / μ₀),
wave polarization states (Linear, Circular RCP/LCP, Elliptical), Stokes parameters,
radiation pressure, and Quarter-Wave / Half-Wave birefringent retarder plates.
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
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


# Fundamental physical constants (SI)
C_LIGHT = 299792458.0          # m/s
EPSILON_0 = 8.8541878128e-12   # F/m
MU_0 = 1.25663706212e-6        # H/m or N/A²


@dataclass(frozen=True)
class EMWavePreset:
    code: str
    title_vi: str
    badge_label: str
    e0x_v_m: float           # Electric field amplitude X (V/m)
    e0y_v_m: float           # Electric field amplitude Y (V/m)
    delta_rad: float         # Phase shift delta between Ey and Ex (radians)
    wavelength_nm: float     # Wavelength in nanometers (e.g. 532 nm Green, 632.8 nm He-Ne Red)
    polarization_name: str   # "Tuyến tính (Linear)", "Tròn phải (RCP)", "Tròn trái (LCP)", "Elip (Elliptical)"
    retarder_plate: Optional[str] # None, "QUARTER_WAVE_PLATE", "HALF_WAVE_PLATE"
    description: str


EM_PRESETS: dict[str, EMWavePreset] = {
    "LINEAR_POLARIZATION_45_DEG": EMWavePreset(
        code="LINEAR_POLARIZATION_45_DEG",
        title_vi="Phân cực Tuyến tính 45° — Dao động Cùng pha (δ = 0)",
        badge_label="LINEAR 45°",
        e0x_v_m=100.0,
        e0y_v_m=100.0,
        delta_rad=0.0,
        wavelength_nm=532.0,  # Green laser
        polarization_name="Tuyến tính (Linear 45°)",
        retarder_plate=None,
        description="Vectơ điện trường E_x và E_y dao động hoàn toàn cùng pha (δ = 0). Vectơ tổng E⃗ nằm trên mặt phẳng nghiêng 45° so với trục tọa độ. Vectơ Poynting S⃗ truyền thẳng theo phương +z.",
    ),
    "CIRCULAR_RIGHT_HANDED_RCP": EMWavePreset(
        code="CIRCULAR_RIGHT_HANDED_RCP",
        title_vi="Phân cực Tròn Phải (RCP) — Lệch pha δ = -π/2 & Xoắn ốc Theo chiều Kim Đồng hồ",
        badge_label="CIRCULAR RCP",
        e0x_v_m=100.0,
        e0y_v_m=100.0,
        delta_rad=-math.pi / 2.0,  # -90°
        wavelength_nm=632.8,  # He-Ne Red laser
        polarization_name="Tròn Phải (RCP)",
        retarder_plate=None,
        description="Biên độ E_0x = E_0y và lệch pha δ = -π/2. Đầu mút vectơ điện trường E⃗ quét thành đường xoắn ốc theo chiều kim đồng hồ khi nhìn ngược hướng truyền sóng, mang mômen động lượng spin quang học +ℏ.",
    ),
    "CIRCULAR_LEFT_HANDED_LCP": EMWavePreset(
        code="CIRCULAR_LEFT_HANDED_LCP",
        title_vi="Phân cực Tròn Trái (LCP) — Lệch pha δ = +π/2 & Xoắn ốc Ngược chiều Kim Đồng hồ",
        badge_label="CIRCULAR LCP",
        e0x_v_m=100.0,
        e0y_v_m=100.0,
        delta_rad=math.pi / 2.0,   # +90°
        wavelength_nm=632.8,
        polarization_name="Tròn Trái (LCP)",
        retarder_plate=None,
        description="Biên độ E_0x = E_0y và lệch pha δ = +π/2. Đầu mút vectơ E⃗ xoay ngược chiều kim đồng hồ, ứng dụng trong truyền dẫn vô tuyến vệ tinh chống suy hao do phản xạ ion hóa.",
    ),
    "ELLIPTICAL_GENERAL_POLARIZATION": EMWavePreset(
        code="ELLIPTICAL_GENERAL_POLARIZATION",
        title_vi="Phân cực Elip Tổng quát — Biên độ Lệch & Pha Tùy ý (δ = π/4)",
        badge_label="ELLIPTICAL",
        e0x_v_m=120.0,
        e0y_v_m=70.0,
        delta_rad=math.pi / 4.0,   # 45°
        wavelength_nm=450.0,  # Blue laser
        polarization_name="Elip Tổng quát (Elliptical)",
        retarder_plate=None,
        description="Dạng phân cực tổng quát nhất khi E_0x ≠ E_0y hoặc góc lệch pha δ bất kỳ khác 0, ±π/2, π. Quỹ đạo đầu mút vẽ nên hình elip nghiêng trên mặt phẳng chắn ngang.",
    ),
    "QUARTER_WAVE_RETARDER_CRYSTAL": EMWavePreset(
        code="QUARTER_WAVE_RETARDER_CRYSTAL",
        title_vi="Bản làm chậm Một phần tư bước sóng (λ/4) — Biến đổi Tuyến tính thành Phân cực Tròn",
        badge_label="λ/4 RETARDER",
        e0x_v_m=100.0,
        e0y_v_m=100.0,
        delta_rad=-math.pi / 2.0,
        wavelength_nm=532.0,
        polarization_name="Biến đổi qua Bản λ/4",
        retarder_plate="QUARTER_WAVE_PLATE",
        description="Tinh thể lưỡng chiết quang học (Birefringent crystal) có trục quang học nhanh và chậm lệch pha đúng Δφ = 2π(n_e - n_o)d/λ = π/2, chuyển sóng phân cực phẳng thành phân cực tròn hoàn hảo.",
    ),
}


class InteractiveElectromagneticWaveItem(QGraphicsObject):
    """
    Interactive Maxwell's Electromagnetic Waves & Poynting Vector Simulator.
    Simulates transverse E-field and B-field vector oscillations (E ⊥ B ⊥ k),
    Poynting vector energy flux S = (E x B) / μ₀, radiation pressure,
    and Stokes polarization parameters in 3D isometric projection.
    """

    preset_changed = Signal(str)
    wave_parameters_changed = Signal(float, float, float)  # e0, wavelength_nm, delta_rad
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

        # Simulation Preset
        self._active_preset_code: str = "LINEAR_POLARIZATION_45_DEG"
        self._preset: EMWavePreset = EM_PRESETS[self._active_preset_code]

        # Wave Parameters
        self._e0x: float = self._preset.e0x_v_m
        self._e0y: float = self._preset.e0y_v_m
        self._delta: float = self._preset.delta_rad
        self._wavelength_nm: float = self._preset.wavelength_nm

        # Simulation Time & Animation
        self._sim_time_fs: float = 0.0   # femtoseconds (1 fs = 1e-15 s)
        self._is_running: bool = True
        self._show_b_field: bool = True
        self._show_poynting: bool = True
        self._show_polarization_screen: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 60 Hz animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physical Properties & Computations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> EMWavePreset:
        return self._preset

    @property
    def frequency_hz(self) -> float:
        """Frequency f = c / lambda (Hz)."""
        lam_m = self._wavelength_nm * 1e-9
        return C_LIGHT / lam_m

    @property
    def period_fs(self) -> float:
        """Wave period T in femtoseconds (1 fs = 1e-15 s)."""
        return (1.0 / self.frequency_hz) * 1e15

    @property
    def b0_tesla(self) -> float:
        """Magnetic field amplitude B0 = E0 / c (Tesla)."""
        e0_total = math.hypot(self._e0x, self._e0y)
        return e0_total / C_LIGHT

    @property
    def b0_microtesla(self) -> float:
        """Magnetic field amplitude in microtesla (μT)."""
        return self.b0_tesla * 1e6

    @property
    def poynting_intensity_w_m2(self) -> float:
        """Time-averaged Poynting intensity <S> = (1/2) * c * ε₀ * E₀² (W/m²)."""
        e0_total_sq = self._e0x**2 + self._e0y**2
        return 0.5 * C_LIGHT * EPSILON_0 * e0_total_sq

    @property
    def radiation_pressure_micro_pa(self) -> float:
        """Radiation pressure on absorbing surface P_rad = <S> / c (μPa)."""
        p_pa = self.poynting_intensity_w_m2 / C_LIGHT
        return p_pa * 1e6

    @property
    def stokes_parameters(self) -> tuple[float, float, float, float]:
        """
        Calculates Stokes parameters (S0, S1, S2, S3):
        S0 = E0x² + E0y²
        S1 = E0x² - E0y²
        S2 = 2 E0x E0y cos(δ)
        S3 = 2 E0x E0y sin(δ)
        """
        s0 = self._e0x**2 + self._e0y**2
        s1 = self._e0x**2 - self._e0y**2
        s2 = 2.0 * self._e0x * self._e0y * math.cos(self._delta)
        s3 = 2.0 * self._e0x * self._e0y * math.sin(self._delta)
        return (s0, s1, s2, s3)

    def get_polarization_classification(self) -> str:
        """Classifies polarization state from Stokes parameters."""
        s0, s1, s2, s3 = self.stokes_parameters
        if s0 < 1e-6:
            return "Không phân cực"

        norm_s3 = s3 / s0
        if math.isclose(norm_s3, 0.0, abs_tol=0.08):
            angle_deg = 0.5 * math.degrees(math.atan2(s2, s1))
            return f"Tuyến tính ({angle_deg:+.1f}°)"
        elif math.isclose(norm_s3, -1.0, abs_tol=0.08):
            return "Phân cực Tròn Phải (RCP, -90°)"
        elif math.isclose(norm_s3, 1.0, abs_tol=0.08):
            return "Phân cực Tròn Trái (LCP, +90°)"
        elif norm_s3 < 0:
            return "Phân cực Elip Phải (Right Elliptical)"
        else:
            return "Phân cực Elip Trái (Left Elliptical)"

    # -------------------------------------------------------------------------
    # Simulation Animation Tick
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if self._is_running:
            # Advance time by ~0.08 fs per tick
            self._sim_time_fs += 0.08
        self.update()

    # -------------------------------------------------------------------------
    # Presets & Interactive Steppers
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads an electromagnetic wave preset."""
        if preset_code not in EM_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = EM_PRESETS[preset_code]
        self._e0x = self._preset.e0x_v_m
        self._e0y = self._preset.e0y_v_m
        self._delta = self._preset.delta_rad
        self._wavelength_nm = self._preset.wavelength_nm
        self.preset_changed.emit(preset_code)
        self.update()

    def step_amplitude(self, delta_v_m: float) -> None:
        """Steps electric field amplitude E0."""
        new_ex = max(10.0, min(500.0, self._e0x + delta_v_m))
        new_ey = max(10.0, min(500.0, self._e0y + delta_v_m))
        self._e0x = new_ex
        self._e0y = new_ey
        self.update()

    def step_phase_delta(self, delta_rad: float) -> None:
        """Steps phase difference delta."""
        self._delta = (self._delta + delta_rad) % (2.0 * math.pi)
        if self._delta > math.pi:
            self._delta -= 2.0 * math.pi
        self.update()

    def step_wavelength(self, delta_nm: float) -> None:
        """Steps optical wavelength."""
        self._wavelength_nm = max(380.0, min(780.0, self._wavelength_nm + delta_nm))
        self.update()

    def toggle_animation(self) -> None:
        """Toggles wave propagation animation."""
        self._is_running = not self._is_running
        self.update()

    def toggle_b_field(self) -> None:
        """Toggles magnetic field vector display."""
        self._show_b_field = not self._show_b_field
        self.update()

    def toggle_poynting(self) -> None:
        """Toggles Poynting vector display."""
        self._show_poynting = not self._show_poynting
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Pinch Handling
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_e_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_e_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_phase_rcp"] = QRectF(290.0, bar_y, 65.0, 34.0)
        self._buttons["btn_phase_lcp"] = QRectF(360.0, bar_y, 65.0, 34.0)
        self._buttons["btn_toggle_b"] = QRectF(432.0, bar_y, 75.0, 34.0)
        self._buttons["btn_toggle_s"] = QRectF(513.0, bar_y, 85.0, 34.0)
        self._buttons["btn_p_rcp"] = QRectF(604.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_retarder"] = QRectF(685.0, bar_y, 80.0, 34.0)
        self._buttons["btn_play"] = QRectF(771.0, bar_y, 65.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(842.0, bar_y, 88.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        """Handles button clicks."""
        preset_keys = list(EM_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_e_down":
            self.step_amplitude(-20.0)
        elif key == "btn_e_up":
            self.step_amplitude(20.0)
        elif key == "btn_phase_rcp":
            self._delta = -math.pi / 2.0
        elif key == "btn_phase_lcp":
            self._delta = math.pi / 2.0
        elif key == "btn_toggle_b":
            self.toggle_b_field()
        elif key == "btn_toggle_s":
            self.toggle_poynting()
        elif key == "btn_p_rcp":
            self.load_preset("CIRCULAR_RIGHT_HANDED_RCP")
        elif key == "btn_p_retarder":
            self.load_preset("QUARTER_WAVE_RETARDER_CRYSTAL")
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
        """Emits signal requesting permanent vector stamping of current wave diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "e0x": self._e0x,
            "e0y": self._e0y,
            "delta_rad": self._delta,
            "wavelength_nm": self._wavelength_nm,
            "intensity_w_m2": self.poynting_intensity_w_m2,
            "b0_ut": self.b0_microtesla,
            "rad_pressure_upa": self.radiation_pressure_micro_pa,
            "polarization_name": self.get_polarization_classification(),
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
        self._paint_3d_wave_viewport(painter)
        self._paint_transverse_polarization_screen(painter)
        self._paint_analytical_metrics(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(10, 15, 26, 252))
        grad.setColorAt(1.0, QColor(6, 10, 18, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"ĐIỆN ĐỘNG LỰC HỌC MAXWELL: SÓNG ĐIỆN TỪ & VECTƠ POYNTING – {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(22, 48, f"Phương trình sóng: ∇²E⃗ = (1/c²)∂²E⃗/∂t² | E⃗ ⊥ B⃗ ⊥ k⃗ | Cường độ năng lượng S⃗ = (1/μ₀)(E⃗ × B⃗) | Bước sóng λ = {self._wavelength_nm:.1f} nm")

    def _paint_3d_wave_viewport(self, painter: QPainter) -> None:
        """
        Renders 3D isometric view of transverse E-field and B-field oscillations along +z axis:
        z-axis: from left-back to right-front
        x-axis (E-field): vertical axis
        y-axis (B-field): horizontal transverse axis
        """
        vx = 20.0
        vy = 60.0
        vw = 610.0
        vh = 440.0

        # Viewport background
        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(vx, vy, vw, vh, 8.0)

        # Coordinate axes origin & direction
        # z: propagation axis (along line from (vx + 50, vy + vh*0.55) to (vx + vw - 40, vy + vh*0.55))
        origin_x = vx + 50.0
        origin_y = vy + vh * 0.55
        axis_len = vw - 90.0

        # Draw Z-axis (Wave propagation k)
        painter.setPen(QPen(QColor(148, 163, 184, 180), 1.5))
        painter.drawLine(origin_x, origin_y, origin_x + axis_len, origin_y)

        # Draw k-vector arrow at end
        painter.setPen(QPen(QColor(245, 158, 11), 2.2))
        end_x = origin_x + axis_len
        painter.drawLine(end_x, origin_y, end_x - 10.0, origin_y - 6.0)
        painter.drawLine(end_x, origin_y, end_x - 10.0, origin_y + 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.drawText(end_x + 6.0, origin_y + 4.0, "k⃗ (Lan truyền)")

        # Wave calculation parameters
        # Two complete wavelengths across the viewport
        num_wavelengths = 2.0
        k = (2.0 * math.pi * num_wavelengths) / axis_len
        omega = 2.0 * math.pi / 2.0  # Normalized angular frequency
        t = self._sim_time_fs

        # Scale amplitudes for pixel rendering
        scale_e = 55.0 / max(1.0, math.hypot(self._e0x, self._e0y))
        scale_b = 40.0

        num_samples = 45
        step_z = axis_len / float(num_samples)

        # Retarder plate indicator if active
        if self._preset.retarder_plate == "QUARTER_WAVE_PLATE":
            plate_x = origin_x + axis_len * 0.45
            plate_rect = QRectF(plate_x - 10.0, origin_y - 80.0, 20.0, 160.0)
            painter.setBrush(QBrush(QColor(168, 85, 247, 60)))
            painter.setPen(QPen(QColor(168, 85, 247), 1.5, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(plate_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(168, 85, 247)))
            painter.drawText(plate_x - 25.0, origin_y - 85.0, "Bản λ/4 (Retarder)")

        # Plot discrete vectors along Z
        for i in range(num_samples + 1):
            z = i * step_z
            px = origin_x + z
            phase = k * z - omega * t

            ex = self._e0x * math.cos(phase)
            ey = self._e0y * math.cos(phase + self._delta)

            # 3D isometric projection:
            # Ex is along vertical axis (upwards in screen: -y)
            # Ey is along tilted transverse axis (down-right at 30°: +x*0.5, +y*0.3)
            screen_ex = -ex * scale_e
            screen_ey_x = ey * scale_e * 0.5
            screen_ey_y = ey * scale_e * 0.3

            # Total Electric field vector E(z, t)
            tip_e_x = px + screen_ey_x
            tip_e_y = origin_y + screen_ex + screen_ey_y

            # Draw E-field vector (Red / Crimson)
            painter.setPen(QPen(QColor(239, 68, 68, 180), 1.6))
            painter.drawLine(px, origin_y, tip_e_x, tip_e_y)

            # Draw E-arrow head
            painter.setBrush(QBrush(QColor(239, 68, 68)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(tip_e_x, tip_e_y), 2.5, 2.5)

            # Draw B-field vector (Cyan / Blue, B = (k x E) / c, orthogonal in transverse plane)
            if self._show_b_field:
                # Bx = -Ey/c, By = Ex/c
                bx = -ey * scale_b * 0.01
                by = ex * scale_b * 0.01

                tip_b_x = px + by * 2.0
                tip_b_y = origin_y - bx * 1.5

                painter.setPen(QPen(QColor(56, 189, 248, 140), 1.2))
                painter.drawLine(px, origin_y, tip_b_x, tip_b_y)
                painter.setBrush(QBrush(QColor(56, 189, 248)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(tip_b_x, tip_b_y), 2.0, 2.0)

            # Draw Poynting vector S (Yellow arrow along +z at intervals)
            if self._show_poynting and (i % 9 == 0) and (i > 0):
                s_len = 22.0
                painter.setPen(QPen(QColor(245, 158, 11, 220), 2.0))
                painter.drawLine(px, origin_y - 25.0, px + s_len, origin_y - 25.0)
                painter.drawLine(px + s_len, origin_y - 25.0, px + s_len - 5.0, origin_y - 28.0)
                painter.drawLine(px + s_len, origin_y - 25.0, px + s_len - 5.0, origin_y - 22.0)
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                painter.setPen(QPen(QColor(245, 158, 11)))
                painter.drawText(px + 4.0, origin_y - 30.0, "S⃗")

        # Legend Overlay
        leg_x = vx + 16.0
        leg_y = vy + 18.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))

        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.drawText(leg_x, leg_y, "■ Vectơ Điện trường E⃗ (V/m)")

        if self._show_b_field:
            painter.setPen(QPen(QColor(56, 189, 248)))
            painter.drawText(leg_x + 150.0, leg_y, "■ Vectơ Từ trường B⃗ (μT)")

        if self._show_poynting:
            painter.setPen(QPen(QColor(245, 158, 11)))
            painter.drawText(leg_x + 290.0, leg_y, "■ Dòng Năng lượng Poynting S⃗ (W/m²)")

    def _paint_transverse_polarization_screen(self, painter: QPainter) -> None:
        """
        Renders transverse projection screen (looking down the +z axis):
        Traces Lissajous trajectory (Ex vs Ey) showing Linear, Circular, or Elliptical locus.
        """
        sx = 645.0
        sy = 60.0
        sw = 285.0
        sh = 200.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(sx, sy, sw, sh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(sx + 12.0, sy + 20.0, "MẶT PHẲNG PHÂN CỰC NGANG (EX - EY):")

        # Center of polarization crosshair
        center_x = sx + sw / 2.0
        center_y = sy + sh / 2.0 + 8.0
        axis_radius = 65.0

        # Crosshair axes
        painter.setPen(QPen(QColor(51, 65, 85), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(center_x - axis_radius, center_y, center_x + axis_radius, center_y)
        painter.drawLine(center_x, center_y - axis_radius, center_x, center_y + axis_radius)

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(center_x + axis_radius + 4.0, center_y + 3.0, "Ey")
        painter.drawText(center_x - 8.0, center_y - axis_radius - 4.0, "Ex")

        # Draw full polarization orbit path (Lissajous ellipse / circle / line)
        max_amp = max(1.0, math.hypot(self._e0x, self._e0y))
        scale = 55.0 / max_amp

        orbit_path = QPainterPath()
        for deg in range(361):
            theta = math.radians(deg)
            x_val = self._e0y * math.cos(theta + self._delta) * scale
            y_val = -self._e0x * math.cos(theta) * scale
            if deg == 0:
                orbit_path.moveTo(center_x + x_val, center_y + y_val)
            else:
                orbit_path.lineTo(center_x + x_val, center_y + y_val)

        painter.setPen(QPen(QColor(168, 85, 247), 2.2))
        painter.setBrush(QBrush(QColor(168, 85, 247, 25)))
        painter.drawPath(orbit_path)

        # Current instantaneous vector position
        t_phase = -2.0 * math.pi / 2.0 * self._sim_time_fs
        curr_ex = self._e0x * math.cos(t_phase) * scale
        curr_ey = self._e0y * math.cos(t_phase + self._delta) * scale

        curr_x = center_x + curr_ey
        curr_y = center_y - curr_ex

        # Vector line from center to tip
        painter.setPen(QPen(QColor(239, 68, 68), 2.0))
        painter.drawLine(center_x, center_y, curr_x, curr_y)

        # Tip dot
        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(curr_x, curr_y), 4.5, 4.5)

        # Polarization state name banner
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(sx + 12.0, sy + sh - 10.0, f"Dạng: {self.get_polarization_classification()}")

    def _paint_analytical_metrics(self, painter: QPainter) -> None:
        """Renders exact electrodynamics analytical quantities and Stokes parameters."""
        rx = 645.0
        ry = 270.0
        rw = 285.0
        rh = 230.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(rx, ry, rw, rh, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(rx + 12.0, ry + 22.0, "THÔNG SỐ ĐIỆN TỪ & NĂNG LƯỢNG:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(226, 232, 240)))

        e0_total = math.hypot(self._e0x, self._e0y)
        f_thz = self.frequency_hz * 1e-12
        intensity = self.poynting_intensity_w_m2
        rad_p = self.radiation_pressure_micro_pa
        s0, s1, s2, s3 = self.stokes_parameters

        lines = [
            f"• Cường độ điện trường E₀: {e0_total:.1f} V/m (Ex={self._e0x:.0f}, Ey={self._e0y:.0f})",
            f"• Cảm ứng từ B₀ = E₀/c: {self.b0_microtesla:.3f} μT ({self.b0_tesla:.2e} T)",
            f"• Tần số f: {f_thz:.1f} THz | Chu kỳ T: {self.period_fs:.2f} fs",
            f"• Độ lệch pha δ: {math.degrees(self._delta):.1f}° ({self._delta/math.pi:.2f}π rad)",
            f"• Cường độ bức xạ <S>: {intensity:.2f} W/m²",
            f"• Áp suất bức xạ P_rad: {rad_p:.3f} μPa (Hấp thụ hoàn toàn)",
            f"• Tham số Stokes chuẩn hóa: [S₀, S₁, S₂, S₃]",
            f"   = [1.0, {s1/s0:.2f}, {s2/s0:.2f}, {s3/s0:.2f}]",
        ]
        curr_y = ry + 42.0
        for line in lines:
            painter.drawText(rx + 12.0, curr_y, line)
            curr_y += 18.0

        # Description note
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        desc_rect = QRectF(rx + 12.0, curr_y + 4.0, rw - 24.0, 38.0)
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        self._layout_buttons()
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        # 1. Preset Selector
        self._draw_btn(painter, self._buttons["btn_preset"], "btn_preset", f"Preset: {self._preset.badge_label}")

        # 2. Amplitude Steppers
        self._draw_btn(painter, self._buttons["btn_e_down"], "btn_e_down", "-E₀")
        self._draw_btn(painter, self._buttons["btn_e_up"], "btn_e_up", "+E₀")

        # 3. Phase Selectors (RCP / LCP)
        self._draw_btn(painter, self._buttons["btn_phase_rcp"], "btn_phase_rcp", "RCP (-90°)")
        self._draw_btn(painter, self._buttons["btn_phase_lcp"], "btn_phase_lcp", "LCP (+90°)")

        # 4. Toggles
        b_label = "Từ trường: BẬT" if self._show_b_field else "Từ trường: ẨN"
        self._draw_btn(painter, self._buttons["btn_toggle_b"], "btn_toggle_b", b_label)

        s_label = "Poynting: BẬT" if self._show_poynting else "Poynting: ẨN"
        self._draw_btn(painter, self._buttons["btn_toggle_s"], "btn_toggle_s", s_label)

        # 5. Quick Presets
        self._draw_btn(painter, self._buttons["btn_p_rcp"], "btn_p_rcp", "Sóng Tròn")
        self._draw_btn(painter, self._buttons["btn_p_retarder"], "btn_p_retarder", "Bản λ/4")

        # 6. Play / Pause
        play_label = "⏸ Dừng" if self._is_running else "▶ Tiếp"
        self._draw_btn(painter, self._buttons["btn_play"], "btn_play", play_label)

        # 7. Stamp Diagram
        self._draw_btn(painter, self._buttons["btn_stamp"], "btn_stamp", "📌 Chèn Vector")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, text: str) -> None:
        is_hovered = self._hovered_btn == key
        bg_col = QColor(30, 41, 59, 220) if not is_hovered else QColor(51, 65, 85, 240)
        border_col = QColor(56, 189, 248) if is_hovered else QColor(71, 85, 105)

        painter.setBrush(QBrush(bg_col))
        painter.setPen(QPen(border_col, 1.2))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor(248, 250, 252) if is_hovered else QColor(203, 213, 225)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # Mouse fallbacks
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        new_hover = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                new_hover = key
                break
        if new_hover != self._hovered_btn:
            self._hovered_btn = new_hover
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._handle_button_click(key)
                event.accept()
                return
        super().mousePressEvent(event)
