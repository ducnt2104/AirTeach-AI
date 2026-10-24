# src/ui/interactive_kepler_orbits.py
"""
Interactive Astrophysics & Planetary Science: Gravitational Orbits,
Kepler's Laws (Elliptical Orbits, Equal Areas in Equal Times, Harmonic Law),
Orbital Velocities (Apoapsis, Periapsis, Escape Velocity) & Satellite Trajectory Simulator.
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
class OrbitPreset:
    """Represents a celestial orbit scenario."""
    code: str
    title_vi: str
    body_name: str
    central_body: str             # "Sun" or "Earth"
    mu_km3_s2: float              # Standard gravitational parameter (km^3/s^2)
    semi_major_axis_km: float     # a in km
    eccentricity: float           # e [0.0, 0.98]
    scale_km_to_px: float         # Scale factor for diagram rendering
    accent_color: QColor          # Orbiting body color
    central_color: QColor         # Central star/planet color
    description: str

    @property
    def semi_minor_axis_km(self) -> float:
        """b = a * sqrt(1 - e^2) in km."""
        return self.semi_major_axis_km * math.sqrt(max(1e-6, 1.0 - self.eccentricity ** 2))

    @property
    def periapsis_km(self) -> float:
        """r_p = a * (1 - e) in km."""
        return self.semi_major_axis_km * (1.0 - self.eccentricity)

    @property
    def apoapsis_km(self) -> float:
        """r_a = a * (1 + e) in km."""
        return self.semi_major_axis_km * (1.0 + self.eccentricity)

    @property
    def period_seconds(self) -> float:
        """Kepler's Third Law: T = 2 * pi * sqrt(a^3 / mu)."""
        return 2.0 * math.pi * math.sqrt((self.semi_major_axis_km ** 3) / self.mu_km3_s2)

    @property
    def periapsis_velocity_km_s(self) -> float:
        """v_p = sqrt(mu * (1 + e) / (a * (1 - e))) in km/s."""
        return math.sqrt(self.mu_km3_s2 * (1.0 + self.eccentricity) / (self.semi_major_axis_km * max(1e-4, 1.0 - self.eccentricity)))

    @property
    def apoapsis_velocity_km_s(self) -> float:
        """v_a = sqrt(mu * (1 - e) / (a * (1 + e))) in km/s."""
        return math.sqrt(self.mu_km3_s2 * (1.0 - self.eccentricity) / (self.semi_major_axis_km * (1.0 + self.eccentricity)))


# Physical Astronomical Constants:
# Sun mu = G * M_sun = 1.32712440018e11 km^3/s^2, 1 AU = 149597870.7 km
# Earth mu = G * M_earth = 398600.4418 km^3/s^2, R_earth = 6378.137 km

MU_SUN = 1.32712440018e11
MU_EARTH = 398600.4418

ORBIT_PRESETS: dict[str, OrbitPreset] = {
    "EARTH_AROUND_SUN": OrbitPreset(
        code="EARTH_AROUND_SUN",
        title_vi="Trái Đất Quanh Mặt Trời (Quỹ Đạo Gần Tròn)",
        body_name="Trái Đất (Earth)",
        central_body="Mặt Trời (Sun)",
        mu_km3_s2=MU_SUN,
        semi_major_axis_km=149597870.7,   # 1.0 AU
        eccentricity=0.0167,
        scale_km_to_px=115.0 / 149597870.7,
        accent_color=QColor(56, 189, 248),
        central_color=QColor(251, 191, 36),
        description="Quỹ đạo gần tròn giữ cho năng lượng bức xạ Mặt Trời nhận được luôn ổn định, duy trì sinh quyển nước lỏng.",
    ),
    "HALLEY_COMET_EXTREME_ECCENTRICITY": OrbitPreset(
        code="HALLEY_COMET_EXTREME_ECCENTRICITY",
        title_vi="Sao Chổi Halley (Độ Lệch Tâm Cực Lớn e = 0.967)",
        body_name="Sao Chổi Halley (1P/Halley)",
        central_body="Mặt Trời (Sun)",
        mu_km3_s2=MU_SUN,
        semi_major_axis_km=17.834 * 149597870.7,
        eccentricity=0.967,
        scale_km_to_px=130.0 / (35.0 * 149597870.7),
        accent_color=QColor(244, 63, 94),
        central_color=QColor(251, 191, 36),
        description="Chu kỳ 75.3 năm; điểm cận nhật 0.586 AU lao vào trong quỹ đạo Kim Tinh, điểm viễn nhật 35.1 AU vượt qua Hải Vương Tinh.",
    ),
    "ISS_LOW_EARTH_ORBIT_LEO": OrbitPreset(
        code="ISS_LOW_EARTH_ORBIT_LEO",
        title_vi="Trạm Vũ Trụ ISS (Quỹ Đạo Trái Đất Tầm Thấp - LEO)",
        body_name="Trạm ISS (Altitude 420 km)",
        central_body="Trái Đất (Earth)",
        mu_km3_s2=MU_EARTH,
        semi_major_axis_km=6378.137 + 420.0,  # 6798 km
        eccentricity=0.0006,
        scale_km_to_px=110.0 / 6798.0,
        accent_color=QColor(34, 197, 94),
        central_color=QColor(59, 130, 246),
        description="Quỹ đạo LEO độ cao 420 km, vận tốc quỹ đạo 7.66 km/s, bay tròn 1 vòng quanh Trái Đất chỉ trong 92.7 phút.",
    ),
    "GEOSTATIONARY_ORBIT_GEO": OrbitPreset(
        code="GEOSTATIONARY_ORBIT_GEO",
        title_vi="Vệ Tinh Địa Tĩnh GEO (Độ Cao 35,786 km)",
        body_name="Vệ Tinh Địa Tĩnh (GEO Sat)",
        central_body="Trái Đất (Earth)",
        mu_km3_s2=MU_EARTH,
        semi_major_axis_km=42164.0,           # 42,164 km
        eccentricity=0.0000,
        scale_km_to_px=115.0 / 42164.0,
        accent_color=QColor(236, 72, 153),
        central_color=QColor(59, 130, 246),
        description="Chu kỳ đúng bằng 1 ngày sao (23h 56m 04s), vệ tinh đứng yên tương đối so với mặt đất, ứng dụng truyền thông viễn thông.",
    ),
    "MOLNIYA_HIGH_INCLINATION_COMM": OrbitPreset(
        code="MOLNIYA_HIGH_INCLINATION_COMM",
        title_vi="Quỹ Đạo Molniya (Độ Elip Cao e = 0.72 - Viễn Địa Kéo Dài)",
        body_name="Vệ Tinh Molniya",
        central_body="Trái Đất (Earth)",
        mu_km3_s2=MU_EARTH,
        semi_major_axis_km=26554.0,
        eccentricity=0.720,
        scale_km_to_px=125.0 / (26554.0 * 1.72),
        accent_color=QColor(168, 85, 247),
        central_color=QColor(59, 130, 246),
        description="Quỹ đạo 12 giờ của Nga; theo Định luật Kepler II, vệ tinh di chuyển cực chậm gần điểm viễn địa, phủ sóng 8 tiếng liên tục trên bán cầu Bắc.",
    ),
}


class InteractiveKeplerOrbitsItem(QGraphicsObject):
    """
    Pedagogical Interactive Gravitational Orbits, Kepler's Laws (I, II, III),
    Vis-Viva Orbital Velocity and Satellite Trajectory Simulator.
    """

    orbit_computed = Signal(dict)
    stamp_requested = Signal(dict)

    CARD_W = 680.0
    CARD_H = 380.0

    def __init__(self, parent: Optional[QGraphicsItem] = None, pos: tuple[float, float] = (80.0, 40.0)):
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(150.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)

        self._active_preset_code: str = "HALLEY_COMET_EXTREME_ECCENTRICITY"
        self._preset: OrbitPreset = ORBIT_PRESETS[self._active_preset_code]

        # Orbital state variables:
        self._semi_major_axis_km: float = self._preset.semi_major_axis_km
        self._eccentricity: float = self._preset.eccentricity
        self._mean_anomaly_rad: float = 0.0     # M in radians [0, 2*pi]

        # Visualization toggles
        self._show_equal_areas: bool = True     # Kepler's 2nd Law swept triangles
        self._show_velocity_vector: bool = True # Vis-Viva velocity arrow
        self._is_animating: bool = True
        self._sim_speed: float = 0.02           # Radian step per tick

        # Animation timer (30 Hz)
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._layout_buttons()

    # ------------------------------------------------------------------
    # Mathematical & Physical Core Calculations
    # ------------------------------------------------------------------

    @property
    def semi_major_axis_km(self) -> float:
        return self._semi_major_axis_km

    @property
    def eccentricity(self) -> float:
        return self._eccentricity

    @property
    def semi_minor_axis_km(self) -> float:
        return self._semi_major_axis_km * math.sqrt(max(1e-6, 1.0 - self._eccentricity ** 2))

    @property
    def periapsis_km(self) -> float:
        return self._semi_major_axis_km * (1.0 - self._eccentricity)

    @property
    def apoapsis_km(self) -> float:
        return self._semi_major_axis_km * (1.0 + self._eccentricity)

    @property
    def orbital_period_s(self) -> float:
        """Kepler's Third Law: T = 2*pi*sqrt(a^3 / mu)."""
        return 2.0 * math.pi * math.sqrt((self._semi_major_axis_km ** 3) / self._preset.mu_km3_s2)

    @property
    def mean_motion_rad_s(self) -> float:
        """n = 2*pi / T."""
        return (2.0 * math.pi) / max(1e-4, self.orbital_period_s)

    def solve_kepler_eccentric_anomaly(self, mean_anomaly: float) -> float:
        """
        Solves Kepler's equation M = E - e*sin(E) for Eccentric Anomaly E
        using high-precision Newton-Raphson iteration.
        """
        m = mean_anomaly % (2.0 * math.pi)
        e = self._eccentricity
        # Initial guess
        ecc = m if e < 0.8 else math.pi
        for _ in range(25):
            f = ecc - e * math.sin(ecc) - m
            f_prime = 1.0 - e * math.cos(ecc)
            delta = f / f_prime
            ecc -= delta
            if abs(delta) < 1e-10:
                break
        return ecc

    def true_anomaly_rad(self, eccentric_anomaly: float) -> float:
        """Converts Eccentric Anomaly E to True Anomaly nu."""
        e = self._eccentricity
        half_e = eccentric_anomaly / 2.0
        tan_half_nu = math.sqrt(max(1e-6, (1.0 + e) / max(1e-6, 1.0 - e))) * math.tan(half_e)
        nu = 2.0 * math.atan(tan_half_nu)
        return nu % (2.0 * math.pi)

    def radius_at_true_anomaly_km(self, nu_rad: float) -> float:
        """Polar orbit equation r(nu) = a*(1 - e^2) / (1 + e*cos(nu))."""
        p = self._semi_major_axis_km * (1.0 - self._eccentricity ** 2)
        return p / (1.0 + self._eccentricity * math.cos(nu_rad))

    def orbital_velocity_vis_viva_km_s(self, r_km: float) -> float:
        """Vis-Viva equation: v = sqrt(mu * (2/r - 1/a))."""
        mu = self._preset.mu_km3_s2
        term = (2.0 / max(1e-2, r_km)) - (1.0 / self._semi_major_axis_km)
        return math.sqrt(max(0.0, mu * term))

    def escape_velocity_km_s(self, r_km: float) -> float:
        """Parabolic escape velocity v_esc = sqrt(2 * mu / r)."""
        mu = self._preset.mu_km3_s2
        return math.sqrt((2.0 * mu) / max(1e-2, r_km))

    @property
    def current_radius_km(self) -> float:
        ecc = self.solve_kepler_eccentric_anomaly(self._mean_anomaly_rad)
        nu = self.true_anomaly_rad(ecc)
        return self.radius_at_true_anomaly_km(nu)

    @property
    def current_velocity_km_s(self) -> float:
        return self.orbital_velocity_vis_viva_km_s(self.current_radius_km)

    @property
    def current_true_anomaly_deg(self) -> float:
        ecc = self.solve_kepler_eccentric_anomaly(self._mean_anomaly_rad)
        return math.degrees(self.true_anomaly_rad(ecc))

    # ------------------------------------------------------------------
    # State Mutations & Actions
    # ------------------------------------------------------------------

    def set_preset(self, preset_code: str) -> None:
        """Switches celestial orbit preset."""
        if preset_code in ORBIT_PRESETS:
            self._active_preset_code = preset_code
            self._preset = ORBIT_PRESETS[preset_code]
            self._semi_major_axis_km = self._preset.semi_major_axis_km
            self._eccentricity = self._preset.eccentricity
            self._mean_anomaly_rad = 0.0
            self._emit_computed_values()
            self.update()

    def set_orbit_preset(self, preset_code: str) -> None:
        """Alias for set_preset."""
        self.set_preset(preset_code)

    def set_eccentricity(self, e: float) -> None:
        """Sets orbital eccentricity in [0.0, 0.98]."""
        self._eccentricity = max(0.0, min(0.98, round(e, 4)))
        self._emit_computed_values()
        self.update()

    def step_eccentricity(self, delta: float) -> None:
        """Adjusts eccentricity by delta."""
        self.set_eccentricity(self._eccentricity + delta)

    def step_semi_major_axis(self, factor: float) -> None:
        """Scales semi-major axis by factor (e.g. 1.10 or 0.90)."""
        self._semi_major_axis_km = max(100.0, self._semi_major_axis_km * factor)
        self._emit_computed_values()
        self.update()

    def toggle_equal_areas(self) -> None:
        """Toggles Kepler's 2nd Law equal-area sector display."""
        self._show_equal_areas = not self._show_equal_areas
        self.update()

    def toggle_animation(self) -> None:
        """Pauses or resumes simulation animation."""
        self._is_animating = not self._is_animating

    def reset_orbit(self) -> None:
        """Resets to default preset parameters."""
        self.set_preset(self._active_preset_code)

    def cycle_preset(self) -> None:
        """Cycles to next orbit preset."""
        self._handle_button_click("preset_next")

    def _emit_computed_values(self) -> None:
        data = {
            "preset_code": self._active_preset_code,
            "title_vi": self._preset.title_vi,
            "body_name": self._preset.body_name,
            "central_body": self._preset.central_body,
            "semi_major_axis_km": self._semi_major_axis_km,
            "eccentricity": self._eccentricity,
            "current_radius_km": self.current_radius_km,
            "current_velocity_km_s": self.current_velocity_km_s,
            "orbital_period_s": self.orbital_period_s,
            "periapsis_km": self.periapsis_km,
            "apoapsis_km": self.apoapsis_km,
        }
        self.orbit_computed.emit(data)

    def _on_sim_tick(self) -> None:
        """Advances mean anomaly."""
        if self._is_animating:
            self._mean_anomaly_rad = (self._mean_anomaly_rad + self._sim_speed) % (2.0 * math.pi)
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
        self._buttons["anim_toggle"] = QRectF(ctrl_x + 104.0, row1_y, 172.0, btn_h)

        row2_y = 278.0
        self._buttons["ecc_plus"] = QRectF(ctrl_x, row2_y, btn_w_half, btn_h)
        self._buttons["ecc_minus"] = QRectF(ctrl_x + 78.0, row2_y, btn_w_half, btn_h)
        self._buttons["toggle_areas"] = QRectF(ctrl_x + 158.0, row2_y, 118.0, btn_h)

        row3_y = 308.0
        self._buttons["sma_plus"] = QRectF(ctrl_x, row3_y, 85.0, btn_h)
        self._buttons["sma_minus"] = QRectF(ctrl_x + 92.0, row3_y, 85.0, btn_h)
        self._buttons["set_circular"] = QRectF(ctrl_x + 184.0, row3_y, 92.0, btn_h)

        row4_y = 342.0
        self._buttons["reset"] = QRectF(ctrl_x, row4_y, 100.0, 26.0)
        self._buttons["stamp"] = QRectF(ctrl_x + 108.0, row4_y, 168.0, 26.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(10, 15, 30))
        grad.setColorAt(1.0, QColor(5, 8, 20))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(40, 55, 90), 1.5))
        painter.drawRoundedRect(self.boundingRect(), 10.0, 10.0)

        # 2. Header
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(16.0, 24.0, "🪐 QUỸ ĐẠO HẤP DẪN & 3 ĐỊNH LUẬT KEPLER (ORBITAL MECHANICS)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        sub_text = f"{self._preset.title_vi} | Vệ tinh: {self._preset.body_name}"
        painter.drawText(16.0, 42.0, sub_text)

        # 3. Left Panel: Orbit Simulator Canvas
        canvas_rect = QRectF(16.0, 52.0, 350.0, 312.0)
        painter.setBrush(QBrush(QColor(10, 15, 28)))
        painter.setPen(QPen(QColor(35, 50, 80), 1.0))
        painter.drawRoundedRect(canvas_rect, 6.0, 6.0)

        self._paint_orbit_canvas(painter, canvas_rect)

        # 4. Right Panel: Live Astrodynamics Telemetry
        self._paint_telemetry(painter)

        # 5. Buttons
        self._paint_buttons(painter)

    def _paint_orbit_canvas(self, painter: QPainter, rect: QRectF) -> None:
        center_x = rect.x() + rect.width() * 0.50
        center_y = rect.y() + rect.height() * 0.50

        # Scale semi-major and semi-minor axis to pixel canvas
        max_dim = min(rect.width() * 0.42, rect.height() * 0.42)
        a_px = max_dim
        b_px = a_px * math.sqrt(max(1e-4, 1.0 - self._eccentricity ** 2))
        c_px = a_px * self._eccentricity # Distance from center to focus

        # Central body sits at primary focus: (center_x + c_px, center_y) or (center_x, center_y)
        # Let's place Primary Focus F1 at center_x + c_px, center_y
        f1_x = center_x + c_px * 0.4
        f1_y = center_y
        ellipse_cx = f1_x - c_px
        ellipse_cy = f1_y

        # Draw Major & Minor Axes
        painter.setPen(QPen(QColor(30, 45, 75), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(ellipse_cx - a_px, ellipse_cy, ellipse_cx + a_px, ellipse_cy)
        painter.drawLine(ellipse_cx, ellipse_cy - b_px, ellipse_cx, ellipse_cy + b_px)

        # Kepler's Second Law: Equal Swept Areas (A1 near periapsis, A2 near apoapsis)
        if self._show_equal_areas:
            delta_t_rad = 0.35
            # Sector 1: Near Periapsis (nu around 0)
            p1_path = QPainterPath()
            p1_path.moveTo(f1_x, f1_y)
            for step in range(21):
                ang = -delta_t_rad * 0.5 + (step / 20.0) * delta_t_rad
                r_ang = a_px * (1.0 - self._eccentricity ** 2) / (1.0 + self._eccentricity * math.cos(ang))
                px = f1_x + r_ang * math.cos(ang)
                py = f1_y - r_ang * math.sin(ang)
                p1_path.lineTo(px, py)
            p1_path.closeSubpath()
            painter.setBrush(QBrush(QColor(56, 189, 248, 50)))
            painter.setPen(QPen(QColor(56, 189, 248, 120), 1.0))
            painter.drawPath(p1_path)

            # Sector 2: Near Apoapsis (nu around pi)
            p2_path = QPainterPath()
            p2_path.moveTo(f1_x, f1_y)
            for step in range(21):
                ang = math.pi - delta_t_rad * 0.5 + (step / 20.0) * delta_t_rad
                r_ang = a_px * (1.0 - self._eccentricity ** 2) / (1.0 + self._eccentricity * math.cos(ang))
                px = f1_x + r_ang * math.cos(ang)
                py = f1_y - r_ang * math.sin(ang)
                p2_path.lineTo(px, py)
            p2_path.closeSubpath()
            painter.setBrush(QBrush(QColor(244, 63, 94, 50)))
            painter.setPen(QPen(QColor(244, 63, 94, 120), 1.0))
            painter.drawPath(p2_path)

            # Label Areas
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor(56, 189, 248))
            painter.drawText(f1_x + 15.0, f1_y - 12.0, "Diện tích S₁ (Cận Nhật)")
            painter.setPen(QColor(244, 63, 94))
            painter.drawText(ellipse_cx - a_px + 10.0, ellipse_cy - 12.0, "Diện tích S₂ = S₁ (Viễn Nhật)")

        # Draw Elliptical Orbit Path
        orbit_path = QPainterPath()
        num_pts = 100
        for i in range(num_pts + 1):
            theta = (i / float(num_pts)) * 2.0 * math.pi
            r_th = a_px * (1.0 - self._eccentricity ** 2) / (1.0 + self._eccentricity * math.cos(theta))
            ox = f1_x + r_th * math.cos(theta)
            oy = f1_y - r_th * math.sin(theta)
            if i == 0:
                orbit_path.moveTo(ox, oy)
            else:
                orbit_path.lineTo(ox, oy)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._preset.accent_color.lighter(130), 1.6))
        painter.drawPath(orbit_path)

        # Draw Central Body (Sun / Earth) at Focus F1
        c_radius = 12.0
        c_grad = QRadialGradient(f1_x, f1_y, c_radius)
        c_grad.setColorAt(0.0, self._preset.central_color.lighter(150))
        c_grad.setColorAt(0.7, self._preset.central_color)
        c_grad.setColorAt(1.0, self._preset.central_color.darker(160))
        painter.setBrush(QBrush(c_grad))
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.drawEllipse(QPointF(f1_x, f1_y), c_radius, c_radius)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(self._preset.central_color)
        painter.drawText(f1_x - 22.0, f1_y + 24.0, self._preset.central_body.split()[0])

        # Current Orbiting Body Position
        ecc_now = self.solve_kepler_eccentric_anomaly(self._mean_anomaly_rad)
        nu_now = self.true_anomaly_rad(ecc_now)
        r_now_px = a_px * (1.0 - self._eccentricity ** 2) / (1.0 + self._eccentricity * math.cos(nu_now))
        sat_x = f1_x + r_now_px * math.cos(nu_now)
        sat_y = f1_y - r_now_px * math.sin(nu_now)

        # Radius Vector r(t)
        painter.setPen(QPen(QColor(148, 163, 184, 160), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(f1_x, f1_y, sat_x, sat_y)

        # Draw Satellite Body
        sat_radius = 6.5
        sat_grad = QRadialGradient(sat_x, sat_y, sat_radius)
        sat_grad.setColorAt(0.0, QColor(255, 255, 255))
        sat_grad.setColorAt(0.6, self._preset.accent_color)
        sat_grad.setColorAt(1.0, self._preset.accent_color.darker(150))
        painter.setBrush(QBrush(sat_grad))
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.drawEllipse(QPointF(sat_x, sat_y), sat_radius, sat_radius)

        # Draw Velocity Vector Arrow
        if self._show_velocity_vector:
            # Velocity direction is tangent to ellipse: dr/dt and r*dnu/dt
            # Tangent angle phi:
            e = self._eccentricity
            gamma = math.atan2(e * math.sin(nu_now), 1.0 + e * math.cos(nu_now))
            flight_ang = nu_now + math.pi * 0.5 - gamma
            v_len = min(40.0, max(12.0, self.current_velocity_km_s * 1.5))
            vx = sat_x - v_len * math.sin(flight_ang)
            vy = sat_y - v_len * math.cos(flight_ang)

            painter.setPen(QPen(QColor(251, 191, 36), 2.0))
            painter.drawLine(sat_x, sat_y, vx, vy)
            # Arrow head
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor(251, 191, 36))
            painter.drawText(vx + 3.0, vy - 2.0, f"v = {self.current_velocity_km_s:.1f} km/s")

        # Periapsis & Apoapsis Marker Pins
        peri_x = f1_x + (a_px * (1.0 - self._eccentricity))
        apo_x = f1_x - (a_px * (1.0 + self._eccentricity))

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(peri_x - 10.0, f1_y + 14.0, "Cận nhật (P)")
        painter.setPen(QColor(244, 63, 94))
        painter.drawText(apo_x - 12.0, f1_y + 14.0, "Viễn nhật (A)")

    def _paint_telemetry(self, painter: QPainter) -> None:
        p_x = 385.0
        p_y = 56.0
        p_w = 278.0
        p_h = 182.0

        painter.setBrush(QBrush(QColor(15, 23, 42)))
        painter.setPen(QPen(QColor(40, 55, 90), 1.0))
        painter.drawRoundedRect(QRectF(p_x, p_y, p_w, p_h), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(p_x + 10.0, p_y + 18.0, "📊 THÔNG SỐ VŨ TRỤ HỌC THỜI GIAN THỰC")

        # Period formatting (seconds to minutes, hours, days or years)
        t_sec = self.orbital_period_s
        if t_sec > 365.25 * 86400.0:
            period_str = f"{t_sec / (365.25 * 86400.0):.2f} năm"
        elif t_sec > 86400.0:
            period_str = f"{t_sec / 86400.0:.2f} ngày"
        elif t_sec > 3600.0:
            period_str = f"{t_sec / 3600.0:.2f} giờ"
        else:
            period_str = f"{t_sec / 60.0:.1f} phút"

        lines = [
            ("Bán trục lớn (a):", f"{self._semi_major_axis_km:,.0f} km"),
            ("Độ lệch tâm (e):", f"{self._eccentricity:.4f}"),
            ("Bán trục nhỏ (b):", f"{self.semi_minor_axis_km:,.0f} km"),
            ("Khoảng cách cận điểm (r_p):", f"{self.periapsis_km:,.0f} km"),
            ("Khoảng cách viễn điểm (r_a):", f"{self.apoapsis_km:,.0f} km"),
            ("Chu kỳ quỹ đạo (T):", period_str),
            ("Bán kính tức thời r(t):", f"{self.current_radius_km:,.0f} km"),
            ("Vận tốc tức thời v(t):", f"{self.current_velocity_km_s:.2f} km/s"),
            ("Vận tốc vũ trụ cấp 2 (v_esc):", f"{self.escape_velocity_km_s(self.current_radius_km):.2f} km/s"),
        ]

        painter.setFont(QFont("Segoe UI", 7))
        y_cursor = p_y + 36.0
        for lbl, val in lines:
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(p_x + 10.0, y_cursor, lbl)
            painter.setPen(QColor(248, 250, 252))
            painter.drawText(p_x + 175.0, y_cursor, val)
            y_cursor += 15.5

    def _paint_buttons(self, painter: QPainter) -> None:
        btn_labels = {
            "preset_prev": "◀ Trước",
            "preset_next": "Sau ▶",
            "anim_toggle": "⏸ Tạm Dừng" if self._is_animating else "▶ Tiếp Tục",
            "ecc_plus": "e +0.05",
            "ecc_minus": "e -0.05",
            "toggle_areas": "S₁=S₂ (Kepler 2)",
            "sma_plus": "Bán Kính +10%",
            "sma_minus": "Bán Kính -10%",
            "set_circular": "Tròn (e = 0)",
            "reset": "🔄 Đặt Lại",
            "stamp": "📌 Đóng Dấu Bài Học",
        }

        for key, rect in self._buttons.items():
            is_hovered = (self._hovered_btn == key)
            if key == "stamp":
                bg_col = QColor(14, 165, 233) if not is_hovered else QColor(56, 189, 248)
                txt_col = QColor(255, 255, 255)
            elif key == "anim_toggle":
                bg_col = QColor(16, 185, 129) if not is_hovered else QColor(52, 211, 153)
                txt_col = QColor(255, 255, 255)
            else:
                bg_col = QColor(30, 41, 59) if not is_hovered else QColor(51, 65, 85)
                txt_col = QColor(226, 232, 240)

            painter.setBrush(QBrush(bg_col))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.setPen(txt_col)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, btn_labels.get(key, key))

    # ------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_h = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_h != self._hovered_btn:
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
        preset_keys = list(ORBIT_PRESETS.keys())
        curr_idx = preset_keys.index(self._active_preset_code)

        if btn_key == "preset_prev":
            new_idx = (curr_idx - 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "preset_next":
            new_idx = (curr_idx + 1) % len(preset_keys)
            self.set_preset(preset_keys[new_idx])
        elif btn_key == "anim_toggle":
            self.toggle_animation()
        elif btn_key == "ecc_plus":
            self.step_eccentricity(0.05)
        elif btn_key == "ecc_minus":
            self.step_eccentricity(-0.05)
        elif btn_key == "toggle_areas":
            self.toggle_equal_areas()
        elif btn_key == "sma_plus":
            self.step_semi_major_axis(1.10)
        elif btn_key == "sma_minus":
            self.step_semi_major_axis(0.90)
        elif btn_key == "set_circular":
            self.set_eccentricity(0.0)
        elif btn_key == "reset":
            self.reset_orbit()
        elif btn_key == "stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits stamp signal for LessonBoard permanent vector diagram rendering."""
        data = {
            "preset_code": self._active_preset_code,
            "title_vi": self._preset.title_vi,
            "body_name": self._preset.body_name,
            "central_body": self._preset.central_body,
            "semi_major_axis_km": self._semi_major_axis_km,
            "eccentricity": self._eccentricity,
            "semi_minor_axis_km": self.semi_minor_axis_km,
            "periapsis_km": self.periapsis_km,
            "apoapsis_km": self.apoapsis_km,
            "orbital_period_s": self.orbital_period_s,
            "periapsis_velocity_km_s": self._preset.periapsis_velocity_km_s,
            "apoapsis_velocity_km_s": self._preset.apoapsis_velocity_km_s,
        }
        self.stamp_requested.emit(data)
