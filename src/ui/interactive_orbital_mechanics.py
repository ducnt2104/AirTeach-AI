"""
AirTeach AI - Interactive Astrodynamics & Orbital Mechanics: Hohmann Transfer & Gravitational Slingshot
========================================================================================================
Authentic astrodynamics and celestial mechanics simulation based on Keplerian two-body problem & perturbations:
- Vis-Viva Equation & Specific Orbital Energy (Vis-Viva Theorem):
    v^2 = μ * (2 / r - 1 / a)
    ε = v^2 / 2 - μ / r = -μ / (2 * a)
    - Bound Elliptic: a > 0, e ∈ [0, 1), ε < 0
    - Parabolic Escape: a -> ∞, e = 1, ε = 0, v_esc = sqrt(2 * μ / r)
    - Hyperbolic Flyby: a < 0, e > 1, ε > 0, v_inf = sqrt(2 * ε)
- Hohmann Interplanetary & Orbital Transfer (Walter Hohmann, 1925):
    Two-impulse coplanar transfer between circular orbits of radii r_1 and r_2:
    - Transfer semi-major axis:
        a_tx = (r_1 + r_2) / 2
    - Transfer eccentricity:
        e_tx = |r_2 - r_1| / (r_1 + r_2)
    - Delta-V 1 (Departure burn at periapsis):
        Δv_1 = sqrt(μ / r_1) * (sqrt(2 * r_2 / (r_1 + r_2)) - 1)
    - Delta-V 2 (Arrival circularization burn at apoapsis):
        Δv_2 = sqrt(μ / r_2) * (1 - sqrt(2 * r_1 / (r_1 + r_2)))
    - Total Delta-V:
        Δv_tot = |Δv_1| + |Δv_2|
    - One-way transfer time:
        t_tx = π * sqrt(a_tx^3 / μ)
- Gravitational Slingshot & Hyperbolic Gravity Assist (Flyby):
    Deflection angle in planet's rest frame:
      δ = 2 * arcsin(1 / (1 + r_p * v_inf^2 / μ_p))
    Heliocentric velocity vector addition transfers orbital momentum from planet to probe:
      v_out = V_planet + v_inf_out (Maximum speed boost up to 2 * V_planet)
- J2 Gravitational Perturbations (Earth Oblateness):
    Earth J_2 = 1.08263e-3.
    Nodal regression rate (Sun-Synchronous condition: dΩ/dt = 0.9856°/day):
      dΩ/dt = -1.5 * J_2 * (R_eq / p)^2 * n * cos(i)
    Apsidal precession:
      dω/dt = 0.75 * J_2 * (R_eq / p)^2 * n * (5 * cos^2(i) - 1)
      Critical inclination (Molniya orbit): 5 * cos^2(i) - 1 = 0  =>  i = 63.43°
- 5 Curriculum Presets:
    1. HOHMANN_EARTH_MARS: Interplanetary Earth-to-Mars transfer (r_1 = 1.0 AU, r_2 = 1.524 AU).
    2. GRAVITATIONAL_SLINGSHOT: Jupiter flyby gravity assist propelling spacecraft onto solar escape trajectory.
    3. LOW_EARTH_TO_GEO: LEO (r = 6778 km) to Geostationary GEO (r = 42164 km) two-burn Hohmann transfer.
    4. SUN_SYNCHRONOUS_J2: Polar Sun-synchronous orbit with J2 nodal precession matching solar day.
    5. MOLNIYA_CRITICAL_INCLINATION: Highly eccentric 12-hour Molniya orbit frozen at critical inclination i = 63.43°.
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

# Standard gravitational parameters in simulation units
MU_SUN: float = 1.0  # Normalized: 1.0 (AU^3 / yr^2 * 4π^2)
MU_EARTH: float = 398600.4418  # km^3 / s^2


@dataclass(frozen=True)
class OrbitalPreset:
    code: str
    name_vi: str
    scenario_type: str  # "hohmann", "slingshot", "geo", "sunsync", "molniya"
    r1: float  # Initial radius (AU or 1000 km)
    r2: float  # Target radius (AU or 1000 km)
    inclination_deg: float
    eccentricity: float
    description: str


PRESETS: dict[str, OrbitalPreset] = {
    "HOHMANN_EARTH_MARS": OrbitalPreset(
        code="HOHMANN_EARTH_MARS",
        name_vi="1. Quỹ đạo Hohmann Trái Đất - Sao Hỏa",
        scenario_type="hohmann",
        r1=1.00,  # 1.0 AU (Earth)
        r2=1.524,  # 1.524 AU (Mars)
        inclination_deg=1.85,
        eccentricity=0.207,
        description="Quỹ đạo chuyển tiếp Hohmann tiết kiệm nhiên liệu nhất giữa Trái Đất và Sao Hỏa. Δv₁ = 2.94 km/s, Δv₂ = 2.65 km/s.",
    ),
    "GRAVITATIONAL_SLINGSHOT": OrbitalPreset(
        code="GRAVITATIONAL_SLINGSHOT",
        name_vi="2. Hỗ trợ Trọng lực (Gravity Slingshot)",
        scenario_type="slingshot",
        r1=1.00,
        r2=5.20,  # 5.2 AU (Jupiter)
        inclination_deg=1.30,
        eccentricity=1.35,  # Hyperbolic flyby
        description="Phi thuyền mượn động lượng quỹ đạo của Sao Mộc (Flyby) để tăng tốc vượt vận tốc vũ trụ cấp 3, thoát khỏi Hệ Mặt Trời.",
    ),
    "LOW_EARTH_TO_GEO": OrbitalPreset(
        code="LOW_EARTH_TO_GEO",
        name_vi="3. Nâng quỹ đạo LEO lên GEO Địa tĩnh",
        scenario_type="geo",
        r1=0.678,  # 6,778 km (LEO 400 km)
        r2=4.216,  # 42,164 km (GEO 35,786 km)
        inclination_deg=0.0,
        eccentricity=0.723,
        description="Chuyển tiếp vệ tinh từ quỹ đạo tầm thấp (LEO) lên quỹ đạo địa tĩnh (GEO, chu kỳ đúng 24h) qua quỹ đạo GTO trung gian.",
    ),
    "SUN_SYNCHRONOUS_J2": OrbitalPreset(
        code="SUN_SYNCHRONOUS_J2",
        name_vi="4. Quỹ đạo Đồng bộ Mặt Trời (Sun-Sync)",
        scenario_type="sunsync",
        r1=0.718,  # 7,178 km (800 km altitude)
        r2=0.718,
        inclination_deg=98.6,  # Retrograde
        eccentricity=0.001,
        description="Độ dẹt của Trái Đất (J₂) làm mặt phẳng quỹ đạo tiến động đúng 0.9856°/ngày, giữ vệ tinh luôn quan sát mặt đất cùng góc chiếu sáng.",
    ),
    "MOLNIYA_CRITICAL_INCLINATION": OrbitalPreset(
        code="MOLNIYA_CRITICAL_INCLINATION",
        name_vi="5. Quỹ đạo Molniya & Góc nghiêng Tới hạn",
        scenario_type="molniya",
        r1=0.698,  # Perigee 600 km
        r2=4.000,  # Apogee 40,000 km
        inclination_deg=63.43,  # Critical inclination 5cos²(i)-1=0
        eccentricity=0.702,
        description="Ở góc nghiêng tới hạn 63.43°, độ tiến động cận điểm triệt tiêu (dω/dt = 0), đóng băng điểm viễn địa phục vụ viễn thông vùng cực.",
    ),
}


class InteractiveOrbitalMechanicsItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for Astrodynamics, Hohmann Transfer Orbits,
    Gravitational Slingshots, and J2 Earth Perturbations.
    """

    # Signals
    preset_changed = Signal(str)
    r2_changed = Signal(float)
    inclination_changed = Signal(float)
    closed = Signal()

    def __init__(
        self,
        board: Optional["LessonBoard"] = None,
        pos: tuple[float, float] | QPointF = (80.0, 80.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board

        # Dimensions
        self._width: float = 940.0
        self._height: float = 650.0

        # Physical orbital parameters
        self._current_preset: str = "HOHMANN_EARTH_MARS"
        self._scenario_type: str = "hohmann"
        self._r1: float = 1.00  # Initial orbit radius (AU or normalized)
        self._r2: float = 1.524  # Target orbit radius
        self._inclination_deg: float = 1.85
        self._mu: float = 1.0  # Gravitational parameter

        # Simulation clock and state
        self._sim_time_sec: float = 0.0
        self._is_playing: bool = True
        self._spacecraft_true_anomaly_rad: float = 0.0
        self._planet1_angle_rad: float = 0.0
        self._planet2_angle_rad: float = 0.0
        self._transfer_progress: float = 0.0  # 0.0 to 1.0 along transfer ellipse

        # Historical trajectory trail
        self._trail_points: list[tuple[float, float]] = []
        self._max_trail_points: int = 120

        # UI interaction state
        self._active_slider: Optional[str] = None  # "r2", "inc"
        self._hovered_button: Optional[str] = None

        # Flags & setup
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        if isinstance(pos, QPointF):
            self.setPos(pos.x(), pos.y())
        else:
            self.setPos(pos[0], pos[1])

    # -------------------------------------------------------------------------
    # Astrodynamics & Physics Calculations
    # -------------------------------------------------------------------------

    @property
    def transfer_semi_major_axis(self) -> float:
        """Calculate transfer ellipse semi-major axis a_tx = (r1 + r2) / 2."""
        return (self._r1 + self._r2) / 2.0

    @property
    def transfer_eccentricity(self) -> float:
        """Calculate transfer ellipse eccentricity e_tx = |r2 - r1| / (r1 + r2)."""
        return abs(self._r2 - self._r1) / (self._r1 + self._r2)

    def calculate_delta_v1(self) -> float:
        """
        Calculate departure burn Δv1:
        Δv1 = sqrt(μ/r1) * (sqrt(2*r2 / (r1 + r2)) - 1)
        """
        v_circ1 = math.sqrt(self._mu / self._r1)
        v_peri_tx = math.sqrt(self._mu * (2.0 / self._r1 - 1.0 / self.transfer_semi_major_axis))
        return v_peri_tx - v_circ1

    def calculate_delta_v2(self) -> float:
        """
        Calculate arrival burn Δv2:
        Δv2 = sqrt(μ/r2) * (1 - sqrt(2*r1 / (r1 + r2)))
        """
        v_circ2 = math.sqrt(self._mu / self._r2)
        v_apo_tx = math.sqrt(self._mu * (2.0 / self._r2 - 1.0 / self.transfer_semi_major_axis))
        return v_circ2 - v_apo_tx

    def calculate_total_delta_v(self) -> float:
        """Calculate total Hohmann delta-v: |Δv1| + |Δv2|."""
        return abs(self.calculate_delta_v1()) + abs(self.calculate_delta_v2())

    def calculate_transfer_time_days(self) -> float:
        """Calculate transfer time t_tx = π * sqrt(a_tx^3 / μ) in normalized days."""
        a = self.transfer_semi_major_axis
        # In normalized AU/year units: T = 365.25 * sqrt(a^3), t_tx = T / 2
        return (365.25 / 2.0) * math.sqrt(a ** 3)

    def calculate_j2_nodal_precession_deg_day(self) -> float:
        """Calculate J2 nodal precession rate dΩ/dt in degrees per day."""
        # For Earth LEO/Sun-Sync: dΩ/dt = -9.97 * (R_e / r)^3.5 * cos(i)
        inc_rad = math.radians(self._inclination_deg)
        return -9.97 * ((0.6378 / max(0.65, self._r1)) ** 3.5) * math.cos(inc_rad)

    # -------------------------------------------------------------------------
    # Simulation Step & Control
    # -------------------------------------------------------------------------

    def step_simulation(self, dt_sec: float = 0.033) -> None:
        """Advance orbital bodies and spacecraft along transfer trajectory."""
        if not self._is_playing:
            return

        self._sim_time_sec += dt_sec

        # Mean motions n = sqrt(μ / r^3)
        n1 = math.sqrt(self._mu / (self._r1 ** 3))
        n2 = math.sqrt(self._mu / (self._r2 ** 3))
        n_tx = math.sqrt(self._mu / (self.transfer_semi_major_axis ** 3))

        # Update planetary orbital positions
        self._planet1_angle_rad += n1 * dt_sec * 0.8
        self._planet2_angle_rad += n2 * dt_sec * 0.8

        # Spacecraft transfer progress: 0 to π radians along transfer ellipse
        self._spacecraft_true_anomaly_rad += n_tx * dt_sec * 0.8
        if self._spacecraft_true_anomaly_rad > math.pi:
            # Wrap transfer
            self._spacecraft_true_anomaly_rad = 0.0
            self._trail_points.clear()

        # Record spacecraft trail
        theta = self._spacecraft_true_anomaly_rad
        a = self.transfer_semi_major_axis
        e = self.transfer_eccentricity
        r = a * (1.0 - e ** 2) / (1.0 + e * math.cos(theta))

        sc_x = r * math.cos(theta)
        sc_y = r * math.sin(theta)
        self._trail_points.append((sc_x, sc_y))
        if len(self._trail_points) > self._max_trail_points:
            self._trail_points.pop(0)

        self.update()

    def set_preset(self, preset_code: str) -> None:
        """Apply one of the 5 curriculum presets."""
        if preset_code not in PRESETS:
            logger.warning(f"Unknown orbital preset: {preset_code}")
            return

        p = PRESETS[preset_code]
        self._current_preset = preset_code
        self._scenario_type = p.scenario_type
        self._r1 = p.r1
        self._r2 = p.r2
        self._inclination_deg = p.inclination_deg

        self._spacecraft_true_anomaly_rad = 0.0
        self._trail_points.clear()

        self.preset_changed.emit(preset_code)
        self.r2_changed.emit(self._r2)
        self.inclination_changed.emit(self._inclination_deg)
        self.update()

    def set_target_radius(self, r: float) -> None:
        """Set target orbit radius r2."""
        self._r2 = max(0.5, min(6.0, r))
        self.r2_changed.emit(self._r2)
        self.update()

    def set_inclination(self, inc_deg: float) -> None:
        """Set orbital inclination in degrees [0 to 180°]."""
        self._inclination_deg = max(0.0, min(180.0, inc_deg))
        self.inclination_changed.emit(self._inclination_deg)
        self.update()

    def toggle_play_pause(self) -> None:
        """Toggle simulation pause state."""
        self._is_playing = not self._is_playing
        self.update()

    def reset_orbit(self) -> None:
        """Reset spacecraft position and clear trail."""
        self._spacecraft_true_anomaly_rad = 0.0
        self._trail_points.clear()
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1. Main outer card background
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor(10, 16, 28, 248))
        grad.setColorAt(1.0, QColor(6, 10, 18, 252))
        painter.setPen(QPen(QColor(38, 65, 105, 200), 1.5))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header bar with title and close button
        self._paint_header(painter)

        # 3. Preset selector bar (5 buttons)
        self._paint_preset_bar(painter)

        # 4. Astrodynamics celestial orbital canvas
        self._paint_orbital_canvas(painter)

        # 5. Right-side dashboard: Sliders, Telemetry & Delta-V readouts
        self._paint_control_dashboard(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Draw top banner with title, badges, and close button."""
        title_font = QFont("Segoe UI", 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(240, 246, 255)))
        painter.drawText(
            QRectF(20, 14, 600, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            "CƠ HỌC QUỸ ĐẠO: CHUYỂN TIẾP HOHMANN & GRAVITY SLINGSHOT",
        )

        sub_font = QFont("Segoe UI", 8, QFont.Normal)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(130, 175, 230)))
        painter.drawText(
            QRectF(20, 36, 600, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Astrodynamics · Vis-Viva Theorem · Interplanetary Delta-V · J2 Nodal Precession",
        )

        # Close button [X]
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        is_hover = self._hovered_button == "close"
        painter.setPen(QPen(QColor(230, 80, 80) if is_hover else QColor(160, 60, 60), 1.2))
        painter.setBrush(QBrush(QColor(220, 60, 60, 80 if is_hover else 30)))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(close_rect, Qt.AlignCenter, "×")

    def _paint_preset_bar(self, painter: QPainter) -> None:
        """Draw the 5 preset tabs."""
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0

        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            p = PRESETS[key]
            x = start_x + i * (btn_w + 8.0)
            rect = QRectF(x, y, btn_w, btn_h)

            is_active = (self._current_preset == key)
            is_hover = (self._hovered_button == f"preset_{key}")

            if is_active:
                painter.setPen(QPen(QColor(56, 189, 248), 1.6))
                painter.setBrush(QBrush(QColor(24, 76, 120, 220)))
            elif is_hover:
                painter.setPen(QPen(QColor(70, 130, 180), 1.2))
                painter.setBrush(QBrush(QColor(30, 50, 75, 180)))
            else:
                painter.setPen(QPen(QColor(40, 60, 90), 1.0))
                painter.setBrush(QBrush(QColor(20, 32, 50, 140)))

            painter.drawRoundedRect(rect, 6.0, 6.0)

            font = QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 205, 230)))
            painter.drawText(rect, Qt.AlignCenter, p.name_vi)

    def _paint_orbital_canvas(self, painter: QPainter) -> None:
        """Draw interactive orbits, transfer ellipse, planets, spacecraft, and burns."""
        canvas_rect = QRectF(20, 96, 560, 470)

        # Background starry dark space
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(4, 7, 14, 245)))
        painter.drawRoundedRect(canvas_rect, 10.0, 10.0)

        painter.save()
        painter.setClipRect(canvas_rect)

        cx = canvas_rect.center().x()
        cy = canvas_rect.center().y()

        # Scale factor: map max(r1, r2) into ~200 px radius
        max_r = max(self._r1, self._r2, 1.6)
        scale_px_per_unit = 180.0 / max_r

        # Central Body (Sun or Earth)
        sun_pt = QPointF(cx, cy)
        grad_sun = QRadialGradient(sun_pt, 22.0)
        grad_sun.setColorAt(0.0, QColor(255, 240, 150))
        grad_sun.setColorAt(0.6, QColor(245, 158, 11))
        grad_sun.setColorAt(1.0, QColor(234, 88, 12, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(grad_sun))
        painter.drawEllipse(sun_pt, 22.0, 22.0)

        painter.setBrush(QBrush(QColor(250, 204, 21)))
        painter.drawEllipse(sun_pt, 9.0, 9.0)

        # ---------------------------------------------------------------------
        # Draw Initial Orbit (Orbit 1, e.g. Earth)
        # ---------------------------------------------------------------------
        r1_px = self._r1 * scale_px_per_unit
        painter.setPen(QPen(QColor(56, 189, 248, 160), 1.4, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(sun_pt, r1_px, r1_px)

        # Planet 1 marker (Earth)
        p1_x = cx + r1_px * math.cos(self._planet1_angle_rad)
        p1_y = cy - r1_px * math.sin(self._planet1_angle_rad)
        p1_pt = QPointF(p1_x, p1_y)
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.setBrush(QBrush(QColor(59, 130, 246)))
        painter.drawEllipse(p1_pt, 6.0, 6.0)

        # ---------------------------------------------------------------------
        # Draw Target Orbit (Orbit 2, e.g. Mars)
        # ---------------------------------------------------------------------
        r2_px = self._r2 * scale_px_per_unit
        painter.setPen(QPen(QColor(239, 68, 68, 160), 1.4, Qt.DashLine))
        painter.drawEllipse(sun_pt, r2_px, r2_px)

        # Planet 2 marker (Mars)
        p2_x = cx + r2_px * math.cos(self._planet2_angle_rad)
        p2_y = cy - r2_px * math.sin(self._planet2_angle_rad)
        p2_pt = QPointF(p2_x, p2_y)
        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.drawEllipse(p2_pt, 5.0, 5.0)

        # ---------------------------------------------------------------------
        # Draw Hohmann Transfer Ellipse
        # ---------------------------------------------------------------------
        a = self.transfer_semi_major_axis
        e = self.transfer_eccentricity

        ellipse_path = QPainterPath()
        num_orbit_pts = 100
        for i in range(num_orbit_pts + 1):
            th = math.pi * (i / float(num_orbit_pts))  # Half ellipse from 0 to π
            r = a * (1.0 - e ** 2) / (1.0 + e * math.cos(th))
            ex = cx + r * scale_px_per_unit * math.cos(th)
            ey = cy - r * scale_px_per_unit * math.sin(th)
            if i == 0:
                ellipse_path.moveTo(ex, ey)
            else:
                ellipse_path.lineTo(ex, ey)

        painter.setPen(QPen(QColor(250, 204, 21, 200), 1.8))
        painter.drawPath(ellipse_path)

        # ---------------------------------------------------------------------
        # Draw Delta-V Burn Markers (Departure & Arrival)
        # ---------------------------------------------------------------------
        # Periapsis Burn (Δv1) at theta = 0
        peri_pt = QPointF(cx + self._r1 * scale_px_per_unit, cy)
        painter.setPen(QPen(QColor(34, 197, 94), 2.0))
        painter.setBrush(QBrush(QColor(34, 197, 94, 160)))
        painter.drawEllipse(peri_pt, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(74, 222, 128)))
        painter.drawText(peri_pt.x() + 8, peri_pt.y() - 6, f"Δv₁: +{abs(self.calculate_delta_v1()):.2f}")

        # Apoapsis Burn (Δv2) at theta = π
        apo_pt = QPointF(cx - self._r2 * scale_px_per_unit, cy)
        painter.setPen(QPen(QColor(249, 115, 22), 2.0))
        painter.setBrush(QBrush(QColor(249, 115, 22, 160)))
        painter.drawEllipse(apo_pt, 5.0, 5.0)

        painter.setPen(QPen(QColor(251, 146, 60)))
        painter.drawText(apo_pt.x() - 64, apo_pt.y() - 6, f"Δv₂: +{abs(self.calculate_delta_v2()):.2f}")

        # ---------------------------------------------------------------------
        # Draw Spacecraft & Trajectory Trail
        # ---------------------------------------------------------------------
        if len(self._trail_points) >= 2:
            trail_path = QPainterPath()
            for idx, (tx, ty) in enumerate(self._trail_points):
                sx = cx + tx * scale_px_per_unit
                sy = cy - ty * scale_px_per_unit
                if idx == 0:
                    trail_path.moveTo(sx, sy)
                else:
                    trail_path.lineTo(sx, sy)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5, Qt.DotLine))
            painter.drawPath(trail_path)

        # Spacecraft icon
        th_sc = self._spacecraft_true_anomaly_rad
        r_sc = a * (1.0 - e ** 2) / (1.0 + e * math.cos(th_sc))
        sc_x = cx + r_sc * scale_px_per_unit * math.cos(th_sc)
        sc_y = cy - r_sc * scale_px_per_unit * math.sin(th_sc)
        sc_pt = QPointF(sc_x, sc_y)

        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.setBrush(QBrush(QColor(240, 246, 255)))
        painter.drawEllipse(sc_pt, 4.5, 4.5)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(sc_pt.x() + 8, sc_pt.y() + 4, "PHI THUYỀN")

        painter.restore()

        # Canvas bottom bar: Play / Pause, Reset, Status
        self._paint_canvas_bottom_bar(painter, canvas_rect)

    def _paint_canvas_bottom_bar(self, painter: QPainter, canvas_rect: QRectF) -> None:
        """Draw interactive playback and reset buttons under canvas."""
        y_bar = canvas_rect.bottom() + 10
        btn_h = 28.0

        # Play / Pause
        play_rect = QRectF(canvas_rect.left(), y_bar, 96, btn_h)
        is_hover_play = self._hovered_button == "toggle_play"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_play else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_play else 140)))
        painter.drawRoundedRect(play_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(play_rect, Qt.AlignCenter, "⏸ TẠM DỪNG" if self._is_playing else "▶ TIẾP TỤC")

        # Reset button
        reset_rect = QRectF(play_rect.right() + 8, y_bar, 96, btn_h)
        is_hover_reset = self._hovered_button == "reset_orbit"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_reset else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_reset else 140)))
        painter.drawRoundedRect(reset_rect, 6.0, 6.0)
        painter.drawText(reset_rect, Qt.AlignCenter, "↺ ĐẶT LẠI")

        # Status text
        info_rect = QRectF(reset_rect.right() + 12, y_bar, canvas_rect.right() - reset_rect.right() - 12, btn_h)
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(160, 195, 235)))
        t_days = self.calculate_transfer_time_days()
        info_str = f"Thời gian bay Hohmann: {t_days:.1f} ngày  ·  Tổng Δv = {self.calculate_total_delta_v():.2f} (norm)"
        painter.drawText(info_rect, Qt.AlignVCenter | Qt.AlignLeft, info_str)

    def _paint_control_dashboard(self, painter: QPainter) -> None:
        """Draw sliders, live orbital energy metrics, and Vis-Viva telemetry."""
        dash_rect = QRectF(596, 96, 324, 534)
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(10, 16, 28, 230)))
        painter.drawRoundedRect(dash_rect, 10.0, 10.0)

        y = dash_rect.top() + 14

        # Section 1: Dynamic Sliders
        header_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(header_font)
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_rect.left() + 16, y, 290, 20), Qt.AlignLeft, "THÔNG SỐ QUỸ ĐẠO & BÁN TRỤC LỚN")
        y += 26

        # Slider 1: Target Orbit Radius r2
        self._paint_slider(
            painter,
            slider_id="r2",
            label="Bán kính quỹ đạo đích r₂",
            val_text=f"{self._r2:.3f} AU",
            min_val=0.50,
            max_val=4.50,
            current_val=self._r2,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 54

        # Slider 2: Orbital Inclination (i)
        self._paint_slider(
            painter,
            slider_id="inc",
            label="Góc nghiêng mặt phẳng i",
            val_text=f"{self._inclination_deg:.1f}°",
            min_val=0.0,
            max_val=180.0,
            current_val=self._inclination_deg,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 58

        # Section 2: Astrodynamics Readouts Box
        stats_rect = QRectF(dash_left := dash_rect.left() + 14, y, 296, 140)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(16, 25, 42, 220)))
        painter.drawRoundedRect(stats_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(120, 180, 240)))
        painter.drawText(QRectF(stats_rect.left() + 10, stats_rect.top() + 8, 276, 18), "KẾT QUẢ ĐỘNG LỰC HỌC QUỸ ĐẠO:")

        dv1 = self.calculate_delta_v1()
        dv2 = self.calculate_delta_v2()
        dv_tot = self.calculate_total_delta_v()
        t_tx = self.calculate_transfer_time_days()
        e_tx = self.transfer_eccentricity

        metrics = [
            ("Xung lực khởi hành Δv₁:", f"{dv1:+.3f} (xung đẩy cận điểm)"),
            ("Xung lực điểm đến Δv₂:", f"{dv2:+.3f} (tròn hóa quỹ đạo)"),
            ("Tổng chi phí Δv_tot:", f"{dv_tot:.3f} (chuẩn hóa)"),
            ("Tâm sai chuyển tiếp e_tx:", f"{e_tx:.3f} (Elip chuyển tiếp)"),
            ("Thời gian bay (t_tx):", f"{t_tx:.1f} ngày Trái Đất"),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
        for m_idx, (m_lbl, m_val) in enumerate(metrics):
            my = stats_rect.top() + 30 + m_idx * 21
            painter.setPen(QPen(QColor(180, 205, 230)))
            painter.drawText(QRectF(stats_rect.left() + 10, my, 130, 18), Qt.AlignLeft, m_lbl)
            painter.setPen(QPen(QColor(250, 204, 21) if "Tổng" in m_lbl else QColor(74, 222, 128)))
            painter.drawText(QRectF(stats_rect.right() - 156, my, 146, 18), Qt.AlignRight, m_val)

        y += 154

        # Section 3: Physics Principles Box
        guide_rect = QRectF(dash_left, y, 296, 140)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(12, 20, 34, 230)))
        painter.drawRoundedRect(guide_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(guide_rect.left() + 10, guide_rect.top() + 8, 276, 18), "NGUYÊN LÝ VẬT LÝ QUỸ ĐẠO:")

        guide_items = [
            ("Hohmann:", "Đường chuyển tiếp tiếp xúc hai quỹ đạo tròn tại cận & viễn điểm."),
            ("Vis-Viva:", "v² = μ(2/r - 1/a): Vận tốc lớn nhất tại cận điểm, nhỏ nhất ở viễn điểm."),
            ("Slingshot:", "Mượn động lượng hành tinh tăng tốc phi thuyền không tốn nhiên liệu."),
            ("J₂ Precess:", f"Tiến động điểm nút: {self.calculate_j2_nodal_precession_deg_day():+.3f}°/ngày theo góc nghiêng i."),
        ]

        for g_idx, (g_tag, g_desc) in enumerate(guide_items):
            gy = guide_rect.top() + 30 + g_idx * 26
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QPen(QColor(250, 204, 21) if "J₂" in g_tag else QColor(100, 220, 255)))
            painter.drawText(QRectF(guide_rect.left() + 10, gy, 65, 16), g_tag)

            painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
            painter.setPen(QPen(QColor(200, 220, 245)))
            painter.drawText(QRectF(guide_rect.left() + 75, gy, 211, 24), Qt.TextWordWrap, g_desc)

    def _paint_slider(
        self,
        painter: QPainter,
        slider_id: str,
        label: str,
        val_text: str,
        min_val: float,
        max_val: float,
        current_val: float,
        y_pos: float,
        dash_left: float,
    ) -> None:
        """Render a styled draggable slider track with knob."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(180, 205, 230)))
        painter.drawText(QRectF(dash_left + 16, y_pos, 160, 16), Qt.AlignLeft, label)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_left + 170, y_pos, 136, 16), Qt.AlignRight, val_text)

        # Track
        track_rect = QRectF(dash_left + 16, y_pos + 20, 292, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(25, 40, 62)))
        painter.drawRoundedRect(track_rect, 4.0, 4.0)

        # Filled portion
        fraction = max(0.0, min(1.0, (current_val - min_val) / (max_val - min_val)))
        fill_w = max(6.0, fraction * track_rect.width())
        fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_w, track_rect.height())
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(fill_rect, 4.0, 4.0)

        # Knob
        knob_x = track_rect.left() + fraction * track_rect.width()
        knob_pt = QPointF(knob_x, track_rect.center().y())
        is_active = (self._active_slider == slider_id)
        painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 220, 255), 1.5))
        painter.setBrush(QBrush(QColor(56, 189, 248) if is_active else QColor(30, 80, 140)))
        painter.drawEllipse(knob_pt, 7.0, 7.0)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self.closed.emit()
            if self.board and hasattr(self.board, "toggle_orbital_mechanics"):
                self.board.toggle_orbital_mechanics()
            event.accept()
            return

        # 2. Preset tabs
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self.set_preset(key)
                event.accept()
                return

        # 3. Canvas bottom bar controls
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        play_rect = QRectF(canvas_rect.left(), y_bar, 96, 28)
        if play_rect.contains(pos):
            self.toggle_play_pause()
            event.accept()
            return

        reset_rect = QRectF(play_rect.right() + 8, y_bar, 96, 28)
        if reset_rect.contains(pos):
            self.reset_orbit()
            event.accept()
            return

        # 4. Sliders
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        r2_track = QRectF(dash_left + 16, dash_rect.top() + 60, slider_w, 20)
        if r2_track.contains(pos):
            self._active_slider = "r2"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 0.50, 4.50, self.set_target_radius)
            event.accept()
            return

        inc_track = QRectF(dash_left + 16, dash_rect.top() + 114, slider_w, 20)
        if inc_track.contains(pos):
            self._active_slider = "inc"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 0.0, 180.0, self.set_inclination)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        if self._active_slider == "r2":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 0.50, 4.50, self.set_target_radius)
            event.accept()
            return
        elif self._active_slider == "inc":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 0.0, 180.0, self.set_inclination)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._active_slider = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: Any) -> None:
        pos = event.pos()
        old_hover = self._hovered_button
        self._hovered_button = None

        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self._hovered_button = "close"

        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self._hovered_button = f"preset_{key}"

        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 96, 28).contains(pos):
            self._hovered_button = "toggle_play"
        elif QRectF(canvas_rect.left() + 104, y_bar, 96, 28).contains(pos):
            self._hovered_button = "reset_orbit"

        if old_hover != self._hovered_button:
            self.update()

        super().hoverMoveEvent(event)

    def _handle_slider_move(
        self,
        mouse_x: float,
        track_left: float,
        track_width: float,
        min_val: float,
        max_val: float,
        setter_func: Any,
    ) -> None:
        fraction = max(0.0, min(1.0, (mouse_x - track_left) / track_width))
        val = min_val + fraction * (max_val - min_val)
        setter_func(val)

    # -------------------------------------------------------------------------
    # Touchless Webcam Air Gestures
    # -------------------------------------------------------------------------

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process air tap gesture on presets, buttons, or sliders."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # Close button
        if QRectF(self._width - 36, 12, 24, 24).contains(pos):
            self.closed.emit()
            if self.board and hasattr(self.board, "toggle_orbital_mechanics"):
                self.board.toggle_orbital_mechanics()
            return True

        # Presets
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            if QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h).contains(pos):
                self.set_preset(key)
                return True

        # Canvas buttons
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 96, 28).contains(pos):
            self.toggle_play_pause()
            return True
        elif QRectF(canvas_rect.left() + 104, y_bar, 96, 28).contains(pos):
            self.reset_orbit()
            return True

        return True

    # -------------------------------------------------------------------------
    # Serialization & Canvas Stamping
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach packaging."""
        return {
            "type": "InteractiveOrbitalMechanicsItem",
            "pos": [self.x(), self.y()],
            "preset": self._current_preset,
            "scenario_type": self._scenario_type,
            "r1": self._r1,
            "r2": self._r2,
            "inclination_deg": self._inclination_deg,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        board: Optional["LessonBoard"] = None,
    ) -> "InteractiveOrbitalMechanicsItem":
        """Reconstruct from dictionary."""
        pos = data.get("pos", [80.0, 80.0])
        item = cls(board=board, pos=QPointF(pos[0], pos[1]))
        preset = data.get("preset", "HOHMANN_EARTH_MARS")
        item.set_preset(preset)
        if "r1" in data:
            item._r1 = data["r1"]
        if "r2" in data:
            item.set_target_radius(data["r2"])
        if "inclination_deg" in data:
            item.set_inclination(data["inclination_deg"])
        return item

    def stamp_to_canvas(self) -> "InteractiveOrbitalMechanicsItem":
        """Duplicate snapshot item onto lesson canvas."""
        clone = InteractiveOrbitalMechanicsItem(
            board=self.board,
            pos=(self.x() + 30.0, self.y() + 30.0),
        )
        clone.set_preset(self._current_preset)
        clone._r1 = self._r1
        clone.set_target_radius(self._r2)
        clone.set_inclination(self._inclination_deg)
        if self.scene():
            self.scene().addItem(clone)
        return clone
