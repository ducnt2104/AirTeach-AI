"""
AirTeach AI - Interactive Astronomy Celestial Mechanics & Planetary Orbit Kepler Simulator
==========================================================================================
Provides an interactive astronomy & celestial physics simulation demonstrating:
  - Kepler's First Law (Law of Ellipses):
      Planets move in elliptical orbits with the Sun at one focus.
      Polar equation: r(theta) = a * (1 - e^2) / (1 + e * cos(theta)).
  - Kepler's Second Law (Law of Equal Areas):
      A line segment joining a planet and the Sun sweeps out equal areas during equal intervals of time.
      Area velocity: dA/dt = const => Speed is maximum at perihelion (cận nhật) and minimum at aphelion (viễn nhật).
  - Kepler's Third Law (Harmonic Law):
      The square of the orbital period T is proportional to the cube of the semi-major axis a (T^2 / a^3 = const).
  - Interactive eccentricity slider e in [0.0, 0.85].
  - Real solar system presets: Trái Đất (0.017), Sao Hỏa (0.093), Sao Thủy (0.206), Sao Chổi Halley (0.800).
  - Real-time numerical Keplerian orbital motion with equal-area sector sweeps.
  - Stamping permanent vector celestial mechanics diagrams onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

# Real Solar System Astronomical Presets
KEPLER_PRESETS: dict[str, tuple[str, float, QColor]] = {
    # key: (name_vi, eccentricity_e, planet_color)
    "EARTH": ("Trái Đất", 0.017, QColor("#38bdf8")),
    "MARS": ("Sao Hỏa", 0.093, QColor("#f97316")),
    "MERCURY": ("Sao Thủy", 0.206, QColor("#cbd5e1")),
    "HALLEY": ("Sao Chổi Halley", 0.800, QColor("#a855f7")),
}


class InteractiveKeplerOrbitItem(QGraphicsObject):
    """
    Interactive celestial mechanics simulation of Kepler's laws of planetary motion,
    featuring live elliptical orbital motion, equal-area sector sweeps, and vector stamping.
    """

    eccentricity_changed = Signal(float)  # (eccentricity e: 0.0 to 0.85)
    planet_moved = Signal(float, float)   # (distance r, velocity v)

    def __init__(
        self,
        pos: tuple[float, float] = (510.0, 330.0),
        width: float = 640.0,
        height: float = 430.0,
        semi_major_axis: float = 190.0,
        eccentricity: float = 0.45,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Orbital Parameters
        self._a = float(semi_major_axis)  # Semi-major axis (pixels)
        self._e = max(0.0, min(0.85, float(eccentricity)))  # Eccentricity
        self._active_preset = "CUSTOM"

        # Center of ellipse in local item coordinates
        self._ellipse_cx = self._width * 0.44
        self._ellipse_cy = self._height * 0.54

        # True anomaly angle theta (radians) and orbital period
        self._theta = 0.0  # Angle measured from perihelion (theta = 0)
        self._period = 10.0  # seconds per orbit
        self._is_orbiting = True
        self._show_equal_areas = True

        # Orbit simulation timer
        self._timer = QTimer()
        self._timer.setInterval(25)  # 40 FPS
        self._timer.timeout.connect(self._on_orbit_tick)
        self._timer.start()

        # Dragging state
        self._active_drag: Optional[str] = None  # "ECCENTRICITY", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_play = QRectF(self._width - 245.0, 10.0, 90.0, 26.0)
        self._rect_btn_sector = QRectF(self._width - 365.0, 10.0, 115.0, 26.0)

        # Preset selector buttons across left side of top row
        self._preset_keys = ["EARTH", "MARS", "MERCURY", "HALLEY"]
        self._rect_btn_presets: list[QRectF] = []
        btn_w = 60.0
        start_x = 18.0
        for i in range(len(self._preset_keys)):
            self._rect_btn_presets.append(QRectF(start_x + i * (btn_w + 4.0), 10.0, btn_w, 26.0))

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Celestial Mechanics Mathematics ─────────────────────────────────────

    def semi_major_axis(self) -> float:
        """Semi-major axis a (pixels)."""
        return self._a

    def semi_minor_axis(self) -> float:
        """Semi-minor axis b = a * sqrt(1 - e^2)."""
        return self._a * math.sqrt(max(0.01, 1.0 - self._e * self._e))

    def focal_distance(self) -> float:
        """Linear distance c = a * e from center to Sun focus."""
        return self._a * self._e

    def eccentricity(self) -> float:
        """Orbital eccentricity e in [0.0, 0.85]."""
        return self._e

    def sun_position(self) -> QPointF:
        """Focus point position of the Sun (local item coordinates)."""
        # Sun is at one focus: (-c, 0) relative to ellipse center
        return QPointF(self._ellipse_cx - self.focal_distance(), self._ellipse_cy)

    def empty_focus_position(self) -> QPointF:
        """Secondary empty focus of the ellipse."""
        return QPointF(self._ellipse_cx + self.focal_distance(), self._ellipse_cy)

    def radius_at_theta(self, theta_rad: float) -> float:
        """
        Calculates distance r from Sun to planet using polar Keplerian equation:
          r(theta) = a * (1 - e^2) / (1 + e * cos(theta))
        """
        denom = 1.0 + self._e * math.cos(theta_rad)
        if abs(denom) < 1e-4:
            denom = 1e-4
        return (self._a * (1.0 - self._e * self._e)) / denom

    def planet_position_at_theta(self, theta_rad: float) -> QPointF:
        """Returns planet screen coordinates at true anomaly theta."""
        sun_pt = self.sun_position()
        r = self.radius_at_theta(theta_rad)
        # In our coordinate system, theta = 0 is perihelion (left of Sun)
        px = sun_pt.x() - r * math.cos(theta_rad)
        py = sun_pt.y() - r * math.sin(theta_rad)
        return QPointF(px, py)

    def current_planet_position(self) -> QPointF:
        """Returns current planet coordinates."""
        return self.planet_position_at_theta(self._theta)

    def perihelion_distance(self) -> float:
        """Distance at perihelion (closest approach, theta = 0): r_p = a * (1 - e)."""
        return self._a * (1.0 - self._e)

    def aphelion_distance(self) -> float:
        """Distance at aphelion (furthest distance, theta = pi): r_a = a * (1 + e)."""
        return self._a * (1.0 + self._e)

    def orbital_velocity(self) -> float:
        """
        Calculates relative orbital velocity from Vis-Viva equation:
          v = sqrt(GM * (2/r - 1/a)).
        Normalized so average velocity = 1.0.
        """
        r = self.radius_at_theta(self._theta)
        val = 2.0 / r - 1.0 / self._a
        return math.sqrt(max(0.01, val * self._a))

    def is_orbiting(self) -> bool:
        """Returns True if the celestial orbit animation is active."""
        return self._is_orbiting

    # ─── Mutators & Controls ─────────────────────────────────────────────────

    def set_eccentricity(self, e: float) -> None:
        """Sets orbital eccentricity in [0.0, 0.85]."""
        self._e = max(0.0, min(0.85, float(e)))
        self._active_preset = "CUSTOM"
        self.eccentricity_changed.emit(self._e)
        self.update()

    def set_preset(self, preset_key: str) -> None:
        """Applies a real astronomical preset (EARTH, MARS, MERCURY, HALLEY)."""
        if preset_key in KEPLER_PRESETS:
            name, e_val, _ = KEPLER_PRESETS[preset_key]
            self._e = e_val
            self._active_preset = preset_key
            self.eccentricity_changed.emit(self._e)
            self.update()

    def toggle_orbit(self) -> bool:
        """Toggles play/pause state for planetary motion."""
        self._is_orbiting = not self._is_orbiting
        if self._is_orbiting:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()
        return self._is_orbiting

    def toggle_equal_areas(self) -> bool:
        """Toggles Kepler's Second Law equal area sector wedges."""
        self._show_equal_areas = not self._show_equal_areas
        self.update()
        return self._show_equal_areas

    def _on_orbit_tick(self) -> None:
        """
        Advances true anomaly theta according to Kepler's Second Law (dtheta/dt = h / r^2).
        Planets move faster at perihelion and slower at aphelion.
        """
        if self._is_orbiting:
            r = self.radius_at_theta(self._theta)
            # Area constant h = 2 * pi * a * b / T
            b = self.semi_minor_axis()
            h_area = (2.0 * math.pi * self._a * b) / self._period
            dt = 0.025
            d_theta = (h_area / (r * r)) * dt
            self._theta = (self._theta + d_theta) % (2.0 * math.pi)
            self.planet_moved.emit(r, self.orbital_velocity())
            self.update()

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive Kepler celestial mechanics simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height

        # 1. Container Frosted Glass Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Render Elliptical Orbit & Major/Minor Axes
        self._draw_orbital_axes_and_ellipse(painter)

        # 3. Kepler's Second Law Equal-Area Sector Sweeps
        if self._show_equal_areas:
            self._draw_equal_area_sectors(painter)

        # 4. Sun & Foci
        self._draw_sun_and_foci(painter)

        # 5. Radius Vector & Orbiting Planet
        self._draw_planet_and_radius_vector(painter)

        # 6. Celestial HUD Badge (Bottom Right)
        self._draw_celestial_hud(painter)

        # 7. Control Buttons
        self._draw_buttons(painter)

        painter.restore()

    def _draw_orbital_axes_and_ellipse(self, painter: QPainter) -> None:
        """Draws major axis line, minor axis line, and crisp orbit ellipse."""
        cx = self._ellipse_cx
        cy = self._ellipse_cy
        a = self._a
        b = self.semi_minor_axis()

        # Major axis line (dashed)
        pen_axis = QPen(QColor("#475569"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen_axis)
        painter.drawLine(QPointF(cx - a - 25.0, cy), QPointF(cx + a + 25.0, cy))

        # Minor axis line (dashed)
        painter.drawLine(QPointF(cx, cy - b - 15.0), QPointF(cx, cy + b + 15.0))

        # Ellipse Orbit Path
        pen_orbit_glow = QPen(QColor(56, 189, 248, 60), 5.0, Qt.PenStyle.SolidLine)
        painter.setPen(pen_orbit_glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), a, b)

        pen_orbit = QPen(QColor("#0284c7"), 2.2, Qt.PenStyle.SolidLine)
        painter.setPen(pen_orbit)
        painter.drawEllipse(QPointF(cx, cy), a, b)

        # Perihelion (Cận nhật) and Aphelion (Viễn nhật) text markers
        f_lbl = QFont("Segoe UI", 8)
        f_lbl.setBold(True)
        painter.setFont(f_lbl)
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QPointF(cx - a - 20.0, cy - 8.0), "Cận nhật (P)")
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QPointF(cx + a - 45.0, cy - 8.0), "Viễn nhật (A)")

    def _draw_equal_area_sectors(self, painter: QPainter) -> None:
        """
        Draws two equal-area sector sweeps demonstrating Kepler's Second Law:
          - Sector 1: Near perihelion (wide angle sweep, short radius)
          - Sector 2: Near aphelion (narrow angle sweep, long radius)
        """
        sun = self.sun_position()

        # Sector 1: Near Perihelion (theta around 0)
        p_sector1 = QPainterPath()
        p_sector1.moveTo(sun)
        ang_step = math.radians(24)
        for step in range(13):
            theta_sub = -ang_step * 0.5 + (step / 12.0) * ang_step
            pt = self.planet_position_at_theta(theta_sub)
            p_sector1.lineTo(pt)
        p_sector1.closeSubpath()

        painter.setPen(QPen(QColor("#06b6d4"), 1.2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(6, 182, 212, 50)))
        painter.drawPath(p_sector1)

        # Sector 2: Near Aphelion (theta around pi)
        # Because radius is longer (1+e)/(1-e), angle is smaller for same area: dA = 0.5 * r^2 * dtheta
        ratio = ((1.0 - self._e) / max(0.1, 1.0 + self._e)) ** 2
        ang_step2 = max(math.radians(3), ang_step * ratio)

        p_sector2 = QPainterPath()
        p_sector2.moveTo(sun)
        for step in range(13):
            theta_sub2 = math.pi - ang_step2 * 0.5 + (step / 12.0) * ang_step2
            pt = self.planet_position_at_theta(theta_sub2)
            p_sector2.lineTo(pt)
        p_sector2.closeSubpath()

        painter.setPen(QPen(QColor("#f59e0b"), 1.2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(245, 158, 11, 50)))
        painter.drawPath(p_sector2)

        # Equal area badges: A1 = A2
        f_area = QFont("Segoe UI", 8)
        f_area.setBold(True)
        painter.setFont(f_area)
        painter.setPen(QPen(QColor("#67e8f9")))
        pt_sec1 = self.planet_position_at_theta(0.0)
        painter.drawText(QPointF((sun.x() + pt_sec1.x()) * 0.5 - 10.0, sun.y() - 14.0), "S₁")

        painter.setPen(QPen(QColor("#fde68a")))
        pt_sec2 = self.planet_position_at_theta(math.pi)
        painter.drawText(QPointF((sun.x() + pt_sec2.x()) * 0.5 - 10.0, sun.y() - 14.0), "S₂ = S₁")

    def _draw_sun_and_foci(self, painter: QPainter) -> None:
        """Renders the Sun at one focus with glowing corona and empty focus dot."""
        sun_pt = self.sun_position()
        r_sun = 14.0

        # Sun Corona Glow
        grad_sun = QRadialGradient(sun_pt.x(), sun_pt.y(), r_sun * 2.2)
        grad_sun.setColorAt(0.0, QColor(254, 240, 138, 240))
        grad_sun.setColorAt(0.4, QColor(245, 158, 11, 200))
        grad_sun.setColorAt(1.0, QColor(234, 88, 12, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad_sun))
        painter.drawEllipse(sun_pt, r_sun * 2.2, r_sun * 2.2)

        # Sun Core Sphere
        grad_core = QRadialGradient(sun_pt.x() - r_sun * 0.3, sun_pt.y() - r_sun * 0.3, r_sun * 1.3)
        grad_core.setColorAt(0.0, QColor("#ffffff"))
        grad_core.setColorAt(0.3, QColor("#fbbf24"))
        grad_core.setColorAt(1.0, QColor("#ea580c"))

        painter.setPen(QPen(QColor("#d97706"), 1.2))
        painter.setBrush(QBrush(grad_core))
        painter.drawEllipse(sun_pt, r_sun, r_sun)

        # Label: Mặt Trời
        f_sun = QFont("Segoe UI", 8)
        f_sun.setBold(True)
        painter.setFont(f_sun)
        painter.setPen(QPen(QColor("#fef08a")))
        painter.drawText(QPointF(sun_pt.x() - 20.0, sun_pt.y() + r_sun + 14.0), "Mặt Trời (F₁)")

        # Empty Focus (F2)
        f2_pt = self.empty_focus_position()
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor(100, 116, 139, 120)))
        painter.drawEllipse(f2_pt, 3.5, 3.5)
        painter.drawText(QPointF(f2_pt.x() - 6.0, f2_pt.y() + 14.0), "F₂")

    def _draw_planet_and_radius_vector(self, painter: QPainter) -> None:
        """Renders the orbiting planet with shaded sphere and radius vector r."""
        sun_pt = self.sun_position()
        planet_pt = self.current_planet_position()
        r_planet = 9.0

        # Radius Vector Line (from Sun to Planet)
        pen_vec = QPen(QColor("#38bdf8"), 1.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen_vec)
        painter.drawLine(sun_pt, planet_pt)

        # Planet Sphere with specular radial gradient
        color_info = KEPLER_PRESETS.get(self._active_preset, ("Hành tinh", 0.0, QColor("#38bdf8")))[2]
        grad_planet = QRadialGradient(planet_pt.x() - r_planet * 0.3, planet_pt.y() - r_planet * 0.3, r_planet * 1.3)
        grad_planet.setColorAt(0.0, QColor("#ffffff"))
        grad_planet.setColorAt(0.3, color_info)
        grad_planet.setColorAt(1.0, color_info.darker(150))

        painter.setPen(QPen(color_info.darker(160), 1.2))
        painter.setBrush(QBrush(grad_planet))
        painter.drawEllipse(planet_pt, r_planet, r_planet)

        # Velocity Arrow Vector (Tangent to orbit)
        vel = self.orbital_velocity()
        v_scale = 22.0 * vel
        # Compute tangent direction
        r_curr = self.radius_at_theta(self._theta)
        d_theta_test = 0.05
        pt_next = self.planet_position_at_theta(self._theta + d_theta_test)
        tdx = pt_next.x() - planet_pt.x()
        tdy = pt_next.y() - planet_pt.y()
        t_len = math.hypot(tdx, tdy)
        if t_len > 1e-2:
            tux, tuy = tdx / t_len, tdy / t_len
            v_end = QPointF(planet_pt.x() + tux * v_scale, planet_pt.y() + tuy * v_scale)
            pen_v = QPen(QColor("#22c55e"), 2.0, Qt.PenStyle.SolidLine)
            painter.setPen(pen_v)
            painter.drawLine(planet_pt, v_end)
            # Arrowhead
            self._draw_arrowhead(painter, planet_pt, v_end, QColor("#22c55e"))

    def _draw_arrowhead(self, painter: QPainter, p_start: QPointF, p_end: QPointF, color: QColor) -> None:
        """Draws small directional arrow on velocity vector."""
        dx = p_end.x() - p_start.x()
        dy = p_end.y() - p_start.y()
        length = math.hypot(dx, dy)
        if length < 1e-2:
            return

        ux, uy = dx / length, dy / length
        arr_size = 7.0
        wing_ang = math.radians(28)
        left_ang = math.atan2(uy, ux) + math.pi - wing_ang
        right_ang = math.atan2(uy, ux) + math.pi + wing_ang

        path = QPainterPath()
        path.moveTo(p_end)
        path.lineTo(p_end.x() + arr_size * math.cos(left_ang), p_end.y() + arr_size * math.sin(left_ang))
        path.lineTo(p_end.x() + arr_size * math.cos(right_ang), p_end.y() + arr_size * math.sin(right_ang))
        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _draw_celestial_hud(self, painter: QPainter) -> None:
        """Renders Kepler's Laws information card."""
        hud_w = 260.0
        hud_h = 135.0
        hud_x = self._width - hud_w - 18.0
        hud_y = self._height - hud_h - 18.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        preset_name = KEPLER_PRESETS.get(self._active_preset, ("Tùy chỉnh", 0.0, None))[0]
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 8.0, hud_w - 20.0, 18.0), f"3 Định Luật Kepler ({preset_name})")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        # Law 1
        r_p = self.perihelion_distance()
        r_a = self.aphelion_distance()
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 28.0, hud_w - 20.0, 16.0), f"1. Quỹ đạo Elip: Tâm sai e = {self._e:.3f}")
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 44.0, hud_w - 20.0, 16.0), f"   Cận nhật rp = {r_p:.0f}px | Viễn nhật ra = {r_a:.0f}px")

        # Law 2
        vel = self.orbital_velocity()
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 62.0, hud_w - 20.0, 16.0), f"2. Vận tốc diện tích: dA/dt = const")
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 78.0, hud_w - 20.0, 16.0), f"   Vận tốc tức thời: v = {vel:.2f}·v_tb (max tại P)")

        # Law 3
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 96.0, hud_w - 20.0, 16.0), f"3. Chu kỳ điều hòa: T²/a³ = 4π²/(G·M)")
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 112.0, hud_w - 20.0, 16.0), f"   Bán trục: a = {self._a:.0f}px | b = {self.semi_minor_axis():.0f}px")

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders celestial preset buttons, play/pause, area toggle, stamp, close."""
        f_btn = QFont("Segoe UI", 8)
        f_btn.setBold(True)
        painter.setFont(f_btn)

        # Close button [✕]
        painter.setPen(QPen(QColor("#ef4444"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
        painter.drawRoundedRect(self._rect_btn_close, 5.0, 5.0)
        painter.setPen(QPen(QColor("#ef4444")))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # Stamp button [📌 Đóng dấu]
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.setBrush(QBrush(QColor(56, 189, 248, 40)))
        painter.drawRoundedRect(self._rect_btn_stamp, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0f2fe")))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # Play / Pause button
        play_txt = "⏸ Tạm dừng" if self._is_orbiting else "▶ Chuyển động"
        play_col = QColor("#f59e0b") if self._is_orbiting else QColor("#10b981")
        painter.setPen(QPen(play_col, 1.2))
        painter.setBrush(QBrush(QColor(play_col.red(), play_col.green(), play_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_play, 6.0, 6.0)
        painter.setPen(QPen(play_col))
        painter.drawText(self._rect_btn_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # Sector sweep toggle button [📐 Quét S]
        sec_col = QColor("#06b6d4") if self._show_equal_areas else QColor("#64748b")
        painter.setPen(QPen(sec_col, 1.2))
        painter.setBrush(QBrush(QColor(sec_col.red(), sec_col.green(), sec_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_sector, 6.0, 6.0)
        painter.setPen(QPen(sec_col))
        painter.drawText(self._rect_btn_sector, Qt.AlignmentFlag.AlignCenter, "📐 Quét S₁=S₂")

        # Presets buttons: EARTH, MARS, MERCURY, HALLEY
        for i, key in enumerate(self._preset_keys):
            rect = self._rect_btn_presets[i]
            is_active = (key == self._active_preset)
            name_vi, _, col = KEPLER_PRESETS[key]

            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.4))
                painter.setBrush(QBrush(QColor(56, 189, 248, 120)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, name_vi)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles button clicks or eccentricity dragging."""
        pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_play.contains(pos):
            self.toggle_orbit()
            event.accept()
            return

        if self._rect_btn_sector.contains(pos):
            self.toggle_equal_areas()
            event.accept()
            return

        # Check preset buttons
        for i, rect in enumerate(self._rect_btn_presets):
            if rect.contains(pos):
                self.set_preset(self._preset_keys[i])
                event.accept()
                return

        # Check stamp button
        if self._rect_btn_stamp.contains(pos):
            scene = self.scene()
            if scene is not None:
                board = getattr(scene, "board", None)
                if board is None and hasattr(scene, "views") and scene.views():
                    view = scene.views()[0]
                    board = view if hasattr(view, "addItem") else getattr(view, "board", None)
                if board is not None:
                    self.stamp_to_canvas(board)
            event.accept()
            return

        # Perihelion dragging to modify eccentricity
        pt_peri = self.planet_position_at_theta(0.0)
        if (pos - pt_peri).manhattanLength() < 28.0:
            self._active_drag = "ECCENTRICITY"
            event.accept()
            return

        # Background drag
        self._active_drag = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates eccentricity during dragging."""
        if self._active_drag == "ECCENTRICITY":
            # Distance from center to cursor x
            cx = self._ellipse_cx
            cursor_x = event.pos().x()
            # In our system perihelion is to the left: cx - a
            # Change eccentricity e = (cx - cursor_x - a) / ...
            delta_p = (cx - cursor_x) / self._a
            new_e = max(0.0, min(0.85, delta_p - 0.5))
            self.set_eccentricity(new_e)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases drag state."""
        self._active_drag = None
        super().mouseReleaseEvent(event)

    # ─── Touchless Air Gesture Integration ───────────────────────────────────

    def set_air_hover_pos(self, scene_pos: Optional[QPointF]) -> None:
        """Sets touchless air pointer hover position."""
        if scene_pos is not None:
            self._air_hover_pos = self.mapFromScene(scene_pos)
        else:
            self._air_hover_pos = None
        self.update()

    def handle_air_drag(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch drag of eccentricity or entire item."""
        pos = self.mapFromScene(scene_pos)

        if self._active_drag is None:
            pt_peri = self.planet_position_at_theta(0.0)
            if (pos - pt_peri).manhattanLength() < 35.0:
                self._active_drag = "ECCENTRICITY"
            else:
                self._active_drag = "BODY"
                self._drag_start_item_pos = pos

        if self._active_drag == "ECCENTRICITY":
            cx = self._ellipse_cx
            delta_p = (cx - pos.x()) / self._a
            new_e = max(0.0, min(0.85, delta_p - 0.5))
            self.set_eccentricity(new_e)
            return True

        if self._active_drag == "BODY":
            new_pos = scene_pos - self._drag_start_item_pos
            self.setPos(new_pos)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Touchless air tap targeting control buttons."""
        pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(pos):
            self.close()
            return True

        if self._rect_btn_play.contains(pos):
            self.toggle_orbit()
            return True

        if self._rect_btn_sector.contains(pos):
            self.toggle_equal_areas()
            return True

        for i, rect in enumerate(self._rect_btn_presets):
            if rect.contains(pos):
                self.set_preset(self._preset_keys[i])
                return True

        if self._rect_btn_stamp.contains(pos):
            target_board = board
            if target_board is None and self.scene() is not None:
                scene = self.scene()
                target_board = getattr(scene, "board", None)
            if target_board is not None:
                self.stamp_to_canvas(target_board)
                return True

        return False

    def close(self) -> None:
        """Stops orbit timer and removes this interactive item from its scene."""
        self._timer.stop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps vector elliptical orbit, Sun, foci, and Keplerian formulas
        onto LessonBoard via MathTools.create_conic_section.
        """
        from src.ui.stem_tools import MathTools
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem

        stamped: list[QGraphicsItem] = []
        origin = self.pos()
        cx = origin.x() + self._ellipse_cx
        cy = origin.y() + self._ellipse_cy
        a = self._a
        b = self.semi_minor_axis()

        # 1. Stamped Ellipse
        conic_items = MathTools.create_conic_section(
            center=(cx, cy),
            conic_type="ellipse",
            a=a,
            b=b,
            show_foci=True,
            show_axes=True,
            pen_color=QColor("#0284c7"),
        )
        for itm in conic_items:
            board.addItem(itm)
            stamped.append(itm)

        # 2. Sun Marker at Focus F1
        sun_pt = origin + self.sun_position()
        sun_item = QGraphicsEllipseItem(sun_pt.x() - 12.0, sun_pt.y() - 12.0, 24.0, 24.0)
        sun_item.setPen(QPen(QColor("#d97706"), 1.5))
        sun_item.setBrush(QBrush(QColor("#fbbf24")))
        sun_item.setZValue(12.0)
        board.addItem(sun_item)
        stamped.append(sun_item)

        # 3. Text Card with Kepler's Three Laws
        preset_name = KEPLER_PRESETS.get(self._active_preset, ("Tùy chỉnh", 0.0, None))[0]
        lbl_text = (
            f"3 Định Luật Kepler Về Chuyển Động Thiên Thể ({preset_name}):\n"
            f"1. Quỹ đạo elip: r(θ) = a(1-e²)/(1+e cos θ) [e = {self._e:.3f}]\n"
            f"   Cận nhật: rp = {self.perihelion_distance():.0f} px | Viễn nhật: ra = {self.aphelion_distance():.0f} px\n"
            f"2. Vận tốc diện tích: dA/dt = const (quét diện tích bằng nhau)\n"
            f"3. Chu kỳ điều hòa: T²/a³ = const = 4π²/(GM)"
        )
        txt_item = QGraphicsTextItem(lbl_text)
        txt_font = QFont("Segoe UI", 10)
        txt_font.setBold(True)
        txt_item.setFont(txt_font)
        txt_item.setDefaultTextColor(QColor("#f8fafc"))
        txt_item.setPos(cx - a, cy + b + 25.0)
        txt_item.setZValue(12.5)
        board.addItem(txt_item)
        stamped.append(txt_item)

        return stamped
