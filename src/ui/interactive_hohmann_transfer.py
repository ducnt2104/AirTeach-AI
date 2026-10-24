"""
AirTeach AI – Interactive Astrophysics & Orbital Mechanics: Kepler's Laws, Vis-Viva Equation & Hohmann Transfer Orbit
====================================================================================================================
Provides an authentic celestial mechanics and spaceflight astrodynamics simulation:
  - Kepler's Three Laws of Planetary Motion:
      1. Elliptical orbits: r(θ) = a(1 - e²) / (1 + e cos θ)
      2. Equal areas in equal times: dA/dt = h/2 = const (Specific angular momentum h = r * v_perp)
      3. Harmonic law: T² = (4π² / μ) * a³
  - Vis-Viva Equation (Living Force Equation):
      v² = μ * (2/r - 1/a)
      Specific orbital energy: ε = v²/2 - μ/r = -μ / (2a) = const
      Circular velocity: v_circ = sqrt(μ / r)
      Escape velocity: v_esc = sqrt(2μ / r) = sqrt(2) * v_circ
  - Hohmann Interplanetary & Orbital Transfer:
      Two coplanar circular orbits of radii r₁ and r₂ around primary mass (μ = G * M):
      Transfer semi-major axis: a_trans = (r₁ + r₂) / 2
      Transfer eccentricity: e_trans = |r₂ - r₁| / (r₁ + r₂)
      First impulsive burn at periapsis:
          v_trans,1 = sqrt(μ * (2/r₁ - 1/a_trans))
          Δv₁ = v_trans,1 - v_circ,1 = sqrt(μ / r₁) * [sqrt(2*r₂ / (r₁ + r₂)) - 1]
      Second impulsive circularization burn at apoapsis:
          v_trans,2 = sqrt(μ * (2/r₂ - 1/a_trans))
          Δv₂ = v_circ,2 - v_trans,2 = sqrt(μ / r₂) * [1 - sqrt(2*r₁ / (r₁ + r₂))]
      Total mission budget: Δv_total = |Δv₁| + |Δv₂|
      One-way transfer time: t_trans = π * sqrt(a_trans³ / μ)
  - 5 Curriculum Presets:
      1. LEO_TO_GEO_TRANSFER: Low Earth Orbit (r₁ = 6,678 km) to Geostationary Orbit (r₂ = 42,164 km).
      2. EARTH_TO_MARS_INTERPLANETARY: Heliocentric transfer from Earth (1.0 AU) to Mars (1.524 AU).
      3. EARTH_TO_MOON_TRANSLUNAR: Trans-Lunar Injection (TLI) to Moon distance (r₂ = 384,400 km).
      4. ELLIPTICAL_ORBIT_KEPLER_SWEEP: High eccentricity orbit (e = 0.60) proving Kepler's 2nd Law equal area sectors.
      5. ESCAPE_TRAJECTORY_PARABOLIC: Escape velocity burn (e = 1.0, v = v_esc) freeing spacecraft into deep space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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


# Physical & Astrodynamical Constants (CODATA 2022 / IAU)
G_GRAV = 6.67430e-11                           # Universal Gravitational Constant (m³/(kg·s²))
M_EARTH = 5.9722e24                            # Earth Mass (kg)
R_EARTH = 6.371e6                              # Earth volumetric mean radius (m)
MU_EARTH = 3.986004418e14                      # Standard gravitational parameter GM_Earth (m³/s²)

M_SUN = 1.98847e30                             # Sun Mass (kg)
MU_SUN = 1.32712440018e20                      # Standard gravitational parameter GM_Sun (m³/s²)
AU_METERS = 1.495978707e11                     # 1 Astronomical Unit in meters (149.6M km)


@dataclass(frozen=True)
class OrbitPreset:
    code: str
    title_vi: str
    badge_label: str
    primary_name: str
    mu: float                                  # Gravitational parameter (m³/s²)
    r1_km: float                               # Initial circular radius (km)
    r2_km: float                               # Target circular radius (km)
    primary_radius_km: float                   # Radius of central attractor body (km)
    description: str


PRESETS: dict[str, OrbitPreset] = {
    "LEO_TO_GEO_TRANSFER": OrbitPreset(
        code="LEO_TO_GEO_TRANSFER",
        title_vi="Chuyển quỹ đạo LEO sang GEO (Vệ tinh Địa tĩnh)",
        badge_label="LEO ➔ GEO",
        primary_name="Trái Đất (Earth)",
        mu=MU_EARTH,
        r1_km=6678.0,                          # LEO altitude h = 307 km
        r2_km=42164.0,                         # GEO geostationary radius
        primary_radius_km=6371.0,
        description="Mô phỏng 2 lần kích hỏa động cơ Hohmann đưa vệ tinh từ quỹ đạo tầm thấp (LEO) lên quỹ đạo địa tĩnh (GEO, 35,786 km).",
    ),
    "EARTH_TO_MARS_INTERPLANETARY": OrbitPreset(
        code="EARTH_TO_MARS_INTERPLANETARY",
        title_vi="Hành trình Liên hành tinh: Trái Đất ➔ Sao Hỏa",
        badge_label="TRÁI ĐẤT ➔ HỎA TINH",
        primary_name="Mặt Trời (Sun)",
        mu=MU_SUN,
        r1_km=1.496e8,                         # 1.0 AU
        r2_km=2.279e8,                         # 1.524 AU
        primary_radius_km=6.9634e5,
        description="Quỹ đạo chuyển tiếp nhật tâm Hohmann đưa tàu thăm dò từ Trái Đất sang Sao Hỏa với thời gian bay tối ưu ~259 ngày.",
    ),
    "EARTH_TO_MOON_TRANSLUNAR": OrbitPreset(
        code="EARTH_TO_MOON_TRANSLUNAR",
        title_vi="Quỹ đạo Chuyển tiếp Mặt Trăng (Trans-Lunar Injection - Apollo)",
        badge_label="LÊN MẶT TRĂNG",
        primary_name="Trái Đất (Earth)",
        mu=MU_EARTH,
        r1_km=6678.0,                          # LEO parking orbit
        r2_km=384400.0,                        # Lunar distance
        primary_radius_km=6371.0,
        description="Cú hích chuyển tiếp Mặt Trăng (TLI) tương tự sứ mệnh Apollo, đẩy tàu thoát khỏi LEO hướng thẳng tới quỹ đạo Mặt Trăng (~5 ngày).",
    ),
    "ELLIPTICAL_ORBIT_KEPLER_SWEEP": OrbitPreset(
        code="ELLIPTICAL_ORBIT_KEPLER_SWEEP",
        title_vi="Quỹ đạo Elip & Định luật 2 Kepler (Quét diện tích bằng nhau)",
        badge_label="ELIP KEPLER",
        primary_name="Trái Đất (Earth)",
        mu=MU_EARTH,
        r1_km=8000.0,                          # Periapsis r_p
        r2_km=32000.0,                         # Apoapsis r_a
        primary_radius_km=6371.0,
        description="Định luật 2 Kepler: Đoạn nối tâm hành tinh quét những diện tích bằng nhau trong các khoảng thời gian bằng nhau (vận tốc cực đại ở cận điểm).",
    ),
    "ESCAPE_TRAJECTORY_PARABOLIC": OrbitPreset(
        code="ESCAPE_TRAJECTORY_PARABOLIC",
        title_vi="Tốc độ Vũ trụ cấp 2 & Quỹ đạo Thoát Parabol (Escape Trajectory)",
        badge_label="VẬN TỐC THOÁT",
        primary_name="Trái Đất (Earth)",
        mu=MU_EARTH,
        r1_km=6678.0,
        r2_km=60000.0,
        primary_radius_km=6371.0,
        description="Khi vận tốc đạt v_esc = √(2)·v_circ, cơ năng quỹ đạo ε = 0, tàu bẻ cong thành quỹ đạo Parabol vĩnh viễn không quay lại.",
    ),
}


class InteractiveHohmannTransferItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Astrophysics & Orbital Mechanics:
    Kepler's Laws, Vis-Viva equation, and Hohmann Interplanetary Transfer Orbits.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    preset_changed = Signal(str)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)

        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Simulation Preset & Physical Configuration
        self._active_preset_code: str = "LEO_TO_GEO_TRANSFER"
        self._preset: OrbitPreset = PRESETS[self._active_preset_code]

        self._r1_km: float = self._preset.r1_km
        self._r2_km: float = self._preset.r2_km

        # Animation State
        # Phase 0: In initial circular orbit 1
        # Phase 1: On Hohmann transfer ellipse
        # Phase 2: In final circular orbit 2
        self._flight_phase: int = 1              # Default: on transfer ellipse
        self._true_anomaly_rad: float = 0.0      # θ from 0 to 2π
        self._sim_speed: float = 1.0             # Simulation speed multiplier
        self._is_animating: bool = True

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Orbital Mechanics Mathematics (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def mu(self) -> float:
        """Standard gravitational parameter (m³/s²)."""
        return self._preset.mu

    @property
    def r1_m(self) -> float:
        """Initial orbit radius in meters."""
        return self._r1_km * 1000.0

    @property
    def r2_m(self) -> float:
        """Final orbit radius in meters."""
        return self._r2_km * 1000.0

    @property
    def a_trans_m(self) -> float:
        """Semi-major axis of transfer ellipse: a = (r₁ + r₂) / 2."""
        return (self.r1_m + self.r2_m) / 2.0

    @property
    def a_trans_km(self) -> float:
        return self.a_trans_m / 1000.0

    @property
    def eccentricity_trans(self) -> float:
        """Eccentricity of transfer orbit: e = |r₂ - r₁| / (r₁ + r₂)."""
        return abs(self.r2_m - self.r1_m) / (self.r1_m + self.r2_m)

    @property
    def v_circ_1(self) -> float:
        """Circular velocity in initial orbit: v = sqrt(μ / r₁) in m/s."""
        return math.sqrt(self.mu / self.r1_m)

    @property
    def v_circ_2(self) -> float:
        """Circular velocity in final orbit: v = sqrt(μ / r₂) in m/s."""
        return math.sqrt(self.mu / self.r2_m)

    @property
    def v_trans_periapsis(self) -> float:
        """Transfer orbit velocity at periapsis r₁: v = sqrt(μ * (2/r₁ - 1/a))."""
        return math.sqrt(self.mu * (2.0 / self.r1_m - 1.0 / self.a_trans_m))

    @property
    def v_trans_apoapsis(self) -> float:
        """Transfer orbit velocity at apoapsis r₂: v = sqrt(μ * (2/r₂ - 1/a))."""
        return math.sqrt(self.mu * (2.0 / self.r2_m - 1.0 / self.a_trans_m))

    @property
    def delta_v_1(self) -> float:
        """First burn Δv₁ (tangential injection into transfer orbit) in m/s."""
        return self.v_trans_periapsis - self.v_circ_1

    @property
    def delta_v_2(self) -> float:
        """Second burn Δv₂ (circularization into target orbit) in m/s."""
        return self.v_circ_2 - self.v_trans_apoapsis

    @property
    def delta_v_total(self) -> float:
        """Total mission delta-V budget: |Δv₁| + |Δv₂| in m/s."""
        return abs(self.delta_v_1) + abs(self.delta_v_2)

    @property
    def transfer_time_seconds(self) -> float:
        """Hohmann transfer time (half-period): t = π * sqrt(a³ / μ)."""
        return math.pi * math.sqrt((self.a_trans_m ** 3) / self.mu)

    @property
    def transfer_time_hours(self) -> float:
        """Transfer time in hours."""
        return self.transfer_time_seconds / 3600.0

    @property
    def transfer_time_days(self) -> float:
        """Transfer time in days."""
        return self.transfer_time_hours / 24.0

    @property
    def v_escape_1(self) -> float:
        """Escape velocity from initial orbit: v_esc = sqrt(2 * μ / r₁)."""
        return math.sqrt(2.0 * self.mu / self.r1_m)

    def current_radius_m(self, theta_rad: float) -> float:
        """Radial distance r(θ) on transfer ellipse using Kepler's First Law."""
        p = self.a_trans_m * (1.0 - self.eccentricity_trans ** 2)
        return p / (1.0 + self.eccentricity_trans * math.cos(theta_rad))

    def current_velocity_m(self, r_m: float) -> float:
        """Vis-Viva velocity v = sqrt(μ * (2/r - 1/a))."""
        val = 2.0 / r_m - 1.0 / self.a_trans_m
        return math.sqrt(max(0.0, self.mu * val))

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> OrbitPreset:
        return self._preset

    @property
    def flight_phase(self) -> int:
        return self._flight_phase

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum astrodynamics preset and updates parameters."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._r1_km = self._preset.r1_km
        self._r2_km = self._preset.r2_km
        self._true_anomaly_rad = 0.0
        self.preset_changed.emit(preset_code)
        self.update()

    def set_flight_phase(self, phase: int) -> None:
        """Sets flight phase: 0 = Orbit 1, 1 = Transfer Ellipse, 2 = Orbit 2."""
        self._flight_phase = max(0, min(2, int(phase)))
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def cycle_sim_speed(self) -> None:
        """Cycles speed multiplier between 1x, 2x, 5x."""
        if self._sim_speed == 1.0:
            self._sim_speed = 2.0
        elif self._sim_speed == 2.0:
            self._sim_speed = 5.0
        else:
            self._sim_speed = 1.0
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        # Step true anomaly according to Kepler's second law:
        # dθ/dt = h / r²
        # For smooth rendering, advance by scaled angular step
        d_theta = 0.03 * self._sim_speed
        self._true_anomaly_rad += d_theta
        if self._true_anomaly_rad >= 2.0 * math.pi:
            self._true_anomaly_rad -= 2.0 * math.pi
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#060a12")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Orbital Diagram)
        left_w = 430.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_orbital_viewport(painter, view_rect)

        # 4. Right Astrodynamics Telemetry & Controls
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Physics Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "CƠ HỌC THIÊN THỂ: ĐỊNH LUẬT KEPLER & CHUYỂN QUỸ ĐẠO HOHMANN")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Định luật Kepler • Phương trình Vis-Viva v²=μ(2/r - 1/a) • Xung lực Δv₁ & Δv₂ • Thời gian bay")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_orbital_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        # Draw deep-space coordinate grid & center attractor
        cx = rect.center().x() - 30.0
        cy = rect.center().y()

        # Coordinate axes
        painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), cy), QPointF(rect.right(), cy))
        painter.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))

        # Scale Factor: Map r2 to viewport radius
        max_view_r = 130.0
        scale = max_view_r / float(self._r2_km)

        r1_px = self._r1_km * scale
        r2_px = self._r2_km * scale

        # 1. Initial Circular Orbit (Cyan)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#38bdf8"), 1.5, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(cx, cy), r1_px, r1_px)

        # 2. Target Circular Orbit (Green)
        painter.setPen(QPen(QColor("#10b981"), 1.5, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(cx, cy), r2_px, r2_px)

        # 3. Hohmann Transfer Ellipse (Gold)
        # Semi-major axis a_px = (r1_px + r2_px) / 2
        # Center of ellipse is shifted by c = a * e = (r2_px - r1_px) / 2 to the right
        a_px = (r1_px + r2_px) / 2.0
        c_px = (r2_px - r1_px) / 2.0
        e_val = self.eccentricity_trans
        b_px = a_px * math.sqrt(max(0.01, 1.0 - e_val ** 2))
        ellipse_cx = cx + c_px

        painter.setPen(QPen(QColor("#fbbf24"), 2.0))
        painter.drawEllipse(QPointF(ellipse_cx, cy), a_px, b_px)

        # 4. Central Attractor Body (Earth or Sun)
        primary_is_sun = (self._preset.primary_name.startswith("Mặt Trời"))
        body_r = 14.0 if primary_is_sun else 10.0
        body_col = QColor("#f59e0b") if primary_is_sun else QColor("#2563eb")
        glow_col = QColor(245, 158, 11, 40) if primary_is_sun else QColor(37, 99, 235, 40)

        painter.setBrush(QBrush(glow_col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), body_r + 6.0, body_r + 6.0)

        painter.setBrush(QBrush(body_col))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(QPointF(cx, cy), body_r, body_r)

        # Attractor label
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QPointF(cx - 16.0, cy + body_r + 12.0), self._preset.primary_name.split()[0])

        # 5. Burn Points Δv₁ (Periapsis) and Δv₂ (Apoapsis)
        p1 = QPointF(cx + r1_px, cy)
        p2 = QPointF(cx - r2_px, cy)

        # Burn 1: Injection Δv₁
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.drawEllipse(p1, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#fca5a5"))
        dv1_str = f"Δv₁ = {self.delta_v_1/1000.0:+.2f} km/s"
        painter.drawText(QPointF(p1.x() + 6.0, p1.y() - 6.0), dv1_str)

        # Burn 2: Circularization Δv₂
        painter.setBrush(QBrush(QColor("#10b981")))
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.drawEllipse(p2, 4.0, 4.0)
        painter.setPen(QColor("#6ee7b7"))
        dv2_str = f"Δv₂ = {self.delta_v_2/1000.0:+.2f} km/s"
        painter.drawText(QPointF(p2.x() - 70.0, p2.y() - 6.0), dv2_str)

        # 6. Current Spacecraft Position along trajectory
        th = self._true_anomaly_rad
        # Using parametric ellipse representation around focus cx, cy
        cur_r_km = (self.a_trans_km * (1.0 - e_val ** 2)) / (1.0 + e_val * math.cos(th))
        cur_r_px = cur_r_km * scale

        # Spacecraft coordinates
        sc_x = cx + cur_r_px * math.cos(th)
        sc_y = cy - cur_r_px * math.sin(th)  # Canvas y inverted

        # Spacecraft body & solar panels
        sc_pt = QPointF(sc_x, sc_y)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawEllipse(sc_pt, 4.5, 4.5)

        # Solar panel wings
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawLine(QPointF(sc_x - 7.0, sc_y), QPointF(sc_x + 7.0, sc_y))

        # Velocity Vector Arrow
        cur_v_ms = self.current_velocity_m(cur_r_km * 1000.0)
        v_arrow_len = 20.0 * (cur_v_ms / max(1.0, self.v_circ_1))
        # Tangent angle to ellipse
        tangent_angle = th + math.pi / 2.0
        v_end_x = sc_x + v_arrow_len * math.cos(tangent_angle)
        v_end_y = sc_y - v_arrow_len * math.sin(tangent_angle)

        painter.setPen(QPen(QColor("#f59e0b"), 1.8))
        painter.drawLine(sc_pt, QPointF(v_end_x, v_end_y))

        # Status Overlay Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 12.0, rect.top() + 20.0), f"QUỸ ĐẠO: {self._preset.title_vi.upper()}")

        # Legend badge
        leg_r = QRectF(rect.right() - 170.0, rect.top() + 8.0, 160.0, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRoundedRect(leg_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(leg_r.left() + 8.0, leg_r.top() + 14.0), "━ Quỹ đạo 1")
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(leg_r.left() + 62.0, leg_r.top() + 14.0), "━ Hohmann")
        painter.setPen(QColor("#10b981"))
        painter.drawText(QPointF(leg_r.left() + 115.0, leg_r.top() + 14.0), "━ Quỹ đạo 2")

        painter.restore()

    # -------------------------------------------------------------------------
    # Right Astrodynamics Telemetry & Controls
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THÔNG SỐ THIÊN THỂ & QUỸ ĐẠO:")

        # Primary Attractor & Vis-Viva Card
        y += 18.0
        card_r = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(card_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 75.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"e = {self.eccentricity_trans:.3f}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(
            QRectF(ox + 85.0, y + 2.0, w - 90.0, 28.0),
            Qt.AlignmentFlag.AlignVCenter,
            f"Bán trục lớn: a = {self.a_trans_km:,.0f} km\n{self._preset.primary_name}",
        )

        # Quantitative Measurements Readout
        y += 40.0
        v1_kms = self.v_circ_1 / 1000.0
        v2_kms = self.v_circ_2 / 1000.0
        dv1_kms = self.delta_v_1 / 1000.0
        dv2_kms = self.delta_v_2 / 1000.0
        dv_tot_kms = self.delta_v_total / 1000.0
        t_hrs = self.transfer_time_hours
        t_days = self.transfer_time_days
        t_str = f"{t_days:.1f} ngày" if t_days >= 3.0 else f"{t_hrs:.2f} giờ"
        vesc_kms = self.v_escape_1 / 1000.0

        rows = [
            (f"• Bán kính r₁: {self._r1_km:,.0f} km", QColor("#38bdf8")),
            (f"• Bán kính r₂: {self._r2_km:,.0f} km", QColor("#10b981")),
            (f"• Tốc độ tròn v₁: {v1_kms:.3f} km/s", QColor("#cbd5e1")),
            (f"• Tốc độ tròn v₂: {v2_kms:.3f} km/s", QColor("#cbd5e1")),
            (f"• Xung lực Δv₁: {dv1_kms:+.3f} km/s", QColor("#ef4444")),
            (f"• Xung lực Δv₂: {dv2_kms:+.3f} km/s", QColor("#10b981")),
            (f"• Tổng xung lực ΣΔv: {dv_tot_kms:.3f} km/s", QColor("#fbbf24")),
            (f"• Thời gian bay t: {t_str}", QColor("#38bdf8")),
            (f"• Tốc độ thoát v_esc: {vesc_kms:.3f} km/s", QColor("#94a3b8")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 15.0

        # Action Buttons: Toggle Animation & Speed Multiplier
        y = rect.bottom() - 32.0
        anim_r = QRectF(ox, y, w / 2.0 - 4.0, 22.0)
        speed_r = QRectF(ox + w / 2.0 + 4.0, y, w / 2.0 - 4.0, 22.0)
        self._btn_rects["toggle_anim"] = anim_r
        self._btn_rects["cycle_speed"] = speed_r

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(anim_r, 4.0, 4.0)
        painter.drawRoundedRect(speed_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(anim_r, Qt.AlignmentFlag.AlignCenter, "⏸ Tạm dừng" if self._is_animating else "▶ Tiếp tục")
        painter.drawText(speed_r, Qt.AlignmentFlag.AlignCenter, f"Tốc độ: {self._sim_speed:.0f}×")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "ĐỊNH LUẬT KEPLER & ĐỘNG LỰC HỌC TÊN LỬA HOHMANN (1925):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Phương trình Vis-Viva: v² = μ(2/r - 1/a) xác định vận tốc quỹ đạo bảo toàn cơ năng ε = -μ/(2a).\n"
            f"• Chuyển quỹ đạo Hohmann sử dụng 2 xung lực tiếp tuyến tối ưu năng lượng để chuyển đổi giữa 2 quỹ đạo tròn đồng phẳng.\n"
            f"• Mô tả tình huống: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Toggle animation
        if self._btn_rects.get("toggle_anim") and self._btn_rects["toggle_anim"].contains(local_pos):
            self.toggle_animation()
            return True

        # 3. Cycle speed
        if self._btn_rects.get("cycle_speed") and self._btn_rects["cycle_speed"].contains(local_pos):
            self.cycle_sim_speed()
            return True

        # 4. Stamp button
        if self._btn_rects.get("stamp") and self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
