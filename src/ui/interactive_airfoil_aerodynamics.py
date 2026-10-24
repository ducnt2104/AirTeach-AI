"""
AirTeach AI — Interactive Wind Tunnel Aerodynamics & NACA Airfoil Lift / Drag Simulator
Authentic NACA 4-digit airfoil geometry generation, thin airfoil theory, stall hysteresis,
Navier-Stokes/Bernoulli streamlines, lift/drag polar calculation, and touchless air gesture integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Callable

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
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class AirfoilPreset:
    code: str
    name_vi: str
    naca_digits: str
    max_camber_m: float  # Fraction of chord c (e.g. 0.02 for 2%)
    camber_loc_p: float  # Fraction of chord c (e.g. 0.4 for 40%)
    thickness_t: float   # Fraction of chord c (e.g. 0.12 for 12%)
    alpha_0_deg: float   # Zero-lift angle of attack in degrees
    alpha_stall_deg: float  # Positive stall angle of attack
    cd0: float           # Parasitic / skin friction drag coefficient
    aspect_ratio: float  # Wing aspect ratio AR
    description: str


PRESETS: dict[str, AirfoilPreset] = {
    "NACA_2412_GENERAL_AVIATION": AirfoilPreset(
        code="NACA_2412_GENERAL_AVIATION",
        name_vi="Cánh phổ thông (NACA 2412 — Cessna 172)",
        naca_digits="2412",
        max_camber_m=0.02,
        camber_loc_p=0.40,
        thickness_t=0.12,
        alpha_0_deg=-2.1,
        alpha_stall_deg=15.0,
        cd0=0.008,
        aspect_ratio=7.3,
        description="Biên dạng cánh máy bay dân dụng phổ thông, tỉ số L/D cao ở tốc độ bay hành trình.",
    ),
    "NACA_0012_SYMMETRIC": AirfoilPreset(
        code="NACA_0012_SYMMETRIC",
        name_vi="Cánh đối xứng (NACA 0012 — Nhào lộn & Trực thăng)",
        naca_digits="0012",
        max_camber_m=0.00,
        camber_loc_p=0.00,
        thickness_t=0.12,
        alpha_0_deg=0.0,
        alpha_stall_deg=14.0,
        cd0=0.007,
        aspect_ratio=6.0,
        description="Không có độ võng, lực nâng bằng 0 khi góc đón bằng 0; lý tưởng cho cánh đuôi và nhào lộn.",
    ),
    "NACA_4415_HIGH_LIFT": AirfoilPreset(
        code="NACA_4415_HIGH_LIFT",
        name_vi="Lực nâng lớn (NACA 4415 — Vận tải & Tàu lượn)",
        naca_digits="4415",
        max_camber_m=0.04,
        camber_loc_p=0.40,
        thickness_t=0.15,
        alpha_0_deg=-4.2,
        alpha_stall_deg=16.5,
        cd0=0.010,
        aspect_ratio=8.5,
        description="Độ võng 4% và bề dày 15% mang lại hệ số lực nâng cực đại C_L max vượt trội ở vận tốc thấp.",
    ),
    "SUPERCRITICAL_TRANSONIC": AirfoilPreset(
        code="SUPERCRITICAL_TRANSONIC",
        name_vi="Cận âm siêu tới hạn (Supercritical Airfoil)",
        naca_digits="SC(2)-0710",
        max_camber_m=0.015,
        camber_loc_p=0.60,
        thickness_t=0.10,
        alpha_0_deg=-1.5,
        alpha_stall_deg=13.0,
        cd0=0.009,
        aspect_ratio=9.2,
        description="Lưng cánh phẳng làm chậm quá trình hình thành sóng xung kích ở dải tốc độ cận âm.",
    ),
    "SYMMETRIC_THIN_AEROFOIL": AirfoilPreset(
        code="SYMMETRIC_THIN_AEROFOIL",
        name_vi="Cánh mỏng tốc độ cao (NACA 0006)",
        naca_digits="0006",
        max_camber_m=0.00,
        camber_loc_p=0.00,
        thickness_t=0.06,
        alpha_0_deg=0.0,
        alpha_stall_deg=11.5,
        cd0=0.005,
        aspect_ratio=5.0,
        description="Cánh siêu mỏng 6% giảm tối đa lực cản sóng và ma sát bề mặt ở vận tốc cao.",
    ),
}


class InteractiveAirfoilAerodynamicsItem(QGraphicsObject):
    """
    Interactive Wind Tunnel Aerodynamics & NACA Airfoil Lift/Drag Simulator.
    Supports touchless air pinch for Angle of Attack (AoA), wind speed modulation,
    smoke stream particles, aerodynamic stall visualization, and LessonBoard vector stamping.
    """

    WIDTH = 640.0
    HEIGHT = 380.0

    stamp_requested = Signal()
    aoa_changed = Signal(float)  # alpha in degrees

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (180.0, 100.0),
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

        # State Variables
        self._active_preset_code: str = "NACA_2412_GENERAL_AVIATION"
        self._preset: AirfoilPreset = PRESETS[self._active_preset_code]

        # Aerodynamic parameters
        self._aoa_deg: float = 4.0        # Angle of attack alpha in degrees [-10.0 .. 25.0]
        self._wind_speed_ms: float = 35.0 # Free-stream velocity v_inf in m/s [10.0 .. 80.0]
        self._air_density: float = 1.225  # kg/m^3 (Sea level ISA)
        self._chord_m: float = 1.0        # Characteristic chord length
        self._wing_area_s: float = 1.5    # Wing reference area in m^2

        # Smoke tracer streamline particles
        self._particles: list[dict[str, float]] = []
        self._init_particles()

        # Animation state
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)  # ~30 FPS
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Precomputes button rects for click and air pinch interaction."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0
        preset_names = [
            ("NACA_2412_GENERAL_AVIATION", "NACA 2412"),
            ("NACA_0012_SYMMETRIC", "NACA 0012"),
            ("NACA_4415_HIGH_LIFT", "NACA 4415"),
            ("SUPERCRITICAL_TRANSONIC", "Supercritical"),
            ("SYMMETRIC_THIN_AEROFOIL", "NACA 0006"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 5.0

        # Controls on top right
        ctrl_w = 40.0
        r_play = QRectF(cur_x + 10.0, btn_y, ctrl_w, btn_h)
        self._btn_rects["toggle_anim"] = r_play
        r_aoa_down = QRectF(r_play.right() + 5.0, btn_y, 28.0, btn_h)
        self._btn_rects["aoa_down"] = r_aoa_down
        r_aoa_up = QRectF(r_aoa_down.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["aoa_up"] = r_aoa_up

        r_v_down = QRectF(r_aoa_up.right() + 6.0, btn_y, 28.0, btn_h)
        self._btn_rects["v_down"] = r_v_down
        r_v_up = QRectF(r_v_down.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["v_up"] = r_v_up

        r_stamp = QRectF(r_v_up.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

    def _init_particles(self) -> None:
        """Initializes streamlines smoke particles."""
        self._particles.clear()
        # 16 streamline rows spanning tunnel height
        for i in range(16):
            y_stream = 90.0 + i * 11.0
            for j in range(4):
                self._particles.append({
                    "x": 20.0 + j * 75.0 + (i % 3) * 15.0,
                    "y": y_stream,
                    "base_y": y_stream,
                    "speed_mult": 1.0,
                })

    def cleanup(self) -> None:
        """Stops animation timer to avoid memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Analytical Aerodynamics & NACA Formula Engine
    # -------------------------------------------------------------------------
    def generate_naca_coordinates(self, num_points: int = 40) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Computes authentic (x, y) coordinates for upper and lower airfoil surfaces
        using the analytical NACA 4-digit formulation over chord x in [0.0, 1.0].
        """
        m = self._preset.max_camber_m
        p = self._preset.camber_loc_p
        t = self._preset.thickness_t

        upper_pts: list[tuple[float, float]] = []
        lower_pts: list[tuple[float, float]] = []

        for i in range(num_points + 1):
            # Cosine spacing for high leading edge resolution
            beta = (math.pi * i) / float(num_points)
            x = 0.5 * (1.0 - math.cos(beta))

            # Half-thickness y_t
            yt = 5.0 * t * (
                0.2969 * math.sqrt(max(0.0, x))
                - 0.1260 * x
                - 0.3516 * (x ** 2)
                + 0.2843 * (x ** 3)
                - 0.1036 * (x ** 4)
            )

            # Camber line y_c and slope dy_c/dx
            if p > 0.0 and m > 0.0:
                if x < p:
                    yc = (m / (p ** 2)) * (2.0 * p * x - x ** 2)
                    dyc_dx = (2.0 * m / (p ** 2)) * (p - x)
                else:
                    yc = (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2)
                    dyc_dx = (2.0 * m / ((1.0 - p) ** 2)) * (p - x)
            else:
                yc = 0.0
                dyc_dx = 0.0

            theta = math.atan(dyc_dx)

            xu = x - yt * math.sin(theta)
            yu = yc + yt * math.cos(theta)

            xl = x + yt * math.sin(theta)
            yl = yc - yt * math.cos(theta)

            upper_pts.append((xu, yu))
            lower_pts.append((xl, yl))

        return upper_pts, lower_pts

    @property
    def is_stalled(self) -> bool:
        """Determines if the airfoil is currently in aerodynamic stall."""
        return self._aoa_deg > self._preset.alpha_stall_deg or self._aoa_deg < -self._preset.alpha_stall_deg

    @property
    def lift_coefficient(self) -> float:
        """
        Computes lift coefficient C_L with linear slope 2*pi and stall drop-off:
        C_L = 2*pi*(alpha - alpha_0) * (1 / (1 + exp(k * (alpha - alpha_stall))))
        """
        eff_aoa_rad = math.radians(self._aoa_deg - self._preset.alpha_0_deg)
        stall_rad = math.radians(self._preset.alpha_stall_deg)

        cl_linear = 2.0 * math.pi * eff_aoa_rad

        # Sigmoidal stall factor
        k_stall = 15.0
        if self._aoa_deg >= 0.0:
            stall_factor = 1.0 / (1.0 + math.exp(k_stall * (math.radians(self._aoa_deg) - stall_rad)))
            cl_stalled = 1.1 * math.sin(2.0 * eff_aoa_rad)
            cl = cl_linear * stall_factor + cl_stalled * (1.0 - stall_factor)
        else:
            stall_factor = 1.0 / (1.0 + math.exp(-k_stall * (math.radians(self._aoa_deg) + stall_rad)))
            cl_stalled = 1.1 * math.sin(2.0 * eff_aoa_rad)
            cl = cl_linear * stall_factor + cl_stalled * (1.0 - stall_factor)

        return float(cl)

    @property
    def drag_coefficient(self) -> float:
        """
        Computes total drag coefficient C_D:
        C_D = C_D0 + C_L^2 / (pi * e * AR) + C_D,separation
        """
        cl = self.lift_coefficient
        e = 0.85  # Oswald efficiency factor
        ar = self._preset.aspect_ratio
        cd_induced = (cl ** 2) / (math.pi * e * ar)

        # Separation drag surge past stall
        stall_rad = math.radians(self._preset.alpha_stall_deg)
        cur_rad = abs(math.radians(self._aoa_deg))
        if cur_rad > stall_rad:
            cd_sep = 1.6 * math.sin(cur_rad - stall_rad)
        else:
            cd_sep = 0.0

        return float(self._preset.cd0 + cd_induced + cd_sep)

    @property
    def dynamic_pressure_pa(self) -> float:
        """Dynamic pressure q = 0.5 * rho * v^2 in Pascals."""
        return 0.5 * self._air_density * (self._wind_speed_ms ** 2)

    @property
    def lift_force_n(self) -> float:
        """Total aerodynamic lift force L = C_L * q * S in Newtons."""
        return self.lift_coefficient * self.dynamic_pressure_pa * self._wing_area_s

    @property
    def drag_force_n(self) -> float:
        """Total aerodynamic drag force D = C_D * q * S in Newtons."""
        return self.drag_coefficient * self.dynamic_pressure_pa * self._wing_area_s

    @property
    def lift_to_drag_ratio(self) -> float:
        """Aerodynamic efficiency L / D."""
        cd = max(1e-4, self.drag_coefficient)
        return self.lift_coefficient / cd

    # -------------------------------------------------------------------------
    # Properties & Setters
    # -------------------------------------------------------------------------
    @property
    def angle_of_attack_deg(self) -> float:
        return self._aoa_deg

    @property
    def wind_speed_ms(self) -> float:
        return self._wind_speed_ms

    @property
    def active_preset(self) -> AirfoilPreset:
        return self._preset

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined airfoil preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self.update()

    def set_angle_of_attack(self, aoa_deg: float) -> None:
        """Sets Angle of Attack alpha (clamped between -10.0 and 25.0 degrees)."""
        self._aoa_deg = max(-10.0, min(25.0, float(aoa_deg)))
        self.aoa_changed.emit(self._aoa_deg)
        self.update()

    def set_wind_speed(self, v_ms: float) -> None:
        """Sets free-stream wind speed v_inf (clamped between 10.0 and 80.0 m/s)."""
        self._wind_speed_ms = max(10.0, min(80.0, float(v_ms)))
        self.update()

    def toggle_animation(self) -> bool:
        """Toggles wind tunnel airflow particle simulation."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    # -------------------------------------------------------------------------
    # Wind Tunnel Streamlines & Physics Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Updates positions of smoke streak particles flowing through tunnel."""
        if not self._is_animating:
            return

        dt_scale = (self._wind_speed_ms / 35.0) * 3.5

        # Tunnel airfoil position
        chord_px = 160.0
        af_cx = 190.0
        af_cy = 175.0

        for p in self._particles:
            px = p["x"]
            py = p["base_y"]

            # Deflection around airfoil
            dx = px - af_cx
            if abs(dx) < chord_px * 0.8:
                # Flow curvature proportional to AoA
                curvature = math.sin(math.radians(self._aoa_deg)) * 25.0
                dist_factor = 1.0 - (abs(dx) / (chord_px * 0.8))
                if py < af_cy:
                    # Suction upper side accelerates
                    p["speed_mult"] = 1.0 + (0.4 if not self.is_stalled else -0.2)
                    p["y"] = py - curvature * dist_factor
                else:
                    # Compression lower side decelerates
                    p["speed_mult"] = 0.85
                    p["y"] = py - curvature * 0.4 * dist_factor
            else:
                p["speed_mult"] = 1.0
                p["y"] = py

            p["x"] += dt_scale * p["speed_mult"]

            # Wrap around left entrance
            if p["x"] > 340.0:
                p["x"] = 20.0

        self.update()

    # -------------------------------------------------------------------------
    # Touchless Air Gesture Handling & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Airfoil Aerodynamics diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import PhysicsTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = PhysicsTools.create_airfoil_diagram(
            center=center,
            preset_code=self._active_preset_code,
            aoa_deg=self._aoa_deg,
            wind_speed_ms=self._wind_speed_ms,
            cl=self.lift_coefficient,
            cd=self.drag_coefficient,
            is_stalled=self.is_stalled,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. AoA steppers
        r_aoa_down = self._btn_rects.get("aoa_down")
        if r_aoa_down and r_aoa_down.contains(local_pos):
            self.set_angle_of_attack(self._aoa_deg - 1.0)
            return True

        r_aoa_up = self._btn_rects.get("aoa_up")
        if r_aoa_up and r_aoa_up.contains(local_pos):
            self.set_angle_of_attack(self._aoa_deg + 1.0)
            return True

        # 3. Wind speed steppers
        r_v_down = self._btn_rects.get("v_down")
        if r_v_down and r_v_down.contains(local_pos):
            self.set_wind_speed(self._wind_speed_ms - 5.0)
            return True

        r_v_up = self._btn_rects.get("v_up")
        if r_v_up and r_v_up.contains(local_pos):
            self.set_wind_speed(self._wind_speed_ms + 5.0)
            return True

        # 4. Play/Pause toggle
        r_play = self._btn_rects.get("toggle_anim")
        if r_play and r_play.contains(local_pos):
            self.toggle_animation()
            return True

        # 5. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        # 6. Direct Air Pinch on Airfoil Test Chamber to set AoA
        chamber_rect = QRectF(20.0, 75.0, 320.0, 195.0)
        if chamber_rect.contains(local_pos):
            # Calculate angle relative to airfoil center (190, 175)
            dy = -(local_pos.y() - 175.0)
            target_aoa = (dy / 70.0) * 15.0
            self.set_angle_of_attack(target_aoa)
            return True

        return True

    # -------------------------------------------------------------------------
    # Rendering & Painting
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

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, self.WIDTH - 28.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"✈️ KHÍ ĐỘNG HỌC CÁNH MÁY BAY: LỰC NÂNG & LỰC CẢN — {self._preset.name_vi}",
        )

        # 3. Preset Buttons
        btn_y = 44.0
        btn_h = 22.0
        preset_names = [
            ("NACA_2412_GENERAL_AVIATION", "NACA 2412"),
            ("NACA_0012_SYMMETRIC", "NACA 0012"),
            ("NACA_4415_HIGH_LIFT", "NACA 4415"),
            ("SUPERCRITICAL_TRANSONIC", "Supercritical"),
            ("SYMMETRIC_THIN_AEROFOIL", "NACA 0006"),
        ]
        for p_code, p_lbl in preset_names:
            r_btn = self._btn_rects[f"preset_{p_code}"]
            is_active = self._active_preset_code == p_code

            painter.setBrush(QBrush(QColor("#0284c7") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r_btn, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r_btn, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Controls
        r_play = self._btn_rects["toggle_anim"]
        painter.setBrush(QBrush(QColor("#10b981") if self._is_animating else QColor("#334155")))
        painter.setPen(QPen(QColor("#34d399"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "⏸" if self._is_animating else "▶")

        r_ad = self._btn_rects["aoa_down"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_ad, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_ad, Qt.AlignmentFlag.AlignCenter, "α-")

        r_au = self._btn_rects["aoa_up"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_au, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_au, Qt.AlignmentFlag.AlignCenter, "α+")

        r_vd = self._btn_rects["v_down"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_vd, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_vd, Qt.AlignmentFlag.AlignCenter, "v-")

        r_vu = self._btn_rects["v_up"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_vu, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_vu, Qt.AlignmentFlag.AlignCenter, "v+")

        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng")

        # ---------------------------------------------------------------------
        # 4. Wind Tunnel Test Chamber (Left Panel)
        # ---------------------------------------------------------------------
        tunnel_x = 18.0
        tunnel_y = 75.0
        tunnel_w = 325.0
        tunnel_h = 195.0

        tunnel_rect = QRectF(tunnel_x, tunnel_y, tunnel_w, tunnel_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#ef4444") if self.is_stalled else QColor("#334155"), 1.5))
        painter.drawRoundedRect(tunnel_rect, 6.0, 6.0)

        # Tunnel wind inlet label
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(tunnel_x + 8.0, tunnel_y + 16.0), f"💨 Ống khí động học: v∞ = {self._wind_speed_ms:.0f} m/s ({self._wind_speed_ms * 3.6:.0f} km/h)")

        # Stall warning banner if stalled
        if self.is_stalled:
            stall_badge = QRectF(tunnel_x + tunnel_w - 110.0, tunnel_y + 6.0, 102.0, 18.0)
            painter.setBrush(QBrush(QColor("#7f1d1d")))
            painter.setPen(QPen(QColor("#f87171"), 1.0))
            painter.drawRoundedRect(stall_badge, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#fecaca"))
            painter.drawText(stall_badge, Qt.AlignmentFlag.AlignCenter, "⚠️ THẤT TỐC (STALL)")

        # Render smoke streak particles
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self._particles:
            if tunnel_rect.contains(QPointF(p["x"], p["y"])):
                # Color code: cyan for laminar, amber/red for stalled boundary layer
                if self.is_stalled and p["x"] > 190.0 and p["y"] < 175.0:
                    p_col = QColor(248, 113, 113, 160)
                else:
                    p_col = QColor(56, 189, 248, 140)
                painter.setBrush(QBrush(p_col))
                painter.drawEllipse(QPointF(p["x"], p["y"]), 2.2, 2.2)

        # Render NACA Airfoil Body
        chord_px = 150.0
        af_cx = tunnel_x + tunnel_w * 0.52
        af_cy = tunnel_y + tunnel_h * 0.52

        painter.save()
        painter.translate(af_cx, af_cy)
        # Rotate by -alpha (positive AoA tilts nose up)
        painter.rotate(-self._aoa_deg)

        upper_pts, lower_pts = self.generate_naca_coordinates(35)
        airfoil_path = QPainterPath()

        # Origin at leading edge (-chord/2, 0)
        x_le = -chord_px / 2.0

        # Upper surface (LE to TE)
        first = True
        for xu, yu in upper_pts:
            px = x_le + xu * chord_px
            py = -yu * chord_px
            if first:
                airfoil_path.moveTo(px, py)
                first = False
            else:
                airfoil_path.lineTo(px, py)

        # Lower surface (TE back to LE)
        for xl, yl in reversed(lower_pts):
            px = x_le + xl * chord_px
            py = -yl * chord_px
            airfoil_path.lineTo(px, py)

        airfoil_path.closeSubpath()

        # Paint Airfoil with metallic gradient
        painter.setBrush(QBrush(QColor("#334155")))
        painter.setPen(QPen(QColor("#94a3b8"), 1.8))
        painter.drawPath(airfoil_path)

        # Chord line (dashed)
        chord_pen = QPen(QColor("#64748b"), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(chord_pen)
        painter.drawLine(QPointF(x_le, 0.0), QPointF(x_le + chord_px, 0.0))

        painter.restore()

        # Aerodynamic Force Vectors at Aerodynamic Center (quarter-chord)
        ac_x = af_cx - 0.25 * chord_px * math.cos(math.radians(-self._aoa_deg))
        ac_y = af_cy - 0.25 * chord_px * math.sin(math.radians(-self._aoa_deg))

        # Vector scaling
        lift_n = self.lift_force_n
        drag_n = self.drag_force_n

        lift_vec_len = max(-50.0, min(65.0, (lift_n / 400.0) * 45.0))
        drag_vec_len = max(5.0, min(50.0, (drag_n / 200.0) * 35.0))

        # Lift Vector L (perpendicular to freestream = strictly vertical)
        if abs(lift_vec_len) > 2.0:
            painter.setPen(QPen(QColor("#10b981"), 2.2))
            painter.drawLine(QPointF(ac_x, ac_y), QPointF(ac_x, ac_y - lift_vec_len))
            # Arrow head
            arrow_dir = -1.0 if lift_vec_len > 0.0 else 1.0
            tip_y = ac_y - lift_vec_len
            painter.drawLine(QPointF(ac_x, tip_y), QPointF(ac_x - 4.0, tip_y - arrow_dir * 6.0))
            painter.drawLine(QPointF(ac_x, tip_y), QPointF(ac_x + 4.0, tip_y - arrow_dir * 6.0))

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#34d399"))
            painter.drawText(QPointF(ac_x + 6.0, tip_y + 4.0), f"L = {lift_n:.0f} N")

        # Drag Vector D (parallel to freestream = strictly horizontal to the right)
        painter.setPen(QPen(QColor("#ef4444"), 2.2))
        painter.drawLine(QPointF(ac_x, ac_y), QPointF(ac_x + drag_vec_len, ac_y))
        tip_x = ac_x + drag_vec_len
        painter.drawLine(QPointF(tip_x, ac_y), QPointF(tip_x - 6.0, ac_y - 4.0))
        painter.drawLine(QPointF(tip_x, ac_y), QPointF(tip_x - 6.0, ac_y + 4.0))

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f87171"))
        painter.drawText(QPointF(tip_x + 6.0, ac_y + 4.0), f"D = {drag_n:.0f} N")

        # ---------------------------------------------------------------------
        # 5. Diagnostic Polar Curve & Metrics Card (Right Panel)
        # ---------------------------------------------------------------------
        card_x = tunnel_x + tunnel_w + 14.0
        card_y = tunnel_y
        card_w = self.WIDTH - card_x - 14.0
        card_h = tunnel_h

        card_rect = QRectF(card_x, card_y, card_w, card_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        cl = self.lift_coefficient
        cd = self.drag_coefficient
        ld_ratio = self.lift_to_drag_ratio

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(card_x + 10.0, card_y + 18.0), "THÔNG SỐ KHÍ ĐỘNG HỌC")

        metrics_lines = [
            (f"• Góc đón α: {self._aoa_deg:+.1f}° (α₀ = {self._preset.alpha_0_deg:.1f}°)", QColor("#fbbf24"), True),
            (f"• Hệ số lực nâng C_L: {cl:+.3f}", QColor("#10b981"), True),
            (f"• Hệ số lực cản C_D: {cd:.4f}", QColor("#f43f5e"), False),
            (f"• Tỉ số hiệu quả L/D: {ld_ratio:+.1f}", QColor("#38bdf8"), True),
            (f"• Lực nâng L: {lift_n:+.1f} N", QColor("#34d399"), False),
            (f"• Lực cản D: {drag_n:.1f} N", QColor("#f87171"), False),
            (f"• Áp suất động q: {self.dynamic_pressure_pa:.1f} Pa", QColor("#cbd5e1"), False),
            (f"• Trạng thái: {'THẤT TỐC ⚠️' if self.is_stalled else 'Ổn định ✔'}", QColor("#ef4444" if self.is_stalled else "#22c55e"), True),
        ]
        my = card_y + 36.0
        for m_txt, m_col, is_bld in metrics_lines:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_bld else QFont.Weight.Normal))
            painter.setPen(m_col)
            painter.drawText(QPointF(card_x + 12.0, my), m_txt)
            my += 18.0

        # ---------------------------------------------------------------------
        # 6. Theory & Aerodynamic Principles Card (Bottom Panel)
        # ---------------------------------------------------------------------
        bot_y = tunnel_y + tunnel_h + 12.0
        bot_h = self.HEIGHT - bot_y - 12.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "NGUYÊN LÝ VẬT LÝ VÀ SƯ PHẠM (BERNOULLI & NAVIER-STOKES):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        p_desc = (
            f"• Định luật Bernoulli: Vận tốc dòng khí mặt lưng cao hơn mặt bụng (v_upper > v_lower) tạo ra chênh lệch áp suất nâng cánh: ΔP = ½ρ(v_upper² - v_lower²).\n"
            f"• Hiện tượng thất tốc (Stall): Khi góc đón α vượt quá {self._preset.alpha_stall_deg:.0f}°, lớp biên phân tách khỏi mặt lưng gây cuộn xoáy hỗn loạn, làm sụt giảm lực nâng đột ngột.\n"
            f"• Đặc điểm biên dạng: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 48.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            p_desc,
        )
