"""
AirTeach AI - Interactive Physics Optics Snell's Law & Refraction Simulator
===========================================================================
Provides an interactive physics optics simulation demonstrating:
  - Light refraction across two transparent media (Snell's Law: n1 * sin(i) = n2 * sin(r)).
  - Law of reflection (i' = i).
  - Critical angle computation (i_cr = arcsin(n2 / n1) when n1 > n2).
  - Total Internal Reflection (TIR / Hiện tượng Phản xạ Toàn phần) when i > i_cr.
  - Interactive incident ray source handle S and incidence point I.
  - Real-time refractive index selection, media swapping, and angle readout HUD.
  - Stamping permanent vector optics diagrams onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsSceneMouseEvent,
    QGraphicsTextItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Standard Optical Media Presets
OPTICAL_MEDIA: list[tuple[str, float, QColor]] = [
    ("Không khí (Air)", 1.000, QColor(240, 249, 255, 30)),
    ("Nước (Water)", 1.333, QColor(56, 189, 248, 45)),
    ("Thủy tinh (Crown Glass)", 1.520, QColor(99, 102, 241, 55)),
    ("Dầu hỏa (Kerosene)", 1.480, QColor(234, 179, 8, 45)),
    ("Kim cương (Diamond)", 2.417, QColor(168, 85, 247, 65)),
]


class InteractiveOpticsRefractionItem(QGraphicsObject):
    """
    Interactive physics optics item simulating Snell's Law, reflection,
    and total internal reflection between two transparent optical media.
    """

    angles_changed = Signal(float, float, float)  # (i_deg, r_deg, i_prime_deg)
    tir_changed = Signal(bool)                    # (is_total_internal_reflection)
    media_changed = Signal(float, float)          # (n1, n2)

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 640.0,
        height: float = 440.0,
        n1: float = 1.000,
        n2: float = 1.333,
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

        # Refractive indices and media selection
        self._idx_m1 = 0
        self._idx_m2 = 1
        self._n1 = float(n1)
        self._n2 = float(n2)
        self._m1_name = "Không khí (Air)"
        self._m2_name = "Nước (Water)"

        # Geometry within item coordinates:
        # Interface boundary line is horizontal at y_interface
        self._y_interface = self._height * 0.50
        # Incidence point I on interface
        self._point_i = QPointF(self._width * 0.45, self._y_interface)
        # Source handle S in Medium 1 (y < y_interface)
        self._point_s = QPointF(self._width * 0.22, self._y_interface - 140.0)

        # Dragging states
        self._active_handle: Optional[str] = None  # "S", "I", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Button bounding rects (local coords)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_swap = QRectF(self._width - 250.0, 10.0, 94.0, 26.0)
        self._rect_btn_m1 = QRectF(20.0, 12.0, 170.0, 26.0)
        self._rect_btn_m2 = QRectF(20.0, self._height - 38.0, 170.0, 26.0)

        # Match initial presets if exact matches exist
        for idx, (name, n_val, _) in enumerate(OPTICAL_MEDIA):
            if abs(n_val - self._n1) < 1e-3:
                self._idx_m1 = idx
                self._m1_name = name
            if abs(n_val - self._n2) < 1e-3:
                self._idx_m2 = idx
                self._m2_name = name

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin for glow and handles."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Physics & Optical Calculations ──────────────────────────────────────

    def incidence_point(self) -> QPointF:
        """Point of incidence I on the interface."""
        return self._point_i

    def source_point(self) -> QPointF:
        """Source handle point S in Medium 1."""
        return self._point_s

    def normal_endpoints(self) -> tuple[QPointF, QPointF]:
        """Returns (N_top, N_bottom) endpoints of the normal line NN'."""
        ix = self._point_i.x()
        iy = self._point_i.y()
        n_top = QPointF(ix, max(45.0, iy - 160.0))
        n_bot = QPointF(ix, min(self._height - 45.0, iy + 160.0))
        return n_top, n_bot

    def get_incident_angle(self) -> float:
        """
        Calculates angle of incidence i in degrees [0°, 89.9°]
        measured between incident ray SI and normal NN'.
        """
        dx = self._point_i.x() - self._point_s.x()
        dy = self._point_i.y() - self._point_s.y()  # dy > 0 since S is above interface
        if dy <= 1e-4:
            return 89.9
        rad = math.atan2(abs(dx), dy)
        deg = math.degrees(rad)
        return max(0.0, min(89.9, deg))

    def get_reflection_angle(self) -> float:
        """By law of reflection, angle of reflection i' = i."""
        return self.get_incident_angle()

    def get_critical_angle(self) -> Optional[float]:
        """
        Calculates critical angle i_cr in degrees if n1 > n2:
          sin(i_cr) = n2 / n1.
        Returns None if n1 <= n2.
        """
        if self._n1 > self._n2 and self._n1 > 1e-4:
            ratio = self._n2 / self._n1
            if ratio <= 1.0:
                return math.degrees(math.asin(ratio))
        return None

    def is_total_internal_reflection(self) -> bool:
        """Returns True if light undergoes Total Internal Reflection (TIR)."""
        i_crit = self.get_critical_angle()
        if i_crit is not None:
            return self.get_incident_angle() >= (i_crit - 1e-4)
        return False

    def get_refraction_angle(self) -> Optional[float]:
        """
        Calculates angle of refraction r in degrees using Snell's Law:
          sin(r) = (n1 / n2) * sin(i).
        Returns None if Total Internal Reflection occurs.
        """
        if self.is_total_internal_reflection():
            return None

        i_rad = math.radians(self.get_incident_angle())
        sin_r = (self._n1 / self._n2) * math.sin(i_rad)
        sin_r = max(-1.0, min(1.0, sin_r))
        return math.degrees(math.asin(sin_r))

    def reflected_ray_endpoint(self, length: float = 180.0) -> QPointF:
        """Endpoint of reflected ray IR in Medium 1."""
        i_deg = self.get_incident_angle()
        i_rad = math.radians(i_deg)
        # Sign of light propagation: if S is left of I, reflected ray goes right (+1)
        dir_x = 1.0 if (self._point_s.x() <= self._point_i.x()) else -1.0
        rx = self._point_i.x() + dir_x * length * math.sin(i_rad)
        ry = self._point_i.y() - length * math.cos(i_rad)
        return QPointF(rx, ry)

    def refracted_ray_endpoint(self, length: float = 180.0) -> Optional[QPointF]:
        """Endpoint of refracted ray IK in Medium 2 (None if TIR)."""
        r_deg = self.get_refraction_angle()
        if r_deg is None:
            return None
        r_rad = math.radians(r_deg)
        dir_x = 1.0 if (self._point_s.x() <= self._point_i.x()) else -1.0
        kx = self._point_i.x() + dir_x * length * math.sin(r_rad)
        ky = self._point_i.y() + length * math.cos(r_rad)
        return QPointF(kx, ky)

    def speed_of_light_in_media(self) -> tuple[float, float]:
        """Calculates light speed v = c / n in units of 10^8 m/s (c = 3.00)."""
        c = 3.00
        v1 = c / self._n1 if self._n1 > 0 else 0.0
        v2 = c / self._n2 if self._n2 > 0 else 0.0
        return v1, v2

    # ─── Media & Parameter Mutators ──────────────────────────────────────────

    def set_medium_1(self, name: str, n: float) -> None:
        """Sets Medium 1 parameters."""
        self._m1_name = name
        self._n1 = max(1.000, float(n))
        self._notify_changes()
        self.update()

    def set_medium_2(self, name: str, n: float) -> None:
        """Sets Medium 2 parameters."""
        self._m2_name = name
        self._n2 = max(1.000, float(n))
        self._notify_changes()
        self.update()

    def swap_media(self) -> None:
        """Swaps Medium 1 and Medium 2 (n1 <-> n2) to demonstrate TIR immediately."""
        self._n1, self._n2 = self._n2, self._n1
        self._m1_name, self._m2_name = self._m2_name, self._m1_name
        self._idx_m1, self._idx_m2 = self._idx_m2, self._idx_m1
        self._notify_changes()
        self.update()

    def cycle_medium_1(self) -> None:
        """Cycles Medium 1 through standard optical media presets."""
        self._idx_m1 = (self._idx_m1 + 1) % len(OPTICAL_MEDIA)
        name, n_val, _ = OPTICAL_MEDIA[self._idx_m1]
        self.set_medium_1(name, n_val)

    def cycle_medium_2(self) -> None:
        """Cycles Medium 2 through standard optical media presets."""
        self._idx_m2 = (self._idx_m2 + 1) % len(OPTICAL_MEDIA)
        name, n_val, _ = OPTICAL_MEDIA[self._idx_m2]
        self.set_medium_2(name, n_val)

    def set_incident_angle(self, deg: float) -> None:
        """Sets the angle of incidence i by repositioning source handle S."""
        deg_clamped = max(0.0, min(85.0, float(deg)))
        rad = math.radians(deg_clamped)
        dist = 160.0
        # Preserve side of incidence
        sign_x = -1.0 if (self._point_s.x() <= self._point_i.x()) else 1.0
        sx = self._point_i.x() + sign_x * dist * math.sin(rad)
        sy = self._point_i.y() - dist * math.cos(rad)
        self._point_s = QPointF(sx, sy)
        self._notify_changes()
        self.update()

    def _notify_changes(self) -> None:
        """Emits update signals."""
        i_deg = self.get_incident_angle()
        r_deg = self.get_refraction_angle()
        self.angles_changed.emit(i_deg, r_deg if r_deg is not None else -1.0, i_deg)
        self.tir_changed.emit(self.is_total_internal_reflection())
        self.media_changed.emit(self._n1, self._n2)

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive optics simulation canvas."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height
        y_int = self._y_interface

        # 1. Container Frosted Glass Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Medium 1 (Top half) & Medium 2 (Bottom half) Tints
        # Medium 1 background tint
        m1_col = OPTICAL_MEDIA[self._idx_m1][2] if self._idx_m1 < len(OPTICAL_MEDIA) else QColor(240, 249, 255, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(m1_col))
        painter.drawRoundedRect(QRectF(4.0, 4.0, w - 8.0, y_int - 4.0), 10.0, 10.0)

        # Medium 2 background tint
        m2_col = OPTICAL_MEDIA[self._idx_m2][2] if self._idx_m2 < len(OPTICAL_MEDIA) else QColor(56, 189, 248, 45)
        painter.setBrush(QBrush(m2_col))
        painter.drawRoundedRect(QRectF(4.0, y_int, w - 8.0, h - y_int - 4.0), 10.0, 10.0)

        # 3. Boundary Interface Line
        pen_interface = QPen(QColor("#94a3b8"), 2.2, Qt.PenStyle.SolidLine)
        painter.setPen(pen_interface)
        painter.drawLine(QPointF(10.0, y_int), QPointF(w - 10.0, y_int))

        # Interface label
        f_small = QFont("Segoe UI", 9)
        f_small.setBold(True)
        painter.setFont(f_small)
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QRectF(w - 160.0, y_int - 20.0, 150.0, 18.0), Qt.AlignmentFlag.AlignRight, "Mặt phân cách")

        # 4. Normal Line NN' (Pháp tuyến)
        n_top, n_bot = self.normal_endpoints()
        pen_normal = QPen(QColor("#64748b"), 1.8, Qt.PenStyle.DashLine)
        painter.setPen(pen_normal)
        painter.drawLine(n_top, n_bot)

        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(QPointF(n_top.x() + 6.0, n_top.y() + 12.0), "N")
        painter.drawText(QPointF(n_bot.x() + 6.0, n_bot.y() - 4.0), "N'")

        # 5. Incident Ray SI (Tia tới)
        i_deg = self.get_incident_angle()
        pen_inc_glow = QPen(QColor(239, 68, 68, 80), 6.0)
        painter.setPen(pen_inc_glow)
        painter.drawLine(self._point_s, self._point_i)

        pen_inc = QPen(QColor("#ef4444"), 2.6, Qt.PenStyle.SolidLine)
        painter.setPen(pen_inc)
        painter.drawLine(self._point_s, self._point_i)

        # Arrowhead on incident ray (midpoint)
        mid_si = QPointF(
            (self._point_s.x() + self._point_i.x()) * 0.5,
            (self._point_s.y() + self._point_i.y()) * 0.5,
        )
        self._draw_arrowhead(painter, self._point_s, self._point_i, mid_si, QColor("#ef4444"))

        # 6. Reflected Ray IR (Tia phản xạ)
        is_tir = self.is_total_internal_reflection()
        pt_r = self.reflected_ray_endpoint()
        refl_color = QColor("#f59e0b") if not is_tir else QColor("#ef4444")
        refl_width = 2.4 if not is_tir else 3.2

        if is_tir:
            # Neon pulse glow for TIR
            pen_tir_glow = QPen(QColor(239, 68, 68, 120), 8.0)
            painter.setPen(pen_tir_glow)
            painter.drawLine(self._point_i, pt_r)

        pen_refl = QPen(refl_color, refl_width, Qt.PenStyle.SolidLine)
        painter.setPen(pen_refl)
        painter.drawLine(self._point_i, pt_r)

        mid_ir = QPointF(
            (self._point_i.x() + pt_r.x()) * 0.5,
            (self._point_i.y() + pt_r.y()) * 0.5,
        )
        self._draw_arrowhead(painter, self._point_i, pt_r, mid_ir, refl_color)

        # 7. Refracted Ray IK (Tia khúc xạ)
        r_deg = self.get_refraction_angle()
        pt_k = self.refracted_ray_endpoint()
        if pt_k is not None and not is_tir:
            pen_refr_glow = QPen(QColor(16, 185, 129, 80), 6.0)
            painter.setPen(pen_refr_glow)
            painter.drawLine(self._point_i, pt_k)

            pen_refr = QPen(QColor("#10b981"), 2.6, Qt.PenStyle.SolidLine)
            painter.setPen(pen_refr)
            painter.drawLine(self._point_i, pt_k)

            mid_ik = QPointF(
                (self._point_i.x() + pt_k.x()) * 0.5,
                (self._point_i.y() + pt_k.y()) * 0.5,
            )
            self._draw_arrowhead(painter, self._point_i, pt_k, mid_ik, QColor("#10b981"))

        # 8. Angle Arcs (i, r, i')
        self._draw_angle_arcs(painter, i_deg, r_deg)

        # 9. Optical Parameter HUD Card
        self._draw_optical_hud(painter, i_deg, r_deg, is_tir)

        # 10. Buttons & Interactive Controls
        self._draw_buttons(painter)

        # 11. Draggable Handles (S and I)
        self._draw_handle(painter, self._point_s, "S (Nguồn)", QColor("#ef4444"), self._active_handle == "S")
        self._draw_handle(painter, self._point_i, "I (Điểm tới)", QColor("#06b6d4"), self._active_handle == "I")

        painter.restore()

    def _draw_arrowhead(
        self,
        painter: QPainter,
        p_start: QPointF,
        p_end: QPointF,
        midpoint: QPointF,
        color: QColor,
    ) -> None:
        """Draws a directional triangle arrowhead at the midpoint along a ray."""
        dx = p_end.x() - p_start.x()
        dy = p_end.y() - p_start.y()
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return

        ux, uy = dx / length, dy / length
        arrow_size = 9.0
        wing_angle = math.radians(28)

        # Points of the arrow triangle
        tip = midpoint
        left_rad = math.atan2(uy, ux) + math.pi - wing_angle
        right_rad = math.atan2(uy, ux) + math.pi + wing_angle
        p_left = QPointF(tip.x() + arrow_size * math.cos(left_rad), tip.y() + arrow_size * math.sin(left_rad))
        p_right = QPointF(tip.x() + arrow_size * math.cos(right_rad), tip.y() + arrow_size * math.sin(right_rad))

        arr_path = QPainterPath()
        arr_path.moveTo(tip)
        arr_path.lineTo(p_left)
        arr_path.lineTo(p_right)
        arr_path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(arr_path)

    def _draw_angle_arcs(self, painter: QPainter, i_deg: float, r_deg: Optional[float]) -> None:
        """Draws circular arcs denoting angles i, i', and r."""
        ix = self._point_i.x()
        iy = self._point_i.y()
        arc_r = 38.0

        dir_s = -1.0 if (self._point_s.x() <= ix) else 1.0

        # Angle of incidence arc (i)
        # Normal is at 90° (upwards in Qt: angle from 3 o'clock is 90° or 270°)
        # Qt drawArc: 0° is 3 o'clock, positive angles are counter-clockwise.
        # Up is 90°, Left is 180°, Down is 270°, Right is 0°.
        if dir_s < 0:
            # S is to the left: arc from 90° towards 180° (counter-clockwise)
            start_ang = 90.0
            span_ang = i_deg
        else:
            # S is to the right: arc from 90° towards 0° (clockwise, negative span)
            start_ang = 90.0
            span_ang = -i_deg

        pen_arc_i = QPen(QColor("#f87171"), 1.8)
        painter.setPen(pen_arc_i)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            QRectF(ix - arc_r, iy - arc_r, arc_r * 2.0, arc_r * 2.0),
            int(start_ang * 16),
            int(span_ang * 16),
        )

        # Label i
        f_arc = QFont("Segoe UI", 9)
        f_arc.setBold(True)
        painter.setFont(f_arc)
        label_rad = math.radians(90.0 + (span_ang * 0.5))
        lx = ix + (arc_r + 14.0) * math.cos(label_rad)
        ly = iy - (arc_r + 14.0) * math.sin(label_rad)
        painter.setPen(QPen(QColor("#fca5a5")))
        painter.drawText(QPointF(lx - 8.0, ly + 4.0), f"i={i_deg:.1f}°")

        # Refraction angle arc (r)
        if r_deg is not None:
            arc_rr = 34.0
            pen_arc_r = QPen(QColor("#34d399"), 1.8)
            painter.setPen(pen_arc_r)
            if dir_s < 0:
                # Transmitted to right of normal: normal downwards is 270°
                # clockwise towards 0° / 360°
                start_r = 270.0
                span_r = -r_deg
            else:
                # Transmitted to left of normal: towards 180°
                start_r = 270.0
                span_r = r_deg

            painter.drawArc(
                QRectF(ix - arc_rr, iy - arc_rr, arc_rr * 2.0, arc_rr * 2.0),
                int(start_r * 16),
                int(span_r * 16),
            )
            lbl_r_rad = math.radians(270.0 + (span_r * 0.5))
            rx = ix + (arc_rr + 14.0) * math.cos(lbl_r_rad)
            ry = iy - (arc_rr + 14.0) * math.sin(lbl_r_rad)
            painter.setPen(QPen(QColor("#6ee7b7")))
            painter.drawText(QPointF(rx - 8.0, ry + 4.0), f"r={r_deg:.1f}°")

    def _draw_optical_hud(
        self,
        painter: QPainter,
        i_deg: float,
        r_deg: Optional[float],
        is_tir: bool,
    ) -> None:
        """Renders the frosted glass HUD badge with Snell's law equation & values."""
        hud_w = 210.0
        hud_h = 130.0
        hud_x = self._width - hud_w - 18.0
        hud_y = self._height - hud_h - 18.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 8.0, hud_w - 20.0, 18.0), "Định luật Snell (Khúc xạ)")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        line1 = f"n₁ sin(i) = n₂ sin(r)"
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 28.0, hud_w - 20.0, 16.0), line1)

        i_crit = self.get_critical_angle()
        crit_str = f"{i_crit:.1f}°" if i_crit is not None else "Không có"
        line2 = f"Góc tới hạn igh: {crit_str}"
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 46.0, hud_w - 20.0, 16.0), line2)

        v1, v2 = self.speed_of_light_in_media()
        line3 = f"v₁ = {v1:.2f}c | v₂ = {v2:.2f}c"
        painter.drawText(QRectF(hud_x + 10.0, hud_y + 64.0, hud_w - 20.0, 16.0), line3)

        # Status badge
        badge_rect = QRectF(hud_x + 10.0, hud_y + 88.0, hud_w - 20.0, 28.0)
        if is_tir:
            painter.setPen(QPen(QColor("#ef4444"), 1.4))
            painter.setBrush(QBrush(QColor(239, 68, 68, 50)))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QPen(QColor("#fca5a5")))
            f_badge = QFont("Segoe UI", 8)
            f_badge.setBold(True)
            painter.setFont(f_badge)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "⚡ PHẢN XẠ TOÀN PHẦN")
        else:
            painter.setPen(QPen(QColor("#10b981"), 1.4))
            painter.setBrush(QBrush(QColor(16, 185, 129, 40)))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QPen(QColor("#6ee7b7")))
            f_badge = QFont("Segoe UI", 8)
            f_badge.setBold(True)
            painter.setFont(f_badge)
            r_str = f"r = {r_deg:.1f}°" if r_deg is not None else ""
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"Khúc xạ bình thường ({r_str})")

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders interactive control buttons."""
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

        # Swap button [🔄 Đổi n₁ ⇄ n₂]
        painter.setPen(QPen(QColor("#818cf8"), 1.2))
        painter.setBrush(QBrush(QColor(99, 102, 241, 40)))
        painter.drawRoundedRect(self._rect_btn_swap, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0e7ff")))
        painter.drawText(self._rect_btn_swap, Qt.AlignmentFlag.AlignCenter, "🔄 Đổi n₁⇄n₂")

        # Medium 1 selector button
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
        painter.drawRoundedRect(self._rect_btn_m1, 6.0, 6.0)
        painter.setPen(QPen(QColor("#bae6fd")))
        painter.drawText(self._rect_btn_m1, Qt.AlignmentFlag.AlignCenter, f"n₁: {self._m1_name} ({self._n1:.2f})")

        # Medium 2 selector button
        painter.setPen(QPen(QColor("#34d399"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
        painter.drawRoundedRect(self._rect_btn_m2, 6.0, 6.0)
        painter.setPen(QPen(QColor("#a7f3d0")))
        painter.drawText(self._rect_btn_m2, Qt.AlignmentFlag.AlignCenter, f"n₂: {self._m2_name} ({self._n2:.2f})")

    def _draw_handle(
        self,
        painter: QPainter,
        pos: QPointF,
        label: str,
        color: QColor,
        is_active: bool,
    ) -> None:
        """Renders a draggable circular handle with label."""
        radius = 8.5 if not is_active else 11.0
        glow_radius = radius + 4.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 70)))
        painter.drawEllipse(pos, glow_radius, glow_radius)

        painter.setPen(QPen(QColor("#ffffff"), 1.8))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(pos, radius, radius)

        # Label tag
        f_lbl = QFont("Segoe UI", 8)
        f_lbl.setBold(True)
        painter.setFont(f_lbl)
        painter.setPen(QPen(color))
        painter.drawText(QPointF(pos.x() + 12.0, pos.y() + 4.0), label)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles handle dragging or button clicks."""
        pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_swap.contains(pos):
            self.swap_media()
            event.accept()
            return

        if self._rect_btn_m1.contains(pos):
            self.cycle_medium_1()
            event.accept()
            return

        if self._rect_btn_m2.contains(pos):
            self.cycle_medium_2()
            event.accept()
            return

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

        # Check handle S
        if (pos - self._point_s).manhattanLength() < 24.0:
            self._active_handle = "S"
            event.accept()
            return

        # Check handle I
        if (pos - self._point_i).manhattanLength() < 24.0:
            self._active_handle = "I"
            event.accept()
            return

        # Background drag
        self._active_handle = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates handle positions during dragging."""
        pos = event.pos()

        if self._active_handle == "S":
            # Clamped to Medium 1 (y < y_interface - 10)
            sx = max(20.0, min(self._width - 20.0, pos.x()))
            sy = max(40.0, min(self._y_interface - 15.0, pos.y()))
            self._point_s = QPointF(sx, sy)
            self._notify_changes()
            self.update()
            event.accept()
            return

        if self._active_handle == "I":
            # Clamped along horizontal interface
            ix = max(40.0, min(self._width - 40.0, pos.x()))
            self._point_i = QPointF(ix, self._y_interface)
            self._notify_changes()
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases handle drag state."""
        self._active_handle = None
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
        """Touchless air pinch drag of handle S, I, or entire item."""
        pos = self.mapFromScene(scene_pos)

        if self._active_handle is None:
            if (pos - self._point_s).manhattanLength() < 30.0:
                self._active_handle = "S"
            elif (pos - self._point_i).manhattanLength() < 30.0:
                self._active_handle = "I"
            else:
                self._active_handle = "BODY"
                self._drag_start_item_pos = pos

        if self._active_handle == "S":
            sx = max(20.0, min(self._width - 20.0, pos.x()))
            sy = max(40.0, min(self._y_interface - 15.0, pos.y()))
            self._point_s = QPointF(sx, sy)
            self._notify_changes()
            self.update()
            return True

        if self._active_handle == "I":
            ix = max(40.0, min(self._width - 40.0, pos.x()))
            self._point_i = QPointF(ix, self._y_interface)
            self._notify_changes()
            self.update()
            return True

        if self._active_handle == "BODY":
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

        if self._rect_btn_swap.contains(pos):
            self.swap_media()
            return True

        if self._rect_btn_m1.contains(pos):
            self.cycle_medium_1()
            return True

        if self._rect_btn_m2.contains(pos):
            self.cycle_medium_2()
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
        """Removes this interactive item from its scene."""
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps crisp vector optics lines onto LessonBoard:
          - Interface boundary line
          - Normal line NN'
          - Incident ray with arrowhead
          - Reflected ray with arrowhead
          - Refracted ray with arrowhead (if not TIR)
          - Numerical angle and media information text labels
        """
        from src.ui.stem_tools import PhysicsTools

        stamped: list[QGraphicsItem] = []
        origin = self.pos()

        # Scene coordinates of key points
        pt_i_scene = origin + self._point_i
        pt_s_scene = origin + self._point_s
        pt_r_scene = origin + self.reflected_ray_endpoint()
        pt_k_scene = origin + self.refracted_ray_endpoint() if not self.is_total_internal_reflection() else None

        n_top_local, n_bot_local = self.normal_endpoints()
        n_top_scene = origin + n_top_local
        n_bot_scene = origin + n_bot_local

        # 1. Interface Boundary Line
        w_half = self._width * 0.45
        line_int = QGraphicsLineItem(
            pt_i_scene.x() - w_half,
            pt_i_scene.y(),
            pt_i_scene.x() + w_half,
            pt_i_scene.y(),
        )
        line_int.setPen(QPen(QColor("#64748b"), 2.2, Qt.PenStyle.SolidLine))
        line_int.setZValue(10.0)
        board.addItem(line_int)
        stamped.append(line_int)

        # 2. Normal Line NN'
        line_norm = QGraphicsLineItem(
            n_top_scene.x(),
            n_top_scene.y(),
            n_bot_scene.x(),
            n_bot_scene.y(),
        )
        line_norm.setPen(QPen(QColor("#94a3b8"), 1.8, Qt.PenStyle.DashLine))
        line_norm.setZValue(10.5)
        board.addItem(line_norm)
        stamped.append(line_norm)

        # 3. Incident Ray (SI)
        inc_items = PhysicsTools.create_force_vector(
            start=(pt_s_scene.x(), pt_s_scene.y()),
            end=(pt_i_scene.x(), pt_i_scene.y()),
            label="Tia tới (SI)",
            color=QColor("#ef4444"),
            width=2.8,
        )
        for itm in inc_items:
            board.addItem(itm)
            stamped.append(itm)

        # 4. Reflected Ray (IR)
        is_tir = self.is_total_internal_reflection()
        refl_items = PhysicsTools.create_force_vector(
            start=(pt_i_scene.x(), pt_i_scene.y()),
            end=(pt_r_scene.x(), pt_r_scene.y()),
            label="Tia phản xạ" if not is_tir else "Phản xạ toàn phần",
            color=QColor("#f59e0b") if not is_tir else QColor("#ef4444"),
            width=2.8 if not is_tir else 3.5,
        )
        for itm in refl_items:
            board.addItem(itm)
            stamped.append(itm)

        # 5. Refracted Ray (IK)
        if pt_k_scene is not None and not is_tir:
            refr_items = PhysicsTools.create_force_vector(
                start=(pt_i_scene.x(), pt_i_scene.y()),
                end=(pt_k_scene.x(), pt_k_scene.y()),
                label="Tia khúc xạ (IK)",
                color=QColor("#10b981"),
                width=2.8,
            )
            for itm in refr_items:
                board.addItem(itm)
                stamped.append(itm)

        # 6. Snell Law & Angle Information Label
        i_deg = self.get_incident_angle()
        r_deg = self.get_refraction_angle()
        lbl_text = (
            f"Định luật Khúc xạ Ánh sáng:\n"
            f"Môi trường 1: {self._m1_name} (n₁ = {self._n1:.3f})\n"
            f"Môi trường 2: {self._m2_name} (n₂ = {self._n2:.3f})\n"
            f"Góc tới: i = {i_deg:.1f}° | Góc phản xạ: i' = {i_deg:.1f}°\n"
        )
        if is_tir:
            crit = self.get_critical_angle()
            crit_s = f"{crit:.1f}°" if crit else ""
            lbl_text += f"⚡ HIỆN TƯỢNG PHẢN XẠ TOÀN PHẦN (i ≥ igh = {crit_s})"
        else:
            lbl_text += f"Góc khúc xạ: r = {r_deg:.1f}° (n₁ sin i = n₂ sin r)"

        txt_item = QGraphicsTextItem(lbl_text)
        txt_font = QFont("Segoe UI", 10)
        txt_font.setBold(True)
        txt_item.setFont(txt_font)
        txt_item.setDefaultTextColor(QColor("#f8fafc"))
        txt_item.setPos(pt_i_scene.x() + 40.0, pt_i_scene.y() + 20.0)
        txt_item.setZValue(12.0)
        board.addItem(txt_item)
        stamped.append(txt_item)

        return stamped
