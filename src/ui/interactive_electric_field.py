"""
AirTeach AI - Interactive Physics Electric Field Lines & Point Charges Coulomb Simulator
========================================================================================
Provides an authentic physics simulation of electrostatics and Coulomb's Law:
  - System of discrete 2D point charges q_i with Coulomb's Law:
      E(r) = (1 / 4*pi*eps_0) * sum( q_i * (r - r_i) / |r - r_i|^3 )
      V(r) = (1 / 4*pi*eps_0) * sum( q_i / |r - r_i| )
  - Numerical 4th-Order Runge-Kutta (RK4) electric field line integration.
  - Authentic field line behavior: lines radiate from positive charges and
    terminate at negative charges or infinity, never intersecting.
  - Interactive draggable charges with real-time field deformation.
  - Movable electric field & potential test probe with vector arrow readout.
  - Real electrostatic presets: Dipole (Lưỡng cực), Like charges (Hai điện tích cùng dấu), Quadrupole (Tứ cực).
  - Touchless air pinch dragging and air tap controls.
  - Vector diagram stamping onto LessonBoard via PhysicsTools.create_electric_field_diagram.
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

# Coulomb constant scaling factor for screen coordinates (pixels and microCoulombs)
K_COULOMB = 65000.0


@dataclass
class PointCharge:
    """Electrostatic point charge."""
    charge_id: str
    pos: QPointF  # Local item coordinates
    q: float      # Charge in microCoulombs (µC)


ELECTROSTATIC_PRESETS: dict[str, dict] = {
    "DIPOLE": {
        "name": "Lưỡng Cực Điện (Dipole)",
        "charges": [
            ("Q1", 220.0, 220.0, 1.0),
            ("Q2", 440.0, 220.0, -1.0),
        ],
    },
    "TWO_POSITIVE": {
        "name": "Cùng Dấu Dương (+ / +)",
        "charges": [
            ("Q1", 220.0, 220.0, 1.0),
            ("Q2", 440.0, 220.0, 1.0),
        ],
    },
    "TWO_NEGATIVE": {
        "name": "Cùng Dấu Âm (− / −)",
        "charges": [
            ("Q1", 220.0, 220.0, -1.0),
            ("Q2", 440.0, 220.0, -1.0),
        ],
    },
    "QUADRUPOLE": {
        "name": "Tứ Cực Điện (Quadrupole)",
        "charges": [
            ("Q1", 230.0, 150.0, 1.0),
            ("Q2", 430.0, 150.0, -1.0),
            ("Q3", 230.0, 290.0, -1.0),
            ("Q4", 430.0, 290.0, 1.0),
        ],
    },
}


class InteractiveElectricFieldItem(QGraphicsObject):
    """
    Interactive Physics Coulomb's Law & Electric Field Line simulator,
    featuring live RK4 numerical streamline integration, draggable charges,
    probe sensor readout, and vector stamping onto LessonBoard.
    """

    charge_moved = Signal(str, float, float)     # (charge_id, x, y)
    probe_updated = Signal(float, float, float)  # (magnitude E, angle_deg, potential V)

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 660.0,
        height: float = 440.0,
        preset: str = "DIPOLE",
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

        self._active_preset = preset if preset in ELECTROSTATIC_PRESETS else "DIPOLE"
        self._charges: list[PointCharge] = []

        # Interactive Field Probe Point P(x, y)
        self._probe_pos = QPointF(330.0, 140.0)
        self._show_probe = True

        # Precomputed RK4 Streamlines
        self._field_lines: list[list[QPointF]] = []

        # Dragging state
        self._active_drag: Optional[str] = None  # "CHARGE_<id>", "PROBE", "BODY"
        self._dragged_charge_id: Optional[str] = None
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_probe = QRectF(self._width - 250.0, 10.0, 95.0, 26.0)
        self._rect_btn_invert = QRectF(self._width - 360.0, 10.0, 105.0, 26.0)

        # Preset selector buttons across bottom
        self._preset_keys = ["DIPOLE", "TWO_POSITIVE", "TWO_NEGATIVE", "QUADRUPOLE"]
        self._rect_btn_presets: list[QRectF] = []
        btn_w = 95.0
        start_x = 18.0
        for i in range(len(self._preset_keys)):
            self._rect_btn_presets.append(QRectF(start_x + i * (btn_w + 6.0), self._height - 36.0, btn_w, 24.0))

        # Load preset and compute initial field lines
        self._load_preset(self._active_preset)

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Electrostatic Physics Calculations ──────────────────────────────────

    def _load_preset(self, preset_key: str) -> None:
        """Loads point charge configuration from preset."""
        data = ELECTROSTATIC_PRESETS.get(preset_key, ELECTROSTATIC_PRESETS["DIPOLE"])
        self._active_preset = preset_key
        self._charges.clear()
        for cid, cx, cy, q in data["charges"]:
            self._charges.append(PointCharge(charge_id=cid, pos=QPointF(cx, cy), q=q))

        # Default probe location between charges
        if len(self._charges) >= 2:
            mid_x = (self._charges[0].pos.x() + self._charges[1].pos.x()) / 2.0
            mid_y = min(self._charges[0].pos.y(), self._charges[1].pos.y()) - 70.0
            self._probe_pos = QPointF(mid_x, max(60.0, mid_y))

        self.recompute_field_lines()

    def set_preset(self, preset_key: str) -> None:
        """Switches to another electrostatic preset."""
        if preset_key in ELECTROSTATIC_PRESETS:
            self._load_preset(preset_key)
            self.update()

    def invert_charges(self) -> None:
        """Inverts polarity (+q <-> -q) of all charges."""
        for c in self._charges:
            c.q = -c.q
        self.recompute_field_lines()
        self.update()

    def toggle_probe(self) -> bool:
        """Toggles electric field probe visibility."""
        self._show_probe = not self._show_probe
        self.update()
        return self._show_probe

    def electric_field_at(self, x: float, y: float) -> tuple[float, float]:
        """
        Calculates electric field vector (Ex, Ey) at point (x, y) using
        superposition of Coulomb's Law:
          E = sum( k_e * q_i * (r - r_i) / |r - r_i|^3 )
        """
        ex = 0.0
        ey = 0.0
        for charge in self._charges:
            dx = x - charge.pos.x()
            dy = y - charge.pos.y()
            r2 = dx * dx + dy * dy
            r = math.sqrt(r2)
            # Softening distance to prevent singularity at charge origin
            r_soft = max(14.0, r)
            r3 = r_soft * r_soft * r_soft
            coeff = (K_COULOMB * charge.q) / r3
            ex += coeff * dx
            ey += coeff * dy

        return ex, ey

    def electric_potential_at(self, x: float, y: float) -> float:
        """
        Calculates scalar electric potential V at point (x, y):
          V = sum( k_e * q_i / |r - r_i| )
        """
        v = 0.0
        for charge in self._charges:
            dx = x - charge.pos.x()
            dy = y - charge.pos.y()
            r = max(14.0, math.hypot(dx, dy))
            v += (K_COULOMB * charge.q) / r
        return v

    def recompute_field_lines(self) -> None:
        """
        Performs 4th-Order Runge-Kutta (RK4) numerical streamline integration
        to construct authentic electric field lines.
        """
        self._field_lines.clear()

        # Check if there are any positive charges
        has_positive = any(c.q > 0 for c in self._charges)
        direction_sign = 1.0 if has_positive else -1.0

        sim_bounds = QRectF(15.0, 42.0, self._width - 30.0, self._height - 90.0)
        step_size = 3.6
        max_steps = 130
        capture_radius = 14.0

        # Emitters: positive charges (or negative charges if all negative)
        emitters = [c for c in self._charges if (c.q > 0 if has_positive else c.q < 0)]
        absorbers = [c for c in self._charges if (c.q < 0 if has_positive else c.q > 0)]

        for charge in emitters:
            num_rays = int(max(8, min(24, round(abs(charge.q) * 14))))
            r_seed = 17.0

            for ray_idx in range(num_rays):
                ang = 2.0 * math.pi * (ray_idx / float(num_rays))
                curr_x = charge.pos.x() + r_seed * math.cos(ang)
                curr_y = charge.pos.y() + r_seed * math.sin(ang)

                line_points: list[QPointF] = [QPointF(curr_x, curr_y)]

                for _ in range(max_steps):
                    # Check boundary exit
                    if not sim_bounds.contains(curr_x, curr_y):
                        break

                    # Check capture by absorbing charges
                    captured = False
                    for abs_charge in absorbers:
                        if math.hypot(curr_x - abs_charge.pos.x(), curr_y - abs_charge.pos.y()) < capture_radius:
                            line_points.append(QPointF(abs_charge.pos.x(), abs_charge.pos.y()))
                            captured = True
                            break
                    if captured:
                        break

                    # RK4 Integration Step
                    # k1
                    ex1, ey1 = self.electric_field_at(curr_x, curr_y)
                    emag1 = math.hypot(ex1, ey1)
                    if emag1 < 1e-4:
                        break
                    u1x, u1y = (ex1 / emag1) * direction_sign, (ey1 / emag1) * direction_sign

                    # k2
                    x2 = curr_x + 0.5 * step_size * u1x
                    y2 = curr_y + 0.5 * step_size * u1y
                    ex2, ey2 = self.electric_field_at(x2, y2)
                    emag2 = math.hypot(ex2, ey2)
                    if emag2 < 1e-4:
                        break
                    u2x, u2y = (ex2 / emag2) * direction_sign, (ey2 / emag2) * direction_sign

                    # k3
                    x3 = curr_x + 0.5 * step_size * u2x
                    y3 = curr_y + 0.5 * step_size * u2y
                    ex3, ey3 = self.electric_field_at(x3, y3)
                    emag3 = math.hypot(ex3, ey3)
                    if emag3 < 1e-4:
                        break
                    u3x, u3y = (ex3 / emag3) * direction_sign, (ey3 / emag3) * direction_sign

                    # k4
                    x4 = curr_x + step_size * u3x
                    y4 = curr_y + step_size * u3y
                    ex4, ey4 = self.electric_field_at(x4, y4)
                    emag4 = math.hypot(ex4, ey4)
                    if emag4 < 1e-4:
                        break
                    u4x, u4y = (ex4 / emag4) * direction_sign, (ey4 / emag4) * direction_sign

                    # Average tangent
                    dx = (step_size / 6.0) * (u1x + 2.0 * u2x + 2.0 * u3x + u4x)
                    dy = (step_size / 6.0) * (u1y + 2.0 * u2y + 2.0 * u3y + u4y)

                    curr_x += dx
                    curr_y += dy
                    line_points.append(QPointF(curr_x, curr_y))

                if len(line_points) >= 3:
                    self._field_lines.append(line_points)

        # Notify probe update
        pex, pey = self.electric_field_at(self._probe_pos.x(), self._probe_pos.y())
        p_mag = math.hypot(pex, pey)
        p_ang = math.degrees(math.atan2(pey, pex))
        p_v = self.electric_potential_at(self._probe_pos.x(), self._probe_pos.y())
        self.probe_updated.emit(p_mag, p_ang, p_v)

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive Electrostatic Coulomb simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height

        # 1. Frosted Glass Container Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Render RK4 Streamlines (Electric Field Lines)
        self._draw_field_lines(painter)

        # 3. Render Point Charges (+ / −)
        self._draw_point_charges(painter)

        # 4. Render Electric Field Probe & Readout
        if self._show_probe:
            self._draw_probe(painter)

        # 5. Electrostatic HUD Badge (Top Right)
        self._draw_coulomb_hud(painter)

        # 6. Control & Preset Buttons
        self._draw_buttons(painter)

        painter.restore()

    def _draw_field_lines(self, painter: QPainter) -> None:
        """Renders continuous electric field lines with directional arrowheads."""
        pen_line = QPen(QColor(56, 189, 248, 180), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_line)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for line_pts in self._field_lines:
            if len(line_pts) < 2:
                continue
            path = QPainterPath()
            path.moveTo(line_pts[0])
            for pt in line_pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

            # Midpoint directional arrowhead
            if len(line_pts) >= 6:
                mid_idx = len(line_pts) // 2
                p_prev = line_pts[mid_idx - 1]
                p_curr = line_pts[mid_idx]
                dx = p_curr.x() - p_prev.x()
                dy = p_curr.y() - p_prev.y()
                length = math.hypot(dx, dy)
                if length > 1e-2:
                    ux, uy = dx / length, dy / length
                    arr_len = 6.0
                    wing_ang = math.radians(26)
                    ang = math.atan2(uy, ux)

                    arr_path = QPainterPath()
                    arr_path.moveTo(p_curr)
                    arr_path.lineTo(
                        p_curr.x() - arr_len * math.cos(ang - wing_ang),
                        p_curr.y() - arr_len * math.sin(ang - wing_ang),
                    )
                    arr_path.lineTo(
                        p_curr.x() - arr_len * math.cos(ang + wing_ang),
                        p_curr.y() - arr_len * math.sin(ang + wing_ang),
                    )
                    arr_path.closeSubpath()

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor("#38bdf8")))
                    painter.drawPath(arr_path)
                    painter.setPen(pen_line)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_point_charges(self, painter: QPainter) -> None:
        """Renders point charges with radial gradient glow and polarity sign."""
        r_charge = 15.0

        for charge in self._charges:
            pos = charge.pos
            is_pos = (charge.q > 0)
            c_color = QColor("#ef4444") if is_pos else QColor("#3b82f6")

            # Radial Glow Halo
            grad_glow = QRadialGradient(pos.x(), pos.y(), r_charge * 2.2)
            grad_glow.setColorAt(0.0, QColor(c_color.red(), c_color.green(), c_color.blue(), 160))
            grad_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad_glow))
            painter.drawEllipse(pos, r_charge * 2.2, r_charge * 2.2)

            # Core Sphere
            grad_core = QRadialGradient(pos.x() - r_charge * 0.3, pos.y() - r_charge * 0.3, r_charge * 1.3)
            grad_core.setColorAt(0.0, QColor("#ffffff"))
            grad_core.setColorAt(0.3, c_color)
            grad_core.setColorAt(1.0, c_color.darker(150))
            painter.setPen(QPen(c_color.darker(160), 1.5))
            painter.setBrush(QBrush(grad_core))
            painter.drawEllipse(pos, r_charge, r_charge)

            # Sign: '+' or '−'
            f_sign = QFont("Segoe UI", 13)
            f_sign.setBold(True)
            painter.setFont(f_sign)
            painter.setPen(QPen(QColor("#ffffff")))
            sign_char = "+" if is_pos else "−"
            painter.drawText(QRectF(pos.x() - r_charge, pos.y() - r_charge, r_charge * 2, r_charge * 2), Qt.AlignmentFlag.AlignCenter, sign_char)

            # Value label below
            f_val = QFont("Segoe UI", 8)
            f_val.setBold(True)
            painter.setFont(f_val)
            painter.setPen(QPen(c_color))
            val_text = f"{charge.q:+.1f} µC"
            painter.drawText(QRectF(pos.x() - 30.0, pos.y() + r_charge + 2.0, 60.0, 14.0), Qt.AlignmentFlag.AlignCenter, val_text)

    def _draw_probe(self, painter: QPainter) -> None:
        """Renders movable test probe point with electric field vector arrow."""
        p = self._probe_pos
        ex, ey = self.electric_field_at(p.x(), p.y())
        emag = math.hypot(ex, ey)
        pot = self.electric_potential_at(p.x(), p.y())

        # Probe reticle dot
        r_probe = 6.0
        painter.setPen(QPen(QColor("#22c55e"), 1.5))
        painter.setBrush(QBrush(QColor(34, 197, 94, 80)))
        painter.drawEllipse(p, r_probe, r_probe)

        # Field Vector Arrow (scaled for visibility)
        v_scale = min(40.0, emag * 0.08)
        if emag > 1e-2 and v_scale > 2.0:
            ux, uy = ex / emag, ey / emag
            p_end = QPointF(p.x() + ux * v_scale, p.y() + uy * v_scale)

            # Vector line
            pen_arrow = QPen(QColor("#22c55e"), 2.2, Qt.PenStyle.SolidLine)
            painter.setPen(pen_arrow)
            painter.drawLine(p, p_end)

            # Arrowhead
            arr_len = 7.0
            wing_ang = math.radians(26)
            ang = math.atan2(uy, ux)
            arr_path = QPainterPath()
            arr_path.moveTo(p_end)
            arr_path.lineTo(
                p_end.x() - arr_len * math.cos(ang - wing_ang),
                p_end.y() - arr_len * math.sin(ang - wing_ang),
            )
            arr_path.lineTo(
                p_end.x() - arr_len * math.cos(ang + wing_ang),
                p_end.y() - arr_len * math.sin(ang + wing_ang),
            )
            arr_path.closeSubpath()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#22c55e")))
            painter.drawPath(arr_path)

        # Probe HUD Badge
        hud_w = 145.0
        hud_h = 44.0
        hud_rect = QRectF(p.x() + 12.0, p.y() - 22.0, hud_w, hud_h)
        painter.setPen(QPen(QColor(34, 197, 94, 140), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 6.0, 6.0)

        f_hud = QFont("Segoe UI", 7)
        f_hud.setBold(True)
        painter.setFont(f_hud)
        painter.setPen(QPen(QColor("#86efac")))
        painter.drawText(QRectF(hud_rect.x() + 6.0, hud_rect.y() + 4.0, hud_w - 12.0, 16.0), f"Đầu dò E⃗: {emag:.1f} V/m")
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(QRectF(hud_rect.x() + 6.0, hud_rect.y() + 22.0, hud_w - 12.0, 16.0), f"Điện thế V: {pot:.1f} V")

    def _draw_coulomb_hud(self, painter: QPainter) -> None:
        """Renders Coulomb's Law & Electrostatic parameters card."""
        hud_w = 210.0
        hud_h = 100.0
        hud_x = self._width - hud_w - 18.0
        hud_y = 48.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(hud_rect, 10.0, 10.0)

        # Title
        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))
        preset_name = ELECTROSTATIC_PRESETS.get(self._active_preset, {}).get("name", "Tĩnh điện")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 8.0, hud_w - 16.0, 18.0), f"Định Luật Coulomb ({preset_name})")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        painter.drawText(QRectF(hud_x + 8.0, hud_y + 28.0, hud_w - 16.0, 16.0), "E⃗ = k·Σ(qᵢ/rᵢ²)·r̂ᵢ")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 46.0, hud_w - 16.0, 16.0), f"Số điện tích: {len(self._charges)} hạt")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 64.0, hud_w - 16.0, 16.0), f"Đường sức vẽ RK4: {len(self._field_lines)} đường")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 82.0, hud_w - 16.0, 16.0), "Kéo hạt để uốn cong điện trường")

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders header controls and preset selectors."""
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

        # Probe button [🎯 Đầu dò]
        probe_col = QColor("#22c55e") if self._show_probe else QColor("#64748b")
        painter.setPen(QPen(probe_col, 1.2))
        painter.setBrush(QBrush(QColor(probe_col.red(), probe_col.green(), probe_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_probe, 6.0, 6.0)
        painter.setPen(QPen(probe_col))
        painter.drawText(self._rect_btn_probe, Qt.AlignmentFlag.AlignCenter, "🎯 Đầu dò E⃗")

        # Invert Polarity button [± Đổi dấu]
        painter.setPen(QPen(QColor("#f59e0b"), 1.2))
        painter.setBrush(QBrush(QColor(245, 158, 11, 40)))
        painter.drawRoundedRect(self._rect_btn_invert, 6.0, 6.0)
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(self._rect_btn_invert, Qt.AlignmentFlag.AlignCenter, "± Đổi dấu")

        # Preset buttons across bottom
        for i, key in enumerate(self._preset_keys):
            rect = self._rect_btn_presets[i]
            is_active = (key == self._active_preset)
            p_data = ELECTROSTATIC_PRESETS.get(key, {})
            name = p_data.get("name", key)

            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.4))
                painter.setBrush(QBrush(QColor(56, 189, 248, 110)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, name)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles button clicks or charge/probe dragging."""
        pos = event.pos()

        # Check buttons
        if self._rect_btn_close.contains(pos):
            self.close()
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

        if self._rect_btn_probe.contains(pos):
            self.toggle_probe()
            event.accept()
            return

        if self._rect_btn_invert.contains(pos):
            self.invert_charges()
            event.accept()
            return

        for i, rect in enumerate(self._rect_btn_presets):
            if rect.contains(pos):
                self.set_preset(self._preset_keys[i])
                event.accept()
                return

        # Check probe click
        if self._show_probe and (pos - self._probe_pos).manhattanLength() < 20.0:
            self._active_drag = "PROBE"
            event.accept()
            return

        # Check charge click
        for charge in self._charges:
            if (pos - charge.pos).manhattanLength() < 24.0:
                self._active_drag = "CHARGE"
                self._dragged_charge_id = charge.charge_id
                event.accept()
                return

        # Background drag
        self._active_drag = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles charge or probe movement during dragging."""
        pos = event.pos()

        if self._active_drag == "PROBE":
            new_x = max(20.0, min(self._width - 20.0, pos.x()))
            new_y = max(45.0, min(self._height - 45.0, pos.y()))
            self._probe_pos = QPointF(new_x, new_y)
            pex, pey = self.electric_field_at(new_x, new_y)
            self.probe_updated.emit(math.hypot(pex, pey), math.degrees(math.atan2(pey, pex)), self.electric_potential_at(new_x, new_y))
            self.update()
            event.accept()
            return

        if self._active_drag == "CHARGE" and self._dragged_charge_id:
            for charge in self._charges:
                if charge.charge_id == self._dragged_charge_id:
                    new_x = max(30.0, min(self._width - 30.0, pos.x()))
                    new_y = max(50.0, min(self._height - 50.0, pos.y()))
                    charge.pos = QPointF(new_x, new_y)
                    self.charge_moved.emit(charge.charge_id, new_x, new_y)
                    self.recompute_field_lines()
                    self.update()
                    event.accept()
                    return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Releases drag state."""
        self._active_drag = None
        self._dragged_charge_id = None
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
        """Touchless air pinch drag of charges, probe, or entire card."""
        pos = self.mapFromScene(scene_pos)

        if self._active_drag is None:
            # Check probe
            if self._show_probe and (pos - self._probe_pos).manhattanLength() < 26.0:
                self._active_drag = "PROBE"
            else:
                for charge in self._charges:
                    if (pos - charge.pos).manhattanLength() < 28.0:
                        self._active_drag = "CHARGE"
                        self._dragged_charge_id = charge.charge_id
                        break
            if self._active_drag is None:
                self._active_drag = "BODY"
                self._drag_start_item_pos = pos

        if self._active_drag == "PROBE":
            new_x = max(20.0, min(self._width - 20.0, pos.x()))
            new_y = max(45.0, min(self._height - 45.0, pos.y()))
            self._probe_pos = QPointF(new_x, new_y)
            self.update()
            return True

        if self._active_drag == "CHARGE" and self._dragged_charge_id:
            for charge in self._charges:
                if charge.charge_id == self._dragged_charge_id:
                    new_x = max(30.0, min(self._width - 30.0, pos.x()))
                    new_y = max(50.0, min(self._height - 50.0, pos.y()))
                    charge.pos = QPointF(new_x, new_y)
                    self.recompute_field_lines()
                    self.update()
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

        if self._rect_btn_probe.contains(pos):
            self.toggle_probe()
            return True

        if self._rect_btn_invert.contains(pos):
            self.invert_charges()
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
        """Removes this interactive item from its scene."""
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps vector electric field lines, point charges,
        and Coulomb's law formulas onto LessonBoard.
        """
        from src.ui.stem_tools import PhysicsTools
        from PySide6.QtWidgets import QGraphicsTextItem

        stamped: list[QGraphicsItem] = []
        origin = self.pos()

        # Build absolute charge tuples: (x, y, q)
        scene_charges: list[tuple[float, float, float]] = []
        for c in self._charges:
            scene_charges.append((origin.x() + c.pos.x(), origin.y() + c.pos.y(), c.q))

        # Build absolute field line polylines
        scene_lines: list[list[tuple[float, float]]] = []
        for line in self._field_lines:
            line_pts = [(origin.x() + pt.x(), origin.y() + pt.y()) for pt in line]
            scene_lines.append(line_pts)

        # Stamp vector field diagram
        field_items = PhysicsTools.create_electric_field_diagram(
            charges=scene_charges,
            field_lines=scene_lines,
            pen_color=QColor("#38bdf8"),
        )
        for itm in field_items:
            board.addItem(itm)
            stamped.append(itm)

        # Stamped Electrostatic Formulas Text Card
        preset_name = ELECTROSTATIC_PRESETS.get(self._active_preset, {}).get("name", "Tĩnh điện")
        lbl_text = (
            f"Điện Trường & Định Luật Coulomb ({preset_name}):\n"
            f"1. Cường độ điện trường: E⃗ = (1 / 4πε₀) · Σ(qᵢ / rᵢ²) · r̂ᵢ\n"
            f"2. Điện thế tĩnh điện: V = (1 / 4πε₀) · Σ(qᵢ / rᵢ)\n"
            f"3. Đặc điểm đường sức điện: Xuất phát từ (+), kết thúc tại (−) hoặc vô cực."
        )
        txt_item = QGraphicsTextItem(lbl_text)
        txt_font = QFont("Segoe UI", 10)
        txt_font.setBold(True)
        txt_item.setFont(txt_font)
        txt_item.setDefaultTextColor(QColor("#f8fafc"))
        txt_item.setPos(origin.x() + 20.0, origin.y() + self._height + 15.0)
        txt_item.setZValue(12.5)
        board.addItem(txt_item)
        stamped.append(txt_item)

        return stamped
