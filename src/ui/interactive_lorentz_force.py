"""
AirTeach AI - Interactive Physics Magnetism & Lorentz Force Particle Deflection Simulator
========================================================================================
Provides an authentic physics simulation of magnetic fields and Lorentz force:
  - Lorentz Force Law: F⃗_L = q * (v⃗ x B⃗).
  - Pure magnetic deflection: F⃗_L is always perpendicular to velocity v⃗,
    hence work done W = 0 and kinetic energy E_k = 0.5 * m * v^2 is strictly conserved.
  - Uniform circular motion with Cyclotron (Larmor) radius:
      R = (m * v) / (|q| * B)
      T = (2 * pi * m) / (|q| * B)
  - Interactive particle types: Electron (e⁻), Proton (p⁺), Alpha particle (He²⁺).
  - Reversible magnetic field direction: ⊗ (Into page) vs ⊙ (Out of page).
  - Real-time numerical integration with velocity rotation (Boris solver) ensuring zero artificial energy drift.
  - Interactive velocity and magnetic field strength adjustments.
  - Live vector arrows for velocity v⃗ (green) and Lorentz force F⃗_L (red).
  - Touchless air pinch dragging and air tap controls.
  - Vector diagram stamping onto LessonBoard via PhysicsTools.create_lorentz_force_diagram.
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

# Particle Specifications: (name_vi, charge_q, mass_m, color)
PARTICLE_SPECS: dict[str, tuple[str, float, float, QColor]] = {
    "ELECTRON": ("Electron (e⁻)", -1.0, 1.0, QColor("#38bdf8")),
    "PROTON": ("Proton (p⁺)", 1.0, 6.0, QColor("#ef4444")),
    "ALPHA": ("Hạt Alpha (He²⁺)", 2.0, 12.0, QColor("#f59e0b")),
}


class InteractiveLorentzForceItem(QGraphicsObject):
    """
    Interactive Physics Magnetism & Lorentz Force Particle Deflection Simulator,
    featuring live circular cyclotron orbits, reversible B-field direction (⊗ / ⊙),
    velocity & force vectors, and vector stamping onto LessonBoard.
    """

    particle_moved = Signal(float, float, float)  # (x, y, radius)
    parameters_changed = Signal(float, float)     # (velocity v, b_field B)

    def __init__(
        self,
        pos: tuple[float, float] = (500.0, 320.0),
        width: float = 680.0,
        height: float = 440.0,
        particle_type: str = "ELECTRON",
        b_field: float = 1.0,
        velocity: float = 160.0,
        b_direction: str = "INTO",
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

        # Physics Parameters
        self._particle_type = particle_type if particle_type in PARTICLE_SPECS else "ELECTRON"
        self._b_field = max(0.2, min(3.0, float(b_field)))       # Tesla (relative)
        self._speed = max(40.0, min(350.0, float(velocity)))     # Pixels/sec
        self._b_direction = "OUT" if b_direction.upper() == "OUT" else "INTO"

        # Magnetic Field Chamber Boundary (local coordinates)
        self._chamber_rect = QRectF(25.0, 48.0, 440.0, 320.0)

        # Particle State
        self._launch_pos = QPointF(245.0, 208.0)
        self._pos = QPointF(self._launch_pos)
        self._vel = QPointF(self._speed, 0.0)  # Starts moving right (+x)
        self._traj: list[QPointF] = [QPointF(self._pos)]
        self._max_traj_len = 120

        # Animation State
        self._is_animating = True
        self._timer = QTimer()
        self._timer.setInterval(25)  # 40 FPS
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Dragging state
        self._active_drag: Optional[str] = None  # "PARTICLE", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_play = QRectF(self._width - 245.0, 10.0, 90.0, 26.0)
        self._rect_btn_reset = QRectF(self._width - 335.0, 10.0, 85.0, 26.0)
        self._rect_btn_bdir = QRectF(self._width - 460.0, 10.0, 120.0, 26.0)

        # Particle Type Selector Buttons (Right side card)
        self._particle_keys = ["ELECTRON", "PROTON", "ALPHA"]
        self._rect_btn_particles: list[QRectF] = []
        p_card_x = self._width - 195.0
        p_card_y = 195.0
        btn_h = 24.0
        for i in range(len(self._particle_keys)):
            self._rect_btn_particles.append(QRectF(p_card_x, p_card_y + i * (btn_h + 6.0), 175.0, btn_h))

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Physical Calculations ────────────────────────────────────────────────

    def cyclotron_radius(self) -> float:
        """
        Calculates Cyclotron (Larmor) radius:
          R = (m * v) / (|q| * B)
        """
        _, q, m, _ = PARTICLE_SPECS[self._particle_type]
        q_mag = abs(q)
        b_eff = self._b_field * 0.016  # scale factor for screen dimensions
        if q_mag < 1e-4 or b_eff < 1e-4:
            return 9999.0
        return (m * self._speed) / (q_mag * b_eff * 100.0)

    def cyclotron_period(self) -> float:
        """
        Calculates Cyclotron period:
          T = (2 * pi * m) / (|q| * B)
        """
        _, q, m, _ = PARTICLE_SPECS[self._particle_type]
        q_mag = abs(q)
        if q_mag < 1e-4 or self._b_field < 1e-4:
            return 9999.0
        return (2.0 * math.pi * m) / (q_mag * self._b_field)

    def lorentz_force_vector(self) -> tuple[float, float]:
        """
        Calculates Lorentz force vector F⃗_L = q * (v⃗ x B⃗).
        For v = (vx, vy, 0) and B = (0, 0, Bz):
          Fx = q * vy * Bz
          Fy = -q * vx * Bz
        """
        _, q, _, _ = PARTICLE_SPECS[self._particle_type]
        bz = -self._b_field if self._b_direction == "INTO" else self._b_field
        scale_f = 0.22
        fx = q * self._vel.y() * bz * scale_f
        fy = -q * self._vel.x() * bz * scale_f
        return fx, fy

    def is_inside_chamber(self, pt: QPointF) -> bool:
        """Checks if point is within the magnetic field chamber."""
        return self._chamber_rect.contains(pt)

    # ─── Mutators & Controls ─────────────────────────────────────────────────

    def set_particle_type(self, p_type: str) -> None:
        """Selects active particle: ELECTRON, PROTON, or ALPHA."""
        if p_type in PARTICLE_SPECS:
            self._particle_type = p_type
            self.reset_particle()
            self.update()

    def toggle_b_direction(self) -> str:
        """Toggles magnetic field direction between INTO (⊗) and OUT (⊙)."""
        self._b_direction = "OUT" if self._b_direction == "INTO" else "INTO"
        self.update()
        return self._b_direction

    def toggle_animation(self) -> bool:
        """Toggles simulation play / pause."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()
        return self._is_animating

    def reset_particle(self) -> None:
        """Resets particle to launch position and initial velocity."""
        self._pos = QPointF(self._launch_pos)
        self._vel = QPointF(self._speed, 0.0)
        self._traj = [QPointF(self._pos)]
        self.update()

    def _on_sim_tick(self) -> None:
        """
        Boris / Velocity rotation integration:
        Rotates velocity vector v⃗ by angle dtheta = -q * Bz / m * dt,
        guaranteeing strict kinetic energy conservation without radius drift.
        """
        if not self._is_animating:
            return

        dt = 0.025
        in_field = self.is_inside_chamber(self._pos)

        if in_field:
            _, q, m, _ = PARTICLE_SPECS[self._particle_type]
            bz = -self._b_field if self._b_direction == "INTO" else self._b_field
            # Angular velocity omega = q * Bz / m
            omega = (q * bz * 2.8) / m
            d_theta = omega * dt

            # Rotate velocity
            vx = self._vel.x()
            vy = self._vel.y()
            cos_a = math.cos(d_theta)
            sin_a = math.sin(d_theta)
            new_vx = vx * cos_a - vy * sin_a
            new_vy = vx * sin_a + vy * cos_a
            self._vel = QPointF(new_vx, new_vy)

        # Advance position
        new_x = self._pos.x() + self._vel.x() * dt
        new_y = self._pos.y() + self._vel.y() * dt
        self._pos = QPointF(new_x, new_y)

        # Chamber wall reflection / bounce so simulation runs continuously
        r_pad = 12.0
        c = self._chamber_rect
        if new_x < c.left() + r_pad:
            self._pos.setX(c.left() + r_pad)
            self._vel.setX(abs(self._vel.x()))
        elif new_x > c.right() - r_pad:
            self._pos.setX(c.right() - r_pad)
            self._vel.setX(-abs(self._vel.x()))

        if new_y < c.top() + r_pad:
            self._pos.setY(c.top() + r_pad)
            self._vel.setY(abs(self._vel.y()))
        elif new_y > c.bottom() - r_pad:
            self._pos.setY(c.bottom() - r_pad)
            self._vel.setY(-abs(self._vel.y()))

        # Update trajectory tail
        self._traj.append(QPointF(self._pos))
        if len(self._traj) > self._max_traj_len:
            self._traj.pop(0)

        self.particle_moved.emit(self._pos.x(), self._pos.y(), self.cyclotron_radius())
        self.update()

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive Lorentz force & magnetic deflection simulation."""
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

        # 2. Render Magnetic Field Chamber (Grid & Symbols)
        self._draw_chamber_and_b_field(painter)

        # 3. Render Particle Trajectory Ribbon
        self._draw_trajectory(painter)

        # 4. Render Particle Sphere & Vectors (v⃗ and F⃗_L)
        self._draw_particle_and_vectors(painter)

        # 5. Lorentz Physics HUD Badge (Right side)
        self._draw_lorentz_hud(painter)

        # 6. Control & Particle Buttons
        self._draw_buttons(painter)

        painter.restore()

    def _draw_chamber_and_b_field(self, painter: QPainter) -> None:
        """Renders magnetic chamber boundary and regular ⊗ / ⊙ field direction symbols."""
        c = self._chamber_rect

        # Chamber Background Box
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.4, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(2, 132, 199, 20)))
        painter.drawRoundedRect(c, 8.0, 8.0)

        # Label: Vùng từ trường đều B
        f_lbl = QFont("Segoe UI", 8)
        f_lbl.setBold(True)
        painter.setFont(f_lbl)
        painter.setPen(QPen(QColor("#38bdf8")))
        dir_sym = "⊗ (Hướng vào trong)" if self._b_direction == "INTO" else "⊙ (Hướng ra ngoài)"
        painter.drawText(QPointF(c.left() + 10.0, c.top() + 16.0), f"Vùng từ trường đều B⃗ {dir_sym}")

        # Regular Grid of Symbols
        f_sym = QFont("Segoe UI", 10)
        f_sym.setBold(True)
        painter.setFont(f_sym)
        painter.setPen(QPen(QColor(71, 85, 105, 130)))

        is_into = (self._b_direction == "INTO")
        step_g = 48.0
        curr_x = c.left() + 32.0
        while curr_x < c.right() - 20.0:
            curr_y = c.top() + 36.0
            while curr_y < c.bottom() - 15.0:
                sym_str = "⊗" if is_into else "⊙"
                painter.drawText(QPointF(curr_x - 6.0, curr_y + 4.0), sym_str)
                curr_y += step_g
            curr_x += step_g

    def _draw_trajectory(self, painter: QPainter) -> None:
        """Draws glowing particle trail with fading alpha ribbon."""
        if len(self._traj) < 2:
            return

        path = QPainterPath()
        path.moveTo(self._traj[0])
        for pt in self._traj[1:]:
            path.lineTo(pt)

        # Outer neon glow
        _, _, _, p_col = PARTICLE_SPECS[self._particle_type]
        pen_glow = QPen(QColor(p_col.red(), p_col.green(), p_col.blue(), 60), 4.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen_glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Core trajectory line
        pen_core = QPen(p_col, 1.8, Qt.PenStyle.SolidLine)
        painter.setPen(pen_core)
        painter.drawPath(path)

    def _draw_particle_and_vectors(self, painter: QPainter) -> None:
        """Renders orbiting particle with velocity arrow (green) and Lorentz force (red)."""
        p = self._pos
        r_part = 10.0
        name, q, _, p_col = PARTICLE_SPECS[self._particle_type]

        # Particle Glow Aura
        grad_glow = QRadialGradient(p.x(), p.y(), r_part * 2.0)
        grad_glow.setColorAt(0.0, QColor(p_col.red(), p_col.green(), p_col.blue(), 180))
        grad_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad_glow))
        painter.drawEllipse(p, r_part * 2.0, r_part * 2.0)

        # Particle Sphere
        grad_core = QRadialGradient(p.x() - r_part * 0.3, p.y() - r_part * 0.3, r_part * 1.3)
        grad_core.setColorAt(0.0, QColor("#ffffff"))
        grad_core.setColorAt(0.3, p_col)
        grad_core.setColorAt(1.0, p_col.darker(150))
        painter.setPen(QPen(p_col.darker(160), 1.2))
        painter.setBrush(QBrush(grad_core))
        painter.drawEllipse(p, r_part, r_part)

        # Sign on particle: '+' or '−'
        f_sign = QFont("Segoe UI", 9)
        f_sign.setBold(True)
        painter.setFont(f_sign)
        painter.setPen(QPen(QColor("#ffffff")))
        sign_ch = "−" if q < 0 else "+"
        painter.drawText(QRectF(p.x() - r_part, p.y() - r_part, r_part * 2, r_part * 2), Qt.AlignmentFlag.AlignCenter, sign_ch)

        # 1. Velocity Vector v⃗ (Green)
        vx = self._vel.x()
        vy = self._vel.y()
        v_mag = math.hypot(vx, vy)
        if v_mag > 1e-2:
            v_scale = 32.0
            ux, uy = vx / v_mag, vy / v_mag
            v_end = QPointF(p.x() + ux * v_scale, p.y() + uy * v_scale)

            pen_v = QPen(QColor("#10b981"), 2.2, Qt.PenStyle.SolidLine)
            painter.setPen(pen_v)
            painter.drawLine(p, v_end)
            self._draw_arrowhead(painter, p, v_end, QColor("#10b981"))

            # Label v⃗
            f_vec = QFont("Segoe UI", 8)
            f_vec.setBold(True)
            painter.setFont(f_vec)
            painter.setPen(QPen(QColor("#10b981")))
            painter.drawText(QPointF(v_end.x() + 3.0, v_end.y() - 4.0), "v⃗")

        # 2. Lorentz Force Vector F⃗_L (Red)
        fx, fy = self.lorentz_force_vector()
        f_mag = math.hypot(fx, fy)
        if f_mag > 1e-2:
            f_scale = min(36.0, f_mag * 1.5)
            f_ux, f_uy = fx / f_mag, fy / f_mag
            f_end = QPointF(p.x() + f_ux * f_scale, p.y() + f_uy * f_scale)

            pen_f = QPen(QColor("#ef4444"), 2.2, Qt.PenStyle.SolidLine)
            painter.setPen(pen_f)
            painter.drawLine(p, f_end)
            self._draw_arrowhead(painter, p, f_end, QColor("#ef4444"))

            # Label F⃗_L
            painter.setPen(QPen(QColor("#ef4444")))
            painter.drawText(QPointF(f_end.x() + 3.0, f_end.y() - 4.0), "F⃗_L")

    def _draw_arrowhead(self, painter: QPainter, p_start: QPointF, p_end: QPointF, color: QColor) -> None:
        """Draws small directional arrow on vector end."""
        dx = p_end.x() - p_start.x()
        dy = p_end.y() - p_start.y()
        length = math.hypot(dx, dy)
        if length < 1e-2:
            return

        ux, uy = dx / length, dy / length
        arr_size = 6.5
        wing_ang = math.radians(26)
        ang = math.atan2(uy, ux)

        path = QPainterPath()
        path.moveTo(p_end)
        path.lineTo(
            p_end.x() - arr_size * math.cos(ang - wing_ang),
            p_end.y() - arr_size * math.sin(ang - wing_ang),
        )
        path.lineTo(
            p_end.x() - arr_size * math.cos(ang + wing_ang),
            p_end.y() - arr_size * math.sin(ang + wing_ang),
        )
        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _draw_lorentz_hud(self, painter: QPainter) -> None:
        """Renders Lorentz Physics & Cyclotron parameters card."""
        hud_w = 185.0
        hud_h = 135.0
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
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 8.0, hud_w - 16.0, 18.0), "Định Luật Lorentz & Cyclotron")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        r_cyclotron = self.cyclotron_radius()
        t_cyclotron = self.cyclotron_period()
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 28.0, hud_w - 16.0, 16.0), "1. Lực: F⃗_L = q(v⃗ × B⃗) ⊥ v⃗")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 46.0, hud_w - 16.0, 16.0), f"2. Bán kính: R = {r_cyclotron:.1f} px")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 64.0, hud_w - 16.0, 16.0), f"3. Chu kỳ quay: T = {t_cyclotron:.2f} s")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 82.0, hud_w - 16.0, 16.0), f"4. Vận tốc v = {self._speed:.0f} px/s")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 100.0, hud_w - 16.0, 16.0), f"   Từ trường B = {self._b_field:.1f} T")
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 116.0, hud_w - 16.0, 16.0), "   Công thực hiện A = 0 J (v=const)")

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders header controls and particle selector buttons."""
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
        play_txt = "⏸ Tạm dừng" if self._is_animating else "▶ Chuyển động"
        play_col = QColor("#f59e0b") if self._is_animating else QColor("#10b981")
        painter.setPen(QPen(play_col, 1.2))
        painter.setBrush(QBrush(QColor(play_col.red(), play_col.green(), play_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_play, 6.0, 6.0)
        painter.setPen(QPen(play_col))
        painter.drawText(self._rect_btn_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # Reset button [↺ Đặt lại]
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QBrush(QColor(148, 163, 184, 30)))
        painter.drawRoundedRect(self._rect_btn_reset, 6.0, 6.0)
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

        # Magnetic Field Direction toggle button [⊗ Vào / ⊙ Ra]
        bdir_txt = "Từ trường: ⊗ Vào" if self._b_direction == "INTO" else "Từ trường: ⊙ Ra"
        painter.setPen(QPen(QColor("#06b6d4"), 1.2))
        painter.setBrush(QBrush(QColor(6, 182, 212, 40)))
        painter.drawRoundedRect(self._rect_btn_bdir, 6.0, 6.0)
        painter.setPen(QPen(QColor("#06b6d4")))
        painter.drawText(self._rect_btn_bdir, Qt.AlignmentFlag.AlignCenter, bdir_txt)

        # Particle Selector Buttons
        for i, key in enumerate(self._particle_keys):
            rect = self._rect_btn_particles[i]
            is_active = (key == self._particle_type)
            p_name, _, _, p_col = PARTICLE_SPECS[key]

            if is_active:
                painter.setPen(QPen(p_col, 1.4))
                painter.setBrush(QBrush(QColor(p_col.red(), p_col.green(), p_col.blue(), 90)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, p_name)

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles button clicks or particle/chamber dragging."""
        pos = event.pos()

        # Check buttons
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_play.contains(pos):
            self.toggle_animation()
            event.accept()
            return

        if self._rect_btn_reset.contains(pos):
            self.reset_particle()
            event.accept()
            return

        if self._rect_btn_bdir.contains(pos):
            self.toggle_b_direction()
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

        for i, rect in enumerate(self._rect_btn_particles):
            if rect.contains(pos):
                self.set_particle_type(self._particle_keys[i])
                event.accept()
                return

        # Check particle click to relocate
        if (pos - self._pos).manhattanLength() < 24.0:
            self._active_drag = "PARTICLE"
            event.accept()
            return

        # Background drag
        self._active_drag = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles particle repositioning during dragging."""
        pos = event.pos()

        if self._active_drag == "PARTICLE":
            new_x = max(self._chamber_rect.left() + 15.0, min(self._chamber_rect.right() - 15.0, pos.x()))
            new_y = max(self._chamber_rect.top() + 15.0, min(self._chamber_rect.bottom() - 15.0, pos.y()))
            self._launch_pos = QPointF(new_x, new_y)
            self._pos = QPointF(new_x, new_y)
            self._traj = [QPointF(self._pos)]
            self.update()
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
        """Touchless air pinch drag of particle launch position or entire card."""
        pos = self.mapFromScene(scene_pos)

        if self._active_drag is None:
            if (pos - self._pos).manhattanLength() < 30.0:
                self._active_drag = "PARTICLE"
            else:
                self._active_drag = "BODY"
                self._drag_start_item_pos = pos

        if self._active_drag == "PARTICLE":
            new_x = max(self._chamber_rect.left() + 15.0, min(self._chamber_rect.right() - 15.0, pos.x()))
            new_y = max(self._chamber_rect.top() + 15.0, min(self._chamber_rect.bottom() - 15.0, pos.y()))
            self._launch_pos = QPointF(new_x, new_y)
            self._pos = QPointF(new_x, new_y)
            self._traj = [QPointF(self._pos)]
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

        if self._rect_btn_play.contains(pos):
            self.toggle_animation()
            return True

        if self._rect_btn_reset.contains(pos):
            self.reset_particle()
            return True

        if self._rect_btn_bdir.contains(pos):
            self.toggle_b_direction()
            return True

        for i, rect in enumerate(self._rect_btn_particles):
            if rect.contains(pos):
                self.set_particle_type(self._particle_keys[i])
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
        """Stops timer and removes this interactive item from its scene."""
        self._timer.stop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self)

    # ─── Stamping onto LessonBoard ───────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """
        Permanently stamps vector magnetic field chamber, particle orbit,
        velocity & Lorentz force vectors, and formulas onto LessonBoard.
        """
        from src.ui.stem_tools import PhysicsTools
        from PySide6.QtWidgets import QGraphicsTextItem

        stamped: list[QGraphicsItem] = []
        origin = self.pos()

        # Build absolute chamber rect
        c = self._chamber_rect
        abs_chamber = (origin.x() + c.x(), origin.y() + c.y(), c.width(), c.height())

        # Build trajectory points
        traj_pts = [(origin.x() + pt.x(), origin.y() + pt.y()) for pt in self._traj]
        particle_pt = (origin.x() + self._pos.x(), origin.y() + self._pos.y())

        # Velocity & Force vectors
        v_vec = (self._vel.x() * 0.2, self._vel.y() * 0.2)
        fx, fy = self.lorentz_force_vector()
        f_vec = (fx * 1.5, fy * 1.5)

        lorentz_items = PhysicsTools.create_lorentz_force_diagram(
            field_rect=abs_chamber,
            b_direction=self._b_direction,
            trajectory=traj_pts,
            particle_pos=particle_pt,
            particle_type=self._particle_type,
            v_vector=v_vec,
            f_vector=f_vec,
            radius=self.cyclotron_radius(),
        )
        for itm in lorentz_items:
            board.addItem(itm)
            stamped.append(itm)

        # Formula & Law Text Card
        p_name, q_val, m_val, _ = PARTICLE_SPECS[self._particle_type]
        r_c = self.cyclotron_radius()
        t_c = self.cyclotron_period()
        lbl_text = (
            f"Lực Lorentz & Chuyển Động Hạt Mang Điện Trong Từ Trường ({p_name}):\n"
            f"1. Lực Lorentz: F⃗_L = q(v⃗ × B⃗) [Độ lớn F = |q|·v·B·sin α]\n"
            f"2. Bán kính Cyclotron: R = (m·v) / (|q|·B) = {r_c:.1f} px\n"
            f"3. Chu kỳ Cyclotron: T = 2πm / (|q|·B) = {t_c:.2f} s\n"
            f"4. Bảo toàn động năng: F⃗_L ⊥ v⃗ ⇒ Công A = 0, tốc độ v = const ({self._speed:.0f} px/s)"
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
