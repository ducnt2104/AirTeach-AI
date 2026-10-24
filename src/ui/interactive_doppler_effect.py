"""
AirTeach AI – Interactive Wave Mechanics, Acoustics & Astrophysics: Doppler Effect, Sonic Boom & Mach Cone Simulator
====================================================================================================================
Provides an authentic physics simulation of wave mechanics and the Doppler effect:
  - Classical Acoustic Doppler Effect (Christian Doppler, 1842):
      f' = f_0 * (v ± v_o) / (v ∓ v_s)
      When observer is stationary (v_o = 0):
        Ahead (approaching):  f'_ahead = f_0 * v / (v - v_s) = f_0 / (1 - M)
        Behind (receding):     f'_behind = f_0 * v / (v + v_s) = f_0 / (1 + M)
        At arbitrary angle θ:  f'(θ) = f_0 * v / (v - v_s * cos θ)
  - Wavelength Compression & Dilation:
      λ_ahead = λ_0 * (1 - M)
      λ_behind = λ_0 * (1 + M)
  - Aerodynamic Mach Number & Shock Wave Cone (Ernst Mach, 1887):
      M = v_s / v_sound
      Subsonic (M < 1): Spherical wavefronts crowd ahead without intersecting.
      Sonic Barrier (M = 1): Wavefronts pile up at the leading edge creating a sonic barrier.
      Supersonic (M > 1): Wavefronts form an envelope tangent cone with half-angle:
          sin(μ) = 1 / M  ==>  μ = arcsin(1 / M)
          Acoustic pressure discontinuity produces sonic boom shock front.
  - Double Doppler Radar & Echocardiography:
      f_reflected = f_0 * (c + v) / (c - v) ==> Beat frequency Δf ≈ 2 * f_0 * v / c
  - Relativistic Doppler Effect & Cosmological Redshift:
      ν_obs = ν_0 * sqrt((1 - β) / (1 + β)) where β = v / c
      Redshift parameter z = sqrt((1 + β) / (1 - β)) - 1
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


# Physical Constants
SOUND_SPEED_AIR_20C = 343.0          # m/s at 20°C
SOUND_SPEED_WATER = 1482.0           # m/s in water
SPEED_OF_LIGHT = 299792458.0         # m/s in vacuum


@dataclass(frozen=True)
class DopplerPreset:
    code: str
    title_vi: str
    badge_label: str
    mach_number: float
    description: str


PRESETS: dict[str, DopplerPreset] = {
    "SUBSONIC_MOVING_SOURCE": DopplerPreset(
        code="SUBSONIC_MOVING_SOURCE",
        title_vi="Nguồn âm dưới âm tốc (Subsonic M = 0.5)",
        badge_label="DƯỚI ÂM",
        mach_number=0.50,
        description="M < 1: Sóng âm bị dồn ép phía trước (f tăng, λ giảm) và giãn rộng phía sau (f giảm, λ tăng).",
    ),
    "SONIC_BARRIER_MACH_ONE": DopplerPreset(
        code="SONIC_BARRIER_MACH_ONE",
        title_vi="Bức tường âm thanh (Sonic Barrier M = 1.0)",
        badge_label="BỨC TƯỜNG ÂM",
        mach_number=1.00,
        description="M = 1: Vận tốc nguồn bằng vận tốc âm thanh. Các đỉnh sóng chồng chập tạo rào cản áp suất cực đại.",
    ),
    "SUPERSONIC_MACH_CONE": DopplerPreset(
        code="SUPERSONIC_MACH_CONE",
        title_vi="Nón sóng chấn động Mach (Supersonic M = 1.5)",
        badge_label="SIÊU ÂM",
        mach_number=1.50,
        description="M > 1: Nguồn bay nhanh hơn âm thanh, tạo mặt bao nón Mach sin(μ) = 1/M kèm tiếng nổ Sonic Boom.",
    ),
    "DOPPLER_RADAR_ECHOCARDIOGRAPHY": DopplerPreset(
        code="DOPPLER_RADAR_ECHOCARDIOGRAPHY",
        title_vi="Súng Radar Doppler & Siêu âm Tim mạch",
        badge_label="RADAR DOPPLER",
        mach_number=0.20,
        description="Hiệu ứng Doppler kép hai lần: Sóng phản xạ từ vật thể chuyển động có tần số phách Δf ≈ 2 f_0 v / c.",
    ),
    "RELATIVISTIC_COSMOLOGICAL_REDSHIFT": DopplerPreset(
        code="RELATIVISTIC_COSMOLOGICAL_REDSHIFT",
        title_vi="Dịch chuyển đỏ Vũ trụ học (Cosmological Redshift)",
        badge_label="DỊCH ĐỎ",
        mach_number=0.85,
        description="Thiên hà lùi xa ở vận tốc tương đối tính: Vạch quang phổ H_alpha dịch về phía đỏ theo định luật Hubble.",
    ),
}


class InteractiveDopplerEffectItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Wave Mechanics, Acoustic Doppler Effect,
    Sonic Boom, Mach Cone Shockwaves, and Cosmological Redshift.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    mach_changed = Signal(float)

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

        # Active Preset & Physical Parameters
        self._active_preset_code: str = "SUBSONIC_MOVING_SOURCE"
        self._preset: DopplerPreset = PRESETS[self._active_preset_code]

        self._mach_number: float = 0.50          # M = v_s / v
        self._source_frequency_hz: float = 440.0 # f_0 (Musical note A4)
        self._sound_speed_mps: float = 343.0    # Speed of sound in air (m/s)

        # Simulation Geometry & Animation State
        self._source_x: float = 80.0             # Current x position of moving source (in viewport coords)
        self._direction: float = 1.0             # +1.0 moving right, -1.0 moving left
        self._is_animating: bool = True
        self._anim_phase: float = 0.0

        # Emitted circular wave history: list of (emit_x, emit_y, emit_time)
        self._wavefronts: list[tuple[float, float, float]] = []
        self._sim_time: float = 0.0
        self._emission_interval: float = 0.25    # Seconds between wavefront emissions
        self._last_emission_time: float = 0.0

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physics Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def source_velocity_mps(self) -> float:
        """Source speed v_s = M * v_sound in m/s."""
        return self._mach_number * self._sound_speed_mps

    @property
    def observed_frequency_ahead_hz(self) -> float:
        """
        Observed frequency ahead of approaching source:
        f'_ahead = f_0 / (1 - M) for M < 1.0.
        If M >= 1.0, sound waves form a shock front singularity.
        """
        if self._mach_number >= 1.0:
            return float("inf")
        return self._source_frequency_hz / (1.0 - self._mach_number)

    @property
    def observed_frequency_behind_hz(self) -> float:
        """
        Observed frequency behind receding source:
        f'_behind = f_0 / (1 + M).
        """
        return self._source_frequency_hz / (1.0 + self._mach_number)

    @property
    def wavelength_ahead_ratio(self) -> float:
        """Ratio λ_ahead / λ_0 = (1 - M)."""
        return max(0.0, 1.0 - self._mach_number)

    @property
    def wavelength_behind_ratio(self) -> float:
        """Ratio λ_behind / λ_0 = (1 + M)."""
        return 1.0 + self._mach_number

    @property
    def mach_angle_deg(self) -> Optional[float]:
        """
        Mach cone half-angle μ in degrees:
        sin(μ) = 1 / M (only exists when M >= 1.0).
        """
        if self._mach_number < 1.0:
            return None
        return math.degrees(math.asin(1.0 / self._mach_number))

    @property
    def beat_frequency_radar_hz(self) -> float:
        """
        Doppler radar beat frequency for 24 GHz microwave radar:
        Δf = 2 * f_radar * (v_s / c).
        """
        f_radar = 24.125e9  # 24.125 GHz K-band police radar
        return 2.0 * f_radar * (self.source_velocity_mps / SPEED_OF_LIGHT)

    @property
    def cosmological_redshift_z(self) -> float:
        """
        Relativistic redshift parameter:
        z = sqrt((1 + β) / (1 - β)) - 1 where β = M * 0.9 (relativistic speed fraction).
        """
        beta = min(0.98, self._mach_number * 0.6)
        return math.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> DopplerPreset:
        return self._preset

    @property
    def mach_number(self) -> float:
        return self._mach_number

    @property
    def source_frequency_hz(self) -> float:
        return self._source_frequency_hz

    @property
    def sound_speed_mps(self) -> float:
        return self._sound_speed_mps

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum preset and sets corresponding physical parameters."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self.set_mach_number(self._preset.mach_number)
        self.reset_wavefronts()
        self.update()

    def set_mach_number(self, mach: float) -> None:
        """Sets Mach number clamped between 0.0 and 2.5."""
        self._mach_number = max(0.0, min(2.5, round(mach, 2)))
        self.mach_changed.emit(self._mach_number)
        self.update()

    def set_source_frequency_hz(self, freq: float) -> None:
        self._source_frequency_hz = max(100.0, min(2000.0, freq))
        self.update()

    def set_sound_speed_mps(self, speed: float) -> None:
        self._sound_speed_mps = max(100.0, min(3000.0, speed))
        self.update()

    def toggle_direction(self) -> None:
        self._direction = -self._direction
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def reset_wavefronts(self) -> None:
        self._wavefronts.clear()
        self._source_x = 80.0 if self._direction > 0 else 360.0
        self._sim_time = 0.0
        self._last_emission_time = 0.0
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
        dt = 0.033
        self._sim_time += dt
        self._anim_phase += 0.1
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi

        # Move source
        vp_min_x = 50.0
        vp_max_x = 390.0
        speed_px_per_sec = 80.0 * self._mach_number

        self._source_x += self._direction * speed_px_per_sec * dt

        # Wrap around or bounce at viewport edges
        if self._source_x > vp_max_x:
            self._source_x = vp_min_x
            self._wavefronts.clear()
        elif self._source_x < vp_min_x:
            self._source_x = vp_max_x
            self._wavefronts.clear()

        # Emit new circular wavefront
        if (self._sim_time - self._last_emission_time) >= self._emission_interval:
            cy = 195.0
            self._wavefronts.append((self._source_x, cy, self._sim_time))
            self._last_emission_time = self._sim_time

            # Prune old waves that expanded beyond view
            max_age = 4.0
            self._wavefronts = [w for w in self._wavefronts if (self._sim_time - w[2]) <= max_age]

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
        painter.setBrush(QBrush(QColor("#070b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Acoustic Wavefronts & Mach Cone)
        left_w = 445.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_viewport(painter, view_rect)

        # 4. Right Control & Physical Readout Panel
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
        painter.drawText(QPointF(16.0, 24.0), "HIỆU ỨNG DOPPLER & NÓN SÓNG CHẤN ĐỘNG MACH (WAVE MECHANICS)")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Nén bước sóng • Bức tường âm thanh (M=1) • Nón Mach sin(μ)=1/M • Dịch đỏ Vũ trụ")

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

    def _draw_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        if self._active_preset_code == "RELATIVISTIC_COSMOLOGICAL_REDSHIFT":
            self._draw_cosmological_redshift(painter, rect)
            painter.restore()
            return

        # 1. Subtle horizontal flight axis & grid
        cy = rect.center().y()
        painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), cy), QPointF(rect.right(), cy))

        # 2. Draw Emitted Expanding Wavefront Rings
        wave_speed_px = 80.0  # Wave propagation velocity in canvas pixels per second
        for emit_x, emit_y, emit_t in self._wavefronts:
            age = self._sim_time - emit_t
            if age <= 0.0:
                continue
            radius = wave_speed_px * age

            # Alpha fade as waves expand
            alpha = max(20, int(220 * (1.0 - age / 4.0)))

            if self._mach_number > 1.0:
                wave_col = QColor(245, 158, 11, alpha)  # Amber for supersonic
            elif abs(self._mach_number - 1.0) < 0.05:
                wave_col = QColor(239, 68, 68, alpha)   # Red for sonic barrier
            else:
                wave_col = QColor(56, 189, 248, alpha)  # Cyan for subsonic

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(wave_col, 1.3))
            painter.drawEllipse(QPointF(emit_x, emit_y), radius, radius)

        # 3. Draw Mach Cone Tangent Lines (When M >= 1.0)
        if self._mach_number >= 1.0:
            self._draw_mach_cone(painter, rect, QPointF(self._source_x, cy))

        # 4. Draw Observers (Ahead and Behind)
        obs_ahead_x = rect.right() - 25.0 if self._direction > 0 else rect.left() + 25.0
        obs_behind_x = rect.left() + 25.0 if self._direction > 0 else rect.right() - 25.0

        self._draw_observer(painter, QPointF(obs_ahead_x, cy), "A (Phía trước)", QColor("#10b981"))
        self._draw_observer(painter, QPointF(obs_behind_x, cy), "B (Phía sau)", QColor("#f43f5e"))

        # 5. Draw Moving Source (Aircraft / Sound Emitter)
        self._draw_moving_source(painter, QPointF(self._source_x, cy))

        # 6. Top Status Banner inside Viewport
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        m_col = QColor("#10b981") if self._mach_number < 1.0 else (QColor("#ef4444") if abs(self._mach_number - 1.0) < 0.05 else QColor("#f59e0b"))
        painter.setPen(m_col)

        status_str = f"Chế độ: {self._preset.badge_label} (M = {self._mach_number:.2f}) • v_nguồn = {self.source_velocity_mps:.1f} m/s"
        if self._mach_number >= 1.0 and self.mach_angle_deg is not None:
            status_str += f" • Góc nón Mach μ = {self.mach_angle_deg:.1f}°"
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 22.0), status_str)

        painter.restore()

    def _draw_mach_cone(self, painter: QPainter, rect: QRectF, vertex: QPointF) -> None:
        """Draws the tangent shock wave envelope (Mach Cone)."""
        mu_deg = self.mach_angle_deg
        if mu_deg is None:
            return

        mu_rad = math.radians(mu_deg)
        cone_length = 350.0

        # When moving right (+1.0), cone trails to the left
        dx = -self._direction * cone_length
        dy1 = cone_length * math.tan(mu_rad)
        dy2 = -dy1

        p_top = QPointF(vertex.x() + dx, vertex.y() - dy1)
        p_bot = QPointF(vertex.x() + dx, vertex.y() + dy1)

        # Semi-transparent shock wave region inside cone
        cone_path = QPainterPath()
        cone_path.moveTo(vertex)
        cone_path.lineTo(p_top)
        cone_path.lineTo(p_bot)
        cone_path.closeSubpath()

        painter.setBrush(QBrush(QColor(239, 68, 68, 30)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(cone_path)

        # Discontinuity Shock wavefronts lines (strong amber/red)
        shock_pen = QPen(QColor("#ef4444"), 2.2)
        painter.setPen(shock_pen)
        painter.drawLine(vertex, p_top)
        painter.drawLine(vertex, p_bot)

        # Label Mach angle
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        lbl_x = vertex.x() - self._direction * 45.0
        painter.drawText(QPointF(lbl_x, vertex.y() - 14.0), f"μ = {mu_deg:.1f}°")

        # Sonic boom badge
        boom_r = QRectF(vertex.x() - self._direction * 110.0, vertex.y() + 25.0, 95.0, 18.0)
        painter.setBrush(QBrush(QColor("#7f1d1d")))
        painter.setPen(QPen(QColor("#ef4444"), 1.0))
        painter.drawRoundedRect(boom_r, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(boom_r, Qt.AlignmentFlag.AlignCenter, "💥 SONIC BOOM")

    def _draw_moving_source(self, painter: QPainter, pos: QPointF) -> None:
        """Draws the supersonic jet or acoustic emitter icon."""
        painter.save()
        painter.translate(pos)
        if self._direction < 0:
            painter.scale(-1.0, 1.0)

        # Fuselage / Jet profile
        jet_path = QPainterPath()
        jet_path.moveTo(14.0, 0.0)      # Nose
        jet_path.lineTo(2.0, -4.0)
        jet_path.lineTo(-10.0, -14.0)   # Wing tip
        jet_path.lineTo(-6.0, -3.0)
        jet_path.lineTo(-14.0, -2.0)    # Tail
        jet_path.lineTo(-14.0, 2.0)
        jet_path.lineTo(-6.0, 3.0)
        jet_path.lineTo(-10.0, 14.0)    # Wing tip
        jet_path.lineTo(2.0, 4.0)
        jet_path.closeSubpath()

        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawPath(jet_path)

        # Afterburner engine thrust flame if supersonic
        if self._mach_number >= 1.0:
            flame_len = 8.0 + 4.0 * math.sin(self._anim_phase * 4.0)
            flame_path = QPainterPath()
            flame_path.moveTo(-14.0, -2.0)
            flame_path.lineTo(-14.0 - flame_len, 0.0)
            flame_path.lineTo(-14.0, 2.0)
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(flame_path)

        # Center dot
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(QPointF(0.0, 0.0), 2.5, 2.5)

        painter.restore()

    def _draw_observer(self, painter: QPainter, pos: QPointF, label: str, col: QColor) -> None:
        """Draws a listening observer microphone station."""
        painter.setBrush(QBrush(col))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(pos, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(col)
        painter.drawText(QPointF(pos.x() - 25.0, pos.y() + 18.0), label)

    def _draw_cosmological_redshift(self, painter: QPainter, rect: QRectF) -> None:
        """Draws relativistic optical spectrum redshift comparison (Hydrogen Balmer series)."""
        ox = rect.left() + 20.0
        w = rect.width() - 40.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(ox, rect.top() + 24.0), "DỊCH CHUYỂN ĐỎ VŨ TRỤ HỌC (RELATIVISTIC COSMOLOGICAL REDSHIFT)")

        z = self.cosmological_redshift_z
        beta = min(0.98, self._mach_number * 0.6)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(ox, rect.top() + 40.0), f"Vận tốc lùi xa v = {beta:.2f} c • Thông số dịch đỏ z = Δλ / λ_0 = {z:.3f} • Định luật Hubble v = H_0·d")

        # 1. Laboratory Reference Spectrum (Stationary Source)
        lab_y = rect.top() + 75.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, lab_y - 6.0), "Quang phổ chuẩn trong phòng thí nghiệm (Đứng yên v = 0):")

        spec_rect_lab = QRectF(ox, lab_y, w, 40.0)
        self._draw_rainbow_spectrum_bar(painter, spec_rect_lab)

        # Hydrogen Balmer absorption lines in laboratory:
        # H-alpha: 656.3 nm (Red), H-beta: 486.1 nm (Cyan), H-gamma: 434.0 nm (Violet)
        self._draw_spectral_line(painter, spec_rect_lab, 656.3, "H_α (656 nm)")
        self._draw_spectral_line(painter, spec_rect_lab, 486.1, "H_β (486 nm)")
        self._draw_spectral_line(painter, spec_rect_lab, 434.0, "H_γ (434 nm)")

        # 2. Shifted Spectrum from Receding Galaxy
        gal_y = lab_y + 85.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ef4444"))
        painter.drawText(QPointF(ox, gal_y - 6.0), f"Quang phổ quan sát từ Thiên hà lùi xa (z = {z:.3f}):")

        spec_rect_gal = QRectF(ox, gal_y, w, 40.0)
        self._draw_rainbow_spectrum_bar(painter, spec_rect_gal)

        # Shifted wavelengths: λ_obs = λ_0 * (1 + z)
        self._draw_spectral_line(painter, spec_rect_gal, 656.3 * (1.0 + z), f"{656.3*(1+z):.0f} nm", is_shifted=True)
        self._draw_spectral_line(painter, spec_rect_gal, 486.1 * (1.0 + z), f"{486.1*(1+z):.0f} nm", is_shifted=True)
        self._draw_spectral_line(painter, spec_rect_gal, 434.0 * (1.0 + z), f"{434.0*(1+z):.0f} nm", is_shifted=True)

        # Big arrow connecting H-alpha shift
        x_lab_ha = ox + ((656.3 - 400.0) / 300.0) * w
        x_gal_ha = min(rect.right() - 10.0, ox + (((656.3 * (1.0 + z)) - 400.0) / 300.0) * w)

        shift_pen = QPen(QColor("#ef4444"), 2.0, Qt.PenStyle.DashLine)
        painter.setPen(shift_pen)
        painter.drawLine(QPointF(x_lab_ha, lab_y + 40.0), QPointF(x_gal_ha, gal_y))

        # Bottom Astrophysics card
        card_r = QRectF(ox, rect.bottom() - 48.0, w, 36.0)
        painter.setBrush(QBrush(QColor("#451a03")))
        painter.setPen(QPen(QColor("#f59e0b"), 1.2))
        painter.drawRoundedRect(card_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            card_r,
            Qt.AlignmentFlag.AlignCenter,
            f"KẾT LUẬN THỰC NGHIỆM: Vạch H_α dịch chuyển mạnh về phía Hồng ngoại (Δλ = +{656.3*z:.1f} nm).\n"
            f"Bằng chứng trực tiếp chứng minh Vũ trụ đang giãn nở có gia tốc (Giải Nobel Vật lý 2011)."
        )

    def _draw_rainbow_spectrum_bar(self, painter: QPainter, rect: QRectF) -> None:
        """Draws visual visible spectrum band from 400 nm to 700 nm."""
        painter.setPen(QPen(QColor("#475569"), 1.0))
        n_strips = 60
        dx = rect.width() / float(n_strips)

        for i in range(n_strips):
            wl = 400.0 + (i / float(n_strips)) * 300.0
            r, g, b = self._wavelength_to_rgb(wl)
            painter.setBrush(QBrush(QColor(r, g, b)))
            strip_rect = QRectF(rect.left() + i * dx, rect.top(), dx + 0.5, rect.height())
            painter.drawRect(strip_rect)

    def _draw_spectral_line(self, painter: QPainter, rect: QRectF, wl: float, label: str, is_shifted: bool = False) -> None:
        """Draws absorption line on spectrum."""
        if wl < 400.0 or wl > 700.0:
            if wl > 700.0 and is_shifted:
                # Beyond visible spectrum in infrared
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                painter.setPen(QColor("#ef4444"))
                painter.drawText(QPointF(rect.right() - 75.0, rect.top() - 4.0), f"➔ Hồng ngoại ({label})")
            return

        x = rect.left() + ((wl - 400.0) / 300.0) * rect.width()
        painter.setPen(QPen(QColor("#000000"), 2.2))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff") if not is_shifted else QColor("#ef4444"))
        painter.drawText(QPointF(x - 14.0, rect.top() - 4.0), label)

    @staticmethod
    def _wavelength_to_rgb(wavelength: float) -> tuple[int, int, int]:
        """Converts optical wavelength (400-700 nm) to approximate RGB."""
        w = float(wavelength)
        if 400.0 <= w < 440.0:
            r = -(w - 440.0) / (440.0 - 400.0)
            g = 0.0
            b = 1.0
        elif 440.0 <= w < 490.0:
            r = 0.0
            g = (w - 440.0) / (490.0 - 440.0)
            b = 1.0
        elif 490.0 <= w < 510.0:
            r = 0.0
            g = 1.0
            b = -(w - 510.0) / (510.0 - 490.0)
        elif 510.0 <= w < 580.0:
            r = (w - 510.0) / (580.0 - 510.0)
            g = 1.0
            b = 0.0
        elif 580.0 <= w < 645.0:
            r = 1.0
            g = -(w - 645.0) / (645.0 - 580.0)
            b = 0.0
        elif 645.0 <= w <= 700.0:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            return 30, 30, 30
        return int(r * 255), int(g * 255), int(b * 255)

    # -------------------------------------------------------------------------
    # Right Control & Readout Panel
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
        painter.drawText(QPointF(ox, y), "THÔNG SỐ & VẬN TỐC:")

        # Mach Number Card
        y += 18.0
        mach_card = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(mach_card, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        m_col = QColor("#10b981") if self._mach_number < 1.0 else (QColor("#ef4444") if abs(self._mach_number - 1.0) < 0.05 else QColor("#f59e0b"))
        painter.setPen(m_col)
        painter.drawText(QRectF(ox + 8.0, y, 75.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"M={self._mach_number:.2f}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        state_txt = "Dưới âm" if self._mach_number < 1.0 else ("Bức tường âm" if abs(self._mach_number - 1.0) < 0.05 else "Siêu âm tốc")
        painter.drawText(QRectF(ox + 85.0, y + 2.0, w - 90.0, 28.0), Qt.AlignmentFlag.AlignVCenter, f"{state_txt}\nv = {self.source_velocity_mps:.0f} m/s")

        # Steppers for Mach number
        y += 38.0
        m_m05 = QRectF(ox, y, 32.0, 20.0)
        m_m01 = QRectF(ox + 36.0, y, 32.0, 20.0)
        m_p01 = QRectF(ox + 100.0, y, 32.0, 20.0)
        m_p05 = QRectF(ox + 136.0, y, 32.0, 20.0)

        self._btn_rects["m_m05"] = m_m05
        self._btn_rects["m_m01"] = m_m01
        self._btn_rects["m_p01"] = m_p01
        self._btn_rects["m_p05"] = m_p05

        for b_r, b_txt in [(m_m05, "-0.5"), (m_m01, "-0.1"), (m_p01, "+0.1"), (m_p05, "+0.5")]:
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)

        # Quantitative Physical Readouts
        y += 28.0
        f0 = self._source_frequency_hz
        f_ah = self.observed_frequency_ahead_hz
        f_be = self.observed_frequency_behind_hz

        f_ah_str = f"{f_ah:.1f} Hz ({f_ah/f0:.2f}×)" if f_ah != float("inf") else "Vô cực (Sóng xung)"
        f_be_str = f"{f_be:.1f} Hz ({f_be/f0:.2f}×)"

        mu_str = f"{self.mach_angle_deg:.1f}°" if self.mach_angle_deg is not None else "Không có (M < 1)"

        rows = [
            (f"• Tần số gốc f₀: {f0:.0f} Hz", QColor("#38bdf8")),
            (f"• Phía trước f'A: {f_ah_str}", QColor("#10b981")),
            (f"• Phía sau f'B: {f_be_str}", QColor("#f43f5e")),
            (f"• Tỷ lệ λ phía trước: {self.wavelength_ahead_ratio*100:.0f}%", QColor("#cbd5e1")),
            (f"• Tỷ lệ λ phía sau: {self.wavelength_behind_ratio*100:.0f}%", QColor("#cbd5e1")),
            (f"• Góc nón Mach μ: {mu_str}", QColor("#fbbf24")),
            (f"• Vận tốc âm v: {self._sound_speed_mps:.0f} m/s", QColor("#94a3b8")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 16.0

        # Action Buttons
        y = rect.bottom() - 48.0
        dir_r = QRectF(ox, y, w, 22.0)
        self._btn_rects["toggle_dir"] = dir_r
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(dir_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(dir_r, Qt.AlignmentFlag.AlignCenter, f"✈ ĐỔI HƯỚNG BAY ({'SANG PHẢI ➔' if self._direction > 0 else '⬅ SANG TRÁI'})")

        rst_r = QRectF(ox, rect.bottom() - 22.0, w / 2.0 - 4.0, 18.0)
        anim_r = QRectF(ox + w / 2.0 + 4.0, rect.bottom() - 22.0, w / 2.0 - 4.0, 18.0)
        self._btn_rects["reset"] = rst_r
        self._btn_rects["toggle_anim"] = anim_r

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(rst_r, 3.0, 3.0)
        painter.drawRoundedRect(anim_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rst_r, Qt.AlignmentFlag.AlignCenter, "Xóa sóng")
        painter.drawText(anim_r, Qt.AlignmentFlag.AlignCenter, "Dừng" if self._is_animating else "Chạy")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ HIỆU ỨNG DOPPLER (1842) & NÓN SÓNG CHẤN ĐỘNG MACH (1887):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Hiệu ứng Doppler: f' = f_0·v / (v ∓ v_s). Khi nguồn tiến lại gần, bước sóng bị nén lại làm âm thanh bổng hơn; khi ra xa, âm thanh trầm xuống.\n"
            f"• Khi v_s ≥ v (M ≥ 1), các mặt sóng hình cầu giao nhau tạo thành một hình nón bao gọi là Nón Mach với nửa góc sin(μ) = 1/M. "
            f"Sự biến thiên đột ngột của áp suất trên mặt nón gây ra tiếng nổ Sonic Boom chấn động.\n"
            f"• Mô tả chế độ: {self._preset.description}"
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

        # 2. Mach steppers
        if self._btn_rects.get("m_m05") and self._btn_rects["m_m05"].contains(local_pos):
            self.set_mach_number(self._mach_number - 0.5)
            return True
        if self._btn_rects.get("m_m01") and self._btn_rects["m_m01"].contains(local_pos):
            self.set_mach_number(self._mach_number - 0.1)
            return True
        if self._btn_rects.get("m_p01") and self._btn_rects["m_p01"].contains(local_pos):
            self.set_mach_number(self._mach_number + 0.1)
            return True
        if self._btn_rects.get("m_p05") and self._btn_rects["m_p05"].contains(local_pos):
            self.set_mach_number(self._mach_number + 0.5)
            return True

        # 3. Direction toggle
        if self._btn_rects.get("toggle_dir") and self._btn_rects["toggle_dir"].contains(local_pos):
            self.toggle_direction()
            return True

        # 4. Action buttons
        if self._btn_rects.get("reset") and self._btn_rects["reset"].contains(local_pos):
            self.reset_wavefronts()
            return True
        if self._btn_rects.get("toggle_anim") and self._btn_rects["toggle_anim"].contains(local_pos):
            self.toggle_animation()
            return True
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
