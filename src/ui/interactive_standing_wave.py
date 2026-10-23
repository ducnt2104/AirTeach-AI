"""
AirTeach AI - Interactive Physics Wave Motion & Standing Wave Resonance Simulator
=================================================================================
Provides an interactive physics wave mechanics simulation demonstrating:
  - Transverse wave propagation and standing wave resonance (Hiện tượng Sóng Dừng).
  - Standing wave on a string with two fixed ends: L = k * (lambda / 2).
  - Standing wave with one fixed end, one free end: L = (2k + 1) * (lambda / 4).
  - Harmonic mode resonance (k = 1, 2, 3, 4, 5, 6).
  - Nodes (Nút sóng, u = 0) and Antinodes (Bụng sóng, |u| = 2A).
  - Dynamic time oscillation: u(x, t) = 2A * sin(2*pi*x / lambda) * cos(omega * t).
  - Interactive harmonic mode selector, amplitude handle, play/pause controls.
  - Stamping permanent vector wave diagrams onto LessonBoard.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
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


class InteractiveStandingWaveItem(QGraphicsObject):
    """
    Interactive standing wave physics simulator with real-time vibration,
    harmonic modes, node/antinode tracking, and LessonBoard vector stamping.
    """

    harmonic_changed = Signal(int)             # (k_harmonic)
    amplitude_changed = Signal(float)          # (amplitude_px)
    mode_changed = Signal(str)                 # ("fixed_fixed" | "fixed_free")

    def __init__(
        self,
        pos: tuple[float, float] = (520.0, 340.0),
        width: float = 640.0,
        height: float = 400.0,
        string_length: float = 500.0,
        harmonic_k: int = 3,
        amplitude: float = 35.0,
        frequency_hz: float = 1.5,
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
        self._string_length = float(string_length)
        self._k = max(1, min(6, int(harmonic_k)))
        self._amplitude = max(10.0, min(65.0, float(amplitude)))
        self._frequency = max(0.2, min(5.0, float(frequency_hz)))
        self._is_fixed_free = False  # False: Hai đầu cố định, True: Một đầu tự do

        # Time tracking & Animation timer
        self._time = 0.0
        self._is_playing = True
        self._timer = QTimer()
        self._timer.setInterval(25)  # 40 FPS
        self._timer.timeout.connect(self._on_animation_tick)
        self._timer.start()

        # Geometry within item coordinates:
        # String baseline is centered vertically
        self._y_baseline = self._height * 0.52
        self._x_string_start = (self._width - self._string_length) * 0.5
        self._x_string_end = self._x_string_start + self._string_length

        # Interactive dragging states
        self._active_handle: Optional[str] = None  # "AMP", "BODY"
        self._drag_start_item_pos = QPointF()
        self._air_hover_pos: Optional[QPointF] = None

        # Control button bounding rects (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)
        self._rect_btn_stamp = QRectF(self._width - 150.0, 10.0, 110.0, 26.0)
        self._rect_btn_play = QRectF(self._width - 240.0, 10.0, 84.0, 26.0)
        self._rect_btn_boundary = QRectF(20.0, 10.0, 160.0, 26.0)

        # Harmonic mode buttons: k = 1 to 6
        self._rect_btn_harmonics: list[QRectF] = []
        btn_w = 40.0
        start_hx = 190.0
        for i in range(6):
            self._rect_btn_harmonics.append(QRectF(start_hx + i * (btn_w + 5.0), 10.0, btn_w, 26.0))

    # ─── Bounding Rect ────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        """Returns the item boundary with generous margin for glow and handles."""
        return QRectF(-15.0, -15.0, self._width + 30.0, self._height + 30.0)

    # ─── Physical Wave Calculations ──────────────────────────────────────────

    def harmonic_k(self) -> int:
        """Harmonic resonance index k (number of half-wavelength loops if fixed-fixed)."""
        return self._k

    def amplitude(self) -> float:
        """Wave amplitude A in pixels."""
        return self._amplitude

    def frequency(self) -> float:
        """Wave oscillation frequency f in Hertz."""
        return self._frequency

    def is_fixed_free(self) -> bool:
        """Returns True if one end is free; False if both ends are fixed."""
        return self._is_fixed_free

    def wavelength(self) -> float:
        """
        Calculates wavelength lambda in pixels:
          - Both ends fixed: L = k * (lambda / 2) => lambda = 2L / k.
          - One end fixed, one free: L = (2k + 1) * (lambda / 4) => lambda = 4L / (2k + 1).
        """
        L = self._string_length
        if not self._is_fixed_free:
            return (2.0 * L) / float(self._k)
        else:
            return (4.0 * L) / float(2 * self._k + 1)

    def wave_speed(self) -> float:
        """Calculates phase velocity v = lambda * f in pixels/second."""
        return self.wavelength() * self._frequency

    def count_nodes_and_antinodes(self) -> tuple[int, int]:
        """
        Returns (num_nodes, num_antinodes).
          - Fixed-Fixed: k antinodes, (k + 1) nodes.
          - Fixed-Free: (k + 1) antinodes, (k + 1) nodes.
        """
        if not self._is_fixed_free:
            return self._k + 1, self._k
        else:
            return self._k + 1, self._k + 1

    def node_positions(self) -> list[float]:
        """
        Returns the x-coordinates (local item space) of all stationary wave nodes (u = 0).
        """
        positions: list[float] = []
        L = self._string_length
        x0 = self._x_string_start
        lam = self.wavelength()

        if not self._is_fixed_free:
            # Nodes at x = n * (lambda / 2) for n in 0..k
            step = L / float(self._k)
            for n in range(self._k + 1):
                positions.append(x0 + n * step)
        else:
            # Nodes at x = n * (lambda / 2)
            half_lam = lam * 0.5
            n = 0
            while True:
                pos = x0 + n * half_lam
                if pos > self._x_string_end + 1e-3:
                    break
                positions.append(pos)
                n += 1

        return positions

    def antinode_positions(self) -> list[float]:
        """
        Returns the x-coordinates (local item space) of all wave antinodes (maximum amplitude).
        """
        positions: list[float] = []
        x0 = self._x_string_start
        L = self._string_length

        if not self._is_fixed_free:
            # Antinodes at midpoint of each loop: (2m + 1) * (L / (2k))
            step = L / float(self._k)
            for m in range(self._k):
                positions.append(x0 + (m + 0.5) * step)
        else:
            lam = self.wavelength()
            quarter_lam = lam * 0.25
            m = 0
            while True:
                pos = x0 + quarter_lam + m * (lam * 0.5)
                if pos > self._x_string_end + 1e-3:
                    break
                positions.append(pos)
                m += 1

        return positions

    def displacement_at(self, x_local: float, t: float) -> float:
        """
        Calculates instantaneous displacement u(x, t) in pixels:
          u(x, t) = 2A * sin(2*pi*x / lambda) * cos(omega * t)
        """
        x_rel = x_local - self._x_string_start
        if x_rel < 0.0 or x_rel > self._string_length:
            return 0.0

        lam = self.wavelength()
        k_wave = (2.0 * math.pi) / lam
        omega = 2.0 * math.pi * self._frequency
        spatial = math.sin(k_wave * x_rel)
        temporal = math.cos(omega * t)
        return self._amplitude * spatial * temporal

    def envelope_at(self, x_local: float) -> float:
        """Calculates maximum amplitude envelope |2A * sin(k*x)| at x."""
        x_rel = x_local - self._x_string_start
        if x_rel < 0.0 or x_rel > self._string_length:
            return 0.0

        lam = self.wavelength()
        k_wave = (2.0 * math.pi) / lam
        return self._amplitude * abs(math.sin(k_wave * x_rel))

    # ─── Mutators & Controls ─────────────────────────────────────────────────

    def set_harmonic_k(self, k: int) -> None:
        """Sets the harmonic resonance mode k (1 to 6)."""
        self._k = max(1, min(6, int(k)))
        self.harmonic_changed.emit(self._k)
        self.update()

    def set_amplitude(self, amp: float) -> None:
        """Sets wave amplitude A in pixels."""
        self._amplitude = max(10.0, min(65.0, float(amp)))
        self.amplitude_changed.emit(self._amplitude)
        self.update()

    def set_frequency(self, freq: float) -> None:
        """Sets wave frequency in Hertz."""
        self._frequency = max(0.2, min(5.0, float(freq)))
        self.update()

    def toggle_boundary_mode(self) -> None:
        """Toggles between 'fixed_fixed' and 'fixed_free' boundaries."""
        self._is_fixed_free = not self._is_fixed_free
        mode_str = "fixed_free" if self._is_fixed_free else "fixed_fixed"
        self.mode_changed.emit(mode_str)
        self.update()

    def toggle_play(self) -> bool:
        """Toggles play/pause state for the wave vibration animation."""
        self._is_playing = not self._is_playing
        if self._is_playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()
        return self._is_playing

    def is_playing(self) -> bool:
        """Returns True if the wave animation is actively oscillating."""
        return self._is_playing

    def _on_animation_tick(self) -> None:
        """Timer callback updating wave time step."""
        if self._is_playing:
            self._time += 0.025
            self.update()

    # ─── Paint Implementation ────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Renders the comprehensive standing wave mechanics simulation."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._height
        y0 = self._y_baseline
        x_start = self._x_string_start
        x_end = self._x_string_end

        # 1. Container Frosted Glass Card
        card_rect = QRectF(0.0, 0.0, w, h)
        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. String Equilibrium Baseline (Vị trí cân bằng)
        pen_base = QPen(QColor("#475569"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen_base)
        painter.drawLine(QPointF(x_start - 20.0, y0), QPointF(x_end + 20.0, y0))

        # 3. Wave Envelope (Đường bao biên độ đối xứng)
        self._draw_wave_envelope(painter)

        # 4. Instantaneous Vibrating String (Sóng dừng chuyển động)
        self._draw_vibrating_string(painter)

        # 5. Fixed End Mechanical Clamps (Ngàm cố định đầu dây)
        self._draw_clamps(painter)

        # 6. Nodes & Antinodes (Nút sóng & Bụng sóng)
        self._draw_nodes_and_antinodes(painter)

        # 7. Wavelength Dimension Bracket (Thước đo bước sóng lambda)
        self._draw_wavelength_bracket(painter)

        # 8. Physical Theory & HUD Badge
        self._draw_hud_badge(painter)

        # 9. Control Buttons
        self._draw_buttons(painter)

        # 10. Draggable Amplitude Handle on first antinode
        self._draw_amplitude_handle(painter)

        painter.restore()

    def _draw_wave_envelope(self, painter: QPainter) -> None:
        """Draws the static symmetric upper and lower envelope curves."""
        x_start = self._x_string_start
        x_end = self._x_string_end
        y0 = self._y_baseline
        num_pts = 120
        dx = (x_end - x_start) / float(num_pts)

        path_upper = QPainterPath()
        path_lower = QPainterPath()
        path_upper.moveTo(x_start, y0)
        path_lower.moveTo(x_start, y0)

        for i in range(num_pts + 1):
            px = x_start + i * dx
            env = self.envelope_at(px)
            path_upper.lineTo(px, y0 - env)
            path_lower.lineTo(px, y0 + env)

        pen_env = QPen(QColor(6, 182, 212, 70), 1.6, Qt.PenStyle.DotLine)
        painter.setPen(pen_env)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_upper)
        painter.drawPath(path_lower)

    def _draw_vibrating_string(self, painter: QPainter) -> None:
        """Draws the live vibrating string curve with neon halo glow."""
        x_start = self._x_string_start
        x_end = self._x_string_end
        y0 = self._y_baseline
        num_pts = 140
        dx = (x_end - x_start) / float(num_pts)

        path_string = QPainterPath()
        path_string.moveTo(x_start, y0 + self.displacement_at(x_start, self._time))

        for i in range(1, num_pts + 1):
            px = x_start + i * dx
            py = y0 - self.displacement_at(px, self._time)
            path_string.lineTo(px, py)

        # Neon glow outer pass
        pen_glow = QPen(QColor(56, 189, 248, 85), 7.0, Qt.PenStyle.SolidLine)
        painter.setPen(pen_glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_string)

        # Crisp inner string pass
        pen_string = QPen(QColor("#38bdf8"), 2.8, Qt.PenStyle.SolidLine)
        painter.setPen(pen_string)
        painter.drawPath(path_string)

    def _draw_clamps(self, painter: QPainter) -> None:
        """Draws mechanical fixed clamp blocks at ends."""
        y0 = self._y_baseline
        x_start = self._x_string_start
        x_end = self._x_string_end
        clamp_w = 12.0
        clamp_h = 44.0

        # Left Clamp (Always fixed)
        rect_left = QRectF(x_start - clamp_w, y0 - clamp_h * 0.5, clamp_w, clamp_h)
        painter.setPen(QPen(QColor("#94a3b8"), 1.5))
        painter.setBrush(QBrush(QColor(71, 85, 105, 200)))
        painter.drawRoundedRect(rect_left, 3.0, 3.0)

        # Right End: Fixed clamp or Free ring
        if not self._is_fixed_free:
            rect_right = QRectF(x_end, y0 - clamp_h * 0.5, clamp_w, clamp_h)
            painter.setPen(QPen(QColor("#94a3b8"), 1.5))
            painter.setBrush(QBrush(QColor(71, 85, 105, 200)))
            painter.drawRoundedRect(rect_right, 3.0, 3.0)
        else:
            # Free ring slider on vertical rod
            painter.setPen(QPen(QColor("#cbd5e1"), 2.0))
            painter.drawLine(QPointF(x_end, y0 - clamp_h * 0.6), QPointF(x_end, y0 + clamp_h * 0.6))
            disp_end = self.displacement_at(x_end, self._time)
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.drawEllipse(QPointF(x_end, y0 - disp_end), 5.0, 5.0)

    def _draw_nodes_and_antinodes(self, painter: QPainter) -> None:
        """Renders circular markers for nodes (N) and antinodes (B)."""
        y0 = self._y_baseline
        f_marker = QFont("Segoe UI", 8)
        f_marker.setBold(True)
        painter.setFont(f_marker)

        # 1. Nodes (Nút sóng) - stationary points at baseline
        nodes = self.node_positions()
        painter.setPen(QPen(QColor("#ef4444"), 1.5))
        painter.setBrush(QBrush(QColor("#fee2e2")))
        for nx in nodes:
            painter.drawEllipse(QPointF(nx, y0), 4.5, 4.5)
            painter.setPen(QPen(QColor("#fca5a5")))
            painter.drawText(QPointF(nx - 4.0, y0 + 16.0), "N")
            painter.setPen(QPen(QColor("#ef4444"), 1.5))

        # 2. Antinodes (Bụng sóng) - points of peak amplitude
        antinodes = self.antinode_positions()
        for bx in antinodes:
            # Instantaneous position on vibrating string
            disp = self.displacement_at(bx, self._time)
            by = y0 - disp
            painter.setPen(QPen(QColor("#10b981"), 1.5))
            painter.setBrush(QBrush(QColor("#d1fae5")))
            painter.drawEllipse(QPointF(bx, by), 4.0, 4.0)

            # Antinode label below baseline
            painter.setPen(QPen(QColor("#6ee7b7")))
            painter.drawText(QPointF(bx - 3.5, y0 + 16.0), "B")

    def _draw_wavelength_bracket(self, painter: QPainter) -> None:
        """Draws a horizontal dimension bracket indicating wavelength lambda or lambda/2."""
        nodes = self.node_positions()
        if len(nodes) < 2:
            return

        x1 = nodes[0]
        # Show lambda if k >= 2 (spanning two loops), else show lambda/2 (one loop)
        if len(nodes) >= 3 and self._k >= 2:
            x2 = nodes[2]
            lbl = "Bước sóng λ"
        else:
            x2 = nodes[1]
            lbl = "λ / 2"

        bracket_y = self._y_baseline + self._amplitude + 32.0
        pen_dim = QPen(QColor("#f59e0b"), 1.4)
        painter.setPen(pen_dim)

        # Main horizontal bracket line
        painter.drawLine(QPointF(x1, bracket_y), QPointF(x2, bracket_y))
        # End ticks
        painter.drawLine(QPointF(x1, bracket_y - 6.0), QPointF(x1, bracket_y + 6.0))
        painter.drawLine(QPointF(x2, bracket_y - 6.0), QPointF(x2, bracket_y + 6.0))

        # Label text
        f_dim = QFont("Segoe UI", 8)
        f_dim.setBold(True)
        painter.setFont(f_dim)
        painter.setPen(QPen(QColor("#fde68a")))
        mid_x = (x1 + x2) * 0.5
        painter.drawText(QPointF(mid_x - 24.0, bracket_y - 4.0), lbl)

    def _draw_hud_badge(self, painter: QPainter) -> None:
        """Renders physics formulas, node/antinode count, and frequency information."""
        hud_w = 260.0
        hud_h = 80.0
        hud_x = 20.0
        hud_y = self._height - hud_h - 15.0
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.setPen(QPen(QColor(71, 85, 105, 180), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
        painter.drawRoundedRect(hud_rect, 8.0, 8.0)

        f_title = QFont("Segoe UI", 9)
        f_title.setBold(True)
        painter.setFont(f_title)
        painter.setPen(QPen(QColor("#38bdf8")))

        mode_name = "Hai đầu cố định" if not self._is_fixed_free else "Một đầu tự do"
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 6.0, hud_w - 16.0, 16.0), f"Sóng dừng: {mode_name}")

        f_vals = QFont("Segoe UI", 8)
        painter.setFont(f_vals)
        painter.setPen(QPen(QColor("#cbd5e1")))

        num_nodes, num_antis = self.count_nodes_and_antinodes()
        lam_px = self.wavelength()
        v_px = self.wave_speed()

        if not self._is_fixed_free:
            formula_str = f"L = k·(λ/2) = {self._k}·({lam_px:.1f}/2) px"
        else:
            formula_str = f"L = (2k+1)·(λ/4) = {2*self._k+1}·({lam_px:.1f}/4) px"

        painter.drawText(QRectF(hud_x + 8.0, hud_y + 24.0, hud_w - 16.0, 16.0), formula_str)
        line2 = f"Số nút: {num_nodes} (N) | Số bụng: {num_antis} (B)"
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 40.0, hud_w - 16.0, 16.0), line2)
        line3 = f"f = {self._frequency:.1f} Hz | v = λ·f = {v_px:.1f} px/s"
        painter.drawText(QRectF(hud_x + 8.0, hud_y + 56.0, hud_w - 16.0, 16.0), line3)

    def _draw_buttons(self, painter: QPainter) -> None:
        """Renders interactive buttons: harmonic selector, play/pause, boundary toggle, stamp."""
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
        play_txt = "⏸ Tạm dừng" if self._is_playing else "▶ Chạy"
        play_col = QColor("#f59e0b") if self._is_playing else QColor("#10b981")
        painter.setPen(QPen(play_col, 1.2))
        painter.setBrush(QBrush(QColor(play_col.red(), play_col.green(), play_col.blue(), 40)))
        painter.drawRoundedRect(self._rect_btn_play, 6.0, 6.0)
        painter.setPen(QPen(play_col))
        painter.drawText(self._rect_btn_play, Qt.AlignmentFlag.AlignCenter, play_txt)

        # Boundary mode toggle button
        b_txt = "2 Đầu cố định" if not self._is_fixed_free else "1 Đầu tự do"
        painter.setPen(QPen(QColor("#818cf8"), 1.2))
        painter.setBrush(QBrush(QColor(99, 102, 241, 40)))
        painter.drawRoundedRect(self._rect_btn_boundary, 6.0, 6.0)
        painter.setPen(QPen(QColor("#e0e7ff")))
        painter.drawText(self._rect_btn_boundary, Qt.AlignmentFlag.AlignCenter, f"Mode: {b_txt}")

        # Harmonic mode selector buttons [k=1]..[k=6]
        for i, rect in enumerate(self._rect_btn_harmonics):
            val_k = i + 1
            is_active = (val_k == self._k)
            if is_active:
                painter.setPen(QPen(QColor("#38bdf8"), 1.4))
                painter.setBrush(QBrush(QColor(56, 189, 248, 120)))
                painter.setPen(QPen(QColor("#ffffff")))
            else:
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
                painter.setPen(QPen(QColor("#94a3b8")))

            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"k={val_k}")

    def _draw_amplitude_handle(self, painter: QPainter) -> None:
        """Draws draggable circular handle on the first antinode to adjust amplitude A."""
        antinodes = self.antinode_positions()
        if not antinodes:
            return

        bx = antinodes[0]
        by = self._y_baseline - self._amplitude
        radius = 8.0 if self._active_handle != "AMP" else 10.5
        handle_pos = QPointF(bx, by)

        # Outer glow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(245, 158, 11, 70)))
        painter.drawEllipse(handle_pos, radius + 4.0, radius + 4.0)

        # Handle circle
        painter.setPen(QPen(QColor("#ffffff"), 1.8))
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.drawEllipse(handle_pos, radius, radius)

        # Tag
        f_amp = QFont("Segoe UI", 8)
        f_amp.setBold(True)
        painter.setFont(f_amp)
        painter.setPen(QPen(QColor("#fde68a")))
        painter.drawText(QPointF(bx + 12.0, by + 4.0), f"A={self._amplitude:.0f}px")

    # ─── Mouse Interactions ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles handle dragging or button clicks."""
        pos = event.pos()

        # Check button clicks
        if self._rect_btn_close.contains(pos):
            self.close()
            event.accept()
            return

        if self._rect_btn_play.contains(pos):
            self.toggle_play()
            event.accept()
            return

        if self._rect_btn_boundary.contains(pos):
            self.toggle_boundary_mode()
            event.accept()
            return

        # Check harmonic buttons
        for i, rect in enumerate(self._rect_btn_harmonics):
            if rect.contains(pos):
                self.set_harmonic_k(i + 1)
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

        # Check amplitude handle
        antinodes = self.antinode_positions()
        if antinodes:
            bx = antinodes[0]
            by = self._y_baseline - self._amplitude
            if (pos - QPointF(bx, by)).manhattanLength() < 22.0:
                self._active_handle = "AMP"
                event.accept()
                return

        # Background drag
        self._active_handle = "BODY"
        self._drag_start_item_pos = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Updates amplitude during dragging."""
        pos = event.pos()

        if self._active_handle == "AMP":
            new_amp = self._y_baseline - pos.y()
            self.set_amplitude(new_amp)
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
        """Touchless air pinch drag of amplitude handle or entire item."""
        pos = self.mapFromScene(scene_pos)

        if self._active_handle is None:
            antinodes = self.antinode_positions()
            if antinodes:
                bx = antinodes[0]
                by = self._y_baseline - self._amplitude
                if (pos - QPointF(bx, by)).manhattanLength() < 28.0:
                    self._active_handle = "AMP"
                else:
                    self._active_handle = "BODY"
                    self._drag_start_item_pos = pos

        if self._active_handle == "AMP":
            new_amp = self._y_baseline - pos.y()
            self.set_amplitude(new_amp)
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

        if self._rect_btn_play.contains(pos):
            self.toggle_play()
            return True

        if self._rect_btn_boundary.contains(pos):
            self.toggle_boundary_mode()
            return True

        for i, rect in enumerate(self._rect_btn_harmonics):
            if rect.contains(pos):
                self.set_harmonic_k(i + 1)
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
        Permanently stamps crisp vector standing wave lines onto LessonBoard:
          - Equilibrium baseline
          - Symmetric envelope curves
          - Instantaneous vibrating string profile
          - Node & antinode markers
          - Physics equation and harmonic explanation text card
        """
        stamped: list[QGraphicsItem] = []
        origin = self.pos()
        y0 = origin.y() + self._y_baseline
        x_start = origin.x() + self._x_string_start
        x_end = origin.x() + self._x_string_end

        # 1. Equilibrium Line
        line_base = QGraphicsLineItem(x_start, y0, x_end, y0)
        line_base.setPen(QPen(QColor("#64748b"), 1.4, Qt.PenStyle.DashLine))
        line_base.setZValue(10.0)
        board.addItem(line_base)
        stamped.append(line_base)

        # 2. Symmetric Envelope Curves
        num_pts = 100
        dx = (x_end - x_start) / float(num_pts)
        p_up = QPainterPath()
        p_dn = QPainterPath()
        p_up.moveTo(x_start, y0)
        p_dn.moveTo(x_start, y0)

        for i in range(num_pts + 1):
            px_local = self._x_string_start + i * dx
            px_scene = origin.x() + px_local
            env = self.envelope_at(px_local)
            p_up.lineTo(px_scene, y0 - env)
            p_dn.lineTo(px_scene, y0 + env)

        pen_env = QPen(QColor(6, 182, 212, 160), 1.6, Qt.PenStyle.DotLine)
        item_up = QGraphicsPathItem(p_up)
        item_up.setPen(pen_env)
        item_up.setZValue(10.5)
        board.addItem(item_up)
        stamped.append(item_up)

        item_dn = QGraphicsPathItem(p_dn)
        item_dn.setPen(pen_env)
        item_dn.setZValue(10.5)
        board.addItem(item_dn)
        stamped.append(item_dn)

        # 3. Wave Profile (Current Snapshot)
        p_wave = QPainterPath()
        p_wave.moveTo(x_start, y0 - self.displacement_at(self._x_string_start, self._time))
        for i in range(1, num_pts + 1):
            px_local = self._x_string_start + i * dx
            px_scene = origin.x() + px_local
            disp = self.displacement_at(px_local, self._time)
            p_wave.lineTo(px_scene, y0 - disp)

        pen_wave = QPen(QColor("#38bdf8"), 2.6, Qt.PenStyle.SolidLine)
        item_wave = QGraphicsPathItem(p_wave)
        item_wave.setPen(pen_wave)
        item_wave.setZValue(11.0)
        board.addItem(item_wave)
        stamped.append(item_wave)

        # 4. Text Card with Standing Wave Formulas
        nodes = self.node_positions()
        antis = self.antinode_positions()
        mode_str = "Hai đầu cố định" if not self._is_fixed_free else "Một đầu tự do"
        lam = self.wavelength()

        lbl_text = (
            f"Hiện Tượng Sóng Dừng Trên Dây ({mode_str}):\n"
            f"Điều kiện sóng dừng: L = {self._k}·(λ/2) [λ = {lam:.1f} px]\n"
            f"Họa âm bậc: k = {self._k} | Biên độ bụng: 2A = {2*self._amplitude:.1f} px\n"
            f"Số nút sóng: {len(nodes)} (N) | Số bụng sóng: {len(antis)} (B)\n"
            f"Tần số: f = {self._frequency:.1f} Hz | Vận tốc truyền sóng: v = {self.wave_speed():.1f} px/s"
        )
        txt_item = QGraphicsTextItem(lbl_text)
        txt_font = QFont("Segoe UI", 10)
        txt_font.setBold(True)
        txt_item.setFont(txt_font)
        txt_item.setDefaultTextColor(QColor("#f8fafc"))
        txt_item.setPos(x_start, y0 + self._amplitude + 24.0)
        txt_item.setZValue(12.0)
        board.addItem(txt_item)
        stamped.append(txt_item)

        return stamped
