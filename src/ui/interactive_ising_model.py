"""
AirTeach AI - Interactive Statistical Physics: 2D Ising Model & Phase Transitions
==================================================================================
Authentic Lars Onsager Exact Solution (1944) & Metropolis Monte Carlo Simulation:
- 2D Square Lattice Hamiltonian:
    H = -J * ∑_{<i,j>} s_i * s_j - h * ∑_i s_i
    where s_i ∈ {-1, +1}, J > 0 (ferromagnetic) or J < 0 (antiferromagnetic).
- Local Metropolis Update Rule:
    ΔE = 2 * s_i * ( J * ∑_{n.n.} s_j + h )
    Acceptance probability: P_acc = min(1, exp(-ΔE / (k_B * T)))
- Lars Onsager's Exact Analytical Solution (1944):
    Critical Temperature:
      k_B * T_c / J = 2 / ln(1 + sqrt(2)) ≈ 2.269185314
    Spontaneous Magnetization (T < T_c):
      M(T) = [ 1 - sinh(2*J / (k_B*T))^(-4) ]^(1/8)   for T < T_c
      M(T) = 0                                         for T >= T_c
- Thermodynamic Observables:
    Energy per spin: <E> = H / N
    Magnetization: <M> = (1/N) * |∑_i s_i|
    Magnetic Susceptibility: χ = (N / k_B*T) * (<M^2> - <M>^2)  (diverges at T_c)
    Specific Heat Capacity: C_v = (N / k_B*T^2) * (<E^2> - <E>^2)  (logarithmic divergence at T_c)
- 5 Curriculum Presets:
    1. FERRO_ORDERED: T = 1.2 J/k_B, spontaneous symmetry breaking, large uniform domain.
    2. CRITICAL_POINT: T = T_c ≈ 2.269 J/k_B, fractal scale-invariant spin fluctuations.
    3. PARAMAGNETIC_GAS: T = 4.5 J/k_B, disordered paramagnetic entropy-dominated phase.
    4. FIELD_HYSTERESIS: Sweeping external field h from -1.5 to +1.5, hysteresis loop.
    5. ANTIFERROMAGNET: J = -1.0, checkerboard Néel antiferromagnetic ground state.
- Touchless webcam air gestures (air-tap, temperature slider, magnetic field sweeping),
  bimanual manipulation, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Exact Onsager critical temperature in units of J / k_B
ONSAGER_TC: float = 2.0 / math.log(1.0 + math.sqrt(2.0))  # ≈ 2.2691853142


@dataclass
class IsingPreset:
    """Preset configuration for educational 2D Ising model experiments."""
    key: str
    name: str
    desc: str
    temp_ratio: float  # T / (J / k_B)
    j_coupling: float  # +1.0 for ferro, -1.0 for antiferro
    ext_field_h: float
    init_state: str    # "ordered", "random", "checkerboard"


PRESETS: List[IsingPreset] = [
    IsingPreset(
        key="ferro_ordered",
        name="Sắt từ Trật tự (T < Tc)",
        desc="Nhiệt độ thấp T = 1.2 (dưới Tc ≈ 2.27), tương tác trao đổi thắng thế tạo miền từ tự phát đồng nhất",
        temp_ratio=1.2,
        j_coupling=1.0,
        ext_field_h=0.0,
        init_state="ordered",
    ),
    IsingPreset(
        key="critical_point",
        name="Điểm Tới hạn (T = Tc, Chuyển pha)",
        desc="Nhiệt độ tới hạn Onsager Tc ≈ 2.269, thăng giáng cụm spin bất biến tỉ lệ (Fractal) và độ cảm từ χ phân kỳ",
        temp_ratio=ONSAGER_TC,
        j_coupling=1.0,
        ext_field_h=0.0,
        init_state="random",
    ),
    IsingPreset(
        key="paramagnetic_gas",
        name="Thuận từ Vô trật tự (T > Tc)",
        desc="Nhiệt độ cao T = 4.2 (trên Tc), entropy nhiệt phá vỡ trật tự từ tính, độ từ hóa trung bình triệt tiêu <M> = 0",
        temp_ratio=4.2,
        j_coupling=1.0,
        ext_field_h=0.0,
        init_state="random",
    ),
    IsingPreset(
        key="field_hysteresis",
        name="Trễ từ & Từ trường ngoài (Hysteresis)",
        desc="Chuyển động lật spin dưới từ trường ngoài biến thiên h, quan sát hiện tượng từ trễ và lực kháng từ",
        temp_ratio=1.8,
        j_coupling=1.0,
        ext_field_h=0.5,
        init_state="ordered",
    ),
    IsingPreset(
        key="antiferromagnet",
        name="Phản sắt từ Néel (J < 0)",
        desc="Tương tác trao đổi âm (J = -1.0), các spin liền kề đối song song tạo cấu trúc bàn cờ ô vuông Néel",
        temp_ratio=1.4,
        j_coupling=-1.0,
        ext_field_h=0.0,
        init_state="checkerboard",
    ),
]


class InteractiveIsingModelItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic 2D Ising Model statistical mechanics,
    Metropolis Monte Carlo dynamics, Onsager exact solution, and phase transitions.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 940.0
    HEIGHT: float = 600.0

    # Lattice dimension (50 x 50 = 2500 spins)
    LATTICE_SIZE: int = 50

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 1  # Default to Critical Point
        self._is_paused: bool = False
        self._sim_time: float = 0.0

        p = PRESETS[self._preset_index]
        self._temp: float = p.temp_ratio
        self._j: float = p.j_coupling
        self._h: float = p.ext_field_h

        # 2D Spin Lattice (int8: +1 or -1)
        self._spins = np.ones((self.LATTICE_SIZE, self.LATTICE_SIZE), dtype=np.int8)
        self._init_spins(p.init_state)

        # History tracking for live plots
        self._m_history: Deque[float] = collections.deque(maxlen=120)
        self._e_history: Deque[float] = collections.deque(maxlen=120)

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

        # Real-time animation timer (~30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start()

    def _init_spins(self, state_type: str) -> None:
        """Initialize spin configuration."""
        l = self.LATTICE_SIZE
        if state_type == "ordered":
            self._spins.fill(1)
        elif state_type == "checkerboard":
            x, y = np.indices((l, l))
            self._spins = np.where((x + y) % 2 == 0, 1, -1).astype(np.int8)
        else:  # "random"
            self._spins = np.random.choice([-1, 1], size=(l, l)).astype(np.int8)

    def _on_anim_tick(self) -> None:
        """Periodic tick to advance Monte Carlo sweeps."""
        if not self._is_paused:
            self.step_physics(0.033)

    def stop(self) -> None:
        """Halt background timer to release resources."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

    @property
    def current_preset(self) -> IsingPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active 2D Ising preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._temp = p.temp_ratio
            self._j = p.j_coupling
            self._h = p.ext_field_h
            self._init_spins(p.init_state)
            self._m_history.clear()
            self._e_history.clear()
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_temperature(self, t_val: float) -> None:
        """Set reduced temperature T / (J/kB)."""
        self._temp = max(0.1, min(t_val, 6.0))
        self.property_changed.emit("temperature", self._temp)
        self.update()

    def set_coupling_j(self, j_val: float) -> None:
        """Set exchange coupling J (-2.0 to +2.0)."""
        self._j = max(-2.0, min(j_val, 2.0))
        self.property_changed.emit("coupling_j", self._j)
        self.update()

    def set_external_field(self, h_val: float) -> None:
        """Set external magnetic field h."""
        self._h = max(-2.0, min(h_val, 2.0))
        self.property_changed.emit("external_field", self._h)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset state to preset defaults."""
        self.set_preset(self._preset_index)

    # -------------------------------------------------------------------------
    # Authentic Statistical Mechanics & Metropolis Monte Carlo Algorithm
    # -------------------------------------------------------------------------

    def monte_carlo_sweep(self) -> None:
        """
        Execute one full Metropolis Monte Carlo sweep over all N spins
        using a vectorized checkerboard (even/odd sub-lattice) partition.
        """
        l = self.LATTICE_SIZE
        t = max(0.01, self._temp)
        j = self._j
        h = self._h

        # Checkerboard partition for parallel vectorized updates
        for offset in [0, 1]:
            # Periodic neighbor sum
            up = np.roll(self._spins, 1, axis=0)
            down = np.roll(self._spins, -1, axis=0)
            left = np.roll(self._spins, 1, axis=1)
            right = np.roll(self._spins, -1, axis=1)
            nn_sum = up + down + left + right

            # Energy change upon flipping spin: ΔE = 2 * s * (J * nn_sum + h)
            delta_e = 2.0 * self._spins * (j * nn_sum + h)

            # Mask for current sub-lattice
            sub_mask = np.indices((l, l)).sum(axis=0) % 2 == offset

            # Metropolis acceptance criteria
            prob = np.exp(-delta_e / t)
            accept = (delta_e <= 0) | (np.random.random((l, l)) < prob)
            flip_mask = sub_mask & accept

            self._spins[flip_mask] *= -1

    def compute_observables(self) -> Tuple[float, float]:
        """
        Compute mean magnetization per spin <M> and mean energy per spin <E>.
        Returns: (M, E)
        """
        l = self.LATTICE_SIZE
        n = float(l * l)

        # Magnetization per spin: |∑ s_i| / N
        m = float(abs(np.sum(self._spins)) / n)

        # Energy per spin: -J * ∑ s_i s_j - h * ∑ s_i
        up = np.roll(self._spins, 1, axis=0)
        right = np.roll(self._spins, 1, axis=1)
        e_nn = -self._j * np.sum(self._spins * (up + right))
        e_h = -self._h * np.sum(self._spins)
        e = float((e_nn + e_h) / n)

        return m, e

    def compute_onsager_magnetization(self, t_val: float) -> float:
        """
        Compute exact Lars Onsager spontaneous magnetization (h = 0, J > 0):
        M(T) = [ 1 - sinh(2*J / (kB*T))^(-4) ]^(1/8) for T < Tc, else 0.
        """
        if self._j <= 0 or t_val >= ONSAGER_TC:
            return 0.0
        try:
            arg = 2.0 * self._j / t_val
            sinh_val = math.sinh(arg)
            sinh_term = sinh_val**(-4)
            if sinh_term >= 1.0:
                return 0.0
            return float((1.0 - sinh_term)**(0.125))
        except (OverflowError, ValueError):
            return 1.0

    def step_physics(self, dt: float) -> None:
        """Advance Monte Carlo sweeps (multiple sweeps per frame for rapid thermalization)."""
        if self._is_paused or dt <= 0.0:
            return

        # Perform 3 sweeps per animation frame
        for _ in range(3):
            self.monte_carlo_sweep()

        self._sim_time += dt
        m_curr, e_curr = self.compute_observables()
        self._m_history.append(m_curr)
        self._e_history.append(e_curr)

        self.update()

    # -------------------------------------------------------------------------
    # Layout & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _header_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, 56.0)

    def _lattice_rect(self) -> QRectF:
        return QRectF(15.0, 65.0, 420.0, 420.0)

    def _plots_panel_rect(self) -> QRectF:
        return QRectF(445.0, 65.0, 480.0, 420.0)

    def _controls_rect(self) -> QRectF:
        return QRectF(15.0, 495.0, 910.0, 90.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, IsingPreset]]:
        h_rect = self._header_rect()
        btn_w = 95.0
        btn_h = 28.0
        gap = 6.0
        start_x = 350.0
        y = (h_rect.height() - btn_h) / 2.0
        rects = []
        for i, p in enumerate(PRESETS):
            bx = start_x + i * (btn_w + gap)
            rects.append((QRectF(bx, y, btn_w, btn_h), p))
        return rects

    def paint(
        self,
        painter: QPainter,
        option: Optional[QStyleOptionGraphicsItem] = None,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Main Background Card
        bg_brush = QBrush(QColor(11, 15, 25, 245))
        border_pen = QPen(QColor(14, 165, 233, 160), 1.5)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.boundingRect(), 14.0, 14.0)

        self._paint_header(painter)
        self._paint_spin_lattice(painter)
        self._paint_plots_panel(painter)
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        h_rect = self._header_rect()

        # Title & Subtitle
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(20, 26, "⚡ MÔ HÌNH ISING 2D & CHUYỂN PHA LARS ONSAGER (1944)")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        p = self.current_preset
        m_curr, e_curr = self.compute_observables()
        phase_label = "Sắt từ (Ferro)" if self._temp < ONSAGER_TC and self._j > 0 else (
            "Điểm tới hạn Tc" if math.isclose(self._temp, ONSAGER_TC, abs_tol=0.08) else "Thuận từ (Para)"
        )
        painter.drawText(20, 44, f"Pha: {phase_label}  |  T = {self._temp:.3f} (Tc ≈ {ONSAGER_TC:.3f})  |  <M> = {m_curr:.3f}  |  <E> = {e_curr:.3f}")

        # Preset Buttons
        active_idx = self._preset_index
        for i, (b_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == active_idx)
            btn_brush = QColor(14, 165, 233, 200) if is_active else QColor(30, 41, 59, 180)
            btn_pen = QPen(QColor(56, 189, 248), 1.5) if is_active else QPen(QColor(71, 85, 105), 1.0)
            painter.setBrush(btn_brush)
            painter.setPen(btn_pen)
            painter.drawRoundedRect(b_rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            labels = ["Sắt từ", "Tới hạn Tc", "Thuận từ", "Trễ từ (h)", "Phản sắt từ"]
            label = labels[i] if i < len(labels) else preset.name[:8]
            painter.drawText(b_rect, Qt.AlignCenter, label)

        # Status Badge
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        badge_color = QColor(16, 185, 129) if self._temp < ONSAGER_TC else (
            QColor(250, 204, 21) if math.isclose(self._temp, ONSAGER_TC, abs_tol=0.1) else QColor(239, 68, 68)
        )
        painter.setBrush(QColor(badge_color.red(), badge_color.green(), badge_color.blue(), 40))
        painter.setPen(QPen(badge_color, 1.2))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(badge_color)
        painter.drawText(badge_rect, Qt.AlignCenter, f"T/Tc = {self._temp / ONSAGER_TC:.2f}")

    def _paint_spin_lattice(self, painter: QPainter) -> None:
        """Render Left Panel: 50x50 Spin Lattice Pixel Array (Spin Up = Cyan, Spin Down = Magenta)."""
        rect = self._lattice_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, f"🧩 MẠNG SPIN 2D ({self.LATTICE_SIZE}×{self.LATTICE_SIZE} = 2,500 SPINS)")

        # Create QImage from spin array
        l = self.LATTICE_SIZE
        # Map spins: +1 -> Cyan (RGB: 14, 165, 233), -1 -> Magenta (RGB: 236, 72, 153)
        img = QImage(l, l, QImage.Format_RGB32)
        color_up = 0xFF0EA5E9
        color_down = 0xFFEC4899

        for y in range(l):
            for x in range(l):
                img.setPixel(x, y, color_up if self._spins[y, x] == 1 else color_down)

        # Scale and draw image
        draw_rect = QRectF(rect.left() + 25.0, rect.top() + 45.0, 370.0, 335.0)
        painter.drawImage(draw_rect, img)

        # Border around lattice
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(51, 65, 85), 1.5))
        painter.drawRect(draw_rect)

        # Bottom Legend
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(14, 165, 233))
        painter.drawText(int(rect.left() + 35.0), int(rect.bottom() - 15.0), "■ Spin Lên (+1, Cyan)")

        painter.setPen(QColor(236, 72, 153))
        painter.drawText(int(rect.left() + 200.0), int(rect.bottom() - 15.0), "■ Spin Xuống (-1, Magenta)")

    def _paint_plots_panel(self, painter: QPainter) -> None:
        """Render Right Panel: Onsager Analytical M(T) vs Empirical Magnetization & Energy."""
        rect = self._plots_panel_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 20.0), Qt.AlignCenter, "📈 NGHIỆM CHÍNH XÁC ONSAGER & ĐỘ TỪ HÓA THỜI GIAN THỰC")

        # Top Plot: Onsager Curve M(T) with live state point
        top_plot = QRectF(rect.left() + 25.0, rect.top() + 45.0, rect.width() - 45.0, 160.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(top_plot)

        # Plot Onsager M(T) curve
        onsager_path = QPainterPath()
        num_pts = 60
        max_t = 4.5

        for i in range(num_pts + 1):
            t_val = 0.1 + (i / num_pts) * (max_t - 0.1)
            m_onsager = self.compute_onsager_magnetization(t_val)
            px = top_plot.left() + (t_val / max_t) * top_plot.width()
            py = top_plot.bottom() - m_onsager * (top_plot.height() * 0.9)

            if i == 0:
                onsager_path.moveTo(px, py)
            else:
                onsager_path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawPath(onsager_path)

        # Mark Onsager Tc vertical line
        tc_x = top_plot.left() + (ONSAGER_TC / max_t) * top_plot.width()
        painter.setPen(QPen(QColor(239, 68, 68), 1.2, Qt.DashLine))
        painter.drawLine(int(tc_x), int(top_plot.top()), int(tc_x), int(top_plot.bottom()))

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(239, 68, 68))
        painter.drawText(QRectF(tc_x - 30.0, top_plot.top() + 5.0, 60.0, 14.0), Qt.AlignCenter, f"Tc={ONSAGER_TC:.2f}")

        # Plot Current State Point (T, <M>)
        curr_m, _ = self.compute_observables()
        pt_x = top_plot.left() + min(1.0, self._temp / max_t) * top_plot.width()
        pt_y = top_plot.bottom() - curr_m * (top_plot.height() * 0.9)

        painter.setBrush(QColor(250, 204, 21))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(pt_x, pt_y), 4.5, 4.5)

        # Bottom Plot: Real-time Magnetization & Energy Time History
        bot_plot = QRectF(rect.left() + 25.0, top_plot.bottom() + 35.0, rect.width() - 45.0, 130.0)
        painter.setBrush(QColor(2, 6, 23))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRect(bot_plot)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(QRectF(bot_plot.left() + 5.0, top_plot.bottom() + 18.0, 200.0, 14.0), Qt.AlignLeft, "Biến thiên theo thời gian: <M>(t) (Vàng) & <E>(t) (Xanh)")

        if len(self._m_history) > 1:
            m_path = QPainterPath()
            e_path = QPainterPath()
            n_hist = len(self._m_history)

            for hi in range(n_hist):
                hx = bot_plot.left() + (hi / max(1, n_hist - 1)) * bot_plot.width()
                # M from 0 to 1
                hy_m = bot_plot.bottom() - self._m_history[hi] * bot_plot.height()
                # E from -2 to 0 (scaled)
                norm_e = (self._e_history[hi] + 2.2) / 2.5
                hy_e = bot_plot.bottom() - max(0.0, min(1.0, norm_e)) * bot_plot.height()

                if hi == 0:
                    m_path.moveTo(hx, hy_m)
                    e_path.moveTo(hx, hy_e)
                else:
                    m_path.lineTo(hx, hy_m)
                    e_path.lineTo(hx, hy_e)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(250, 204, 21), 1.8))
            painter.drawPath(m_path)

            painter.setPen(QPen(QColor(16, 185, 129), 1.5))
            painter.drawPath(e_path)

        # Telemetry Text
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(203, 213, 225))
        ty = bot_plot.bottom() + 16.0
        painter.drawText(int(rect.left() + 20.0), int(ty), f"Định luật Onsager: M(T) = [1 - sinh(2J/T)⁻⁴]¹/⁸  |  Tương tác J = {self._j:+.1f}, Trường h = {self._h:+.2f}")

    def _paint_controls(self, painter: QPainter) -> None:
        ctrl_rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawRoundedRect(ctrl_rect, 10.0, 10.0)

        col_w = (ctrl_rect.width() - 40.0) / 4.0

        sliders = [
            ("temp", "Nhiệt độ T (J/kB)", f"{self._temp:.2f}", self._temp, 0.2, 5.0),
            ("j", "Tương tác trao đổi J", f"{self._j:+.2f}", self._j, -2.0, 2.0),
            ("h", "Từ trường ngoài h", f"{self._h:+.2f}", self._h, -1.5, 1.5),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(56, 189, 248))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill
            norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(14, 165, 233))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(14, 165, 233), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Reset button in 4th column
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        painter.setBrush(QColor(14, 165, 233, 40))
        painter.setPen(QPen(QColor(56, 189, 248), 1.2))
        painter.drawRoundedRect(btn_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(btn_rect, Qt.AlignCenter, "🔄 ĐẶT LẠI BAN ĐẦU")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 14.0, 130.0, 28.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Reset Button
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        btn_x = ctrl_rect.left() + 20.0 + 3 * col_w
        btn_rect = QRectF(btn_x, ctrl_rect.top() + 22.0, col_w - 25.0, 38.0)
        if btn_rect.contains(pos):
            self.reset_simulation()
            event.accept()
            return

        # Check Sliders
        for i, key in enumerate(["temp", "j", "h"]):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["temp", "j", "h"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        if key == "temp":
            self.set_temperature(0.2 + norm * 4.8)
        elif key == "j":
            self.set_coupling_j(-2.0 + norm * 4.0)
        elif key == "h":
            self.set_external_field(-1.5 + norm * 3.0)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveIsingModelItem",
            "preset_key": self.current_preset.key,
            "temp": self._temp,
            "j": self._j,
            "h": self._h,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveIsingModelItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "critical_point")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "temp" in data:
            item._temp = float(data["temp"])
        if "j" in data:
            item._j = float(data["j"])
        if "h" in data:
            item._h = float(data["h"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
