"""
AirTeach AI – Interactive Biophysics & Physiology: Hodgkin-Huxley Giant Squid Axon Action Potential
===================================================================================================
Provides an authentic biophysical simulation of the Nobel-Prize winning Hodgkin-Huxley model (1952/1963)
of action potential generation in the giant axon of the squid (Loligo pealeii):
  - Membrane Capacitance & Current Balance:
      C_m * (dV/dt) = I_inj(t) - I_Na - I_K - I_L
  - Voltage-Gated Ionic Currents:
      I_Na = g_Na_max * m³ * h * (V - E_Na)    (E_Na = +50.0 mV, g_Na_max = 120.0 mS/cm²)
      I_K  = g_K_max  * n⁴     * (V - E_K)     (E_K  = -77.0 mV, g_K_max  = 36.0 mS/cm²)
      I_L  = g_L      * (V - E_L)              (E_L  = -54.4 mV, g_L      = 0.3 mS/cm²)
  - Gating Kinetics (m, h, n):
      dm/dt = α_m(v) * (1 - m) - β_m(v) * m
      dh/dt = α_h(v) * (1 - h) - β_h(v) * h
      dn/dt = α_n(v) * (1 - n) - β_n(v) * n
      where v = V - V_rest (depolarization from resting V_rest = -65.0 mV).
  - Rate Coefficients α(v) and β(v):
      α_m(v) = 0.1 * (25 - v) / (exp((25 - v) / 10) - 1)
      β_m(v) = 4.0 * exp(-v / 18)
      α_h(v) = 0.07 * exp(-v / 20)
      β_h(v) = 1.0 / (exp((30 - v) / 10) + 1)
      α_n(v) = 0.01 * (10 - v) / (exp((10 - v) / 10) - 1)
      β_n(v) = 0.125 * exp(-v / 80)
  - Temperature Scaling (Q10 = 3.0):
      phi = 3.0 ** ((T_celsius - 6.3) / 10.0)
  - 5 Authentic Physiological Presets:
      1. STANDARD_ACTION_POTENTIAL: Single 10 μA/cm² pulse producing classic 110 mV spike.
      2. SUBTHRESHOLD_STIMULATION: Weak 3 μA/cm² stimulus decaying without firing (All-or-None).
      3. REPETITIVE_SPIKE_TRAIN: Constant 12 μA/cm² current generating rhythmic train (~65 Hz).
      4. TETRODOTOXIN_TTX_BLOCK: Pufferfish toxin blocking Na⁺ channels (g_Na -> 0).
      5. REFRACTORY_PERIOD_PROBING: Paired pulses demonstrating absolute and relative refractory windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
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


# Physical / Biophysical Constants
C_MEMBRANE = 1.0            # Membrane capacitance (μF/cm²)
V_REST = -65.0              # Resting membrane potential (mV)
E_NA = 50.0                 # Sodium Nernst equilibrium potential (mV)
E_K = -77.0                 # Potassium Nernst equilibrium potential (mV)
E_LEAK = -54.4              # Leak equilibrium potential (mV)
G_NA_MAX_BASE = 120.0       # Max Na⁺ conductance (mS/cm²)
G_K_MAX_BASE = 36.0         # Max K⁺ conductance (mS/cm²)
G_LEAK_BASE = 0.3           # Leak conductance (mS/cm²)


class Neurotoxin(str, Enum):
    NONE = "none"
    TTX = "ttx"             # Tetrodotoxin: blocks voltage-gated Na⁺ channels
    TEA = "tea"             # Tetraethylammonium: blocks voltage-gated K⁺ channels
    LIDOCAINE = "lidocaine" # Local anesthetic: partial Na⁺ block & threshold elevation


@dataclass(frozen=True)
class ActionPotentialPreset:
    code: str
    title_vi: str
    badge_label: str
    i_inj_ua: float
    pulse_duration_ms: float
    is_sustained: bool
    is_paired: bool
    paired_delay_ms: float
    toxin: Neurotoxin
    description: str


PRESETS: dict[str, ActionPotentialPreset] = {
    "STANDARD_ACTION_POTENTIAL": ActionPotentialPreset(
        code="STANDARD_ACTION_POTENTIAL",
        title_vi="Điện thế Hoạt động Điển hình (Xung Đơn Kích thích)",
        badge_label="ĐIỆN THẾ HOẠT ĐỘNG",
        i_inj_ua=10.0,
        pulse_duration_ms=1.5,
        is_sustained=False,
        is_paired=False,
        paired_delay_ms=0.0,
        toxin=Neurotoxin.NONE,
        description="Xung kích thích 10 μA/cm² vượt ngưỡng (-55 mV) kích hoạt chu trình Hodgkin: kênh Na⁺ mở ồ ạt gây khử cực và đảo cực tới +30 mV, theo sau bởi tái phân cực và ưu phân cực.",
    ),
    "SUBTHRESHOLD_STIMULATION": ActionPotentialPreset(
        code="SUBTHRESHOLD_STIMULATION",
        title_vi="Kích thích Dưới ngưỡng (Quy luật Tất cả hoặc Không)",
        badge_label="DƯỚI NGƯỠNG (KHÔNG PHÁT XUNG)",
        i_inj_ua=3.0,
        pulse_duration_ms=1.5,
        is_sustained=False,
        is_paired=False,
        paired_delay_ms=0.0,
        toxin=Neurotoxin.NONE,
        description="Xung 3 μA/cm² không đủ nâng điện thế màng lên ngưỡng kích hoạt (-55 mV). Điện thế biến thiên thụ động và tắt dần về điện thế nghỉ theo hàm mũ.",
    ),
    "REPETITIVE_SPIKE_TRAIN": ActionPotentialPreset(
        code="REPETITIVE_SPIKE_TRAIN",
        title_vi="Chuỗi Xung Nhịp nhàng (Kích thích Duy trì Liên tục)",
        badge_label="CHUỖI XUNG NHỊP (~65 HZ)",
        i_inj_ua=12.0,
        pulse_duration_ms=40.0,
        is_sustained=True,
        is_paired=False,
        paired_delay_ms=0.0,
        toxin=Neurotoxin.NONE,
        description="Dòng điện kích thích liên tục duy trì trạng thái khử cực, tạo chuỗi xung thần kinh nhịp nhàng với tần số tỷ lệ thuận với cường độ dòng điện (Mã hóa tần số nơron).",
    ),
    "TETRODOTOXIN_TTX_BLOCK": ActionPotentialPreset(
        code="TETRODOTOXIN_TTX_BLOCK",
        title_vi="Nhiễm độc Tetrodotoxin (Độc tố Cá nóc Khóa Kênh Na⁺)",
        badge_label="KHÓA KÊNH NA⁺ (TTX)",
        i_inj_ua=15.0,
        pulse_duration_ms=2.0,
        is_sustained=False,
        is_paired=False,
        paired_delay_ms=0.0,
        toxin=Neurotoxin.TTX,
        description="Độc tố TTX gắn chọn lọc vào lỗ ngoài kênh Na⁺ phụ thuộc điện thế (g_Na = 0). Nơron mất hoàn toàn khả năng phát xung hoạt động dù kích thích cực mạnh.",
    ),
    "REFRACTORY_PERIOD_PROBING": ActionPotentialPreset(
        code="REFRACTORY_PERIOD_PROBING",
        title_vi="Khảo sát Giai đoạn Trơ Tuyệt đối & Trơ Tương đối (Xung Đôi)",
        badge_label="GIAI ĐOẠN TRƠ (PAIRED PULSES)",
        i_inj_ua=10.0,
        pulse_duration_ms=1.2,
        is_sustained=False,
        is_paired=True,
        paired_delay_ms=4.0,
        toxin=Neurotoxin.NONE,
        description="Phát hai xung kích thích cách nhau Δt. Ở giai đoạn trơ tuyệt đối (kênh Na⁺ bất hoạt h ≈ 0), xung thứ hai hoàn toàn không phát sinh được điện thế hoạt động.",
    ),
}


class InteractiveActionPotentialItem(QGraphicsObject):
    """
    Interactive Hodgkin-Huxley Action Potential Simulator.
    Provides numerical integration of membrane potential Vm(t), gating kinetics (m, h, n),
    ionic conductances (gNa, gK), and neuropharmacological channel blocks.
    """

    preset_changed = Signal(str)
    voltage_updated = Signal(float)
    stamp_requested = Signal(dict)

    CARD_W = 940.0
    CARD_H = 630.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation State
        self._active_preset_code: str = "STANDARD_ACTION_POTENTIAL"
        self._preset: ActionPotentialPreset = PRESETS[self._active_preset_code]
        self._toxin: Neurotoxin = self._preset.toxin
        self._temp_c: float = 6.3            # Squid physiological baseline temperature

        self._i_inj_ua: float = self._preset.i_inj_ua
        self._pulse_duration_ms: float = self._preset.pulse_duration_ms
        self._is_sustained: bool = self._preset.is_sustained
        self._is_paired: bool = self._preset.is_paired
        self._paired_delay_ms: float = self._preset.paired_delay_ms

        # State Variables
        self._vm: float = V_REST
        self._m: float = 0.05
        self._h: float = 0.60
        self._n: float = 0.32
        self._time_ms: float = 0.0

        # Trace buffers for oscilloscope rendering (50 ms window)
        self._time_history: list[float] = []
        self._vm_history: list[float] = []
        self._gna_history: list[float] = []
        self._gk_history: list[float] = []

        # Playback control
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Run numerical integration for initial 50 ms window
        self._simulate_initial_window()

        # Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Hodgkin-Huxley Numerical Mathematics
    # -------------------------------------------------------------------------
    @property
    def vm(self) -> float:
        return self._vm

    @property
    def active_preset(self) -> ActionPotentialPreset:
        return self._preset

    @property
    def toxin(self) -> Neurotoxin:
        return self._toxin

    @property
    def temp_c(self) -> float:
        return self._temp_c

    def _alpha_m(self, v: float) -> float:
        val = 25.0 - v
        if abs(val) < 1e-6:
            return 1.0
        return 0.1 * val / (math.exp(val / 10.0) - 1.0)

    def _beta_m(self, v: float) -> float:
        return 4.0 * math.exp(-v / 18.0)

    def _alpha_h(self, v: float) -> float:
        return 0.07 * math.exp(-v / 20.0)

    def _beta_h(self, v: float) -> float:
        return 1.0 / (math.exp((30.0 - v) / 10.0) + 1.0)

    def _alpha_n(self, v: float) -> float:
        val = 10.0 - v
        if abs(val) < 1e-6:
            return 0.1
        return 0.01 * val / (math.exp(val / 10.0) - 1.0)

    def _beta_n(self, v: float) -> float:
        return 0.125 * math.exp(-v / 80.0)

    def _get_injected_current(self, t: float) -> float:
        """Determines injected stimulus current at time t (ms)."""
        if self._is_sustained:
            return self._i_inj_ua if t >= 2.0 else 0.0

        if not self._is_paired:
            # Single pulse between 2.0 ms and 2.0 + duration
            return self._i_inj_ua if 2.0 <= t <= (2.0 + self._pulse_duration_ms) else 0.0
        else:
            # Paired pulses: first at 2.0 ms, second at 2.0 + delay
            p1 = 2.0 <= t <= (2.0 + self._pulse_duration_ms)
            t2 = 2.0 + self._paired_delay_ms
            p2 = t2 <= t <= (t2 + self._pulse_duration_ms)
            return self._i_inj_ua if (p1 or p2) else 0.0

    def step_rk4(self, dt: float) -> None:
        """Executes a 4th-order Runge-Kutta numerical integration step."""
        t = self._time_ms
        v_depol = self._vm - V_REST
        phi = 3.0 ** ((self._temp_c - 6.3) / 10.0)

        # Apply pharmacological conductance modifications
        g_na_factor = 1.0
        g_k_factor = 1.0
        if self._toxin == Neurotoxin.TTX:
            g_na_factor = 0.02   # 98% Na⁺ block
        elif self._toxin == Neurotoxin.TEA:
            g_k_factor = 0.05    # 95% K⁺ block
        elif self._toxin == Neurotoxin.LIDOCAINE:
            g_na_factor = 0.35   # Partial Na⁺ block

        g_na = G_NA_MAX_BASE * g_na_factor * (self._m ** 3) * self._h
        g_k = G_K_MAX_BASE * g_k_factor * (self._n ** 4)
        g_l = G_LEAK_BASE

        i_na = g_na * (self._vm - E_NA)
        i_k = g_k * (self._vm - E_K)
        i_l = g_l * (self._vm - E_LEAK)
        i_stim = self._get_injected_current(t)

        dvm_dt = (i_stim - i_na - i_k - i_l) / C_MEMBRANE
        dm_dt = phi * (self._alpha_m(v_depol) * (1.0 - self._m) - self._beta_m(v_depol) * self._m)
        dh_dt = phi * (self._alpha_h(v_depol) * (1.0 - self._h) - self._beta_h(v_depol) * self._h)
        dn_dt = phi * (self._alpha_n(v_depol) * (1.0 - self._n) - self._beta_n(v_depol) * self._n)

        # Euler / Midpoint update
        self._vm += dvm_dt * dt
        self._m = max(0.0, min(1.0, self._m + dm_dt * dt))
        self._h = max(0.0, min(1.0, self._h + dh_dt * dt))
        self._n = max(0.0, min(1.0, self._n + dn_dt * dt))
        self._time_ms += dt

        # Log history
        self._time_history.append(self._time_ms)
        self._vm_history.append(self._vm)
        self._gna_history.append(g_na)
        self._gk_history.append(g_k)

        # Keep trace bounded to last 50 ms window
        while len(self._time_history) > 1000:
            self._time_history.pop(0)
            self._vm_history.pop(0)
            self._gna_history.pop(0)
            self._gk_history.pop(0)

    def _simulate_initial_window(self) -> None:
        """Simulates initial 35 ms from resting conditions."""
        self.reset_state()
        dt = 0.05  # 50 μs timestep
        steps = int(35.0 / dt)
        for _ in range(steps):
            self.step_rk4(dt)

    def reset_state(self) -> None:
        """Resets variables back to resting state."""
        self._vm = V_REST
        v_depol = 0.0
        # Steady-state gating at rest
        self._m = self._alpha_m(v_depol) / (self._alpha_m(v_depol) + self._beta_m(v_depol))
        self._h = self._alpha_h(v_depol) / (self._alpha_h(v_depol) + self._beta_h(v_depol))
        self._n = self._alpha_n(v_depol) / (self._alpha_n(v_depol) + self._beta_n(v_depol))
        self._time_ms = 0.0
        self._time_history.clear()
        self._vm_history.clear()
        self._gna_history.clear()
        self._gk_history.clear()

    # -------------------------------------------------------------------------
    # Presets & Control Methods
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads designated action potential curriculum preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._toxin = self._preset.toxin
        self._i_inj_ua = self._preset.i_inj_ua
        self._pulse_duration_ms = self._preset.pulse_duration_ms
        self._is_sustained = self._preset.is_sustained
        self._is_paired = self._preset.is_paired
        self._paired_delay_ms = self._preset.paired_delay_ms

        self._simulate_initial_window()
        self.preset_changed.emit(preset_code)
        self.update()

    def set_toxin(self, toxin: Neurotoxin) -> None:
        """Applies selective pharmacological ion channel block."""
        self._toxin = toxin
        self._simulate_initial_window()
        self.update()

    def step_stimulus_current(self, delta_ua: float) -> None:
        """Adjusts injected stimulus current amplitude."""
        self._i_inj_ua = max(0.0, min(50.0, round(self._i_inj_ua + delta_ua, 1)))
        self._simulate_initial_window()
        self.update()

    def toggle_animation(self) -> None:
        """Toggles continuous integration playback."""
        self._is_running = not self._is_running
        self.update()

    def _on_sim_tick(self) -> None:
        if not self._is_running:
            return
        # Advance simulation by 0.5 ms per frame
        dt = 0.05
        for _ in range(10):
            self.step_rk4(dt)
        self.voltage_updated.emit(self._vm)
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Background Card & Acrylic Frame
        card_rect = self.boundingRect()
        bg_brush = QBrush(QColor(10, 15, 29, 245))
        border_pen = QPen(QColor(56, 189, 248, 200) if self.isSelected() else QColor(30, 41, 59, 220), 1.8)
        painter.setBrush(bg_brush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header & Badges
        self._paint_header(painter)

        # 3. Left Panel: Voltage Oscilloscope & Conductance Curves
        self._paint_oscilloscope_panel(painter)

        # 4. Right Panel: Axon Membrane Channels & Ion Gradients
        self._paint_membrane_panel(painter)

        # 5. Bottom Digital Readout Bar
        self._paint_readout_bar(painter)

        # 6. Interactive In-Air Controls Bar
        self._paint_controls_bar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QPen(QColor("#f8fafc")))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 28, "⚡ Điện thế Hoạt động & Mô hình Hodgkin-Huxley Mực ống Khổng lồ")

        # Badge
        badge_text = self._preset.badge_label
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        badge_rect = QRectF(600.0, 12.0, 185.0, 22.0)
        painter.setBrush(QBrush(QColor(244, 63, 94, 50)))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # Subtitle / Description
        painter.setPen(QPen(QColor("#94a3b8")))
        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.drawText(20, 46, self._preset.description)

    def _paint_oscilloscope_panel(self, painter: QPainter) -> None:
        """Renders live membrane potential Vm(t) oscilloscope and conductance traces."""
        osc_rect = QRectF(20.0, 56.0, 480.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(osc_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(34, 78, "DAO ĐỘNG KÝ ĐIỆN THẾ MÀNG V_m(t) & ĐỘ DẪN ION g(t)")

        # Voltage Display Area
        v_box = QRectF(34.0, 92.0, 452.0, 210.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(v_box, 6.0, 6.0)

        # Horizontal Grid Lines (-80 mV to +50 mV)
        # Mapping: V in [-90, +60] -> Y in [v_box.bottom() - 10, v_box.top() + 10]
        def v_to_y(v_val: float) -> float:
            norm = (v_val - (-90.0)) / (60.0 - (-90.0))
            return v_box.bottom() - 15.0 - norm * (v_box.height() - 30.0)

        painter.setFont(QFont("Segoe UI", 7))
        for v_grid, v_lbl, v_col in [
            (50.0, "+50 mV (E_Na)", "#f59e0b"),
            (0.0, "0 mV", "#475569"),
            (-55.0, "-55 mV (Ngưỡng)", "#ef4444"),
            (-65.0, "-65 mV (Nghỉ V_rest)", "#38bdf8"),
            (-77.0, "-77 mV (E_K)", "#10b981"),
        ]:
            y_pos = v_to_y(v_grid)
            dash_style = Qt.PenStyle.DashLine if v_grid != 0.0 else Qt.PenStyle.SolidLine
            painter.setPen(QPen(QColor(v_col), 0.8, dash_style))
            painter.drawLine(QPointF(v_box.left() + 45.0, y_pos), QPointF(v_box.right() - 10.0, y_pos))

            painter.setPen(QPen(QColor(v_col)))
            painter.drawText(int(v_box.left() + 4.0), int(y_pos + 3.0), v_lbl)

        # Plot Voltage Trace Vm(t)
        if len(self._vm_history) > 1:
            v_path = QPainterPath()
            t_min = self._time_history[0]
            t_max = self._time_history[-1]
            t_span = max(10.0, t_max - t_min)

            w_plot = v_box.width() - 65.0
            x_origin = v_box.left() + 50.0

            for i, (t_val, vm_val) in enumerate(zip(self._time_history, self._vm_history)):
                px = x_origin + ((t_val - t_min) / t_span) * w_plot
                py = v_to_y(vm_val)
                if i == 0:
                    v_path.moveTo(px, py)
                else:
                    v_path.lineTo(px, py)

            painter.setPen(QPen(QColor("#f43f5e"), 2.2))
            painter.drawPath(v_path)

        # Bottom Sub-Panel: Conductance Traces gNa(t) and gK(t)
        cond_box = QRectF(34.0, 314.0, 452.0, 118.0)
        painter.setBrush(QBrush(QColor(2, 6, 23, 200)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.0))
        painter.drawRoundedRect(cond_box, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(46, 330, "ĐỘ DẪN ION g_Na (Cam) & g_K (Xanh lá) [mS/cm²]")

        if len(self._gna_history) > 1:
            gna_path = QPainterPath()
            gk_path = QPainterPath()

            t_min = self._time_history[0]
            t_max = self._time_history[-1]
            t_span = max(10.0, t_max - t_min)

            w_cond = cond_box.width() - 40.0
            x_c_origin = cond_box.left() + 20.0
            base_cond_y = cond_box.bottom() - 15.0

            for i, (t_val, gna, gk) in enumerate(zip(self._time_history, self._gna_history, self._gk_history)):
                px = x_c_origin + ((t_val - t_min) / t_span) * w_cond
                py_na = base_cond_y - (gna / 120.0) * 75.0
                py_k = base_cond_y - (gk / 36.0) * 75.0

                if i == 0:
                    gna_path.moveTo(px, py_na)
                    gk_path.moveTo(px, py_k)
                else:
                    gna_path.lineTo(px, py_na)
                    gk_path.lineTo(px, py_k)

            painter.setPen(QPen(QColor("#f59e0b"), 1.8))
            painter.drawPath(gna_path)

            painter.setPen(QPen(QColor("#10b981"), 1.8))
            painter.drawPath(gk_path)

    def _paint_membrane_panel(self, painter: QPainter) -> None:
        """Renders lipid bilayer membrane, ion channel conformations, and concentrations."""
        mem_rect = QRectF(514.0, 56.0, 406.0, 390.0)
        painter.setBrush(QBrush(QColor(6, 11, 23, 240)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mem_rect, 10.0, 10.0)

        # Title
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(528, 78, "MÀNG TẾ BÀO NƠRON & KÊNH ION PHỤ THUỘC ĐIỆN THẾ")

        # Lipid Bilayer Canvas
        bilayer_y = 190.0
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(528, 108, "NGOẠI BÀO (EXTRACELLULAR): [Na⁺] cao (440 mM) | [K⁺] thấp (20 mM)")
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(528, 295, "NỘI BÀO (INTRACELLULAR): [Na⁺] thấp (50 mM) | [K⁺] cao (400 mM)")

        # Upper & Lower Leaflets of Bilayer
        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRect(QRectF(528.0, bilayer_y - 20.0, 378.0, 16.0))
        painter.drawRect(QRectF(528.0, bilayer_y + 14.0, 378.0, 16.0))

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(532, int(bilayer_y + 5.0), "Lớp Kép Photpholipid (C_m ≈ 1 μF/cm²)")

        # 1. Voltage-Gated Na+ Channel (Left)
        na_x = 600.0
        na_open = (self._m ** 3) * self._h > 0.15
        na_rect = QRectF(na_x - 22.0, bilayer_y - 26.0, 44.0, 62.0)
        painter.setBrush(QBrush(QColor(245, 158, 11, 230) if na_open else QColor(100, 116, 139, 200)))
        painter.setPen(QPen(QColor("#f59e0b") if na_open else QColor("#ffffff"), 1.5))
        painter.drawRoundedRect(na_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#020617")))
        painter.drawText(na_rect, Qt.AlignmentFlag.AlignCenter, f"Kênh Na⁺\nm³h={((self._m**3)*self._h):.2f}")

        # 2. Voltage-Gated K+ Channel (Right)
        k_x = 760.0
        k_open = (self._n ** 4) > 0.2
        k_rect = QRectF(k_x - 22.0, bilayer_y - 26.0, 44.0, 62.0)
        painter.setBrush(QBrush(QColor(16, 185, 129, 230) if k_open else QColor(100, 116, 139, 200)))
        painter.setPen(QPen(QColor("#10b981") if k_open else QColor("#ffffff"), 1.5))
        painter.drawRoundedRect(k_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#020617")))
        painter.drawText(k_rect, Qt.AlignmentFlag.AlignCenter, f"Kênh K⁺\nn⁴={(self._n**4):.2f}")

        # Neurotoxin Indicator
        if self._toxin != Neurotoxin.NONE:
            tox_rect = QRectF(528.0, 315.0, 378.0, 30.0)
            painter.setBrush(QBrush(QColor(239, 68, 68, 50)))
            painter.setPen(QPen(QColor("#ef4444"), 1.2))
            painter.drawRoundedRect(tox_rect, 6.0, 6.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ef4444")))
            painter.drawText(tox_rect, Qt.AlignmentFlag.AlignCenter, f"⚠️ ĐANG DÙNG ĐỘC TỐ: {self._toxin.value.upper()} (ỨC CHẾ KÊNH)")

        # Gating Variable Gauge Bar (m, h, n)
        gy = 360.0
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(532, int(gy), f"Cổng m (Kích hoạt Na⁺): {self._m:.3f}")
        painter.drawText(532, int(gy + 20), f"Cổng h (Bất hoạt Na⁺): {self._h:.3f}")
        painter.drawText(532, int(gy + 40), f"Cổng n (Kích hoạt K⁺): {self._n:.3f}")

        # Draw mini progress bars for m, h, n
        for idx, (val, col) in enumerate([(self._m, "#38bdf8"), (self._h, "#f59e0b"), (self._n, "#10b981")]):
            bar_rect = QRectF(700.0, gy - 8.0 + idx * 20.0, 190.0, 10.0)
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(QColor(51, 65, 85), 0.8))
            painter.drawRoundedRect(bar_rect, 3.0, 3.0)

            fill_w = val * 190.0
            painter.setBrush(QBrush(QColor(col)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(700.0, gy - 8.0 + idx * 20.0, fill_w, 10.0), 3.0, 3.0)

    def _paint_readout_bar(self, painter: QPainter) -> None:
        """Draws digital parameter readouts and biophysical formulas."""
        bar_rect = QRectF(20.0, 456.0, 900.0, 68.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bar_rect, 8.0, 8.0)

        v_now = self._vm
        i_stim = self._get_injected_current(self._time_ms)
        gna_now = G_NA_MAX_BASE * (self._m ** 3) * self._h
        gk_now = G_K_MAX_BASE * (self._n ** 4)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(36, 478, f"ĐIỆN THẾ MÀNG TỨC THỜI: V_m = {v_now:+.1f} mV (Nghỉ: {V_REST:.1f} mV)")
        painter.drawText(36, 498, f"DÒNG KÍCH THÍCH BƠM VÀO: I_inj = {i_stim:.1f} μA/cm² (Thời lượng: {self._pulse_duration_ms:.1f} ms)")
        painter.drawText(36, 516, f"PHƯƠNG TRÌNH DÒNG MÀNG: C_m·(dV/dt) = I_inj - I_Na - I_K - I_L")

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(480, 478, f"ĐỘ DẪN KÊNH NA⁺: g_Na = {gna_now:.2f} mS/cm² (E_Na = +50.0 mV)")
        painter.drawText(480, 498, f"ĐỘ DẪN KÊNH K⁺: g_K = {gk_now:.2f} mS/cm² (E_K = -77.0 mV)")
        painter.drawText(480, 516, f"NHIỆT ĐỘ MỰC ỐNG: T = {self._temp_c:.1f} °C (Hệ số Q₁₀ = 3.0)")

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        bar_y = 536.0
        self._buttons.clear()

        btn_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(btn_font)

        # 1. Preset Selector Button
        preset_rect = QRectF(20.0, bar_y, 160.0, 32.0)
        self._draw_btn(painter, preset_rect, "btn_preset", f"Presets: {self._preset.badge_label}")

        # 2. Stimulus Inject Button
        inject_rect = QRectF(188.0, bar_y, 110.0, 32.0)
        self._draw_btn(painter, inject_rect, "btn_inject", "⚡ Bơm Xung (Stim)")

        # 3. Current Steppers
        self._draw_btn(painter, QRectF(306.0, bar_y, 65.0, 32.0), "btn_i_minus", "-2 μA")
        self._draw_btn(painter, QRectF(377.0, bar_y, 65.0, 32.0), "btn_i_plus", "+2 μA")

        # 4. Toxin Selector
        tox_label = "Độc tố: " + self._toxin.value.upper()
        tox_rect = QRectF(450.0, bar_y, 105.0, 32.0)
        self._draw_btn(painter, tox_rect, "btn_toxin", tox_label)

        # 5. Play / Pause
        play_label = "⏸ Tạm dừng" if self._is_running else "▶ Tiếp tục"
        self._draw_btn(painter, QRectF(563.0, bar_y, 90.0, 32.0), "btn_play", play_label)

        # 6. Reset
        self._draw_btn(painter, QRectF(661.0, bar_y, 65.0, 32.0), "btn_reset", "↺ Đặt lại")

        # 7. Temperature Switcher
        self._draw_btn(painter, QRectF(734.0, bar_y, 55.0, 32.0), "btn_temp", f"{self._temp_c:.0f}°C")

        # 8. Stamp to Canvas Button
        stamp_rect = QRectF(797.0, bar_y, 85.0, 32.0)
        self._draw_btn(painter, stamp_rect, "btn_stamp", "📌 Đóng dấu")

        # 9. Close Button (Top right)
        close_rect = QRectF(self.CARD_W - 38.0, 10.0, 26.0, 26.0)
        self._draw_btn(painter, close_rect, "btn_close", "✕")

    def _draw_btn(self, painter: QPainter, rect: QRectF, key: str, label: str) -> None:
        self._buttons[key] = rect
        is_hovered = self._hovered_btn == key

        btn_bg = QColor(30, 41, 59, 220) if not is_hovered else QColor(244, 63, 94, 220)
        btn_pen = QPen(QColor("#f43f5e") if is_hovered else QColor(51, 65, 85), 1.2)

        painter.setBrush(QBrush(btn_bg))
        painter.setPen(btn_pen)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setPen(QPen(QColor("#f8fafc")))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Mouse Events
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_hover = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if old_hover != self._hovered_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered_btn = None
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._handle_button_click(key)
                    event.accept()
                    return
            self._is_dragging = True
            self._last_mouse_pos = event.scenePos()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_dragging:
            delta = event.scenePos() - self._last_mouse_pos
            self._last_mouse_pos = event.scenePos()
            self.setPos(self.pos() + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging = False
        super().mouseReleaseEvent(event)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_inject":
            self._simulate_initial_window()
        elif key == "btn_i_minus":
            self.step_stimulus_current(-2.0)
        elif key == "btn_i_plus":
            self.step_stimulus_current(2.0)
        elif key == "btn_toxin":
            toxins = [Neurotoxin.NONE, Neurotoxin.TTX, Neurotoxin.TEA, Neurotoxin.LIDOCAINE]
            curr_idx = toxins.index(self._toxin)
            self.set_toxin(toxins[(curr_idx + 1) % len(toxins)])
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_reset":
            self._simulate_initial_window()
        elif key == "btn_temp":
            temps = [6.3, 18.0, 37.0]
            curr_idx = temps.index(self._temp_c) if self._temp_c in temps else 0
            self._temp_c = temps[(curr_idx + 1) % len(temps)]
            self._simulate_initial_window()
        elif key == "btn_stamp":
            self.stamp_diagram()
        elif key == "btn_close":
            self.setVisible(False)

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current state onto canvas."""
        v_peak = max(self._vm_history) if self._vm_history else self._vm
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 640.0,
            "height": 340.0,
            "preset_code": self._active_preset_code,
            "v_peak_mv": v_peak,
            "i_inj_ua": self._i_inj_ua,
            "toxin": self._toxin.value,
            "temp_c": self._temp_c,
        }
        self.stamp_requested.emit(stamp_data)
