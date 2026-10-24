"""
AirTeach AI - Interactive Electrophysiology: Hodgkin-Huxley Neuron ODE Simulator
==================================================================================
Simulates the authentic Nobel Prize-winning Hodgkin-Huxley (1952) 4-variable ODE system
for squid giant axon action potential generation, ion channel gating dynamics,
refractory periods, and pharmacological channel blockers (TTX / TEA).

Biophysical Model:
  1. Membrane Capacitive Current:
       C_m * dV/dt = I_inj - I_Na - I_K - I_L
  2. Ionic Currents:
       I_Na = g_Na_max * m^3 * h * (V - E_Na)
       I_K  = g_K_max  * n^4 * (V - E_K)
       I_L  = g_L      * (V - E_L)
  3. Gating Kinetics (dx/dt = alpha_x * (1 - x) - beta_x * x):
       v = V - V_rest (where V_rest = -65 mV)
       alpha_m(v) = 0.1 * (25 - v) / (exp((25 - v) / 10) - 1)
       beta_m(v)  = 4.0 * exp(-v / 18)
       alpha_h(v) = 0.07 * exp(-v / 20)
       beta_h(v)  = 1.0 / (exp((30 - v) / 10) + 1)
       alpha_n(v) = 0.01 * (10 - v) / (exp((10 - v) / 10) - 1)
       beta_n(v)  = 0.125 * exp(-v / 80)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent

# Standard Biophysical Constants (Squid Giant Axon at 6.3 °C)
V_REST = -65.0  # mV
C_M = 1.0  # uF/cm^2
G_NA_MAX = 120.0  # mS/cm^2
G_K_MAX = 36.0  # mS/cm^2
G_L = 0.3  # mS/cm^2
E_NA = 50.0  # mV
E_K = -77.0  # mV
E_L = -54.4  # mV


@dataclass(frozen=True)
class NeuroPreset:
    name: str
    description: str
    i_inj_amp: float  # uA/cm^2
    stim_start_ms: float  # ms
    stim_duration_ms: float  # ms
    is_continuous_train: bool
    ttx_block_na: bool  # Tetrodotoxin blocks Na+
    tea_block_k: bool  # Tetraethylammonium blocks K+


HODGKIN_PRESETS: Dict[str, NeuroPreset] = {
    "SINGLE_ACTION_POTENTIAL": NeuroPreset(
        name="Điện Thế Hoạt Động Đơn Cực (Action Potential)",
        description="Xung kích thích 10 μA/cm² khử cực màng vượt ngưỡng (-55 mV), bùng phát đỉnh spike +35 mV.",
        i_inj_amp=10.0,
        stim_start_ms=10.0,
        stim_duration_ms=2.0,
        is_continuous_train=False,
        ttx_block_na=False,
        tea_block_k=False,
    ),
    "REPETITIVE_TRAIN_FIRING": NeuroPreset(
        name="Chuỗi Xung Nhịp Nhàng (Repetitive Train)",
        description="Dòng kích thích liên tục duy trì trạng thái dao động chu kỳ giới hạn (Limit Cycle Oscillation).",
        i_inj_amp=12.0,
        stim_start_ms=5.0,
        stim_duration_ms=45.0,
        is_continuous_train=True,
        ttx_block_na=False,
        tea_block_k=False,
    ),
    "SUBTHRESHOLD_STIMULUS": NeuroPreset(
        name="Kích Thích Dưới Ngưỡng (Subthreshold)",
        description="Dòng 3 μA/cm² không đủ mở cổng Na⁺ hàng loạt, màng trở về trạng thái nghỉ mà không tạo spike.",
        i_inj_amp=3.0,
        stim_start_ms=10.0,
        stim_duration_ms=2.0,
        is_continuous_train=False,
        ttx_block_na=False,
        tea_block_k=False,
    ),
    "TTX_SODIUM_BLOCK": NeuroPreset(
        name="Độc Tố Cá Nóc TTX (Khóa Kênh Na⁺)",
        description="Tetrodotoxin khóa hoàn toàn kênh Na⁺ (g_Na = 0), triệt tiêu hoàn toàn khả năng phát xung thần kinh.",
        i_inj_amp=15.0,
        stim_start_ms=10.0,
        stim_duration_ms=3.0,
        is_continuous_train=False,
        ttx_block_na=True,
        tea_block_k=False,
    ),
    "TEA_POTASSIUM_BLOCK": NeuroPreset(
        name="Hóa Chất TEA (Khóa Kênh K⁺)",
        description="Tetraethylammonium khóa kênh K⁺ (g_K = 0), ngăn tái cực nhanh khiến xung kéo dài bất thường.",
        i_inj_amp=10.0,
        stim_start_ms=10.0,
        stim_duration_ms=2.0,
        is_continuous_train=False,
        ttx_block_na=False,
        tea_block_k=True,
    ),
}


class InteractiveHodgkinHuxleyItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating the Hodgkin-Huxley Action Potential ODE System.
    """

    stamp_requested = Signal(dict)
    simulation_computed = Signal(dict)

    def __init__(
        self,
        preset_key: str = "SINGLE_ACTION_POTENTIAL",
        stim_amp: Optional[float] = None,
        pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        if pos is not None:
            self.setPos(pos[0], pos[1])

        self._active_preset_key = preset_key if preset_key in HODGKIN_PRESETS else "SINGLE_ACTION_POTENTIAL"
        preset = HODGKIN_PRESETS[self._active_preset_key]

        self._stim_amp = float(stim_amp if stim_amp is not None else preset.i_inj_amp)
        self._stim_duration_ms = preset.stim_duration_ms
        self._ttx_blocked = preset.ttx_block_na
        self._tea_blocked = preset.tea_block_k

        self._width = 700.0
        self._height = 430.0

        # UI rects
        self._stim_slider_rect = QRectF(0, 0, 0, 0)
        self._ttx_btn_rect = QRectF(0, 0, 0, 0)
        self._tea_btn_rect = QRectF(0, 0, 0, 0)
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)
        self._preset_btn_rects: Dict[str, QRectF] = {}

        self._is_dragging_stim = False

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Hodgkin-Huxley Rate Functions ────────────────────────────────────────

    @staticmethod
    def alpha_m(v: float) -> float:
        # 0.1 * (25 - v) / (exp((25 - v) / 10) - 1)
        dv = 25.0 - v
        if abs(dv) < 1e-5:
            return 1.0  # L'Hopital limit: 0.1 * 10 = 1.0
        return 0.1 * dv / (math.exp(dv / 10.0) - 1.0)

    @staticmethod
    def beta_m(v: float) -> float:
        return 4.0 * math.exp(-v / 18.0)

    @staticmethod
    def alpha_h(v: float) -> float:
        return 0.07 * math.exp(-v / 20.0)

    @staticmethod
    def beta_h(v: float) -> float:
        return 1.0 / (math.exp((30.0 - v) / 10.0) + 1.0)

    @staticmethod
    def alpha_n(v: float) -> float:
        # 0.01 * (10 - v) / (exp((10 - v) / 10) - 1)
        dv = 10.0 - v
        if abs(dv) < 1e-5:
            return 0.1  # L'Hopital limit: 0.01 * 10 = 0.1
        return 0.01 * dv / (math.exp(dv / 10.0) - 1.0)

    @staticmethod
    def beta_n(v: float) -> float:
        return 0.125 * math.exp(-v / 80.0)

    # ── Simulation Solver (4th-order Runge-Kutta / Euler) ────────────────────

    @property
    def stim_amp(self) -> float:
        return self._stim_amp

    def set_stim_amp(self, amp: float) -> None:
        self._stim_amp = max(0.0, min(30.0, float(amp)))
        self.simulation_computed.emit(self.compute_simulation())
        self.update()

    def toggle_ttx(self) -> None:
        self._ttx_blocked = not self._ttx_blocked
        self.update()

    def toggle_tea(self) -> None:
        self._tea_blocked = not self._tea_blocked
        self.update()

    def apply_preset(self, preset_key: str) -> None:
        if preset_key in HODGKIN_PRESETS:
            self._active_preset_key = preset_key
            preset = HODGKIN_PRESETS[preset_key]
            self._stim_amp = preset.i_inj_amp
            self._stim_duration_ms = preset.stim_duration_ms
            self._ttx_blocked = preset.ttx_block_na
            self._tea_blocked = preset.tea_block_k
            self.simulation_computed.emit(self.compute_simulation())
            self.update()

    def compute_simulation(self) -> Dict[str, any]:
        """
        Integrates Hodgkin-Huxley ODEs over 50 ms with dt = 0.04 ms.
        Returns time series for V(t), m(t), h(t), n(t), g_Na(t), g_K(t).
        """
        t_max = 50.0  # ms
        dt = 0.04  # ms
        n_steps = int(t_max / dt)

        t_pts: List[float] = [0.0] * n_steps
        v_pts: List[float] = [0.0] * n_steps
        m_pts: List[float] = [0.0] * n_steps
        h_pts: List[float] = [0.0] * n_steps
        n_pts: List[float] = [0.0] * n_steps

        # Initial steady state at V_rest (-65 mV, v = 0)
        v0 = 0.0
        am0 = self.alpha_m(v0)
        bm0 = self.beta_m(v0)
        m = am0 / (am0 + bm0)

        ah0 = self.alpha_h(v0)
        bh0 = self.beta_h(v0)
        h = ah0 / (ah0 + bh0)

        an0 = self.alpha_n(v0)
        bn0 = self.beta_n(v0)
        n = an0 / (an0 + bn0)

        V = V_REST  # -65.0 mV

        g_na_eff = 0.0 if self._ttx_blocked else G_NA_MAX
        g_k_eff = 0.0 if self._tea_blocked else G_K_MAX

        preset = HODGKIN_PRESETS[self._active_preset_key]
        stim_start = preset.stim_start_ms
        stim_dur = self._stim_duration_ms

        v_max = V
        v_min = V

        for i in range(n_steps):
            t = i * dt
            t_pts[i] = t

            # Determine stimulus injection
            if preset.is_continuous_train:
                i_inj = self._stim_amp if t >= stim_start else 0.0
            else:
                i_inj = self._stim_amp if (stim_start <= t <= stim_start + stim_dur) else 0.0

            # Membrane voltage relative to resting
            v_rel = V - V_REST

            # Rates
            am = self.alpha_m(v_rel)
            bm = self.beta_m(v_rel)
            ah = self.alpha_h(v_rel)
            bh = self.beta_h(v_rel)
            an = self.alpha_n(v_rel)
            bn = self.beta_n(v_rel)

            # Currents
            i_na = g_na_eff * (m**3) * h * (V - E_NA)
            i_k = g_k_eff * (n**4) * (V - E_K)
            i_l = G_L * (V - E_L)

            # Derivatives
            dv_dt = (i_inj - i_na - i_k - i_l) / C_M
            dm_dt = am * (1.0 - m) - bm * m
            dh_dt = ah * (1.0 - h) - bh * h
            dn_dt = an * (1.0 - n) - bn * n

            # Euler integration step
            V += dv_dt * dt
            m += dm_dt * dt
            h += dh_dt * dt
            n += dn_dt * dt

            # Clamp gating variables in [0, 1]
            m = max(0.0, min(1.0, m))
            h = max(0.0, min(1.0, h))
            n = max(0.0, min(1.0, n))

            v_pts[i] = V
            m_pts[i] = m
            h_pts[i] = h
            n_pts[i] = n

            if V > v_max:
                v_max = V
            if V < v_min:
                v_min = V

        has_spike = v_max > -20.0
        return {
            "t_pts": t_pts,
            "v_pts": v_pts,
            "m_pts": m_pts,
            "h_pts": h_pts,
            "n_pts": n_pts,
            "v_max": v_max,
            "v_min": v_min,
            "has_spike": has_spike,
            "v_rest": V_REST,
            "g_na_eff": g_na_eff,
            "g_k_eff": g_k_eff,
        }

    def request_stamp(self) -> None:
        sim = self.compute_simulation()
        preset = HODGKIN_PRESETS[self._active_preset_key]
        data = {
            "center": (self.x(), self.y()),
            "preset_key": self._active_preset_key,
            "preset_name": preset.name,
            "description": preset.description,
            "stim_amp": self._stim_amp,
            "stim_duration_ms": self._stim_duration_ms,
            "v_max": sim["v_max"],
            "v_min": sim["v_min"],
            "has_spike": sim["has_spike"],
            "ttx_blocked": self._ttx_blocked,
            "tea_blocked": self._tea_blocked,
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        sim = self.compute_simulation()
        preset = HODGKIN_PRESETS[self._active_preset_key]

        hw = self._width / 2.0
        hh = self._height / 2.0

        # 1. Outer Container
        bg_grad = QLinearGradient(-hw, -hh, -hw, hh)
        bg_grad.setColorAt(0.0, QColor(10, 16, 28))
        bg_grad.setColorAt(1.0, QColor(5, 8, 16))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(28, 44, 72), 1.5))
        painter.drawRoundedRect(-hw, -hh, self._width, self._height, 10.0, 10.0)

        # 2. Header
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(-hw + 20, -hh + 12, self._width - 40, 22),
            Qt.AlignLeft,
            f"⚡ HỆ PHƯƠNG TRÌNH HODGKIN-HUXLEY (1952): {preset.name.upper()}",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(
            QRectF(-hw + 20, -hh + 34, self._width - 40, 18),
            Qt.AlignLeft,
            f"Mô hình điện sinh lý sợi trục nơ-ron mực ống | I_inj = {self._stim_amp:.1f} μA/cm² | V_max = {sim['v_max']:.1f} mV",
        )

        # 3. Presets Tabs
        self._draw_preset_tabs(painter, -hw + 20, -hh + 54)

        # 4. Left Dual Oscilloscope (Voltage & Gating Traces)
        scope_rect = QRectF(-hw + 20, -hh + 84, 400, 230)
        self._draw_oscilloscope(painter, scope_rect, sim)

        # 5. Right Electrophysiology Panel
        panel_rect = QRectF(scope_rect.right() + 14, -hh + 84, self._width - scope_rect.width() - 54, 230)
        self._draw_neuro_panel(painter, panel_rect, sim, preset)

        # 6. Bottom Controls
        controls_rect = QRectF(-hw + 20, hh - 100, self._width - 40, 88)
        self._draw_controls(painter, controls_rect, sim)

    def _draw_preset_tabs(self, painter: QPainter, x: float, y: float) -> None:
        self._preset_btn_rects.clear()
        tab_w = 128.0
        tab_h = 22.0

        keys = list(HODGKIN_PRESETS.keys())
        for i, key in enumerate(keys):
            preset = HODGKIN_PRESETS[key]
            rect = QRectF(x + i * (tab_w + 4), y, tab_w, tab_h)
            self._preset_btn_rects[key] = rect

            is_active = key == self._active_preset_key
            if is_active:
                painter.setBrush(QBrush(QColor(2, 132, 199)))
                painter.setPen(QPen(QColor(56, 189, 248), 1.5))
            else:
                painter.setBrush(QBrush(QColor(16, 24, 39)))
                painter.setPen(QPen(QColor(31, 46, 70), 1.0))

            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(241, 245, 249) if is_active else QColor(148, 163, 184))
            short_name = preset.name.split(" (")[0]
            painter.drawText(rect, Qt.AlignCenter, short_name)

    def _draw_oscilloscope(self, painter: QPainter, rect: QRectF, sim: Dict[str, any]) -> None:
        painter.setBrush(QBrush(QColor(6, 10, 18)))
        painter.setPen(QPen(QColor(24, 38, 60), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Trace 1: Membrane Voltage V(t) [-90 mV to +50 mV]
        ox = rect.left() + 38.0
        oy = rect.top() + 140.0
        pw = rect.width() - 50.0
        ph = 120.0

        # Grid lines
        painter.setPen(QPen(QColor(20, 32, 48), 1.0, Qt.DotLine))
        for v_level in [-80.0, -65.0, -50.0, 0.0, 40.0]:
            gy = oy - ((v_level - (-90.0)) / 140.0) * ph
            painter.drawLine(QPointF(ox, gy), QPointF(ox + pw, gy))
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor(100, 116, 139))
            painter.drawText(QRectF(rect.left() + 2, gy - 7, 34, 14), Qt.AlignRight, f"{int(v_level)}")

        # Threshold (-55 mV) dashed line
        thresh_y = oy - ((-55.0 - (-90.0)) / 140.0) * ph
        painter.setPen(QPen(QColor(234, 179, 8, 150), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(ox, thresh_y), QPointF(ox + pw, thresh_y))

        # Resting (-65 mV) line
        rest_y = oy - ((-65.0 - (-90.0)) / 140.0) * ph
        painter.setPen(QPen(QColor(74, 222, 128, 120), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(ox, rest_y), QPointF(ox + pw, rest_y))

        # Plot Voltage Curve
        t_pts = sim["t_pts"]
        v_pts = sim["v_pts"]
        v_path = QPainterPath()
        for idx, (t, v) in enumerate(zip(t_pts, v_pts)):
            px = ox + (t / 50.0) * pw
            py = oy - ((v - (-90.0)) / 140.0) * ph
            if idx == 0:
                v_path.moveTo(px, py)
            else:
                v_path.lineTo(px, py)

        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(v_path)

        # Trace 2: Gating Variables m, h, n [0.0 to 1.0]
        gy_box = rect.top() + 155.0
        gh = 60.0
        painter.setPen(QPen(QColor(20, 32, 48), 1.0, Qt.DotLine))
        painter.drawLine(QPointF(ox, gy_box + gh), QPointF(ox + pw, gy_box + gh))
        painter.drawLine(QPointF(ox, gy_box), QPointF(ox + pw, gy_box))

        m_path = QPainterPath()
        h_path = QPainterPath()
        n_path = QPainterPath()

        for idx, (t, m_val, h_val, n_val) in enumerate(
            zip(sim["t_pts"], sim["m_pts"], sim["h_pts"], sim["n_pts"])
        ):
            px = ox + (t / 50.0) * pw
            my = gy_box + gh - m_val * gh
            hy = gy_box + gh - h_val * gh
            ny = gy_box + gh - n_val * gh

            if idx == 0:
                m_path.moveTo(px, my)
                h_path.moveTo(px, hy)
                n_path.moveTo(px, ny)
            else:
                m_path.lineTo(px, my)
                h_path.lineTo(px, hy)
                n_path.lineTo(px, ny)

        # Draw m(t) [Sodium activation, cyan]
        painter.setPen(QPen(QColor(34, 211, 238), 1.4))
        painter.drawPath(m_path)

        # Draw h(t) [Sodium inactivation, coral]
        painter.setPen(QPen(QColor(251, 113, 133), 1.4))
        painter.drawPath(h_path)

        # Draw n(t) [Potassium activation, emerald]
        painter.setPen(QPen(QColor(74, 222, 128), 1.4))
        painter.drawPath(n_path)

        # Gating legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(34, 211, 238))
        painter.drawText(QRectF(ox + 4, gy_box + 2, 40, 12), Qt.AlignLeft, "m (Na+)")

        painter.setPen(QColor(251, 113, 133))
        painter.drawText(QRectF(ox + 50, gy_box + 2, 40, 12), Qt.AlignLeft, "h (Na-inact)")

        painter.setPen(QColor(74, 222, 128))
        painter.drawText(QRectF(ox + 115, gy_box + 2, 40, 12), Qt.AlignLeft, "n (K+)")

    def _draw_neuro_panel(
        self, painter: QPainter, rect: QRectF, sim: Dict[str, any], preset: NeuroPreset
    ) -> None:
        painter.setBrush(QBrush(QColor(8, 14, 24)))
        painter.setPen(QPen(QColor(24, 38, 60), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, 16),
            Qt.AlignLeft,
            "🧠 ĐIỆN SINH LÝ TẾ BÀO THẦN KINH",
        )

        spike_status = "ĐẠT ĐỈNH SPIKE" if sim["has_spike"] else "DƯỚI NGƯỠNG"
        spike_col = QColor(74, 222, 128) if sim["has_spike"] else QColor(250, 204, 21)

        rows = [
            ("Trạng thái điện thế:", spike_status, spike_col),
            ("Điện thế nghỉ (V_rest):", f"{sim['v_rest']:.1f} mV", QColor(148, 163, 184)),
            ("Điện thế cực đại (V_max):", f"{sim['v_max']:.1f} mV", QColor(56, 189, 248)),
            ("Tái cực sâu (V_min):", f"{sim['v_min']:.1f} mV", QColor(168, 85, 247)),
            ("Kênh Na⁺ (g_Na):", f"{sim['g_na_eff']:.1f} mS/cm²", QColor(34, 211, 238)),
            ("Kênh K⁺ (g_K):", f"{sim['g_k_eff']:.1f} mS/cm²", QColor(74, 222, 128)),
            ("Khóa TTX (Na⁺):", "BỊ KHÓA" if self._ttx_blocked else "Bình thường", QColor(244, 63, 94) if self._ttx_blocked else QColor(148, 163, 184)),
            ("Khóa TEA (K⁺):", "BỊ KHÓA" if self._tea_blocked else "Bình thường", QColor(244, 63, 94) if self._tea_blocked else QColor(148, 163, 184)),
        ]

        y = rect.top() + 26.0
        painter.setFont(QFont("Segoe UI", 7))
        for label, val, col in rows:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 8, y, 120, 13), Qt.AlignLeft, label)
            painter.setPen(col)
            painter.drawText(QRectF(rect.right() - 110, y, 102, 13), Qt.AlignRight, val)
            y += 15.0

        # Math ODE formulas box
        f_box = QRectF(rect.left() + 6, y + 4, rect.width() - 12, 70)
        painter.setBrush(QBrush(QColor(6, 10, 18)))
        painter.setPen(QPen(QColor(20, 32, 50), 1.0))
        painter.drawRoundedRect(f_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(f_box.left() + 6, f_box.top() + 4, f_box.width() - 12, 12), Qt.AlignLeft, "HỆ PHƯƠNG TRÌNH VI PHÂN:")

        painter.setFont(QFont("Segoe UI", 6.8))
        painter.setPen(QColor(203, 213, 225))
        formulas = [
            "• C_m · dV/dt = I_inj - I_Na - I_K - I_L",
            "• I_Na = g_Na·m³h(V - E_Na)",
            "• I_K = g_K·n⁴(V - E_K)",
            "• dx/dt = α_x(1 - x) - β_x·x,  x ∈ {m, h, n}",
        ]
        fy = f_box.top() + 18.0
        for form in formulas:
            painter.drawText(QRectF(f_box.left() + 6, fy, f_box.width() - 12, 12), Qt.AlignLeft, form)
            fy += 13.0

    def _draw_controls(self, painter: QPainter, rect: QRectF, sim: Dict[str, any]) -> None:
        painter.setBrush(QBrush(QColor(10, 16, 28)))
        painter.setPen(QPen(QColor(28, 44, 72), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # 1. Stimulus Slider I_inj (0 to 30 uA/cm^2)
        s_y = rect.top() + 14.0
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(245, 158, 11))
        painter.drawText(QRectF(rect.left() + 12, s_y - 2, 150, 16), Qt.AlignLeft, f"Dòng kích thích: {self._stim_amp:.1f} μA/cm²")

        s_track = QRectF(rect.left() + 165, s_y + 2, 190, 8)
        self._stim_slider_rect = s_track
        painter.setBrush(QBrush(QColor(24, 34, 53)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(s_track, 4.0, 4.0)

        s_norm = self._stim_amp / 30.0
        s_fill_w = max(0.0, min(1.0, s_norm)) * s_track.width()
        painter.setBrush(QBrush(QColor(245, 158, 11)))
        painter.drawRoundedRect(QRectF(s_track.left(), s_track.top(), s_fill_w, 8), 4.0, 4.0)

        # 2. Blockers TTX / TEA buttons
        ttx_btn = QRectF(rect.left() + 370, rect.top() + 12, 95, 26)
        self._ttx_btn_rect = ttx_btn
        if self._ttx_blocked:
            painter.setBrush(QBrush(QColor(239, 68, 68, 80)))
            painter.setPen(QPen(QColor(239, 68, 68), 1.5))
        else:
            painter.setBrush(QBrush(QColor(24, 34, 53)))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(ttx_btn, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(254, 202, 202) if self._ttx_blocked else QColor(148, 163, 184))
        painter.drawText(ttx_btn, Qt.AlignCenter, "🐟 TTX (Na⁺): " + ("KHÓA" if self._ttx_blocked else "MỞ"))

        tea_btn = QRectF(rect.left() + 475, rect.top() + 12, 95, 26)
        self._tea_btn_rect = tea_btn
        if self._tea_blocked:
            painter.setBrush(QBrush(QColor(239, 68, 68, 80)))
            painter.setPen(QPen(QColor(239, 68, 68), 1.5))
        else:
            painter.setBrush(QBrush(QColor(24, 34, 53)))
            painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(tea_btn, 4.0, 4.0)

        painter.setPen(QColor(254, 202, 202) if self._tea_blocked else QColor(148, 163, 184))
        painter.drawText(tea_btn, Qt.AlignCenter, "🧪 TEA (K⁺): " + ("KHÓA" if self._tea_blocked else "MỞ"))

        # 3. Stamp Button
        stamp_btn = QRectF(rect.right() - 115, rect.top() + 12, 105, 26)
        self._stamp_btn_rect = stamp_btn
        painter.setBrush(QBrush(QColor(14, 116, 144)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(stamp_btn, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(stamp_btn, Qt.AlignCenter, "📌 ĐÓNG DẤU BÀI DẠY")

        # Instruction Hint
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(100, 116, 139))
        painter.drawText(
            QRectF(rect.left() + 14, rect.top() + 48, rect.width() - 28, 20),
            Qt.AlignLeft,
            "💡 Tăng I_inj trên 7 μA/cm² để kích hoạt điện thế hoạt động; thử nghiệm khóa TTX (mất spike) hoặc TEA (xung kéo dài).",
        )

    # ── Mouse Interaction ────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        for key, rect in self._preset_btn_rects.items():
            if rect.contains(pos):
                self.apply_preset(key)
                event.accept()
                return

        if self._ttx_btn_rect.contains(pos):
            self.toggle_ttx()
            event.accept()
            return

        if self._tea_btn_rect.contains(pos):
            self.toggle_tea()
            event.accept()
            return

        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        if self._stim_slider_rect.contains(pos) or (
            abs(pos.y() - self._stim_slider_rect.center().y()) < 12
            and self._stim_slider_rect.left() <= pos.x() <= self._stim_slider_rect.right()
        ):
            self._is_dragging_stim = True
            norm = (pos.x() - self._stim_slider_rect.left()) / self._stim_slider_rect.width()
            self.set_stim_amp(norm * 30.0)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._is_dragging_stim:
            norm = (pos.x() - self._stim_slider_rect.left()) / self._stim_slider_rect.width()
            self.set_stim_amp(norm * 30.0)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging_stim = False
        super().mouseReleaseEvent(event)
